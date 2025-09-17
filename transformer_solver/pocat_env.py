# transformer_solver/pocat_env.py

import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Optional, List

from torchrl.data import Unbounded , UnboundedDiscrete
from torchrl.data import CompositeSpec



from common.pocat_defs import (
    SCALAR_PROMPT_FEATURE_DIM, FEATURE_DIM, FEATURE_INDEX,
    NODE_TYPE_BATTERY, NODE_TYPE_IC, NODE_TYPE_LOAD
)

BATTERY_NODE_IDX = 0

class PocatEnv(EnvBase):
    name = "pocat"

    def __init__(self, generator_params: dict = {}, device: str = "cpu", **kwargs):
        super().__init__(device=device)
        from .pocat_generator import PocatGenerator
        self.generator = PocatGenerator(**generator_params)
        self._make_spec()
        self._set_seed(None) # 생성자에서 호출은 되어 있으나, 아래에 메소드 정의가 필요합니다.
        self.trajectory_head_stacks: List[List[int]] = []


    def _make_spec(self):
        """환경의 observation, action, reward 스펙을 정의합니다."""
        num_nodes = self.generator.num_nodes
        
        self.observation_spec = CompositeSpec({
            "nodes": Unbounded(shape=(num_nodes, FEATURE_DIM)),
            "scalar_prompt_features": Unbounded(shape=(SCALAR_PROMPT_FEATURE_DIM,)),
            "matrix_prompt_features": Unbounded(shape=(num_nodes, num_nodes)),
            "connectivity_matrix": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.bool),
            "adj_matrix": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.bool),
            "unconnected_loads_mask": Unbounded(shape=(num_nodes,), dtype=torch.bool),
            "trajectory_head": UnboundedDiscrete(shape=(1,)),
            "step_count": UnboundedDiscrete(shape=(1,)),
            "node_stages": UnboundedDiscrete(shape=(num_nodes,)),
        })
        
        self.action_spec = UnboundedDiscrete(shape=(1,))
        self.reward_spec = Unbounded(shape=(1,))

    def _set_seed(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)

    def select_start_nodes(self, td: TensorDict):
        node_types = td["nodes"][0, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        start_nodes_idx = torch.where(node_types == NODE_TYPE_LOAD)[0]
        return len(start_nodes_idx), start_nodes_idx
    
    def _trace_path_batch(self, start_nodes: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """배치 전체에 대해 start_node들의 모든 조상을 찾아 마스크로 반환합니다."""
        batch_size, num_nodes, _ = adj_matrix.shape
        path_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)
        path_mask.scatter_(1, start_nodes.unsqueeze(-1), True)
        
        # 행렬 곱셈을 이용해 그래프를 거슬러 올라가며 모든 조상을 찾습니다.
        for _ in range(num_nodes):
            # 현재 경로에 포함된 노드들의 부모를 찾습니다.
            parents_mask = (adj_matrix.float() @ path_mask.float().unsqueeze(-1)).squeeze(-1).bool()
            # 더 이상 새로운 부모가 없으면 (경로의 끝에 도달하면) 종료합니다.
            if (parents_mask & ~path_mask).sum() == 0: break
            # 새로 찾은 부모들을 경로 마스크에 추가합니다.
            path_mask |= parents_mask
        return path_mask            

    def _reset(self, td: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        batch_size = kwargs.get("batch_size", self.batch_size)
        if isinstance(batch_size, tuple): batch_size = batch_size[0]
        
        td_initial = self.generator(batch_size=batch_size).to(self.device)
        num_nodes = td_initial["nodes"].shape[1]

        self.trajectory_head_stacks = [[] for _ in range(batch_size)]

        # --- 💡 1. Trajectory 기반 상태(state) 재정의 ---
        reset_td = TensorDict({
            "nodes": td_initial["nodes"],
            "scalar_prompt_features": td_initial["scalar_prompt_features"],
            "matrix_prompt_features": td_initial["matrix_prompt_features"],
            "connectivity_matrix": td_initial["connectivity_matrix"],
            "adj_matrix": torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=self.device),
            "trajectory_head": torch.full((batch_size, 1), BATTERY_NODE_IDX, dtype=torch.long, device=self.device),
            "unconnected_loads_mask": torch.ones(batch_size, num_nodes, dtype=torch.bool, device=self.device),
            "step_count": torch.zeros(batch_size, 1, dtype=torch.long, device=self.device),
            "node_stages": torch.full((batch_size, num_nodes), -1, dtype=torch.long, device=self.device),
        }, batch_size=[batch_size], device=self.device)
        
        reset_td["node_stages"][:, BATTERY_NODE_IDX] = 0
        
        # 배터리(인덱스 0)는 항상 메인 트리에 포함
        node_types = td_initial["nodes"][0, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        is_load = node_types == NODE_TYPE_LOAD
        reset_td["unconnected_loads_mask"][:, ~is_load] = False
        reset_td.set("done", torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device))
        return reset_td

    # 💡 추가된 step 메소드: 배치 크기 검사를 우회합니다.
    def step(self, tensordict: TensorDict) -> TensorDict:
        return self._step(tensordict)

    def _calculate_power_loss(self, ic_node_features: torch.Tensor, i_out: torch.Tensor) -> torch.Tensor:
        ic_type = ic_node_features[:, :, FEATURE_INDEX["ic_type_idx"]]
        vin = ic_node_features[:, :, FEATURE_INDEX["vin_min"]]
        vout = ic_node_features[:, :, FEATURE_INDEX["vout_min"]]

        power_loss = torch.zeros_like(i_out)
        
        # LDO
        ldo_mask = ic_type == 1.0
        if ldo_mask.any():
            op_current = ic_node_features[:, :, FEATURE_INDEX["op_current"]]
            power_loss[ldo_mask] = (vin[ldo_mask] - vout[ldo_mask]) * i_out[ldo_mask] + vin[ldo_mask] * op_current[ldo_mask]
        
        # Buck
        buck_mask = ic_type == 2.0
        if buck_mask.any():
            s, e = FEATURE_INDEX["efficiency_params"]
            a, b, c = ic_node_features[:, :, s:e].permute(2, 0, 1)
            i_out_buck = i_out[buck_mask]
            power_loss[buck_mask] = a[buck_mask] * (i_out_buck**2) + b[buck_mask] * i_out_buck + c[buck_mask]
            
        return power_loss


    def _step(self, td: TensorDict) -> TensorDict:
        new_batch_size = td.batch_size[0]
        if new_batch_size > len(self.trajectory_head_stacks):
            num_repeats = new_batch_size // len(self.trajectory_head_stacks)
            self.trajectory_head_stacks = [
                s.copy() for s in self.trajectory_head_stacks for _ in range(num_repeats)
            ]

        action = td["action"].squeeze(-1)
        current_head = td["trajectory_head"].squeeze(-1)
        next_obs = td.clone()
        
        b_idx = torch.arange(new_batch_size, device=self.device)

        # Mode 1: 새 Load 선택
        head_is_battery = (current_head == BATTERY_NODE_IDX)
        if head_is_battery.any():
            selected_load = action[head_is_battery]
            next_obs["trajectory_head"][head_is_battery] = selected_load.unsqueeze(-1)
            next_obs["unconnected_loads_mask"][head_is_battery, selected_load] = False
            # 스택 업데이트 (for 루프 유지)
            for i in torch.where(head_is_battery)[0].tolist():
                load_idx = action[i].item()
                # --- 👇 [핵심 수정] 선택된 노드가 배터리이면 스택 로직을 건너뜁니다. ---
                if load_idx == BATTERY_NODE_IDX:
                    continue
                # --- 수정 완료 ---
                load_config_idx = load_idx - (1 + self.generator.num_ics)
                if self.generator.config.loads[load_config_idx].get("independent_rail_type") is not None:
                    self.trajectory_head_stacks[i].append(BATTERY_NODE_IDX)

        # Mode 2: 부모 노드 연결
        head_is_node = ~head_is_battery
        if head_is_node.any():
            # 먼저 벡터화된 연산으로 adj_matrix와 stage를 한 번에 업데이트
            child_idx_vec = current_head[head_is_node]
            parent_idx_vec = action[head_is_node]
            b_idx_node_vec = b_idx[head_is_node]

            next_obs["adj_matrix"][b_idx_node_vec, parent_idx_vec, child_idx_vec] = True
            parent_stages = next_obs["node_stages"][b_idx_node_vec, parent_idx_vec]
            next_obs["node_stages"][b_idx_node_vec, child_idx_vec] = parent_stages + 1
            
            # 그 다음, 스택과 같이 개별 상태 전환이 필요한 부분만 for 루프로 처리
            for i in torch.where(head_is_node)[0].tolist():
                parent_for_i = action[i].item()
                # 메인 트리 연결 여부를 루프 안에서 직접 확인
                is_parent_connected = (parent_for_i == BATTERY_NODE_IDX) or \
                                      (next_obs["adj_matrix"][i, :, parent_for_i].any())

                if is_parent_connected:
                    if self.trajectory_head_stacks[i]:
                        next_obs["trajectory_head"][i] = self.trajectory_head_stacks[i].pop()
                    else:
                        next_obs["trajectory_head"][i] = BATTERY_NODE_IDX
                else:
                    next_obs["trajectory_head"][i] = parent_for_i


        # 전류 및 온도 업데이트 (벡터화)
        active_currents = next_obs["nodes"][:, :, FEATURE_INDEX["current_active"]]
        new_current_out = (next_obs["adj_matrix"].float().transpose(-1, -2) @ active_currents.float().unsqueeze(-1)).squeeze(-1)
        next_obs["nodes"][:, :, FEATURE_INDEX["current_out"]] = new_current_out
        
        power_loss = self._calculate_power_loss(next_obs["nodes"], new_current_out)
        theta_ja = next_obs["nodes"][:, :, FEATURE_INDEX["theta_ja"]]
        ambient_temp = self.generator.config.constraints.get("ambient_temperature", 25.0)
        new_temp = ambient_temp + power_loss * theta_ja
        next_obs["nodes"][:, :, FEATURE_INDEX["junction_temp"]] = new_temp
        
        next_obs.set("step_count", td["step_count"] + 1)

        # 종료 조건
        next_mask = self.get_action_mask(next_obs)
        is_stuck_or_finished = ~next_mask.any(dim=-1)
        all_loads_connected = (next_obs["unconnected_loads_mask"].sum(dim=1) == 0)
        trajectory_finished = (next_obs["trajectory_head"].squeeze(-1) == BATTERY_NODE_IDX)
        done_successfully = all_loads_connected & trajectory_finished
        max_steps = 2 * self.generator.num_nodes
        timed_out = (next_obs["step_count"] > max_steps).squeeze(-1)
        is_done = done_successfully | timed_out | is_stuck_or_finished
        next_obs["done"] = is_done.unsqueeze(-1)
        
        return TensorDict({
            "next": next_obs,
            "reward": self.get_reward(next_obs, done_successfully, timed_out, is_stuck_or_finished),
            "done": next_obs["done"],
        }, batch_size=new_batch_size)

        
    # 💡 *** 여기가 핵심 수정 부분입니다 ***
    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        batch_size, num_nodes, _ = td["nodes"].shape
        mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)
        current_head = td["trajectory_head"].squeeze(-1)

        # Mode 1: 새 Load 선택 마스크
        head_is_battery = (current_head == BATTERY_NODE_IDX)
        if head_is_battery.any():
            # mask[head_is_battery] = td["unconnected_loads_mask"][head_is_battery]

    
            # 현재 배터리에 위치한 인스턴스들의 미연결 Load 마스크를 가져옵니다.
            unconnected_loads_mask_subset = td["unconnected_loads_mask"][head_is_battery]
            
            # 아직 연결할 Load가 남았는지 확인합니다.
            has_unconnected_loads = unconnected_loads_mask_subset.any(dim=-1)
            
            # --- 👇 [핵심 로직] ---
            # 1. 아직 연결할 Load가 남은 경우 (has_unconnected_loads == True)
            #    -> 오직 미연결 Load만 선택할 수 있습니다.
            if has_unconnected_loads.any():
                # 배터리에 있으면서, 연결할 Load가 남은 인스턴스들의 인덱스를 찾습니다.
                instances_with_loads = torch.where(head_is_battery)[0][has_unconnected_loads]
                # 해당 인스턴스들의 마스크는 미연결 Load 마스크가 됩니다.
                mask[instances_with_loads] = td["unconnected_loads_mask"][instances_with_loads]

            # 2. 모든 Load 연결이 끝난 경우 (has_unconnected_loads == False)
            #    -> 오직 배터리 자신만 선택할 수 있습니다.
            if (~has_unconnected_loads).any():
                # 배터리에 있으면서, 모든 Load 연결이 끝난 인스턴스들의 인덱스를 찾습니다.
                finished_instances = torch.where(head_is_battery)[0][~has_unconnected_loads]
                # 해당 인스턴스들의 마스크는 배터리 위치(인덱스 0)만 True가 됩니다.
                mask[finished_instances, BATTERY_NODE_IDX] = True
    

        # Mode 2: 부모 노드 선택 마스크
        head_is_node = ~head_is_battery
        if head_is_node.any():
            b_idx_node = torch.where(head_is_node)[0]
            child_nodes = current_head[head_is_node]
            
            can_be_parent = torch.ones(len(b_idx_node), num_nodes, dtype=torch.bool, device=self.device)

            # 1. 전압 호환성
            # connectivity_matrix[batch, parent, child] -> [b_idx_node, :, child_nodes]
            # PyTorch의 gather를 사용하여 각 배치 항목에 맞는 child 슬라이스를 선택
            connectivity = td["connectivity_matrix"][b_idx_node] # (N_node, n, n)
            child_indices_exp = child_nodes.view(-1, 1, 1).expand(-1, num_nodes, 1)
            volt_ok = torch.gather(connectivity, 2, child_indices_exp).squeeze(-1)
            can_be_parent &= volt_ok

            # 2. 사이클 방지
            path_mask = self._trace_path_batch(child_nodes, td["adj_matrix"][b_idx_node])
            can_be_parent &= ~path_mask

            mask[head_is_node] = can_be_parent
            
        return mask
    

    
    def get_reward(self, td: TensorDict, done_successfully: torch.Tensor, timed_out: torch.Tensor, is_stuck_or_finished: torch.Tensor) -> torch.Tensor:
        """
        보상을 계산합니다. 성공, 타임아웃, 갇힘 상태에 따라 다른 보상을 부여합니다.
        """
        batch_size = td.batch_size[0]
        reward = torch.zeros(batch_size, device=self.device)
        
        # 성공적으로 완료된 경우: 사용된 IC 비용의 음수값
        if done_successfully.any():
            is_used_mask = td["adj_matrix"][done_successfully].any(dim=2)
            node_costs = td["nodes"][done_successfully, :, FEATURE_INDEX["cost"]]
            ic_mask = td["nodes"][done_successfully, :, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] == 1
            used_ic_mask = is_used_mask & ic_mask
            total_cost = (node_costs * used_ic_mask).sum(dim=-1)
            reward[done_successfully] = -total_cost

        # 중간에 갇히거나 타임아웃으로 실패한 경우: 큰 패널티
        failed = (timed_out | is_stuck_or_finished) & ~done_successfully
        if failed.any():
            reward[failed] -= 100.0 # 예시 패널티 값
            
        return reward
