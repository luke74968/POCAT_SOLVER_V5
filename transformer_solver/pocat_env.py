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
    
    def _trace_path_batch(self, b_idx, start_nodes, adj_matrix):
        """배치 전체에 대해 start_node들의 모든 조상을 찾아 마스크로 반환합니다."""
        num_nodes = adj_matrix.shape[-1]
        
        # 선택된 배치 항목들에 대한 인접 행렬
        adj_b = adj_matrix[b_idx]
        
        # 경로 마스크 초기화 (시작 노드만 True)
        path_mask = torch.zeros(len(b_idx), num_nodes, dtype=torch.bool, device=self.device)
        path_mask[torch.arange(len(b_idx)), start_nodes] = True
        
        # 행렬 곱셈을 이용해 그래프를 거슬러 올라가며 모든 조상을 찾습니다.
        for _ in range(num_nodes):
            # 현재 경로에 포함된 노드들의 부모를 찾습니다.
            parents_mask = (adj_b.float() @ path_mask.float().unsqueeze(-1)).squeeze(-1) > 0
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

    def _calculate_power_loss(self, ic_node_feature: torch.Tensor, i_out: float) -> float:
        ic_type = ic_node_feature[FEATURE_INDEX["ic_type_idx"]].item()
        vin = ic_node_feature[FEATURE_INDEX["vin_min"]].item()
        vout = ic_node_feature[FEATURE_INDEX["vout_min"]].item()
        if ic_type == 1.0: # LDO
            op_current = ic_node_feature[FEATURE_INDEX["op_current"]].item()
            return (vin - vout) * i_out + vin * op_current
        elif ic_type == 2.0: # Buck
            s, e = FEATURE_INDEX["efficiency_params"]
            a, b, c = ic_node_feature[s:e]
            return a * (i_out**2) + b * i_out + c
        return 0


    def _step(self, td: TensorDict) -> TensorDict:
        new_batch_size = td.batch_size[0]
        if new_batch_size > len(self.trajectory_head_stacks):
            num_repeats = new_batch_size // len(self.trajectory_head_stacks)
            self.trajectory_head_stacks = [
                stack.copy() for stack in self.trajectory_head_stacks for _ in range(num_repeats)
            ]

        action = td["action"].squeeze(-1)
        current_head = td["trajectory_head"].squeeze(-1)
        next_obs = td.clone()
        batch_size = td.batch_size[0]
        num_nodes = td["nodes"].shape[1]
        
        for i in range(batch_size):
            head, act = current_head[i].item(), action[i].item()
            if head == BATTERY_NODE_IDX:
                load_idx_in_config = act - (1 + self.generator.num_ics)
                load_info = self.generator.config.loads[load_idx_in_config]
                if load_info.get("independent_rail_type") is not None:
                    self.trajectory_head_stacks[i].append(head)
                next_obs["trajectory_head"][i] = act
                next_obs["unconnected_loads_mask"][i, act] = False
            else:
                child_idx, parent_idx = head, act
                next_obs["adj_matrix"][i, parent_idx, child_idx] = True
                parent_stage = next_obs["node_stages"][i, parent_idx].item()
                next_obs["node_stages"][i, child_idx] = parent_stage + 1
                
                is_parent_connected = (parent_idx == BATTERY_NODE_IDX) or \
                                      (next_obs["adj_matrix"][i, :, parent_idx].any())
                
                if is_parent_connected:
                    if self.trajectory_head_stacks[i]:
                        next_obs["trajectory_head"][i] = self.trajectory_head_stacks[i].pop()
                    else:
                        next_obs["trajectory_head"][i] = BATTERY_NODE_IDX
                else:
                    next_obs["trajectory_head"][i] = parent_idx
        
        # 전류 및 온도 업데이트 (배치 연산)
        new_current_out = (next_obs["adj_matrix"].float().transpose(-1, -2) @ \
                           next_obs["nodes"][:, :, FEATURE_INDEX["current_active"]].float().unsqueeze(-1)).squeeze(-1)
        next_obs["nodes"][:, :, FEATURE_INDEX["current_out"]] = new_current_out
        
        ambient_temp = self.generator.config.constraints.get("ambient_temperature", 25.0)
        for i in range(batch_size):
            for n_idx in range(num_nodes):
                node_feat = next_obs["nodes"][i, n_idx]
                if node_feat[FEATURE_INDEX["node_type"][0]+NODE_TYPE_IC]:
                    power_loss = self._calculate_power_loss(node_feat, new_current_out[i, n_idx].item())
                    theta_ja = node_feat[FEATURE_INDEX["theta_ja"]].item()
                    next_obs["nodes"][i, n_idx, FEATURE_INDEX["junction_temp"]] = ambient_temp + power_loss * theta_ja
        
        next_obs.set("step_count", td["step_count"] + 1)
        all_loads_connected = (next_obs["unconnected_loads_mask"].sum(dim=1) == 0)
        is_done = all_loads_connected & (next_obs["trajectory_head"].squeeze(-1) == BATTERY_NODE_IDX)
        
        return TensorDict({"next": next_obs, "reward": self.get_reward(next_obs, is_done), "done": is_done.unsqueeze(-1)}, batch_size=td.batch_size)

        
    # 💡 *** 여기가 핵심 수정 부분입니다 ***
    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        batch_size, num_nodes, _ = td["nodes"].shape
        mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)
        current_head = td["trajectory_head"].squeeze(-1)
        load_configs = self.generator.config.loads
        num_ics = self.generator.num_ics

        b_idx = torch.arange(batch_size, device=self.device)

        # --- 👇 [핵심 수정] Single-Trajectory 액션 마스킹 로직 ---
        # Case 1: 현재 헤드가 배터리 -> '새 Load 선택' 마스크 생성
        head_is_battery_mask = (current_head == BATTERY_NODE_IDX)
        if head_is_battery_mask.any():
            mask[head_is_battery_mask] = td["unconnected_loads_mask"][head_is_battery_mask]

        # Phase 1: 현재 경로를 이을 부모 노드 선택
        head_is_node_mask = ~head_is_battery_mask
        if head_is_node_mask.any():
            b_select_parent = b_idx[head_is_node_mask]
            child_indices = current_head[b_select_parent]

            node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
            

            # 1. 기본 자격 정의: 부모는 IC 또는 배터리여야만 함 (Load 원천 배제)
            parent_candidate_mask = (node_types == NODE_TYPE_IC) | (node_types == NODE_TYPE_BATTERY)
            can_be_parent = parent_candidate_mask.unsqueeze(0).expand(len(b_idx), -1).clone()

            # 2. 사이클 방지: 현재 경로의 조상 및 자손은 부모가 될 수 없음
            #    자기 자신도 포함하여 확실히 제외합니다.
            ancestor_mask = self._trace_path_batch(b_idx, child_indices, td["adj_matrix"])
            can_be_parent &= ~ancestor_mask

            # 3. 전압 호환성 검사
            is_voltage_compatible = td["connectivity_matrix"][b_idx, :, child_indices]
            can_be_parent &= is_voltage_compatible

            # 4. 전류 한계 검사
            current_path_mask = self._trace_path_batch(b_idx, child_indices, td["adj_matrix"])
            path_nodes_currents = (td["nodes"][b_idx, :, FEATURE_INDEX["current_active"]] * current_path_mask).sum(dim=1)
            current_draw = td["nodes"][b_idx, :, FEATURE_INDEX["current_out"]]
            prospective_draw = current_draw + path_nodes_currents.unsqueeze(1)
            parent_limits = td["nodes"][b_idx, :, FEATURE_INDEX["i_limit"]]
            can_be_parent &= (prospective_draw <= parent_limits) | (parent_limits == 0)

            # 5. 기타 제약조건 (Power Sequence, Independent Rail)
            constraints, loads_info, node_names = self.generator.config.constraints, self.generator.config.loads, self.generator.config.node_names
            # ancestors 텐서 생성 시 device를 지정합니다.
            ancestors = td["adj_matrix"][b_idx].clone().to(self.device) 
            for _ in range(num_nodes): # 최악의 경우를 대비해 num_nodes 만큼 반복
                ancestors = ancestors | (ancestors.float() @ ancestors.float()).bool()
            
            head_load_idx = child_indices - (1 + len(self.generator.config.available_ics))

            # 아래의 for문은 배치별로 순회하므로 병렬 처리가 어렵지만,
            # 그 안의 텐서 연산은 이미 GPU에서 수행되도록 변경되었습니다.

            for idx, b in enumerate(b_idx.tolist()):
                if 0 <= head_load_idx[idx] < len(loads_info):
                    load = loads_info[head_load_idx[idx]]
                    rail_type = load.get("independent_rail_type")
                    # << 수정: 배터리(노드 0)는 이 제약에서 제외하도록 수정 >>
                    is_not_battery_mask = torch.ones(num_nodes, dtype=torch.bool, device=self.device)
                    is_not_battery_mask[0] = False
                    
                    if rail_type == "exclusive_supplier":
                        # 이미 자식이 있는 노드는 부모가 될 수 없음 (단, 배터리는 예외)
                        no_existing_children_mask = td["adj_matrix"][b].sum(dim=1) == 0
                        can_be_parent[idx] &= (no_existing_children_mask | ~is_not_battery_mask)
                    elif rail_type == "exclusive_path":
                        # 이미 자식이 2개 이상 있는 노드는 부모가 될 수 없음 (단, 배터리는 예외)
                        less_than_two_children_mask = td["adj_matrix"][b].sum(dim=1) <= 1
                        can_be_parent[idx] &= (less_than_two_children_mask | ~is_not_battery_mask)

            
            for seq in constraints.get("power_sequences", []):
                if seq.get("f") != 1: continue
                j_name, k_name = seq.get("j"), seq.get("k")
                if j_name not in node_names or k_name not in node_names: continue
                j_idx, k_idx = node_names.index(j_name), node_names.index(k_name)
                is_head_k_mask = child_indices == k_idx
                if is_head_k_mask.any():
                    can_be_parent[is_head_k_mask] &= ~ancestors[is_head_k_mask, :, j_idx]
                        # 1. 부하는 부모가 될 수 없음
            is_load = (node_types == NODE_TYPE_LOAD)
            can_be_parent &= ~is_load.unsqueeze(0)


            mask[b_idx] = can_be_parent
            
        return mask
    

    
    def get_reward(self, td: TensorDict, timed_out: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reward based on the final state of the power tree.
        The reward is the negative of the total cost of used ICs.
        This function is called only when an episode is done.
        """
        reward = torch.zeros(td.batch_size[0], device=self.device)
        done = td["done"].squeeze(-1)
        
        # 성공적으로 완료된 경우
        done_success = done & ~timed_out
        if done_success.any():
            is_used_mask = td["adj_matrix"][done_success].any(dim=1) | td["adj_matrix"][done_success].any(dim=2)
            node_costs = td["nodes"][done_success, :, FEATURE_INDEX["cost"]]
            ic_mask = td["nodes"][done_success, :, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] == 1
            used_ic_mask = is_used_mask & ic_mask
            total_cost = (node_costs * used_ic_mask).sum(dim=-1)
            reward[done_success] = -total_cost

        # 💡 *** 여기가 핵심 수정 부분입니다 (2/2) ***
        # 시간 초과로 실패한 경우 패널티를 부여합니다.
        if timed_out.any():
            # 연결하지 못한 Load의 수만큼 큰 패널티를 부여합니다.
            unconnected_loads = td["unconnected_loads_mask"][timed_out].sum(dim=1).float()
            reward[timed_out] -= unconnected_loads * 10.0 # 패널티 값
            
        return reward