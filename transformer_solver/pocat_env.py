# transformer_solver/pocat_env.py
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Optional, Tuple
from torchrl.data import UnboundedContinuousTensorSpec as Unbounded, \
    UnboundedDiscreteTensorSpec as UnboundedDiscrete, \
    DiscreteTensorSpec as Categorical, \
    CompositeSpec as Composite

from common.pocat_defs import SCALAR_PROMPT_FEATURE_DIM

from common.pocat_defs import (
    NODE_TYPE_BATTERY, NODE_TYPE_IC, NODE_TYPE_LOAD,
    FEATURE_DIM, FEATURE_INDEX
)


class PocatEnv(EnvBase):
    name = "pocat"

    def __init__(self, generator_params: dict = {}, device: str = "cpu", **kwargs):
        super().__init__(device=device)
        from .pocat_generator import PocatGenerator
        self.generator = PocatGenerator(**generator_params)
        self._make_spec()
        self._set_seed(None) # 생성자에서 호출은 되어 있으나, 아래에 메소드 정의가 필요합니다.

    def _make_spec(self):
        """환경의 observation, action, reward 스펙을 정의합니다."""
        num_nodes = self.generator.num_nodes
        
        self.observation_spec = Composite({
            "nodes": Unbounded(shape=(num_nodes, FEATURE_DIM), dtype=torch.float32),
            "scalar_prompt_features": Unbounded(shape=(SCALAR_PROMPT_FEATURE_DIM,), dtype=torch.float32),
            "matrix_prompt_features": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.float32),
            "adj_matrix": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.bool),
            "main_tree_mask": Unbounded(shape=(num_nodes,), dtype=torch.bool),
            "ic_current_draw": Unbounded(shape=(num_nodes,), dtype=torch.float32),
            "decoding_phase": Categorical(shape=(1,), n=2, dtype=torch.long),
            "trajectory_head": UnboundedDiscrete(shape=(1,), dtype=torch.long),
            "unconnected_loads_mask": Unbounded(shape=(num_nodes,), dtype=torch.bool),
            "step_count": UnboundedDiscrete(shape=(1,), dtype=torch.long),
        })
        
        self.action_spec = UnboundedDiscrete(shape=(2,), dtype=torch.long)
        self.reward_spec = Unbounded(shape=(1,))

    def _set_seed(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)

    def select_start_nodes(self, td: TensorDict) -> Tuple[int, torch.Tensor]:
        node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
        start_nodes_idx = torch.where(node_types == NODE_TYPE_LOAD)[0]
        return len(start_nodes_idx), start_nodes_idx
    
    def _trace_path_batch(self, b_idx: torch.Tensor, start_nodes: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
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
            if (parents_mask & ~path_mask).sum() == 0:
                break
            
            # 새로 찾은 부모들을 경로 마스크에 추가합니다.
            path_mask |= parents_mask
            
        return path_mask            

    def _reset(self, td: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        if td is None:
            batch_size = kwargs.get("batch_size", self.batch_size)
            if not isinstance(batch_size, int): batch_size = batch_size[0]
            td = self.generator(batch_size=batch_size).to(self.device)
            
        num_nodes = td["nodes"].shape[1]
        batch_size = td.batch_size[0]
        
        # --- 💡 1. Trajectory 기반 상태(state) 재정의 ---
        reset_td = TensorDict({
            "nodes": td["nodes"],
            "scalar_prompt_features": td["scalar_prompt_features"],
            "matrix_prompt_features": td["matrix_prompt_features"],
            "adj_matrix": torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=self.device),
            "main_tree_mask": torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device),
            "ic_current_draw": torch.zeros(batch_size, num_nodes, device=self.device),
            
            # --- 새로운 상태 변수 ---
            # 0: 새 Load 선택 단계, 1: Trajectory(경로) 구축 단계
            "decoding_phase": torch.zeros(batch_size, 1, dtype=torch.long, device=self.device),
            # 현재 만들고 있는 경로의 가장 끝 노드 (Load에서 배터리 방향으로)
            "trajectory_head": torch.full((batch_size, 1), -1, dtype=torch.long, device=self.device),
            # 아직 트리에 연결되지 않은 Load들의 마스크
            "unconnected_loads_mask": torch.ones(batch_size, num_nodes, dtype=torch.bool, device=self.device),
            
            "step_count": torch.zeros(batch_size, 1, dtype=torch.long, device=self.device),
        }, batch_size=[batch_size], device=self.device)
        
        reset_td.set("done", torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device))
        
        # 배터리(인덱스 0)는 항상 메인 트리에 포함
        reset_td["main_tree_mask"][:, 0] = True
        
        node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
        is_load = node_types == NODE_TYPE_LOAD
        reset_td["unconnected_loads_mask"][:, ~is_load] = False
        
        return reset_td

    # 💡 추가된 step 메소드: 배치 크기 검사를 우회합니다.
    def step(self, tensordict: TensorDict) -> TensorDict:
        return self._step(tensordict)

    def _step(self, td: TensorDict) -> TensorDict:
        action = td["action"]
        b_idx = torch.arange(td.batch_size[0], device=self.device)
        phase = td["decoding_phase"].squeeze(-1)
        next_obs = td.clone()

        # Phase 0: 새 Load 선택
        phase0_mask = phase == 0
        if phase0_mask.any():
            b_phase0 = b_idx[phase0_mask]
            selected_load = action[b_phase0, 0]
            next_obs["trajectory_head"][b_phase0] = selected_load.unsqueeze(-1)
            next_obs["unconnected_loads_mask"][b_phase0, selected_load] = False
            next_obs["decoding_phase"][b_phase0] = 1

        # Phase 1: Trajectory(경로) 구축
        phase1_mask = phase == 1
        phase1_mask = phase == 1
        if phase1_mask.any():
            b_phase1 = b_idx[phase1_mask]
            child_idx, parent_idx = action[b_phase1, 0], action[b_phase1, 1]
            next_obs["adj_matrix"][b_phase1, parent_idx, child_idx] = True

            path_nodes_mask = self._trace_path_batch(b_phase1, child_idx, next_obs["adj_matrix"])
            path_nodes_currents = (td["nodes"][b_phase1] * path_nodes_mask.unsqueeze(-1))[:, :, FEATURE_INDEX["current_active"]]
            total_child_currents = path_nodes_currents.sum(dim=1)
            
            ancestor_mask = self._trace_path_batch(b_phase1, parent_idx, next_obs["adj_matrix"])
            battery_mask = torch.zeros_like(ancestor_mask); battery_mask[:, 0] = True
            ancestor_mask_no_battery = ancestor_mask & ~battery_mask
            
            current_draw_update = ancestor_mask_no_battery.float().transpose(-1, -2) @ total_child_currents.float().unsqueeze(-1)
            next_obs["ic_current_draw"][b_phase1] += current_draw_update.squeeze(-1)

            is_parent_in_main_tree = next_obs["main_tree_mask"][b_phase1, parent_idx]
            
            b_connected = b_phase1[is_parent_in_main_tree]
            if b_connected.numel() > 0:
                child_connected = child_idx[is_parent_in_main_tree]
                newly_connected_path_mask = self._trace_path_batch(b_connected, child_connected, next_obs["adj_matrix"])
                next_obs["main_tree_mask"][b_connected] |= newly_connected_path_mask
                next_obs["decoding_phase"][b_connected] = 0

            b_not_connected = b_phase1[~is_parent_in_main_tree]
            if b_not_connected.numel() > 0:
                parent_not_connected = parent_idx[~is_parent_in_main_tree]
                next_obs["trajectory_head"][b_not_connected] = parent_not_connected.unsqueeze(-1)
        
        next_obs.set("step_count", td["step_count"] + 1)

        # 💡 *** 여기가 핵심 수정 부분입니다 (1/2) ***
        # 1. 모든 부하가 연결되었는지 확인
        all_loads_connected = (next_obs["unconnected_loads_mask"].sum(dim=1) == 0)
        # 2. 현재 새로운 경로를 만들고 있지 않은지 확인 (모든 경로가 주 전력망에 연결되었는지)
        in_selection_phase = (next_obs["decoding_phase"].squeeze(-1) == 0)
        # 3. 두 조건을 모두 만족해야 성공적으로 종료
        done_successfully = all_loads_connected & in_selection_phase
        
        # 4. 타임아웃(안전망) 확인
        max_steps = 2 * self.generator.num_nodes
        timed_out = (next_obs["step_count"] > max_steps).squeeze(-1)
        
        is_done = done_successfully | timed_out
        next_obs["done"] = is_done.unsqueeze(-1)
        
        return TensorDict({
            "next": next_obs,
            "reward": self.get_reward(next_obs, timed_out),
            "done": next_obs["done"],
        }, batch_size=td.batch_size)
        
    # 💡 *** 여기가 핵심 수정 부분입니다 ***
    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        batch_size, num_nodes, _ = td["nodes"].shape
        # 텐서 생성 시 device를 명시적으로 지정합니다.
        mask = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=self.device)

        # Phase 0: 아직 연결되지 않은 Load만 선택 가능
        phase0_mask = (td["decoding_phase"].squeeze(-1) == 0)
        if phase0_mask.any():
            mask[phase0_mask, :, 0] = td["unconnected_loads_mask"][phase0_mask]

        # Phase 1: 현재 경로를 이을 부모 노드 선택
        phase1_mask = ~phase0_mask
        if phase1_mask.any():
            b_idx = torch.where(phase1_mask)[0]
            child_indices = td["trajectory_head"][b_idx].squeeze(-1)

            # can_be_parent 텐서 생성 시 device를 지정합니다.
            can_be_parent = torch.ones(len(b_idx), num_nodes, dtype=torch.bool, device=self.device)
            node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
            


            # 2. 현재 만들고 있는 경로에 포함된 노드는 부모가 될 수 없음 (사이클 방지)
            current_path_mask = self._trace_path_batch(b_idx, child_indices, td["adj_matrix"])
            can_be_parent &= ~current_path_mask


            # 3. 전압 호환성 검사
            child_vin_min = td["nodes"][b_idx, child_indices, FEATURE_INDEX["vin_min"]]
            child_vin_max = td["nodes"][b_idx, child_indices, FEATURE_INDEX["vin_max"]]
            
            parent_vout_min = td["nodes"][b_idx, :, FEATURE_INDEX["vout_min"]]
            parent_vout_max = td["nodes"][b_idx, :, FEATURE_INDEX["vout_max"]]

            is_voltage_compatible = (parent_vout_min <= child_vin_max.unsqueeze(1)) & \
                                    (parent_vout_max >= child_vin_min.unsqueeze(1))
            can_be_parent &= is_voltage_compatible

            # 4. 전류 한계 검사
            path_nodes_currents = (td["nodes"][b_idx, :, FEATURE_INDEX["current_active"]] * current_path_mask).sum(dim=1)
            prospective_draw = td["ic_current_draw"][b_idx] + path_nodes_currents.unsqueeze(1)
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
                    if rail_type == "exclusive_supplier":
                        # adj_matrix.sum() 연산이 GPU에서 수행됩니다.
                        can_be_parent[idx] &= td["adj_matrix"][b].sum(dim=1) == 0
                    elif rail_type == "exclusive_path":
                        can_be_parent[idx] &= td["adj_matrix"][b].sum(dim=1) <= 1
            
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

            mask[b_idx, :, child_indices] = can_be_parent
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