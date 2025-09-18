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
        self._load_constraint_info()


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
            "children_count": UnboundedDiscrete(shape=(num_nodes,)),
            "is_exclusive_path": Unbounded(shape=(num_nodes,), dtype=torch.bool),

        })
        
        self.action_spec = UnboundedDiscrete(shape=(1,))
        self.reward_spec = Unbounded(shape=(1,))

    def _set_seed(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)

    # 💡 **[변경 3]** 제약조건 정보를 미리 가공하여 저장하는 헬퍼 함수
    def _load_constraint_info(self):
        """config 파일에서 제약조건 정보를 로드하고 마스킹에 사용하기 쉽게 가공합니다."""
        self.node_name_to_idx = {name: i for i, name in enumerate(self.generator.config.node_names)}
        
        # Independent Rail 정보
        self.exclusive_supplier_loads = set()
        self.exclusive_path_loads = set()

        self.exclusive_path_loads_tensor = torch.tensor([], dtype=torch.long, device=self.device)
        loads_config = self.generator.config.loads
        if loads_config:
            for i, load_cfg in enumerate(loads_config):
                load_idx = 1 + self.generator.num_ics + i
                if load_cfg.get("independent_rail_type") == "exclusive_supplier":
                    self.exclusive_supplier_loads.add(load_idx)
                elif load_cfg.get("independent_rail_type") == "exclusive_path":
                    self.exclusive_path_loads.add(load_idx)
            # set에 정보가 채워진 후 tensor를 생성합니다.
            self.exclusive_path_loads_tensor = torch.tensor(list(self.exclusive_path_loads), dtype=torch.long, device=self.device)

        # Power Sequence 정보에 f 플래그(동시 허용 여부) 추가
        self.power_sequences = []
        for seq in self.generator.config.constraints.get("power_sequences", []):
            f_flag = seq.get("f", 1)
            j_idx = self.node_name_to_idx.get(seq['j'])
            k_idx = self.node_name_to_idx.get(seq['k'])
            if j_idx is not None and k_idx is not None:
                self.power_sequences.append((j_idx, k_idx, f_flag))




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
        if td is None:
            batch_size = kwargs.get("batch_size", self.batch_size)
            if isinstance(batch_size, tuple): batch_size = batch_size[0]
            td_initial = self.generator(batch_size=batch_size).to(self.device)
        # td가 인자로 들어오면, 그 td를 초기 상태로 사용합니다.
        else:
            td_initial = td
            # 배치 크기도 들어온 td에서 가져옵니다.
            batch_size = td_initial.batch_size[0]

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
            "children_count": torch.zeros(batch_size, num_nodes, dtype=torch.long, device=self.device),
            "is_exclusive_path": torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device),

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

        action = td["action"].view(-1)
        current_head = td["trajectory_head"].view(-1)
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
                if load_idx in self.exclusive_path_loads:
                    self.trajectory_head_stacks[i].append(BATTERY_NODE_IDX)

        # Mode 2: 부모 노드 연결
        head_is_node = ~head_is_battery
        if head_is_node.any():
            # 먼저 벡터화된 연산으로 adj_matrix와 stage를 한 번에 업데이트
            child_idx_vec = current_head[head_is_node]
            parent_idx_vec = action[head_is_node]
            b_idx_node_vec = b_idx[head_is_node]

            next_obs["adj_matrix"][b_idx_node_vec, parent_idx_vec, child_idx_vec] = True

            # 💡 **[변경 6]** 자식 수 및 경로 상태 업데이트
            next_obs["children_count"][b_idx_node_vec, parent_idx_vec] += 1
            # 💡 **[버그 수정]** exclusive_path 상태 전파 로직 개선
            # 독립 경로로 지정된 부하가 연결되었는지 확인
            is_child_initially_exclusive = (child_idx_vec.unsqueeze(1) == self.exclusive_path_loads_tensor).any(dim=1)
            path_starts = torch.where(is_child_initially_exclusive, child_idx_vec, -1)

            # 독립 경로가 시작되었다면
            if (path_starts != -1).any():
                active_indices = torch.where(path_starts != -1)[0]
                # 해당 노드의 모든 조상을 찾음
                ancestors = self._trace_path_batch(path_starts[active_indices], next_obs["adj_matrix"][b_idx_node_vec[active_indices]])
                # 모든 조상에게 is_exclusive_path 상태를 True로 전파
                next_obs["is_exclusive_path"][b_idx_node_vec[active_indices]] |= ancestors

            
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

        # 💡 디버깅 로그를 위한 설정
        is_debug_instance = td.batch_size[0] > 0 and td.get("log_mode", "progress") == "detail"
        debug_idx = td.get("log_idx", 0) if is_debug_instance else -1


        # Mode 1: 새 Load 선택 마스크
        head_is_battery = (current_head == BATTERY_NODE_IDX)
        if head_is_battery.any():
            # 💡 **[버그 수정]** 텐서 크기 불일치 오류 해결
            all_has_unconnected = td["unconnected_loads_mask"].any(dim=-1)
            is_active = head_is_battery & all_has_unconnected
            is_finished = head_is_battery & ~all_has_unconnected
            
            if is_active.any():
                mask[is_active] = td["unconnected_loads_mask"][is_active]
            
            if is_finished.any():
                mask[is_finished, BATTERY_NODE_IDX] = True

    

        # Mode 2: 부모 노드 선택 마스크
        head_is_node = ~head_is_battery
        if head_is_node.any():
            b_idx_node = torch.where(head_is_node)[0]
            child_nodes = current_head[head_is_node]
            
            can_be_parent = torch.ones(len(b_idx_node), num_nodes, dtype=torch.bool, device=self.device)

            if debug_idx != -1 and debug_idx in b_idx_node.tolist():
                local_debug_idx = (b_idx_node == debug_idx).nonzero().item()
                if can_be_parent[local_debug_idx, BATTERY_NODE_IDX]: print("    - [DEBUG] Battery is valid before any constraint.")


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

            if debug_idx != -1 and debug_idx in b_idx_node.tolist():
                local_debug_idx = (b_idx_node == debug_idx).nonzero().item()
                if not can_be_parent[local_debug_idx, BATTERY_NODE_IDX]: print("    - [DEBUG] Battery REJECTED by Voltage/Cycle.")

            

            # 3. 전류 한계 마스킹
            parent_i_limit = td["nodes"][:, :, FEATURE_INDEX["i_limit"]]
            parent_i_out = td["nodes"][:, :, FEATURE_INDEX["current_out"]]
            remaining_capacity = (parent_i_limit - parent_i_out)[b_idx_node] # (N_node, num_nodes)
            child_current_draw = td["nodes"][b_idx_node, child_nodes, FEATURE_INDEX["current_active"]].unsqueeze(1)
            current_ok = remaining_capacity >= child_current_draw
            can_be_parent &= current_ok
            if debug_idx != -1 and debug_idx in b_idx_node.tolist():
                local_debug_idx = (b_idx_node == debug_idx).nonzero().item()
                if not can_be_parent[local_debug_idx, BATTERY_NODE_IDX]: print("    - [DEBUG] Battery REJECTED by Current Limit.")

            
            # 4. Independent Rail 마스킹
            # 4a. exclusive_supplier: 해당 Load를 자식으로 가지는 부모는 다른 자식을 가질 수 없음
            for load_idx in self.exclusive_supplier_loads:
                is_this_child = (child_nodes == load_idx)
                if is_this_child.any():
                    # 부모가 될 후보 IC들의 현재 자식 수를 확인
                    parent_children_count = td["children_count"][b_idx_node[is_this_child]]
                    # 자식이 이미 1명 이상인 부모는 선택 불가
                    can_be_parent[is_this_child] &= (parent_children_count == 0)

            # 4b. exclusive_path: 경로 상의 모든 부모는 자식을 1명만 가질 수 있음
            is_child_exclusive = td["is_exclusive_path"][b_idx_node, child_nodes]
            if is_child_exclusive.any():
                instances_to_constrain = torch.where(is_child_exclusive)[0]
                if len(instances_to_constrain) > 0:
                    b_idx_constr = b_idx_node[instances_to_constrain]

                    
                    is_battery_mask = torch.arange(num_nodes, device=self.device) == BATTERY_NODE_IDX
                    
                    parent_children_count = td["children_count"][b_idx_constr]
                    children_ok = (parent_children_count == 0) | is_battery_mask
                    can_be_parent[instances_to_constrain] &= children_ok
                    
                    parent_is_exclusive = td["is_exclusive_path"][b_idx_constr]
                    exclusive_ok = ~parent_is_exclusive | is_battery_mask
                    can_be_parent[instances_to_constrain] &= exclusive_ok
            if debug_idx != -1 and debug_idx in b_idx_node.tolist():
                local_debug_idx = (b_idx_node == debug_idx).nonzero().item()
                if not can_be_parent[local_debug_idx, BATTERY_NODE_IDX]: print("    - [DEBUG] Battery REJECTED by Independent Rail.")



            # 5. Power Sequence 마스킹
            for j_idx, k_idx, f_flag in self.power_sequences: # Rule: J before K
                # Case 1: 현재 child가 'k'일 때 (k의 부모를 찾는 중)
                is_k = (child_nodes == k_idx)
                if is_k.any():
                    is_j_connected = td["adj_matrix"][b_idx_node[is_k], :, j_idx].any(dim=-1)
                    if is_j_connected.any():
                        instances_to_constrain = torch.where(is_k & is_j_connected.unsqueeze(0).transpose(0, 1))[0]
                        if len(instances_to_constrain) > 0:
                            b_idx_constr = b_idx_node[instances_to_constrain]
                            parent_of_j_idx = td["adj_matrix"][b_idx_constr, :, j_idx].long().argmax(-1)
                            stage_of_j_parent = td["node_stages"][b_idx_constr, parent_of_j_idx]
                            candidate_parent_stages = td["node_stages"][b_idx_constr]
                            is_candidate_unconnected = (candidate_parent_stages == -1)
                            stage_ok = (candidate_parent_stages > stage_of_j_parent.unsqueeze(1)) if f_flag == 1 else (candidate_parent_stages >= stage_of_j_parent.unsqueeze(1))
                            can_be_parent[instances_to_constrain] &= is_candidate_unconnected | stage_ok

                    
                # Case 2: 현재 child가 'j'일 때 (j의 부모를 찾는 중)
                is_j = (child_nodes == j_idx)
                if is_j.any():
                    is_k_connected = td["adj_matrix"][b_idx_node[is_j], :, k_idx].any(dim=-1)
                    if is_k_connected.any():
                        instances_to_constrain = torch.where(is_j & is_k_connected.unsqueeze(0).transpose(0, 1))[0]
                        if len(instances_to_constrain) > 0:
                            b_idx_constr = b_idx_node[instances_to_constrain]
                            parent_of_k_idx = td["adj_matrix"][b_idx_constr, :, k_idx].long().argmax(-1)
                            stage_of_k_parent = td["node_stages"][b_idx_constr, parent_of_k_idx]
                            candidate_parent_stages = td["node_stages"][b_idx_constr]
                            is_candidate_unconnected = (candidate_parent_stages == -1)
                            stage_ok = (candidate_parent_stages < stage_of_k_parent.unsqueeze(1)) if f_flag == 1 else (candidate_parent_stages <= stage_of_k_parent.unsqueeze(1))
                            can_be_parent[instances_to_constrain] &= is_candidate_unconnected | stage_ok
                    
                    if f_flag == 1: # 엄격한 순서일때만 같은 부모 금지
                        is_parent_of_k = td["adj_matrix"][b_idx_node[is_j], :, k_idx]
                        can_be_parent[is_j] &= ~is_parent_of_k
            if debug_idx != -1 and debug_idx in b_idx_node.tolist():
                local_debug_idx = (b_idx_node == debug_idx).nonzero().item()
                if not can_be_parent[local_debug_idx, BATTERY_NODE_IDX]: print("    - [DEBUG] Battery REJECTED by Power Sequence.")
   

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