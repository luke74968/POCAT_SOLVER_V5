# transformer_solver/pocat_env.py

import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Optional, Dict, Union

from torchrl.data import Unbounded, UnboundedDiscrete, CompositeSpec

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
        
        # 버퍼는 _ensure_buffers에서 동적으로 생성되므로 __init__에서는 None으로 초기화
        self.register_buffer("arange_nodes", None, persistent=False)
        self.register_buffer("node_type_tensor", None, persistent=False)
        self.register_buffer("rail_types", None, persistent=False)

        self._make_spec()
        self._load_constraint_info()

    # --- [개선] 버퍼 크기 동기화 함수 추가 ---
    def _ensure_buffers(self, td: TensorDict):
        """에피소드마다 그래프/로드 수가 바뀔 경우를 대비해 버퍼를 동기화합니다."""
        num_nodes = td["nodes"].shape[1]

        if self.arange_nodes is None or self.arange_nodes.numel() != num_nodes:
            self.arange_nodes = torch.arange(num_nodes, device=self.device)
        
        # node_type_tensor는 config에서 오므로 고정, __init__에서 한 번만 생성되도록 수정
        if self.node_type_tensor is None:
            node_types_list = [self.generator.config.node_types[i] for i in range(num_nodes)]
            self.node_type_tensor = torch.tensor(node_types_list, dtype=torch.long, device=self.device)

        # rail_types도 config에서 오므로 고정
        if self.rail_types is None:
            rail_type_map = {"exclusive_supplier": 1, "exclusive_path": 2}
            load_configs = self.generator.config.loads
            rail_types_list = [rail_type_map.get(cfg.get("independent_rail_type"), 0) for cfg in load_configs]
            self.rail_types = torch.tensor(rail_types_list, dtype=torch.long, device=self.device)

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
            # --- 👇 [여기에 새로운 상태 명세를 추가합니다] ---
            "current_cost": Unbounded(shape=(1,)),
            "is_used_ic_mask": Unbounded(shape=(num_nodes,), dtype=torch.bool),
            "is_locked_ic_mask": Unbounded(shape=(num_nodes,), dtype=torch.bool),
            "current_target_load": UnboundedDiscrete(shape=(1,)),
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

        loads_config = self.generator.config.loads
        if loads_config:
            load_start_idx = 1 + self.generator.num_ics
            for i, load_cfg in enumerate(loads_config):
                load_idx = load_start_idx + i
                if load_cfg.get("independent_rail_type") == "exclusive_supplier":
                    self.exclusive_supplier_loads.add(load_idx)
                elif load_cfg.get("independent_rail_type") == "exclusive_path":
                    self.exclusive_path_loads.add(load_idx)
            # set에 정보가 채워진 후 tensor를 생성합니다.
            if self.exclusive_path_loads:
                self.exclusive_path_loads_tensor = torch.tensor(
                    sorted(self.exclusive_path_loads), dtype=torch.long, device=self.device
                )
        if self.exclusive_supplier_loads:
            self.exclusive_supplier_loads_tensor = torch.tensor(
                sorted(self.exclusive_supplier_loads), dtype=torch.long, device=self.device
            )
        else:
            self.exclusive_supplier_loads_tensor = torch.tensor([], dtype=torch.long, device=self.device)
        if not self.exclusive_path_loads:
            self.exclusive_path_loads_tensor = torch.tensor([], dtype=torch.long, device=self.device)

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

        # start_nodes가 비어있지 않을 때만 scatter_ 실행
        if start_nodes.numel() > 0:
            path_mask.scatter_(1, start_nodes.unsqueeze(-1), True)

        # 행렬 곱셈을 이용해 그래프를 거슬러 올라가며 모든 조상을 찾습니다.
        for _ in range(num_nodes):
            # 현재 경로에 포함된 노드들의 부모를 찾습니다.
            parents_mask = (
                # Use the transpose to follow incoming edges when accumulating parents.
                adj_matrix.transpose(-1, -2).float() @ path_mask.float().unsqueeze(-1)
            ).squeeze(-1).bool()            # 더 이상 새로운 부모가 없으면 (경로의 끝에 도달하면) 종료합니다.
            if (parents_mask & ~path_mask).sum() == 0: break
            # 새로 찾은 부모들을 경로 마스크에 추가합니다.
            path_mask |= parents_mask
        return path_mask

    def _propagate_exclusive_path_upward(
        self,
        obs: TensorDict,
        rows: torch.Tensor,
        parent_indices: torch.Tensor,
        child_indices: torch.Tensor,
    ) -> None:
        if rows.numel() == 0:
            return

        ancestors = self._trace_path_batch(parent_indices, obs["adj_matrix"][rows])
        obs["is_exclusive_path"][rows] |= ancestors
        obs["is_exclusive_path"][rows, parent_indices] = True
        obs["is_exclusive_path"][rows, child_indices] = True

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
            # --- 👇 [여기에 새로운 상태 초기값을 추가합니다] ---
            "current_cost": torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device),
            "is_used_ic_mask": torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device),
            "is_locked_ic_mask": torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device),
            "current_target_load": torch.full((batch_size, 1), -1, dtype=torch.long, device=self.device),
        }, batch_size=[batch_size], device=self.device)
       
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
        batch_size = td.batch_size[0]
        action = td["action"].reshape(batch_size)
        current_head = td["trajectory_head"].reshape(batch_size)
        next_obs = td.clone()
        batch_indices = torch.arange(batch_size, device=self.device)

        # 1. 액션 타입에 따라 상태 업데이트
        head_is_battery = current_head == BATTERY_NODE_IDX
        if head_is_battery.any():
            # [Select New Load]
            battery_rows = batch_indices[head_is_battery]
            selected_load = action[head_is_battery]
            next_obs["trajectory_head"][battery_rows, 0] = selected_load
            next_obs["unconnected_loads_mask"][battery_rows, selected_load] = False
            next_obs["current_target_load"][battery_rows, 0] = selected_load


        head_is_node = ~head_is_battery
        if head_is_node.any():
            # [Find Parent]
            node_rows = batch_indices[head_is_node]
            child_node = current_head[head_is_node]
            parent_node = action[head_is_node]

            # 2. 연결 정보 및 사용 여부 업데이트
            next_obs["adj_matrix"][node_rows, parent_node, child_node] = True
            node_types = td["nodes"][0, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
            is_parent_ic = (node_types[parent_node] == NODE_TYPE_IC)
            if is_parent_ic.any():
                ic_rows = node_rows[is_parent_ic]
                ic_indices = parent_node[is_parent_ic]
                next_obs["is_used_ic_mask"][ic_rows, ic_indices] = True

            # 3. [핵심] Independent 조건에 따라 '잠금' 상태 업데이트
            target_load_idx = td["current_target_load"].squeeze(-1)[head_is_node]
            load_configs = self.generator.config.loads
            load_start_idx = 1 + self.generator.num_ics

            for i, b_idx in enumerate(node_rows):
                target_idx = target_load_idx[i].item()
                if target_idx == -1: continue
                
                config_idx = target_idx - load_start_idx
                if 0 <= config_idx < len(load_configs):
                    rail_type = load_configs[config_idx].get("independent_rail_type")
                    p_idx = parent_node[i].item() # 선택된 부모 IC

                    if rail_type == "exclusive_supplier" and child_node[i].item() == target_idx:
                        # 부하의 '직접' 부모만 잠금
                        next_obs["is_locked_ic_mask"][b_idx, p_idx] = True
                    elif rail_type == "exclusive_path":
                        # 경로상의 모든 IC를 잠금
                        next_obs["is_locked_ic_mask"][b_idx, p_idx] = True

            # 4. 다음 헤드 결정 및 작업 목표 초기화
            parent_is_battery = (parent_node == BATTERY_NODE_IDX)
            next_obs["trajectory_head"][node_rows, 0] = torch.where(parent_is_battery, BATTERY_NODE_IDX, parent_node)
            if parent_is_battery.any():
                finished_rows = node_rows[parent_is_battery]
                next_obs["current_target_load"][finished_rows, 0] = -1

        # 5. 전류, 온도, 비용 업데이트
        active_currents = next_obs["nodes"][:, :, FEATURE_INDEX["current_active"]]
        new_current_out = (next_obs["adj_matrix"].float().transpose(-1, -2) @ active_currents.float().unsqueeze(-1)).squeeze(-1)
        next_obs["nodes"][:, :, FEATURE_INDEX["current_out"]] = new_current_out
        
        power_loss = self._calculate_power_loss(next_obs["nodes"], new_current_out)
        theta_ja = next_obs["nodes"][:, :, FEATURE_INDEX["theta_ja"]]
        ambient_temp = self.generator.config.constraints.get("ambient_temperature", 25.0)
        new_temp = ambient_temp + power_loss * theta_ja
        next_obs["nodes"][:, :, FEATURE_INDEX["junction_temp"]] = new_temp
        
        node_costs = next_obs["nodes"][:, :, FEATURE_INDEX["cost"]]
        next_obs["current_cost"] = (next_obs["is_used_ic_mask"].float() * node_costs).sum(dim=1, keepdim=True)
        
        next_obs.set("step_count", td["step_count"] + 1)


        # 6. 종료 조건
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
        }, batch_size=batch_size)
        
    # 💡 *** 여기가 핵심 수정 부분입니다 ***
    def get_action_mask(self, td: TensorDict, debug: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        self._ensure_buffers(td) # 맨 앞에서 버퍼 동기화
        
        batch_size, num_nodes, _ = td["nodes"].shape
        mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)
        current_head = td["trajectory_head"].squeeze(-1)

        # 💡 디버깅 로그를 위한 설정
        is_debug_instance = td.batch_size[0] > 0 and td.get("log_mode", "progress") == "detail"
        debug_idx = td.get("log_idx", 0) if is_debug_instance else -1


        # --- 1. [Select New Load] 모드 마스킹 ---
        head_is_battery = (current_head == BATTERY_NODE_IDX)
        if head_is_battery.any():
            all_has_unconnected = td["unconnected_loads_mask"].any(dim=-1)
            is_active = head_is_battery & all_has_unconnected
            is_finished = head_is_battery & ~all_has_unconnected
            mask[is_active] = td["unconnected_loads_mask"][is_active]
            mask[is_finished, BATTERY_NODE_IDX] = True

        # --- 2. [Find Parent] 모드 마스킹 (모든 조건을 한번에 계산) ---
        head_is_node = ~head_is_battery
        if head_is_node.any():
            b_idx_node = torch.where(head_is_node)[0]
            child_nodes = current_head[head_is_node]
            B_act = len(b_idx_node)

            # 조건 0a: 부모는 Load 타입이 아니어야 함
            not_load_parent = self.node_type_tensor.unsqueeze(0) != NODE_TYPE_LOAD

            # 조건 0b: 자기 자신은 부모가 될 수 없음
            not_self_parent = self.arange_nodes.unsqueeze(0) != child_nodes.unsqueeze(1)


            # --- 👇 [핵심 수정] 모든 마스킹 조건을 개별적으로 계산 ---

            # 1. 전압 호환성
            # connectivity_matrix[batch, parent, child] -> [b_idx_node, :, child_nodes]
            # PyTorch의 gather를 사용하여 각 배치 항목에 맞는 child 슬라이스를 선택
            connectivity = td["connectivity_matrix"][b_idx_node]
            child_indices_exp = child_nodes.view(-1, 1, 1).expand(-1, num_nodes, 1)
            volt_ok = torch.gather(connectivity, 2, child_indices_exp).squeeze(-1)

            # 조건 2: 사이클 방지 (_trace_path_batch는 자기 자신을 포함하므로 not_self_parent와 중복되지만, 명시적으로 둠)
            path_mask = self._trace_path_batch(child_nodes, td["adj_matrix"][b_idx_node])
            cycle_ok = ~path_mask

            # 조건 3: 전류 한계
            nodes_slice = td["nodes"][b_idx_node]
            rows = torch.arange(B_act, device=self.device)
            remaining_capacity = nodes_slice[:, :, FEATURE_INDEX["i_limit"]] - nodes_slice[:, :, FEATURE_INDEX["current_out"]]
            child_current_draw = nodes_slice[rows, child_nodes, FEATURE_INDEX["current_active"]].unsqueeze(1)
            current_ok = remaining_capacity >= child_current_draw

            # 조건 4: Independent Rail (전역 규칙 - 잠긴 IC 제외)
            not_locked = ~td["is_locked_ic_mask"][b_idx_node]

            # 조건 5: Independent Rail (상황 규칙 - exclusive 경로의 경우 사용된 IC 제외)
            target_load_idx = td["current_target_load"].squeeze(-1)[head_is_node]
            load_start_idx = 1 + self.generator.num_ics
            
            target_rail_types = torch.zeros_like(target_load_idx, dtype=torch.long)
            valid_target_mask = (target_load_idx != -1)
            
            # clamp_ 대신 안전한 마스킹으로 인덱싱
            if valid_target_mask.any():
                config_indices = target_load_idx[valid_target_mask] - load_start_idx
                in_range_mask = (config_indices >= 0) & (config_indices < len(self.rail_types))
                
                # 범위 내에 있는 유효한 인덱스에 대해서만 rail_type을 할당
                if in_range_mask.any():
                    final_indices = config_indices[in_range_mask]
                    temp_types = torch.zeros_like(config_indices, dtype=torch.long)
                    temp_types[in_range_mask] = self.rail_types[final_indices]
                    target_rail_types[valid_target_mask] = temp_types

            # `exclusive_supplier` (type 1)
            children_count = td["adj_matrix"][b_idx_node].sum(dim=-1)
            is_parent_free = (children_count == 0)
            is_exclusive_supplier = (target_rail_types == 1).unsqueeze(1)
            supplier_ok = ~(is_exclusive_supplier & ~is_parent_free)

            # `exclusive_path` (type 2)
            is_used_mask_slice = td["is_used_ic_mask"][b_idx_node]
            is_exclusive_path = (target_rail_types == 2).unsqueeze(1)
            path_ok = ~(is_exclusive_path & is_used_mask_slice)

            exclusive_ok = supplier_ok & path_ok & not_locked

            # --- 모든 벡터화 가능 조건을 단 한 번의 연산으로 결합 ---
            if debug:
                reasons = {
                    "not_load_parent": not_load_parent, "not_self_parent": not_self_parent,
                    "volt_ok": volt_ok, "cycle_ok": cycle_ok, "current_ok": current_ok,
                    "exclusive_ok": exclusive_ok,
                }
                can_be_parent = torch.ones_like(volt_ok, dtype=torch.bool)
                for v in reasons.values():
                    can_be_parent &= v
            else:
                can_be_parent = (
                    not_load_parent & not_self_parent & volt_ok & cycle_ok & 
                    current_ok & exclusive_ok
                )

            # --- 2-2. 루프가 필요한 Power Sequence만 따로 처리 ---
            for j_idx, k_idx, f_flag in self.power_sequences:
                # Case 1: 현재 child가 'k'일 때 (k의 부모를 찾는 중)
                is_k_mask = (child_nodes == k_idx)
                if is_k_mask.any():
                    instances_to_check = torch.where(is_k_mask)[0]
                    b_idx_check = b_idx_node[instances_to_check]
                    adj_j = td["adj_matrix"][b_idx_check, :, j_idx]
                    parent_exists = adj_j.any(dim=-1)

                    if parent_exists.any():
                        b_constr = b_idx_check[parent_exists]
                        inst_constr = instances_to_check[parent_exists]
                        parent_of_j_idx = adj_j[parent_exists].long().argmax(-1)
                        
                        anc_mask = self._trace_path_batch(parent_of_j_idx, td["adj_matrix"][b_constr])
                        can_be_parent[inst_constr] &= ~anc_mask
                        
                        if f_flag == 1:
                            same_parent_mask = (self.arange_nodes == parent_of_j_idx.unsqueeze(1))
                            can_be_parent[inst_constr] &= ~same_parent_mask

                # Case 2: 현재 child가 'j'일 때 (j의 부모를 찾는 중)
                is_j_mask = (child_nodes == j_idx)
                if is_j_mask.any():
                    instances_to_check = torch.where(is_j_mask)[0]
                    b_idx_check = b_idx_node[instances_to_check]
                    adj_k = td["adj_matrix"][b_idx_check, :, k_idx]
                    parent_exists = adj_k.any(dim=-1)

                    if parent_exists.any():
                        b_constr = b_idx_check[parent_exists]
                        inst_constr = instances_to_check[parent_exists]
                        parent_of_k_idx = adj_k[parent_exists].long().argmax(-1)
                        
                        anc_mask = self._trace_path_batch(parent_of_k_idx, td["adj_matrix"][b_constr])
                        can_be_parent[inst_constr] &= ~anc_mask
                        
                        if f_flag == 1:
                            same_parent_mask = (self.arange_nodes == parent_of_k_idx.unsqueeze(1))
                            can_be_parent[inst_constr] &= ~same_parent_mask

            mask[head_is_node] = can_be_parent
        if debug:
            return {"mask": mask, "reasons": reasons, "b_idx_node": b_idx_node, "child_nodes": child_nodes}

        return mask


    
    def get_reward(self, td: TensorDict, done_successfully: torch.Tensor, timed_out: torch.Tensor, is_stuck_or_finished: torch.Tensor) -> torch.Tensor:
        """
        보상을 계산합니다. 성공, 타임아웃, 갇힘 상태에 따라 다른 보상을 부여합니다.
        """
        batch_size = td.batch_size[0]
        reward = torch.zeros(batch_size, device=self.device)
        
        # 1. 성공적으로 완료된 경우: 비용 기반으로 기본 보상 계산
        if done_successfully.any():
            is_used_mask = td["adj_matrix"][done_successfully].any(dim=2)
            node_costs = td["nodes"][done_successfully, :, FEATURE_INDEX["cost"]]
            ic_mask = td["nodes"][done_successfully, :, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] == 1
            used_ic_mask = is_used_mask & ic_mask
            total_cost = (node_costs * used_ic_mask).sum(dim=-1)
            reward[done_successfully] = -total_cost

        # 3. 중간에 갇히거나 타임아웃으로 실패한 경우: 큰 페널티
        failed = (timed_out | is_stuck_or_finished) & ~done_successfully
        if failed.any():
            reward[failed] -= 100.0 # 예시 패널티 값
            
        return reward