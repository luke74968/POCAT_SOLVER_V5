# transformer_solver/pocat_env.py
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Optional, List

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
        self._make_spec()
        self._set_seed(None)
        self.trajectory_head_stacks: List[List[int]] = []
        
        self._load_constraint_info()

    def _make_spec(self):
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
    
    def _load_constraint_info(self):
        self.node_name_to_idx = {name: i for i, name in enumerate(self.generator.config.node_names)}
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
            self.exclusive_path_loads_tensor = torch.tensor(list(self.exclusive_path_loads), dtype=torch.long, device=self.device)

        self.power_sequences = []
        for seq in self.generator.config.constraints.get("power_sequences", []):
            f_flag = seq.get("f", 1)
            j_idx = self.node_name_to_idx.get(seq['j'])
            k_idx = self.node_name_to_idx.get(seq['k'])
            if j_idx is not None and k_idx is not None:
                self.power_sequences.append((j_idx, k_idx, f_flag))

    def _trace_path_batch(self, start_nodes: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = adj_matrix.shape
        path_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)
        if start_nodes.numel() > 0:
            path_mask.scatter_(1, start_nodes.view(batch_size, -1), True)
        for _ in range(num_nodes):
            parents_mask = (adj_matrix.float() @ path_mask.float().unsqueeze(-1)).squeeze(-1).bool()
            newly_added = (parents_mask & ~path_mask)
            if not newly_added.any(): break
            path_mask |= newly_added
        return path_mask

    def _reset(self, td: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        batch_size = kwargs.get("batch_size", self.batch_size)
        if isinstance(batch_size, tuple): batch_size = batch_size[0]
        td_initial = self.generator(batch_size=batch_size).to(self.device)
        num_nodes = td_initial["nodes"].shape[1]
        self.trajectory_head_stacks = [[] for _ in range(batch_size)]
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
        node_types = td_initial["nodes"][0, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        is_load = node_types == NODE_TYPE_LOAD
        reset_td["unconnected_loads_mask"][:, ~is_load] = False
        reset_td.set("done", torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device))
        return reset_td

    def step(self, tensordict: TensorDict) -> TensorDict:
        return self._step(tensordict)

    def _calculate_power_loss(self, ic_node_features: torch.Tensor, i_out: torch.Tensor) -> torch.Tensor:
        ic_type = ic_node_features[:, :, FEATURE_INDEX["ic_type_idx"]]
        vin = ic_node_features[:, :, FEATURE_INDEX["vin_min"]]
        vout = ic_node_features[:, :, FEATURE_INDEX["vout_min"]]
        power_loss = torch.zeros_like(i_out)
        ldo_mask = ic_type == 1.0
        if ldo_mask.any():
            op_current = ic_node_features[:, :, FEATURE_INDEX["op_current"]]
            power_loss[ldo_mask] = (vin[ldo_mask] - vout[ldo_mask]) * i_out[ldo_mask] + vin[ldo_mask] * op_current[ldo_mask]
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
            self.trajectory_head_stacks = [s.copy() for s in self.trajectory_head_stacks for _ in range(num_repeats)]

        action = td["action"].squeeze(-1)
        current_head = td["trajectory_head"].squeeze(-1)
        next_obs = td.clone()
        b_idx = torch.arange(new_batch_size, device=self.device)

        head_is_battery = (current_head == BATTERY_NODE_IDX)
        if head_is_battery.any():
            selected_load = action[head_is_battery]
            next_obs["trajectory_head"][head_is_battery] = selected_load.unsqueeze(-1)
            next_obs["unconnected_loads_mask"][head_is_battery, selected_load] = False
            for i in torch.where(head_is_battery)[0].tolist():
                load_idx = action[i].item()
                if load_idx == BATTERY_NODE_IDX: continue
                if load_idx in self.exclusive_path_loads:
                    self.trajectory_head_stacks[i].append(BATTERY_NODE_IDX)

        head_is_node = ~head_is_battery
        if head_is_node.any():
            child_idx_vec = current_head[head_is_node]
            parent_idx_vec = action[head_is_node]
            b_idx_node_vec = b_idx[head_is_node]
            next_obs["adj_matrix"][b_idx_node_vec, parent_idx_vec, child_idx_vec] = True
            next_obs["children_count"][b_idx_node_vec, parent_idx_vec] += 1
            is_child_initially_exclusive = (child_idx_vec.unsqueeze(1) == self.exclusive_path_loads_tensor).any(dim=1)
            path_starts = torch.where(is_child_initially_exclusive, child_idx_vec, -1)
            if (path_starts != -1).any():
                active_indices = torch.where(path_starts != -1)[0]
                ancestors = self._trace_path_batch(path_starts[active_indices], next_obs["adj_matrix"][b_idx_node_vec[active_indices]])
                next_obs["is_exclusive_path"][b_idx_node_vec[active_indices]] |= ancestors
            
            # 💡 **[핵심 버그 수정 1]** 누락되었던 node_stages 업데이트 로직
            parent_stages = next_obs["node_stages"][b_idx_node_vec, parent_idx_vec]
            next_obs["node_stages"][b_idx_node_vec, child_idx_vec] = parent_stages + 1
            
            # 💡 **[핵심 버그 수정 2]** 스택 관리 로직 수정
            for i in torch.where(head_is_node)[0].tolist():
                parent_for_i = action[i].item()
                # 부모가 이미 연결되었는지 여부를 node_stages로 정확히 판단
                is_parent_connected = (parent_for_i == BATTERY_NODE_IDX) or (next_obs["node_stages"][i, parent_for_i] != -1)
                if is_parent_connected:
                    # 부모가 이미 연결되었으면, 스택에서 다음 작업을 꺼냄
                    if self.trajectory_head_stacks[i]: next_obs["trajectory_head"][i] = self.trajectory_head_stacks[i].pop()
                    else: next_obs["trajectory_head"][i] = BATTERY_NODE_IDX
                else:
                    # 부모가 새 노드이면, 현재 노드를 스택에 넣고 부모를 먼저 처리
                    self.trajectory_head_stacks[i].append(current_head[i].item())
                    next_obs["trajectory_head"][i] = parent_for_i

        active_currents = next_obs["nodes"][:, :, FEATURE_INDEX["current_active"]]
        new_current_out = (next_obs["adj_matrix"].float().transpose(-1, -2) @ active_currents.float().unsqueeze(-1)).squeeze(-1)
        next_obs["nodes"][:, :, FEATURE_INDEX["current_out"]] = new_current_out
        power_loss = self._calculate_power_loss(next_obs["nodes"], new_current_out)
        theta_ja = next_obs["nodes"][:, :, FEATURE_INDEX["theta_ja"]]
        ambient_temp = self.generator.config.constraints.get("ambient_temperature", 25.0)
        new_temp = ambient_temp + power_loss * theta_ja
        next_obs["nodes"][:, :, FEATURE_INDEX["junction_temp"]] = new_temp
        next_obs.set("step_count", td["step_count"] + 1)
        
        # 💡 **[수정]** get_action_mask 호출 시 인자 전달
        next_mask = self.get_action_mask(next_obs, td.get("log_mode", "progress"), td.get("log_idx", 0))
        is_stuck = ~next_mask.any(dim=-1)
        all_loads_connected = (next_obs["unconnected_loads_mask"].sum(dim=1) == 0)
        trajectory_finished = (next_obs["trajectory_head"].squeeze(-1) == BATTERY_NODE_IDX)
        done_successfully = all_loads_connected & trajectory_finished
        max_steps = 2 * self.generator.num_nodes
        timed_out = (next_obs["step_count"] > max_steps).squeeze(-1)
        is_done = done_successfully | timed_out | is_stuck
        next_obs["done"] = is_done.unsqueeze(-1)
        
        return TensorDict({
            "next": next_obs,
            "reward": self.get_reward(next_obs, done_successfully, timed_out, is_stuck),
            "done": next_obs["done"],
        }, batch_size=new_batch_size)
        
    def get_action_mask(self, td: TensorDict, log_mode: str = "progress", log_idx: int = 0) -> torch.Tensor:
        batch_size, num_nodes, _ = td["nodes"].shape
        mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)
        current_head = td["trajectory_head"].squeeze(-1)
        is_debug_instance = batch_size > 0 and log_mode == "detail"
        debug_idx = log_idx if is_debug_instance else -1

        head_is_battery = (current_head == BATTERY_NODE_IDX)
        if head_is_battery.any():
            all_has_unconnected = td["unconnected_loads_mask"].any(dim=-1)
            is_active = head_is_battery & all_has_unconnected
            is_finished = head_is_battery & ~all_has_unconnected
            if is_active.any(): mask[is_active] = td["unconnected_loads_mask"][is_active]
            if is_finished.any(): mask[is_finished, BATTERY_NODE_IDX] = True

        head_is_node = ~head_is_battery
        if head_is_node.any():
            b_idx_node = torch.where(head_is_node)[0]
            child_nodes = current_head[head_is_node]
            can_be_parent = torch.ones(len(b_idx_node), num_nodes, dtype=torch.bool, device=self.device)
            local_debug_idx = (b_idx_node == debug_idx).nonzero().item() if debug_idx != -1 and debug_idx in b_idx_node else -1

            def log_rejection(constraint_name):
                if local_debug_idx != -1 and len(can_be_parent) > local_debug_idx and not can_be_parent[local_debug_idx, BATTERY_NODE_IDX]:
                    print(f"    - [DEBUG] Battery REJECTED by {constraint_name}.")

            connectivity = td["connectivity_matrix"][b_idx_node]
            child_indices_exp = child_nodes.view(-1, 1, 1).expand(-1, num_nodes, 1)
            can_be_parent &= torch.gather(connectivity, 2, child_indices_exp).squeeze(-1)
            path_mask = self._trace_path_batch(child_nodes, td["adj_matrix"][b_idx_node])
            can_be_parent &= ~path_mask
            log_rejection("Voltage/Cycle")

            parent_i_limit = td["nodes"][:, :, FEATURE_INDEX["i_limit"]]
            parent_i_out = td["nodes"][:, :, FEATURE_INDEX["current_out"]]
            remaining_capacity = (parent_i_limit - parent_i_out)[b_idx_node]
            child_current_draw = td["nodes"][b_idx_node, child_nodes, FEATURE_INDEX["current_active"]].unsqueeze(1)
            can_be_parent &= remaining_capacity >= child_current_draw
            log_rejection("Current Limit")

            for load_idx in self.exclusive_supplier_loads:
                is_this_child = (child_nodes == load_idx)
                if is_this_child.any():
                    parent_children_count = td["children_count"][b_idx_node[is_this_child]]
                    can_be_parent[is_this_child] &= (parent_children_count == 0)
            
            is_child_exclusive = td["is_exclusive_path"][b_idx_node, child_nodes]
            if is_child_exclusive.any():
                instances_to_constrain = torch.where(is_child_exclusive)[0]
                if len(instances_to_constrain) > 0:
                    b_idx_constr = b_idx_node[instances_to_constrain]
                    is_battery_mask = torch.arange(num_nodes, device=self.device) == BATTERY_NODE_IDX
                    parent_children_count = td["children_count"][b_idx_constr]
                    can_be_parent[instances_to_constrain] &= (parent_children_count == 0) | is_battery_mask
                    parent_is_exclusive = td["is_exclusive_path"][b_idx_constr]
                    can_be_parent[instances_to_constrain] &= ~parent_is_exclusive | is_battery_mask
            log_rejection("Independent Rail")

            for j_idx, k_idx, f_flag in self.power_sequences:
                is_k = (child_nodes == k_idx)
                if is_k.any():
                    is_j_connected = td["adj_matrix"][b_idx_node[is_k], :, j_idx].any(dim=-1)
                    if is_j_connected.any():
                        instances_to_constrain = torch.where(is_k & is_j_connected.unsqueeze(0).transpose(0,1))[0]
                        if len(instances_to_constrain) > 0:
                            b_idx_constr = b_idx_node[instances_to_constrain]
                            parent_of_j_idx = td["adj_matrix"][b_idx_constr, :, j_idx].long().argmax(-1)
                            stage_of_j_parent = td["node_stages"][b_idx_constr, parent_of_j_idx]
                            candidate_parent_stages = td["node_stages"][b_idx_constr]
                            is_candidate_unconnected = (candidate_parent_stages == -1)
                            stage_ok = (candidate_parent_stages > stage_of_j_parent.unsqueeze(1)) if f_flag == 1 else (candidate_parent_stages >= stage_of_j_parent.unsqueeze(1))
                            can_be_parent[instances_to_constrain] &= is_candidate_unconnected | stage_ok
                
                is_j = (child_nodes == j_idx)
                if is_j.any():
                    is_k_connected = td["adj_matrix"][b_idx_node[is_j], :, k_idx].any(dim=-1)
                    if is_k_connected.any():
                        instances_to_constrain = torch.where(is_j & is_k_connected.unsqueeze(0).transpose(0,1))[0]
                        if len(instances_to_constrain) > 0:
                            b_idx_constr = b_idx_node[instances_to_constrain]
                            parent_of_k_idx = td["adj_matrix"][b_idx_constr, :, k_idx].long().argmax(-1)
                            stage_of_k_parent = td["node_stages"][b_idx_constr, parent_of_k_idx]
                            candidate_parent_stages = td["node_stages"][b_idx_constr]
                            is_candidate_unconnected = (candidate_parent_stages == -1)
                            stage_ok = (candidate_parent_stages < stage_of_k_parent.unsqueeze(1)) if f_flag == 1 else (candidate_parent_stages <= stage_of_k_parent.unsqueeze(1))
                            can_be_parent[instances_to_constrain] &= is_candidate_unconnected | stage_ok
                    if f_flag == 1:
                        is_parent_of_k = td["adj_matrix"][b_idx_node[is_j], :, k_idx]
                        can_be_parent[is_j] &= ~is_parent_of_k
            log_rejection("Power Sequence")

            mask[head_is_node] = can_be_parent
            
        return mask
    
    def get_reward(self, td: TensorDict, done_successfully: torch.Tensor, timed_out: torch.Tensor, is_stuck: torch.Tensor) -> torch.Tensor:
        batch_size = td.batch_size[0]
        reward = torch.zeros(batch_size, device=self.device)
        
        if done_successfully.any():
            is_used_mask = td["adj_matrix"][done_successfully].any(dim=2)
            node_costs = td["nodes"][done_successfully, :, FEATURE_INDEX["cost"]]
            ic_mask = td["nodes"][done_successfully, :, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] == 1
            used_ic_mask = is_used_mask & ic_mask
            total_cost = (node_costs * used_ic_mask).sum(dim=-1)
            reward[done_successfully] = -total_cost

        failed = (timed_out | is_stuck) & ~done_successfully
        if failed.any():
            reward[failed] -= 100.0
            
        return reward.unsqueeze(-1)

    def select_start_nodes(self, td):
        node_types = td["nodes"][0, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        load_indices = torch.where(node_types == NODE_TYPE_LOAD)[0]
        return len(load_indices), load_indices