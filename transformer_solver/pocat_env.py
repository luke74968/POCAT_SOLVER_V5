# pocat_env.py
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Optional, Tuple
from torchrl.data import UnboundedContinuousTensorSpec as Unbounded, \
    UnboundedDiscreteTensorSpec as UnboundedDiscrete, \
    DiscreteTensorSpec as Categorical, \
    CompositeSpec as Composite

from common.pocat_defs import (
    NODE_TYPE_BATTERY, NODE_TYPE_IC, NODE_TYPE_LOAD,
    FEATURE_DIM, FEATURE_INDEX
)

class PocatEnv(EnvBase):
    name = "pocat"

    def __init__(
        self,
        generator_params: dict = {},
        device: str = "cpu",
        instance_repeats: int = 1,
        **kwargs,
    ):
        super().__init__(device=device)
        from .pocat_generator import PocatGenerator

        self.generator = PocatGenerator(**generator_params)
        self.instance_repeats = instance_repeats
        self._make_spec()
        self._set_seed(None)

    def _reset(self, td: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        if td is None:
            batch_size = kwargs.get("batch_size", self.batch_size)
            if not isinstance(batch_size, int):
                batch_size = batch_size[0]
            instance_repeats = kwargs.get("instance_repeats", self.instance_repeats)
            td = self.generator(
                batch_size=batch_size, instance_repeats=instance_repeats
            ).to(self.device)
            td = td.reshape(batch_size * instance_repeats)
        num_nodes = td["nodes"].shape[1]
        
        reset_td = TensorDict({
            "nodes": td["nodes"],
            "prompt_features": td["prompt_features"],
            "connections": torch.zeros(td.batch_size[0], num_nodes - 1, 2, dtype=torch.long, device=self.device),
            "connected_nodes_mask": torch.zeros(td.batch_size[0], num_nodes, dtype=torch.bool, device=self.device),
            "ic_current_draw": torch.zeros(td.batch_size[0], num_nodes, device=self.device),
            "step_count": torch.zeros(td.batch_size[0], 1, dtype=torch.long, device=self.device),
        }, batch_size=[td.batch_size[0]], device=self.device)
        reset_td.set("done", torch.zeros(td.batch_size[0], 1, dtype=torch.bool, device=self.device))
        return reset_td

    # 💡 추가된 step 메소드: 배치 크기 검사를 우회합니다.
    def step(self, tensordict: TensorDict) -> TensorDict:
        return self._step(tensordict)

    def _step(self, td: TensorDict) -> TensorDict:
        action = td["action"]
        child_idx, parent_idx = action[:, 0], action[:, 1]
        b_idx = torch.arange(td.batch_size[0], device=self.device)

        step = td["step_count"].squeeze(-1)
        if step.numel() > 0 and step[0] < td["connections"].shape[1]:
            td["connections"][b_idx, step] = action

        td["connected_nodes_mask"][b_idx, child_idx] = True
        parent_is_not_battery = td["nodes"][b_idx, parent_idx, FEATURE_INDEX["node_type"][0] + NODE_TYPE_BATTERY] != 1
        td["connected_nodes_mask"][b_idx[parent_is_not_battery], parent_idx[parent_is_not_battery]] = True

        # Update current draw for parent ICs and propagate upstream
        child_currents = td["nodes"][b_idx, child_idx, FEATURE_INDEX["current_active"]]
        for batch, curr_parent in enumerate(parent_idx.tolist()):
            increment = child_currents[batch]
            steps_used = step[batch].item() + 1
            while True:
                td["ic_current_draw"][batch, curr_parent] += increment
                # find if current parent is itself a child in existing connections
                existing = td["connections"][batch, :steps_used]
                matches = existing[existing[:, 0] == curr_parent]
                if matches.numel() == 0:
                    break
                curr_parent = matches[0, 1].item()

        load_indices = torch.where(td["nodes"][0, :, FEATURE_INDEX["node_type"][0] + NODE_TYPE_LOAD] == 1)[0]
        all_loads_connected = td["connected_nodes_mask"][:, load_indices].all(dim=1)
        
        next_obs = td.clone()
        next_obs.set("step_count", td["step_count"] + 1)
        
        return TensorDict({
                "next": next_obs,
                "reward": self.get_reward(td),
                "done": all_loads_connected.unsqueeze(-1),
            }, batch_size=td.batch_size)

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        num_nodes = td["nodes"].shape[1]
        mask = torch.zeros(td.batch_size[0], num_nodes, num_nodes, dtype=torch.bool, device=self.device)
        node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
        is_ic_or_load = (node_types == NODE_TYPE_IC) | (node_types == NODE_TYPE_LOAD)
        unconnected_mask = ~td["connected_nodes_mask"]
        valid_child_mask = unconnected_mask & is_ic_or_load.unsqueeze(0)
        is_battery_or_ic = (node_types == NODE_TYPE_BATTERY) | (node_types == NODE_TYPE_IC)
        can_be_parent_mask = (node_types == NODE_TYPE_BATTERY).unsqueeze(0) | td["connected_nodes_mask"]
        valid_parent_mask = can_be_parent_mask & is_battery_or_ic.unsqueeze(0)
        parent_vout_min = td["nodes"][:, :, FEATURE_INDEX["vout_min"]].unsqueeze(1)
        parent_vout_max = td["nodes"][:, :, FEATURE_INDEX["vout_max"]].unsqueeze(1)
        child_vin_min = td["nodes"][:, :, FEATURE_INDEX["vin_min"]].unsqueeze(2)
        child_vin_max = td["nodes"][:, :, FEATURE_INDEX["vin_max"]].unsqueeze(2)
        voltage_ok = (parent_vout_min <= child_vin_max) & (parent_vout_max >= child_vin_min)

        child_currents = td["nodes"][:, :, FEATURE_INDEX["current_active"]].unsqueeze(2)
        prospective_draw = td["ic_current_draw"].unsqueeze(1) + child_currents
        parent_limits = td["nodes"][:, :, FEATURE_INDEX["i_limit"]]
        inf = torch.full_like(parent_limits, float("inf"))
        parent_limits = torch.where(parent_limits > 0, parent_limits, inf).unsqueeze(1)
        current_ok = prospective_draw <= parent_limits
       
        # --- 타입 규칙 ---
        # 배터리 → 자식은 오직 IC만 허용 (B→L 금지)
        parent_is_batt = (node_types == NODE_TYPE_BATTERY).unsqueeze(0).unsqueeze(1)
        child_is_load  = (node_types == NODE_TYPE_LOAD).unsqueeze(0).unsqueeze(2)
        type_ok = ~(parent_is_batt & child_is_load)

        final_mask = (
                valid_child_mask.unsqueeze(2)
                & valid_parent_mask.unsqueeze(1)
                & voltage_ok
                & current_ok
                & type_ok
        )
        
        final_mask[:, torch.arange(num_nodes), torch.arange(num_nodes)] = False
        return final_mask
        
    def get_reward(self, td: TensorDict) -> torch.Tensor:
        reward = torch.zeros(td.batch_size[0], device=self.device)
        load_indices = torch.where(td["nodes"][0, :, FEATURE_INDEX["node_type"][0] + NODE_TYPE_LOAD] == 1)[0]
        done = td["connected_nodes_mask"][:, load_indices].all(dim=1)
        if done.any():
            used_nodes_mask = td["connected_nodes_mask"][done]
            node_costs = td["nodes"][done, :, FEATURE_INDEX["cost"]]
            ic_mask = td["nodes"][done, :, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] == 1
            used_ic_mask = used_nodes_mask & ic_mask
            total_cost = (node_costs * used_ic_mask).sum(dim=-1)
            reward[done] = -total_cost
        return reward.unsqueeze(-1)
        
    def _make_spec(self):
        num_nodes = self.generator.num_nodes
        self.observation_spec = Composite({
            "nodes": Unbounded(shape=(num_nodes, FEATURE_DIM), device=self.device),
            "prompt_features": Unbounded(shape=(2,), device=self.device),
            "connections": UnboundedDiscrete(shape=(num_nodes - 1, 2), dtype=torch.long, device=self.device),
            "connected_nodes_mask": Categorical(n=2, shape=(num_nodes,), dtype=torch.bool, device=self.device),
            "ic_current_draw": Unbounded(shape=(num_nodes,), device=self.device),
            "step_count": UnboundedDiscrete(shape=(1,), dtype=torch.long, device=self.device),
        })
        self.action_spec = UnboundedDiscrete(shape=(2,), dtype=torch.long, device=self.device)
        self.reward_spec = Unbounded(shape=(1,), device=self.device)
        self.done_spec = Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device)

    def _set_seed(self, seed: Optional[int]):
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        rng = torch.manual_seed(seed)
        self.rng = rng

    def select_start_nodes(self, td: TensorDict) -> Tuple[int, torch.Tensor]:
        node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
        load_indices = torch.where(node_types == NODE_TYPE_LOAD)[0]
        num_starts = len(load_indices)
        start_nodes = load_indices.repeat_interleave(td.batch_size[0])
        return num_starts, start_nodes
