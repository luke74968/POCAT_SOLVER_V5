import torch
import sys
import os

# 💡 테스트를 위해 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer_solver.pocat_env import PocatEnv, BATTERY_NODE_IDX
from common.pocat_defs import FEATURE_INDEX, NODE_TYPE_LOAD


def _step_with_action(env: PocatEnv, td, action_idx: int):
    action_tensor = torch.tensor([[action_idx]], dtype=torch.long)
    td_with_action = td.clone().set("action", action_tensor)
    return env.step(td_with_action)["next"]


def _find_node_index_by_name(node_names, keyword: str) -> int:
    for idx, name in enumerate(node_names):
        if keyword in name:
            return idx
    raise ValueError(f"Node containing '{keyword}' not found")


def test_current_limit_mask():
    env = PocatEnv(generator_params={"config_file_path": "config.json"})
    td = env.reset()

    node_names = env.generator.config.node_names
    ldo_idx = _find_node_index_by_name(node_names, "LDO_X_Gen")
    first_load = node_names.index("MCU_Main")
    big_load = node_names.index("DSP_Core")

    # Select the first load from the battery
    td = _step_with_action(env, td, first_load)
    mask_for_first = env.get_action_mask(td)[0]
    assert mask_for_first[ldo_idx], "Expected LDO to be a valid parent for the first load"

    # Connect the load to the LDO and then attach the LDO to the battery
    td = _step_with_action(env, td, ldo_idx)
    td = _step_with_action(env, td, BATTERY_NODE_IDX)

    child_current = td["nodes"][0, first_load, FEATURE_INDEX["current_active"]]
    ldo_current_out = td["nodes"][0, ldo_idx, FEATURE_INDEX["current_out"]]
    assert torch.isclose(ldo_current_out, child_current)

    remaining_capacity = (
        td["nodes"][0, ldo_idx, FEATURE_INDEX["i_limit"]] - ldo_current_out
    )

    # Check that a high-current load is masked out due to the remaining capacity
    td_big = _step_with_action(env, td.clone(), big_load)
    mask_big = env.get_action_mask(td_big)[0]
    big_current = td_big["nodes"][0, big_load, FEATURE_INDEX["current_active"]]
    expected_big_mask = big_current <= remaining_capacity + 1e-6
    assert mask_big[ldo_idx].item() == expected_big_mask


def test_load_cannot_be_parent():
    """Ensure load nodes are never allowed to be parents."""
    env = PocatEnv(generator_params={"config_file_path": "config.json"})
    td = env.reset()

    load_indices = torch.where(td["unconnected_loads_mask"][0])[0]
    selected_load = load_indices[0].item()
    td = _step_with_action(env, td, selected_load)

    mask = env.get_action_mask(td)[0]
    node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
    load_nodes = torch.where(node_types == NODE_TYPE_LOAD)[0]

    assert not mask[load_nodes].any()


def test_exclusive_path_stack_behaviour():
    env = PocatEnv(generator_params={"config_file_path": "config.json"})
    td = env.reset()

    assert env.exclusive_path_loads, "Exclusive path loads must be defined for this test"
    exclusive_load_idx = sorted(env.exclusive_path_loads)[0]

    # Select the exclusive load from the battery
    td = _step_with_action(env, td, exclusive_load_idx)
    assert td["trajectory_head"][0, 0] == exclusive_load_idx
    assert td["trajectory_stack_depth"][0, 0] == 1
    assert td["trajectory_stack_parents"][0, 0] == BATTERY_NODE_IDX

    mask = env.get_action_mask(td)[0]
    candidate_parents = torch.where(mask)[0]
    non_battery_parents = candidate_parents[candidate_parents != BATTERY_NODE_IDX]
    assert non_battery_parents.numel() > 0
    parent_idx = non_battery_parents[0].item()

    # Connect load -> parent
    td = _step_with_action(env, td, parent_idx)
    assert td["trajectory_head"][0, 0] == parent_idx
    assert td["trajectory_stack_depth"][0, 0] == 1

    # Connect parent -> battery to finish the path
    mask_parent = env.get_action_mask(td)[0]
    assert mask_parent[BATTERY_NODE_IDX]
    td = _step_with_action(env, td, BATTERY_NODE_IDX)

    assert td["trajectory_head"][0, 0] == BATTERY_NODE_IDX
    assert td["trajectory_stack_depth"][0, 0] == 0
    assert td["trajectory_stack_parents"][0, 0] == -1


def test_regular_path_no_stack_growth():
    env = PocatEnv(generator_params={"config_file_path": "config.json"})
    td = env.reset()

    load_candidates = torch.where(td["unconnected_loads_mask"][0])[0]
    regular_load_idx = None
    for idx in load_candidates.tolist():
        if idx not in env.exclusive_path_loads:
            regular_load_idx = idx
            break

    assert regular_load_idx is not None, "No regular load available for the test"

    td = _step_with_action(env, td, regular_load_idx)
    assert td["trajectory_stack_depth"][0, 0] == 0

    mask = env.get_action_mask(td)[0]
    candidate_parents = torch.where(mask)[0]
    parent_idx = BATTERY_NODE_IDX
    non_battery_parents = candidate_parents[candidate_parents != BATTERY_NODE_IDX]
    if non_battery_parents.numel() > 0:
        parent_idx = non_battery_parents[0].item()

    td = _step_with_action(env, td, parent_idx)
    assert td["trajectory_stack_depth"][0, 0] == 0

    mask_parent = env.get_action_mask(td)[0]
    if mask_parent[BATTERY_NODE_IDX]:
        td = _step_with_action(env, td, BATTERY_NODE_IDX)
        assert td["trajectory_stack_depth"][0, 0] == 0