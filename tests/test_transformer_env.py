import torch
import sys
import os

# 💡 테스트를 위해 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer_solver.pocat_env import PocatEnv
from common.pocat_defs import FEATURE_INDEX, NODE_TYPE_LOAD


def test_current_limit_mask():
    env = PocatEnv(generator_params={"config_file_path": "config.json"})
    td = env.reset()

    battery_idx = 0
    ldo_idx = 5  # LDO_X_Gen with i_limit=0.3
    first_load = 9   # MCU_Main current_active=0.150
    second_load = 17  # eMMC_Memory current_active=0.250
    big_load = 15  # DSP_Core current_active=0.800

    # Connect LDO to battery so it can act as a parent
    td = td.set("action", torch.tensor([[ldo_idx, battery_idx]], dtype=torch.long))
    td = env.step(td)["next"]

    mask = env.get_action_mask(td)
    assert not mask[0, big_load, ldo_idx]
    assert mask[0, second_load, ldo_idx]

    # Connect first load to LDO
    td = td.set("action", torch.tensor([[first_load, ldo_idx]], dtype=torch.long))
    td = env.step(td)["next"]

    # Current draw of LDO should equal first load's current
    child_current = td["nodes"][0, first_load, FEATURE_INDEX["current_active"]]
    assert torch.isclose(td["ic_current_draw"][0, ldo_idx], child_current)

    mask2 = env.get_action_mask(td)

    # Remaining capacity on the LDO after connecting the first load
    remaining = (
        td["nodes"][0, ldo_idx, FEATURE_INDEX["i_limit"]]
        - td["ic_current_draw"][0, ldo_idx]
    )
    child_currents = td["nodes"][0, :, FEATURE_INDEX["current_active"]]

    # Any child drawing more than the remaining capacity should be masked out
    over_limit = child_currents > remaining
    assert torch.all(mask2[0, over_limit, ldo_idx] == 0)


def test_load_cannot_be_parent():
    """Ensure load nodes are never allowed to be parents."""
    env = PocatEnv(generator_params={"config_file_path": "config.json"})
    td = env.reset()

    load_indices = torch.where(td["unconnected_loads_mask"][0])[0]
    selected_load = load_indices[0]
    td = td.set("action", torch.tensor([[selected_load, 0]], dtype=torch.long))
    td = env.step(td)["next"]

    mask = env.get_action_mask(td)
    node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
    load_nodes = torch.where(node_types == NODE_TYPE_LOAD)[0]

    load_parent_mask = mask[0, selected_load, load_nodes]
    print("Load parent mask:", load_parent_mask)
    assert not load_parent_mask.any()