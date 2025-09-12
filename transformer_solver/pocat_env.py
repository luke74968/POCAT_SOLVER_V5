# transformer_solver/pocat_env.py

import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Optional, Tuple
from torchrl.data import Unbounded, Categorical, Composite

from .pocat_generator import PocatGenerator
from common.pocat_defs import (
    NODE_TYPE_BATTERY, NODE_TYPE_IC, NODE_TYPE_LOAD,
    FEATURE_DIM, FEATURE_INDEX, SCALAR_PROMPT_FEATURE_DIM
)

class PocatEnv(EnvBase):
    name = "pocat"

    def __init__(self, generator_params: dict = {}, device: str = "cpu", **kwargs):
        super().__init__(device=device)
        self.generator = PocatGenerator(**generator_params)
        self._make_spec()

    def _make_spec(self):
        num_nodes = self.generator.num_nodes
        
        self.observation_spec = Composite({
            "nodes": Unbounded(shape=(num_nodes, FEATURE_DIM), dtype=torch.float32),
            "scalar_prompt_features": Unbounded(shape=(SCALAR_PROMPT_FEATURE_DIM,), dtype=torch.float32),
            "matrix_prompt_features": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.float32),
            # 💡 아래 상태들은 이제 모델 내부에서 관리되므로 스펙에서 제거해도 무방하지만,
            #    호환성을 위해 남겨두거나 혹은 아래 _reset에서만 생성하도록 변경할 수 있습니다.
        })
        
        self.action_spec = Unbounded(shape=(2,), dtype=torch.long)
        self.reward_spec = Unbounded(shape=(1,))

    def _set_seed(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)

    def select_start_nodes(self, td: TensorDict) -> Tuple[int, torch.Tensor]:
        node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
        start_nodes_idx = torch.where(node_types == NODE_TYPE_LOAD)[0]
        num_starts = len(start_nodes_idx)
        return num_starts, start_nodes_idx

    def _reset(self, td: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        if td is None:
            batch_size = kwargs.get("batch_size", self.batch_size)
            if not isinstance(batch_size, int): batch_size = batch_size[0]
            td = self.generator(batch_size=batch_size).to(self.device)
            
        # 💡 Env는 이제 초기 텐서들만 제공합니다.
        #    나머지 상태(adj_matrix 등)는 모델의 forward 패스에서 초기화됩니다.
        return td

    # 💡 step과 get_action_mask는 더 이상 사용되지 않으므로 삭제하거나 비워둡니다.
    def _step(self, td: TensorDict) -> TensorDict:
        # 이 로직은 이제 모델로 이전되었습니다.
        raise NotImplementedError("Step logic has been moved to the model.")

    def get_reward(self, td: TensorDict) -> torch.Tensor:
        """
        에피소드가 완료되었을 때 최종 보상을 계산합니다.
        """
        reward = torch.zeros(td.batch_size[0], device=self.device)
        done = td["done"].squeeze(-1)
        
        if done.any():
            is_used_mask = td["adj_matrix"][done].any(dim=1) | td["adj_matrix"][done].any(dim=2)
            node_costs = td["nodes"][done, :, FEATURE_INDEX["cost"]]
            ic_mask = td["nodes"][done, :, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] == 1
            
            used_ic_mask = is_used_mask & ic_mask
            total_cost = (node_costs * used_ic_mask).sum(dim=-1)
            reward[done] = -total_cost
            
        return reward