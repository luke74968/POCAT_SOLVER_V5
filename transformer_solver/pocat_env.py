# transformer_solver/pocat_env.py
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

    def __init__(self, generator_params: dict = {}, device: str = "cpu", **kwargs):
        super().__init__(device=device)
        from .pocat_generator import PocatGenerator
        self.generator = PocatGenerator(**generator_params)
        self._make_spec()
        self._set_seed(None)

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
            "prompt_features": td["prompt_features"],
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
        phase = td["decoding_phase"][0, 0].item()
        
        next_obs = td.clone()
        
        if phase == 0:  # 새 Load 선택 단계
            selected_load = action[:, 0]
            next_obs["trajectory_head"] = selected_load.unsqueeze(-1)
            next_obs["unconnected_loads_mask"][b_idx, selected_load] = False
            next_obs["decoding_phase"][:, 0] = 1 # 다음은 경로 구축 단계로

        elif phase == 1:  # Trajectory 구축 단계
            child_idx, parent_idx = action[:, 0], action[:, 1]
            next_obs["adj_matrix"][b_idx, parent_idx, child_idx] = True
            
            # 전류 전파 로직 (기존과 유사)
            # ...
            
            is_parent_in_main_tree = next_obs["main_tree_mask"][b_idx, parent_idx]
            
            for i in range(td.batch_size[0]):
                if is_parent_in_main_tree[i]:
                    # 경로가 메인 트리에 연결됨 -> Trajectory 완성
                    # 완성된 경로의 모든 노드를 메인 트리에 편입
                    path_nodes = self._trace_path(i, child_idx[i], next_obs["adj_matrix"])
                    next_obs["main_tree_mask"][i, path_nodes] = True
                    
                    if next_obs["unconnected_loads_mask"][i].sum() == 0:
                        # 모든 Load가 연결되었으면 종료
                        next_obs["done"][i] = True
                    else:
                        # 아니면 다음 Load 선택 단계로
                        next_obs["decoding_phase"][i] = 0
                else:
                    # 아직 경로 구축 중 -> 경로 헤드만 업데이트
                    next_obs["trajectory_head"][i] = parent_idx[i]

        next_obs.set("step_count", td["step_count"] + 1)
        
        return TensorDict({
            "next": next_obs,
            "reward": self.get_reward(next_obs),
            "done": next_obs["done"],
        }, batch_size=td.batch_size)
    
    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        batch_size, num_nodes, _ = td["nodes"].shape
        phase = td["decoding_phase"][0, 0].item()
        
        mask = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=self.device)

        if phase == 0:  # 새 Load 선택 단계
            # 자식: 아직 연결 안된 Load만 가능, 부모: 사용 안함 (0으로 고정)
            mask[:, 0, :] = td["unconnected_loads_mask"]
            return mask.transpose(1, 2)

        elif phase == 1:  # Trajectory 구축 단계
            # 자식: 현재 Trajectory의 헤드 노드만 가능
            child_indices = td["trajectory_head"].squeeze(-1)
            b_idx = torch.arange(batch_size, device=self.device)
            
            can_be_parent = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=self.device)
            # 자기 자신, 이미 경로에 포함된 노드는 부모가 될 수 없음
            can_be_parent[b_idx, child_indices] = False
            
            path_nodes = self._trace_path_batch(b_idx, child_indices, td["adj_matrix"])
            can_be_parent[path_nodes] = False

            # --- 기본 마스크 (전압, 전류, 기본 연결 규칙) ---
            node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
            is_load = node_types == NODE_TYPE_LOAD
            is_ic = node_types == NODE_TYPE_IC
            is_battery = node_types == NODE_TYPE_BATTERY
            
            can_be_child = ~td["connected_nodes_mask"] & (is_ic | is_load)
            can_be_parent = td["connected_nodes_mask"] & (is_ic | is_battery)
            
            base_mask = can_be_child.unsqueeze(1) & can_be_parent.unsqueeze(2)
            base_mask[:, torch.arange(num_nodes), torch.arange(num_nodes)] = False # 자기 자신에게 연결 금지

            # 전압 호환성
            parent_vout_max = td["nodes"][:, :, FEATURE_INDEX["vout_max"]].unsqueeze(2)
            child_vin_min = td["nodes"][:, :, FEATURE_INDEX["vin_min"]].unsqueeze(1)
            base_mask &= (parent_vout_max >= child_vin_min)

            # 전류 한계 (이제 열 마진이 반영된 i_limit 사용)
            child_currents = td["nodes"][:, :, FEATURE_INDEX["current_active"]].unsqueeze(1)
            prospective_draw = td["ic_current_draw"].unsqueeze(2) + child_currents
            parent_limits = td["nodes"][:, :, FEATURE_INDEX["i_limit"]].unsqueeze(2)
            # i_limit=0인 노드(부하, 배터리)는 무한대로 처리하여 전류 검사 통과
            inf_limits = torch.where(parent_limits > 0, parent_limits, float("inf"))
            base_mask &= (prospective_draw.transpose(1, 2) <= inf_limits)

            # --- 💡 3. 복잡한 전역 제약조건 마스킹 ---
            final_mask = base_mask
            
            # config.json의 제약조건 파싱
            constraints = self.generator.config.constraints
            loads_info = self.generator.config.loads
            load_map = {l['name']: l for l in loads_info}
            node_names = self.generator.config.node_names
            
            # 조상(ancestor) 행렬 계산: 플로이드-워셜 알고리즘과 유사
            ancestors = td["adj_matrix"].clone()
            for k in range(num_nodes):
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        ancestors[:, i, j] |= ancestors[:, i, k] & ancestors[:, k, j]

            # 1. 독립 레일 (Independent Rail)
            for i, load in enumerate(loads_info):
                rail_type = load.get("independent_rail_type")
                if not rail_type: continue
                
                load_idx = self.generator.num_nodes - self.generator.num_loads + i
                
                if rail_type == 'exclusive_supplier':
                    # 이 부하(load_idx)를 자식으로 가지려는 부모(p)는 다른 자식이 있으면 안 됨
                    # 즉, adj_matrix[p]의 out-degree가 0이어야 함
                    num_children = td["adj_matrix"].sum(dim=2) # (B, N)
                    is_ok_parent = (num_children == 0)
                    final_mask[:, :, load_idx] &= is_ok_parent

                elif rail_type == 'exclusive_path':
                    # 이 부하의 조상들은 다른 자식을 가질 수 없음
                    load_ancestors = ancestors[:, :, load_idx] # (B, N)
                    num_children = td["adj_matrix"].sum(dim=2)
                    # 조상이면서 자식이 1개 초과인 노드 찾기 (배터리 제외)
                    violating_ancestors = load_ancestors & (num_children > 1) & is_ic.unsqueeze(0)
                    # 위반하는 조상을 부모로 삼으려는 모든 연결을 금지
                    final_mask &= ~violating_ancestors.unsqueeze(2)

            # 2. 전원 인가 순서 (Power Sequence)
            for seq in constraints.get("power_sequences", []):
                j_name, k_name = seq['j'], seq['k']
                if j_name not in node_names or k_name not in node_names: continue
                j_idx, k_idx = node_names.index(j_name), node_names.index(k_name)

                # k의 부모(p)는 j의 조상이 될 수 없음
                j_ancestors = ancestors[:, :, j_idx] # (B, N)
                final_mask[:, :, k_idx] &= ~j_ancestors

                # j와 k는 같은 부모를 가질 수 없음
                is_k_parent_mask = td["adj_matrix"][:, :, k_idx]  # (B, N) 모양의 불리언 마스크
                # final_mask에서 자식이 j_idx인 슬라이스 (B, N)를 선택한 뒤,
                # k의 부모인 위치에 False를 적용합니다.
                final_mask[:, :, j_idx][is_k_parent_mask] = False

        return final_mask
    
    def get_reward(self, td: TensorDict, done: torch.Tensor) -> torch.Tensor:
        reward = torch.zeros(td.batch_size[0], device=self.device)
        if done.any():
            # 비용 계산
            node_costs = td["nodes"][done, :, FEATURE_INDEX["cost"]]
            ic_mask = td["nodes"][done, :, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] == 1
            # is_used_mask: connected_nodes_mask에서 배터리 제외
            is_used_mask = td["connected_nodes_mask"][done].clone()
            is_used_mask[:, 0] = False
            
            used_ic_mask = is_used_mask & ic_mask
            total_cost = (node_costs * used_ic_mask).sum(dim=-1)
            reward[done] = -total_cost
            
            # --- 💡 4. 슬립 전류 제약 위반 시 페널티 ---
            max_sleep_current = self.generator.config.constraints.get("max_sleep_current", 0.0)
            if max_sleep_current > 0:
                # (구현 간소화를 위해 간단한 로직 적용, 실제로는 더 정교한 계산 필요)
                # Always-on 부하들의 슬립 전류 합계만으로 간단히 계산
                loads_info = self.generator.config.loads
                always_on_loads_current = sum(
                    l['current_sleep'] for l in loads_info if l.get('always_on_in_sleep')
                )
                # 실제로는 IC의 quiescent/operating current도 전파해야 함
                if always_on_loads_current > max_sleep_current:
                    reward[done] -= 100.0 # 큰 페널티
                    
        return reward.unsqueeze(-1)
        
    def _make_spec(self):
        # ... (기존과 동일, 단 instance_repeats 제거) ...
        num_nodes = self.generator.num_nodes
        self.observation_spec = Composite({
            "nodes": Unbounded(shape=(num_nodes, FEATURE_DIM)),
            "prompt_features": Unbounded(shape=(2,)),
            "adj_matrix": Categorical(n=2, shape=(num_nodes, num_nodes), dtype=torch.bool),
            "connected_nodes_mask": Categorical(n=2, shape=(num_nodes,), dtype=torch.bool),
            "ic_current_draw": Unbounded(shape=(num_nodes,)),
            "step_count": UnboundedDiscrete(shape=(1,), dtype=torch.long),
        })
        self.action_spec = UnboundedDiscrete(shape=(2,), dtype=torch.long)
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Categorical(n=2, shape=(1,), dtype=torch.bool)

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

    def _trace_path(self, batch_idx, head_idx, adj_matrix):
        path = [head_idx.item()]
        current = head_idx
        for _ in range(adj_matrix.shape[1]): # 무한루프 방지
            parents = adj_matrix[batch_idx, :, current].nonzero(as_tuple=True)[0]
            if parents.numel() == 0: break
            parent = parents[0]
            path.append(parent.item())
            current = parent
        return torch.tensor(path, device=self.device)

    def _trace_path_batch(self, b_idx, head_indices, adj_matrix):
        """배치 전체에 대해, 각 경로의 노드들을 찾는 마스크 반환"""
        num_nodes = adj_matrix.shape[1]
        path_mask = torch.zeros_like(adj_matrix[:, :, 0], dtype=torch.bool)
        path_mask[b_idx, head_indices] = True
        
        current_heads = head_indices
        for _ in range(num_nodes): # 무한루프 방지
            # 행렬 곱셈을 이용한 효율적인 부모 노드 탐색
            adj_T = adj_matrix.transpose(1, 2)
            parents = torch.where(adj_T[b_idx, current_heads])[1]
            if parents.numel() == 0: break
            path_mask[b_idx, parents] = True
            current_heads = parents
        return path_mask