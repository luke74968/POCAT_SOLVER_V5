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
        self._set_seed(None) # 생성자에서 호출은 되어 있으나, 아래에 메소드 정의가 필요합니다.

    # --- 👇 1. 누락된 _make_spec 메소드 추가 ---
    def _make_spec(self):
        """환경의 observation, action, reward 스펙을 정의합니다."""
        num_nodes = self.generator.num_nodes
        
        # 관측 공간(Observation Space) 정의
        self.observation_spec = Composite({
            "nodes": Unbounded(
                shape=(num_nodes, FEATURE_DIM),
                dtype=torch.float32,
            ),
            "prompt_features": Unbounded(
                shape=(5,), # ambient_temp, max_sleep_current, current_margin, thermal_margin_percent, power_sequence_count
                dtype=torch.float32,
            ),
            "adj_matrix": Unbounded(
                shape=(num_nodes, num_nodes),
                dtype=torch.bool,
            ),
            "main_tree_mask": Unbounded(
                shape=(num_nodes,),
                dtype=torch.bool,
            ),
            "ic_current_draw": Unbounded(
                shape=(num_nodes,),
                dtype=torch.float32,
            ),
            "decoding_phase": Categorical(
                shape=(1,),
                n=2, # 0: 새 Load 선택, 1: Trajectory 구축
                dtype=torch.long,
            ),
            "trajectory_head": UnboundedDiscrete(
                shape=(1,),
                dtype=torch.long,
            ),
            "unconnected_loads_mask": Unbounded(
                shape=(num_nodes,),
                dtype=torch.bool,
            ),
            "step_count": UnboundedDiscrete(
                shape=(1,),
                dtype=torch.long,
            ),
        })
        
        # 행동 공간(Action Space) 정의: [자식 노드, 부모 노드]
        self.action_spec = UnboundedDiscrete(
            shape=(2,),
            dtype=torch.long,
        )
        
        # 보상(Reward) 스펙 정의
        self.reward_spec = Unbounded(shape=(1,))

    # --- 👇 2. 누락된 _set_seed 메소드 추가 ---
    def _set_seed(self, seed: Optional[int] = None):
        """환경의 랜덤 시드를 설정합니다. (torchrl 필수 구현)"""
        # 현재 환경은 자체적인 랜덤 요소가 없으므로 특별한 로직은 필요 없습니다.
        # 하지만 EnvBase를 상속받기 위해 반드시 구현해야 합니다.
        if seed is not None:
            torch.manual_seed(seed)

    # --- 👇 1. 누락되었던 select_start_nodes 메소드 추가 ---
    def select_start_nodes(self, td: TensorDict) -> Tuple[int, torch.Tensor]:
        """POMO decoding을 위해 시작 노드(모든 Load)를 선택합니다."""
        # 노드 타입 정보는 배치 내에서 동일하므로 0번 인덱스만 사용합니다.
        node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
        start_nodes_idx = torch.where(node_types == NODE_TYPE_LOAD)[0]
        num_starts = len(start_nodes_idx)
        return num_starts, start_nodes_idx

    # --- 👇 2. 누락되었던 경로 추적 헬퍼 메소드들 추가 ---
    def _trace_path(self, b_idx: int, start_node: int, adj_matrix: torch.Tensor) -> list[int]:
        """단일 배치 항목에 대해 start_node에서 시작하는 경로를 역추적하여 노드 인덱스 리스트를 반환합니다."""
        path = [start_node]
        current_node = start_node
        # adj_matrix[b_idx, parent, child] 형태이므로, current_node를 자식으로 갖는 부모를 찾습니다.
        while True:
            parents = adj_matrix[b_idx, :, current_node].nonzero(as_tuple=True)[0]
            if parents.numel() == 0:
                break
            parent_node = parents[0].item() # 경로는 하나뿐이라고 가정
            path.append(parent_node)
            current_node = parent_node
        return path

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
        phase = td["decoding_phase"].squeeze(-1)

        next_obs = td.clone()

        phase0_mask = phase == 0
        phase1_mask = phase == 1
        b_phase0 = b_idx[phase0_mask]
        b_phase1 = b_idx[phase1_mask]

        if b_phase0.numel() > 0:  # 새 Load 선택 단계
            selected_load = action[b_phase0, 0]
            next_obs["trajectory_head"][b_phase0] = selected_load.unsqueeze(-1)
            next_obs["unconnected_loads_mask"][b_phase0, selected_load] = False
            next_obs["decoding_phase"][b_phase0, 0] = 1  # 다음은 경로 구축 단계로

        if b_phase1.numel() > 0:  # Trajectory 구축 단계
            child_idx, parent_idx = action[b_phase1, 0], action[b_phase1, 1]
            next_obs["adj_matrix"][b_phase1[:, None], parent_idx, child_idx] = True

            # [수정] 전류 전파 로직 구현
            path_nodes_mask = self._trace_path_batch(b_phase1, child_idx, next_obs["adj_matrix"])
            path_nodes_currents = (
                td["nodes"][b_phase1] * path_nodes_mask.unsqueeze(-1)
            )[:, :, FEATURE_INDEX["current_active"]]

            for idx, b in enumerate(b_phase1.tolist()):
                total_child_current = path_nodes_currents[idx].sum()
                ancestor = parent_idx[idx].item()
                while ancestor != 0:
                    next_obs["ic_current_draw"][b, ancestor] += total_child_current
                    ancestors_of_ancestor = next_obs["adj_matrix"][b, :, ancestor].nonzero(as_tuple=True)[0]
                    if ancestors_of_ancestor.numel() == 0:
                        break
                    ancestor = ancestors_of_ancestor[0].item()

            is_parent_in_main_tree = next_obs["main_tree_mask"][b_phase1, parent_idx]

            for idx, b in enumerate(b_phase1.tolist()):
                if is_parent_in_main_tree[idx]:
                    path_nodes_indices = self._trace_path(b, child_idx[idx], next_obs["adj_matrix"])
                    next_obs["main_tree_mask"][b, path_nodes_indices] = True
                    if next_obs["unconnected_loads_mask"][b].sum() == 0:
                        next_obs["done"][b] = True
                    else:
                        next_obs["decoding_phase"][b] = 0
                else:
                    next_obs["trajectory_head"][b] = parent_idx[idx]

        next_obs.set("step_count", td["step_count"] + 1)
        
        return TensorDict({
            "next": next_obs,
            "reward": self.get_reward(next_obs),
            "done": next_obs["done"],
        }, batch_size=td.batch_size)
    
    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        batch_size, num_nodes, _ = td["nodes"].shape
        phase = td["decoding_phase"].squeeze(-1)

        mask = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=self.device)

        phase0_idx = torch.where(phase == 0)[0]
        phase1_idx = torch.where(phase == 1)[0]

        if phase0_idx.numel() > 0:
            # 자식: 아직 연결 안된 Load만 가능, 부모: 사용 안함 (0으로 고정)
            mask[phase0_idx, :, 0] = td["unconnected_loads_mask"][phase0_idx]

        if phase1_idx.numel() > 0:
            b_idx = phase1_idx
            child_indices = td["trajectory_head"][b_idx].squeeze(-1)

            # [수정] 'can_be_parent' 변수 초기화
            can_be_parent = torch.ones(len(b_idx), num_nodes, dtype=torch.bool, device=self.device)

            node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
            is_ic = node_types == NODE_TYPE_IC

            current_path_mask = self._trace_path_batch(b_idx, child_indices, td["adj_matrix"])

            # 기본 후보: (아직 사용되지 않은 IC) 또는 (이미 메인 트리에 속한 노드)
            can_be_parent &= (
                is_ic.unsqueeze(0) & ~current_path_mask & ~td["main_tree_mask"][b_idx]
            ) | td["main_tree_mask"][b_idx]

            # 자기 자신, 이미 경로에 포함된 노드는 부모가 될 수 없음
            can_be_parent[torch.arange(len(b_idx), device=self.device), child_indices] = False

            # 1. 전압 호환성
            head_node_vin_min = td["nodes"][b_idx, child_indices, FEATURE_INDEX["vin_min"]]
            parent_vout_max = td["nodes"][b_idx, :, FEATURE_INDEX["vout_max"]]
            can_be_parent &= parent_vout_max >= head_node_vin_min.unsqueeze(1)

            # 2. 전류 한계
            path_nodes_currents = (
                td["nodes"][b_idx, :, FEATURE_INDEX["current_active"]] * current_path_mask
            ).sum(dim=1)
            prospective_draw = td["ic_current_draw"][b_idx] + path_nodes_currents.unsqueeze(1)
            parent_limits = td["nodes"][b_idx, :, FEATURE_INDEX["i_limit"]]
            inf_limits = torch.where(parent_limits > 0, parent_limits, float("inf"))
            can_be_parent &= prospective_draw <= inf_limits

            # 3. 전역 제약조건 (Independent Rail, Power Sequence)
            constraints = self.generator.config.constraints
            loads_info = self.generator.config.loads
            node_names = self.generator.config.node_names

            # 조상(ancestor) 행렬 계산: 현재 메인 트리에 대해서만 계산
            ancestors = td["adj_matrix"][b_idx].clone()
            for k in range(num_nodes):
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        ancestors[:, i, j] |= ancestors[:, i, k] & ancestors[:, k, j]

            # 3a. Independent Rail
            head_load_idx = child_indices - (1 + len(self.generator.config.available_ics))
            for idx, b in enumerate(b_idx.tolist()):
                if 0 <= head_load_idx[idx] < len(loads_info):
                    load = loads_info[head_load_idx[idx]]
                    rail_type = load.get("independent_rail_type")
                    if rail_type == "exclusive_supplier":
                        num_children = td["adj_matrix"][b].sum(dim=1)
                        can_be_parent[idx] &= num_children == 0
                    elif rail_type == "exclusive_path":
                        # 경로에 속할 모든 잠재적 부모들이 다른 자식을 가지면 안됨
                        num_children = td["adj_matrix"][b].sum(dim=1)
                        can_be_parent[idx] &= num_children <= 1

            # 3b. Power Sequence
            for seq in constraints.get("power_sequences", []):
                if seq.get("f") != 1:
                    continue
                j_name, k_name = seq["j"], seq["k"]
                if j_name not in node_names or k_name not in node_names:
                    continue
                j_idx, k_idx = node_names.index(j_name), node_names.index(k_name)

                # 현재 head가 k_idx인 경우, j의 조상이 될 수 있는 노드는 부모가 될 수 없음
                is_head_k = child_indices == k_idx
                if is_head_k.any():
                    j_ancestors = ancestors[is_head_k, :, j_idx]
                    can_be_parent[is_head_k] &= ~j_ancestors

            mask[b_idx, :, child_indices] = can_be_parent

        return mask
    
    def get_reward(self, td: TensorDict) -> torch.Tensor:
        """
        Calculates the reward based on the final state of the power tree.
        The reward is the negative of the total cost of used ICs.
        This function is called only when an episode is done.
        """
        reward = torch.zeros(td.batch_size[0], device=self.device)
        done = td["done"].squeeze(-1)
        
        if done.any():
            # Calculate cost based on the final adjacency matrix
            is_used_mask = td["adj_matrix"][done].any(dim=1) | td["adj_matrix"][done].any(dim=2)
            
            node_costs = td["nodes"][done, :, FEATURE_INDEX["cost"]]
            ic_mask = td["nodes"][done, :, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] == 1
            
            used_ic_mask = is_used_mask & ic_mask
            total_cost = (node_costs * used_ic_mask).sum(dim=-1)
            reward[done] = -total_cost
            
            # (Optional) Add penalty for violating sleep current constraint
            max_sleep_current = self.generator.config.constraints.get("max_sleep_current", 0.0)
            if max_sleep_current > 0:
                loads_info = self.generator.config.loads
