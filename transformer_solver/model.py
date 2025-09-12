# transformer_solver/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensordict import TensorDict

from common.pocat_defs import FEATURE_DIM, FEATURE_INDEX, NODE_TYPE_BATTERY, NODE_TYPE_IC, NODE_TYPE_LOAD
from common.utils.common import batchify
# 💡 PocatEnv는 더 이상 forward 내부에서 직접 사용되지 않습니다.
# from .pocat_env import PocatEnv 

# ... (RMSNorm, Normalization, EncoderLayer 등 다른 클래스는 이전과 동일하게 유지) ...
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Normalization(nn.Module):
    def __init__(self, embedding_dim, norm_type='rms', **kwargs):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == 'layer': self.norm = nn.LayerNorm(embedding_dim)
        elif self.norm_type == 'rms': self.norm = RMSNorm(embedding_dim)
        elif self.norm_type == 'instance': self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        else: raise NotImplementedError
    def forward(self, x):
        if self.norm_type == 'instance': return self.norm(x.transpose(1, 2)).transpose(1, 2)
        else: return self.norm(x)

class ParallelGatedMLP(nn.Module):
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        inner_size = int(2 * hidden_size * 4 / 3)
        multiple_of = 256
        inner_size = multiple_of * ((inner_size + multiple_of - 1) // multiple_of)
        self.l1, self.l2, self.l3 = nn.Linear(hidden_size, inner_size, bias=False), nn.Linear(hidden_size, inner_size, bias=False), nn.Linear(inner_size, hidden_size, bias=False)
        self.act = F.silu
    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        return self.l3(self.act(z1) * z2)

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_hidden_dim, **kwargs):
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)
    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))

def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    batch_s, n = qkv.size(0), qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    return q_reshaped.transpose(1, 2)

def multi_head_attention(q, k, v, attention_mask=None, sparse_type=None):
    batch_s, head_num, n, key_dim = q.shape

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / (key_dim ** 0.5)
    
    if attention_mask is not None:
        score_scaled = score_scaled.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)
        
    if sparse_type == 'topk':
        seq_len = score_scaled.size(-1)
        k_for_topk = max(1, seq_len // 2) 

        top_k_values, top_k_indices = torch.topk(score_scaled, k=k_for_topk, dim=-1)
        
        topk_mask = torch.zeros_like(score_scaled, dtype=torch.bool).scatter_(-1, top_k_indices, True)
        attention_weights = score_scaled.masked_fill(~topk_mask, -1e9)
        weights = nn.Softmax(dim=3)(attention_weights)
    else:
        weights = nn.Softmax(dim=3)(score_scaled)
        
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    return out_transposed.contiguous().view(batch_s, n, head_num * key_dim)

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim, ffd='siglu', use_sparse=False, **model_params):
        super().__init__()
        self.embedding_dim, self.head_num, self.qkv_dim = embedding_dim, head_num, qkv_dim
        self.Wq, self.Wk, self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False), nn.Linear(embedding_dim, head_num * qkv_dim, bias=False), nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.normalization1 = Normalization(embedding_dim, **model_params)
        if ffd == 'siglu': self.feed_forward = ParallelGatedMLP(hidden_size=embedding_dim, **model_params)
        else: self.feed_forward = FeedForward(embedding_dim=embedding_dim, **model_params)
        self.normalization2 = Normalization(embedding_dim, **model_params)
        self.use_sparse = use_sparse

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        q, k, v = reshape_by_heads(self.Wq(x), self.head_num), reshape_by_heads(self.Wk(x), self.head_num), reshape_by_heads(self.Wv(x), self.head_num)
        sparse_type = 'topk' if self.use_sparse else None
        mha_out = self.multi_head_combine(multi_head_attention(q, k, v, attention_mask=attention_mask, sparse_type=sparse_type))
        h = self.normalization1(x + mha_out)
        return self.normalization2(h + self.feed_forward(h))

class PocatPromptNet(nn.Module):
    def __init__(self, embedding_dim: int, num_nodes: int, **kwargs):
        super().__init__()
        self.scalar_net = nn.Sequential(
            nn.Linear(4, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )
        self.matrix_net = nn.Sequential(
            nn.Linear(num_nodes * num_nodes, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2)
        )
        self.final_processor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )

    def forward(self, scalar_features: torch.Tensor, matrix_features: torch.Tensor) -> torch.Tensor:
        scalar_embedding = self.scalar_net(scalar_features)
        batch_size = matrix_features.shape[0]
        matrix_flat = matrix_features.view(batch_size, -1)
        matrix_embedding = self.matrix_net(matrix_flat)
        combined_embedding = torch.cat([scalar_embedding, matrix_embedding], dim=-1)
        final_prompt_embedding = self.final_processor(combined_embedding)
        return final_prompt_embedding.unsqueeze(1)


class PocatEncoder(nn.Module):
    def __init__(self, embedding_dim: int, encoder_layer_num: int = 6, **model_params):
        super().__init__()
        self.embedding_layer = nn.Linear(FEATURE_DIM, embedding_dim)
        
        sparse_params = model_params.copy()
        sparse_params['use_sparse'] = True
        
        global_params = model_params.copy()
        global_params['use_sparse'] = False
        
        self.sparse_layers = nn.ModuleList([EncoderLayer(embedding_dim=embedding_dim, **sparse_params) for _ in range(encoder_layer_num)])
        self.global_layers = nn.ModuleList([EncoderLayer(embedding_dim=embedding_dim, **global_params) for _ in range(encoder_layer_num)])
        
        self.sparse_fusion = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(encoder_layer_num)])
        self.global_fusion = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(encoder_layer_num - 1)])

    def _create_connectivity_mask(self, td: TensorDict) -> torch.Tensor:
        nodes = td['nodes']
        batch_size, num_nodes, _ = nodes.shape
        node_types = nodes[0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
        is_parent = (node_types == NODE_TYPE_IC) | (node_types == NODE_TYPE_BATTERY)
        is_child = (node_types == NODE_TYPE_IC) | (node_types == NODE_TYPE_LOAD)
        parent_mask = is_parent.unsqueeze(0).unsqueeze(2).expand(batch_size, num_nodes, num_nodes)
        child_mask = is_child.unsqueeze(0).unsqueeze(1).expand(batch_size, num_nodes, num_nodes)
        parent_vout_min, parent_vout_max = nodes[:, :, FEATURE_INDEX["vout_min"]].unsqueeze(2), nodes[:, :, FEATURE_INDEX["vout_max"]].unsqueeze(2)
        child_vin_min, child_vin_max = nodes[:, :, FEATURE_INDEX["vin_min"]].unsqueeze(1), nodes[:, :, FEATURE_INDEX["vin_max"]].unsqueeze(1)
        voltage_compatible = (parent_vout_min <= child_vin_max) & (parent_vout_max >= child_vin_min)
        mask = parent_mask & child_mask & voltage_compatible
        mask.diagonal(dim1=-2, dim2=-1).fill_(False)
        return mask

    def forward(self, td: TensorDict, prompt_embedding: torch.Tensor) -> torch.Tensor:
        node_features = td['nodes']
        batch_size, num_nodes, _ = node_features.shape
        node_embeddings = self.embedding_layer(node_features)
        connectivity_mask = self._create_connectivity_mask(td)
        global_input = torch.cat((node_embeddings, prompt_embedding), dim=1)
        global_attention_mask = torch.ones(batch_size, num_nodes + 1, num_nodes + 1, dtype=torch.bool, device=node_embeddings.device)
        global_attention_mask[:, :num_nodes, :num_nodes] = connectivity_mask
        sparse_out = node_embeddings
        global_out = global_input
        
        for i in range(len(self.sparse_layers)):
            sparse_out = self.sparse_layers[i](sparse_out, attention_mask=connectivity_mask)
            global_out = self.global_layers[i](global_out, attention_mask=global_attention_mask)
            sparse_out = sparse_out + self.sparse_fusion[i](global_out[:, :num_nodes])
            if i < len(self.global_layers) - 1:
                global_nodes = global_out[:, :num_nodes] + self.global_fusion[i](sparse_out)
                global_out = torch.cat((global_nodes, global_out[:, num_nodes:]), dim=1)  
        return sparse_out

class PocatDecoder(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim, **model_params):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.Wq_load_select = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_parent_select = nn.Linear(embedding_dim * 2, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_logit = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

    def forward(self, context_embedding, encoded_nodes, mask, phase):
        k = reshape_by_heads(self.Wk(encoded_nodes), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(encoded_nodes), head_num=self.head_num)

        if phase == 0:
            q = reshape_by_heads(self.Wq_load_select(context_embedding), head_num=self.head_num)
        elif phase == 1:
            q = reshape_by_heads(self.Wq_parent_select(context_embedding), head_num=self.head_num)
        
        mha_out = multi_head_attention(q, k, v, attention_mask=mask)
        mh_atten_out = self.multi_head_combine(mha_out)
        
        logit_k = self.Wk_logit(encoded_nodes).transpose(1, 2)
        scores = torch.matmul(mh_atten_out, logit_k).squeeze(1) / (self.embedding_dim ** 0.5)
        
        return scores

class PocatModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.prompt_net = PocatPromptNet(embedding_dim=model_params['embedding_dim'], num_nodes=model_params['num_nodes'])
        self.encoder = PocatEncoder(**model_params)
        self.decoder = PocatDecoder(**model_params)
        self.context_gru = nn.GRUCell(model_params['embedding_dim'] * 2, model_params['embedding_dim'])

    # 💡 1. `env` 인자를 제거하고, forward 메서드 시그니처를 수정합니다.
    def forward(self, td: TensorDict, decode_type: str = 'greedy', pbar: object = None, status_msg: str = "", log_fn=None):
        
        # 💡 2. 디코딩 상태를 저장할 텐서들을 여기서 직접 초기화합니다.
        batch_size, num_nodes, _ = td["nodes"].shape
        b_idx = torch.arange(batch_size, device=td.device)
        
        adj_matrix = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=td.device)
        main_tree_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=td.device)
        main_tree_mask[:, 0] = True # 배터리는 항상 메인 트리에 포함
        ic_current_draw = torch.zeros(batch_size, num_nodes, device=td.device)
        decoding_phase = torch.zeros(batch_size, 1, dtype=torch.long, device=td.device)
        trajectory_head = torch.full((batch_size, 1), -1, dtype=torch.long, device=td.device)
        
        node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
        is_load = node_types == NODE_TYPE_LOAD
        unconnected_loads_mask = is_load.unsqueeze(0).expand(batch_size, -1)
        
        done = torch.zeros(batch_size, 1, dtype=torch.bool, device=td.device)

        # --- 인코딩 ---
        prompt_embedding = self.prompt_net(td["scalar_prompt_features"], td["matrix_prompt_features"])
        encoded_nodes = self.encoder(td, prompt_embedding)

        # --- POMO 준비 ---
        start_nodes_idx = torch.where(is_load)[0]
        num_starts = len(start_nodes_idx)
        
        # POMO를 위해 모든 상태 텐서를 확장합니다.
        td = batchify(td, num_starts)
        encoded_nodes = batchify(encoded_nodes, num_starts)
        adj_matrix = batchify(adj_matrix, num_starts)
        main_tree_mask = batchify(main_tree_mask, num_starts)
        ic_current_draw = batchify(ic_current_draw, num_starts)
        decoding_phase = batchify(decoding_phase, num_starts)
        trajectory_head = batchify(trajectory_head, num_starts)
        unconnected_loads_mask = batchify(unconnected_loads_mask, num_starts)
        done = batchify(done, num_starts)
        
        expanded_batch_size = td.batch_size[0]
        context_embedding = encoded_nodes.mean(dim=1)
        log_probs, actions = [], []
        
        # --- 첫 번째 액션(Load 선택) 및 상태 업데이트 ---
        action_part1 = start_nodes_idx.repeat(batch_size)
        action_part2 = torch.zeros_like(action_part1)
        action = torch.stack([action_part1, action_part2], dim=1)
        
        trajectory_head = action_part1.unsqueeze(-1)
        unconnected_loads_mask[torch.arange(expanded_batch_size), action_part1] = False
        decoding_phase[:, 0] = 1

        actions.append(action)
        log_probs.append(torch.zeros(expanded_batch_size, device=td.device))

        # --- 디코딩 루프 시작 ---
        for step in range(num_nodes * 2): # 최대 스텝 수 제한
            if done.all():
                break

            # 💡 3. 마스크 계산 로직을 여기에 통합 (벡터화)
            mask = torch.zeros(expanded_batch_size, num_nodes, num_nodes, dtype=torch.bool, device=td.device)
            phase = decoding_phase.squeeze(-1)
            b_idx_loop = torch.arange(expanded_batch_size, device=td.device)
            
            # Phase 0: 새 Load 선택
            phase0_mask = (phase == 0)
            if phase0_mask.any():
                mask[phase0_mask, :, 0] = unconnected_loads_mask[phase0_mask]

            # Phase 1: 부모 선택
            phase1_mask = (phase == 1)
            if phase1_mask.any():
                b_p1 = b_idx_loop[phase1_mask]
                child_indices = trajectory_head[b_p1].squeeze(-1)
                
                can_be_parent = ~is_load.unsqueeze(0).expand(len(b_p1), -1)
                
                child_vin_min = td["nodes"][b_p1, child_indices, FEATURE_INDEX["vin_min"]]
                child_vin_max = td["nodes"][b_p1, child_indices, FEATURE_INDEX["vin_max"]]
                parent_vout_min = td["nodes"][b_p1, :, FEATURE_INDEX["vout_min"]]
                parent_vout_max = td["nodes"][b_p1, :, FEATURE_INDEX["vout_max"]]
                is_voltage_compatible = (parent_vout_min <= child_vin_max.unsqueeze(1)) & \
                                        (parent_vout_max >= child_vin_min.unsqueeze(1))
                can_be_parent &= is_voltage_compatible
                
                # 전류 한계 체크 (간소화된 버전)
                # 실제로는 경로 전체 전류를 계산해야 하지만, 일단 가장 간단한 버전으로 구현
                child_current = td["nodes"][b_p1, child_indices, FEATURE_INDEX["current_active"]]
                prospective_draw = ic_current_draw[b_p1] + child_current.unsqueeze(1)
                parent_limits = td["nodes"][b_p1, :, FEATURE_INDEX["i_limit"]]
                can_be_parent &= (prospective_draw <= parent_limits) | (parent_limits == 0)

                # 자신을 부모로 선택하는 것 방지
                can_be_parent[torch.arange(len(b_p1)), child_indices] = False
                
                mask[b_p1, :, child_indices] = can_be_parent
                
            # --- 디코더 호출 ---
            if phase0_mask.any():
                main_tree_nodes = encoded_nodes[phase0_mask] * main_tree_mask[phase0_mask].unsqueeze(-1)
                context = main_tree_nodes.sum(1) / (main_tree_mask[phase0_mask].sum(1, keepdim=True) + 1e-9)
                mask_for_load_select = mask[phase0_mask, :, 0]
                scores = self.decoder(context.unsqueeze(1), encoded_nodes[phase0_mask], mask_for_load_select.unsqueeze(1), 0)
                scores.masked_fill_(~mask_for_load_select, -1e9)
                
                log_prob = F.log_softmax(scores, dim=-1)
                selected_node = Categorical(probs=log_prob.exp()).sample() if decode_type == 'sampling' else log_prob.argmax(dim=-1)
                action = torch.stack([selected_node, torch.zeros_like(selected_node)], dim=1)
                log_prob_val = log_prob.gather(1, selected_node.unsqueeze(-1)).squeeze(-1)

            if phase1_mask.any():
                trajectory_head_idx = trajectory_head[phase1_mask].squeeze(-1)
                head_emb = encoded_nodes[phase1_mask, trajectory_head_idx]
                context = torch.cat([context_embedding[phase1_mask], head_emb], dim=1)
                
                mask_for_parent_select = mask[phase1_mask, :, trajectory_head_idx]
                scores = self.decoder(context.unsqueeze(1), encoded_nodes[phase1_mask], mask_for_parent_select.unsqueeze(1), 1)
                scores.masked_fill_(~mask_for_parent_select, -1e9)
                
                log_prob = F.log_softmax(scores, dim=-1)
                selected_parent_idx = Categorical(probs=log_prob.exp()).sample() if decode_type == 'sampling' else log_prob.argmax(dim=-1)
                action = torch.stack([trajectory_head_idx, selected_parent_idx], dim=1)
                log_prob_val = log_prob.gather(1, selected_parent_idx.unsqueeze(-1)).squeeze(-1)

            # --- 상태 업데이트 로직 통합 ---
            full_action = torch.zeros_like(actions[0])
            full_log_prob = torch.zeros_like(log_probs[0])
            if phase0_mask.any():
                full_action[phase0_mask] = action
                full_log_prob[phase0_mask] = log_prob_val
            if phase1_mask.any():
                full_action[phase1_mask] = action
                full_log_prob[phase1_mask] = log_prob_val

            # Phase 0 액션에 대한 상태 업데이트
            if phase0_mask.any():
                selected_load = full_action[phase0_mask, 0]
                trajectory_head[phase0_mask] = selected_load.unsqueeze(-1)
                unconnected_loads_mask[phase0_mask, selected_load] = False
                decoding_phase[phase0_mask] = 1

            # Phase 1 액션에 대한 상태 업데이트
            if phase1_mask.any():
                child_idx, parent_idx = full_action[phase1_mask, 0], full_action[phase1_mask, 1]
                adj_matrix[phase1_mask, parent_idx, child_idx] = True

                is_parent_in_main_tree = main_tree_mask[phase1_mask, parent_idx]
                
                # 경로가 메인 트리에 합류된 경우
                b_merged = torch.where(phase1_mask)[0][is_parent_in_main_tree]
                if b_merged.numel() > 0:
                    decoding_phase[b_merged] = 0 # 다음 스텝은 새 Load 선택
                    # 경로 전체를 메인 트리에 추가 (단순화된 버전)
                    # 실제로는 재귀적으로 모든 조상을 찾아야 함
                    main_tree_mask[b_merged, child_idx[is_parent_in_main_tree]] = True
                    main_tree_mask[b_merged, parent_idx[is_parent_in_main_tree]] = True

                # 경로가 아직 연결되지 않은 경우
                b_unmerged = torch.where(phase1_mask)[0][~is_parent_in_main_tree]
                if b_unmerged.numel() > 0:
                    trajectory_head[b_unmerged] = parent_idx[~is_parent_in_main_tree].unsqueeze(-1)
            
            # 완료 조건 체크
            done = (unconnected_loads_mask.sum(dim=1) == 0).unsqueeze(-1)
            
            actions.append(full_action)
            log_probs.append(full_log_prob)
            
            child_emb = encoded_nodes[b_idx_loop, full_action[:, 0]]
            parent_emb = encoded_nodes[b_idx_loop, full_action[:, 1]]
            context_embedding = self.context_gru(torch.cat([child_emb, parent_emb], dim=1), context_embedding)

        # --- 최종 보상 계산 ---
        is_used_mask = adj_matrix.any(dim=1) | adj_matrix.any(dim=2)
        node_costs = td["nodes"][:, :, FEATURE_INDEX["cost"]]
        ic_mask = td["nodes"][:, :, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] == 1
        used_ic_mask = is_used_mask & ic_mask
        final_reward = -(node_costs * used_ic_mask).sum(dim=-1)

        return {
            "reward": final_reward,
            "log_likelihood": torch.stack(log_probs, 1).sum(1),
            "actions": torch.stack(actions, 1)
        }