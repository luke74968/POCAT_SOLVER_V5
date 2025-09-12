# transformer_solver/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensordict import TensorDict

from common.pocat_defs import FEATURE_DIM, FEATURE_INDEX, NODE_TYPE_BATTERY, NODE_TYPE_IC, NODE_TYPE_LOAD
from common.utils.common import batchify

# --- 이전과 동일한 클래스 정의 (Encoder, Decoder 등) ---
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

    def forward(self, td: TensorDict, decode_type: str = 'greedy', pbar: object = None, log_fn=None):
        batch_size, num_nodes, _ = td["nodes"].shape
        device = td.device
        
        adj_matrix = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=device)
        main_tree_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
        main_tree_mask[:, 0] = True
        ic_current_draw = torch.zeros(batch_size, num_nodes, device=device)
        decoding_phase = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        trajectory_head = torch.full((batch_size, 1), -1, dtype=torch.long, device=device)
        
        node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
        is_load = node_types == NODE_TYPE_LOAD
        unconnected_loads_mask = is_load.unsqueeze(0).expand(batch_size, -1)
        
        done = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        prompt_embedding = self.prompt_net(td["scalar_prompt_features"], td["matrix_prompt_features"])
        encoded_nodes = self.encoder(td, prompt_embedding)

        start_nodes_idx = torch.where(is_load)[0]
        num_starts = len(start_nodes_idx)
        
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
        
        action_part1 = start_nodes_idx.repeat(batch_size)
        action = torch.stack([action_part1, torch.zeros_like(action_part1)], dim=1)
        
        trajectory_head = action_part1.unsqueeze(-1)
        unconnected_loads_mask[torch.arange(expanded_batch_size), action_part1] = False
        decoding_phase[:, 0] = 1

        actions.append(action)
        log_probs.append(torch.zeros(expanded_batch_size, device=device))

        for _ in range(num_nodes * 2):
            if done.all():
                break

            mask = torch.zeros(expanded_batch_size, num_nodes, num_nodes, dtype=torch.bool, device=device)
            phase = decoding_phase.squeeze(-1)
            b_idx_loop = torch.arange(expanded_batch_size, device=device)
            
            phase0_mask = (phase == 0)
            phase1_mask = (phase == 1)

            # 💡 --- 핵심 수정 시작 ---
            # 1. 각 Phase별로 액션과 로그 확률을 저장할 변수를 분리합니다.
            action_p0, action_p1 = None, None
            log_prob_val_p0, log_prob_val_p1 = None, None

            # 2. 각 if 블록 안에서 계산을 수행합니다.
            if phase0_mask.any():
                b_p0 = b_idx_loop[phase0_mask]
                main_tree_nodes = encoded_nodes[b_p0] * main_tree_mask[b_p0].unsqueeze(-1)
                context = main_tree_nodes.sum(1) / (main_tree_mask[b_p0].sum(1, keepdim=True) + 1e-9)
                mask_for_load_select = unconnected_loads_mask[b_p0]
                
                scores = self.decoder(context.unsqueeze(1), encoded_nodes[b_p0], mask_for_load_select.unsqueeze(1), 0)
                scores.masked_fill_(~mask_for_load_select, -1e9)
                
                log_prob = F.log_softmax(scores, dim=-1)
                selected_load = Categorical(probs=log_prob.exp()).sample() if decode_type == 'sampling' else log_prob.argmax(dim=-1)
                action_p0 = torch.stack([selected_load, torch.zeros_like(selected_load)], dim=1)
                log_prob_val_p0 = log_prob.gather(1, selected_load.unsqueeze(-1)).squeeze(-1)

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
                can_be_parent[torch.arange(len(b_p1)), child_indices] = False
                
                head_emb = encoded_nodes[b_p1, child_indices]
                context = torch.cat([context_embedding[b_p1], head_emb], dim=1)
                
                mask_for_parent_select = can_be_parent
                scores = self.decoder(context.unsqueeze(1), encoded_nodes[b_p1], mask_for_parent_select.unsqueeze(1), 1)
                scores.masked_fill_(~mask_for_parent_select, -1e9)
                
                log_prob = F.log_softmax(scores, dim=-1)
                selected_parent_idx = Categorical(probs=log_prob.exp()).sample() if decode_type == 'sampling' else log_prob.argmax(dim=-1)
                action_p1 = torch.stack([child_indices, selected_parent_idx], dim=1)
                log_prob_val_p1 = log_prob.gather(1, selected_parent_idx.unsqueeze(-1)).squeeze(-1)

            # 3. 계산이 끝난 후, 결과를 전체 텐서에 합칩니다.
            full_action = torch.zeros(expanded_batch_size, 2, dtype=torch.long, device=device)
            full_log_prob = torch.zeros(expanded_batch_size, device=device)
            
            if phase0_mask.any():
                full_action[phase0_mask] = action_p0
                full_log_prob[phase0_mask] = log_prob_val_p0
            if phase1_mask.any():
                full_action[phase1_mask] = action_p1
                full_log_prob[phase1_mask] = log_prob_val_p1

            # --- 상태 업데이트 ---
            if phase0_mask.any():
                selected_load = full_action[phase0_mask, 0]
                trajectory_head[phase0_mask] = selected_load.unsqueeze(-1)
                unconnected_loads_mask[phase0_mask, selected_load] = False
                decoding_phase[phase0_mask] = 1

            if phase1_mask.any():
                child_idx, parent_idx = full_action[phase1_mask, 0], full_action[phase1_mask, 1]
                adj_matrix[phase1_mask, parent_idx, child_idx] = True

                is_parent_in_main_tree = main_tree_mask[phase1_mask, parent_idx]
                
                b_merged = torch.where(phase1_mask)[0][is_parent_in_main_tree]
                if b_merged.numel() > 0:
                    decoding_phase[b_merged] = 0
                    
                    # 경로 전체를 메인 트리에 추가 (재귀적 탐색 대신 행렬 곱셈 사용)
                    path_to_merge = torch.zeros_like(main_tree_mask[b_merged])
                    path_to_merge[torch.arange(len(b_merged)), child_idx[is_parent_in_main_tree]] = True
                    adj_b = adj_matrix[b_merged]
                    for _ in range(num_nodes):
                         new_parents = (adj_b.float().transpose(1,2) @ path_to_merge.float().unsqueeze(-1)).squeeze(-1) > 0
                         if (new_parents & ~path_to_merge).sum() == 0: break
                         path_to_merge |= new_parents
                    main_tree_mask[b_merged] |= path_to_merge
                    
                b_unmerged = torch.where(phase1_mask)[0][~is_parent_in_main_tree]
                if b_unmerged.numel() > 0:
                    trajectory_head[b_unmerged] = parent_idx[~is_parent_in_main_tree].unsqueeze(-1)
            
            # 💡 --- 핵심 수정 끝 ---
            
            done = (unconnected_loads_mask.sum(dim=1) == 0).unsqueeze(-1)
            
            actions.append(full_action)
            log_probs.append(full_log_prob)
            
            child_emb = encoded_nodes[b_idx_loop, full_action[:, 0]]
            parent_emb = encoded_nodes[b_idx_loop, full_action[:, 1]]
            context_embedding = self.context_gru(torch.cat([child_emb, parent_emb], dim=1), context_embedding)
            
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