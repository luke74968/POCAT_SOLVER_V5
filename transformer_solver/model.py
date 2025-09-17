# transformer_solver/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensordict import TensorDict
from dataclasses import dataclass

from common.pocat_defs import FEATURE_DIM, FEATURE_INDEX, NODE_TYPE_BATTERY, NODE_TYPE_IC, NODE_TYPE_LOAD
from common.utils.common import batchify
from .pocat_env import PocatEnv, BATTERY_NODE_IDX

@dataclass
class PrecomputedCache:
    node_embeddings: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor
    def batchify(self, num_starts: int):
        return PrecomputedCache(
            batchify(self.node_embeddings, num_starts),
            batchify(self.glimpse_key, num_starts),
            batchify(self.glimpse_val, num_starts),
            batchify(self.logit_key, num_starts),
        )

# ... (Encoder, Decoder 등 나머지 클래스 정의는 변경 없음) ...
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
        if attention_mask.dim() == 3: attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 2: attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        score_scaled = score_scaled.masked_fill(attention_mask == 0, -1e12)

    if sparse_type == 'topk':
        seq_len = score_scaled.size(-1)
        k_for_topk = max(1, seq_len // 2)
        top_k_values, top_k_indices = torch.topk(score_scaled, k=k_for_topk, dim=-1)
        topk_mask = torch.zeros_like(score_scaled, dtype=torch.bool).scatter_(-1, top_k_indices, True)
        attention_weights = score_scaled.masked_fill(~topk_mask, -1e12)
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
        self.scalar_net = nn.Sequential(nn.Linear(4, embedding_dim // 2), nn.ReLU(), nn.Linear(embedding_dim // 2, embedding_dim // 2))
        self.matrix_net = nn.Sequential(nn.Linear(num_nodes * num_nodes, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim // 2))
        self.final_processor = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.ReLU())

    def forward(self, scalar_features: torch.Tensor, matrix_features: torch.Tensor) -> torch.Tensor:
        scalar_embedding = self.scalar_net(scalar_features)
        matrix_flat = matrix_features.view(matrix_features.shape[0], -1)
        matrix_embedding = self.matrix_net(matrix_flat)
        combined_embedding = torch.cat([scalar_embedding, matrix_embedding], dim=-1)
        final_prompt_embedding = self.final_processor(combined_embedding)
        return final_prompt_embedding.unsqueeze(1)

class PocatEncoder(nn.Module):
    def __init__(self, embedding_dim: int, encoder_layer_num: int = 6, **model_params):
        super().__init__()
        self.embedding_battery = nn.Linear(FEATURE_DIM, embedding_dim)
        self.embedding_ic = nn.Linear(FEATURE_DIM, embedding_dim)
        self.embedding_load = nn.Linear(FEATURE_DIM, embedding_dim)        
        sparse_params = model_params.copy(); sparse_params['use_sparse'] = True
        global_params = model_params.copy(); global_params['use_sparse'] = False
        self.sparse_layers = nn.ModuleList([EncoderLayer(embedding_dim=embedding_dim, **sparse_params) for _ in range(encoder_layer_num)])
        self.global_layers = nn.ModuleList([EncoderLayer(embedding_dim=embedding_dim, **global_params) for _ in range(encoder_layer_num)])
        self.sparse_fusion = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(encoder_layer_num)])
        self.global_fusion = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(encoder_layer_num - 1)])

    def forward(self, td: TensorDict, prompt_embedding: torch.Tensor) -> torch.Tensor:
        node_features = td['nodes']
        batch_size, num_nodes, embedding_dim = node_features.shape[0], node_features.shape[1], self.embedding_battery.out_features
        node_embeddings = torch.zeros(batch_size, num_nodes, embedding_dim, device=node_features.device)
        node_type_indices = node_features[..., FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(dim=-1)
        battery_mask, ic_mask, load_mask = (node_type_indices == NODE_TYPE_BATTERY), (node_type_indices == NODE_TYPE_IC), (node_type_indices == NODE_TYPE_LOAD)
        if battery_mask.any(): node_embeddings[battery_mask] = self.embedding_battery(node_features[battery_mask])
        if ic_mask.any(): node_embeddings[ic_mask] = self.embedding_ic(node_features[ic_mask])
        if load_mask.any(): node_embeddings[load_mask] = self.embedding_load(node_features[load_mask])
        connectivity_mask = td['connectivity_matrix']
        global_input = torch.cat((node_embeddings, prompt_embedding), dim=1)
        global_attention_mask = torch.ones(batch_size, num_nodes + 1, num_nodes + 1, dtype=torch.bool, device=node_embeddings.device)
        global_attention_mask[:, :num_nodes, :num_nodes] = connectivity_mask
        sparse_out, global_out = node_embeddings, global_input
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
        self.embedding_dim, self.head_num, self.qkv_dim = embedding_dim, head_num, qkv_dim
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_logit = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_context = nn.Linear(embedding_dim + 3, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

    def forward(self, td: TensorDict, cache: PrecomputedCache):
        avg_current = td["nodes"][:, :, FEATURE_INDEX["current_out"]].mean(dim=1, keepdim=True)
        unconnected_ratio = td["unconnected_loads_mask"].float().mean(dim=1, keepdim=True)
        step_ratio = td["step_count"].float() / (2 * td["nodes"].shape[1])
        state_features = torch.cat([avg_current, unconnected_ratio, step_ratio], dim=1)
        head_idx = td["trajectory_head"].squeeze(-1)
        head_emb = cache.node_embeddings[torch.arange(td.batch_size[0]), head_idx]
        query_input = torch.cat([head_emb, state_features], dim=1)
        q = reshape_by_heads(self.Wq_context(query_input.unsqueeze(1)), self.head_num)
        mha_out = multi_head_attention(q, cache.glimpse_key, cache.glimpse_val)
        mh_atten_out = self.multi_head_combine(mha_out)
        scores = torch.matmul(mh_atten_out, cache.logit_key).squeeze(1) / (self.embedding_dim ** 0.5)
        return scores

class PocatModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.prompt_net = PocatPromptNet(embedding_dim=model_params['embedding_dim'], num_nodes=model_params['num_nodes'])
        self.encoder = PocatEncoder(**model_params)
        self.decoder = PocatDecoder(**model_params)

    def forward(self, td: TensorDict, env: PocatEnv, decode_type: str = 'greedy', pbar: object = None,
                status_msg: str = "", log_fn=None, log_idx: int = 0, log_mode: str = 'progress'):
        
        base_desc = pbar.desc.split(' | ')[0] if pbar else ""
        if pbar:
            desc = f"{base_desc} | {status_msg} | ▶ Encoding (ing..)"
            pbar.set_description(desc)
            if log_fn and log_mode == 'detail': log_fn(desc)
        
        prompt_embedding = self.prompt_net(td["scalar_prompt_features"], td["matrix_prompt_features"])
        encoded_nodes = self.encoder(td, prompt_embedding)        

        glimpse_key = reshape_by_heads(self.decoder.Wk(encoded_nodes), self.decoder.head_num)
        glimpse_val = reshape_by_heads(self.decoder.Wv(encoded_nodes), self.decoder.head_num)
        logit_key = encoded_nodes.transpose(1, 2)
        cache = PrecomputedCache(encoded_nodes, glimpse_key, glimpse_val, logit_key)

        num_starts, start_nodes_idx = env.select_start_nodes(td)
        node_names = env.generator.config.node_names
        num_total_loads = env.generator.num_loads
        
        # 💡 **[수정]** POMO / Greedy 분기 처리
        batch_size = td.batch_size[0]
        if decode_type != 'greedy':
            td = batchify(td, num_starts)
            cache = cache.batchify(num_starts)
            action = start_nodes_idx.repeat(batch_size).unsqueeze(-1)
        else:
            action = start_nodes_idx[0].unsqueeze(0).unsqueeze(0)
        
        td.set("action", action)
        output_td = env.step(td)
        td = output_td["next"]

        log_probs, actions = [torch.zeros(td.batch_size[0], device=td.device)], [action]

        decoding_step = 0
        while not td["done"].all():
            decoding_step += 1
            scores = self.decoder(td, cache)
            
            # 💡 **[수정]** get_action_mask 호출 시 인자 전달
            mask = env.get_action_mask(td, log_mode=log_mode, log_idx=log_idx)
            
            if log_mode == 'detail' and log_fn:
                if log_idx >= td.batch_size[0]: log_idx = 0
                head_idx = td["trajectory_head"][log_idx].item()
                head_name = node_names[head_idx]
                log_fn(f"\n--- Step {decoding_step}: Head is at '{head_name}'. Action Type: [{'Select New Load' if head_idx == 0 else 'Find Parent'}]")
                valid_indices = torch.where(mask[log_idx])[0]
                log_fn(f"    - Valid actions before masking: {len(valid_indices)}")
                if len(valid_indices) > 0:
                    valid_scores = scores[log_idx, valid_indices]
                    top_k = min(5, len(valid_indices))
                    top_scores, top_indices_in_valid = torch.topk(valid_scores, k=top_k)
                    top_global_indices = valid_indices[top_indices_in_valid]
                    log_fn("    - Top 5 Action Scores (pre-mask):")
                    for i in range(top_k):
                        node_idx = top_global_indices[i].item()
                        score = top_scores[i].item()
                        log_fn(f"        - {node_names[node_idx]:<40s} | Score: {score:.4f}")
            
            elif log_mode == 'progress' and pbar:
                unconnected_loads = td['unconnected_loads_mask'][0].sum().item()
                connected_loads = num_total_loads - unconnected_loads
                progress_msg = f"Connecting Loads ({connected_loads}/{num_total_loads})"
                desc = f"{base_desc} | {status_msg} | {progress_msg}"
                pbar.set_description(desc)

            scores.masked_fill_(~mask, -float('inf'))
            log_prob = F.log_softmax(scores, dim=-1)
            probs = log_prob.exp()
            action = probs.argmax(dim=-1) if decode_type == 'greedy' else Categorical(probs=probs).sample()

            if log_mode == 'detail' and log_fn:
                action_idx_log = action[log_idx].item()
                action_prob = probs[log_idx, action_idx_log].item()
                log_fn(f"    - Action Selected: '{node_names[action_idx_log]}' (Prob: {action_prob:.2%})")
                log_fn("-" * 20)

            td.set("action", action.unsqueeze(-1))
            output_td = env.step(td)
            td = output_td["next"]
            actions.append(action.unsqueeze(-1))
            log_probs.append(log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1))

        return {
            "reward": output_td["reward"],
            "log_likelihood": torch.stack(log_probs, 1).sum(1),
            "actions": torch.cat(actions, 1) # 💡 **[수정]** torch.cat 사용
        }