# transformer_solver/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple
from tensordict import TensorDict
from dataclasses import dataclass

from common.pocat_defs import FEATURE_DIM, FEATURE_INDEX, NODE_TYPE_BATTERY, NODE_TYPE_IC, NODE_TYPE_LOAD
from common.utils.common import batchify
from .pocat_env import PocatEnv


# 💡 [CaDA 장점 적용 1] PrecomputedCache 클래스 추가
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

# ... (RMSNorm, Normalization, ParallelGatedMLP, FeedForward, reshape_by_heads는 이전과 동일) ...
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

# 💡 수정: multi_head_attention이 sparse_type을 인자로 받도록 변경
def multi_head_attention(q, k, v, attention_mask=None, sparse_type=None):
    batch_s, head_num, n, key_dim = q.shape
    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / (key_dim ** 0.5)
    
    if attention_mask is not None:
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        score_scaled = score_scaled.masked_fill(attention_mask == 0, -1e9)


        
    if sparse_type == 'topk':
        # Top-K Sparse Attention
        # 어텐션 스코어가 높은 K개만 선택하여 마스크 생성
        # 💡 [핵심 변경] K 값을 시퀀스 길이의 절반으로 동적 계산
        #    k_top_k 파라미터를 제거하고, score_scaled의 마지막 차원 크기를 사용합니다.
        seq_len = score_scaled.size(-1)
        k_for_topk = max(1, seq_len // 2) # 최소 1개를 보장하면서 절반을 선택

        # 어텐션 스코어가 높은 K개만 선택하여 마스크 생성
        top_k_values, top_k_indices = torch.topk(score_scaled, k=k_for_topk, dim=-1)
        
        # 선택되지 않은 나머지 값들은 -inf로 마스킹
        topk_mask = torch.zeros_like(score_scaled, dtype=torch.bool).scatter_(-1, top_k_indices, True)
        attention_weights = score_scaled.masked_fill(~topk_mask, -1e9)
        weights = nn.Softmax(dim=3)(attention_weights)
    else:
        # Standard (Dense) Attention
        weights = nn.Softmax(dim=3)(score_scaled)
        
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    return out_transposed.contiguous().view(batch_s, n, head_num * key_dim)

# 💡 수정: EncoderLayer가 sparse_type을 인자로 받도록 변경
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
        # 1. 스칼라 제약조건(4개)을 위한 네트워크
        self.scalar_net = nn.Sequential(
            nn.Linear(4, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )
        
        # 2. 시퀀스 제약 행렬(num_nodes * num_nodes)을 위한 네트워크
        self.matrix_net = nn.Sequential(
            nn.Linear(num_nodes * num_nodes, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2)
        )
        
        # 3. 결합된 임베딩을 최종 처리하는 네트워크
        self.final_processor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), # (emb/2 + emb/2) -> emb
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )

    def forward(self, scalar_features: torch.Tensor, matrix_features: torch.Tensor) -> torch.Tensor:
        # 각 네트워크를 통과시켜 임베딩 생성
        scalar_embedding = self.scalar_net(scalar_features)
        
        # 행렬을 1차원으로 펼쳐서 입력
        batch_size = matrix_features.shape[0]
        matrix_flat = matrix_features.view(batch_size, -1)
        matrix_embedding = self.matrix_net(matrix_flat)
        
        # 두 임베딩을 연결(concatenate)
        combined_embedding = torch.cat([scalar_embedding, matrix_embedding], dim=-1)
        
        # 최종 프롬프트 임베딩 생성
        final_prompt_embedding = self.final_processor(combined_embedding)
        
        # (batch, 1, embedding_dim) 형태로 리턴
        return final_prompt_embedding.unsqueeze(1)


# 💡 수정: PocatEncoder를 CaDA와 같은 듀얼 어텐션 구조로 변경
class PocatEncoder(nn.Module):
    def __init__(self, embedding_dim: int, encoder_layer_num: int = 6, **model_params):
        super().__init__()
        self.embedding_layer = nn.Linear(FEATURE_DIM, embedding_dim)
        
        # Sparse 파라미터를 복사하여 수정
        sparse_params = model_params.copy(); sparse_params['use_sparse'] = True
        global_params = model_params.copy(); global_params['use_sparse'] = False
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
        sparse_out, global_out = node_embeddings, global_input

        
        for i in range(len(self.sparse_layers)):
            # 💡 [핵심 수정] Sparse Stream에 'connectivity_mask'를 전달합니다.
            sparse_out = self.sparse_layers[i](sparse_out, attention_mask=connectivity_mask)
            
            # Global Stream: 연결성 마스크 기반 어텐션
            global_out = self.global_layers[i](global_out, attention_mask=global_attention_mask)
            
            # Fusion
            sparse_out = sparse_out + self.sparse_fusion[i](global_out[:, :num_nodes])
            if i < len(self.global_layers) - 1:
                global_nodes = global_out[:, :num_nodes] + self.global_fusion[i](sparse_out)
                global_out = torch.cat((global_nodes, global_out[:, num_nodes:]), dim=1)  
        return global_out


# 💡 [CaDA 장점 적용 2] 디코더 로직 수정
class PocatDecoder(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim, **model_params):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim

        # 각 Phase에 맞는 Query 생성용 Linear 레이어
        self.Wq_load_select = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_parent_select = nn.Linear(embedding_dim * 2, head_num * qkv_dim, bias=False)

        # Key, Value, 최종 Logit Key 생성용 Linear 레이어
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_logit = nn.Linear(embedding_dim, embedding_dim, bias=False)
       
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

    def forward(self, context_embedding, cache: PrecomputedCache, mask, phase):
        # 1. Phase에 따라 Query 생성
        if phase == 0:
            q = reshape_by_heads(self.Wq_load_select(context_embedding), head_num=self.head_num)
        elif phase == 1:
            q = reshape_by_heads(self.Wq_parent_select(context_embedding), head_num=self.head_num)
        
        # 2. Multi-Head Attention 수행 (Key, Value는 캐시에서 가져옴)
        mha_out = multi_head_attention(q, cache.glimpse_key, cache.glimpse_val, attention_mask=mask)
        mh_atten_out = self.multi_head_combine(mha_out)
        
        # 3. 최종 Logits 계산 (Logit Key는 캐시에서 가져옴)
        scores = torch.matmul(mh_atten_out, cache.logit_key).squeeze(1) / (self.embedding_dim ** 0.5)
        
        return scores

class PocatModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.prompt_net = PocatPromptNet(embedding_dim=model_params['embedding_dim'], num_nodes=model_params['num_nodes'])
        self.encoder = PocatEncoder(**model_params)
        self.decoder = PocatDecoder(**model_params)
        # 💡 [CaDA 장점 적용 4] GRUCell 제거 (상태 기반 쿼리로 대체)
        self.context_gru = nn.GRUCell(model_params['embedding_dim'] * 2, model_params['embedding_dim'])


    def forward(self, td: TensorDict, env: PocatEnv, decode_type: str = 'greedy', pbar: object = None, status_msg: str = "", log_fn=None):
        base_desc = pbar.desc.split(' | ')[0] if pbar else ""
        
        if pbar:
            desc = f"{base_desc} | {status_msg} | ▶ Encoding (ing..)"
            pbar.set_description(desc)
            if log_fn: log_fn(desc)
        
        prompt_embedding = self.prompt_net(td["scalar_prompt_features"], td["matrix_prompt_features"])
        encoded_nodes = self.encoder(td, prompt_embedding)        

        num_starts, start_nodes_idx = env.select_start_nodes(td)
        node_names = env.generator.config.node_names
        num_total_loads = env.generator.num_loads
        
        batch_size = td.batch_size[0]
        td = batchify(td, num_starts)
        encoded_nodes = batchify(encoded_nodes, num_starts)
        
        expanded_batch_size = td.batch_size[0]
        context_embedding = encoded_nodes.mean(dim=1)
        log_probs, actions = [], []
        
        action_part1 = start_nodes_idx.repeat(batch_size)
        action_part2 = torch.zeros_like(action_part1)
        action = torch.stack([action_part1, action_part2], dim=1)
        
        td.set("action", action)
        output_td = env.step(td)
        td = output_td["next"]
        actions.append(action)
        log_probs.append(torch.zeros(expanded_batch_size, device=td.device))
        
        decoding_step = 0
        while not td["done"].all():
            decoding_step += 1
            num_connected_loads = num_total_loads - td["unconnected_loads_mask"][0].sum().item()
            phase = td["decoding_phase"][0, 0].item()

            state_description = ""
            if phase == 0:
                state_description = "Select New Load"
            else:
                current_node_idx = td["trajectory_head"][0].item()
                if current_node_idx != -1:
                    current_node_name = node_names[current_node_idx]
                    state_description = f"Find Parent for '{current_node_name}'"

            if pbar and log_fn:
                log_fn(f"{base_desc} | ... | Decoding (Step {decoding_step} State): {state_description}")

            mask = env.get_action_mask(td)

            if phase == 0:
                main_tree_nodes = encoded_nodes * td["main_tree_mask"].unsqueeze(-1)
                context = main_tree_nodes.sum(1) / (td["main_tree_mask"].sum(1, keepdim=True) + 1e-9)
                mask_for_load_select = mask[:, :, 0]
                scores = self.decoder(context.unsqueeze(1), encoded_nodes, mask_for_load_select, phase)
                scores.masked_fill_(~mask_for_load_select, -1e9)
                
                log_prob = F.log_softmax(scores, dim=-1)
                selected_load = Categorical(probs=log_prob.exp()).sample() if decode_type == 'sampling' else log_prob.argmax(dim=-1)
                action = torch.stack([selected_load, torch.zeros_like(selected_load)], dim=1)
                log_prob_val = log_prob.gather(1, selected_load.unsqueeze(-1)).squeeze(-1)

            elif phase == 1:
                trajectory_head_idx = td["trajectory_head"].squeeze(-1)
                head_emb = encoded_nodes[torch.arange(expanded_batch_size), trajectory_head_idx]
                context = torch.cat([context_embedding, head_emb], dim=1)
                
                mask_for_parent_select = mask[torch.arange(expanded_batch_size), :, trajectory_head_idx]
                scores = self.decoder(context.unsqueeze(1), encoded_nodes, mask_for_parent_select, phase)
                scores.masked_fill_(~mask_for_parent_select, -1e9)
                
                log_prob = F.log_softmax(scores, dim=-1)
                selected_parent_idx = Categorical(probs=log_prob.exp()).sample() if decode_type == 'sampling' else log_prob.argmax(dim=-1)
                action = torch.stack([trajectory_head_idx, selected_parent_idx], dim=1)
                log_prob_val = log_prob.gather(1, selected_parent_idx.unsqueeze(-1)).squeeze(-1)
            
            if pbar:
                child_idx = action[0, 0].item()
                child_name = node_names[child_idx]
                action_description = ""
                if phase == 0:
                    action_description = f"Started new path with Load '{child_name}'"
                else:
                    parent_idx = action[0, 1].item()
                    parent_name = node_names[parent_idx]
                    action_description = f"Connected '{child_name}' to '{parent_name}'"
                
                detail_msg = f"◀ Decoding ({num_connected_loads}/{num_total_loads} Loads, Step {decoding_step}): {action_description}"
                desc = f"{base_desc} | {status_msg} | ▶ Encoding (done) | {detail_msg}"
                pbar.set_description(desc)
                if log_fn: log_fn(desc)

            td.set("action", action)
            output_td = env.step(td)
            td = output_td["next"]
            
            actions.append(action)
            log_probs.append(log_prob_val)
            
            child_emb = encoded_nodes[torch.arange(expanded_batch_size), action[:, 0]]
            parent_emb = encoded_nodes[torch.arange(expanded_batch_size), action[:, 1]]
            context_embedding = self.context_gru(torch.cat([child_emb, parent_emb], dim=1), context_embedding)
            
        final_reward = output_td["reward"]

        return {
            "reward": final_reward,
            "log_likelihood": torch.stack(log_probs, 1).sum(1) if log_probs else torch.zeros(expanded_batch_size, device=td.device),
            "actions": torch.stack(actions, 1) if actions else torch.empty(expanded_batch_size, 0, 2, dtype=torch.long, device=td.device)
        }
