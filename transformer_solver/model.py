# transformer_solver/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical # 💡 확률적 샘플링을 위해 추가
from typing import Tuple
from tensordict import TensorDict

from common.pocat_defs import FEATURE_DIM
from common.utils.common import batchify
from .pocat_env import PocatEnv


# ... (RMSNorm, Normalization, EncoderLayer 등 다른 클래스는 이전과 동일) ...
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

def multi_head_attention(q, k, v, ninf_mask=None):
    batch_s, head_num, n, key_dim = q.shape
    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / (key_dim ** 0.5)
    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, None, :, :].expand(batch_s, head_num, n, k.size(2))
    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    return out_transposed.contiguous().view(batch_s, n, head_num * key_dim)

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim, ffd='siglu', **model_params):
        super().__init__()
        self.embedding_dim, self.head_num, self.qkv_dim = embedding_dim, head_num, qkv_dim
        self.Wq, self.Wk, self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False), nn.Linear(embedding_dim, head_num * qkv_dim, bias=False), nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.normalization1 = Normalization(embedding_dim, **model_params)
        if ffd == 'siglu': self.feed_forward = ParallelGatedMLP(hidden_size=embedding_dim, **model_params)
        else: self.feed_forward = FeedForward(embedding_dim=embedding_dim, **model_params)
        self.normalization2 = Normalization(embedding_dim, **model_params)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = reshape_by_heads(self.Wq(x), self.head_num), reshape_by_heads(self.Wk(x), self.head_num), reshape_by_heads(self.Wv(x), self.head_num)
        mha_out = self.multi_head_combine(multi_head_attention(q, k, v))
        h = self.normalization1(x + mha_out)
        return self.normalization2(h + self.feed_forward(h))


# 💡 수정: PocatPromptNet 클래스를 새 구조로 변경
# 💡 수정: PocatPromptNet 클래스를 새 구조로 변경
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


# 💡 수정: PocatEncoder 클래스의 forward 함수 수정
class PocatEncoder(nn.Module):
    def __init__(self, embedding_dim: int, encoder_layer_num: int = 6, **kwargs):
        super().__init__()
        self.embedding_layer = nn.Linear(FEATURE_DIM, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim=embedding_dim, **kwargs) for _ in range(encoder_layer_num)])

    def forward(self, node_features: torch.Tensor, prompt_embedding: torch.Tensor) -> torch.Tensor:
        # 1. 노드 피처를 초기 임베딩으로 변환
        node_embeddings = self.embedding_layer(node_features)
        
        # 2. 노드 임베딩과 프롬프트 임베딩을 연결(concatenate)
        concatenated_embeddings = torch.cat((node_embeddings, prompt_embedding), dim=1)
        
        # 3. 인코더 레이어 통과
        x = concatenated_embeddings
        for layer in self.layers:
            x = layer(x)
            
        # 4. 프롬프트 부분을 제외한 노드 임베딩만 반환
        num_nodes = node_features.shape[1]
        return x[:, :num_nodes, :]

class PocatDecoder(nn.Module):
    def __init__(self, embedding_dim: int, head_num: int = 8, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.parent_wq, self.parent_wk = nn.Linear(embedding_dim * 2, embedding_dim, bias=False), nn.Linear(embedding_dim, embedding_dim, bias=False)


class PocatModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        # 💡 수정: PocatPromptNet 초기화 시 num_nodes 정보 전달
        #    (run.py에서 env.generator.num_nodes를 통해 전달해야 함)
        self.prompt_net = PocatPromptNet(embedding_dim=embedding_dim, num_nodes=model_params['num_nodes'])
        self.encoder = PocatEncoder(**model_params)
        self.decoder = PocatDecoder(**model_params)
        self.context_gru = nn.GRUCell(embedding_dim * 2, embedding_dim)

        self.load_select_wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.load_select_wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
    # --- 👇 [핵심] log_fn 인자 추가 ---
    def forward(self, td: TensorDict, env: PocatEnv, decode_type: str = 'greedy', pbar: object = None, status_msg: str = "", log_fn=None):
        base_desc = pbar.desc.split(' | ')[0] if pbar else ""
        
        if pbar:
            desc = f"{base_desc} | {status_msg} | ▶ Encoding (ing..)"
            pbar.set_description(desc)
            if log_fn: log_fn(desc)
        
        # 💡 수정: 새로운 프롬프트 넷에 스칼라와 행렬 피처를 전달

        prompt_embedding = self.prompt_net(td["scalar_prompt_features"], td["matrix_prompt_features"])
        encoded_nodes = self.encoder(td["nodes"], prompt_embedding)
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
            if pbar:
                num_connected_loads = num_total_loads - td["unconnected_loads_mask"][0].sum().item()
                current_node_idx = td["trajectory_head"][0].item()
                current_node_name = node_names[current_node_idx] if current_node_idx != -1 else "N/A"

                detail_msg = f"◀ Decoding ({num_connected_loads}/{num_total_loads} Loads, Step {decoding_step}: Conn. '{current_node_name}')"
                desc = f"{base_desc} | {status_msg} | ▶ Encoding (done) | {detail_msg}"
                pbar.set_description(desc)
                if log_fn: log_fn(desc)

            mask = env.get_action_mask(td)
            phase = td["decoding_phase"][0, 0].item()

            if phase == 0:
                main_tree_nodes = encoded_nodes * td["main_tree_mask"].unsqueeze(-1)
                context_for_load_select = main_tree_nodes.sum(1) / (td["main_tree_mask"].sum(1, keepdim=True) + 1e-9)
                
                q = self.load_select_wq(context_for_load_select).unsqueeze(1)
                k = self.load_select_wk(encoded_nodes)
                scores = torch.matmul(q, k.transpose(1, 2)).squeeze(1) / (self.decoder.embedding_dim ** 0.5)
                mask_for_load_select = mask[:, :, 0]
                scores = scores.masked_fill(~mask_for_load_select, -1e9)
                
                log_prob = F.log_softmax(scores, dim=-1)
                
                if decode_type == 'sampling':
                    selected_load = Categorical(probs=log_prob.exp()).sample()
                else:
                    selected_load = log_prob.argmax(dim=-1)
                
                action = torch.stack([selected_load, torch.zeros_like(selected_load)], dim=1)
                log_prob_val = log_prob.gather(1, selected_load.unsqueeze(-1)).squeeze(-1)

            elif phase == 1:
                trajectory_head_idx = td["trajectory_head"].squeeze(-1)
                head_emb = encoded_nodes[torch.arange(expanded_batch_size), trajectory_head_idx]
                
                parent_q_in = torch.cat([context_embedding, head_emb], dim=1)
                parent_q = self.decoder.parent_wq(parent_q_in).unsqueeze(1)
                parent_k = self.decoder.parent_wk(encoded_nodes)
                parent_scores = torch.matmul(parent_q, parent_k.transpose(1, 2)).squeeze(1) / (self.decoder.embedding_dim ** 0.5)
                
                mask_for_parent_select = mask[torch.arange(expanded_batch_size), :, trajectory_head_idx]
                parent_scores.masked_fill_(~mask_for_parent_select, -1e9)
                
                parent_log_probs = F.log_softmax(parent_scores, dim=-1)

                if decode_type == 'sampling':
                    selected_parent_idx = Categorical(probs=parent_log_probs.exp()).sample()
                else:
                    selected_parent_idx = parent_log_probs.argmax(dim=-1)
                
                action = torch.stack([trajectory_head_idx, selected_parent_idx], dim=1)
                log_prob_val = parent_log_probs.gather(1, selected_parent_idx.unsqueeze(-1)).squeeze(-1)

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