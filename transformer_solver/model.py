# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class PocatPromptNet(nn.Module):
    def __init__(self, embedding_dim: int, prompt_feature_dim: int = 2, **kwargs):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(prompt_feature_dim, embedding_dim // 2), nn.ReLU(), nn.Linear(embedding_dim // 2, embedding_dim))
    def forward(self, prompt_features: torch.Tensor) -> torch.Tensor:
        return self.model(prompt_features).unsqueeze(1)

class PocatEncoder(nn.Module):
    def __init__(self, embedding_dim: int, encoder_layer_num: int = 6, **kwargs):
        super().__init__()
        self.embedding_layer = nn.Linear(FEATURE_DIM, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim=embedding_dim, **kwargs) for _ in range(encoder_layer_num)])
    def forward(self, node_features: torch.Tensor, prompt_embedding: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(node_features) + prompt_embedding
        for layer in self.layers: x = layer(x)
        return x

class PocatDecoder(nn.Module):
    def __init__(self, embedding_dim: int, head_num: int = 8, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.child_wq, self.child_wk = nn.Linear(embedding_dim, embedding_dim, bias=False), nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.parent_wq, self.parent_wk = nn.Linear(embedding_dim * 2, embedding_dim, bias=False), nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, encoded_nodes, context_embedding, mask):
        child_q = self.child_wq(context_embedding).unsqueeze(1)
        child_k = self.child_wk(encoded_nodes)
        child_scores = torch.matmul(child_q, child_k.transpose(1, 2)).squeeze(1) / (self.embedding_dim ** 0.5)
        child_scores[~mask.any(dim=2)] = -1e9
        child_log_probs = F.log_softmax(child_scores, dim=-1)
        selected_child_idx = child_log_probs.argmax(dim=-1)
        child_emb = encoded_nodes[torch.arange(encoded_nodes.shape[0]), selected_child_idx]
        parent_q_in = torch.cat([context_embedding, child_emb], dim=1)
        parent_q = self.parent_wq(parent_q_in).unsqueeze(1)
        parent_k = self.parent_wk(encoded_nodes)
        parent_scores = torch.matmul(parent_q, parent_k.transpose(1, 2)).squeeze(1) / (self.embedding_dim ** 0.5)
        parent_scores[~mask[torch.arange(mask.shape[0]), selected_child_idx]] = -1e9
        parent_log_probs = F.log_softmax(parent_scores, dim=-1)
        selected_parent_idx = parent_log_probs.argmax(dim=-1)
        action = torch.stack([selected_child_idx, selected_parent_idx], dim=1)
        child_prob = child_log_probs.gather(1, selected_child_idx.unsqueeze(-1)).squeeze(-1)
        parent_prob = parent_log_probs.gather(1, selected_parent_idx.unsqueeze(-1)).squeeze(-1)
        return child_prob + parent_prob, action

class PocatModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.prompt_net = PocatPromptNet(embedding_dim=embedding_dim)
        self.encoder = PocatEncoder(**model_params)
        self.decoder = PocatDecoder(**model_params)
        self.context_gru = nn.GRUCell(embedding_dim * 2, embedding_dim)

    def forward(self, td: TensorDict, env: PocatEnv):
        prompt_embedding = self.prompt_net(td["prompt_features"])
        encoded_nodes = self.encoder(td["nodes"], prompt_embedding)
        num_starts, start_nodes_idx = env.select_start_nodes(td)
        
        td = batchify(td, num_starts)
        encoded_nodes = batchify(encoded_nodes, num_starts)
        
        batch_size = td.batch_size[0]
        context_embedding = encoded_nodes.mean(dim=1)
        log_probs, actions = [], []
        
        # POMO First Step
        start_child_emb = encoded_nodes[torch.arange(batch_size), start_nodes_idx]
        parent_q_in = torch.cat([context_embedding, start_child_emb], dim=1)
        parent_q = self.decoder.parent_wq(parent_q_in).unsqueeze(1)
        parent_k = self.decoder.parent_wk(encoded_nodes)
        parent_scores = torch.matmul(parent_q, parent_k.transpose(1, 2)).squeeze(1) / (self.decoder.embedding_dim ** 0.5)
        
        mask = env.get_action_mask(td)
        parent_scores[~mask[torch.arange(batch_size), start_nodes_idx]] = -1e9
        parent_log_probs = F.log_softmax(parent_scores, dim=-1)
        selected_parent_idx = parent_log_probs.argmax(dim=-1)
        
        action = torch.stack([start_nodes_idx, selected_parent_idx], dim=1)
        td.set("action", action)
        
        # 💡 수정된 부분: step()의 전체 반환값을 받고, 'next' 키로 다음 상태를 업데이트
        output_td = env.step(td)
        td = output_td["next"]
        
        log_prob = parent_log_probs.gather(1, selected_parent_idx.unsqueeze(-1)).squeeze(-1)
        actions.append(action)
        log_probs.append(log_prob)
        
        parent_emb = encoded_nodes[torch.arange(batch_size), selected_parent_idx]
        context_embedding = self.context_gru(torch.cat([start_child_emb, parent_emb], dim=1), context_embedding)
        
        # Decoding Loop
        num_loads = env.generator.num_loads
        for _ in range(1, num_loads):
            if td["done"].all(): break
            log_prob, action = self.decoder(encoded_nodes, context_embedding, env.get_action_mask(td))
            td.set("action", action)
            output_td = env.step(td)
            td = output_td["next"]
            actions.append(action)
            log_probs.append(log_prob)
            
            child_emb = encoded_nodes[torch.arange(batch_size), action[:, 0]]
            parent_emb = encoded_nodes[torch.arange(batch_size), action[:, 1]]
            context_embedding = self.context_gru(torch.cat([child_emb, parent_emb], dim=1), context_embedding)

        # 💡 수정된 부분: 최종 상태의 보상이 아닌, 마지막 step에서 반환된 보상을 사용
        final_reward = output_td["reward"]

        return {
            "reward": final_reward,
            "log_likelihood": torch.stack(log_probs, 1).sum(1) if log_probs else torch.zeros(batch_size, device=td.device),
            "actions": torch.stack(actions, 1) if actions else torch.empty(batch_size, 0, 2, dtype=torch.long, device=td.device)
        }