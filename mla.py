import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import einsum, rearrange

def RoPE(theta, x):
    seq_len, d = x.shape[-2], x.shape[-1]
    freqs = theta ** (-torch.arange(0, d, 2) / d) * torch.arange(0, seq_len)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    x1, x2 = x[..., ::2], x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)

class MLA(nn.Module):

    def __init__(self, hidden_size, q_lora_rank, kv_lora_rank, num_heads, qk_rope_head_dim, qk_nope_head_dim, value_head_dim):
        super().__init__()
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.W_KVD = torch.ones((hidden_size, kv_lora_rank))
        self.W_QD = torch.ones((hidden_size, q_lora_rank))

        self.W_KU = torch.ones((kv_lora_rank, num_heads * qk_nope_head_dim))
        self.W_VU = torch.ones((kv_lora_rank, num_heads * value_head_dim))
        self.W_QU = torch.ones((q_lora_rank, num_heads * qk_nope_head_dim))

        self.W_K = torch.ones((hidden_size, qk_rope_head_dim))
        self.W_Q = torch.ones((q_lora_rank, num_heads * qk_rope_head_dim))

        self.rms_norm = nn.RMSNorm()

        self.WO = torch.ones((num_heads * value_head_dim, hidden_size))

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[-2]

        C = self.rms_norm(x @ self.W_KVD)
        QC = self.rms_norm(x @ self.W_QD)

        KR = (x @ self.W_K).squeeze(-3)
        QR = QC @ self.W_Q
        QR = rearrange("... T (H D) -> ... H T D", QR)

        KR = RoPE(10000.0, KR)
        QR = RoPE(10000.0, QR)

        KN = C @ self.W_KU
        KN = rearrange("... T (H D) -> ... H T D", KN)
        VN = C @ self.W_VU
        VN = rearrange("... T (H D) -> ... H T D", VN)
        QN = QC @ self.W_QU
        QN = rearrange("... T (H D) -> ... H T D", QN)

        scores = (QN @ KN.transpose(-2, -1) + QR @ KR.transpose(-2, -1)) / torch.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=True)
        scores = scores.masked_fill(mask, float("-inf"))
        scores = torch.softmax(scores, dim=-1)
        out = scores @ VN
        out = rearrange("... H T D -> ... T (H D)", out)
        out = out @ self.WO

        return out
