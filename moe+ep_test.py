import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):

    def __init__(self, hidden: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(hidden, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        weights, ids = torch.topk(torch.softmax(logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, ids

def expert_ffn(x, w13, w2):
    gate, up = (x @ w13.t()).chunk(2, dim=-1)
    return (F.silu(gate) * up) @ w2.t()

class EPMoeLayer(nn.Module):

    def __init__(self, hidden: int, inner: int, num_experts: int, top_k: int, world_size: int, rank: int):
        super().__init__()
        self.router = Router(hidden, num_experts, top_k)
        assert num_experts % world_size == 0
        self.W, self.E, self.rank = world_size, num_experts, rank
        self.epr = num_experts // world_size
        self.expert_start = self.epr * rank
        self.top_k = top_k
        self.w13 = nn.Parameter(torch.ones(self.epr, hidden, inner))
        self.w2 = nn.Parameter(torch.ones(self.epr, inner, hidden))

    def dispatch_local(self, x) -> dict:
        T = x.shape[0]
        weights, ids = self.router(x)
        flat_weights, flat_ids = weights.reshape(-1), ids.reshape(-1)
        tok = torch.arange(0, T).unsqueeze(dim=0).expand(T, self.top_k).reshpae(-1)
        dest = flat_ids // self.epr # 属于第几个rank
        bundles = {}
        for r in range(self.W):
            mask = dest == r
            if mask.any():
                bundles[r] = (x[tok[mask]], flat_ids[mask], tok[mask], flat_weights[mask])
        return bundles

    def compute_local(self, tokens, expert_ids):
        out = torch.empty_like(tokens)
        local = expert_ids - self.expert_start
        for e in range(self.epr):
            mask = local == e
            if mask.any():
                out[mask] = expert_ffn(tokens[mask], self.w13[e], self.w2[e])
        return out

    def combine_reduce(self, x, token_idx, weights, partial_rows):
        out = torch.zeros_like(x)
        out.index_add_(0, token_idx, partial_rows * weights.unsqueeze(1))
        return out

    def forward(self, x):
        assert torch.distributed.is_initialized(), "use torchrun --dist"
        d = torch.distributed

        bundles = self.dispatch_local(x)
        order = sorted(bundles.keys())
        send_counts = torch.tensor(
            [bundles[r][0].shape[0] for r in range(self.W)], dtype=torch.long
        )
        send_tokens = torch.cat([bundles[r][0] for r in order]) if bundles else x.new_empty(0, x.shape[1])
        send_experts = torch.cat([bundles[r][1] for r in order]) if bundles else x.new_empty(0, dtype=torch.long)
        send_toks = torch.cat([bundles[r][2] for r in order]) if bundles else x.new_empty(0)
        send_weights = torch.cat([bundles[r][3] for r in order]) if bundles else x.new_empty(0)

        recv_counts = torch.zeros_like(send_counts)
        d.all_to_all_single(
            recv_counts, send_counts
        )

        recv_tokens = torch.empty(
            (int(recv_counts.sum()), x.shape[1]), dtype=x.dtype, device=x.device
        )
        d.all_to_all_single(
            recv_tokens, send_tokens,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist()
        )

        meta = torch.stack([send_experts, send_toks, send_weights], dim=1)
        recv_meta = torch.empty(
            (recv_tokens.shape[0], 3), dtype=meta.dtype, device=meta.device
        )
        d.all_to_all_single(
            recv_meta, meta,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist()
        )

        # phase 2: compute
        rows = self.compute_local(recv_tokens, recv_meta[..., :0].long())

        # phase 3: send_back
        recv_back = torch.empty_like(
            int(send_counts.sum()), rows.shape[1], dtype=rows.dtype, device=rows.device
        )
        d.all_to_all_single(
            recv_back, rows,
            output_split_sizes=send_counts.tolist(),
            input_split_sizes=recv_counts.tolist()
        )
        return self.combine_reduce(x, send_toks, send_weights, recv_back)
