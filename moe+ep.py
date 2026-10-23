"""
A minimal, educational MoE + Expert-Parallelism (EP) implementation in pure PyTorch.

It mirrors the production pipeline discussed for SGLang:

    router (topk)                     # gating: logits -> top-k (expert_id, weight)
      -> dispatch (all_to_all)        # EP: send each (token, expert) copy to the
                                     #     rank owning that expert
      -> local experts                # each rank holds only E/W experts
      -> combine (all_to_all)         # send partial outputs back to the home rank
      -> weighted sum over top-k      # y = sum_k w_k * FFN_e_k(x)

Files map to production concepts:
    dispatch/combine here <-> DeepEP / token_dispatcher in SGLang
    compute_local here     <-> fused_moe runner (one grouped GEMM per expert)

Two run modes:
  1. Simulation (default, single process, CPU, no torch.distributed):
         virtual ranks are separate weight shards; the driver hand-routes
         (token, expert) copies between shards. Verifies the index algebra.
  2. Distributed: `torchrun --nproc_per_node=W simple_ep_moe.py --dist`
         real all_to_all over gloo (CPU) — swap backend for nccl on GPUs.

Correctness check: EPMoE output must equal a NaiveMoE holding ALL experts.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Router(nn.Module):
    """Gating network: hidden -> top-k (expert_id, weight). Replicated on every rank."""

    def __init__(self, hidden: int, num_experts: int, top_k: int):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)                                   # (T, E)
        weights, ids = torch.topk(torch.softmax(logits, -1), self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)       # renormalize
        return ids, weights                                     # (T, k) each


def expert_ffn(x, w13, w2):
    """One expert's SwiGLU FFN. x: (n, D); w13: (2I, D); w2: (D, I)."""
    gate, up = (x @ w13.t()).chunk(2, dim=-1)
    return (F.silu(gate) * up) @ w2.t()                         # (n, D)


# ---------------------------------------------------------------------------
# Reference: naive MoE, ALL experts local, loop over experts
# ---------------------------------------------------------------------------


class NaiveMoE(nn.Module):
    def __init__(self, hidden, inter, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.router = Router(hidden, num_experts, top_k)
        self.w13 = nn.Parameter(torch.randn(num_experts, 2 * inter, hidden) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, hidden, inter) * 0.02)

    def forward(self, x):
        T = x.shape[0]
        ids, weights = self.router(x)                           # (T, k)
        flat_ids = ids.reshape(-1)                              # (T*k,)
        out = torch.zeros_like(x)
        for e in range(self.w13.shape[0]):                      # one GEMM per expert
            pos = torch.nonzero(flat_ids == e).squeeze(1)       # flat positions (t*k+j)
            if pos.numel() == 0:
                continue
            tok = pos // self.top_k
            w = weights.reshape(-1)[pos].unsqueeze(1)           # (n, 1)
            out.index_add_(0, tok, expert_ffn(x[tok], self.w13[e], self.w2[e]) * w)
        return out


# ---------------------------------------------------------------------------
# EP MoE layer: rank r owns experts [r*E/W, (r+1)*E/W)
# ---------------------------------------------------------------------------


class EPMoELayer(nn.Module):
    """
    The layer is split into the three phases of the EP pipeline. In
    distributed mode `forward()` wires them together with real all_to_all;
    in simulation mode the driver below routes copies between shards by hand.
    """

    def __init__(self, hidden, inter, num_experts, top_k, rank, world_size):
        super().__init__()
        assert num_experts % world_size == 0
        self.E, self.W, self.rank = num_experts, world_size, rank
        self.epr = num_experts // world_size                     # experts per rank
        self.top_k = top_k
        self.router = Router(hidden, num_experts, top_k)         # replicated
        self.expert_start = rank * self.epr
        # only MY experts' weights live on this rank
        self.w13 = nn.Parameter(torch.randn(self.epr, 2 * inter, hidden) * 0.02)
        self.w2 = nn.Parameter(torch.randn(self.epr, hidden, inter) * 0.02)

    # ---- phase 1: DISPATCH (local part; needs no communication) -----------

    def dispatch_local(self, x):
        """Route my tokens; return per-destination-rank bundles of
        (token_rows, expert_ids, token_idx, weights)."""
        T = x.shape[0]
        ids, weights = self.router(x)                            # (T, k)
        flat_ids = ids.reshape(-1)                               # (T*k,)
        flat_tok = (
            torch.arange(T, device=x.device).unsqueeze(1)
            .expand(T, self.top_k).reshape(-1)
        )                                                        # (T*k,)
        flat_w = weights.reshape(-1)
        dest = flat_ids // self.epr                              # owner rank per copy
        bundles = {}
        for r in range(self.W):
            sel = dest == r
            if sel.any():
                bundles[r] = (x[flat_tok[sel]], flat_ids[sel], flat_tok[sel], flat_w[sel])
        return bundles

    # ---- phase 2: LOCAL EXPERTS (runs on the owner rank) ------------------

    def compute_local(self, tokens, expert_ids):
        """tokens: (n, D) whose expert_ids (n,) all belong to my shard.
        Group by local expert, one GEMM each."""
        out = torch.empty_like(tokens)
        local = expert_ids - self.expert_start
        for e in range(self.epr):
            mask = local == e
            if mask.any():
                out[mask] = expert_ffn(tokens[mask], self.w13[e], self.w2[e])
        return out                                               # (n, D)

    # ---- phase 3: COMBINE + weighted sum (runs on the home rank) ----------

    def combine_reduce(self, x, token_idx, weights, partial_rows):
        """Scatter partial expert outputs back to their tokens and do the
        top-k weighted sum:  y_t = sum_j w_tj * partial_tj."""
        out = torch.zeros_like(x)
        out.index_add_(0, token_idx, partial_rows * weights.unsqueeze(1))
        return out

    # ---- full pipeline with real communication -----------------------------

    def forward(self, x):
        assert torch.distributed.is_initialized(), "use torchrun --dist"
        d = torch.distributed

        bundles = self.dispatch_local(x)
        order = sorted(bundles.keys())
        send_counts = torch.tensor(
            [bundles[r][0].shape[0] for r in range(self.W)], dtype=torch.long
        )
        send_tokens = torch.cat([bundles[r][0] for r in order]) if bundles else x.new_empty(0, x.shape[1])
        send_expert = torch.cat([bundles[r][1] for r in order]) if bundles else x.new_empty(0, dtype=torch.long)
        send_tok = torch.cat([bundles[r][2] for r in order]) if bundles else x.new_empty(0, dtype=torch.long)
        send_w = torch.cat([bundles[r][3] for r in order]) if bundles else x.new_empty(0)

        # exchange row counts with every rank (W-sized all_to_all)
        recv_counts = torch.zeros_like(send_counts)
        d.all_to_all_single(recv_counts, send_counts)

        recv = torch.empty(
            (int(recv_counts.sum()), x.shape[1]), dtype=x.dtype, device=x.device
        )
        d.all_to_all_single(
            recv, send_tokens,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
        )
        # small tensors (expert id / token idx / router weight): ship as one
        # float matrix to keep the example to a single all_to_all per payload
        meta = torch.stack([send_expert.float(), send_tok.float(), send_w], dim=1)
        recv_meta = torch.empty((recv.shape[0], 3), dtype=meta.dtype, device=x.device)
        d.all_to_all_single(
            recv_meta, meta,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
        )

        # phase 2 on received rows (all belong to my shard by construction)
        rows = self.compute_local(recv, recv_meta[:, 0].long())

        # phase 3: send rows back to their source ranks (reverse the split).
        # all_to_all preserves per-peer block order, so the rows I get back
        # align 1:1 with the copies I sent — the same "src2dst" invariant
        # DeepEP's normal-mode combine relies on. No need to ship indices.
        recv_back = torch.empty(
            (int(send_counts.sum()), rows.shape[1]), dtype=rows.dtype, device=rows.device
        )
        d.all_to_all_single(
            recv_back, rows,
            output_split_sizes=send_counts.tolist(),           # what I sent out
            input_split_sizes=recv_counts.tolist(),            # what I received
        )
        return self.combine_reduce(x, send_tok, send_w, recv_back)


# ---------------------------------------------------------------------------
# Simulation driver: orchestrate the virtual EP by hand (no dist, one process)
# ---------------------------------------------------------------------------


def simulate_ep_forward(shards, xs):
    """Each shards[r] owns tokens xs[r]. Hand-deliver dispatch/combine: this is
    exactly what the all_to_alls do in distributed mode, spelled out in Python."""
    outs = [torch.zeros_like(x) for x in xs]

    for src in range(len(shards)):                       # home rank of the tokens
        shard_src, x = shards[src], xs[src]
        bundles = shard_src.dispatch_local(x)            # dest rank -> bundle

        for dst, (tokens, expert_ids, tok_idx, weights) in bundles.items():
            rows = shards[dst].compute_local(tokens, expert_ids)   # ... computes
            # ... and combine returns the rows to the home rank, which does
            # the top-k weighted sum
            outs[src].index_add_(0, tok_idx, rows * weights.unsqueeze(1))
    return outs


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def run_simulation(W=2, T=12, D=32, I=16, E=8, k=2, seed=0):
    torch.manual_seed(seed)
    shards, xs = [], []
    for r in range(W):
        torch.manual_seed(seed)                          # same router on all ranks
        shards.append(EPMoELayer(D, I, E, k, rank=r, world_size=W))
        xs.append(torch.randn(T, D))

    outs = simulate_ep_forward(shards, xs)

    # reference: full-expert NaiveMoE assembled from the shards' weights
    ref = NaiveMoE(D, I, E, k)
    ref.router.load_state_dict(shards[0].router.state_dict())
    with torch.no_grad():
        ref.w13.copy_(torch.cat([s.w13.data for s in shards]))
        ref.w2.copy_(torch.cat([s.w2.data for s in shards]))
    for r in range(W):
        expected = ref(xs[r])
        err = (outs[r] - expected).abs().max().item()
        print(f"[sim] rank {r}: max abs err vs naive MoE = {err:.3e}")
        assert err < 1e-4, "EP simulation mismatch!"
    print("simulation: PASS")


def run_distributed(W, T, D, I, E, k, seed=0):
    torch.distributed.init_process_group(backend="gloo")  # nccl on GPUs
    d = torch.distributed
    rank = d.get_rank()
    torch.manual_seed(seed)
    shard = EPMoELayer(D, I, E, k, rank=rank, world_size=W)
    x = torch.randn(T, D)
    y = shard(x)

    # correctness: all_gather everyone's expert shards, rebuild the full
    # expert table, and compare against a single-device NaiveMoE
    w13_all = [torch.empty_like(shard.w13.data) for _ in range(W)]
    w2_all = [torch.empty_like(shard.w2.data) for _ in range(W)]
    d.all_gather(w13_all, shard.w13.data)
    d.all_gather(w2_all, shard.w2.data)
    ref = NaiveMoE(D, I, E, k)
    ref.router.load_state_dict(shard.router.state_dict())
    with torch.no_grad():
        ref.w13.copy_(torch.cat(w13_all))
        ref.w2.copy_(torch.cat(w2_all))
    err = (y - ref(x)).abs().max().item()
    print(f"[dist] rank {rank}: shape={tuple(y.shape)} max err vs naive MoE = {err:.3e}")
    assert err < 1e-4, "distributed EP mismatch!"
    if rank == 0:
        print("distributed: PASS")
    d.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dist", action="store_true", help="run with torchrun")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--D", type=int, default=32)
    p.add_argument("--I", type=int, default=16)
    p.add_argument("--E", type=int, default=8)
    p.add_argument("--k", type=int, default=2)
    a = p.parse_args()
    if a.dist:
        run_distributed(a.world_size, a.T, a.D, a.I, a.E, a.k)
    else:
        run_simulation(a.world_size, a.T, a.D, a.I, a.E, a.k)
