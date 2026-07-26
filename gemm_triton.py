import torch
import triton
import triton.language as tl


@triton.jit
def gemm(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_ar, stride_ac,
    stride_br, stride_bc,
    stride_cr, stride_cc,
    alpha, beta,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)

    a_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_ar, stride_ac),
        offsets=(pid_m * BM, 0),
        block_shape=(BM, BK),
        order=(1, 0)
    )
    b_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_br, stride_bc),
        offsets=(0, pid_n * BN),
        block_shape=(BK, BN),
        order=(1, 0)
    )
    
    block_c_ptr = c_ptr + (rm[:, None] * stride_cr + rn[None, :] * stride_cc)
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)

    block_c = tl.load(block_c_ptr, mask=mask_c).to(tl.float32) * beta

    for k in range(0, K, BK):
        block_a = tl.load(a_ptr, boundary_check=(0, 1))
        block_b = tl.load(b_ptr, boundary_check=(0, 1))

        block_c += alpha * tl.dot(block_a, block_b)

        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

    tl.store(block_c_ptr, block_c.to(tl.float16), mask=mask_c)

# a, b, c are tensors on the GPU
def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    BM, BN, BK = 128, 128, 16
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    gemm[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        alpha, beta,
        BM, BN, BK
    )
