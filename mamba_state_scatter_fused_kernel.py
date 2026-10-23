import torch
import triton
import triton.language as tl

@triton.jit
def _fused_mamba_state_scatter_multi_kernel(
    src_ptr,
    dst_ptr,
    dst_indices1_raw_ptr,
    step_indices1_raw_ptr,
    dst_indices2_raw_ptr,
    step_indices2_raw_ptr,
    n1,
    elem_per_entry: tl.constexpr,
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    dst_layer_stride,
    dst_req_stride,
    dst_step_stride,
    src_req_size,
    src_step_size,
    dst_req_size,
    BLOCK_SIZE: tl.constexpr,
    ITERS: tl.constexpr
):
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1).to(tl.int64)
    pid_block = tl.program_id(2).to(tl.int64)

    is2 = pid_req >= n1
    off = tl.where(is2, pid_req - n1, pid_req)
    step_idx = tl.where(
        is2,
        tl.load(step_indices2_raw_ptr + off, mask=is2, other=-1),
        tl.load(step_indices1_raw_ptr + off, mask=not is2, other=-1)
    ).to(tl.int64)
    if step_idx < 0:
        return

    dst_idx = tl.where(
        is2,
        tl.load(dst_indices2_raw_ptr + off, mask=is2, other=-1),
        tl.load(dst_indices1_raw_ptr + off, mask=not is2, other=-1)
    ).to(tl.int64)
    src_idx = off

    if not (
        (src_idx < src_req_size)
        & (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (step_idx < src_step_size)
    ):
        return

    src_offset = (
        pid_layer * src_layer_stride
        + src_idx * src_req_stride
        + step_idx * src_step_stride
    )

    dst_offset = pid_layer * dst_layer_stride + dst_idx * dst_req_stride

    BASE = pid_block * (BLOCK_SIZE * ITERS)

    for it in tl.range(ITERS):
        offsets = BASE + tl.range(0, BLOCK_SIZE)
        mask = offsets < elem_per_entry
        data = tl.load(src_ptr + src_offset + offsets, mask=mask)
        tl.store(dst_ptr + dst_offset + offsets, mask=mask)
