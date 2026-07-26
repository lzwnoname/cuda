import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner_(acc, m_i, l_i, q, K_block_ptr, V_block_ptr,
                     start_m, qk_scale,
                     BLOCK_M, BLOCK_N, HEAD_DIM,
                     offs_m, offs_n, N_CTX):
  lo, hi = 0, N_CTX
  # 调整KV的块指针, q的维度是(Z, H, BLOCK_M, HEAD_DIM)
  K_block_ptr = tl.advance(K_block_ptr, (0, lo))
  V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
  for start_n in range(lo, hi, BLOCK_N):
    # 编译优化
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # 分块进行矩阵乘法
    k = tl.load(K_block_ptr) # (Z, H, HEAD_DIM, BLOCK_N)
    v = tl.load(V_block_ptr) # (Z, H, BLOCK_N, HEAD_DIM)
    qk = tl.dot(q, k) # (Z, H, BLOCK_M, BLOCK_N)
    
    m_ij = tl.maximum(m_i, tl.max(qk, dim=1) * qk_scale) # 更新行最大值
    qk = qk * qk_scale - m_ij[:, None] # 广播机制集体减去行最大值
    p = tl.math.exp2(qk, dim=1)
    l_ij = tl.sum(p, dim=1)
    
    # 更新行和
    alpha = tl.math.exp2(m_ij - m_i)
    l_i = l_i * alpha + l_ij
    # 更新QK分数
    p = p.to(tl.float16)
    acc = acc * alpha[:, None]
    acc = tl.dot(p, v, acc) # (Z, H, BLOCK_M, HEAD_DIM)
    
    m_i = m_ij
    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
  return acc, m_i, l_i

@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd_(Q, K, V, sm_scale, M, O,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_om, stride_on,
              Z, H, N_CTX,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr
              ):
  tl.static_assert(HEAD_DIM <= BLOCK_N)
  start_m = tl.program_id(0) # seq_len 维度
  off_hz = tl.program_id(1) # heads * batch 维度
  off_z = off_hz // H
  off_h = off_hz % H
  # 找到QKV矩阵位置的偏移量
  qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
  # 组装块状指针
  Q_block_ptr = tl.make_block_ptr(
    base=Q + qkv_offset,
    shape=(N_CTX, HEAD_DIM),
    strides=(stride_qm, stride_qk),
    offsets=(start_m * BLOCK_M, 0),
    block_shape=(BLOCK_M, HEAD_DIM),
    order=(1, 0),
  )
  K_block_ptr = tl.make_block_ptr(
    base=K + qkv_offset,
    shape=(HEAD_DIM, N_CTX),
    strides=(stride_kk, stride_kn),
    offsets=(0, 0),
    block_shape=(HEAD_DIM, BLOCK_M),
    order=(0, 1),
  )
  V_block_ptr = tl.make_block_ptr(
    base=V + qkv_offset,
    shape=(N_CTX, HEAD_DIM),
    strides=(stride_vk, stride_vn),
    offsets=(0, 0),
    block_shape=(BLOCK_N, HEAD_DIM),
    order=(1, 0),
  )
  O_block_ptr = tl.make_block_ptr(
    base=O + qkv_offset,
    shape=(N_CTX, HEAD_DIM),
    strides=(stride_om, stride_on),
    offsets=(start_m * BLOCK_M, 0),
    block_shape=(BLOCK_N, HEAD_DIM),
    order=(1, 0),
  )
  # 计算块内位移m和n
  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  # 分配acc, m, l空间
  m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1
  acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
  # 计算qk_scale
  qk_scale = sm_scale
  qk_scale *= 1.44269504 # 1/log(2)
  q = tl.load(Q_block_ptr)
  # 计算attention
  
  acc, m_i, l_i = _attn_fwd_inner_(acc, m_i, l_i, q, K_block_ptr, V_block_ptr,
                                   start_m, qk_scale,
                                   BLOCK_M, BLOCK_N, HEAD_DIM,
                                   offs_m, offs_n, N_CTX)
  
  # 后处理数据
  m_i += tl.math.log2(l_i) # 计算原始softmax分母存入m供backward使用
  acc = acc / l_i[:, None]
  m_ptrs = M + off_hz * N_CTX + offs_m
  tl.store(m_ptrs, m_i)
  tl.store(O_block_ptr, acc.to(O.type.element_ty))
  

class _attention(torch.autograd.Function):
  
  @staticmethod
  def forward(ctx, q, k, v, casual, sm_scale):
    HEAD_DIM_Q, HEAD_DIM_KV = q.head[-1], k.head[-1]
    o = torch.empty_like(q)
    
    # grid 并行参数
    grid = lambda args: (triton.cdiv(q.shape[2], args['BLOCK_M']), q.shape[0] * q.shape[1], 1)
    M = torch.empty(q.shape[0], q.shape[1], q.shape[2], device=q.device, dtype=torch.float32)
    _attn_fwd_[grid](
      q, k, v, casual, sm_scale, M, o,
      q.stride(0), q.stride(1), q.stride(2), q.stride(3),
      k.stride(0), k.stride(1), k.stride(2), k.stride(3),
      v.stride(0), v.stride(1), v.stride(2), v.stride(3),
      q.shape[0], q.shape[1], N_CTX=q.shape[2], HEAD_DIM=HEAD_DIM_KV,
      **exta_kern_args)
    return o