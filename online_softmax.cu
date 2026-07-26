
void warpReduceOnLine(float& m, float& d) {
  for (int offset = 16 ; offset ; offset >>= 1) {
    float m2 = __shfl_down_sync(0xffffffff, m, offset);
    float d2 = __shfl_down_sync(0xffffffff, d2, offset);

    float m_new = fmaxf(m, m2);
    d = d * expf(m - m_new) + expf(m - m2)
    m = m_new
  }
}

__global__ void online_softmax(float *input, float *output, int M, int N) {
  int tid = threadIdx.x;
  int lane = tid % 32;
  int wid = tid / 32;

  for (int row = blockIdx.x ; row < M ; row += gridDim.x) {
    float *x = input + row * N;
    float *y = output + row * N;

    int N4 = N / 4;

    float4 *x4 = reinterpret_cast<float4 *>(x);
    float4 *y4 = reinterpret_cast<float4 *>(y);

    float local_m = -INFINITY;
    float local_d = 0.0f;

    // 递推规约，对每个线程规约它管的范围
    for (int i = tid ; i < N ; i += blockDim.x) {
      float4 data = x4[i];
      float vals[4] = {data.x, data.y, data.z, data.w};
      for (int j = 0 ; j < 4 ; j++) {
        float m_new = fmaxf(local_m, vals[j]);
        local_d = local_d * expf(local_m - m_new) + expf(vals[j] - m_new);
        local_m = m_new;
      }
    }
    // 处理不被4整除的部分
    for (int i = tid + N4 * 4 ; i < N ; i += blockDim.x) {
      float xi = x[i];
      float m_new = fmaxf(local_m, xi);
      local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
      local_m = m_new;
    }
    // 二级规约，先规约到warp0，然后warp0内warp shuffle将结果规约到wid = 0, tid = 0
    warpReduceOnLine(local_m, local_d);

    __shared__ float warp_m[32];
    __shared__ float warp_d[32];

    if (lane == 0) {
      warp_m[wid] = local_m;
      warp_d[wid] = local_d;
    }

    __syncthreads();

    int num_warps = blockIdx.x / 32;
    if (wid == 0) {
      local_m = (lane < num_warps) ? warp_m[lane] : -INFINITY;
      local_d = (lane < num_warps) ? warp_d[lane] : 0.0f;
      warpReduceOnLine(local_m, local_d);
    }

    __shared__ float final_m;
    __shared__ float final_d;

    if (tid == 0) {
      final_m = local_m;
      final_d = local_d;
    }

    __syncthreads();

    float row_max = final_m;
    float inv_sum = 1.0f / final_d;

    for (int i = tid ; i < N ; i += blockDim.x) {
      float4 data = x4[i];
      float4 result;
      result.x = expf(data.x - row_max) * inv_sum;
      result.y = expf(data.y - row_max) * inv_sum;
      result.w = expf(data.w - row_max) * inv_sum;
      result.z = expf(data.z - row_max) * inv_sum;
      y4[i] = result;
    }

    for (int i = tid + N4 ; i < N ; i += blockDim.x) {
      y[i] = expf(x[i] - row_max) * inv_sum;
    }

    __syncthreads();
  }

}