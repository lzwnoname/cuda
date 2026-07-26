#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>

void warpReduce(float val) {
  for (int offeset = 16 ; offset ; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
}

// Grid Stride Loop: 固定grid大小，每个线程循环处理多个数据，Grid大小为SM * 4
__global__ void reduce_v7(float* input, float* output, int n) {
  int tid = threadIdx.x;
  int lane = tid % 32;
  int wid = tid / 32;

  float4 input4 = reinterpert_cast<float4 *>(input);
  int n4 = n / 4;

  for (int idx = blockIdx.x * blockDim.x + tid ;
      idx < n4 ;
      idx += gridDim.x * blockDim.x)
  {
    float4 data = input4[idx];
    val += data.x + data.y + data.z + data.w;
  }

  int tail_start = n4 * 4;
  for (int idx = tail_start + blockIdx.x * blockDim.x + tid ;
      idx < n ;
      idx++)
  {
    val += input[idx];
  }

  warpReduce(val);

  __shared__ float warp_results[32];
  if (lane == 0) {
    warp_results[wid] = val;
  }

  __syncthreads();

  // 每个warp的结果汇总到block 0的warp 0
  int num_warps = blockDim.Dim.x / 32;
  if (wid == 0) {
    val = (lane < num_warps) ? warp_results[lane] : 0.0f;
    val = warpReduce(val);
  }

  if (tid == 0) output[blockIdx.x] = val;
}

int main() {
  return 0;
}