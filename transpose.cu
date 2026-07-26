#include <iostream>
#include <algorithm>
#include <string>
#include <omp.h>
#include <cuda_runtime.h>
#include <chrono>

template <int BLOCK_SZ, int NUM_PER_THREAD>
__global__ void transpose(float *in, float *out, int M, int N) {
  __shared__ float smem[BLOCK_SZ][BLOCK_SZ + 1];
  int x = blockIdx.x * BLOCK_SZ + threadIDx.x;
  int y = blockIdx.y * BLOCK_SZ + threadIDx.y;

  for (int i = 0 ; i < NUM_PER_THREAD; i++)
    if (y + i < M && x < N) {
      smem[threadIdx.y + i][threadIdx.x] = in[(y + i) * N + x];
    }
  __syncthreads();

  x = blockIdx.y * BLOCK_SZ + threadIdx.x;
  y = blockIdx.x * BLOCK_SZ + threadIdx.y;
  // 这里要让smem的索引是 threadIdx.x, threadIdx.y 对应的就是转置后分块矩阵的 (threadIdx.x, threadIdx.y) 这部分位置
  for (int i = 0 ; i < NUM_PER_THREAD; i++)
    if (y + i < N && x < M) {
      out[(y + i) * M + x] = smem[threadIdx.x][threadIdx.y + i];
    }
  return ;
}