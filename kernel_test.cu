#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ void warpReduce(float &val) {
  for (int offset = 16 ; offset ; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
}

__global__ void reduce(const float* input, const float* output, int N) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int wid = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  float val = (x < N) ? input[x] : 0.0f;
  warpReduce(val);

  __shared__ smem[32];
  if (lane_id == 0) {
    smem[wid] = val;
  }

  __syncthreads();

  if (wid == 0) {
    val = (lane_id < blockDim.x / 32) ? smem[lane_id] : 0.0f;
    warpReduce(val);
    if (lane_id == 0) {
      atomic_add(output, val);
    }
  }
}

#define BM 128
#define BN 128
#define TM 8
#define TN 8
#define BK 16

#define THREADS ((BM / TM) * (BN / TN))

__global__ void gemm(const float* a, const float* b, float *c, int M, int N, int K) {
  int br = blockDim.y * BM;
  int bc = blockDim.x * BN;

  int tid = threadIdx.x;
  int ty = tid % (BN / TN), tx = tid / (BN / TN);

  __shared__ amem[BM][BK], bmem[BM][BK];

  float cmem[TM][TN] = {0.0f};
  for (int k = 0 ; k < K ; k += BK) {
    for (int i = tid ; i < BM * BK ; i += THREADS) {
      int row = i / BK, col = i % BK;
      if (row < BM && col < BK) {
        amem[row][col] = a[(br + row) * BK + k + col];
      } else {
        amem[row][col] = 0.0f;
      }
    }
    for (int i = tid ; i < BK * BN ; i += THREADS) {
      int row = i / BN, col = i % BN;
      if (row < BK && col < BN) {
        bmem[row][col] = b[(k + row) * BN + bc + col];
      }
    }
    
    __syncthreads();

    for (int kk = 0 ; kk < BK ; kk++) {
      float ar[BM], br[BN];
      for (int i = 0 ; i < BM ; i++) ar[i] = amem[i][kk];
      for (int j = 0 ; j < BN ; j++) br[j] = bmem[kk][j];
      for (int i = 0 ; i < BM ; i++)
        for (int j = 0 ; j < BN ; j++)
          cmem[i][j] += ar[i] * br[j];
    }

    __syncthreads();
  }

  int dy = br + ty * TM, dx = bc + tx * TN;
  for (int i = 0 ; i < TM ; i++)
    for (int j = 0 ; j < TN ; j++) {
      int row = dy + i, col = dx + j;
      if (row < M && col < N)
        c[row * BN + col] = cmem[row][col];
    }

}