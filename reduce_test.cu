#include <cuda_runtime.h>
#include <math.h>

__device__ __managed__ float emax, esum;

__device__ void warpReduceSum(float &val) {
    for (int offset = 16 ; offset ; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
}

__global__ void reduceSum(const float* input, float* output, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int wid = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    float val = x < N ? input[x] : 0.0f;
    for (int offset = 16 ; offset ; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    
    __shared__ float smem[32];
    if (lane_id == 0) {
        smem[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = (lane_id < blockDim.x / 32) ? smem[lane_id] : 0.0f;
        for (int offset = 16 ; offset ; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane_id == 0)
            atomicAdd(&esum, val);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    emax = -INFINITY;
    esum = 0.0f;

    reduceSum<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
