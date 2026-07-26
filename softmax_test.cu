#include <cuda_runtime.h>

__device__ __managed__ float esum;

__device__ void warpReduce(float &localm, float &locald) {
    for (int offset = 16 ; offset ; offset >>= 1) {
        m = __shfl_down_sync(0xffffffff, local_m, offset);
        d = __shfl_down_sync(0xffffffff, local_d, offset);
        newm = m > local_m ? m : localm;
        locald = locald * __expf(local_m - new_m) + d * __expf(m - new_m);
    }
}

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int wid = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    float localm = x < N ? input[x] : -1000000000.0f;
    float locald = x < N ? 1.0f : 0.0f;
    
    warpReduce(localm, locald);

    __shared__ float mmem[32], dmem[32];
    if (lane_id == 0) {
        smem[wid] = localm;
        dmem[wid] = locald;
    }
    __syncthreads();

    if (wid == 0) {
        local_m = (lane_id < blockDim.x / 32) ? mmem[lane_id] : -1000000000.0f;
        local_d = (lane_id < blockDim.x / 32) ? dmem[lane_id] : 0.0f;
        
        warpReduce(localm, locald);

        
    }

    __syncthreads();

    if (x < N)
        output[x] = __expf(input[x]) / esum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
