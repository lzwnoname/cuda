#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BM 128   // block tile: M 方向
#define BN 128   // block tile: N 方向
#define BK 16    // k 方向每次搬运的厚度
#define TM 8     // thread tile: 每线程算 TM x TN 个输出
#define TN 8

// 16x16 = 256 个线程拼出 128x128 的 block tile
#define THREADS (BM / TM) * (BN / TN)

__global__ void gemm_tile(const half* A, const half* B, half* C,
                          int M, int N, int K, float alpha, float beta) {
    __shared__ float As[BK][BM];  // A 块转置存放，计算时读连续地址
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;                 // 一维 256
    int ty = tid / (BN / TN);              // 线程在 block tile 里的行 (0~15)
    int tx = tid % (BN / TN);              // 线程在 block tile 里的列 (0~15)

    int block_row = blockIdx.y * BM;       // block tile 左上角的全局坐标
    int block_col = blockIdx.x * BN;

    float mC[TM][TN] = {0.0f};             // 寄存器累加器，只有 64 个 float

    for (int k = 0; k < K; k += BK) {
        // ---- 1. 全 block 协作搬 A 块 (BM x BK)，转置写入 As ----
        for (int i = tid; i < BM * BK; i += THREADS) {
            int row = i / BK;              // tile 内的 m 偏移
            int col = i % BK;              // tile 内的 k 偏移
            int gr = block_row + row, gc = k + col;
            As[col][row] = (gr < M && gc < K) ? (float)A[gr * K + gc] : 0.0f;
        }
        // ---- 2. 全 block 协作搬 B 块 (BK x BN) ----
        for (int i = tid; i < BK * BN; i += THREADS) {
            int row = i / BN;              // tile 内的 k 偏移
            int col = i % BN;              // tile 内的 n 偏移
            int gr = k + row, gc = block_col + col;
            Bs[row][col] = (gr < K && gc < N) ? (float)B[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        // ---- 3. 每线程从 smem 读片段，做 8x8 外积累加 ----
        for (int kk = 0; kk < BK; kk++) {
            float rA[TM], rB[TN];
            for (int i = 0; i < TM; i++) rA[i] = As[kk][ty * TM + i];
            for (int j = 0; j < TN; j++) rB[j] = Bs[kk][tx * TN + j];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    mC[i][j] += rA[i] * rB[j];
        }
        __syncthreads();                   // 读完才能覆盖下一轮 smem
    }

    // ---- 4. 写回，带边界检查和 beta=0 短路 ----
    int row0 = block_row + ty * TM;
    int col0 = block_col + tx * TN;
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int r = row0 + i, c = col0 + j;
            if (r < M && c < N) {
                if (beta == 0.0f)
                    C[r * N + c] = (half)(alpha * mC[i][j]);
                else
                    C[r * N + c] = (half)(alpha * mC[i][j] + beta * (float)C[r * N + c]);
            }
        }
    }
}

extern "C" void solve(const half* A, const half* B, half* C,
                      int M, int N, int K, float alpha, float beta) {
    dim3 block(THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_tile<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}
