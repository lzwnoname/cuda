
template <int BM, int BN, int BK, int TM, int TN,
          int PAD_A = 4, int PAD_B = 1>
__global__ void gemm(const float *A, const float *B, float *C, int M, int N, int K) {

  __shared__ float As[BK][BM + PAD_A]; // 转置避免bank conflict
  __shared__ float Bs[BK][BN + PAD_B];  

  // block负责的C子矩阵的坐标
  int c_row = blockIdx.y * BM;
  int c_col = blockIdx.x * BN;

  constexpr int NUM_THREADS = (BM * BN) / (TM * TN);
  const int thread_row = threadIdx.x / (BN / TN); // 每个BLOCK x方向有BN / TN个线程
  const int thread_col = threadIdx.x % (BN / TN);

  float Ct[TM][TN] = {0.0f};
  float reg_m[TM] = {0.0f};
  float reg_n[TN] = {0.0f};

  // 计算协作加载时的索引
  const int inner_row_A = threadIdx.x / (BK / 4);
  const int inner_col_A = threadIdx.x % (BK / 4) * 4;
  constexpr int stride_A = NUM_THREADS / (BK / 4);
  const int inner_row_B = threadIdx.x / (BN / 4);
  const int inner_col_B = threadIdx.x % (BN / 4) * 4;
  constexpr int stride_B = NUM_THREADS / (BN / 4);

  // 每个线程负责 TM * TN 大小的子矩阵
  // 分块矩阵乘，这里外部每一次做 (TM, BK) * (BK, TN)
  for (int k = 0 ; k < K ; k += BK) {

    // 协作加载, 对于矩阵A(BM, BK)，每次循环加载As(i->i+load_offset,k->k+BK), Bs(i->i+load_offset, c_col->c_ol+BN)
    for (int load_offset = 0 ; load_offset < BM ; load_offset += stride_A) {
      int row_id = c_row + inner_row_A + load_offset;
      int col_id = k + inner_col_A;
      if (row_id < M && col_id + 3 < K) {
        float4 tmp = reinterpret_cast<float4*>(&A[row_id][col_id])[0];
        As[inner_col_A + 0][inner_row_A + load_offset] = tmp.x;
        As[inner_col_A + 1][inner_row_A + load_offset] = tmp.y;
        As[inner_col_A + 2][inner_row_A + load_offset] = tmp.z;
        As[inner_col_A + 3][inner_row_A + load_offset] = tmp.w;
      } else {
        for (int i = 0 ; i < 4 ; i++)
          As[inner_col_A + i][inner_row_A + load_offset] = 0;
      }
    }

    for (int load_offset = 0 ; load_offset < BK ; load_offset += stride_B) {
      int row_id = k + inner_row_B + load_offset;
      int col_id = c_col + inner_col_B;
      if (row_id < K && col_id + 3 < N) {
        *reinterpret_cast<float4*>(&Bs[inner_row_B + load_offset][inner_col_B]) =
          reinterpret_cast<float4*>(&B[row_id][col_id])[0];
      } else {
        for (int i = 0 ; i < 4 ; i++)
          Bs[inner_row_B + load_offset][inner_col_B + i] = 0;
      }
    }

    __syncthreads();

    for (int p = 0 ; p < BK ; p++) {

      // 寄存器缓存
      for (int i = 0 ; i < TM ; i += 4) {
        *reinterpret_cast<float4*>(&reg_m[i]) = *reinterpret_cast<float4*>(&As[p][thread_row * TM + i]);
      }
      for (int j = 0 ; j < TN ; j += 4) {
        *reinterpret_cast<float4*>(&reg_n[j]) = *reinterpret_cast<float4*>(&Bs[p][thread_col * TN + j]);
      }

      for (int i = 0 ; i < TM ; i++)
        for (int j = 0 ; j < TN ; j++) {
          Ct[i][j] += reg_m[i] * reg_n[j];
        }
    }

    __syncthreads();
  }

  // 拷贝回结果数组
  for (int i = 0 ; i < TM ; i++) {
    int id_y = thread_row * TM + i + c_row;
    for (int j = 0 ; j < TN ; j += 4) {
      float4 out;
      out.x = Ct[i][j + 0];
      out.y = Ct[i][j + 1];
      out.z = Ct[i][j + 2];
      out.w = Ct[i][j + 3];
      int id_x = thread_col * TN + j + c_col;
      *reinterpret_cast<float4*>(&C[id_y][id_x]) = out;
    }
  }
}