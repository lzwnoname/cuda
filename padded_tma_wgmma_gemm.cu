// SPDX-License-Identifier: BSD-3-Clause
//
// 支持任意 M/N/K 的教学版 TMA + WGMMA GEMM。
//
// 对外接口使用常见的 row-major：
//   A[M,K]、B[K,N]、C[M,N]
// 内部先补零到：
//   Mp = ceil(M / 128) * 128
//   Np = ceil(N / 128) * 128
//   Kp = ceil(K / 64)  * 64，且至少包含 2 个 K tile
//
// padded kernel 只看到完整 tile，因此 TMA 和 WGMMA 主循环不需要边界分支。
// 核心路径：global memory -> TMA -> swizzled shared memory -> WGMMA -> C。
//
// 需要：Hopper GPU、CUDA 12.x、CUTLASS 3.x、-arch=sm_90a。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/mma_sm90.hpp>

#include "cutlass/arch/barrier.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/util/helper_cuda.hpp"

using namespace cute;

namespace {

constexpr int BlockM = 128;
constexpr int BlockN = 128;
constexpr int BlockK = 64;
constexpr int Stages = 2;

using InputElement = cute::half_t;
using OutputElement = float;

constexpr int round_up(int x, int multiple) {
  return ((x + multiple - 1) / multiple) * multiple;
}

// 每个 stage 保存一块 A[BlockM,BlockK] 和 B^T[BlockN,BlockK]。
// layout 的最后一个维度就是循环使用的 pipeline stage。
template <class SmemLayoutA, class SmemLayoutB>
struct PaddedSharedStorage {
  alignas(128) ArrayEngine<InputElement, cosize_v<SmemLayoutA>> A;
  alignas(128) ArrayEngine<InputElement, cosize_v<SmemLayoutB>> B;

  // producer_barrier：TMA 写完某个 stage 后通知 WGMMA。
  // consumer_barrier：WGMMA 读完某个 stage 后允许 TMA 覆盖它。
  uint64_t producer_barrier[Stages];
  uint64_t consumer_barrier[Stages];
};

template <class ProblemShape, class CtaTiler,
          class SmemLayoutA, class TmaA,
          class SmemLayoutB, class TmaB,
          class CStride, class TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void padded_gemm_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    InputElement const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
    InputElement const* Bt, CUTLASS_GRID_CONSTANT TmaB const tma_b,
    OutputElement* C, CStride dC,
    TiledMma mma) {
  auto [M, N, K] = problem_shape;

  // TMA atom 自带 global tensor descriptor，因此 kernel 只需取逻辑视图。
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));   // (Mp,Kp)
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));   // (Np,Kp)，实际是 B^T
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

  // 一个 CTA 负责 C 的一个 128x128 tile，第三维保留 K tile 序列。
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(
      mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(
      mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(
      mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  using Storage = PaddedSharedStorage<SmemLayoutA, SmemLayoutB>;
  Storage& smem = *reinterpret_cast<Storage*>(shared_memory);

  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

  // tAgA/tBgB：global memory 中沿 K 方向排列的完整 tile。
  // tAsA/tBsB：shared memory 中循环复用的 stage。
  auto [tAgA, tAsA] = tma_partition(
      tma_a, Int<0>{}, Layout<_1>{},
      group_modes<0, 2>(sA), group_modes<0, 2>(gA));
  auto [tBgB, tBsB] = tma_partition(
      tma_b, Int<0>{}, Layout<_1>{},
      group_modes<0, 2>(sB), group_modes<0, 2>(gB));

  // 一个 barrier 同时追踪 A tile 和 B tile 的总传输字节数。
  constexpr int transaction_bytes =
      sizeof(make_tensor_like(tensor<0>(tAsA))) +
      sizeof(make_tensor_like(tensor<0>(tBsB)));

  int k_tile_count = size<1>(tAgA);
  int next_k_tile = 0;

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int elected_lane = cute::elect_one_sync();

  using ProducerBarrier = cutlass::arch::ClusterTransactionBarrier;
  using ConsumerBarrier = cutlass::arch::ClusterBarrier;

  // 只有 warp 0 的一个 lane 初始化/发起 TMA；所有 128 个线程参与 WGMMA。
  if (warp_idx == 0 && elected_lane) {
    for (int stage = 0; stage < Stages; ++stage) {
      ProducerBarrier::init(&smem.producer_barrier[stage], 1);
      ConsumerBarrier::init(&smem.consumer_barrier[stage], 128);
    }
  }
  cluster_sync();

  // 先填满两个 stage。host 端保证 Kp 至少为 2*BlockK。
  for (int stage = 0; stage < Stages; ++stage) {
    if (warp_idx == 0 && elected_lane) {
      ProducerBarrier::arrive_and_expect_tx(
          &smem.producer_barrier[stage], transaction_bytes);

      copy(tma_a.with(smem.producer_barrier[stage]),
           tAgA(_, next_k_tile), tAsA(_, stage));
      copy(tma_b.with(smem.producer_barrier[stage]),
           tBgB(_, next_k_tile), tBsB(_, stage));
    }
    --k_tile_count;
    ++next_k_tile;
  }

  // WGMMA 的 A/B fragment 是 shared-memory descriptor；
  // C fragment 分布在一个 warpgroup（4 warps）的寄存器中。
  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  // phase 每绕 pipeline 一圈翻转一次，用来区分同一 barrier 的不同代。
  auto write_state = cutlass::PipelineState<Stages>();
  auto read_state = cutlass::PipelineState<Stages>();

  // 循环条件让最后两个已经预取的 stage 也能被消费完。
  CUTE_NO_UNROLL
  while (k_tile_count > -Stages) {
    int read_stage = read_state.index();

    // TMA 完成后才能让 WGMMA 读取该 shared-memory stage。
    ProducerBarrier::wait(
        &smem.producer_barrier[read_stage], read_state.phase());

    warpgroup_arrive();
    gemm(mma,
         tCrA(_, _, _, read_stage),
         tCrB(_, _, _, read_stage),
         tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    // 128 个消费者都到达后，该 stage 才能重新被 producer 覆盖。
    ConsumerBarrier::arrive(&smem.consumer_barrier[read_stage]);
    ++read_state;

    // 如果还有 K tile，则异步搬入刚刚空出来的 stage。
    if (warp_idx == 0 && elected_lane && k_tile_count > 0) {
      int write_stage = write_state.index();

      ConsumerBarrier::wait(
          &smem.consumer_barrier[write_stage], write_state.phase());

      ProducerBarrier::arrive_and_expect_tx(
          &smem.producer_barrier[write_stage], transaction_bytes);

      copy(tma_a.with(smem.producer_barrier[write_stage]),
           tAgA(_, next_k_tile), tAsA(_, write_stage));
      copy(tma_b.with(smem.producer_barrier[write_stage]),
           tBgB(_, next_k_tile), tBsB(_, write_stage));

      ++write_state;
    }

    --k_tile_count;
    ++next_k_tile;
  }

  // padded C 没有越界 tile，因此可以直接无条件写回完整 128x128 tile。
  axpby(1.0f, tCrC, 0.0f, tCgC);
}

// 只接收已经补齐后的 column-major 张量。
// A_pad:  (Mp,Kp)
// Bt_pad: (Np,Kp)
// C_pad:  (Mp,Np)
void launch_padded_kernel(
    int Mp, int Np, int Kp,
    InputElement const* A_pad,
    InputElement const* Bt_pad,
    OutputElement* C_pad) {
  auto problem_shape = make_shape(Mp, Np, Kp);
  auto cta_tiler = make_shape(
      Int<BlockM>{}, Int<BlockN>{}, Int<BlockK>{});

  // column-major：(row stride=1, column stride=leading dimension)。
  auto dA = make_stride(Int<1>{}, Mp);
  auto dB = make_stride(Int<1>{}, Np);
  auto dC = make_stride(Int<1>{}, Mp);

  Tensor mA = make_tensor(
      A_pad, make_shape(Mp, Kp), dA);
  Tensor mB = make_tensor(
      Bt_pad, make_shape(Np, Kp), dB);

  // SW128 是 Hopper GMMA 的 128-byte shared-memory swizzle atom。
  auto sA = tile_to_shape(
      GMMA::Layout_MN_SW128_Atom<InputElement>{},
      make_shape(Int<BlockM>{}, Int<BlockK>{}, Int<Stages>{}));
  auto sB = tile_to_shape(
      GMMA::Layout_MN_SW128_Atom<InputElement>{},
      make_shape(Int<BlockN>{}, Int<BlockK>{}, Int<Stages>{}));

  // 一个 atom 计算 64x64x16；输入为 FP16，累加器和输出为 FP32。
  // CuTe 自动把 atom 铺成 CTA 的 128x128x64 工作。
  auto tiled_mma = make_tiled_mma(
      SM90_64x64x16_F32F16F16_SS<
          GMMA::Major::MN, GMMA::Major::MN>{});

  // host 端构造 TMA descriptor：global stride、tile shape、shared layout
  // 和 swizzle 都被记录在 atom 中。
  auto tma_a = make_tma_atom(
      SM90_TMA_LOAD{}, mA, sA(_, _, 0),
      make_shape(Int<BlockM>{}, Int<BlockK>{}));
  auto tma_b = make_tma_atom(
      SM90_TMA_LOAD{}, mB, sB(_, _, 0),
      make_shape(Int<BlockN>{}, Int<BlockK>{}));

  using SmemA = decltype(sA);
  using SmemB = decltype(sB);
  int smem_bytes = sizeof(PaddedSharedStorage<SmemA, SmemB>);

  dim3 grid(Mp / BlockM, Np / BlockN, 1);
  dim3 block(size(tiled_mma));
  dim3 cluster(1, 1, 1);

  auto kernel = &padded_gemm_kernel<
      decltype(problem_shape), decltype(cta_tiler),
      SmemA, decltype(tma_a),
      SmemB, decltype(tma_b),
      decltype(dC), decltype(tiled_mma)>;

  cutlass::ClusterLaunchParams params{
      grid, block, cluster, smem_bytes};

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      reinterpret_cast<void const*>(kernel),
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_bytes));

  auto status = cutlass::launch_kernel_on_cluster(
      params, reinterpret_cast<void const*>(kernel),
      problem_shape, cta_tiler,
      A_pad, tma_a, Bt_pad, tma_b, C_pad, dC, tiled_mma);

  CUTE_CHECK_LAST();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "TMA + WGMMA kernel launch failed\n";
    std::exit(EXIT_FAILURE);
  }
}

// 对外的简单 row-major API。
// 为了突出 GEMM 主循环，教学版在 host 上完成 padding/转置；
// 实际服务中应使用 CUDA padding kernel，或让上游直接产生 padded tensor。
std::vector<float> padded_tma_wgmma_gemm(
    std::vector<float> const& A,
    std::vector<float> const& B,
    int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0 ||
      static_cast<int>(A.size()) != M * K ||
      static_cast<int>(B.size()) != K * N) {
    throw std::invalid_argument("invalid GEMM shape or input size");
  }

  int Mp = round_up(M, BlockM);
  int Np = round_up(N, BlockN);
  int Kp = std::max(round_up(K, BlockK), Stages * BlockK);

  // 内部采用 column-major，并把 padding 区域初始化为 0。
  // A_pad[m + k*Mp] = A[m*K + k]
  // Bt_pad[n + k*Np] = B[k*N + n]
  thrust::host_vector<InputElement> hA_pad(Mp * Kp, InputElement(0));
  thrust::host_vector<InputElement> hBt_pad(Np * Kp, InputElement(0));
  thrust::host_vector<OutputElement> hC_pad(Mp * Np, OutputElement(0));

  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      hA_pad[m + k * Mp] = InputElement(A[m * K + k]);
    }
  }
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      hBt_pad[n + k * Np] = InputElement(B[k * N + n]);
    }
  }

  thrust::device_vector<InputElement> dA_pad = hA_pad;
  thrust::device_vector<InputElement> dBt_pad = hBt_pad;
  thrust::device_vector<OutputElement> dC_pad = hC_pad;

  launch_padded_kernel(
      Mp, Np, Kp,
      dA_pad.data().get(),
      dBt_pad.data().get(),
      dC_pad.data().get());

  CUTE_CHECK_LAST();
  hC_pad = dC_pad;

  // 只取有效区域，并恢复为 row-major C[M,N]。
  std::vector<float> C(M * N);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      C[m * N + n] = static_cast<float>(hC_pad[m + n * Mp]);
    }
  }
  return C;
}

std::vector<float> reference_gemm(
    std::vector<float> const& A,
    std::vector<float> const& B,
    int M, int N, int K) {
  std::vector<float> C(M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum;
    }
  }
  return C;
}

}  // namespace

int main(int argc, char** argv) {
  cudaDeviceProp props{};
  CUTE_CHECK_ERROR(cudaGetDeviceProperties(&props, 0));
  if (props.major != 9) {
    std::cout << "This example requires Hopper compute capability 90a.\n";
    return 0;
  }

  // 用非对齐尺寸验证 padding 路径。
  int M = argc > 1 ? std::atoi(argv[1]) : 173;
  int N = argc > 2 ? std::atoi(argv[2]) : 219;
  int K = argc > 3 ? std::atoi(argv[3]) : 97;

  if (M <= 0 || N <= 0 || K <= 0) {
    std::cerr << "usage: padded_tma_wgmma_gemm [M N K]\n";
    return EXIT_FAILURE;
  }

  std::mt19937 rng(7);
  std::uniform_int_distribution<int> sign(0, 1);
  std::vector<float> A(M * K);
  std::vector<float> B(K * N);

  // +/-1 输入可降低 fp16 输出误差，便于先验证 layout 和 padding。
  for (float& x : A) x = sign(rng) ? 1.0f : -1.0f;
  for (float& x : B) x = sign(rng) ? 1.0f : -1.0f;

  auto C = padded_tma_wgmma_gemm(A, B, M, N, K);
  auto ref = reference_gemm(A, B, M, N, K);

  float max_error = 0.0f;
  for (size_t i = 0; i < C.size(); ++i) {
    max_error = std::max(max_error, std::fabs(C[i] - ref[i]));
  }

  std::printf(
      "M=%d N=%d K=%d -> padded=(%d,%d,%d), max_abs_error=%.3f\n",
      M, N, K,
      round_up(M, BlockM),
      round_up(N, BlockN),
      std::max(round_up(K, BlockK), Stages * BlockK),
      max_error);

  return max_error < 1.0f ? EXIT_SUCCESS : EXIT_FAILURE;
}
