# GEMM Thread Tile + Bank Conflict → Double Buffering 优化方案

## 当前代码状态

`gemm.cu` 已实现：
- **Thread Tile**：每线程计算 TM×TN 子矩阵，寄存器累积减少 shared memory 访问
- **A 转置存储 + PAD_A**：消除 A 矩阵 shared memory bank conflict
- **B PAD_B**：摊散 B 矩阵连续迭代间的 bank conflict
- **float4 向量化**：全局加载 / 写回均使用 128-bit 向量化访存

当前瓶颈：**加载和计算串行执行**。每轮 K-tile 的流水线是：

```
Load A/B tile → __syncthreads → Compute → __syncthreads → (下一轮) Load → ...
```

global memory 延迟（~300-800 cycles）完全裸露在关键路径上，SM 在加载阶段大量空闲。

---

## 优化目标

GEMM 的数据流天然是 **三级流水线**：

```
Global Memory ──→ Shared Memory ──→ Register ──→ FMA
   (DRAM)          (on-chip)        (on-chip)
  ~300-800 cyc     ~20-30 cyc       ~0 cyc
```

每一级都有延迟。通过 **双缓冲（double buffering）** 将各级延迟隐藏在计算背后，形成两级重叠：

```
Level 1: global → shared  双缓冲  (方案一/二/三)
Level 2: shared → register 双缓冲  (方案四)
```

两层叠加后的理想流水线：

```
         ┌── Load tile 0 ──┐                         Level 1: global → shared
                  │
          __syncthreads
                  │
    ┌─────────────┼─────────────────────────┐
    │          Compute tile 0               │          Level 1 计算阶段
    │  ┌─── Load tile 1 ───────────────┐   │
    │  │                                │   │
    │  │  ┌ prefetch reg[0] p=0 ──┐    │   │
    │  │  │                        │    │   │
    │  │  │  for p=1..BK-1:        │    │   │
    │  │  │    load reg[1] from    │    │   │      Level 2: shared → register
    │  │  │    As/Bs[p] ──────┐    │    │   │      软件流水线
    │  │  │    FMA reg[0] ←──┘    │    │   │
    │  │  │    swap reg[0]↔reg[1] │    │   │
    │  │  │  FMA reg[0] (last)    │    │   │
    │  │  └───────────────────────┘    │   │
    │  └───────────────────────────────┘   │
    └──────────────────────────────────────┘
                  │
          __syncthreads
                  │
                 ...
```

Level 1 隐藏 global memory 延迟，Level 2 隐藏 shared memory 延迟。最终只有 **Level 2 最后一个 p 的 shared load** 和 **Level 1 最后一个 tile 的 global load** 的延迟裸露在外。

---

## 方案一：cp.async 异步拷贝（SM80+, 推荐）

CUDA 11+ 提供 `cp.async` 指令族，可真正异步地将 global memory 拷贝到 shared memory，不需要线程主动等待。

### 1. Shared memory 布局变化

```cpp
// 原：单份 buffer
__shared__ float As[BK][BM + PAD_A];
__shared__ float Bs[BK][BN + PAD_B];

// 改为：双份 buffer（ping-pong）
__shared__ float As[2][BK][BM + PAD_A];
__shared__ float Bs[2][BK][BN + PAD_B];
```

### 2. 存储空间代价

- A: `2 × BK × (BM + PAD_A) × 4 bytes`
- B: `2 × BK × (BN + PAD_B) × 4 bytes`

Shared memory 用量翻倍，需确保不超过 SM 限制（通常 48KB~164KB，视架构而定）。典型配置 `BM=128, BN=128, BK=8, PAD_A=4, PAD_B=1` 下：
- 原：`8×132×4 + 8×129×4 = 4224 + 4128 = 8352 bytes ≈ 8.2KB`
- 新：`≈ 16.4KB`，仍在合理范围内

### 3. 算法伪码

```cpp
template <int BM, int BN, int BK, int TM, int TN, int PAD_A = 4, int PAD_B = 1>
__global__ void gemm_double_buffering(
    const float *A, const float *B, float *C, int M, int N, int K
) {
    // --- 双缓冲 shared memory ---
    __shared__ float As[2][BK][BM + PAD_A];
    __shared__ float Bs[2][BK][BN + PAD_B];

    // --- 线程 partition & 寄存器 ---
    // （与当前 gemm.cu 相同，省略 thread_row/thread_col/Ct/reg_m/reg_n 定义）
    float Ct[TM][TN] = {0.0f};
    float reg_m[TM] = {0.0f};
    float reg_n[TN] = {0.0f};

    // 协作加载索引（与当前相同）
    const int inner_row_A = threadIdx.x / (BK / 4);
    const int inner_col_A = threadIdx.x % (BK / 4) * 4;
    constexpr int stride_A = NUM_THREADS / (BK / 4);
    const int inner_row_B = threadIdx.x / (BN / 4);
    const int inner_col_B = threadIdx.x % (BN / 4) * 4;
    constexpr int stride_B = NUM_THREADS / (BN / 4);

    // --- Pipeline 状态 ---
    int read_idx  = 0;   // 下一块写入哪个 buffer
    int comp_idx  = 1;   // 当前计算使用哪个 buffer

    // --- 预取第一块 (k=0) 到 read_idx ---
    for (int load_offset = 0; load_offset < BM; load_offset += stride_A) {
        int row_id = c_row + inner_row_A + load_offset;
        int col_id = 0 + inner_col_A;  // k=0
        if (row_id < M && col_id + 3 < K) {
            // cp.async: 从 global(A) 拷贝 16 bytes 到 &As[read_idx][inner_col_A][...]
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n" ::
                "r"((unsigned)(4 * ((inner_col_A + 0) * (BM + PAD_A) + inner_row_A + load_offset)
                               + __cvta_generic_to_shared(&As[read_idx][0][0]))),
                "l"(&A[row_id * K + col_id])
            );
            // ... 类似拷贝As[read_idx][inner_col_A+1..3][...]
        } else {
            // 边界置零：用 cp.async 或直接写零
            for (int i = 0; i < 4; i++)
                As[read_idx][inner_col_A + i][inner_row_A + load_offset] = 0;
        }
    }
    for (int load_offset = 0; load_offset < BK; load_offset += stride_B) {
        // 同理用 cp.async 加载 B → Bs[read_idx]
        // ...
    }
    // 提交 cp.async group
    asm volatile("cp.async.commit_group;\n" ::);

    // --- 主循环：流水化 K 方向 ---
    for (int k = BK; k < K; k += BK) {
        // Step 1: 加载下一块到 read_idx
        //   （与预取逻辑相同，k 换成当前值）
        // ... load A tile at k into As[read_idx] ...
        // ... load B tile at k into Bs[read_idx] ...

        // 等待上一组 cp.async 完成（comp_idx 的数据已就绪）
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();  // cp.async 写入 shared memory 的可见性保证

        // Step 2: 计算 comp_idx 对应的 tile
        for (int p = 0; p < BK; p++) {
            for (int i = 0; i < TM; i += 4) {
                *reinterpret_cast<float4*>(&reg_m[i]) =
                    *reinterpret_cast<float4*>(&As[comp_idx][p][thread_row * TM + i]);
            }
            for (int j = 0; j < TN; j += 4) {
                *reinterpret_cast<float4*>(&reg_n[j]) =
                    *reinterpret_cast<float4*>(&Bs[comp_idx][p][thread_col * TN + j]);
            }
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    Ct[i][j] += reg_m[i] * reg_n[j];
        }

        // Step 3: 交换 buffer index
        int tmp  = read_idx;
        read_idx = comp_idx;
        comp_idx = tmp;
    }

    // --- 消费最后一块（comp_idx）---
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    for (int p = 0; p < BK; p++) {
        // （计算逻辑同上）
    }

    // --- 写回 C（与当前 gemm.cu 相同）---
    // ...
}
```

### 4. 关键点说明

| 要点 | 说明 |
|------|------|
| `cp.async.ca.shared.global` | 从 global memory 异步拷贝到 shared memory，支持 4/8/16 bytes。`ca` = cache-all，适合有 L1 cache 复用场景 |
| `cp.async.commit_group` | 将一组异步拷贝归为一个 group，用于批量等待 |
| `cp.async.wait_group N` | 等待除了最近 N 个 group 外的所有拷贝完成。`wait_group 1` = 等所有之前的，`wait_group 0` = 等全部 |
| `__syncthreads()` 仍需保留 | 按 CUDA 编程模型，`cp.async` 写入 shared memory 后，需 `__syncthreads()` 保证其他线程可见 |
| arm 架构兼容 | `cp.async` 是 SM80+ (A100/RTX30系列) 指令，老卡退化为手动双缓冲（方案二） |

### 5. 性能预期

- 理想情况下，global memory 加载延迟被完全隐藏在计算里
- 实际收益取决于：compute-bound 还是 memory-bound、tile 大小、SM 占用率
- 典型提升 1.3× ~ 2×（对比无 double buffering 的 thread-tile 版本）

---

## 方案二：手动双缓冲（兼容 SM < 80）

对不支持的硬件，用纯软件方式实现类似效果——在同一 K-tile 迭代内交叉安排计算和下一轮的部分加载。

### 核心思路

利用**循环展开**，显式交替加载和计算，但不再额外拆一个 warp 专门负责加载（所有线程仍协作）：

```cpp
__shared__ float As[2][BK][BM + PAD_A];
__shared__ float Bs[2][BK][BN + PAD_B];

// 预取 k=0 → buf[0]
load_tile(/*buf=*/0, /*k=*/0);
__syncthreads();

int buf = 0, next_buf = 1;

// 主循环：每次处理 2 个 K-tile
for (int k = BK; k < K; k += BK) {
    // 加载 k → buf[next_buf]
    load_tile(next_buf, k);
    // 计算 k-BK → buf[buf]
    compute_tile(buf);
    __syncthreads();   // 等本次加载完成

    // 交换
    swap(buf, next_buf);
}

// 尾块
compute_tile(buf);
__syncthreads();
```

### 与方案一的区别

| 维度 | cp.async (方案一) | 手动双缓冲 (方案二) |
|------|------------------|---------------------|
| 硬件要求 | SM ≥ 80 (Ampere+) | 所有 SM |
| 异步程度 | 真异步，加载与计算真正并行 | 伪异步，靠 warp 调度器指令级并行 |
| `__syncthreads()` | 仅 1 次/轮 | 仍需 1 次/轮（加载完成后） |
| 效果 | 接近完全隐藏 load 延迟 | 部分隐藏（受限于 ILP/MLP） |
| 复杂度 | 中（需内联 asm 或使用 cuda::pipeline） | 低（仅 buffer 索引管理） |
| 推荐度 | ⭐⭐⭐⭐⭐ | ⭐⭐ |

---

## 方案三：warp-level 异步流水线（cuda::pipeline, SM80+）

CUDA 12+ 提供 C++ 封装 `cuda::pipeline`，比手写 cp.async 内联汇编更易维护：

```cpp
#include <cuda/pipeline>

__shared__ float As[2][BK][BM + PAD_A];
__shared__ float Bs[2][BK][BN + PAD_B];

// 声明 pipeline，最多 2 个 stage
constexpr int pipeline_stages = 2;
__shared__ cuda::pipeline_shared_state<pipeline_stages> pipe_state;
auto pipe = cuda::make_pipeline(block_thread_count(), block_thread_count(), pipe_state);

// --- 使用 ---
// 异步拷贝
cuda::memcpy_async(/*shared*/ dest, /*global*/ src, sizeof(float4), pipe);
pipe.commit();

// 等待 stage
pipe.wait_prior<1>();
__syncthreads();

// ... compute ...
```

这个封装自动处理 commit_group / wait_group 的索引管理，可读性更好。

---

## 方案四：寄存器双缓冲 / 软件流水线（shared → register）

当前 `gemm.cu` 的 inner loop（每个 K-tile 内部）是严格串行的：

```cpp
for (int p = 0; p < BK; p++) {
    // ① load: shared → register  (~20-30 cycles)
    load reg_m from As[p][*], reg_n from Bs[p][*]
    // ② compute: FMA
    for (i,j) Ct[i][j] += reg_m[i] * reg_n[j];
}
```

shared memory 读取延迟（~20-30 cycles）和 FMA 计算交替执行，无法重叠。

### 方案：寄存器双缓冲

同时持有 **两份** `reg_m` / `reg_n`，让当前 p 的 FMA 与 p+1 的 shared memory 预取并行：

```cpp
// 两份寄存器缓冲，通过索引切换代替物理拷贝
float reg_m[2][TM] = {0.0f};
float reg_n[2][TN] = {0.0f};

int cur = 0, nxt = 1;

// --- 预取 p=0 ---
for (int i = 0; i < TM; i += 4)
    *reinterpret_cast<float4*>(&reg_m[cur][i]) =
        *reinterpret_cast<float4*>(&As[comp_idx][0][thread_row * TM + i]);
for (int j = 0; j < TN; j += 4)
    *reinterpret_cast<float4*>(&reg_n[cur][j]) =
        *reinterpret_cast<float4*>(&Bs[comp_idx][0][thread_col * TN + j]);

// --- 流水线主体：p=1 .. BK-1 ---
for (int p = 1; p < BK; p++) {
    // Step A: 预取 p 到 nxt buffer（与 Step B 的 FMA 可以指令级并行）
    for (int i = 0; i < TM; i += 4)
        *reinterpret_cast<float4*>(&reg_m[nxt][i]) =
            *reinterpret_cast<float4*>(&As[comp_idx][p][thread_row * TM + i]);
    for (int j = 0; j < TN; j += 4)
        *reinterpret_cast<float4*>(&reg_n[nxt][j]) =
            *reinterpret_cast<float4*>(&Bs[comp_idx][p][thread_col * TN + j]);

    // Step B: 用 cur buffer 的 p-1 数据做 FMA
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            Ct[i][j] += reg_m[cur][i] * reg_n[cur][j];

    // Step C: 切换索引
    //   （仅改一个 int，不拷贝寄存器数据）
    cur = 1 - cur;
    nxt = 1 - nxt;
}

// --- 尾块：消费最后一次预取（cur 指向最后一轮数据）---
for (int i = 0; i < TM; i++)
    for (int j = 0; j < TN; j++)
        Ct[i][j] += reg_m[cur][i] * reg_n[cur][j];
```

### 为什么有效

CUDA SM 的 warp 调度器支持**指令级并行（ILP）**：当一条 load 指令发射后，后续不依赖该结果的指令可以立即执行（乱序发射、顺序提交）。寄存器双缓冲正是利用了这一点：

```
cycle 0:   issue load  reg_m[1][0..3] from As[p]     // shared memory 读
cycle 1:   issue load  reg_n[1][0..3] from Bs[p]     // shared memory 读（不依赖 reg_m[1]）
cycle 2-:  issue FMA   Ct[0..3][*] += reg_m[0][*] * reg_n[0][*]  // 不依赖 reg_m[1]/reg_n[1]
           ... reg_m[1]/reg_n[1] 在 ~20 cycles 后返回 ...
cycle 22:  load 结果就绪，但此时 FMA 已经执行了 ~20 条指令
           → shared memory 延迟被完全隐藏 ✅
```

### 寄存器代价

| 项目 | 原 (无寄存器双缓冲) | 寄存器双缓冲 |
|------|---------------------|-------------|
| `reg_m` | TM 个寄存器 | 2 × TM 个寄存器 |
| `reg_n` | TN 个寄存器 | 2 × TN 个寄存器 |
| `Ct` | TM × TN 个寄存器 | TM × TN（不变） |
| `cur/nxt` | 0 | 2 个 int |
| **合计** | TM+TN+TM×TN | **2×(TM+TN)+TM×TN** |

典型配置 `TM=8, TN=8`：
- 原：8 + 8 + 64 = **80 registers**
- 双缓冲：16 + 16 + 64 = **96 registers**

CUDA 每线程寄存器上限通常 255（受限于 SM 的寄存器文件总量 / occupancy），96 仍在安全范围内。但若 `TM=16, TN=16`：
- 双缓冲：32 + 32 + 256 = **320 registers → 会 spilling！**

> ⚠️ 务必用 `nvcc -Xptxas -v` 查看实际寄存器用量，确保无 local memory spilling。

### Shared memory 代价

**零**。寄存器双缓冲不增加 shared memory 用量，仅增加寄存器。

---

## 方案组合：两级双缓冲（推荐完整方案）

将方案三（`cuda::pipeline` Level 1）与方案四（寄存器双缓冲 Level 2）组合：

```
                    ┌─ global 加载 tile k+1 ─┐   Level 1 buffer
                    │                        │
    ┌───────────────┼────────────────────────┼───┐
    │         shared tile k                  │   │
    │  ┌─ prefetch reg[0] p=0 ────┐         │   │
    │  │  for p=1..BK-1:           │         │   │   Level 2 buffer
    │  │    load reg[1] (p)   ─┐   │         │   │
    │  │    FMA   reg[0] (p-1)┘   │         │   │
    │  │    swap                 │         │   │
    │  │  FMA reg[0] (last)      │         │   │
    │  └─────────────────────────┘         │   │
    └──────────────────────────────────────┘───┘
```

每个 K-tile 的 shared memory 延迟和 global memory 延迟同时被隐藏，只裸露出：
- 第一块 K-tile (k=0) 的 global 加载
- 每 tile 内 p=0 的第一个 shared 预取

对大型 GEMM（K >> BK）而言，这两处裸露的延迟可忽略不计。

---

## 推荐实施路径（更新）

```
                      当前 gemm.cu
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     SM ≥ 80?          SM < 80?        需要极致性能?
          │                │                │
          ▼                ▼                ▼
    cuda::pipeline      手动双缓冲     cp.async 内联 asm
    (方案三, Level 1)   (方案二, L1)   (方案一, L1)
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  + 寄存器双缓冲 (方案四) │    ← Level 2，叠加在任意 L1 方案之上
              │    shared → register     │       与前三个方案正交，无冲突
              └────────────────────────┘
                           │
                           ▼
                   两级双缓冲完整方案
```

**建议优先实施**：方案三（`cuda::pipeline`）+ 方案四（寄存器双缓冲），获得最大收益。

---

## 注意事项

1. **Shared memory 容量**：Level 1 双缓冲使 shared memory 用量翻倍，需确认 SM 的 shared memory 上限。若 `BK × (BM + PAD_A) × 4 × 2 + BK × (BN + PAD_B) × 4 × 2 > smem_limit`，需减小 BM/BN/BK 或降回单缓冲。

2. **寄存器压力（新增）**：Level 2 寄存器双缓冲使 `reg_m`/`reg_n` 翻倍。编译后用 `nvcc -Xptxas -v` 检查每个线程的实际寄存器数以及是否出现 `lmem`（local memory spilling）。若 spilling，解决手段：
   - 减小 TM 和 TN（寄存器用量从 `2(TM+TN)+TM×TN` 下降）
   - 放弃 Level 2，仅保留 Level 1 双缓冲
   - 编译器调优：`__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` 或 `-maxrregcount=N`

3. **Occupancy 权衡**：双缓冲增加 shared memory 占用或寄存器占用可能降低 occupancy（一个 SM 能同时运行的 block/warp 数减少）。GEMM 通常 compute-bound + memory-bound 混合场景，用更多片上资源换延迟隐藏是**值得的**。经验法则：occupancy ≥ 25%（每个 SM 至少 1~2 个活跃 block）即可接受。

4. **验证正确性**：与单缓冲版本对比 `cuda-memcheck` 和数值误差（`max(|C_double - C_single|) / max(|C_single|)` 应在 1e-5 以内）。

5. **TL;DR 寄存器用量公式**：给定 TM, TN, BK：
   ```
   双缓冲寄存器数 ≈ (2×TM + 2×TN + TM×TN)   // reg_m, reg_n, Ct
   单缓冲寄存器数 ≈ (TM + TN + TM×TN)
   ```
   目标：双缓冲寄存器数 < 255，且编译后无 spilling。
