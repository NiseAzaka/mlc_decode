# 沐曦 FlashMLA 逆向分析（C500 / xcore1000）

仓库：`https://gitee.com/metax-maca/FlashMLA`（本地副本 `/data/flashmla_metax`，13 MB，MIT）
定位：deepseek-ai/FlashMLA 在 MACA 软件栈上的移植，基线是 **FlashAttention-2 v2.6.3**。
硬件：曦云 **C500 / C600**，要求 MXMACA ≥ 2.27。

---

## 1. 技术栈：mctlass（C++ 模板），不是 TileLang

| 项 | 结论 | 依据 |
| --- | --- | --- |
| 语言 | C 72.7% + C++ 26.6%，CUDA 0.1% | Gitee 语言统计 |
| 模板库 | **mctlass**（MetaX 版 CUTLASS） | `csrc/mctlass` 子模块 + `.gitmodules` |
| 编译器 | MACA clang（`MACA_CLANG_PATH=$MACA_PATH/mxgpu_llvm/bin`） | README |
| CUDA 兼容 | `cu-bridge`（`CUDA_PATH=$MACA_PATH/tools/cu-bridge`） | README |
| TileLang / Triton | ❌ 未使用（Triton 只作 PyTorch 依赖） | 目录与依赖均无 |

## 2. 代码组织

```text
csrc/
├── flash_api/           对外 API（flash_mla.h / flash_api.cpp）
├── flash_dispatch/      launch 模板：flash_fwd_{dispatch,launch,run}_template.h
├── flash_kernel/
│   ├── flash_dense_mla_decode_kernel.h
│   ├── flash_fwd_splitkv_mla_combine_kernel.h     ← combine
│   ├── feature/{mask,softmax}.h
│   ├── kernel_traits.h                            ← shared / tile / stage 账
│   └── xcore1000/                                 ← C500 的 5 个 kernel 变体
├── flash_run/           run_mla.h / run_mla_fwd.cpp
├── instantiations/      模板实例化（每个 shape 一个 .cu）
└── utils/
```

C500 的 kernel 变体（文件名即配置）：

```text
flash_fwd_mla_kernel_k64_16x16_4waves_stage1_xcore1000.h
flash_fwd_mla_kernel_k64_16x16_4waves_xcore1000.h
flash_fwd_mla_kernel_k64_32x16_4waves_xcore1000.h
flash_fwd_mla_kernel_k64_64x16_8waves_xcore1000.h
flash_fwd_sparse_mla_kernel_k64_64x16_8waves_xcore1000.h
```

## 3. 实例化参数（m16n16 这一档）

```cpp
// instantiations/xcore1000/..._m16n16_fp16_True_True_split_xcore1000.cu
template void run_flash_splitkv_fwd_mla_template<
    576,               // kHeadDim      (QK = 512 + 64)
    16,                // kBlockM       → 16 heads
    16,                // kBlockN       → 16 token
    4,                 // kNWarps       → 4 warps = 256 线程
    true,              // Is_Q_in_regs
    true,              // Share_Q_K_smem
    mctlass::half_t,
    true,              // Is_Splits
    512,               // kHeadDimV
    2                  // Num_Stages = 2  ← 双缓冲
>();
```

可选档位：`m16n16 / m32n16 / m64n16`（C500）、`m64n16 / m64n32`（C600），
kernel 侧对应 `4waves / 8waves`、`stage1 / 双 stage`。

## 4. 关键发现一：K 双 stage，V 单 stage

```cpp
// kernel_traits.h:185
using SmemLayoutK424 = tile_to_shape(SmemLayoutAtomK424{},
        Shape<Int<kBlockN>, Int<kHeadDim>, Int<Num_Stages>>{});   // K：带 Num_Stages

// kernel_traits.h:198
using SmemLayoutV = tile_to_shape(SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{});                    // V：没有 Num_Stages
```

即 **K 预取两块、V 只有一块**（V 在 softmax 之后才用，不需要提前预取）。

shared 账（`block_N=16`, fp16）：

| 项 | 计算 | 大小 |
| --- | --- | ---: |
| K（576 × 2 stage） | 16 × 576 × 2 × 2 B | **36.0 KB** |
| V（512 × 1 stage） | 16 × 512 × 1 × 2 B | **16.0 KB** |
| Q | 装载时借 K 的空间，之后复用 | 0 |
| O / LSE | 寄存器 | 0 |
| **合计** | | **52 KB** ✓ |

对比我们：`KV` 合并成一个 buffer（QK 与 PV 共用），`block_N=32` 时 `36 × 2 = 72 KB` ✗
—— 这是“64 KB 下开不了双缓冲”的真正原因，不是硬件限制，而是**K/V 没有拆开**。

## 5. 关键发现二：`Has_cp_async = false`，双缓冲靠寄存器中转

```cpp
// kernel_traits.h:21
static constexpr bool Has_cp_async = false;
```

C500 没有 `cp.async`（这也解释了为什么 TileLang 的 `num_stages=2` 生成 0 个 `cp.async`）。
但他们仍然做了双缓冲：

```cpp
int Ksmem_read_index = 0, Ksmem_write_index = 0;
Tensor tKrK = make_fragment_like(tKgK);              // ← 寄存器中转站

cute::copy(tKrK, tKsK(_, _, _, Ksmem_write_index));  // global→寄存器→shared[stage]
Ksmem_write_index ^= 1;                              // 切换写槽

gemm(acc_s, tSrQ, tSrK, tSsK(_, _, _, Ksmem_read_index), ...);   // MMA 读另一槽
Ksmem_read_index ^= 1;
```

要点：
- **寄存器是预取缓冲**：`ld.global` 先落到 `tKrK`，再 `st.shared`，搬运与 MMA 可以错开；
- `index ^= 1` 在 2 个 stage 间翻转，`tKsK(_,_,_,idx)` 的最后一维就是 `Num_Stages`；
- `1stage` 实例化是该逻辑的特例（索引恒 0）。

## 6. 关键发现三：Q 进寄存器的三段式路径

```cpp
// flash_fwd_mla_kernel_k64_16x16_4waves_xcore1000.h:316
Tensor tQrQ = make_fragment_like(tQgQ);
flash::copy_b64<...>(tQgQ, tQrQ, ...);   // ① global → 寄存器（b64 合并访存）
cute::copy(tQrQ, tQsQ);                   // ② 寄存器 → shared
if constexpr (Kernel_traits::Is_Q_in_regs) {
    flash::sync_threads();
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ);   // ③ shared → 寄存器（ldmatrix）
    flash::sync_threads();
}
```

我们的 `q_in_reg=True` 是**直接 `T.copy(Q, Q_frag)`**（global→fragment 散列，sector 利用率 25%），
官方是 **global→shared→ldmatrix→fragment**：先保证合并访存，再用 ldmatrix 完成布局转换。

并且第③步之后 `sQ` 空闲 —— 这正是 `Share_Q_K_smem` 成立的前提：

```cpp
kSmemSize = Share_Q_K_smem
          ? max(max(kSmemQSize, kSmemKSize), kSmemOSize)   // 复用
          : max(kSmemQSize + kSmemKSize, kSmemOSize);
```

## 7. 关键发现四：4 个 warp 如何分（QK 只有 16 列）

```cpp
static constexpr int kAtomLayoutMS = std::min(kBlockM / 16, kNWarps);   // = 1
```

M 方向只占 1 个 warp，而 QK 的 N=16 只有 1 个 16 列片 ⇒ 推断 **QK（`tiled_mma_s`）与 PV（`tiled_mma_o`）
使用两套不同的 warp 布局**，PV 的 N=512 被 4 个 warp 均分（每 warp 128 列）。

这正是 TileLang 卡死的地方：`T.gemm` 的一个 `policy` 必须同时满足两个 GEMM，
而 QK 需要 1 warp、PV 需要 4 warp，无法统一。

## 8. 代价：KV 流量约 1.9 倍

```text
我们：KV[block_N, 512] 一份，QK 与 PV 共用     → 每块读 576 维
他们：K[block_N, 576] + V[block_N, 512] 两份    → 每块读 1088 维（×1.89）
```

用流量换双缓冲：V 不参与 QK，所以只给 1 stage；K 需要连续预取，给 2 stage。

## 9. combine 侧

```cpp
// flash_fwd_splitkv_mla_combine_kernel.h:23
constexpr int kNThreads = kBlockM == 1 ? 64 : (kBlockM == 2 ? 128 : 256);
__shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];   // +1 避免 bank conflict
```

- 一个 CTA 处理 `kBlockM` 个 head（1/2/4/8/16/32），线程数随之 64/128/256；
- `num_splits` 来自 `params.num_splits_ptr[batch_idx]`（运行时逐序列算，不是 autotune 常量）；
- LSE 用 `kMaxSplits × (kBlockM+1)` 的定长 shared 数组承载。

## 10. 对 TileLang 路线的判定

| 要素 | 官方取值 | TileLang 能否表达 |
| --- | --- | --- |
| `block_N=16` | 是 | 能，但被 warp 约束卡在 1 warp |
| `kNWarps=4` | QK 1 warp / PV 4 warp | ❌ 需要两套 warp 布局 |
| Q 进寄存器 | global→shared→ldmatrix | ⚠️ 能写，但 `T.copy` 走散列路径 |
| K 2 stage | `Shape<..., Num_Stages>` | ❌ shared 布局要手写 |
| V 1 stage | 与 K 分开声明 | ❌ 我们 K/V 共用一个 buffer |
| shared 复用 | `max(Q,K,O)` | ❌ 静态分配，无法复用 |
| split 决策 | 运行时 `num_splits_ptr[]` | 我们靠 autotune 定值 |

⇒ **这套方案（tile 形状 + 4 warp + K/V 分离 stage + shared 复用）只有 mctlass 这类模板库能做**，
这也解释了官方为什么不用 TileLang。

## 11. `num_splits` 的决策公式（与我们的 autotune 交叉验证）

`flash_api.cpp:88`：

```cpp
num_splits = std::max(sm_count / h_k
                      / mctlass::ceil_div(num_q_tokens_per_head_k, block_m), 1);
```

其中 `num_q_tokens_per_head_k = s_q * (h_q / h_k)`（GQA 下每个 kv head 摊到的 q token 数），
`sm_count` = 104（C500 的 AP 数）。

**代入 OJ 场景**（b=1, h_q=16, h_k=1, s_q=1, block_m=16）：

```text
num_q_tokens_per_head_k = 1 × 16 = 16
ceil_div(16, 16) = 1
num_splits = max(104 / 1 / 1, 1) = 104
```

**正好是 104（1 波常驻）** —— 与我们 §2.2.6 实测的「16384 / 32768 在 `ns=104` 最优」
完全一致。这条公式的实质就是「让 split 数刚好填满 SM」，与我们的整波族思路同源。

差异在于：**公式没有考虑每 CTA 的块数（关键路径）**。对 64K（2048 块）来说，
104 splits 意味着每 CTA 19–20 块，实测 216 µs，而 256 splits（8 块）只要 204 µs。
所以该公式适合「batch 大、seqlen 中等」的推理场景，对超长 ctx 的 decode 不是最优 ——
这也是我们坚持逐 shape autotune 而不是套解析式的原因。

## 12. 实测对比（同一 OJ shape）

用预编译的 `flash_mla_cuda`（site-packages 里已有，直接调
`get_mla_metadata` + `flash_mla_with_kvcache`），shape 与 OJ 一致：
`b=1, s_q=1, h_q=16, h_kv=1, d=576, dv=512, block_size=64, fp16, causal=True`。
脚本：`handmade/bench_metax_flashmla.py`。

| kv_ctx | 官方 FlashMLA | 我们的 TileLang | 比值 | 谁快 |
| ---: | ---: | ---: | ---: | --- |
| 2048 | 41.68 | **22.34** | 1.87× | 我们 |
| 4096 | 51.31 | **29.61** | 1.73× | 我们 |
| 8192 | 64.13 | **44.43** | 1.44× | 我们 |
| 16384 | 76.11 | **68.43** | 1.11× | 我们 |
| 32768 | **98.07** | 114.20 | 0.86× | 官方快 14% |
| 65536 | **141.58** | 196.63 | 0.72× | 官方快 28% |

`cos = 1.000000`（2048/4096/8192 三档校验通过），官方 `num_splits = 64/86/103`
（长 ctx 收敛到 103 ≈ 104，与 §11 的公式一致）。

**三条结论**：

1. **短 ctx 我们全面领先（1.1–1.9×）**：官方为 paged KV / GQA / varlen 付了通用性开销，
   而我们的 kernel 只做 dense decode、combine 更轻；
2. **长 ctx 官方领先（32K −14%、64K −28%）**：这正是 §4/§5 那套「K 双 stage + V 单 stage +
   Q 进寄存器」的作用；64K 官方 141.58 µs 对应带宽 **567 GB/s（31%）**，我们只有 408 GB/s（22%）；
3. **逐档取两者最优**（22.34 / 29.61 / 44.43 / 68.43 / 98.07 / 141.58）总分 **97.37** ——
   仍远低于 98.3。说明 98.3 不是「把这个 FlashMLA 跑一遍」就能达到的，
   要么 OJ 的 $T_b$ 口径不同，要么对方用了别的实现/硬件条件。

## 13. 待分析项

1. `flash_dispatch/flash_fwd_launch_template.h` 里 `num_splits` 的决策公式（对比我们的 `_split_plan`）；
2. `flash_kernel/feature/softmax.h` 的 online softmax 实现（有无我们没用的技巧）；
3. `sparse_mla` 变体的 mask 组织（对 OJ 的 causal 场景是否有借鉴）。

---

参考：`docs/mla_decode_best_implementation.md` §12（手写 kernel 的工具链与 MMA 契约）。
