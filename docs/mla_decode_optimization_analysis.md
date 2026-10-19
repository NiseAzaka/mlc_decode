# MLA Decode 相对 baseline_mla 的优化分析与 mcProfiler 证据

> 分析对象：
> - 优化版 `mla_decode.py`
> - 基线 `baseline_mla.py`（同仓库、同一接口，作对照实验）
>
> 硬件口径：[`mxC500.md`](mxC500.md)（MetaX C500：104 AP、wavefront=64、SMEM 64KB、
> DRAM-L2 1843 GB/s、L2 8MB、VL1 32KB）。
>
> mcProfiler 证据（已归档，可直接复算）：
> - 优化版 `mcprofiler_output/mla_decode/mla_ours_16384/output20261016120455/`
> - 基线 `mcprofiler_output/mla_decode/mla_base_16384/output20261016121235/`
> - 对比结果 `mcprofiler_output/mla_decode/compare_16384_ours_vs_base.txt`
> - kernel 源码 `mcprofiler_output/mla_decode/kernels/{ours,baseline}_16384.cu`
>
> 脚本用法见 [`mla_decode_scripts_guide.md`](mla_decode_scripts_guide.md)；
> **只看结论**见 [`mla_decode_best_implementation.md`](mla_decode_best_implementation.md)（当前最优实现的最终结果）。

---

## 0. 结论速览

| 维度 | 结论 |
| --- | --- |
| 端到端加速 | `heads=16`（`test_mla.py` 配置，≈DeepSeek-V2-Lite）：**14.93×–53.27×**（autotune 选型，§1.1）；**`heads=128`（DeepSeek-V2/V3 真实配置）：5.41×–9.22×**（§1.4）。对外引用应以后者为准 |
| 根因 | 基线 grid 第一维 = `heads // block_H`。`heads=16` 时退化为 **(1,1)**、104 个 AP 只有 1 个在跑（**这是小 head 配置的假象**，§3.1）；`heads=128` 时为 **(8,1)**。真正的瓶颈是**单个 CTA 内串行扫完整条 KV 链**。优化版用 split-KV 把 grid 扩到 **(1, heads//block_H, num_split) + (heads,1,8)**，16384@h128 为 (1,8,64)+(128,1,8) |
| 机制 | 从「并行度饥饿 → 单 workgroup 内 MMA/共享内存串行（`wsm_stall` 39.4%）」变为「并行度充足 → 访存流水线为主的延迟隐藏（`vls_pipeline_stall` 89.5%）」 |
| 次要优化 | 共享内存 swizzle（冲突周期 1.32→0.76）、`coalesced_width=8` 向量化（`instruction per cycle` 9.68→16.05）、L2 光栅化（L2C 命中 4.68%→70.24%） |
| ⚠️ 已证伪 | **「Q 入寄存器」不是优化**：消融实测在 2048–65536 全档比 Q 入 shared **慢 13%–64%**，见 §4.5。生产路由 `kv_ctx>2048 → frag_q` 需要复核（已改为 autotune 自动选择，见 §7） |
| 当前瓶颈 | 16384 档有效 DRAM 带宽仅约 130–165 GB/s（峰值 1843 的 ~7–9%），**仍非带宽受限，而是延迟/占用率受限**，仍有继续提升空间 |

---

## 1. 端到端性能数据

### 1.1 `test_mla.py`：优化版 kernel vs 基线 kernel

同一组输入、预分配输出（Zero-Allocation），CUDA event 计时，`batch=1, heads=16, kv_heads=1, dim=512, pe_dim=64`。
「硬编码」= 原 `if/else` 参数 + 2Ki Q 放置规则；「autotune」= §7 的自动选型（**当前默认路径**）。

| kv_ctx | 基线 (µs) | 硬编码 (µs) | **autotune (µs)** | 硬编码 vsRef | **autotune vsRef** | autotune / 硬编码 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 362.28 | 24.17 | **24.19** | 14.99× | **14.93×** | 1.00× |
| 4096 | 724.02 | 43.28 | **31.73** | 16.73× | **22.80×** | **1.36×** |
| 8192 | 1483.18 | 88.93 | **47.03** | 16.68× | **31.51×** | **1.89×** |
| 16384 | 2953.57 | 144.20 | **74.01** | 20.48× | **39.91×** | **1.95×** |
| 32768 | 5900.71 | 244.25 | **140.10** | 24.16× | **42.11×** | **1.74×** |
| 65536 | 11783.93 | 353.17 | **221.15** | 33.37× | **53.27×** | **1.60×** |

> autotune 相对硬编码普遍再快 **1.6–1.95×**，来源有两个：一是 Q 放置由实测决定，
> 二是 `num_split` 不再按 `kv_ctx` 手拍（详见 §7）。

### 1.2 `check_mla.py`：优化版 vs 独立 PyTorch 参考（正确性 + 加速比）

| kv_ctx | 优化版 (µs) | PyTorch (µs) | vs Torch | max_abs_err | cos_sim |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 24.15 | 162.78 | 6.74× | 7e-6 | 1.000000 |
| 4096 | 31.81 | 142.35 | 4.48× | 5e-6 | 1.000000 |
| 8192 | 46.88 | 198.63 | 4.24× | 4e-6 | 1.000000 |
| 16384 | 73.86 | 329.05 | 4.45× | 2e-6 | 1.000000 |
| 32768 | 140.08 | 683.35 | 4.88× | 2e-6 | 1.000000 |
| 65536 | 220.47 | 1216.05 | 5.52× | 1e-6 | 1.000000 |

> 数值与 PyTorch（fp32 计算）对比全部 `cos_sim = 1.000000`，精度无回退。
> 附带修好一处旧问题：**32768/65536 在「硬编码」路径下 `cos` 只有 0.9376**
> （`num_split=312` 不整除 KV 长度，尾部 token 未参与计算）；
> autotune 的候选集强制 `num_split | seqlen_kv`，该问题随之消失（§7.4）。

### 1.3 有效带宽推算（说明「为什么还有空间」）

16384 档单次调用的**逻辑 DRAM 流量** ≈ KV `1×16384×1×512×2B = 16.78 MB`
+ K_pe `1×16384×1×64×2B = 2.10 MB` + Output `16 KB` ≈ **18.9 MB**（`output_partial`/`glse`
是中间量，实测 L2C 命中 70.24%，大部分不落 DRAM）。

加上 split-KV 每 split 重读 Q 的开销（autotune 在 16384 选中 `num_split=64`，
即 64 × (Q 16 KB + Q_pe 2 KB) ≈ 1.15 MB —— 这也是「split 越少越好」的原因之一）：

```text
有效带宽 ≈ (18.9 + 1.15) MB / 73.86 µs ≈ 271 GB/s   （峰值 1843 GB/s 的 ~14.7%）
```

⇒ **当前仍远未到带宽墙**，性能由并行度/延迟决定，还有继续提升的空间。

### 1.4 `heads=128`（DeepSeek 真实配置）基准

`test_mla.py` 用的是 `heads=16`，对应 DeepSeek-**V2-Lite** 的 head 数；真实 DeepSeek-V2 / V3 / R1
是 `num_attention_heads = num_key_value_heads = 128`，吸收（absorbed）成 kernel 形式后即
`heads=128, kv_heads=1`（group=128，Q 为 `[128, 512+64]`、KV latent 只剩 1 个头）。

用 `scripts/bench_mla_heads.py` 在同一计时口径下扫描（`batch=1, dim=512, pe_dim=64`，
warmup=10 / rep=50，候选网格 `chunk∈{64,128,256} × block_N=32 × q_in_reg∈{0,1}`，
每个候选都过 cos_sim 校验；`block_N=64` 因 `64×576×2 = 73728 B > 64 KB` 恒定不可用）：

| kv_ctx | h128 优化版 (µs) | h128 基线 (µs) | **h128 加速比** | h16 优化版 (µs) | h16 基线 (µs) | h16 加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 67.40 | 364.35 | **5.41×** | 26.21 | 364.14 | 13.89× |
| 4096 | 122.28 | 724.59 | **5.93×** | 31.58 | 725.46 | 22.97× |
| 8192 | 196.89 | 1468.80 | **7.46×** | 47.03 | 1481.07 | 31.49× |
| 16384 | 326.49 | 2932.37 | **8.98×** | 73.65 | 2966.29 | 40.27× |
| 32768 | 641.15 | 5852.95 | **9.13×** | 140.35 | 5895.02 | 42.00× |
| 65536 | 1268.72 | 11700.82 | **9.22×** | 220.13 | 11819.20 | 53.69× |

> 24 个 shape 的 `cos_sim` 均为 `1.00000`；原始日志 `bench_heads_128_vs_16.log`。
> h16 列与本文件 §1.1 的 autotune 列（24.19 / 31.73 / 47.03 / 74.01 / 140.10 / 221.15）吻合，
> 且 16384@h16 的 autotune 选型实测 73.63 µs ≈ 扫描最优 73.89 µs，互为印证。

三条必须记住的结论：

1. **基线时延与 head 数几乎无关。** 364.14 → 364.35 µs（2048）、11819 → 11701 µs（65536），
   差异 < 1%。因为基线 grid = `heads // block_H`，head 数翻 8 倍就多开 8 倍 CTA，
   而每个 CTA 的工作量完全不变（都串行扫完整条 KV 链）——**这是完美的弱扩展**。
   所以基线慢的根因是「单 CTA 内串行扫 KV」，不是「整卡并行度不足」。
2. **加速比随 head 数收敛。** `heads=16` 报 13.9×–53.7×，`heads=128` 只有 **5.4×–9.2×**。
   §3.1 的「grid 退化为单 workgroup」只在 `heads=16` 下成立（`16 // 16 = 1`）。
   **对外的性能数字应以 `heads=128` 为准**，`heads=16` 的倍数会显著高估。
3. **优化版自身扩展性良好。** head 数 ×8 只带来时延 ×2.57（2048）、×5.77（65536）的增长，
   且最优 `num_split` 随 ctx 单调增大（8→256），与 `heads=16` 下的趋势一致。

另外，扫描出的最优配置在**全部 12 个 shape 上都是 `q_in_reg=0`（Q 放 shared）**，
两个 head 数下均如此，与 §4.5 的消融结论一致：Q 入寄存器在任何 head 数下都不是优化。

---

## 2. 优化点清单

| # | 优化 | 基线做法 | 优化后做法（`mla_decode.py`） | 证据 |
| --- | --- | --- | --- | --- |
| O1 | **Split-KV + 二次归约** | `num_split=1`，永远走 `main_no_split`（`baseline_mla.py:195`）；grid = `(heads//block_H, 1)`，`heads=16` 时为 **(1,1)** | 按 shape 取 `num_split∈{64,128,256,312}`（`:303-335`），新增 combine kernel | grid `(1,1)`→`(1,8,64)+(128,1,8)`@16384/h128；端到端 **5.4×–9.2×（h128）/ 13.9×–53.7×（h16）**，见 §1.4 |
| O2 | Q 放置：shared vs 寄存器 | `Q_shared`/`Q_pe_shared`（`baseline_mla.py:118,120`） | 生产在 `kv_ctx>2048` 走 `flashattn_frag_q`（`Q_frag`/`Q_pe_frag`，`:187-188,220-221`）；`<=2048` 走 shared_q | ⚠️ **消融实验证伪：frag 反而更慢**（§4.5）。shared load 指令 5.22M→4.61M 的差异确实来自 Q 放置，但方向与预期相反 |
| O3 | **共享内存 swizzle** | 无 layout 注解 | `annotate_layout(make_swizzled_layout)` × KV/K_pe/S（`:56-60,199-203`） | shared 效率 70.35%→**81.08%**；冲突周期/指令 1.32→**0.76** |
| O4 | **`coalesced_width=8` 向量化** | 默认 copy 宽度 | 所有 global copy 加 `coalesced_width=8`（`:63-64,73-74,105,206-207,216-217,248`） | `MTE_16CYCLES` 占比 36.29%→**65.84%**；`instruction per cycle` 9.68→**16.05** |
| O5 | **L2 光栅化 `use_swizzle`** | 无 | `T.use_swizzle(panel_size=10, enable=True)`（`:61,204`） | `L2C Hit Rate` 4.68%→**70.24%**；load 平均延迟 47.81→**38.67** |
| O6 | **倒数替代逐元素除法** | `acc_o[i,j] /= logsum[i]`（`baseline_mla.py:160`） | `inv_logsum[i]=1/logsum[i]` 再乘（`:96-100,239-243`） | `average cycles per instruction` 4.45→**4.02** |
| O7 | **去掉每轮 `-inf` 填充** | `T.fill(scores_max,-inf)` + `reduce_max` + 显式 max（`baseline_mla.py:143-147`） | 直接 `reduce_max(clear=False)` 复用运行最大值（`:80-86`） | 每 KV 块省 1 次 fill + 1 次并行 max |
| O8 | **中间张量 padding** | 无中间张量 | `dim_pad=glse_pad=64` 打破 512B 对齐（`:18-23`） | VL1 partition stall 4/4 均非零（ours）；见 §4.3 说明 |
| O9 | **按 shape 调优 block/流水** | 固定 `block_N=32, num_stages=0` | 按 shape 选 `block_n/num_split/num_stages`（`:303-335`） | 见 O1；`num_stages` 0→1 打开流水 |
| O10 | **Combine 阶段 dim 分块 + 寄存器 FMA** | 无 combine | grid `(heads,batch,8)`，`T.copy→po_local` + 寄存器 FMA（`:107-144,250-290`） | combine 128 个 workgroup，每线程仅 64 个输出 |
| O11 | **Kernel 缓存** | 无（每次重编） | `_CACHE_SHARED`/`_CACHE_FRAG`（`:295-296,341-354`） | 降低 host 侧重 JIT 开销 |

---

## 3. 根因分析：基线为什么慢

### 3.1 基线 grid 在 `heads=16` 下只有 1 个 workgroup（**配置假象，非固有缺陷**）

导出两端 C-like kernel 源码（`mcprofiler_output/mla_decode/kernels/`，均以 `heads=16` 导出）：

```text
# baseline_16384.cu  —— 唯一 kernel，__launch_bounds__(128,1)
__global__ void __launch_bounds__(128, 1) main_no_split_kernel(...)
# 全文件 0 处 blockIdx 引用  →  grid = (1,1)

# ours_16384.cu  —— 两个 kernel
__global__ void __launch_bounds__(128, 1) main_kernel(...)    # 用 blockIdx.x / blockIdx.z
__global__ void __launch_bounds__(64,  1) main_kernel_1(...)  # 用 blockIdx.x
```

基线 `T.Kernel(heads // min(block_H, kv_group_num), batch)` = `(16//16, 1)` = **(1, 1)**：
`heads=16` 被 `block_H=16` 一次性吃满，`batch=1`，于是整卡只启动 1 个 workgroup（128 线程 = 2 wave），
104 个 AP 里只用 1 个。

优化版（同配置）：
- `main_kernel`：`T.Kernel(batch, heads//valid_block_h, num_split)` = `(1, 1, 256)` → 256 个 workgroup；
- `main_kernel_1`（combine）：`T.Kernel(heads, batch, dim_chunk_num)` = `(16, 1, 8)` → 128 个 workgroup；
- 合计 **384 个 workgroup**，足以铺满 104 个 AP。

#### ⚠️ 限定：`grid=(1,1)` 只在 `heads=16` 下成立

基线 grid 的第一维是 `heads // block_H`，**随 head 数线性增长**：

| heads | 基线 grid | workgroup 数（batch=1） |
| ---: | --- | ---: |
| 16（`test_mla.py` 配置） | (1, 1) | **1** |
| 128（DeepSeek-V2/V3 真实配置） | (8, 1) | **8** |

而且「源码里 0 处 `blockIdx`」是 **grid extent = 1 的结果**，不是独立证据：
TVM 在 grid extent 为 1 时把 `blockIdx` 常量折叠为 0，相关读写随之被消除。
`heads=128` 时 grid extent = 8，`blockIdx` 无法折叠，基线源码里出现 3 处 `blockIdx.x`：

```text
# baseline @ heads=128, ctx=16384
#   scripts/dump_kernel_src.py --heads 128 --impl baseline
*(uint4*)(Output + (blockIdx.x * 8192 + i_8 * 1024 + threadIdx.x * 8)) = ...
```

**后果**：基线的瓶颈是「单个 CTA 串行扫完整条 KV 链」，与 head 数无关，
所以它的端到端时延对 head 数几乎不变（§1.4 实测差 < 1%），
**加速比因此随 head 数上升而收敛**：`heads=16` 为 13.9×–53.7×，`heads=128` 只剩 **5.4×–9.2×**。

> 因此「15–33×」是小 head 配置下的数字。对真实 DeepSeek 配置应报 **5.4×–9.2×**（§1.4）。

### 3.2 基线单 workgroup 内部的瓶颈

> 口径说明：本节的 mcProfiler 数据采集于 `heads=16`（此时基线 grid = (1,1)，全卡只有 1 个 workgroup）。
> 它说明的是「单个 CTA 内部为什么慢」，与 §3.1 的限定不冲突——`heads=128` 下基线有 8 个 CTA，
> 但每个 CTA 的结构与瓶颈完全相同，只是总时延被并行摊掉（§1.4）。

| 指标（占比） | 基线 | 优化版 | 解读 |
| --- | ---: | ---: | --- |
| `wsm_stall`（wave/共享调度 stall） | **39.37%** | 7.10% | 基线单 workgroup 在共享/调度上串行等待 |
| `MMA_16CYCLES` 占比 | **37.22%** | 13.74% | 基线矩阵单元占比高但只用 1 个 AP |
| `AP MMA Duty ratio` | **7.13%** | 1.86% | 基线把单个 AP 的 MMA 压得很满，却浪费其余 103 个 |
| `vls_pipeline_stall` | 56.49% | **89.48%** | 优化版并行度足够后，瓶颈转移到访存流水线 |
| `MTE_16CYCLES` 占比 | 36.29% | **65.84%** | 优化版以宽向量访存为主（向量化生效） |
| `BSM`（共享内存指令）占比 | **17.30%** | 8.28% | 基线共享内存指令占比几乎翻倍 |

**结论**：基线 = 并行度饥饿 + 单 workgroup 内 MMA/共享内存串行；优化版 = 并行度充足 + 访存流水线主导。

---

## 4. mcProfiler 逐项证据（kv_ctx=16384）

> ### ⚠️ 归一化说明（重要）
>
> mcProfiler **会为每组 metric 重跑一遍 workload**，报告里的绝对计数器
> （`Total Cycles`、`AP active cycles`、`Global Memory bytes` 等）是**多轮 replay 的累加值**，
> 且按 AP 求和，两次运行之间**不能直接相减比较**。
> 例如优化版 `Total Cycles`(11.86M) 反而大于基线(9.71M)，而墙钟时间快 20×——
> 原因就是优化版用了 104 个 AP（`AP active cycles` 374.6M vs 97.6M），基线只用 1 个 AP。
>
> 因此下文**只用与归一化无关的比率 / 占比 / 单位延迟**作为证据；
> 绝对加速比以 `test_mla.py` 的 CUDA event 计时为准。

### 4.1 指令与发射效率

| 指标 | 基线 | 优化版 | 变化 |
| --- | ---: | ---: | --- |
| `instruction per cycle` | 9.68 | **16.05** | +66% |
| `instruction throughput efficiency` | 2.33% | **3.86%** | +66% |
| `instruction throughput` | 0.09 | **0.15** | +67% |
| `average cycles per instruction` | 4.45 | **4.02** | −9.7% |
| `average conflict cycles per instruction` | 1.32 | **0.76** | **−42%** |

→ 对应 O3（swizzle 显著降低冲突周期）与 O4/O6（更少更宽的操作）。

### 4.2 访存与缓存

| 指标 | 基线 | 优化版 | 解读 |
| --- | ---: | ---: | --- |
| `VL1 Hit Rate` | 88.29% | 87.88% | 基本持平 |
| `L2C Hit Rate` | 4.68% | **70.24%** | 对应 O5：光栅化 + split 中间量驻留 L2 |
| `average latency per load instruction` | 47.81 | **38.67** | −19%，局部性改善 |
| `average cycles per load instruction` | 4.22 | **3.71** | −12% |
| `load instructions`（shared load） | 5,222,800 | **4,608,000** | **−11.8%**，对应 O2（Q 入寄存器） |
| `shared memory access efficiency` | 70.35% | **81.08%** | 对应 O3 |
| `Dnoc Read Average Latency` | 206.78 | 250.99 | 优化版请求更多、并发更高，平均被拉长 |

split-KV 的代价（可从绝对值看出，但需归一化，仅作方向性说明）：
`Global Read Instructions` 0.50M→1.78M、`Global Write Instructions` 0.074M→0.288M、
`Global Memory Write bytes` 37.9MB→101.7MB —— 来自**每 split 重读 Q** 与**写中间张量
`output_partial`**。这部分被 L2C 70.24% 的命中率大幅吸收。

### 4.3 VL1 partition stall 分布

```text
=== VL1 partition stall 分布 ===
              优化版(ours)          基线(base)
pt0         1,664,788 (11.8%)           0 ( 0.0%)
pt1         4,672,848 (33.0%)           0 ( 0.0%)
pt2         2,064,956 (14.6%)           0 ( 0.0%)
pt3         5,742,507 (40.6%)           0 ( 0.0%)
```

优化版四个 partition 均有 stall（说明 4 路 VL1 partition 都在工作、且被 §O8 的 padding
打散到不同 partition）；基线四路全 0 —— 与「基线只有 1 个 workgroup，未对 VL1 partition
仲裁形成压力」一致，因此这一项**不能用来做前后对比**，只能说明优化版的多 workgroup 访问
确实铺满了 4 个 partition。

### 4.4 RoofLine

| 指标 | 基线 | 优化版 |
| --- | ---: | ---: |
| `case_I`（实测计算强度） | 258.65 | 238.66 |
| `MAX_I`（拐点） | 260 | 260 |
| `case_bandwith` | 164.97 GB/s | 171.46 GB/s |
| `case_flops` | 42.67 TFLOPS | 40.92 TFLOPS |

两版 `case_I` 都贴近拐点 `MAX_I=260`，说明 MLA Decode 在 C500 上处在计算/访存平衡点附近。
（注意：`case_bandwith` 是 profiler 依据其内部分析窗口折算的值，与 §1.3 的 163 GB/s 口径不同，
绝对值不宜直接引用，仅看「接近拐点」这一结论。）

### 4.5 消融实验：Q 放 shared 还是寄存器（结论：O2 被证伪）

`mla_decode.py` 原本按 `kv_ctx` 分流：`<=2048` 用 `Q_shared`（shared），`>2048` 用 `Q_frag`（寄存器）。
把两个 kernel 工厂放在**同一进程**里对照（同一份输入、预分配输出、warmup 30 / rep 200、
各测两遍取较小值；`block_n=32, block_h=16, dim_chunk_num=8`，`num_split` 取生产值）：

| kv_ctx | num_split | loop_range | `flashattn_shared_q` (µs) | `flashattn_frag_q` (µs) | frag/shared |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 64 | 1 | **23.86** | 31.50 | 1.32× |
| 4096 | 64 | 2 | **31.41** | 43.09 | 1.37× |
| 8192 | 128 | 2 | **62.01** | 89.62 | 1.45× |
| 16384 | 256 | 2 | **103.64** | 143.47 | 1.38× |
| 32768 | 312 | 4 | **214.86** | 243.75 | 1.13× |
| 65536 | 312 | 7 | **215.20** | 352.31 | 1.64× |

两种放置的数值结果完全一致（`max_abs_err≈7e-6`、`cos=1.0`），且**生产 `_get_kernel` 的实测延迟
在 `<=2048` 命中 shared_q、`>2048` 命中 frag_q** —— 说明差异确实只来自 Q 放置。
（两个工厂源码 diff 只差 Q 放置 + combine 里 `glse` 的装载写法，后者量级 < 5 µs，解释不了 40 µs 的差距。）

**结论：Q 入寄存器在本硬件 + 本版 TileLang 上是劣化，不是优化。**

**机制：Q 的 global→fragment 装载破坏了合并访存。** 对比两端生成的源码：

```c
// shared_q：global → shared，uint4（16B/线程），128 线程连续覆盖 2048B
*(uint4*)(((half_t*)Q_shared)+...) = *(uint4*)(Q + ((i*1024) + ((int)threadIdx.x * 8)));

// frag_q：global → register，uint2（8B/线程），同 wavefront 内相邻线程步长 512 half = 1KB
half_t Q_frag[128];
*(uint2*)(Q_frag + (i*4)) = *(uint2*)(Q + ((((int)threadIdx.x & 15) * 512)
                                 + (i*16) + (((((int)threadIdx.x & 63) >> 4) * 4)));
```

- `shared_q`：线程 `t` 读 `Q[i*1024 + t*8]`，一个 wavefront（64 线程）覆盖连续地址、
  sector 利用率 100%；
- `frag_q`：地址含 `(tid & 15) * 512`，同 wavefront 内相邻线程地址相差 **1 KB**，
  每线程 8 B 落在不同的 32 B sector → **sector 利用率约 25%（读放大 ~4×）**，
  向量宽度也从 `uint4` 掉到 `uint2`。

根源是 MMA 的 A-operand 寄存器布局决定了「哪个线程持有哪几个元素」，直接 global→fragment
只能生成这种散列 pattern；而 global→shared 的拷贝布局是按合并访问挑选的，之后再用廉价的
shared→register 喂给 MMA。

叠加的次要因素：

1. 寄存器压力：`half_t Q_frag[128]` ≈ 64 个寄存器/线程，而 `acc_o`（`[16,512]` fp32）已占 64 个/线程；
2. **2048 是退化档**：`chunk_size = block_N = 32` ⇒ 每个 split 的 KV 长度恰为 32，
   内层 `T.Pipelined` 的 `loop_range = ceildiv(2048//64, 32) = 1`，**只有一次迭代**，
   既无跨迭代重叠、也没有 Q 复用可摊薄这次未合并装载；kernel 只有 ~24 µs，固定开销占比大。
   （代码在 `:312` 的 4096 分支写明了这条规则：`chunk_size` 需 `>= 2*block_n`，否则流水线失效。）

所以「2Ki 边界」实际是 `chunk_size` 跌到 `block_n` 的退化位置。

> **⇒ 后续处置**：不再用 `if/else` 硬编码 Q 放置，交给 autotune 自动选择，见 §7。

### 4.6 澄清：「Q 放 shared 才能开软件流水线」不成立

一个常见的误解是「把 Q 放进 shared memory 才能开启软件流水线」。实测不支持这个说法，
**Q 放置与流水线是正交的两件事**，证据如下。

**（1）代码层面完全相同。** 两个工厂里的流水线语句逐字相同：

```python
loop_range = T.ceildiv(seqlen_kv // num_split, block_N)
for k in T.Pipelined(loop_range, num_stages=num_stages):
```

唯一的差别是 `Q_shared/Q_pe_shared` 与 `Q_frag/Q_pe_frag`（§4.5 的源码 diff 已确认）。

**（2）生成代码层面两者都被流水线化。** 在 `16384, num_split=64, block_N=32, num_stages=1` 下导出源码，
两者的结构一致：主循环前都 peel 了一段 prologue（先把第 0 拍的 `KV`/`K_pe` 抄进 shared），
主循环都是 `for (k = 0; k < 7; ++k)`（= `loop_range - 1`），`__syncthreads` 都是 12 次。

**（3）`num_stages` 的上限不由 Q 决定，而由 KV tile 尺寸决定。**

| 变体 | `num_stages=0` | `num_stages=1` | 2 | 3 |
| --- | --- | --- | --- | --- |
| `shared_q` | 70.83 µs（主循环 8 次，6 次 sync） | 73.59 µs（主循环 7 次，12 次 sync） | `mcErrorInvalidValue` | `mcErrorInvalidValue` |
| `frag_q` | 123.52 µs（主循环 8 次，6 次 sync） | 122.61 µs（主循环 7 次，12 次 sync） | `mcErrorInvalidValue` | `mcErrorInvalidValue` |

要开第 2 级，shared 至少要多放一份 `KV + K_pe`：
`block_N×(dim+pe_dim)×2 = 32×576×2 = 36864 B`，两级就是 **73728 B > 65536 B（C500 上限）**——
**即使 Q 完全不占 shared 也放不下**。所以本 shape 下 `num_stages` 被锁死在 1，与 Q 放哪里无关。

**（4）方向甚至是相反的。** Q 放 shared 会额外占用 `block_H×(dim+pe_dim)×2 = 18432 B`，
更早撞上 64 KB 上限（实测需求：`shared_q` ns=2 ≈ **93.7 KB**，`frag_q` ns=2 ≈ **75.3 KB**，
frag 离上限更近）。也就是说若真要在共享内存预算上做文章，**Q 放寄存器才是「更省 shared」的那一侧**。

**（5）`num_stages=1` 本身并没有收益。** 相对 `num_stages=0`，它只是多了一次 peel prologue
与 6 次 `__syncthreads`：`shared_q` 70.83 → 73.59 µs（略慢）、`frag_q` 基本持平。
这也解释了为什么 autotune 的候选集里 `num_stages=2` 会被 shared 上限过滤掉，只剩 1。

**结论**：真正决定流水线能否生效的是

1. `loop_range = chunk / block_N`（≥2 才有可重叠的迭代）—— 代码 `:312` 的注释写的正是这条：
   *"chunk_size 需 >= 2*block_n，否则每个 split 只剩一次循环，流水线失效"*；
2. shared 预算（每级多花一份 KV tile，而 KV tile 本身就占了上限的一大半）。

两者都与 Q 放哪里无关。

### 4.7 追问：为什么「放弃 2 级流水线、换回 shared」反而更快

先修正前提：**在本 shape（`dim=512`）上，2 级流水线对两种 Q 放置都开不起来。**

实测（`ctx=16384, num_split=64, block_N=32`，动态 shared 需求按 `Q + ns×(KV+K_pe) + S` 折算）：

| 变体 | `num_stages=0` | `num_stages=1` | `num_stages=2` |
| --- | --- | --- | --- |
| `shared_q` | 70.85 µs（需 19 KB） | 73.71 µs（需 55 KB） | **FAIL**（需 **91 KB** > 64 KB） |
| `frag_q` | 123.67 µs（需 1 KB） | 122.67 µs（需 37 KB） | **FAIL**（需 **73 KB** > 64 KB） |

两级 KV 本身就 `2 × 32×576×2 = 73728 B > 65536 B` —— **Q 完全不占 shared 也放不下**。
这与代码注释一致（`:328`）：*"num_stages 降为 1 以压住 shared memory（C500 上限 64KB）"*。
（`baseline_mla.py:51` 的 `main_split` 用的是 `num_stages=2`，「2 级流水线」的印象大概来自那里；
但那条路径在 C500 上同样跑不起来。）

那如果换到**开得下**的配置，2 级流水线到底值多少？把 `dim` 降到 256（两种放置都能开 ns=2）实测：

| 变体 | `num_stages=0` | `num_stages=1` | `num_stages=2` |
| --- | --- | --- | --- |
| `shared_q` | 47.25 µs | 48.85 µs | **53.66 µs（比 ns=0 慢 13.6%）** |
| `frag_q` | 48.20 µs | 50.01 µs | **57.04 µs（比 ns=0 慢 18.3%）** |

**即使能开，2 级流水线在这个 kernel 上也是负收益**，且对两种 Q 放置都一样。原因：
MLA Decode 的延迟隐藏主要来自 **split-KV 的几百个 workgroup 级并发**（§3），
而不是单 workgroup 内的双缓冲；多开一级只增加 shared 占用（压低占用率）
与同步次数（`ns=1` 已把 `__syncthreads` 从 6 次抬到 12 次），换不来额外重叠。

**所以这两个效应是独立的，量级还差一个数量级**（16384 档）：

| 效应 | 量级 | 性质 |
| --- | --- | --- |
| Q 放置（frag → shared） | **1.39×**（143.91 → 103.88 µs） | 一阶，正收益 |
| `num_split` 重调 | **1.42×**（103.88 → 73.17 µs） | 一阶，正收益 |
| `num_stages` 0 → 1 → 2 | 0 … **−18%** | 二阶，无收益甚至负收益 |

⇒ 「Q in register 可以开 2 级流水线」在 **C500 + `dim=512`** 下不成立；即使在共享内存更大的机器上
成立（`frag_q` 的 ns=2 只差 9 KB），流水线那点二阶收益也远不足以抵掉 Q 入寄存器带来的一阶损失
（未合并访存、读放大 ~4×，§4.5）。这就是「放弃流水线、换回 shared」反而更快的原因。

---

## 5. 瓶颈判断与后续空间

1. **已解决（本轮 20× 的来源）**：并行度饥饿。基线 1 workgroup → 优化版 384 workgroup，
   把 104 个 AP 用起来。
2. **当前瓶颈**：有效 DRAM 带宽仅约 130–165 GB/s（峰值 ~9%），`vls_pipeline_stall` 89.5%。
   既不是带宽墙，也不是 MMA 墙（`AP MMA Duty` 仅 1.86%），更可能是
   **workgroup 数仍偏少 / 尾效应 / 访存延迟未被完全隐藏**。
3. **可继续尝试**：
   - 提高 `num_split`（当前 16384 用 256、32768/65536 用 312），或让 combine 更快，压低尾部；
   - 减少 split 带来的 Q 重读（把 Q 一次性广播/驻留，或让一个 workgroup 处理多个 split）；
   - 提高单 block 的 wave 数（`threads` 或每 workgroup 处理的 head 数），提高占用率；
   - 用 `compare_reports.py` 复采，盯 `vls_pipeline_stall`、`L2C Hit Rate`、`instruction per cycle` 三项。
4. **代码层面的两处修正**（本轮提出）：
   - **Q 放置不再硬编码**：原 `kv_ctx>2048 → frag_q` 的分流在实测中劣化 13%–64%（§4.5），
     已改为 autotune 自动选择 `q_in_reg`，见 §7；
   - **`num_split` 与 KV 长度的整除关系**：`num_split=312` 在 32768/65536 上不能整除
     （312×105=32760 < 32768、312×210=65520 < 65536），尾部 token 未被处理，
     实测这两档 `cos` 仅 0.9376。**本轮不改**，留作后续单独处理。

---

## 6. 复现命令

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/mla_decode:$PYTHONPATH
cd /data/mla_decode

# 端到端加速比
python test_mla.py
# 正确性 + vs PyTorch
python scripts/check_mla.py --kv-ctx 2048 4096 8192 16384 32768 65536

# mcProfiler：优化版 / 基线各采一份（约 8–10 min/份），自动归档到 mcprofiler_output/
bash scripts/run_mcprofiler.sh 16384 0 mla_ours_16384
CMD_ARGS="16384 5 20 baseline" bash scripts/run_mcprofiler.sh 16384 0 mla_base_16384

# 解读与对比
python scripts/summarize_report.py mcprofiler_output/mla_decode/mla_ours_16384/output20261016120455
python scripts/compare_reports.py mcprofiler_output/mla_decode/mla_ours_16384/output20261016120455 \
                                  mcprofiler_output/mla_decode/mla_base_16384/output20261016121235

# kernel 源码（核查 grid / 向量宽度）
python scripts/dump_kernel_src.py --kv-ctx 16384 --impl ours     --out mcprofiler_output/mla_decode/kernels/ours_16384.cu
python scripts/dump_kernel_src.py --kv-ctx 16384 --impl baseline --out mcprofiler_output/mla_decode/kernels/baseline_16384.cu
```

采集环境：MetaX C500（整卡，无 sGPU 切片），Driver 3.8.30，MACA 3.7.1.5，
PyTorch 2.8.0+metax3.7.1.3，TileLang 0.1.10（MACA，`/opt/tilelang-metax-v0.1.10`），
mcProfiler 3.8.1.4（本地模式）。

---

## 7. 用 autotune 替代 `if/else` 选型

### 7.1 改了什么

原实现把「Q 放置」和「分片参数」写死在 `_get_kernel` 的 `if kv_ctx == ...` 里。
实测表明其中 Q 放置的选择是错的（§4.5），`num_split` 也远非最优（16384 手拍 256，实测 64 更好）。
现在改为由 TileLang autotune 逐 shape 实测选择：

| 原 `if/else` 硬编码 | 现状 |
| --- | --- |
| `block_n`（32） | 进入候选集，按 shared 上限自动裁剪 |
| `num_split`（64/64/128/256/312/312） | 进入候选集（由 `chunk` 导出），实测选择 |
| `num_stages`（1） | 进入候选集（1/2），按 shared 上限自动裁剪 |
| Q 放置（`<=2048` shared，`>2048` frag） | 参数 `q_in_reg`，实测选择 |
| **主 kernel `threads`（写死 128）** | **参数 `threads`（64/128/256），按 warp 映射约束裁剪，实测选择** |
| `dim_chunk_num`（8） | 仍为固定值（未纳入搜索） |
| combine kernel `threads`（写死 64） | 仍为固定值（`dim_per_chunk = dim/dim_chunk_num = 64`，正好 1 元素/线程） |

代码结构（`mla_decode.py`）：

```text
mla_autotune_configs(...)   # 按 shape 生成合法候选（整除约束 + shared 上限）
flashattn_autotuned(...)    # @autotune + @tilelang.jit 的统一入口
_get_kernel(...)            # 默认走 autotune；失败或 MLA_NO_AUTOTUNE=1 时回退
_legacy_kernel(...)         # 原 if/else 路径，保留为回退
```

### 7.2 候选空间是怎么裁的

`mla_autotune_configs` 只保留同时满足以下条件的组合：

1. `chunk = seqlen_kv / num_split` 必须整除（即 `num_split | seqlen_kv`）且 `chunk % block_N == 0`
   —— 保证每个 split 都是整块，既不越界也不漏算尾部；
2. 主 kernel 的动态 shared（`Q + num_stages×(KV+K_pe) + S`）不超过 C500 的 **64 KB / AP**；
3. `threads` 满足 warp 映射约束 `block_N >= 16 × (threads / 64)`
   —— `acc_s[block_H, block_N]` 用 `FullCol` 策略按 N 切给 `threads/64` 个 warp，
   每个 warp 至少要吃下一个 16 宽 MMA tile。这条也解释了源码注释里
   *"threads=128 需要 >=32，否则 16x16 MMA tile 无法映射 4 个 warp"*。

在 `dim=512, pe_dim=64, block_H=16` 下：

- 第 2 条把 `block_N=64`、`num_stages=2` 全部剪掉；
- 第 3 条把 `threads=256` 剪掉（它要求 `block_N>=64`，而 `block_N=64` 已被第 2 条剪掉）
  —— 这条很关键：**`block_N=32, threads=256` 会触发 MACA 版 TileLang 的内部错误**
  `Check failed: pb->value != 0 ... Divide by zero`，靠规则提前剪掉而不是等编译崩；

最终剩 `block_N=32, num_stages=1` × `chunk∈{32,64,128,256}` × `q_in_reg∈{False,True}`
× `threads∈{64,128}` = **16 个候选**（65536 档因 `num_split>1024` 少一组，为 12 个）。
（`chunk∈{512,1024}` 也验证过，各档都更慢，故未纳入。）

### 7.3 autotune 在各 ctx 的选择情况

`batch=1, heads=16, dim=512, pe_dim=64`。下表为**稳态计时**（预分配输出、warmup 20 / rep 100、
连测两遍取小值），每格是一个候选配置；**粗体**为 autotune 选中的配置，`✅` 表示它同时是稳态最小值
（6 档全部命中，说明 autotune 的计时口径与稳态口径排序一致）。
每个候选的 `block_N`/`num_stages` 由 §7.2 的 shared 上限自动确定为 `32 / 1`。

**ctx = 2048** — 选中 `num_split=64, q_in_reg=False`，**23.77 µs** ✅

| chunk | num_split | `q_in_reg=False` | `q_in_reg=True` |
| ---: | ---: | ---: | ---: |
| 32 | 64 | **23.77** ✅ | 31.50 |
| 64 | 32 | 25.88 | 35.57 |
| 128 | 16 | 35.67 | 51.08 |
| 256 | 8 | 57.55 | 84.82 |

**ctx = 4096** — 选中 `num_split=64, q_in_reg=False`，**31.30 µs** ✅

| chunk | num_split | `q_in_reg=False` | `q_in_reg=True` |
| ---: | ---: | ---: | ---: |
| 32 | 128 | 46.47 | 64.04 |
| 64 | 64 | **31.30** ✅ | 43.02 |
| 128 | 32 | 39.06 | 57.18 |
| 256 | 16 | 60.72 | 92.40 |

**ctx = 8192** — 选中 `num_split=64, q_in_reg=False`，**46.85 µs** ✅

| chunk | num_split | `q_in_reg=False` | `q_in_reg=True` |
| ---: | ---: | ---: | ---: |
| 32 | 256 | 80.76 | 106.17 |
| 64 | 128 | 62.20 | 89.45 |
| 128 | 64 | **46.85** ✅ | 71.86 |
| 256 | 32 | 66.55 | 113.56 |

**ctx = 16384** — 选中 `num_split=64, q_in_reg=False`，**73.17 µs** ✅

| chunk | num_split | `q_in_reg=False` | `q_in_reg=True` |
| ---: | ---: | ---: | ---: |
| 32 | 512 | 147.17 | 188.17 |
| 64 | 256 | **73.17** ✅ | 122.47 |
| 128 | 128 | 89.01 | 139.30 |
| 256 | 64 | 103.88 | 143.91 |

**ctx = 32768** — 选中 `num_split=128, q_in_reg=False`，**139.80 µs** ✅

| chunk | num_split | `q_in_reg=False` | `q_in_reg=True` |
| ---: | ---: | ---: | ---: |
| 32 | 1024 | 289.16 | 366.65 |
| 64 | 512 | 185.89 | 251.55 |
| 128 | 256 | **139.80** ✅ | 237.72 |
| 256 | 128 | 142.57 | 220.32 |

**ctx = 65536** — 选中 `num_split=256, q_in_reg=False`，**220.21 µs** ✅
（`chunk=32` 对应 `num_split=2048`，超过候选集上限 1024 被排除）

| chunk | num_split | `q_in_reg=False` | `q_in_reg=True` |
| ---: | ---: | ---: | ---: |
| 64 | 1024 | 364.94 | 491.70 |
| 128 | 512 | 249.79 | 381.56 |
| 256 | 256 | **220.21** ✅ | 369.34 |

**汇总**

| kv_ctx | 选中 num_split（chunk） | 选中 `q_in_reg` | 选中 `threads` | 延迟 (µs) | 硬编码延迟 (µs) | 提升 |
| ---: | ---: | :---: | :---: | ---: | ---: | ---: |
| 2048 | 64（32） | False | 128 | 23.77 | 24.17 | 1.02× |
| 4096 | 64（64） | False | 128 | 31.30 | 43.28 | 1.38× |
| 8192 | 64（128） | False | 128 | 46.85 | 88.93 | 1.90× |
| 16384 | 64（256） | False | 128 | 73.17 | 144.20 | 1.97× |
| 32768 | 128（256） | False | 128 | 139.80 | 244.25 | 1.75× |
| 65536 | 256（256） | False | 128 | 220.21 | 353.17 | 1.60× |

> 加入 `threads` 候选后重跑一次，6 档选择的仍是 `threads=128`，端到端延迟与加 `threads` 前一致
> （`test_mla.py`：24.19 / 31.84 / 47.00 / 73.88 / 140.54 / 220.60 µs）。`threads` 的
> 消融数据见 §7.8。

三点结论：

1. **`q_in_reg=False`（Q 放 shared）在全部 48 个比较中都胜出**（同 chunk 下 frag/shared ≈ 1.2–1.7×），
   与 §4.5 的消融完全一致 —— autotune 独立复现了「Q 入寄存器是劣化」这个结论，
   `>2048 → frag` 的分流被干净替换掉；
2. **`num_split` 比手拍值小得多**（16384：64 vs 256）。手拍值只按 chunk 拍，没有把 split 带来的
   Q 重读与 combine 开销算进去；最优 chunk 落在 128–256 而不是最小的 32；
3. **`num_split` 随 ctx 单调增长**（2048–16384 都是 64，32768 是 128，65536 是 256）——
   即「在能把机器喂饱的前提下尽量少切」，同时 `chunk` 大致被顶在 32–256 区间。

### 7.4 顺带修好的问题

`num_split=312` 不整除 32768/65536（312×105=32760、312×210=65520），尾部 token 未参与计算，
旧路径这两档 `cos` 只有 **0.9376**。autotune 的候选集强制 `num_split | seqlen_kv` 后，
全档 `cos = 1.000000`。

> ⚠️ 这是**候选集约束的副产品**：split 的尾部处理逻辑本身没改，
> 非整除 shape 的通用修法仍是待办（见 §5.4）。

### 7.5 使用方式

```python
from mla_decode import _get_kernel

kernel = _get_kernel(1, 16, 1, 16384, 512, 64)   # 首次 autotune 并缓存，之后直接命中
```

- 结果按 `(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim)` 缓存
  （进程内 `_TUNE_CACHE` + TileLang 磁盘缓存），同 shape 第二次调用约 0.1 s；
- **旁路 autotune**（用于对照/排错）：设 `MLA_NO_AUTOTUNE=1` 走原 `if/else`；
  或显式传参调用 `flashattn_autotuned(..., block_N=32, num_split=256, q_in_reg=False)`
  —— TileLang 只要发现可调参数已被显式提供就会跳过调优直接编译；
- autotune 抛异常时 `_get_kernel` 会 `try/except` 回退到 `_legacy_kernel`，不会让调用方崩掉。

### 7.6 已知限制

- 候选集的 `chunk` 上限为 256，故 `num_split` 下界是 `seqlen_kv/256`；超长 ctx（如 128K）
  候选会偏少，需要相应放宽；
- `dim_chunk_num`（combine 阶段的 dim 分块数）仍固定为 8，未纳入搜索；
  同理 combine kernel 的 `threads=64` 也没进搜索（`dim_per_chunk = dim/dim_chunk_num = 64`，
  正好 1 元素/线程，调它需要和 `dim_chunk_num` 联动）；
- `threads` 在 C500 + 本 shape 下没有可调空间（见 §7.8），加入该轴只增加编译时间；
- autotune 的计时协议与 `test_mla.py` 的稳态计时不同（它用自己的 supply 输入、不做 L2 预热），
  **选型结论一致但绝对数值偏大约 1.6×**；
- 首次 autotune 需要编译全部候选：16 个/档（65536 档 12 个），单 shape 约 2–4 min（之后走缓存）。

---

### 7.7 收益归因：Q 放置 / num_split / 流水线各占多少

把 §1.1 的「硬编码 → autotune」总收益拆成两个因子（每个都在**固定 `num_split`** 下对比，
因此是条件效应；两点之间按顺序分解，不严格正交）：

- **Q 因子**：把 legacy 的 Q 放置换成 shared，`num_split` 保持 legacy 值；
- **num_split 因子**：Q 换成 shared 后，再把 `num_split` 换成 autotune 的选中值。

| kv_ctx | legacy 实现 | legacy (µs) | ① 换 Q→shared (µs) | ② 再调 num_split (µs) | Q 因子 | num_split 因子 | 合计 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | shared, ns=64 | 24.17 | 23.77 | 23.77（ns=64） | 1.00× | 1.00× | 1.02× |
| 4096 | frag, ns=64 | 43.02 | 31.30 | 31.30（ns=64） | **1.37×** | 1.00× | 1.38× |
| 8192 | frag, ns=128 | 89.45 | 62.20 | 46.85（ns=64） | **1.44×** | **1.33×** | 1.91× |
| 16384 | frag, ns=256 | 143.91 | 103.88 | 73.17（ns=64） | **1.39×** | **1.42×** | 1.97× |
| 32768 | frag, ns=312 | 243.87 | 214.18 | 139.86（ns=128） | 1.14× | **1.53×** | 1.74× |
| 65536 | frag, ns=312 | 351.97 | 215.03 | 220.33（ns=256） | **1.64×** | 0.98× | 1.60× |

> 说明：
> - legacy 列与 §1.1 的 `test_mla.py` 结果一致（如 16384：143.91 vs 144.20），
>   2048 之所以没有 Q 因子，是因为原 `if/else` 在 `<=2048` 本来就走 shared；
> - 32768/65536 的 legacy 用了 `num_split=312`，该值**不整除 KV 长度**（§7.4），
>   它实际会多读一段重叠数据又漏掉尾部，因此只作基线，不参与“最优”判断。

**结论：**

1. **Q 放置是主因子，但它“抵消”的是一处劣化而不是一处收益。**
   原 `>2048 → frag_q` 在全档都更慢（§4.5），所以改回 shared 在 4096–65536 上直接拿回
   **1.37×–1.64×**；2048 本来就走 shared，所以这一因子为 1.00×。
2. **`num_split` 是与 Q 正交的第二个收益来源，大 ctx 上依然有额外收益。**
   在 Q 已是 shared 的前提下，仅调 `num_split` 还能再拿 **16384: 1.42×**、
   **32768: 1.53×**、8192: 1.33×；
   但 **65536 上它没有额外收益（0.98×，略负）** —— 那一档的收益几乎全部来自 Q 放置
   （选定 `num_split=256` 与 legacy 的 312 在时间上基本等价，但 312 是错的，不能选）。
3. **流水线没有被“抵消”，因为它本来就没有收益。**
   legacy 与 autotune 都使用 `num_stages=1`（`num_stages=2` 因 shared 超限被候选集过滤，§4.6），
   而 `num_stages=1` 相对 `0` 实测无收益（§4.6 第 5 点：16384 下 shared 70.83 → 73.59 µs，略慢）。
   即所谓「Q in register / 流水线」两项里：**前者是负收益被移除，后者是 0 收益被原样保留**，
   两者都没有需要“补偿”的正收益。

### 7.8 `threads` 作为可调参数：结论与数据

主 kernel 的 `threads` 原先写死 128，现已参数化并进入候选集
（`_TUNE_THREADS = (64, 128, 256)`，经 `_threads_feasible` 的 warp 映射规则裁剪后实际剩 `{64, 128}`）。

**动机**：`acc_o` 是 `[block_H=16, dim=512]` fp32 = **32 KB/CTA**，在 128 线程下相当于
`8192/128 = 64` 个寄存器/线程；加到 256 线程理论上能减半到 32 个，缓解寄存器压力。

**实测**（`shared_q`，`block_N=32`，`num_split` 取该档选中值）：

| kv_ctx | num_split | `threads=64` | **`threads=128`** | `threads=256` |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 64 | 33.07 µs | **23.88 µs** | — |
| 8192 | 64 | 151.31 µs | **46.61 µs** | — |
| 16384 | 64 | 288.72 µs | **73.36 µs** | FAIL（需 `block_N≥64`） |
| 65536 | 256 | 808.59 µs | **220.01 µs** | — |

（16384 档 `frag_q` 同向：287.92 vs 122.48 µs。）

**机制：每线程寄存器翻倍 → 占用率减半（直接取自生成代码）**

`block` 的工作量不变、线程数减半 ⇒ 每个线程承担的元素数翻倍。生成源码里主 kernel 的
每线程本地数组（`threads=64/128`，`ctx=16384, num_split=64`）：

| `threads` | `acc_o` | `acc_s` | MMA B 操作数（×2 处） | 每线程静态寄存器合计 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | `float[64]` = 64 | `float[4]` = 4 | `half_t[64]` = 32 ×2 | **166 reg** |
| 64 | `float[128]` = 128 | `float[8]` = 8 | `half_t[128]` = 64 ×2 | **306 reg** |

即 **×1.84**。注意 306 < 512 的每线程寄存器上限，所以**不是溢出，是占用率**——
关键在于寄存器堆是**每个 AP 共享**的固定资源（C500：`4 PEU × 512 reg × 64 thread = 131072 reg = 512 KB`，
`Max waves/AP = 32`），可驻留线程数 = `131072 / R`：

| `threads` | 每线程 R | 可驻留线程/AP | 可驻留 wave/AP | 占用率 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | ~166 | ~789 | ~12 | ~37% |
| 64 | ~306 | ~428 | ~6 | ~19% |

⇒ 占用率大致减半，这正是"寄存器翻倍"的直接后果
（不是"总寄存器占用变多"：单位工作量的总寄存器占用是守恒的）。

**但实测差距 3.10–3.93× 大于占用率减半预测的 2×**，还有第二个因素：`threads=64` 时
每个 CTA 只有 **1 个 wave**，CTA 内部没有兄弟 wave 可切换来掩盖延迟，单 wave 的依赖链也被拉长
（`acc_o` 的 `FullCol` 侧从 2 wave × 32 块变成 1 wave × 64 块）。

"单纯加并行度救不回来"也印证了瓶颈在**每线程效率**而非总 wave 数：

| 配置 | CTA 数 | 总线程 | 延迟 |
| --- | ---: | ---: | ---: |
| th=64, ns=64 | 64 | 4,096 | 288.89 µs |
| th=64, ns=128 | 128 | 8,192 | 275.81 µs |
| th=64, ns=256 | 256 | 16,384 | 205.26 µs |
| **th=128, ns=64** | 64 | 8,192 | **73.48 µs** |

`threads=64` 即使把总线程堆到 16384（是 `th=128, ns=64` 的 2 倍），仍慢 **2.8×**。

**结论：**

1. **`threads=128` 全档胜出**，且优势随 ctx 拉大（1.38× → 3.68×）；
2. **`threads=64` 不可用**：每线程静态寄存器 166 → 306（×1.84），
   而寄存器堆是按 AP 共享的固定资源，可驻留 wave 数随之从 ~12 降到 ~6（占用率 ~37% → ~19%）；
   再叠加"单 wave CTA 无兄弟 wave 掩盖延迟"，实测劣化 3.1–3.9×
   （**是占用率问题，不是寄存器溢出**——306 < 512/线程）；
3. **`threads=256` 在本 shape 不可行**：warp 映射规则要求 `block_N ≥ 16×(256/64) = 64`，
   而 `block_N=64` 的 `KV_shared`（`64×576×2 = 73.7 KB`）已超 64 KB smem 上限；
   强行用 `block_N=32, threads=256` 会触发 TileLang 内部错误
   `Check failed: pb->value != 0 ... Divide by zero`；
4. `block_N=64` 在 6 种 `threads` 组合里**全部编译/启动失败**
   （`mcErrorInvalidValue` 或 `Vector size 4 is not divisible by ...`），进一步印证 smem 是硬约束。

⇒ **`threads` 在 C500 + `dim=512` + `block_H=16` 下没有可调空间**（唯一可行值就是 128）。
autotune 加入该轴后选型不变，代价是候选数 8 → 16、编译时间约 2×。
保留该轴的价值在于：换 shape 或换 smem 更大的硬件（`block_N=64` 放得下时）能自动发现不同取值。

---

## 8. `num_split` 的解析设计

> 📄 完整讲解见独立文档 **[`num_split_design.md`](num_split_design.md)**：
> DeepSeek FlashMLA 的 `get_mla_metadata` 算法（`num_sm_parts` / `payload` / 贪心打包）、
> 与我方 kernel 的结构对比、套用公式的偏差、以及闭式解留出验证失败的完整数据。

要点摘录：

- **官方是"算"不是"调"**：`num_sm_parts = sm_count / num_heads_k / ceil_div(num_heads_per_head_k, 64)`
  定并行度上界 → `payload = ceil(total_kv_blocks / num_sm_parts) + 5` 定每份负载 →
  贪心打包跨序列切分 → 逐序列 `num_splits[batch+1]`（前缀和）。这是 varlen 场景的刚需。
- **不能直接抄**：套到我们的 shape 得 `num_sm_parts=104`，与实测最优在短 ctx 偏低、长 ctx 偏低 3 倍
  （原因：bound 类型不同、smem 224 KB vs 64 KB、varlen vs 单定长）。
- **我方闭式解不可靠**：拟合出的 `max(64, ceil(ctx/256))` 在 6 个训练 shape 上全中，
  但在留出的 `ctx=12288` 上失败（预测 64 / 60.15 µs，实际最优 `num_split=96` / 51.79 µs）。
  最优值由 wave 量化与 main/combine 此消彼长决定，两参数模型捕捉不到。
- **决策**：继续用 autotune 选 `num_split`（~8 个候选 + 磁盘缓存）；若将来做 varlen/paged，
  再照搬"运行时逐序列算 `num_splits[]`"的架构。
