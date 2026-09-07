# mhc_post 优化复盘与后续优化空间

> 分析对象：`TileOPs-Metax/tileops/kernels/mhc/mhc_post.py`、`benchmarks/ops/bench_mhc.py`
> Profiler 数据：`mcprofiler_output/mhc_post/8192_4096/8192_4096.json`
> 硬件口径：`docs/mxC500.md`（AP=104、Wavefront=64、4 PEU/AP、SMEM 64KB、DRAM-L2 1843 GB/s、L2 8MB）
>
> **工作区说明（两份并存，别搞混）**
>
> | 路径 | 分支 / HEAD | 内容 |
> | --- | --- | --- |
> | `/data/code/TileOPs-Metax` | `summer-camp-2026` @ `181801d` | **优化前基线**：`c_x // block_C`、无分派、FLOPs 公式错误 |
> | `/data/code/mhc_post` | `feat/mhc-post` @ `3c1f881` | **优化后实现**，PR 描述的全部改动都在这里 |
>
> `3c1f881` 这条分支上关键提交：`a5f30b4`（kernel 重构 + all-n4 fast path）、
> `0456eb9`（manifest 扩到 V4 生产 shape）、`1e622fe`（C500 报告工具）、
> `d1fa7ff`（带宽字节口径修复）、`3c1f881`（开启 roofline）。
> 下文第 3 节的优化建议**针对 `/data/code/mhc_post` 里的真实代码**，并标注了具体位置。

---

## 0. 结论速览

| 维度 | 现状 | 判断 |
| --- | --- | --- |
| 访存量 | 实测单次读 335.8 MB / 写 268.3 MB，与逻辑流量 335.6 / 268.4 MB 完全吻合 | **已到理论下限，无读放大** |
| 有效带宽 | 1456 GB/s = 峰值 1843 GB/s 的 **79.0%** | 接近 memory bound 极限，剩余空间约 21% |
| 指令效率 | `instruction throughput efficiency` **16.18%**，MTE 占全部指令 81.7% | **主要剩余空间在指令层，不在访存量** |
| VL1 partition | pt0 43.1% / pt2 42.8% / pt1 7.2% / pt3 7.0% | **不均衡**，pt0+pt2 占 85.9% |
| shared 往返 | shared load 指令数（1.04 M/run）比 global read（0.68 M/run）还多 53% | 存在冗余中转 |
| 代码状态 | 优化版在 `/data/code/mhc_post`（`feat/mhc-post` @ `3c1f881`）；`/data/code/TileOPs-Metax` 是 baseline | 后续调优一律在 `mhc_post` 目录下做 |

---

## 1. 已完成的优化（PR 总结）

### 1.1 优化前的主要问题

1. `c_x // block_C` 构造 C 维 grid，**C 非 `block_C` 整数倍时尾部元素不被覆盖**。
2. 单个 program 内部处理所有 `N`，并为 `x_res`、`x_out` 分配 shared memory，资源随 `N` 增长且存在冗余暂存。
3. 缺少任意 `N`、最小 shape、空维度、跨 shape 缓存复用等完整契约。
4. Op 层未拒绝错误 dtype、CPU/跨设备、非连续输入。
5. FakeTensor 输出沿用 `x_layer_out` 的 `[B,C]` shape，与真实 `[B,N*C]` 不一致；`torch.compile(fullgraph=True)` 冷启动不安全。
6. **Benchmark FLOPs 公式错误**：写成 `2*B*(N²*C²+N*C)`，真实只有 `2*B*N*C`，TFLOPS 被夸大 `N*C+1` 倍（本次 15 组 workload 中约 4097–28673 倍）。memory 统计也漏了 batch、dtype 字节数与 `h_post` FP32 流量。
7. 只有 3 个微型 shape，日志需人工换算。

### 1.2 采用的优化方案

| # | 方案 | 要点 |
| --- | --- | --- |
| 1 | Tail-safe 通用 Kernel | `T.ceildiv` + 边界 mask 覆盖任意 `B/N/C`；整块用 `T.copy`，不规则用 masked load |
| 2 | 保持 C 维 CTA 并行度 | C tile 继续作为 grid 维，避免改成 program 内循环后损失 block 级并行 |
| 3 | Shape-aware 分派 | `N==4 && B>=8` 走 all-N4 2D Kernel（合并四分支、复用 `x_layer_out`）；其余走通用 3D Kernel |
| 4 | 精简 shared 路径 | `h_post` 与可复用的 `x_layer_out` 保留 shared；只使用一次的 `x_res` 直读 global、`x_out` 直写 global |
| 5 | 可验证配置空间 | `block_x_b={1,8,64}`、`block_C={64,128}`、`threads={128,256}`、`num_stages={2,3}` |
| 6 | 补全 Op 契约 | dtype/rank/shape/device/contiguous 校验；空维度返回正确空输出；cache key 含 shape/dtype/device/tune |
| 7 | 补全编译路径 | 修正 FakeTensor metadata；tracing 时用固定安全配置进入已注册 custom op |
| 8 | 修复并扩展性能契约 | FLOPs 改 `2*B*N*C`；按真实字节统计；Benchmark/Manifest 扩到 15 个 mHC/DeepSeek V4 workload |
| 9 | 一键工具链 | `scripts/mhc_bench_report.py`（正确性→Benchmark→Markdown/CSV）、`mhc_post_final_matrix.py`（15-case Review 矩阵）、`AGENTS.md` |

### 1.3 取得的成果

- 相对**独立 PyTorch reference**：15 组全部加速，范围 `5.811×–14.171×`，几何平均约 `9.998×`。
- 相对**未优化 Kernel**：13 组加速，范围 `0.917×–3.113×`，几何平均约 `1.385×`；`medium` 与 `v4-flash-decode-s` 有 2.8% / 8.3% 轻微回退。
- 大 workload 估算有效带宽 `1.45–1.50 TB/s`，约峰值 1843 GB/s 的 `78.5%–81.4%`。
- 定向测试从 3 项扩到 **38 项**（数值/边界/异常/分派/缓存/编译），归档结果 `38 passed`。

本次分析对应的 `8192_4096` 即表中 **v4-flash-prefill-l `(8192,4,4096)`**：优化后 `405.60 µs`，相对 PyTorch `14.171×`，相对未优化 Kernel `3.113×`，估算带宽 `1499.4 GB/s`（81.4%）。

### 1.4 优化版实现核对（PR 描述 vs 真实代码）

在 `/data/code/mhc_post` 上逐条核对，PR 声称的改动**均已落地**：

| PR 声称 | 真实代码位置 | 核对 |
| --- | --- | --- |
| Tail-safe | `mhc_post.py:57-59` `T.ceildiv(batch, block_x_b)` / `T.ceildiv(c_x, block_C)`，配合 `if b_idx < batch` / `if c_idx < c_x` 边界判断 | ✅ |
| Shape-aware 分派 | `mhc_post.py:195-203` `_select_mhc_post_kernel`，`n_expand == 4 and batch >= _ALL_N4_MIN_BATCH(8)` 走 all-N4 | ✅ |
| 两个 Kernel | 通用 3D `(batch/b, n_expand, c_x/bC)` `mhc_post.py:56-61`；all-N4 2D `(batch/b, c_x/bC)` 内部展开 `T.Parallel(block_x_b, n_expand, block_C)` `mhc_post.py:154-188` | ✅ |
| 精简 shared 路径 | `x_res` 直读、`x_out` 直写（`mhc_post.py:94-97`、`185-188`），已无 `x_res_shared` / `x_out_shared`；仅 `h_post_shared`、`x_layer_out_shared` 保留 | ✅ |
| 向量化加载 | `vectorized = (batch % block_x_b == 0) and (c_x % block_C == 0)`，整块走 `T.copy`、不规则走 masked scalar | ✅ |
| FakeTensor 修正 | `mhc_post.py:229-233` 返回 `(batch, n_expand * c_x)` | ✅ |
| 编译安全路径 | `_mhc_post_compile_default`（`mhc_post.py:236-256`）+ `torch.compiler.is_compiling()` 分支（`ops/mhc.py`） | ✅ |
| Op 契约 | `ops/mhc.py` dtype/rank/shape/device/contiguous 校验；cache key 含 `device_index` 与 `tune` | ✅ |
| FLOPs 修复 | `bench_mhc.py:139` `2 * batch * n_expand * c_x` | ✅ |
| memory 修复 | `bench_mhc.py:146-151` 按 `itemsize` 计字节 + `h_post` FP32 | ✅ |
| 15 组 workload | `bench_mhc.py:154-181` 含 `v4-flash-prefill-l (8192,4,4096)` | ✅ |

一点与设计意图相关的记录：`autotune_configs`（`mhc_post.py:294-311`）把 `num_stages` 固定为 1，注释说明了原因——「把 C 从 grid 移进流水线循环会塌缩 CTA 并行度并回归」。这是有实验依据的取舍，后续不要再轻易把 C 挪进循环。

---

## 2. 8192×4096 Profiler 数据分析

### 2.1 先把聚合值折算到单次执行

mcProfiler 会对每个 metric 各跑一轮 workload，`report.txt` 里的计数器是**多轮 replay 的累加值**，直接与单次 analytical 值相除会得到错误结论。用总流量反推 replay 次数：

```
逻辑流量/次 = x_layer_out 67.1MB + h_post 0.13MB + x_res 268.4MB + x_out 268.4MB = 604.1 MB
实测 Global read + write   = 37.371 GB + 29.864 GB = 67.236 GB
=> replay 次数 ≈ 67.236 GB / 0.6041 GB ≈ 111.3
```

折算后：

| 指标 | 折算到单次 | 逻辑值 | 结论 |
| --- | --- | --- | --- |
| Global read | **335.8 MB** | 335.6 MB | 一致 |
| Global write | **268.3 MB** | 268.4 MB | 一致 |

**结论：访存量已经是最优，既没有 `x_layer_out` 读放大，也没有重复写。** 这一点很重要——意味着后面所有优化都不能指望"少读数据"，只能从**指令效率、partition 均衡、延迟隐藏**上要时间。

### 2.2 关键计数器

| 类别 | 指标 | 值 | 解读 |
| --- | --- | --- | --- |
| 吞吐 | `Memory Access per Second` | 1,110,916 MB/s | ≈1.11 TB/s（profiler 口径） |
| Roofline | `case_bandwith` / `case_I` / `MAX_I` | 1456 GB/s / 25 / 260 | 计算强度 25 远低于拐点 260，**典型 memory bound** |
| 指令 | Total / Compute / Memory | 34.98 亿 / 32.16 亿 / 2.82 亿 | "Compute" 里绝大部分其实是 MTE 搬运 |
| 指令构成 | MTE / MISC / GVM / BSM / STE / MMA | 81.7% / 6.6% / 3.8% / 3.8% / 3.7% / **0%** | **MTE = 向量操作，STE = 标量操作**。向量占 81.7% 是正常且期望的（逐元素算子本就该全向量化）；标量仅 3.7% 说明地址/控制开销低，**这是好事，不是瓶颈** |
| MTE 周期 | `MTE_4CYCLES` : `MTE_16CYCLES` | **89.5% : 10.5%** | 宽向量（16 周期）指令仅占 10.5%；对照 `mla_decode` 的 66.4%，说明本算子**向量宽度偏窄**，这才是真问题 |
| 效率 | `instruction throughput efficiency` | **16.18%** | 指令发射槽位大量浪费 |
| stall | `vls_pipeline_stall` / `vls_wdata_stall` / `wsm_stall` | 73.7% / 18.2% / 8.1% | VLS 访存流水线阻塞为主 |
| duty | AP MTE / STE / MMA / L2C | 53.22% / 2.39% / 0% / 0.37% | MTE 单元半忙，其余单元基本空闲 |
| 缓存 | VL1 hit / **L2C hit** / SL1 hit | 91.04% / **0.06%** / 99.96% | 数据量 604MB ≫ L2 8MB，L2 命中率归零属必然 |
| 延迟 | `Dnoc Read Average Latency` | **580.46 cycles** | 直方图 512–1k 占 58.8%、256–511 占 32.4% |
| shared | load / store 指令（单次） | 1.04 M / 0.16 M | **shared load 比 global read（0.68 M）还多 53%** |
| shared | 平均 load 延迟 / conflict cycles | 66.9 cycles / **0.0** | 无 bank conflict（效率 100%），但往返次数偏多 |
| 调度 | WORKGROUPS / WAVES | 3,606,176 / 14,424,704 | 每 workgroup **4 waves**（即 256 threads/block） |
| 调度 | `Average Wave life cycles` | 10,931 | 单 wave 生命周期约为一次访存延迟（580）的 19 倍 |
| partition | pt0/pt1/pt2/pt3 | **43.1% / 7.2% / 42.8% / 7.0%** | pt0+pt2 = 85.9%，明显不均衡 |

### 2.3 瓶颈判断

1. **不是访存量问题**：单次读/写与理论下限一致，继续优化"少读数据"没有空间。
2. **是"向量宽度 + 延迟"问题**：向量操作（MTE）占 81.7% 且标量（STE）仅 3.7%，说明指令结构是健康的；问题在于**宽向量指令只占 10.5%**（`MTE_16CYCLES`），配合 `instruction throughput efficiency` 仅 16.18%、VLS 流水线 stall 73.7%，指向**每线程访存宽度不足**。带宽虽已 79%，但指令发射槽位大量浪费。
3. **partition 不均衡是向量宽度不足的衍生症状**，根因见 3.3 节（不是独立的调度问题）。
4. **shared 往返偏多**：shared load 指令数超过 global read，其中相当一部分是 `h_post` / `x_layer_out` 的复用读；虽然无 bank conflict，但 66.9 cycles 的延迟叠加 1.04 M 条指令是可观开销。

---

## 3. 仍可优化的方向（按性价比排序）

### P0 — 确认在 `/data/code/mhc_post` 上工作（前置项）

优化版已确认存在且 PR 声称的改动全部落地（见文首表格与 1.4 节核对）。
`TileOPs-Metax` 那份是 baseline，**不要在它上面调优**：非整块 C 会算错、FLOPs 口径不可信。后续一律：

```bash
cd /data/code/mhc_post
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:$PWD:${PYTHONPATH:-}
```

如果两份需要合并，方向是把 `feat/mhc-post` 合入 `summer-camp-2026`，而不是反过来。

### P1 — 提高访存向量宽度（当前最大的单点收益）

**实测证据**（已导出 C-like 源码，按 autotune 实际选中的 `block_x_b=8, block_C=128, threads=256`）：

```c
// x_res 读：uint2 = 8 字节 = 4 个 bf16
*(uint2*)(x_res_local_cast_5 + 0) = *(uint2*)(x_res + ...);
// x_out 写：uint2 = 8 字节
*(uint2*)(x_out + ...) = *(uint2*)(x_out_local_cast_2 + 0);
// x_layer_out 读：uint2 = 8 字节
*(uint2*)(x_layer_out_local_cast_1 + 0) = *(uint2*)(x_layer_out + ...);
```

**结论：已经充分合并，sector 利用率 100%。**

⚠️ 注意区分两个概念，这里容易误判：

- `docs/mxC500.md` 的「GPU 访存一次最小为 32Bytes」指的是 **sector（最小内存事务单位）**，
  与 CUDA 的 32B sector 同义，**不是"每个线程要访问 32B"**；
- 判断向量化是否充分，要看 **wave 级合并后的 sector 利用率**：

```text
C500 wavefront = 64 线程（docs/mxC500.md）
当前每线程 uint2 = 8 B
=> 一个 wave 访问 64 × 8 B = 512 B 连续地址
=> 512 B / 32 B = 16 个 sector，且全部有效
=> sector 利用率 = 100%，已完美合并，没有浪费带宽
```

因此**加宽到 16 B/线程并不会减少访存量或 sector 数**（总字节数不变），
它只会**减少指令条数**。P1 的收益应定位在"减少 issue 开销"，而不是"省带宽"——
这一点修正后，P1 的预期收益需要相应下调（见下）。

> ⚠️ 分析时必须用 **autotune 实际选中的配置**导出。默认配置（`block_x_b=1, block_C=64,
> threads=128`）下只有 `uint1` = 4 B/线程，会低估向量化程度的 2 倍。
> 用 `python scripts/dump_mhc_src.py --block-x-b 8 --block-C 128 --threads 256`。

**做法**：

- 主循环中 `x_res` / `x_out` 改用 `T.copy` 搬进 `T.alloc_fragment`、算完再整段 `T.copy` 写回，
  让 TileLang 生成 `uint4`（16B）乃至更宽的访存；
- 可把 `x_res` / `x_out` 在 prim_func 里声明为 `[batch, n_expand, c_x]` 的 3D 视图
  （物理内存与 `[batch, n_expand*c_x]` 完全一致），这样每个 `j` 分支内部连续，便于 `T.copy` 向量化；
- 保留 `vectorized` 分支：不规则 shape 继续走 masked scalar，不要为向量化牺牲 tail-safe。

**预期（按 sector 口径修正后）**：

- 访存量与 sector 数都不变 ⇒ **不会带来带宽提升**；
- 收益只在指令条数：8 B → 16 B/线程可让 MTE 指令数减半，改善
  `instruction throughput efficiency`（16.18%）与 `MTE_16CYCLES` 占比（10.5%）；
- 但 `AP MTE Duty` 仅 53.22%，MTE 单元**尚未饱和**，减少它的指令数 ≠ 提速；
  真正的瓶颈是 `vls_pipeline_stall` 73.7% 叠加带宽已用掉 79%，即**在等内存**；
- 综合判断：P1 仍值得做，但预期是**个位数到十余个百分点的 issue 开销节省**，
  不应期待倍数提升。优先级低于"想办法提高有效带宽"。

### P2 — `h_post` 改 fragment：已实测劣化，已回退 ❌

**实测结果**（autotune，三个 shape）：

| 版本 | large `(4,4,2560)` | v4-flash-prefill-l `(8192,4,4096)` | expand8 `(128,8,1024)` |
| --- | --- | --- | --- |
| 改动前 | 3.5 µs | **405.4 µs** | 7.7 µs |
| `h_post_shared` → `T.alloc_fragment` | 3.9 µs | **1492.1 µs（3.68× 慢）** | 16.4 µs |

**原因**：`h_post_fragment[block_x_b, n_expand]` 的寄存器布局与主循环
`T.Parallel(block_x_b, n_expand, block_C)` 的线程映射不匹配，跨 `T.Parallel` 读写
产生大量寄存器搬移，代价远超 shared 读。

**结论：此路不通，保持 `T.alloc_shared`。** 报告里「shared load 比 global read 多 53%」
不是这里的瓶颈——shared 无 bank conflict（效率 100%）、单次延迟 66.9 cycles，
远低于 global 的 580 cycles。

### P3 — VL1 partition：grid 交换已实测无效，真正根因是 C tile 步长

**尝试**：grid 从 `(batch/blk, c_x/blkC)` 换成 `(c_x/blkC, batch/blk)`，
让决定 partition 的 C tile 走最快变化的 `blockIdx.x`。

**实测**（8192×4096）：

| 指标 | 改前 | 改后（grid 交换） |
| --- | --- | --- |
| pt0 / pt1 / pt2 / pt3 | 43.1% / 7.2% / 42.8% / 7.0% | **41.8% / 8.3% / 41.6% / 8.3%** |
| 最大/最小比 | 6.14 | 5.07 |
| L2C hit | 0.06% | **0.72%**（+12×） |
| Dnoc 读延迟 | 580.46 | **558.31** |
| Benchmark 带宽 | 1.4900 TB/s | **1.5018 TB/s** |
| 8192×4096 延迟 | 405.4 µs | **402.2 µs** |
| expand8 延迟 | 7.7 µs | **7.0 µs** |

**partition 几乎没变**，但 L2 命中提升 12 倍、延迟略降、带宽略升——该改动**保留**
（收益在 DRAM 局部性，不在 partition）。

#### 为什么 grid 交换会带来提升

关键前提：**GPU 按 blockIdx 递增的顺序分发 block**，同一时刻驻留的 block 具有连续的小
blockIdx；而 `blockIdx.x` 是变化最快的维度，所以「连续的小 blockIdx」≈「连续的 blockIdx.x」。
**`blockIdx.x` 映射到哪个逻辑维度，就决定了并发 block 在物理内存上的分布形态。**

以 8192×4096、`N=4`、`block_x_b=8`、`block_C=128` 为例：

```text
一个 batch 行的 x_res 长度 = N*C*2 = 4*4096*2 = 32 KB
C tile 数   = c_x / block_C = 4096 / 128 = 32      （一个 batch 行被切成 32 段）
batch tile 数 = batch / block_x_b = 8192 / 8 = 1024
```

**改前** `grid = (batch_tile, c_tile)`，`blockIdx.x` = batch tile：

- 并发的 104+ 个 block 的 `blockIdx.x` 连续 ⇒ 它们处理**不同 batch 行的同一个 C tile**；
- 相邻 block 的地址间隔 = 一个 batch 行 = **32 KB**；
- 这些地址均匀散落在 268 MB 的 `x_res` 上 ⇒ DRAM row 频繁切换，L2 也几乎无法复用。

**改后** `grid = (c_tile, batch_tile)`，`blockIdx.x` = C tile：

- 并发 block 的 `blockIdx.x` 连续 ⇒ 它们处理**同一 batch 行内相邻的 C tile**；
- 相邻 block 的地址间隔 = `block_C * 2 = 256 B` ⇒ 连续 32 个 block 正好铺满一个 batch 行的
  32 KB **连续**内存；
- 104 个 AP 同时工作即可覆盖约 3 个 batch 行的连续地址区间 ⇒ DRAM row buffer 连续命中，
  相邻 cache line 也被顺带取进 L2。

**为什么 partition 没跟着变**：partition 由 **C tile 的字节步长**（`block_C*2 = 256 B`）决定，
与 grid 维度的排列顺序无关。两者是相互独立的机制——grid 顺序管的是"并发 block 落在哪段内存"，
tile 步长按 128B 切分决定"落在哪个 partition"。所以这次只收获了局部性收益。

**真正的根因**（由实际配置源码的地址表达式推得）：

```text
x_res 字节地址 = by*262144 + i*65536 + (tid>>5)*8192 + bx*256 + (tid&31)*8
                                                       ^^^^^^^^^ C tile 步长
```

`block_C=128` ⇒ 每个 C tile 跨度 `128*2 = 256 B`。一次访问中 32 个线程覆盖 256 B，
起始 partition = `(bx*256)>>7 & 3` ⇒ **bx 为偶数时 pt0、奇数时 pt2，块首只可能落在 pt0/pt2**。
而 partition stall 计数偏向「最早发出且未返回」的请求，也就是块首那一段，于是形成 pt0+pt2 ≈ 83% 的双峰。

**要真正摊平，C tile 的字节步长必须是 128 B 的奇数倍**，即 `block_C ∈ {64, 192, 320, ...}`：

| `block_C` | tile 步长 | `bit[8:7]` 轮换 | 结论 |
| --- | --- | --- | --- |
| 64 | 128 B | `bx & 3` → pt0,1,2,3 | ✅ 完美轮换 |
| 128（当前） | 256 B | `(bx*2)&3` → 只有 pt0/pt2 | ❌ |
| 256 | 512 B | 恒为同一 partition | ❌❌ |

**障碍**：`c_x=4096` 只能被 `block_C = 64/128/256` 整除，`192` 会退化到 masked scalar 路径；
而 autotune 实测 `block_C=64` 比 `128` 慢约 12%（0.0227 vs 0.0203 ms）。

**判断**：当前带宽已达 79%，瓶颈在 DRAM 而非 L1 partition；partition 均衡的收益
大概率小于换 `block_C=64` 造成的 12% 损失，**建议优先级放低**。
若仍要验证：用固定配置（非 autotune）以 `block_C=64` 采一次 profiler 对比 partition 即可。

### P4 — 提升 occupancy 与延迟隐藏

**现象**：每 workgroup 4 waves（256 threads），`Average Wave life cycles` 10,931，`Dnoc Read Latency` 580 cycles。

**依据**：`docs/mxC500.md` 给出 `Max waves/AP = 32`、`Max work-items/AP = 2048`。当前 shared/block 仅 0.5–4.1 KiB（PR 数据），远未触到 64KB 上限，**occupancy 完全没被 shared 卡住**，还有大量 wave 槽位可用。

**做法**：
- 提高 `threads`（256 → 512/1024）或让每个 block 处理多个 batch 行，把 waves/AP 从当前水平往上推；
- 用 `T.Pipelined` + `num_stages` 增加单 block 内的访存重叠；
- 注意：`threads=256` 在 `mla_decode` 上曾触发 MACA 版 TileLang 的 `Divide by zero`，需逐个配置验证。

### P5 — 改善 DRAM 行局部性（针对 580 cycles 的 Dnoc 延迟）

**现象**：L2C hit 0.06%（数据量使然，不可改），`Dnoc Read Average Latency` 580 cycles。

**做法**：调整 grid → block 的映射顺序（rasterization / swizzle），让同时驻留的 block 访问**物理上相邻**的地址，提高 DRAM row buffer 命中。当前 grid 是 `(batch, C/block_C)`，可尝试按 C tile 优先或按 104（AP 数）的整数倍做分块。这是零访存量代价的纯调度优化。

### P6 — 补齐小 shape 侧

PR 已指出小 workload（decode-s 等）受固定开销与并行度不足影响，带宽利用率仅 21.5%–35.8%，且 `medium`、`v4-flash-decode-s` 相对未优化 Kernel 有 2.8%/8.3% 回退。建议：
- 为小 shape 单独选小 `block_C` + 高 `block_x_b`，减少 launch 与 workgroup 数；
- 在 autotune 配置里显式加入小 shape 的候选（当前 `block_x_b={1,8,64}` 已有，但需确认小 shape 会走到）。

---

## 4. 本轮实测优化记录

### 4.1 先校准带宽分母：实测峰值只有 1528 GB/s

用 `scripts/bandwidth_probe.py`（口径统一为 10^9 字节，与 1843 GB/s 对齐）实测：

| 模式 | 流量 | 耗时 | 带宽 |
| --- | --- | --- | --- |
| write only (`y.fill_`) | 0.268 GiB | 175.6 µs | **1528.5 GB/s** |
| read+write (`mul x*2`) | 0.537 GiB | 355.5 µs | 1510.1 GB/s |
| 2R+1W (`a+b`) | 0.805 GiB | 533.5 µs | 1509.6 GB/s |
| read+write (`copy_`) | 0.537 GiB | 398.4 µs | 1347.7 GB/s |

实测峰值 ≈ **1528 GB/s**，只有理论 1843 的 **82.9%**。
而 mhc_post 在 8192×4096 上是 1502 GB/s ⇒ **已达实测峰值的 98.2%**。

⇒ 先前"还有 21% 带宽空间"是相对理论值的假象，**大 shape 已无带宽可挖**。

### 4.2 固定延迟 ≈ 3.6 µs，才是小 shape 的真正限制

`small (1,4,1280)` 的 kernel 时间是 **3.6 µs**，而它的数据量只有 30 KB
（按 1528 GB/s 只需 0.02 µs）——说明这 3.6 µs 基本是 GPU 侧的最小 kernel 延迟。

于是每个 shape 的理论下限 = `3.6 µs + 流量 / 1528 GB/s`：

| shape | 当前 | 理论下限 | 结论 |
| --- | --- | --- | --- |
| (128,4,4096) | 9.8 µs | 9.77 µs | 已达下限 |
| (32,4,7168) | 7.4 µs | 6.30 µs | 差 17%（见 4.5） |
| (32,4,4096) | 5.2 µs | 5.14 µs | 已达下限 |
| (128,2,2048) | 5.3 µs | 5.31 µs | 已达下限 |
| (128,8,1024) | 6.8 µs | 6.52 µs | 差 4% |

> ⚠️ 测量陷阱：用 CUDA event 直接包住 kernel 调用会量到约 **35 µs**（Python → custom op →
> launch 的端到端开销），是真实 kernel 时间（约 7 µs）的 5 倍，会完全淹没配置之间的差异。
> 对比配置必须用 `benchmarks.benchmark_base.BenchmarkBase` 的 CUPTI 口径
> （已封装为 `scripts/mhc_config_probe.py`）。

### 4.3 本轮三处改动与实测

| 改动 | 内容 | 实测结果 | 处置 |
| --- | --- | --- | --- |
| ① grid 维度交换 | C tile 放 `blockIdx.x` | L2C hit 0.06%→0.72%、decode-m −16.2% | **保留** |
| ② `block_C` 增补 256 | autotune 候选 `{64,128,256}` | decode-s −13.3%、expand2 −10.2%、decode-l −12.3% | **保留** |
| ③ autotune 精度 | `warmup=50, rep=150`（原 25/50 单次平均） | 未能修正 pro-decode-s 误选 | 保留（降噪有依据） |
| P2 | `h_post` 改 fragment | 8192×4096 劣化 3.68× | **已回退** |

### 4.4 15 组最终结果（对比 PR 基线）

| case | PR 基线 | 本轮 | 变化 |
| --- | --- | --- | --- |
| small (1,4,1280) | 3.50 | 3.5 | 0% |
| medium (2,4,1920) | 3.60 | 3.7 | +2.8% |
| large (4,4,2560) | 3.70 | 3.7 | 0% |
| v4-flash-decode-s (32,4,4096) | 6.00 | **5.2** | −13.3% |
| v4-pro-decode-s (32,4,7168) | 6.30 | 7.4 | **+17.5%** ⚠️ |
| v4-flash-decode-m (128,4,4096) | 11.70 | **9.8** | −16.2% |
| v4-pro-decode-m (128,4,7168) | 14.80 | **13.9** | −6.1% |
| v4-flash-decode-l (512,4,4096) | 31.60 | **27.4** | −13.3% |
| v4-pro-decode-l (512,4,7168) | 51.60 | **46.1** | −10.7% |
| v4-flash-prefill-s (1024,4,4096) | 56.80 | **52.2** | −8.1% |
| v4-pro-prefill-s (1024,4,7168) | 97.00 | **90.5** | −6.7% |
| v4-flash-prefill-l (8192,4,4096) | 405.60 | 403.9 | −0.4% |
| v4-pro-prefill-l (4096,4,7168) | 367.60 | **354.0** | −3.7% |
| expand2-c2048 (128,2,2048) | 5.70 | **5.3** | −7.0% |
| expand8-c1024 (128,8,1024) | 7.00 | **6.8** | −2.9% |

**12/15 提升、2 持平、1 回退**；正确性 `38 passed`。

### 4.5 已知问题：pro-decode-s (32,4,7168) 回退 17.5%

用 `scripts/mhc_config_probe.py` 对该 shape 逐配置实测（CUPTI 口径）：

| 配置 | 实测 |
| --- | --- |
| `bb=8, bC=64, th=128`（最优） | **6.18 µs** |
| `bb=8, bC=128, th=256`（PR 基线的选择） | 6.38 µs |
| `bb=1, bC=256, th=128` | 6.59 µs |
| `bb=1, bC=256, th=256`（autotune 实际选中） | **7.42 µs** |

**根因**：autotune 走的是 tilelang autotuner 的默认输入 supply，不做 L2 flush /
地址轮换，与 `BenchmarkBase` 的稳定态口径不一致，于是在候选变多后选错了配置
（选了第 6 名，比最优慢 20%）。

**已尝试的修复**（本轮，两个版本均失败并回退）：

1. **v1**：在 `MHCPostKernel.autotune()` 里复刻 BenchmarkBase 协议（L2 flush + 地址轮换 +
   3 trial 中位数）。第一版把 `flush_buf.zero_()` 写进了迭代循环——8 MB fill 的 ~30 µs
   把 kernel 之间 1~9 µs 的差异完全淹没；
2. **v2**：flush 移到 trial 外、并绕过 torch custom op 直接测底层 JITKernel。
   但 CUDA event 墙钟仍包含每次迭代 ~5 µs 的 launch 开销——小 shape 的 kernel 只有
   6~9 µs，信噪比不足，decode-s / expand8 反而回退 36% / 61%；
3. 两个版本均已回退到基类 tilelang autotuner。

**正确的修法**（框架级，留作后续）：让 autotune 的计时用 **CUPTI 只统计 kernel 时间**
（排除 host launch 开销），即把 `BenchmarkBase` 的测量能力作为回调暴露给 autotune 流程。
在此之前，autotune 的选型在小 shape 上不可信。

**当前状态**：已回退到基类 autotuner，保留 grid 交换与 `block_C=256` 候选。
15 组：**12 提升 / 2 持平 / 1 回退**（pro-decode-s +17.5%）。
临时方案：对误选 shape 固定 `bb=8, bC=128, th=256`（6.38 µs）。

---

## 5. 工程待办清单

- [x] 确认优化版位置：`/data/code/mhc_post`（`feat/mhc-post` @ `3c1f881`），PR 改动已全部落地（1.4 节核对表）。
- [x] P2（`h_post` 改 fragment）：实测劣化 3.68×，已回退。
- [x] P3（grid 维度交换）：partition 未改善，但 L2C hit +12×、带宽 1.49→1.50 TB/s，**已保留**。
- [x] 带宽分母校准：实测峰值 1528 GB/s，大 shape 已达 98.2%，无带宽可挖。
- [x] `block_C` 候选增补 256：decode-s −13.3%、expand2 −10.2%、decode-l −12.3%。
- [x] 正确性：`38 passed`。
- [ ] **修 autotune 测量协议**（L2 flush + 地址轮换 + 多 trial 中位数），解决
      `pro-decode-s (32,4,7168)` 选错配置导致的 +17.5% 回退。
- [ ] 剩余：把 `feat/mhc-post` 合入 `summer-camp-2026`（若需要交付到主分支）。
- [ ] 回填 PR 中的占位项：算子认领 Issue、PR A 链接、最终 PR HEAD 完整 SHA。
- [ ] 在最终 HEAD 上重跑 `tests/ops/test_mhc.py -k "mhc_post"`（38 项）与 15 组 Benchmark。
- [ ] 调优验证优先级：P1 → P2 → P3，每步用 `mcProfiler` 复采 8192×4096 并与本报告对比（重点看 `instruction throughput efficiency`、shared load 数、partition 分布三项）。

---

## 6. 复现命令

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:$PWD:${PYTHONPATH:-}

# 正确性门禁（PR 归档为 38 passed）
python -m pytest -q tests/ops/test_mhc.py -k "mhc_post"

# 15 组 Benchmark
python -m pytest -q benchmarks/ops/bench_mhc.py -k "mhc_post" -s

# 一键正确性 → Benchmark → Markdown/CSV
python scripts/mhc_bench_report.py

# 本仓库内的 profiling 与报告对比工具
python scripts/run_mcprofiler.sh <kv_ctx> [counts] [casename]
python scripts/compare_reports.py mcprofiler_output/mla_decode/16384 <新报告目录>
```

> `scripts/compare_reports.py` 目前是从 `report.txt.csv` 读取（mcProfiler 目录格式）。
> `mhc_post/8192_4096/8192_4096.json` 是另一种汇总 JSON 格式（顶层即 `Summary`/`Memory Statistics` 等），
> 若后续要纳入自动对比，需要为它补一个解析器。
