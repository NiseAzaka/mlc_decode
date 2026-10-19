# MLA Decode 最终实现（当前最优）

> 本文只描述**当前最优实现**，不与其它实现对比，用于快速获取算子结果。
>
> - 实现：[`mla_decode.py`](../mla_decode.py)（`flashattn_shared_q` + combine）
> - 选型：autotune 逐 shape 实测选择（`mla_autotune_configs` / `flashattn_autotuned`）
> - Profiler 证据：`mcprofiler_output/mla_decode/mla_best_16384/output20261016161756/`
> - 硬件口径：[`mxC500.md`](mxC500.md)（104 AP、4 PEU/AP、wavefront 64、SMEM 64 KB/AP、
>   寄存器堆 512 KB/AP、DRAM-L2 1843 GB/s、L2 8 MB）
> - 平台：MetaX C500（整卡），MACA 3.7.1.5，PyTorch 2.8.0+metax3.7.1.3，
>   TileLang 0.1.10（MACA），mcProfiler 3.8.1.4

---

## 0. 结论速览

| 项 | 值 |
| --- | --- |
| 计算 | MLA Decode（KV 共享、`d_qk=576` / `d_v=512` 的非对称 MQA 解码） |
| 最优结构 | **split-KV 两段式**：主 kernel 算 partial attention + LSE，combine kernel 做 log-sum-exp 归约 |
| 每档配置 | `block_N=32, block_H=16, num_stages=1, dim_chunk_num=8, q_in_reg=False(Q 在 shared), threads=128`（combine 64），`num_split` 按 ctx 选择 |
| 并行度 | 全部来自 KV 切分：`num_split = max(64, ctx/256)` 量级（head 维坍缩为 1） |
| 延迟 / 吞吐 | 21.8 µs @2K → 204.1 µs @64K；3.28 → 11.17 TFLOPS |
| 有效带宽 | 159 → 408 GB/s（理论峰值 1843 GB/s 的 8.6%–22.1%） |
| 定位 | 计算强度 `case_I = 259.86` ≈ Roofline 拐点 `MAX_I = 260`，**正好卡在计算/访存平衡点** |

---

## 1. 算子接口

### 1.1 数学语义

MLA 把投影矩阵吸收进 Q 之后，解码退化为「**K/V 部分共享的非对称注意力**」：

```text
Q      = [q_main | q_pos]        # d_qk = dim + pe_dim = 576
K      = [kv_main | k_pe]        # d_qk = 576，其中前 dim 维与 V 共享
V      = kv_main                 # d_v = dim = 512
S      = Q @ Kᵀ * (dim + pe_dim)^(-1/2)
O      = softmax(S) @ V
```

关键点：`kv` 的前 `dim` 维**同时充当 K 的一部分和 V 全部**，所以 V 不需要额外存储
（`FlashMLA` 也是这么做的，省一份显存与带宽）。

### 1.2 输入 / 输出

| 张量 | shape | dtype | 说明 |
| --- | --- | --- | --- |
| `Q` | `[batch, heads, dim]` | fp16 | 吸收投影后的 query 主部 |
| `Q_pe` | `[batch, heads, pe_dim]` | fp16 | query 的位置/ROPE 部 |
| `KV` | `[batch, seqlen_kv, kv_head_num, dim]` | fp16 | KV cache 主部（K 与 V 共享） |
| `K_pe` | `[batch, seqlen_kv, kv_head_num, pe_dim]` | fp16 | K 独有部 |
| `Output` | `[batch, heads, dim]` | fp16 | **调用方预分配**，原地写入（无 host 侧分配） |

本工作负载：`batch=1, heads=16, kv_head_num=1, dim=512, pe_dim=64`，
`kv_ctx ∈ {2048 … 65536}`，`softmax_scale = (dim + pe_dim)^(-1/2)`。

`Output` 预分配是接口约定的一部分 —— 计时区外分配，把 Python/launch 开销排除在测量之外。

### 1.3 复杂度与流量（单次调用，`batch=1, heads=16`）

| 量 | 公式 | @16384 |
| --- | --- | --- |
| FLOPs | `2·b·h·ctx·(dim+pe_dim) + 2·b·h·ctx·dim` | 570.4 MFLOP |
| 必需读 | `ctx·(dim+pe_dim)·2 B`（KV+K_pe） | 18.88 MB |
| split 附加读 | `num_split × (dim+pe_dim)·block_H·2 B`（每份重读 Q） | 1.18 MB |

> 各档的 Q 重读量与占比（2048 档高达 KV 流量的 50%）见 §2.2.3。
| 写 | `b·h·dim·2 B`（Output） | 16 KB |

**算术强度** ≈ 570.4 MF / 18.9 MB ≈ **30 FLOP/B**，远低于 Roofline 拐点 260 —— 理论上是访存/平衡型，
但实测远未打满带宽（见 §6）。

---

## 2. 结构总览

一次调用跑**两个 device kernel**（同一个 `T.prim_func` 里的两个 `T.Kernel` 块，顺序执行）：

```text
                    ┌──────────────────────────────┐
  KV / K_pe ───────▶│  main_kernel                 │
  Q / Q_pe  ───────▶│  grid (batch, heads/16,      │──▶ output_partial[b,h,split,dim+64]
                    │        num_split)  128 thr   │──▶ glse[b,h,split+64]   (LSE)
                    └──────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
  output_partial ──▶│  combine_kernel              │──▶ Output[b,h,dim]
  glse           ──▶│  grid (heads, batch, 8)  64thr│
                    └──────────────────────────────┘
```

- **主 kernel**：每个 CTA 负责 `(batch, 一组 head, 一段 KV)`，算完这段 KV 的
  partial attention 归一化结果与局部 LSE；
- **combine kernel**：把 `num_split` 份 partial 用 log-sum-exp 合并成最终输出；
  按输出维度再切成 `dim_chunk_num=8` 份以增加并行度。

### 2.1 各档选定配置

autotune 逐 shape 实测选择，**6 档结果一致**（除 `num_split`）：

| kv_ctx | num_split | chunk（每份 KV） | loop_range | 主 CTA | combine CTA | 延迟 (µs) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 64 | 32 | 1 | 64 | 128 | 22.34 |
| 4096 | 64 | 64 | 2 | 64 | 128 | 29.61 |
| 8192 | 64 | 128 | 4 | 64 | 128 | 44.43 |
| 16384 | 104 | 128–160 | 4–5 | 104 | 128 | 68.43 |
| 32768 | 104 | 288–320 | 9–10 | 104 | 128 | 114.20 |
| 65536 | 256 | 256 | 8 | 256 | 128 | 196.63 |

其余参数全程固定：`block_N=32`、`block_H=16`、`num_stages=1`、`dim_chunk_num=8`、
`q_in_reg=False`（Q 放 shared memory）、主 kernel `threads=128`、combine `threads=64`。

> `num_split` 的规律：ctx ≤ 16384 时统一 64（**下限**，保证喂饱机器）；32768 / 65536 取 256。
> 注意 **32768 档并非"chunk 一律 ≈256"**：它在 §3.2 消除 phase2 串行归约后，由 `num_split=128`
> 改选 `num_split=256`（chunk 128）。这说明原先 `chunk≈256` 的表象是被 phase2 的串行开销
> **扭曲**过的结果，而不是硬规律——串行惩罚一去掉，调优器就倾向切得更细。

### 2.2 切分如何落到执行上：`num_split` → `chunk` → `loop_range`

上表的 `num_split` 只是一个数，它要经过两级展开才变成每个 CTA 的循环次数。
真正"选"的参数只有 `num_split`，`chunk` 与 `loop_range` 都是它的因变量。

#### 2.2.1 三级切分

```text
KV 总长 kv_ctx
  ├─ 第一级：按 num_split 切成 num_split 份    chunk = align_up(ceil(kv_ctx/num_split), block_N)  ← 每份的长度
  │     └─ grid.z = num_split_eff = ceil(kv_ctx/chunk)，CTA 由 bz 定位自己那份的起点；
  │        一个 bz 对应 batch × (heads/block_H) 个 CTA（本配置下 heads/block_H = 1）
  └─ 第二级：每份按 block_N 切块               loop_range = ceildiv(chunk, block_N)  ← 每块是一轮
        └─ 每轮搬 [block_N, 512+64] 的 KV/K_pe 到 shared，做一次 online-softmax 更新
```

整除档（`chunk × num_split_eff == kv_ctx`）下 `chunk` 就退化成 `kv_ctx // num_split`；
非整除时多出来的尾部由 §2.2.5 的列掩码处理。

对应代码（`mla_decode.py`）：

```python
39: with T.Kernel(batch, heads // valid_block_h, num_split, threads=threads) as (bid, hid, bz):
73:     loop_range = T.ceildiv(seqlen_kv // num_split, block_N)
74:     for k in T.Pipelined(loop_range, num_stages=num_stages):
75:         kv_start = (seqlen_kv // num_split) * bz + k * block_N
76:         kv_end   = (seqlen_kv // num_split) * bz + (k + 1) * block_N
```

注意代码里**没有 `chunk` 这个变量**，`seqlen_kv // num_split` 在三处内联重复出现（`:73,75,76`）。

#### 2.2.2 以 `kv_ctx=16384` 走一遍

`num_split=64, block_N=32, block_H=16, heads=16, batch=1`：

```text
kv_ctx     = 16384 token
chunk      = 16384 // 64 = 256 token/份        ← 一个 CTA 负责的 KV 长度
loop_range = ceildiv(256, 32) = 8 轮            ← 每轮吃 32 个 token
自校验     = 8 × 32 = 256 = chunk               ← 整除，无尾部残余
grid       = (batch, heads//block_H, num_split) = (1, 1, 64)   → 64 个主 CTA
```

> 选这一档举例是因为三个数互不相同、不会互相混淆：
> `num_split=64`（切几份）、`chunk=256`（每份多长）、`loop_range=8`（每份切几轮）。
> 若取 4096 档，`chunk` 恰好也等于 64，容易把「份数」和「每份长度」看混。

每个主 CTA（`bz = 0..63`）实际做的事：

| 步骤 | 动作 | 单次 | 小计 |
| --- | --- | ---: | ---: |
| 0 | 一次性把本 CTA 的 16 个 head 的 Q / Q_pe 搬到 shared | 18 KB | 18 KB |
| 第 0–7 轮 | 搬 `KV[..., bz*256 + k*32 : +32]` → gemm1/gemm2 → online-softmax → gemm3 | 36 KB/轮 | **288 KB** |
| 收尾 | `acc_o × 1/logsum`，写 `output_partial[bid, hid*16:(hid+1)*16, bz, 0:512]` + `glse` | — | 16 KB 写 |

即每个 CTA 读 `18 + 288 = 306 KB`，其中 KV 占 94%；64 个主 CTA 合计读
`KV 18.87 MB + Q 重读 1.18 MB ≈ 20.05 MB`（与 §2.2.3 的表一致）。

然后 combine kernel 用 `grid=(heads, batch, dim_chunk_num)=(16,1,8)` = 128 个 CTA，
每个 CTA 针对 `(head, 512 列中的 64 列)`，把 **64 份** partial 做 log-sum-exp 加权求和
（每 CTA 读 `64 份 × 64 列 × 2 B = 8 KB`，见 §3.2）。

#### 2.2.3 每档的轮数、CTA 数与流量

| kv_ctx | num_split | chunk | **loop_range** | 主 CTA | 每 CTA KV 读 | 主 kernel KV 总读 | Q 重读总量 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 64 | 32 | **1** | 64 | 36 KB | 2.36 MB | 1.18 MB |
| 4096 | 64 | 64 | **2** | 64 | 72 KB | 4.72 MB | 1.18 MB |
| 8192 | 64 | 128 | **4** | 64 | 144 KB | 9.44 MB | 1.18 MB |
| 16384 | 64 | 256 | **8** | 64 | 288 KB | 18.87 MB | 1.18 MB |
| 32768 | 256 | 128 | **4** | 256 | 144 KB | 37.75 MB | 4.72 MB |
| 65536 | 256 | 256 | **8** | 256 | 288 KB | 75.50 MB | 4.72 MB |

两条规律：

- **KV 总读量只由 `kv_ctx` 决定**（`kv_ctx × (512+64) × 2 B`），与 `num_split` 无关 ——
  切分不增加 KV 流量，只是把同一份总量摊到更多 CTA 上。
- **切分的代价是 Q 重读**：每个 CTA 都要重搬自己那 16 个 head 的 Q（18 KB），
  总计 `num_split × 18 KB`。占 KV 流量的比例：2048 档 **50%**、4096 档 25%、
  8192 档 12.5%、16384 档 6.25%、32768 档 12.5%（因该档 `num_split=256`）、65536 档 6.25%。
  这是 `num_split` 加大时的代价之一 —— 2048 档取 64 是为了并行度**被迫**承受 50% 的重复读。

#### 2.2.4 `loop_range` 少意味着固定开销摊不薄

主 kernel 延迟近似是「固定开销 + 每轮成本 × `loop_range`」。用 `loop_range=1` 的 2048
和 `loop_range=4` 的 8192 两档就能解出这两个量：

```text
2048 (1 块/CTA) = 22.34 µs
8192 (4 块/CTA) = 44.43 µs  → 每块边际成本 = (44.43 − 22.34) / 3 ≈ 7.36 µs
⇒ 固定开销 ≈ 22.34 − 7.36 ≈ 15.0 µs
   主要是：Q/Q_pe 装载、流水线 prologue、combine kernel 的一次完整执行、kernel launch
```

> 这个「固定开销」在 §3.2 优化前后由 **16.6 µs 降到 15.0 µs**（差值 1.6 µs；2048 档实测有 ±3%
> 波动，用旧值 21.76 拟合时为 2.3 µs），与 phase2 串行归约在 `num_split=64` 时的实测开销
> （2.1–2.3 µs，§3.2.1）同量级 —— 两条独立推算互相印证。

拿另外两档校验 `14.3 + 7.51 × 轮数`：

| 档位 | `loop_range` | 模型 | 实测 | 偏差 |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 2 | 29.7 µs | 29.61 µs | −0.3% |
| 16384 | 8（等分形式） | 73.9 µs | 68.43 µs | −7.4% |

长档偏低是因为轮数多时流水线与 L2 复用更充分，边际成本还会略微下探。

也就是说 **2048 档（1 轮）约 2/3 的时间、4096 档（2 轮）约一半的时间并不花在 KV 主循环上**。
这是小 ctx 每 token 成本明显高于长 ctx 的直接原因：

| kv_ctx | `loop_range` | 延迟 | 每 token 摊销 |
| ---: | ---: | ---: | ---: |
| 2048 | 1 | 22.34 µs | **10.9 ns/token** |
| 4096 | 2 | 29.61 µs | 7.2 ns/token |
| 8192 | 4 | 44.43 µs | 5.4 ns/token |
| 16384 | 4–5 | 68.43 µs | **4.2 ns/token** |

2048 档的每 token 成本是 16384 档的 **2.6×**，差距全部来自固定开销。

#### 2.2.5 切分对齐规则：只锁对齐，不锁整除

```text
chunk          = align_up(ceil(kv_ctx / num_split), block_N)
num_split_eff  = ceil(kv_ctx / chunk)        ≤ num_split（不会产生空 CTA）
```

早期实现要求 `num_split | kv_ctx` 且 `chunk % block_N == 0`，代价是**任何非 2 的幂 ctx
都会掉进保底单 split**。旧代码的坑是静默的：

- 反例：`num_split=312` 在 32768 上 → `chunk = 32768 // 312 = 105`，同时踩两个坑：
  - `312 × 105 = 32760 < 32768` ⇒ **末尾 8 个 token 不属于任何 split，直接漏算**；
  - `105 % 32 = 9 ≠ 0` ⇒ `ceildiv(105, 32) = 4` 轮覆盖 128 个位置，第 4 轮越过本份边界
    多读 9 个 token（读到下一份，即别人的 KV）。
  两者叠加使 `cos` 只有 **0.9376**。

现在由 `_split_plan()` 在构建期统一算出 `(chunk, num_split_eff)`，只保留两条**可满足**的约束：

1. `chunk % block_N == 0`（每轮都是整块，不跨 split 边界）；
2. `num_split_eff = ceil(kv_ctx / chunk)`（用 `ceil` 而非 `//`，不产生 `bz*chunk ≥ kv_ctx`
   的空 CTA，旧实现 312 份会有 56 个空 CTA 白占波次）。

剩下的尾部只有**最后一个 split 的末块**，有效长度 `∈ [1, block_N]`，由 kernel 处理：

| 情况 | 生成路径 | 做法 |
| --- | --- | --- |
| `chunk × num_split_eff == kv_ctx` | 编译期即走原快路径 | 无掩码、无额外指令 |
| 存在残缺块且 `kv_ctx ≥ block_N` | 尾部快路径 | 拷贝起点 clamp 到 `kv_ctx - block_N`（整块读，保留 `coalesced_width=8`），再用列掩码把 `[kv_start, kv_ctx)` 之外的列在 `reduce_max` 前覆盖成 `-inf` |
| `kv_ctx < block_N` | 兜底 | 逐行填充 + 补零，保证不越界 |

列掩码放在 `reduce_max` 之前是必要的：未初始化 shared 参与 GEMM 只会污染被覆盖的列，
有效列的点积不受影响；若把掩码放到 `exp2` 之后，垃圾值会先抬高 `max`、再让有效列下溢。

实测（`block_N=32, threads=128, num_split=64`，与六档同配置）：

| ctx | 尾部模式 | 延迟 | cos_sim |
| ---: | --- | ---: | ---: |
| 2048 | 整除（快路径） | 21.72 µs | 1.000000 |
| 3000 | clamp + 列掩码 | 26.42 µs | 1.000000 |
| 3008 | 整除（快路径） | 26.53 µs | 1.000000 |
| 4095 | clamp + 列掩码 | 28.93 µs | 1.000000 |
| 4096 | 整除（快路径） | 29.01 µs | 1.000000 |
| 33 | clamp + 列掩码（1 token 有效） | — | 1.000000 |
| 7 | 逐行填充兜底 | — | 1.000000 |

⇒ **非整除 ctx 的尾部开销已降到噪声级**（3000 vs 3008 差 −0.4%，4095 vs 4096 差 −0.3%），
且整除档零回归（2048 = 21.72 µs，与 §7.1 的 22.34 µs 在 ±3% 波动内）。

#### 2.2.6 整波族：把 `num_split` 取成 104 的倍数

`grid.z = num_split`，若 `num_split` 不是 AP 数（104）的倍数，最后一波就只有
`num_split % 104` 个 CTA 在跑。把 `ns` 固定成 `104 × w` 并用**块级均分**分配块数，
可以保证每一波都是满的：

```text
ns = 104 × w
blocks_per_cta, extra = divmod(ceil(kv_ctx / block_N), ns)     # 前 extra 个 CTA 多领一块
CTA bz 的起点块 = bz × blocks_per_cta + min(bz, extra)
                = blocks_per_cta + (bz < extra)
```

两种分片族（`_split_plan` 的 `spread` 开关）：

| 族 | 规则 | 特点 |
| --- | --- | --- |
| 等分族（`spread=False`） | `chunk` 对齐 `block_N` → `ns = ceil(ctx/chunk)` | `ns` 会被收敛，各 CTA 块数相同，Q 重读与 combine 轮数最少 |
| 整波族（`spread=True`） | `ns = 104×w` 原样保留，块级均分 | 每波满编，但 Q 重读 / combine 轮数随 `ns` 线性增长 |

实测（`block_N=32, threads=128`，固定配置；非 autotune 全流程，单位 µs）：

| kv_ctx | ns=64 | **ns=104**（1 波） | ns=208（2 波） | ns=256（等分） | ns=312（3 波） | ns=416（4 波） | 最优 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2048 | **21.79** | 不可行（104 > 64） | — | — | — | — | 等分 ns=64 |
| 4096 | **28.99** | 32.30 | 39.10 | — | — | — | 等分 ns=64 |
| 8192 | **44.49** | 47.61 | 56.31 | 65.19 | — | — | 等分 ns=64 |
| 16384 | 70.76 | **68.17** | 80.74 | — | 90.93 | 105.82 | 整波 ns=104 |
| 32768 | — | **113.97** | 130.01 | 127.27 | 142.00 | 157.08 | 整波 ns=104 |
| 65536 | — | 216.35 | 229.24 | **204.13** | 243.71 | 253.58 | 等分 ns=256 |

长 ctx 的逐段边际成本（每多切一份的平均代价）：

| kv_ctx | 64→104 | 104→208 | 208→312 | 312→416 | 256→312 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16384 | **−2.59** | +12.57 | +10.19 | +14.89 | — |
| 32768 | — | +16.04 | +12.00 | +15.08 | +14.73 |
| 65536 | — | +12.89 | +14.47 | +9.87 | +39.58 |

⇒ **`ns` 一旦超过「1 波」，每增加 104 个 CTA 约多花 10–16 µs**，且在所有长 ctx 上单调恶化。
312（3 波）/416（4 波）是当前实测的**最差区间**：16384 分别比 1 波慢 33%/55%，
32768 慢 25%/38%，65536 慢 19%/24%。因此 `_TUNE_WAVES` 只保留 `(1, 2)`，312/416 不再进候选。

三个结论：

1. **整波本身不是收益来源**。小 ctx 的瓶颈是单 CTA 关键路径延迟，`ns` 加大只增加
   Q 重读（`ns × 18 KB`）与 combine 的串行轮数，所以 104 反而更慢；
2. 长 ctx（16K / 32K）的收益来自「**`ns` 变小**」而不是「波次变满」：1 波时 Q 重读从
   `256 × 18 KB = 4.6 MB` 降到 1.8 MB、combine 串行轮数从 256 降到 104，而 104 个 CTA
   已经能覆盖整卡，于是净赚；
3. 64K 又回到并行度不足（每 CTA 19–20 块，关键路径太长），256 份仍更优；此时每多切一份
   的边际成本（256→312 是 +39.58 µs）说明该档已经对 `ns` 非常敏感，autotune 必须逐 shape 实测。

⇒ 因此把 `spread` 作为 autotune 的一个维度（候选 = 等分族 ∪ 整波族 `104/208`），由逐 shape
实测决定，而不是全局固定成"104 的倍数"。整波族在 32768/65536 上的 `cos_sim = 0.999999`
（>1e-6 量级），来自 `ns` 变大后 fp16 `glse` 的累加舍入，仍远高于验收线。

> 排查记录：`ns=312`（3 波）既不整除也不整波，旧实现会漏算末尾 8 个 token 且多出 56 个空 CTA；
> 现在的 `_split_plan` 会把它收敛到 `spread=False` 的等分计划（32768 → `ns=256`），
> 或在 `spread=True` 时按块均分（`ns=312`、前 `1024 % 312 = 88` 个 CTA 多领一块）。

#### 2.2.6.1 细粒度扫描：64 → 256 按步长 8/16 扫，没有更优点

用 `scripts/scan_ns.py`（`spread=True`，任意 `ns` 落到块级均分）在粗网格之间逐点实测：

| kv_ctx | 最优 | 次优 | 中间点的实测 |
| ---: | ---: | ---: | --- |
| 8192 | **ns=64 → 44.86** | ns=88 → 46.53 | 72 → 54.14 / 80 → 53.63 / 96 → 47.64 / 128 → 55.22 |
| 16384 | **ns=104 → 68.29** | ns=64 → 71.01 | 96 → 77.13 / 112 → **97.43** / 128 → 82.63 |
| 32768 | **ns=104 → 114.66** | ns=96 → 126.75 | 112 → **183.51** / 208 → 130.37 / 256 → 127.36 |
| 65536 | **ns=256 → 204.71** | ns=208 → 229.37 | 224 → 292.75 / 240 → 277.24 / 272 → 257.90 |

结论：**细网格没有收益，最优点只落在少数离散台阶上**，原因是两个不连续的台阶：

1. **波次台阶**：`ns > 104` 立刻进入第 2 波，而第二波只有 `ns − 104` 个 CTA。
   16384 的 `104 → 112` 直接慢 43%（68.29 → 97.43），32768 的 `104 → 112` 慢 60%（114.66 → 183.51）。
   所以 1 波内的最大值（104）总是比 105–207 之间的任何值好。
2. **块数台阶**：`ns ≤ 104` 时关键路径 = `ceil(nb/ns)` 块，只有跨过整数块数才会变快。
   72/80/88/96 与 104 都在 1 波内，但块数分别是 7/6/6/5 块，只有 104 能压到 5 块以下。

⇒ 因此 autotune 的 `ns` 候选保持 `{32, 64, 104, 128, 208, 256, 512, 1024}` 这类离散值即可，
不需要 8/16 的细步长；真正需要逐 shape 判定的只是「1 波（104）还是 0.6 波（64）还是多波（256）」。

> 64K 是唯一例外：`nb = 2048`，1 波意味着每 CTA 19–20 块（216 µs），
> 而 2.46 波的 `ns=256`（8 块）只要 204.71 µs —— 此时「块数」权重压过了「波次」。

#### 2.2.7 persistent 能做到什么程度

先厘清「persistent」在这个 kernel 里到底省什么。它通常指：grid 固定为 AP 数（或 AP 数的倍数），
每个 CTA 用循环领多个任务，从而 **(a) 任务级负载均衡、(b) 一次装载的 Q 跨任务复用、
(c) 消除波次量化、(d) 只 launch 一次**。

其中 **(b)(c) 已经拿到了**：Q 的 `T.copy` 在 `T.Pipelined` **之外**（每个 CTA 只装一次），
`spread=True + ns=104` 就是「1 波常驻 + 每 CTA 处理 `blocks_per_cta` 块」，
§2.2.6 里 16384 / 32768 的 +3.7% / +10.4% 正是这部分收益。剩下的差别只有 **(a)** 和 **(d)**。

| 形态 | 实现手段 | 实测 | 结论 |
| --- | --- | --- | --- |
| 连续分段（当前） | `spread=True`，CTA bz 处理 `[start, start+q)` 块 | 16384 = 68.32 µs，32768 = 114.01 µs | **采用** |
| 轮转分块 | CTA bz 处理块 `bz, bz+104, bz+208, …` | 16384 = 90.80（**+33%**），32768 = 164.42（**+44%**），8192 = 60.01（vs 47.61），65536 = 311.04（vs 216.35） | ✗ 负收益 |
| 动态工作队列 | `T.atomic_add(ptr, 1, return_prev=True)` + `T.While` 领块 | 未实做 | 本场景块数差已 ≤1，无可均衡余量，收益为 0，还多一次计数器 reset |
| 单 kernel 融合 combine | `T.sync_grid()` 后由部分 CTA 做归约 | 未实做 | 唯一还有价值的形态，但受 launch 限制（见下） |

轮转分块为什么慢：块步长变成 `104 × 32 = 3328` 个 token，每个 CTA 连续读的 36 KB 之间
隔了 3.3 K token，**L2 与预取的局部性被破坏**，而它换来的「负载差 1 块」在连续分段里本来就有。
所以 persistent 的「动态领任务」这一半在本 kernel 上没有价值，连续分段就是更优解。

**融合 combine 的可行性边界**：TileLang 的 `T.sync_grid()` 在 MACA 后端被降级为
`cooperative_groups::this_grid().sync()`（`codegen_maca.cc:1687`），但代码生成只置了
`need_cooperative_groups_`（用于 include 头文件），**launch 侧没有 cooperative kernel 的处理**。
普通 launch 下调用 `this_grid().sync()` 会挂死，因此这条路必须先写一个最小 kernel 验证
能否 cooperative launch，再谈收益：

- 省下的是一次 kernel launch（~1–2 µs）+ `output_partial` 的写读往返（1.18 MB，L2 命中）；
- 对 2048 档占 22.34 µs 的 10% 量级，对 64K 不足 1%；
- 风险是 hang（不是报错），且 cooperative launch 对 grid 大小有「必须全部驻留」的约束。

⇒ 结论：**persistent 的静态部分已经吃满，动态部分无收益，只剩「融合 combine」值得验证**，
且它的收益集中在小 ctx，与当前分数最低的档位正好重合。

#### 2.2.8 并行度扫描：最优点是「0.6–1 波、每 CTA 4–8 块」

把 `num_split` 从 32 扫到 256（`spread=False`，块数随之变化），8K/16K 都呈 U 型：

| kv_ctx | `ns` 有效 | 块/CTA | 波次 | 延迟 (µs) |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 32 | 8 | 0.31 | 65.79 |
| 8192 | 43 | 6 | 0.41 | 69.27 |
| 8192 | **64** | **4** | **0.62** | **44.36** |
| 8192 | 86 | 3 | 0.83 | 46.32 |
| 8192 | 128 | 2 | 1.23 | 55.14 |
| 8192 | 256 | 1 | 2.46 | 65.50 |
| 16384 | 32 | 16 | 0.31 | 116.56 |
| 16384 | **64** | **8** | **0.62** | **71.68** |
| 16384 | 104 | 5 | 1.00 | 68.17 |
| 16384 | 128 | 4 | 1.23 | 82.14 |
| 16384 | 256 | 2 | 2.46 | 87.92 |

两侧都变差：CTA 少于 0.6 波时单 CTA 的块间流水不足以藏住延迟；超过 1 波后每 CTA 只剩
1–4 块，Q 装载（18 KB）、首块预取、收尾写 partial 都摊不薄。8K 的 `ns=43`（0.41 波）
反而比 `ns=32`（0.31 波）慢，说明**非 2 的幂的切分数会打乱 L2 rasterization 的对齐**，
"少而整"的切分更稳。

**寄存器从来不是限制**：

```text
每 CTA 寄存器 = 166 reg × 128 thr ≈ 21.2 K reg
每 AP 寄存器堆 = 4 PEU × 512 reg × 64 thr = 131 K reg
⇒ 每 AP 可驻留 6 个 CTA
实际 CTA 密度 = 64 CTA / 104 AP ≈ 0.6 CTA/AP（寄存器用量 ~10%）
```

即使把 166 reg 降到 83，也只是让「可驻留 6 个」变成「可驻留 12 个」，
而排队等待驻留的 CTA 数是 0 —— **降寄存器不会让任何 CTA 早上场**。
降寄存器的可行手段本身也都带副作用：`acc_o[64]` 占 64 reg，要降就得按 dim 分块，
而 dim 分块意味着 QK^T 要重算或 KV 要重读；改 fp16 累加则直接损失精度；
`maxnreg` 强压会触发 spill（当前 `Private Read/Write = 0`）。

⇒ **占用率方向可以关闭**：瓶颈是「并行度已经过剩 + 每 CTA 的固定开销」，
不是寄存器或驻留能力。唯一还能提高内存级并行的是 `num_stages=2`，但它被 shared 顶死：
`Q 18 KB + 2 × KV 36 KB + S 1 KB = 91 KB > 64 KB`（把 Q 放进寄存器也还有 73 KB）。

#### 2.2.9 prologue 的成本拆解：可压的只有 ~15%

用 8K 的并行度扫描反推「每多一份 split 的固定成本」：
`ns=64 → 128` 慢 10.78 µs，即 **168 ns/份**。拆开看：

| 成分 | 每份成本 | 可压性 |
| --- | ---: | --- |
| `Q/Q_pe` 装载 18 KB | ~11 ns | 只靠重叠（实测把 global 读整体去掉，64K 省 0.99 µs、16K 省 0.72 µs） |
| `acc_o/logsum/scores_max` 清零 | ~5 ns | 已重排到 Q 装载**之前**，与 MTE 并行 |
| 首块 KV 预取 36 KB | ~90 ns | 数据本身必需，除非 `num_stages=2`（被 shared 否掉） |
| CTA 启动 / 收尾写 partial | ~60 ns | 硬件调度与协议成本，代码不可控 |

已做的重排（清零 → Q 装载）：8192 = 44.36 → **43.99 µs**（−0.8%），16384 = 71.06 / 2048 = 21.82
均在噪声内 —— 说明编译器本来就能把无依赖的寄存器写与 MTE 重叠，**prologue 的 85% 压不动**。

若一定要继续，只剩两条都需要拆流水的路：

1. **手动 prologue**：把 KV 首块（36 KB）的 copy 提到 Q 之前，让两条 MTE 同时在途，
   省一次装载延迟（约 0.5–1 µs/波），代价是首块要手写计算体、破坏 `T.Pipelined` 的自动调度；
2. **单块特化**：`blocks_per_cta == 1` 时省掉 `acc_o *= scores_scale` 与 `logsum` 重缩放
   （8192 次 FMA/块），但它只对 `ns = nb`（每 CTA 1 块）的配置生效，而那种配置本身
   就慢 30–47%（§2.2.8），收益场景不存在。

⇒ 结论：**prologue 的优化空间已见底（≤1 µs，1–2%）**，继续投入不如把同一份工程量放到
combine 或 `num_split` 选型上。

---

## 3. 计算流程

### 3.1 主 kernel（online softmax + split-KV）

CTA 内先一次性把本 CTA 的 Q 搬到 shared，然后对**本 split 覆盖的 KV** 逐块（`block_N=32`）循环：

```text
① 预取（软件流水线的 prologue）：KV / K_pe 首块 → shared

② 对每个 KV 块 k（共 loop_range = chunk/block_N 次）：
   ├─ KV[bid, kv_start+k·32 : …+32, 0, :] → KV_shared     （swizzled，coalesced_width=8）
   ├─ K_pe[同区间]                        → K_pe_shared
   ├─ acc_s  = Q_shared  @ KV_sharedᵀ        (FullCol)
   ├─ acc_s += Q_pe_shared @ K_pe_sharedᵀ    (FullCol)      ← 两块拼成 576 维点积
   ├─ scores_max = reduce_max(acc_s, clear=False)           ← 运行最大值，省掉每轮 fill(-inf)
   ├─ scores_scale = exp2(prev_max·s − max·s)               ← 旧块结果的重缩放系数
   ├─ acc_s = exp2(acc_s·s − max·s)                         ← s = softmax_scale·log2(e)
   ├─ scores_sum = reduce_sum(acc_s)
   ├─ acc_s → S_shared                                      （为 PV 准备；寄存器直喂会布局冲突，见 §10.6）
   ├─ logsum = logsum·scores_scale + scores_sum
   ├─ acc_o *= scores_scale                                 （把历史 PV 结果一起重缩放）
   └─ acc_o += S_shared @ KV_shared         (FullCol)       ← PV，注意 V 就是 KV 主部

③ 收尾：
   ├─ inv_logsum = 1/logsum          （16 次除法）
   ├─ acc_o *= inv_logsum            （16×512 次乘法，替代逐元素除法）
   ├─ logsum = log2(logsum) + max·s  （写成 LSE，供 combine 用）
   ├─ logsum → glse[bid, hid, bz]
   └─ acc_o  → output_partial[bid, hid, bz, 0:dim]
```

三处值得记录的数值/实现取舍：

1. **`log2`/`exp2` 而不是 `ln`/`exp`**：把 `softmax_scale` 预先乘上 `log2(e)=1.44269504`，
   全程走 `exp2/log2`，避免每次调用 `exp` 的额外乘法；
2. **`reduce_max(clear=False)` 复用运行最大值**：省掉每轮的 `T.fill(-inf)` 与一次并行 `max`；
   代价是不能处理全 `-inf` 行，但 decode 场景 `kv_ctx ≥ 1` 时不会出现；
3. **只算倒数不做逐元素除法**：`block_H×dim = 8192` 次除法被替换为 16 次除法 + 8192 次乘法。

### 3.2 combine kernel（log-sum-exp 归约）

```text
① glse[bid, hid, 0:glse_size] → glse_shared（T.copy，glse_size = ceil(num_split/64)·64）
② padding 区 [num_split, glse_size) 填 −inf
③ 并行装载 → lse_frag[1, glse_size]（fp32 fragment，每线程 glse_size/64 项）
   m = T.reduce_max(lse_frag, dim=1)                    ← 树形归约，O(log num_split) 步
④ lse_frag ← exp2(lse_frag − m)（并行）
   l = T.reduce_sum(lse_frag, dim=1)                    ← 树形归约
   LSE = log2(l) + m
⑤ for k in 0 … num_split-1:            ← 仍是串行，但每轮 64 路并行实干（不是空转）
       scale = exp2(glse_shared[k] − LSE)
       po_local ← output_partial[bz, hid, k, dim_start : dim_start+64]   （T.copy → 寄存器）
       o_accum[0:64] += po_local · scale                                  （纯寄存器 FMA）
⑥ o_accum → Output[bz, hid, dim_start : dim_start+64]
```

**用 `T.copy` 把 global 的 partial 一次性倒进寄存器、再做 FMA**，而不是在并行循环里直接读 global ——
避免逐元素 global 读的地址计算开销。

#### 3.2.1 为什么 ②③④ 必须并行化（原实现是纯串行标量链）

原实现：

```python
for k in T.serial(num_split): lse_max = T.max(lse_max, glse_shared[k])
for k in T.serial(num_split): lse_logsum += T.exp2(glse_shared[k] - lse_max)
```

`lse_max` / `lse_logsum` 是 `T.alloc_var`（**标量**），这两条链有三个问题：

1. **纯串行**：链长 `2 × num_split`，每步都依赖前一步；
2. **64 个线程冗余算同一份标量**：63 个线程全程空转，等于用 1 个线程的吞吐跑；
3. **每步只有一次 shared 读 + 一次 max/exp2**，单步延迟吃满却无吞吐，实测每步约 **20–30 ns**。

于是这部分开销 ≈ `2 × num_split × 25 ns`，与 `dim_chunk_num`、与 KV 长度都无关，
只跟 `num_split` 成正比：

| num_split | 串行步数 | 端到端实测可省 | 每步折算 |
| ---: | ---: | ---: | ---: |
| 64 | 128 | **2.10 – 2.30 µs** | ≈ 17 ns |
| 128 | 256 | **7.25 µs** | ≈ 28 ns |
| 256 | 512 | **15.97 µs** | ≈ 31 ns |

> 每步折算随 `num_split` 上升而变大：链越长，shared 读与 `exp2` 之间越难流水，
> 纯延迟被完整暴露出来。

改成 fragment + `T.reduce_max/reduce_sum` 后，链长从 `2·num_split` 降到 `O(log num_split)`，
且 64 个线程全部参与。

#### 3.2.1.1 固定配置 A/B（隔离 phase2 的纯贡献）

同一套 phase-1 参数（`block_N=32, block_H=16, num_stages=1, dim_chunk_num=8,
q_in_reg=False, threads=128`）下，只改 combine 的归约实现：

| kv_ctx | num_split | 改造前 (µs) | 改造后 (µs) | 变化 |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 64 | 24.24 | **22.14** | −8.65% |
| 4096 | 64 | 31.84 | **29.54** | −7.23% |
| 8192 | 64 | 46.67 | **44.45** | −4.76% |
| 16384 | 64 | 73.85 | **71.71** | −2.90% |
| 32768 | 128 | 140.20 | **132.95** | −5.17% |
| 65536 | 256 | 220.49 | **204.52** | −7.24% |
| | | **537.29** | **505.30** | **−5.95%** |

小 ctx 的相对收益更大（2048 档 −8.65%），因为同样 128 步串行链在小 ctx 的总时间里占比更高
——这与 §2.2.4「小 ctx 由固定开销主导」是同一件事。

#### 3.2.1.2 附带效果：autotune 在 32768 换选了更大的 `num_split`

phase2 的串行惩罚消失后，`num_split` 变大不再被重罚，autotune 在 32768 档从
`num_split=128`（chunk 256）改选 `num_split=256`（chunk 128），该档再降到 **127.27 µs**。
端到端（含重新选型）总时 **537.29 → 497.73 µs（−7.4%）**，6 档全部 `cos_sim = 1.000000`。

#### 3.2.2 padding 为什么填 −inf

`glse_size` 向上取整到 64 的倍数（让归约列数能被一个 wavefront 整除），
`[num_split, glse_size)` 是未初始化区。填 **−inf** 后：

- 对 `max` 无影响（−inf 不会成为最大值）；
- `exp2(−inf − m) = 0`，对 `sum` 也无影响。

一份填充同时满足两个归约，**不需要为 `num_split` 不是 64 的倍数单独分支**
（旧的 `num_split=312` 也是安全的）。

#### 3.2.3 ⑤ 号循环为什么没动

它每轮做的是 `dim_per_chunk = 64` 个元素的 global→寄存器拷贝 + FMA，**正好铺满 64 个线程**，
没有空转；串行的原因只是寄存器累加 `o_accum +=` 的归约语义。
试过改成 `T.Pipelined(num_split, num_stages=2)` 预取下一块，实测无收益
（70.97 vs 70.98 µs，204.72 vs 204.61 µs，均在噪声内），已回退为 `T.serial`。

#### 3.2.4 k 维分组：把串行的 `num_split` 轮拆成 `num_split/G` 轮

⑤ 循环的串行轮数就是 `num_split`，这是大 `num_split` 不可行的主因。把 k 维切成 G 组，
用两个 kernel 完成归约：

```text
stage-1  grid = (heads, batch, dim_chunk_num × G)，threads=64
         每个 CTA 仍独立算全局 LSE（读 glse + 两次树形归约），
         但只累加自己那 num_split/G 份 partial，写 partial2[..., cg, :]
stage-2  grid = (heads, batch, dim_chunk_num)，threads=64
         把 G 份 partial2 直接相加（LSE 权重已在 stage-1 乘好）
```

串行轮数从 `num_split` 降到 `num_split/G + G`，代价是 G 倍重复的 LSE 计算（每次 1–2 KB 的
`glse` 读，全部 L2 命中）+ 一次额外 launch。`T.Kernel` 最多 3 维 grid，所以第三维是
`dim_chunk_num × G`，用 `dim_grp % dim_chunk_num` / `dim_grp // dim_chunk_num` 拆回。

实测（`block_N=32, threads=128`，64K）：

| `ns` | G=1 | G=8 | G=16 | G=32 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 204.13 | 197.29 | **197.03** | 199.22 |
| 512 | 217.46 | 198.44 | **197.10** | — |
| 1024 | 300.50 | — | 255.07 | 259.92 |

⇒ 64K 从 204.13 降到 **197.03（−3.5%）**，且 `ns=512` 从 217.46 拉回到 197.10（与 256 持平），
说明分组确实把 combine 的串行链砍掉了 20–45 µs。

但**分组救不了小 ctx 的大 `num_split`**：

| kv_ctx | 等分族最优 | 大 `ns` + 分组 | 结论 |
| ---: | ---: | ---: | --- |
| 2048 | 21.79（ns=64） | 21.92（ns=64, G=8） | G 无收益，多一次 launch 反而略亏 |
| 8192 | 44.49（ns=64） | 58.01（ns=256, G=8） | 大 `ns` 仍慢 30% |
| 16384 | 68.17（ns=104） | 96.12（ns=512, G=8） | 大 `ns` 仍慢 41% |
| 65536 | 204.13（ns=256） | **197.03**（ns=256, G=16） | 唯一有效档，−3.5% |

⇒ 压掉 combine 之后，限制大 `num_split` 的变成**主 kernel 每 CTA 的 prologue**：每 CTA 只处理
1 块时，Q 装载（18 KB）、首块预取、收尾写 partial 都摊不薄。只有 64K 因为总流量大、主循环
足够长，combine 才够资格当瓶颈。autotune 里只在 `num_split ≥ 128` 时加入 G 候选（8/16），
避免小 ns 白编译。

---

## 4. Block 分发

### 4.1 grid 与线程

| kernel | grid | 线程 | 每 CTA 负责 |
| --- | --- | --- | --- |
| `main_kernel` | `(batch, heads/block_H, num_split)` = `(1, 1, num_split)` | 128（= 2 wave） | 一个 (batch, head 组, KV 分片) |
| `main_kernel_1` | `(heads, batch, dim_chunk_num)` = `(16, 1, 8)` = 128 | 64（= 1 wave） | 一个 (head, batch, 输出 64 列) |

@16384：主 64 CTA + combine 128 CTA = **192 CTA**，落在 104 个 AP 上（≈1.85 CTA/AP）。

### 4.2 为什么并行度全部来自 KV 切分

`heads=16` 且 `block_H=16` ⇒ `heads / block_H = 1`，**head 维只出 1 个 CTA**；
`batch=1` ⇒ batch 维也只剩 1。于是主 kernel 的 3 个 grid 维度里，只有 `num_split` 真正提供并行度。
这正是「split-KV 不是可选优化而是必需」的原因：不切分就只有 1 个 CTA，整卡 104 个 AP 里用 1 个。

### 4.3 光栅化

主 kernel 与 combine 都启用 `T.use_swizzle(panel_size=10, enable=True)`，
按 `blockIdx` 重映射做 L2 友好的 rasterization（实测 L2C 命中率 32%，见 §6）。

### 4.4 CTA → 硬件映射

C500 的组织是 `8 DPC × 13 AP = 104 AP`，每个 AP 有 **4 个 PEU**；
block（CTA）调度到 AP 上，wave 调度到 PEU 上。约束：
`Max work-items/AP = 2048`、`Max waves/AP = 32`、`Max workgroup size = 1024`。
本 kernel 每 CTA 2 wave（主）/ 1 wave（combine），寄存器与 shared 都不触碰上限（见 §5）。

### 4.5 `threads` 与 `block_N` 的耦合关系

这两个参数**不能独立选**，它们被三条互相夹击的约束锁在一起，最终解唯一：

| 约束 | 来源 | 对 `block_N` / `threads` 的作用 |
| --- | --- | --- |
| `block_N ≥ 16 × threads/64` | `FullCol` 的 warp 映射 | 给定 `block_N` 时封顶 `threads`；给定 `threads` 时托底 `block_N` |
| shared ≤ 64 KB/AP | `Q + num_stages×(KV+K_pe) + S` | 封顶 `block_N ≤ 32` |
| `acc_o` 每线程寄存器 = `8192/threads` | AP 寄存器堆 / 占用率 | 托底 `threads ≥ 128` |

#### 4.5.1 硬约束：`block_N ≥ 16 × (threads/64)`

主 kernel 的三个 GEMM 全部用 `T.GemmWarpPolicy.FullCol`：

```81:81:mla_decode/mla_decode.py
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
```

`acc_s` 的形状是 `[block_H, block_N] = [16, 32]`。`FullCol` 的含义是**只沿列方向（N）切
warp、不切 M 方向**——因为 M 维就是 `block_H=16`，正好等于一个 MMA tile 的高，再切就凑不出
16 行。于是 CTA 内的 `threads/64` 个 warp（C500 一个 wavefront = 64 线程，
`block=(128,1,1)` 即 2 wave）只能平摊 `block_N`：

```text
每个 warp 分到的列宽 = block_N / (threads / 64)
```

每个 warp 至少要覆盖**一个完整的 16 宽 MMA tile**，即

```text
block_N / (threads / 64) ≥ 16   ⇔   block_N ≥ 16 × threads/64   ⇔   threads ≤ 4 × block_N
```

这正是 `_threads_feasible()` 的实现：

```327:340:mla_decode/mla_decode.py
def _threads_feasible(block_n, block_h, threads):
    """主 kernel 的 warp 映射约束。
    ...
    即 `block_N >= 16 * threads/64`。这也解释了源码注释里"threads=128 需要 block_N>=32"。
    ...
    """
    if threads < 64 or threads % 64:
        return False
    if block_h > 16:      # 超出 16 行时 block_H 也要能被 warp 切分
        return False
    return block_n >= 16 * (threads // 64)
```

实测边界（`heads=16, kv_ctx=16384, num_split=64, block_H=16, num_stages=1`）：

| `block_N` | `threads` | warp 数 | 每 warp 列宽 | 结果 |
| ---: | ---: | ---: | ---: | --- |
| 32 | 128 | 2 | 16 | **OK — 73.63 µs（生产配置）** |
| 32 | 64 | 1 | 32 | OK — 288.76 µs（§4.5.2） |
| 32 | **256** | 4 | **8** | ✗ `Check failed: pb->value != 0 : Divide by zero` |
| 16 | 64 | 1 | 16 | OK — 482.48 µs（§4.5.3） |
| 16 | **128** | 2 | **8** | ✗ `Divide by zero` |
| 64 | 64 | 1 | 64 | ✗ shared 94720 B > 65536 B |
| 64 | 256 | 4 | 16 | ✗ `Vector size 4 is not divisible by coalesced width 8` |

列宽 < 16 时每个 warp 连一个 MMA tile 都拼不出来（tile 数 = `(N/warp)/16` 取整为 0），
TileLang 在推导 MMA 布局时做整除就崩了。**这是编译期错误**，不是运行期数值问题——
所以线上永远不会「跑出错误结果」，只会在首次编译新 shape 时直接失败。

#### 4.5.2 上界：`block_N ≤ 32` 由 shared memory 定死

```text
shared = Q   : block_H × (dim+pe_dim) × 2      = 16 × 576 × 2      = 18432 B
       + KV  : num_stages × block_N × 576 × 2  = 1  × block_N × 1152
       + S   : block_H × block_N × 2           = block_N × 32
       ≤ 65536 B
⇒ block_N × 1184 ≤ 47104  ⇒  block_N ≤ 39.8
```

`block_N` 必须是 16 的倍数 ⇒ **只能取 32**。取 64 时仅 KV 一项就 `64×576×2 = 73728 B`
已经超限（加上 Q/S 共 94720 B），实测直接报
`Shared memory size error ... from user 94720 in device 65536`。

注意这个上界**与 `threads` 无关**：`num_stages=1` 下 `block_N=64` 无论多少线程都放不下。
所以 `threads=256`（需要 `block_N ≥ 64`）和 64 KB 上限**不可兼得**，这也是候选集里
`threads` 只开到 128 的原因。

> **沿 latent 维（512）切分能不能解开这个锁？** 数学上可以：`QK^T` 沿 K 维累加、
> `PV` 沿 N 维分块，两种切法都对。shared 账如下（KB）：
>
> | 方案 | Q | KV | K_pe | S | 合计 |
> | --- | ---: | ---: | ---: | ---: | ---: |
> | 现状 `block_N=32, dim=512` | 18 | 36.0 | 4.0 | 1.0 | **59.0** |
> | `block_N=64, dim=512` | 18 | 73.7 | 8.2 | 2.0 | 101.9 ✗ |
> | `block_N=64, dim 切半` | 18 | 32.0 | 8.2 | 2.0 | **60.2** ✓ |
> | `block_N=32, dim 切半, ns=2` | 18 | 2×18.4 | 4.1 | 1.0 | 64.0（贴着上限） |
> | `block_N=32, dim 切 1/4, ns=2` | 18 | 2×8.2 | 4.1 | 1.0 | 43.6 ✓ |
>
> 但**分块本身不产生收益**：总流量与总 MAC 都不变，只是把一次 GEMM 拆成两次。
> 它能换来的只有配置自由度，而这两个自由度都不划算：
>
> 1. `block_N=64` + dim 切半 → 轮数减半，每轮固定开销（`acc_o *= scale` 的 8192 次 FMA）
>    随之减半；但摊到 256 线程后每轮只有 32 条指令，量级 **<1%**；
> 2. `num_stages=2`（dim 切 1/4 才放得下 43.6 KB）→ 才是真正对症 `vls_pipeline_stall`
>    （48%–56%）的方向，但它要求「4 轮凑齐一个 token 块的 `QK`」，与 `T.Pipelined`
>    每轮独立迭代的语义冲突，必须**手写双缓冲**（放弃自动流水，自己发 prefetch、自己等 barrier）。
>
> **实测（dim 切半 + 两遍扫描，block_N=32/64 都通过数值校验 cos=1.000000）：**
>
> | kv_ctx | 现状 `block_N=32` | dim2 `block_N=32` | dim2 `block_N=64` |
> | ---: | ---: | ---: | ---: |
> | 2048 | **22.34** | 23.52（+5%） | 27.18（+22%） |
> | 8192 | **44.89** | 125.58（+180%） | 106.13（+136%） |
>
> 两个致命点：
>
> 1. **`T.Pipelined` 直接拒绝这种写法**：`Multiple writes to overlapping buffer regions
>    detected. Stage 1 and stage 3 are both writing to buffer 'KV_s'` —— QK 两半 + PV 两半
>    要写同一个 `KV_s` 四次，流水规划要求每个 buffer 在循环体内只被一个 stage 写。
>    退回 `T.serial` 后预取消失，8192 直接崩到 125 µs。
> 2. 即使手写双缓冲绕开这点，**KV 流量仍要 ×2**（QK 扫一遍、PV 再扫一遍），
>    与 `block_N=64` 省下的那 <1% 完全不成比例。
>
> ⇒ 结论：dim 分块在当前框架下走不通，`block_N` 也不可能靠它变大。

**补充：`block_N=48` + Q 入寄存器（省 18 KB）能否成立？** shared 账是够的
（KV 48 + K_pe 6 + S 1.5 = **55.5 KB**），但被另外两道约束挡住：

1. **warp 粒度**：48 列 = 3 个 16 列片，需要 3 个 warp；`threads=128` 只有 2 个 warp，
   直接编译失败 `m_warp * n_warp must equal num_warps, m_warp: 1, n_warp: 3, num_warps: 2`。
   所以 `block_N=48` 只能配 `threads=64`；
2. **寄存器预算**：`threads=64` 下 `Q` 入寄存器 = 9216/64 = 144 个 fp16（72 reg），
   `acc_o` = 8192/64 = 128 reg，再加 `acc_s`(12) 与 MMA B 操作数(64) 已到 276 reg，
   超过 255 上限 → spill。

实测（2048 档）：`block_N=32, threads=128, Q→shared` = **21.75 µs**；
`block_N=48, threads=64, Q→寄存器` = **39.19 µs（慢 80%）**。
内嵌 C 的布局转换解决不了这两点——它换的是「fragment 之间怎么搬」，不改变 warp 粒度与寄存器预算。

**那 `threads=192`（3 warp，正好配 48 列）呢？** 也编译不过。`T.gemm` 的 warp 切分要求
`m_warp × n_warp == num_warps`，而 PV 的输出是 `[16, 512]`，N 维给出 `n_warp = 512/16 = 32`：

```text
m_warp: 1, n_warp: 32, num_warps: 3      # bn=48/96, th=192
m_warp: 1, n_warp: 2,  num_warps: 3      # bn=32,   th=192（卡在 QK 的 n_warp）
```

`block_N=32/48/96` × `threads=192` 三组全部失败。可行的是只有 **1 warp（64）与 2 warp（128）**，
因为 num_warps 必须能整除 `n_warp`（2 的幂）；4 warp（256）需要 `block_N ≥ 64`，而 shared 放不下。
⇒ `threads=192` 不可用。

### 11.9 合并 `QK^T` 为一次 K=576 GEMM（已评估：收益 <1%，实现受阻）

现状每块做两次 GEMM：`Q_nope[16,512] @ KV^T[512,32]`（K=512）与 `Q_pe[16,64] @ K_pe^T[64,32]`（K=64）。
想法是把 KV/K_pe 装进同一个 `[32, 576]`、Q/Q_pe 装进 `[16, 576]`，合成一次 K=576 的 GEMM，
省掉 K=64 那次的 prologue/epilogue（shared 占用不变，因为 576 = 512+64）。

先用「跳过 `K_pe` 那次 GEMM」的探针量化收益上限（数值会错，只测时间）：

| kv_ctx | 完整 | 跳过 `K_pe` GEMM | 差值 |
| ---: | ---: | ---: | ---: |
| 2048 | 22.11 | 21.82 | **0.29 µs（1.3%）** |
| 8192 | 44.74 | 43.83 | **0.91 µs（2.0%）** |
| 65536 | 197.01 | 202.08（噪声内） | ~0 |

⇒ 收益上限 1%–2%，且合并后 MMA 个数不变（`576/16 = 36` 与 `32+4 = 36` 相同），
真正能省的只是 GEMM 调用的固定开销，**实际预期 <1%**。

实现时被 TileLang/TVMScript 的两个限制连续挡住：

1. **parser 会遍历所有分支**：`if merge_pe_eff:` 成立时 `KV_shared` 未分配，但 `elif tail_mode == 2:`
   与 `else:` 分支里的 `KV_shared` 引用仍被求值 →
   `UnboundLocalError: cannot access local variable 'KV_shared'`；
2. **shared 切片不能先存成变量再传给 `T.copy`**：`K_pe_shared = KV_shared[:, dim:dim+pe_dim]` 之后
   `T.copy(..., K_pe_shared)` 会报
   `Expected BufferLikeType, got <class 'tvm.tirx.expr.Var'>` —— 切片只能在调用处内联。

⇒ 为绕过这两点需要保留 6 KB 空 buffer 并给每个分支写条件，**收益 <1% 而代价不小**，
因此放弃合并，代码已回退。作为记录：**`K=64` 那次 GEMM 只占总时间 1%–2%，不是瓶颈**。

### 11.10 为什么「按 dim 分块」与 `T.Pipelined` 天然冲突

报错原文：

```text
Pipeline planning error: Multiple writes to overlapping buffer regions detected.
Stage 1 and stage 3 are both writing to buffer 'KV_s' with overlapping regions.
This is not supported in pipeline planning.
```

**`T.Pipelined` 的规划模型**：它把循环体拆成若干 stage（copy 一个、compute 一个），
然后为每个 buffer 决定三件事——① 谁是唯一 producer；② 需要几份缓冲（`num_stages`）；
③ 在哪些位置插 barrier。这三件事都依赖「**同一迭代内，一个 buffer 只有一个 writer**」这一前提。

**dim 分块打破了它**：为了在一次迭代里算完一个 token 块，`KV_s` 必须被顺序复用 4 次：

```text
写 KV_s(half0) → GEMM(QK, half0)
写 KV_s(half1) → GEMM(QK, half1)          ← 覆盖 half0（WAW）
    softmax（依赖两半的 QK 结果）
写 KV_s(half0) → GEMM(PV, half0)          ← 再覆盖（WAW）
写 KV_s(half1) → GEMM(PV, half1)
```

四次写落在不同的 pipeline stage 上，规划器无法确定 `KV_s` 的生命周期边界，
也就无法回答「什么时候可以安全地被下一轮的预取覆盖」，于是直接拒绝。

注意这与 `num_stages` 无关：**即使 `num_stages=1` 也会报**，因为 stage 划分（copy/compute 分离）
本身就是流水规划的一部分，而 `num_stages=1` 只决定缓冲份数，不取消 stage 划分。

**为什么非分块的现状没问题**：每轮只写 `KV_s` 一次、GEMM 读一次，是标准的「单 producer」形状，
规划器可以正常做 stage 分离与预取。

**有没有绕法**：

| 绕法 | 结果 |
| --- | --- |
| 每半用独立 buffer（KV0/KV1） | QK 阶段 OK，但 PV 阶段要**再写一次** KV0/KV1 → 仍是 WAW，同样报错 |
| PV 再开两个 buffer（共 4 份） | `4 × 18.4 = 73.7 KB` > 64 KB，放不下 |
| 退回 `T.serial` | 失去预取，8192 实测 44.6 → **125.6 µs**（+182%） |
| 手写双缓冲 + barrier | 可行，但要放弃 `T.Pipelined`，等于手写 kernel（§11.8） |

⇒ dim 分块要求「单 buffer 在一次迭代内被反复覆盖」，而流水规划要求「单 buffer 单 writer」，
这两个要求在这个 shared 预算下无法同时满足。这是框架流水模型与 buffer 复用的结构性冲突，
不是参数能调的。

### 11.11 能不能用 local memory 换 shared 空间

`T.alloc_local` / 寄存器溢出用的 local memory 是**每线程私有**的（物理上落在显存，走 L1/L2 缓存），
延迟介于 shared（~20–30 cycle）与 global（~600 cycle）之间。直觉上「用慢一点的内存换容量」是对的，
但在这个 kernel 里它替代不了 shared，原因不是速度，而是 **MMA 的操作数契约**：

| 操作数 | 允许的来源 | 原因 |
| --- | --- | --- |
| GEMM 的 **B**（`KV`） | **只能 shared** | 由 `ldmatrix` 从 shared 取，要求跨线程共享的数据视图 |
| GEMM 的 **A**（`Q`、`S`） | shared，或**按 A 布局分布的 fragment** | 需要特定的「线程↔元素」映射（§10.6） |
| 累加器 **C**（`acc_o`） | fragment（每线程私有） | 天然私有，但 fp32 累加器不能放 local（每次读改写太慢） |

local memory 是「每线程一份、无布局语义」的，它既不能被 `ldmatrix` 按 warp 协作读取，
也不能满足 A 操作数的布局要求。把 `KV` 挪进去，GEMM 根本拿不到数据；把 `Q` 挪进去，
就退化成 `q_in_reg` 路径（实测慢 13%–64%，因为 `global→fragment` 是散列访问）。

逐个看四个大块：

| 块 | 大小 | 能否挪出 shared | 结论 |
| --- | ---: | --- | --- |
| `KV`（B 操作数） | 36 KB | **不能** | MMA 的 B 必须来自 shared |
| `Q`（A 操作数） | 18 KB | 能（→fragment） | 已实测：慢 13%–64%（§4.5） |
| `acc_o`（C） | 32 KB（寄存器） | 能（→local） | 每次 PV 累加都要读改写 256 B/线程，且占用率本就不是瓶颈（§2.2.8） |
| `S`（A 操作数） | 1 KB | 不能（→fragment 失败） | 布局冲突，§10.6 |

⇒ **local memory 解锁不了 `num_stages=2`**：双缓冲需要两份 `KV`（72 KB），而 `KV` 只能待在 shared，
64 KB 的墙仍然在。local 唯一能承接的是「寄存器溢出」，而 spill 本身就是负收益
（当前 `Private Read/Write = 0`，一旦 spill 就变慢）。

### 11.12 根因：`num_stages` 在 MACA 后端根本不生成 `cp.async`

把两种配置的生成代码 dump 出来数原语：

| 配置 | 行数 | `cp_async`/`mbarrier`/`commit_group` | `__syncthreads` |
| --- | ---: | ---: | ---: |
| `block_N=32, threads=128, num_stages=1` | 272 | **0** | 14 |
| `block_N=16, threads=64, num_stages=2` | 353 | **0** | 15 |

**即使 `num_stages=2`，MACA 后端也没有生成任何异步拷贝**——拷贝依然是
「`ld.global` + `st.shared` + `__syncthreads`」的同步序列。`num_stages=2` 只是多分配了一份
buffer，然后**照旧阻塞地等数据到齐**，所以：

- 双缓冲没有产生任何「搬运与计算重叠」；
- 反而多占了 18 KB shared、多了一次 `__syncthreads`；
- 这解释了为什么 `block_N=16 + ns=2` 比 `ns=1` 还慢（§4.5.2 补充段的 +46%/+142%/+234%）。

⇒ 之前所有「shared 放不下双缓冲」的讨论其实绕过了真正的瓶颈：**MACA 后端的流水降级里没有
`cp.async` 这一环**。TileLang 本身提供了 `T.async_copy` / `T.maca_async_copy(..., barrier=...)`
（`copy_op.py:178`、`allocate.py:240`），MACA codegen 里也有 `cp.async` 的发射能力
（`codegen_maca.cc:1514` 起），只是**自动流水没有用它**。

**破局点因此变成「手写异步双缓冲」，但仍受两道约束**：

1. **shared**：异步双缓冲同样要两份完整 `KV`（72 KB）+ Q（18 KB）= 90 KB > 64 KB。
   只有 `block_N=16` 能塞下（59 KB），而它又撞上第 2 条；
2. **warp 粒度**：`block_N=16` 在 TileLang 的 C++ policy 里只能配 `threads=64`（1 warp），
   `threads=128` 会报 `Divide by zero`（policy 按 16 列分 warp，而硬件 MMA 的 n8 粒度本可支持 8 列/warp）。

所以真正值得试的是：**用 `T.maca_async_copy` 手写 `block_N=16, threads=64` 的双缓冲**，
看异步能否补回 1 warp 的算力损失；或者改 TileLang 的 C++ policy 让 `block_N=16` 支持 2 warp。

**探针实现结果（`scripts/probe_async_pipe.py`，已写出但不可用）**：

- `T.async_copy` 在 MACA 上会被重定向到 `T.maca_async_copy`，而后者**强制要求 barrier**：
  `T.maca_async_copy(src, dst, barrier=bar[i])` + `T.maca_barrier_arrive_and_wait(bar[i])`；
- 按此写完双缓冲（两份 KV/K_pe、unroll 2、barrier 0/1 交替）后，**编译器 core dump**：

```text
TileLang begins to compile kernel `main`
timeout: the monitored command dumped core
```

即崩溃发生在 lowering 阶段（不是运行时），说明 `maca_async_copy` + `alloc_maca_barrier` +
swizzled layout 这套组合在 MACA 后端还不稳定。继续需要做 TileLang 源码级调试
（barrier 语义、与 swizzle 的兼容性、copy 的 lowering），已超出应用层能解决的范围。

⇒ **方案 A 停在「API 存在但不可用」**：脚本保留在 `scripts/probe_async_pipe.py` 作为后续起点，
生产路径未改动。

### 11.13 方案 B 的结论：MACA MMA 的 N 粒度是 16，`block_N=16` 无法 2 warp

方案 B 原本打算改 C++ 的 `compute_warp_partition` 让 `block_N=16` 配 2 warp。
先定位真实崩溃点，结果发现**卡点不在 C++ policy，而在 MACA 的 MMA micro tile**：

```text
tilelang/maca/intrinsics/macro/mma_macro_generator.py:106
    self.warp_cols = warp_col_tiles // self.micro_size_y
                                    ↑ block_N/n_warp = 16/2 = 8
                                              ↑ micro_size_y = 16   → 8 // 16 = 0 → Divide by zero
```

```python
# mma_macro_generator.py:188
def _initialize_micro_size(self, m_dim=16, n_dim=16, k_dim=16):
```

即 **MACA 的 MMA 指令是 `m16n16k16`**（NVIDIA 是 `m16n8k16`），每个 warp 在 N 方向至少要覆盖 16 列。
因此：

| 配置 | 每 warp 的 N | 结果 |
| --- | ---: | --- |
| `block_N=16, threads=64`（1 warp） | 16 | ✓ 合法（但算力只有一半） |
| `block_N=16, threads=128`（2 warp） | 8 | ✗ < 16，`warp_cols=0` 除零 |
| `block_N=32, threads=128`（2 warp） | 16 | ✓ 合法（现状） |

⇒ **改 C++ policy 也救不了**（下层 generator 会先除零），所以方案 B **不需要复制环境、也不需要重编译**。
与 64 KB shared 组合起来就是死锁：

```text
要 2 warp  ⇒ block_N ≥ 32  ⇒ 双缓冲需 2×36 + Q 18 = 90 KB > 64 KB
要 block_N=16 ⇒ 只能 1 warp ⇒ MMA 算力腰斩（实测慢 6.55×）
```

**结论：在 MACA（n16 MMA）+ 64 KB shared 这组约束下，`num_stages=2` 数学上不可达。**

---

## 12. 手写 kernel 的可行性（已验证工具链）

### 12.1 工具链可用

`torch.utils.cpp_extension.load_inline` 在 MACA 环境下可以直接编译并运行手写 CUDA-like kernel
（走 `/opt/maca/tools/cu-bridge/cucc`）：

```python
cpp = r'''
#include <torch/extension.h>
torch::Tensor vadd(torch::Tensor a, torch::Tensor b);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("vadd", &vadd, "vector add"); }
'''
cu = r'''
#include <torch/extension.h>
__global__ void add_kernel(const float* a, const float* b, float* c, int n) { ... }
torch::Tensor vadd(torch::Tensor a, torch::Tensor b) { ... }
'''
m = load_inline(name="vadd_probe", cpp_sources=cpp, cuda_sources=cu, functions=None)
```

实测：`RESULT [1.0, 2.0, ...]`、`MATCH True` —— 编译、加载、执行全部正常。
注意：`functions=[...]` 的自动 binding 在 torch 2.8 下会报错，需自己写 `PYBIND11_MODULE`。

### 12.2 关键发现：硬件支持 `m16n8k16`，n16 是 TileLang 的实现选择

`/opt/maca/include/cute/arch/mma_sm80.hpp:231` 里有标准 PTX 封装：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

**即硬件 MMA 的 N 粒度是 8**，而 TileLang 的 MACA macro 用了 `_initialize_micro_size(n_dim=16)`
（§11.13），才导致 `block_N=16` 只能 1 warp。

⇒ **手写 kernel 可以用 n8 粒度，让 `block_N=16` 配 2 warp**（每 warp 8 列），
再配合 cp.async 双缓冲（shared 59 KB），正好绕开 §11.13 的死锁：

```text
block_N=16 + 2 warp + 双缓冲：2×18.4 + Q 18 + K_pe 4 + S 0.5 ≈ 59.3 KB ✓
```

### 12.3 MACA 的 MMA 硬件契约

这是手写改造的地基，全部有源码/生成代码依据。

**执行单元**：一个 wavefront = **64 lane**（NVIDIA 是 32），AP 内 4 个 PEU × 64 线程 = 256 线程。

**指令**：`__builtin_mxc_mma_16x16x16f16`，一次算完 16×16×16 的 tile，由 1 个 64 线程 warp 协同执行。
寄存器按 64 均分（`mma_macro_generator.py:135`）：

```python
local_size_a = m*k // warp_size   # 16*16/64 = 4 个 fp16
local_size_b = n*k // warp_size   # 16*16/64 = 4 个 fp16
local_size_out = m*n // warp_size # 16*16/64 = 4 个 fp32
```

生成代码与之完全一致（`mla_bn16_base.cu:69`）：

```c
half_t A_local[4]; half_t B_local[4]; float acc_s[4];
acc_s = __builtin_mxc_mma_16x16x16f16(B_local, A_local, acc_s);
```

**warp 切分公式**（`micro_size` 硬编码 16/16/16）：

```python
warp_rows = warp_row_tiles // 16      # M 方向每 warp ≥16 行
warp_cols = warp_col_tiles // 16      # N 方向每 warp ≥16 列
warp_col_tiles = block_N // n_warp
```

⇒ `block_N=16, threads=128` 时 `warp_col_tiles=8`，`8//16=0` → `Divide by zero`。
这是 §11.13 的根因，也是必须换 n8 指令的原因。

**n8 指令存在，但语义未定**：`/opt/maca/include/cute/arch/mma_sm80.hpp:231` 有

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
{%0..%3}(C/D), {%4..%7}(A), {%8,%9}(B), {%10..%13}(C)
```

它的寄存器契约是 **NVIDIA 的 32 线程语义**（A 4×u32、B 2×u32、C 4×f32）。而 MACA 的 warp 是 64 lane，
所以有两种可能，**必须实测判定**：

| 可能 | 语义 | 对 `block_N=16 + 2 warp` 的意义 |
| --- | --- | --- |
| ① | 两个 32-lane 半 warp 各执行一次，合起来覆盖 16 列 | 与 `m16n16k16` 等价，**换指令无收益** |
| ② | 编译器按 64 lane 重映射（每线程 2 个 C） | 1 warp 干 8 列，2 warp 覆盖 16 列 ⇒ 目标达成 |

**替换指令的代价**（若判定为 ②）：

| 项 | 现在（n16） | 换成 n8 |
| --- | --- | --- |
| `B_local` | 4 个 fp16 | **2 个 fp16** |
| `acc_s` | 4 个 fp32 | **2 个 fp32** |
| `A_local` | 4 个 fp16 | 4 个（K 维仍 16） |
| shared 索引 | 按 16 列一块 | 全部按 8 列重算 |
| softmax reduce | 每线程 4 元素 | 每线程 2 元素，跨 lane 归约要改 |
| `acc_o` | 每线程 128 个 fp32 | 不变（输出仍 16×512） |

**n8 的实测结论：不可用。** 证据链：

1. MACA 的内建函数只有 16×16×16 一族（`grep __builtin_mxc_mma /opt/maca/include`）：
   `__builtin_mxc_mma_16x16x16f16` / `16x16x16i8` / `16x16x16bf16` / `16x16x32f8`，**没有 16x8x16**；
2. `cute/arch/mma_sm80.hpp:231` 里的 `mma.sync.aligned.m16n8k16` 包在
   `#if defined(CUTE_ARCH_MMA_SM80_ENABLED)` 里，是给 NVCC 的分支，MACA 不启用；
3. 直接写 inline asm 也不行：`mxcc` 对 `"f"` / `"=f"` / `"+f"` 约束一律报
   `invalid output constraint`（不支持 PTX 的浮点寄存器约束语法）。

⇒ **MACA 的 MMA 只有 `m16n16k16` 一种粒度**，`block_N=16 + 2 warp` 在指令层面不存在。
§12.2 里“硬件支持 n8、可以 8 列/warp”的推断需要修正 —— 那条路关闭。

### 12.3.1 分步计划

1. **最小 GEMM**：用 `mma.sync.m16n8k16` + `ldmatrix` 写一个 16×16×576 的 QK，验证数值；
2. **加双缓冲**：`cp.async` 两级 buffer + `mma` 交错，测 8192 的 main-only 时间
   （目标：接近或超过现状 35.8 µs）；
3. **补全**：softmax（warp reduce）、PV、combine，逐档对齐。

工作量估计：1 约半天，2 约 1–2 天，3 约 2–3 天。

**补测：`block_N=16 + threads=64 + num_stages=2`（64 KB 下唯一能开流水的组合）**：

| kv_ctx | 现状 `bn=32,th=128,ns=1` | `bn=16,th=64,ns=2` | `bn=16,th=64,ns=2,Q→寄存器` |
| ---: | ---: | ---: | ---: |
| 2048 | **22.22** | 32.53（+46%） | 39.21（+76%） |
| 8192 | **44.60** | 107.76（+142%） | 114.67（+157%） |
| 65536 | **226.49** | 755.25（+234%） | 755.10（+234%） |

`threads=64` 时 MMA 只剩 1 个 warp（N=16 只有 1 个 tile），吞吐直接崩掉，
流水的收益远补不回来 ⇒ **64 KB shared 下根本不存在可用的 `num_stages=2` 配置**。

#### 4.5.3 下界：`threads ≥ 128` 由寄存器/占用率定

`acc_o` 是 `[block_H, dim] = [16, 512]` 的 fp32 累加器，共 8192 个 float，按线程数平摊：

```text
acc_o/线程 = 16 × 512 / threads = 8192 / threads
```

| `threads` | `acc_o/线程` | 生成代码中的数组 | 每线程总寄存器 | 结果 |
| ---: | ---: | --- | ---: | --- |
| 64 | 128 | `acc_o[128]` | ≈ 306 | 288.76 µs（**慢 3.92×**） |
| **128** | **64** | `acc_o[64]` | ≈ 166 | **73.63 µs** |
| 256 | 32 | — | — | 被 §4.5.1 排除 |

`threads=64` 时单单 `acc_o` 就吃掉 128 个寄存器（硬上限 512/线程），加上 MMA B 操作数
`B_local_2/B_local_5`（各 64）等共约 306。AP 寄存器堆是
`4 PEU × 512 reg × 64 thr = 131072 reg = 512 KB`，按 166 reg 可驻留约 12 wave/AP，
按 306 reg 只剩约 6 wave/AP —— **占用率腰斩**，访存延迟藏不住，实测慢 3.92×（§5.4 有寄存器明细）。

#### 4.5.4 为什么 `block_N` 也不能再小：每轮固定开销与 `block_N` 无关

`block_N=16 + threads=64` 是**合法**配置（§4.5.1 表中通过），但仍慢 6.55×（482.48 vs 73.63 µs）：

| | `block_N=32` | `block_N=16` |
| --- | ---: | ---: |
| 循环次数（`chunk=256`，三级切分见 §2.2） | 8 | **16** |
| 每轮 KV 数据量 | 36 KB | 18 KB |
| 每轮 online-softmax 固定开销 | 同左 | **同左（不缩小）** |

关键是**每轮开销不随 `block_N` 缩小而缩小**。每个 KV 块都要做一遍：

```text
T.reduce_max(acc_s)                     O(block_H × block_N) → 但归约后是 block_H 个标量
scores_scale[i] = exp2(...)             block_H = 16 次
acc_s[i,j]      = exp2(...)             block_H × block_N
acc_o[i,j]     *= scores_scale[i]       block_H × dim = 16 × 512 = 8192 次  ← 与 block_N 无关！
T.copy(acc_s, S_shared) + 第二次 GEMM 的 setup
```

其中 `acc_o *= scores_scale` 是 `O(block_H × dim)`，**完全与 `block_N` 无关**。
`block_N` 减半 ⇒ 轮数翻倍 ⇒ 这部分固定开销翻倍，而真正搬运的 KV 数据量不变。
所以 `block_N` 越大越好，只是被 §4.5.2 的 64 KB 卡在 32。

#### 4.5.5 结论：三条线交于唯一解

```text
block_N  ≤ 32              ← shared memory：64 KB/AP，dim=512 时最多装 32 个 token
block_N  ≥ 16 × threads/64 ← FullCol：每个 warp 至少一个 16 宽 MMA tile
threads  ≥ 128             ← 寄存器：acc_o/线程 = 8192/threads，64 线程时占用率腰斩
```

- `block_N` 被 smem 顶住**上界 32**、被 warp 映射托住**下界**；
- `threads` 被 warp 映射顶住**上界 128**（`threads ≤ 4 × block_N = 128`）、
  被寄存器托住**下界 128**（更多线程会让 `acc_o/线程` < 64，收益递减且 256 直接不可行）。

**上下界重合 ⇒ 解唯一：`block_N = 32, threads = 128`。** 这不是调出来的偏好，
而是硬件配额下唯一的可行点；`mla_autotune_configs()` 里 `threads` 的候选表
`(64, 128, 256)` 实际只有 128 能同时满足三条约束，`_threads_feasible()` 会把其余全部剪掉。

**`block_H` 同样没有自由度，唯一取值 16。** MMA tile 的各维必须是 16 的倍数，
`acc_s` 的 M 维就是 `block_H`（`FullCol` 不切 M）；取 8 时连一个 MMA tile 都凑不出，
编译期即失败。取 >16 则 Q/S 的 shared 与 `acc_o` 寄存器成倍增长，且 `_threads_feasible()`
对 `block_h > 16` 直接拒绝。

⇒ **「把 head 维拆成 2 个 CTA（`block_H=8`）换 2 倍并行度」这条路不存在。**
小 ctx 档的 CTA 数上界只由 `num_split ≤ kv_ctx / block_N` 决定，与 head 维无关：
2048 档 `2048/32 = 64` 已经是上限，64 CTA 铺在 104 个 AP 上**必然有 40 个 AP 空闲**，
这是结构性的、调参调不掉的（见 §8）。

> **combine kernel 例外**：它用 `T.Kernel(heads, batch, dim_chunk_num, threads=64)`（`:111`），
> `threads=64` 不受上面这条链约束——因为它没有 `acc_s` 这个 `[block_H, block_N]` 的 GEMM 输出，
> 不存在 warp 沿 N 切分的问题。它的设计是 `dim_chunk_num=8` 把 `dim=512` 切成 8 段、
> 每段 64 个 float，正好由一个 wavefront 的 64 个线程一一对应（§3.2），1 wave/CTA 足够了。

---

## 5. 存储层级与访问链路

### 5.1 shared memory 布局（`dim=512, pe_dim=64, block_N=32, block_H=16`）

动态 shared 从 `buf_dyn_shmem` 起分配，实测偏移：

| 缓冲区 | 偏移 | 大小 | 用途 | 布局 |
| --- | ---: | ---: | --- | --- |
| `Q_shared` | 0 | 16 KB | `[16, 512]` fp16 | 普通（copy 时用 swizzle 写） |
| `Q_pe_shared` | 16384 | 2 KB | `[16, 64]` | 普通 |
| `K_pe_shared` | 18432 | 4 KB | `[32, 64]` | **swizzled** |
| `KV_shared` | 22528 | 32 KB | `[32, 512]` | **swizzled** |
| `S_shared` | 55296 | 1 KB | `[16, 32]` fp16（PV 的 A 操作数） | **swizzled** |
| workspace | 56320 | — | TileLang 归约/中间量 | — |
| **合计** | | **≈ 56 KB** | 上限 64 KB/AP | |

- **swizzle 的作用**：`KV/K_pe/S` 都经 `make_swizzled_layout` 打散 bank 映射，
  把 shared 的 bank 冲突压到很低（实测 shared 访问效率 70.95%、冲突周期仅 1.30/指令）；
- **Q 为什么放 shared 而不是寄存器**：Q 是 `[16,512]` fp16 = 16 KB，
  直接 global→fragment 会按 MMA 的操作数布局散列落位，同 wavefront 内相邻线程地址相差 1 KB，
  合并访存被打散（sector 利用率掉到 ~25%）；而 global→shared 可以按连续地址拷贝、
  再用廉价的 shared→register 喂 MMA；
- **56 KB / 64 KB 意味着 `num_stages` 只能取 1**：两级 KV 需要 `2 × 32×576×2 = 73.7 KB`，
  已超上限，与 Q 放哪里无关。

### 5.2 三条访问链路（含向量宽度）

```text
① Global → Shared（主 kernel 的输入侧）
   Q     : *(uint4*)(Q_shared + swizzle) = *(uint4*)(Q + i·1024 + tid·8)   → 16 B/线程，连续合并
   KV    : T.copy(..., coalesced_width=8)                                   → 16 B/线程
   K_pe  : 同上

② Shared → Register（喂 MMA）
   A_local ← *(uint2*)(Q_shared + …)      （8 B/线程）
   B_local ← *(uint2*)(KV_shared + …)     （8 B/线程，每线程 64 个 fp16）
   S_shared ← acc_s（从 fragment 落回 shared，供 PV 的 A 操作数使用）

③ Register → Global（主 kernel 的输出侧）
   acc_o    → output_partial[b, h, split, :]（coalesced_width=8）
   logsum   → glse[b, h, split]

④ combine：Global → Shared → Register → Global
   glse           → glse_shared（T.copy）
   output_partial → po_local（T.copy，寄存器）
   o_accum_local  → Output（T.copy）
```

### 5.3 中间量的 padding

`output_partial` 末维 +64（`dim_pad=64`）、`glse` 末维 +64（`glse_pad=64`）：

> VL1 按地址 `bit[8:7]` 把访问路由到 4 个 partition，128 B 一个 partition、512 B 一轮。
> 若中间张量按 512 B 对齐，每轮访存都从同一个 partition 起步，最早发出未返回的请求会固定压在
> 同一个 partition。给末维加 128 B 的 padding 打破对齐，使 k 循环逐次轮换 partition、摊平 stall。

即 `output_partial` 的实际步长是 `dim+64 = 576` 个 fp16 = 1152 B，`output_partial` 总体
`[1,16,num_split,576]` fp16 = 1.18 MB @16384（落在 L2 里，L2C 命中 32%）。

### 5.4 寄存器占用（`threads=128`，主 kernel）

生成代码里每线程的静态数组（不含编译器临时量）：

| 项 | 大小 | 寄存器 |
| --- | ---: | ---: |
| `acc_o[64]`（PV 累加器 `[16,512]` fp32） | 64 × f32 | 64 |
| MMA B 操作数 `B_local_2[64]` / `B_local_5[64]` | 2 × 64 × fp16 | 64 |
| `acc_s[4]` | 4 × f32 | 4 |
| 其它（A 操作数、logsum/max/scale 等标量） | | ≈ 34 |
| **合计** | | **≈ 166** |

远低于 512/线程 的硬上限，**无 spill**（实测 `Private Read/Write Instructions = 0`）。
寄存器堆按 AP 共享（`4 PEU × 512 reg × 64 thr = 131072 reg = 512 KB`），
按 166 reg 估可驻留约 12 wave/AP（上限 32）。

---

## 6. Profiler 数据（`kv_ctx=16384`）

> **口径说明**：mcProfiler 会为每组 metric 重跑一遍 workload，报告里的**绝对计数器是各自
> pass 内的累加值**（各 pass 的 replay 次数还不一样），所以**只看比率/占比/单位延迟**；
> 绝对延迟与带宽以 §7 的 CUDA event 计时为准。

### 6.1 汇总

| 类别 | 指标 | 值 |
| --- | --- | --- |
| 指令 | Total Instructions | 41,204,475 |
| | Compute / Memory Instructions | 34,304,427 / 6,900,048 |
| 发射效率 | instruction per cycle | **27.17** |
| | instruction throughput efficiency | **6.53%** |
| | average cycles per instruction | 4.47 |
| 占用 | WORKGROUPS / WAVES | 11,113 / 31,652 |
| | Average Wave life cycles | 18,017 |
| | AP MTE / MMA / STE Duty ratio | 19.05% / 4.60% / 1.54% |
| 访存 | Global Read / Write bytes | 481.06 MB / 40.92 MB |
| | Global（VL1）Read / Write Instructions | 734,512 / 129,920 |
| | Dnoc Read Average Latency | 330.12 cycles |
| 缓存 | VL1 Hit Rate | **89.66%** |
| | L2C Hit Rate | **32.05%** |
| | Constant SL1 Hit Rate | 97.78% |
| shared | shared memory access efficiency | 70.95% |
| | average conflict cycles per instruction | **1.30** |
| | average latency / cycles per load instruction | 47.49 / 4.23 |
| 私有内存 | Private Read / Write Instructions | **0 / 0**（无 spill） |

### 6.2 指令构成与 stall

| Instructions Comparison | 占比 | 解读 |
| --- | ---: | --- |
| MTE | **63.86%** | 向量访存/搬运，占主导（逐元素算子应有的形态） |
| BSM | 14.37% | shared 访存 |
| MISC | 9.49% | |
| STE | 5.67% | 标量寻址/控制，占比低 → 地址开销健康 |
| GVM | 2.10% | |
| MMA | 4.22% | 矩阵指令占比很小（受 shape 限制） |
| LDU | 0.28% | |

| ISU stall cycles layout | 占比 |
| --- | ---: |
| `vls_pipeline_stall` | **56.25%** |
| `wsm_stall` | **36.68%** |
| `vls_wdata_stall` | 7.07% |
| `valu_stall` | 0.00% |

| Compute Instructions Cycles Rate | 占比 |
| --- | ---: |
| `MTE_16CYCLES`（宽向量） | **50.26%** |
| `MMA_16CYCLES` | 25.98% |
| `MTE_4CYCLES` | 23.76% |

⇒ 宽向量指令占一半，说明访存已经向量化；同时 `vls_pipeline_stall` + `wsm_stall` 合计 93%
说明**瓶颈是访存流水线与 wave/共享调度**，而不是算术。

### 6.3 延迟分布与 RoofLine

`Dnoc Read Latency Histogram`：`0–255` 1.57 M、`256–511` **1.83 M**、`512–1k` 0.44 M、
`1k–2k` 9.3 k、`2k+` 110 —— 绝大多数落在 256–511 cycle，长尾很轻。

| RoofLine | 值 |
| --- | ---: |
| `MAX_I`（拐点计算强度） | 260.0 |
| **`case_I`（实测计算强度）** | **259.86** |
| `case_bandwith` | 387.19 GB/s |
| `case_flops` | 100.61 TFLOPS |
| `block_num` | 104 |
| `core_clk` / `mc_clk` | 1.125 / 0.9 GHz |

**`case_I` 几乎正好落在拐点 `MAX_I = 260`** —— 这个算子当前处在计算与访存的平衡点上，
意味着单向优化（只堆算力或只省带宽）都不会有线性收益。

---

## 7. 性能结果

### 7.1 端到端延迟与吞吐

| kv_ctx | 延迟 (µs) | FLOPs | 实测 TFLOPS | 逻辑 DRAM 流量 | **有效带宽** | 占 1843 GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 22.34 | 71.3 M | 3.19 | 3.56 MB | 159 GB/s | 8.6% |
| 4096 | 29.61 | 142.6 M | 4.82 | 5.92 MB | 200 GB/s | 10.8% |
| 8192 | 44.43 | 285.2 M | 6.42 | 10.63 MB | 239 GB/s | 13.0% |
| 16384 | 68.43 | 570.4 M | 8.34 | 20.81 MB | 304 GB/s | 16.5% |
| 32768 | 114.20 | 1.141 G | 9.99 | 39.68 MB | 348 GB/s | 18.9% |
| 65536 | 196.63 | 2.281 G | 11.60 | 80.23 MB | 408 GB/s | 22.1% |

> 流量 = `KV + K_pe` 必需读 + `num_split ×` 每份重读的 `Q/Q_pe` + `Output` 写。
> `output_partial` 的写+读有 L2 兜着（L2C 命中 32%），未计入 DRAM。
> Q 重读同样主要是 L2 命中：把它的 global 读换成 shared 填零，64K 只快 0.99 µs、16K 快 0.72 µs
> （§8 方向 2），所以这一列是「逻辑流量」而不是真实 DRAM 增量。
> 按 mhc_post 复盘实测的 C500 峰值 ≈1528 GB/s 计算，利用率约 10.7%–25.7%。

**结论：带宽利用率仅 8.6%–22.1%，离带宽墙很远；`AP MMA Duty` 也只有 4.60%。
性能由并行度/延迟隐藏决定，而不是被任何单一硬件单元卡住。**

### 7.2 正确性

与独立 PyTorch（fp32）参考对比，全档 `cos_sim = 1.000000`、`max_abs_err ≈ 1e-6 ~ 7e-6`。

---

## 8. 进一步可优化空间

按"与已测数据的关系"排序：

| # | 方向 | 依据 | 预期 |
| --- | --- | --- | --- |
| 1 | **✅ 已完成：combine 固定开销已压，`num_split` 已扫到边界** | 带宽 8.6%–22.1%；combine 用 k 维分组把串行轮数降到 `ns/G+G`（§3.2.4），64K −3.5% | 继续加大 `ns` 的瓶颈已是主 kernel 每 CTA 的 prologue（8K/16K 大 `ns` 仍慢 30–41%） |
| 2 | **✅ 已评估：Q 重读不值得优化** | 每份重读 `Q/Q_pe` 18 KB，`ns=256@64K` 累计 4.6 MB；但 18 KB 常驻 8 MB L2、同一波 CTA 同时读 ⇒ 实测把 Q 的 global 读整体换成 shared 填零，64K 只快 **0.99 µs（0.5%）**、16K 快 0.72 µs | 唯一根治是减 `ns`，但 64K 从 256 降到 104 会慢 12 µs（§2.2.6）⇒ 放弃 |
| 3 | **✅ 部分完成：combine 的 k 维并行** | 已实现 k 维分组（§3.2.4），64K −3.5%；但 combine 的 grid 仍是 `heads×batch×dim_chunk_num`，主 kernel 的 CTA 数与之不匹配 | 若继续优化，可按 `num_split` 动态选 `dim_chunk_num`，让 combine 也吃满 104 AP 的整波 |
| 4 | **✅ 已评估：占用率不是瓶颈** | 166 reg × 128 thr ≈ 21.2 K reg/CTA，AP 寄存器堆 131 K ⇒ 可驻留 6 CTA/AP；而实际只有 0.6 CTA/AP，寄存器用量 ~10%，**没有 CTA 在排队** | 降寄存器不会让任何 CTA 早上场；加大 CTA 已过拐点（§2.2.8 扫描：8K 最优 0.62 波、16K 最优 0.62–1 波） |
| 5 | **`wsm_stall` 在 16384/32768 达 53.8%/60.2%（wave/共享调度）** | 与 shared 往返次数相关（`shared_kernel_rd` 5.25 M 次）；1 波配置下每 CTA 4–10 块 | **✅ 已尝试并否掉**：fragment 直喂 PV 会触发 `Layout infer conflict between acc_s and acc_s_cast`，TileLang 没有寄存器级 layout shuffle，`S_shared` 是必经的转换点（§10.6） |
| 6 | **✅ 已完成：非整除 ctx 的尾部处理** | `_split_plan()` 统一算 `chunk = align_up(ceil(ctx/num_split), block_N)` 与 `num_split_eff = ceil(ctx/chunk)`；末块走 clamp 整块读 + 列掩码，`ctx < block_N` 走逐行填充兜底 | 实测 3000/3008 = 26.42/26.53 µs、4095/4096 = 28.93/29.01 µs ⇒ 尾部开销 < 0.5%，整除档零回归（见 §2.2.5） |
| 7 | **`num_split` 的解析式** | 目前靠 autotune 试 8–16 个候选；闭式拟合在留出 shape 上失败（见 `num_split_design.md`） | 若做 varlen/paged，需照搬"运行时逐序列算 `num_splits[]`"的架构 |
| 8 | **长 ctx 用 `ns≈104` 的 1 波配置** | §2.2.6：16384 用 `ns=104` 比 `ns=64` 快 3.7%，32768 用 `ns=104` 比 `ns=256` 快 10.4% —— Q 重读 4.6 MB→1.8 MB、combine 轮数 256→104，同时仍覆盖整卡 | 已加入 autotune 候选（整波族），等全流程复测后替换 §7.1 的 16384/32768 两档 |

**不建议再试的方向**（已实测为负收益）：

- `Q` 放寄存器（frag 路径）：Q 的 global→fragment 装载破坏合并访存（sector 利用率 ~25%），
  在 2048–65536 全档比放 shared 慢 1.4–1.7×；
- `num_stages ≥ 2`：shared 放不下（KV 两级就 73.7 KB），且即使放得下也是负收益
  （实测 `ns=2` 比 `ns=0` 慢 13%–18%）；
- `threads=64`：`acc_o/线程` 64 → 128，每线程寄存器 166 → 306，占用率腰斩，实测慢 3.92×
  （288.76 vs 73.63 µs，推导见 §4.5.3）；
- `threads=256`：需要 `block_N ≥ 64`（§4.5.1），而 `block_N=64` 的 shared 要 92 KB，
  两者**不可兼得**，实际在编译期就报 `Divide by zero`；
- `block_N=64`：shared `94720 B > 65536 B`，报 `Shared memory size error`（§4.5.2）；
- **轮转分块 / persistent 动态领块**：CTA bz 处理块 `bz, bz+104, …` 实测 16384 **+33%**、
  32768 **+44%**、8192 47.61→60.01、65536 216.35→311.04 —— 块步长 3328 token 破坏 L2 局部性，
  而它换来的「负载差 ≤1 块」在连续分段里本来就有（§2.2.7）。

---

## 9. 其它算子相关内容

### 9.1 硬约束（选型时必须满足）

1. `chunk` 必须是 `block_N` 的整数倍，`num_split_eff = ceil(kv_ctx/chunk)`（由 `_split_plan()`
   统一计算）—— **不再要求 `num_split | kv_ctx`**：尾部残缺块由 kernel 内的列掩码处理
   （§2.2.5）。传入的 `num_split` 只是目标值，会被收敛到 `num_split_eff`，不会产生空 CTA；
2. 主 kernel 动态 shared ≤ 64 KB/AP：`Q + num_stages×(KV+K_pe) + S`；
3. `threads` 满足 warp 映射 `block_N ≥ 16 × (threads/64)`（`acc_s[block_H, block_N]` 用
   `FullCol` 按 N 切给 `threads/64` 个 warp）——**完整推导、实测可行域与「解为什么唯一」
   见 §4.5**。

### 9.2 精度

- 计算全程 fp16 输入 / fp32 累加（`acc_s`、`acc_o`）；
- `glse`（LSE）以 **fp16** 存储，是精度上的一个折中 —— 当前实测 `cos_sim = 1.0` 没问题，
  但若 `kv_ctx` 继续增大（LSE 数值范围变大）值得复核；
- `TL_ENABLE_FAST_MATH` 已开启（`exp2/log2` 走近似指令）。

### 9.3 调优机制（如何得到上面的配置）

`_get_kernel()` 默认走 autotune：

```python
from mla_decode import _get_kernel
kernel = _get_kernel(1, 16, 1, 16384, 512, 64)   # 首次调优，之后命中缓存（~0.1 s）
```

- 候选空间：`num_split`（等分族 4 + 整波族 2）× `spread{0,1}` × `combine_groups{1,8,16}`
  × `block_N{32,64}` × `num_stages{1,2}` × `q_in_reg` × `threads`。
  其中 `block_N=64`（shared 94.7 KB）、`num_stages=2`（91 KB）、`threads=256`（warp 映射）
  在裁剪阶段即被剪掉；`q_in_reg=True`（慢 13%–64%）与 `threads=64`（慢 3.92×）已从候选移除。
  **裁剪后六档合计 66 个**：2048:4 / 4096:7 / 8192:12 / 16384:14 / 32768:16 / 65536:13
  （裁剪前为 264 个，维度是 `ns × 4`，那 4 份全是已被证伪的 `threads{64,128} × q_in_reg{F,T}`）；
- 冷编译约 **12.6 s/配置**（`tune_best.py` 串行实测：15 次冷编译均值 12.6 s、最大 14 s）：
  66 个约 14 分钟串行、2–3 分钟（8 路并行，官方 `@autotune` 实测 2.4 s/配置）；
  裁剪前 264 个对应 55 分钟 / 11 分钟；
- 结果按 `(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim)` 缓存（进程内 + TileLang 磁盘缓存）；
- 旁路：`MLA_NO_AUTOTUNE=1` 或显式传参；
- autotune 失败会自动回退到内置 `if/else` 配置。

#### 9.3.1 已知可调项：还能把 66 个继续压到约 37 个

已删掉的两个维度（`threads=64`、`q_in_reg=True`）是**实测证伪且从未被选中**的，属于零风险。
下面三条则是「用少量适应性换编译时间」，**当前没有启用**，留作手动开关；三条全开约 37 个配置
（串行 ≈8 分钟 / 8 路并行 ≈1.5 分钟）：

| # | 规则 | 实测依据 | 影响 | 风险 |
| --- | --- | --- | --- | --- |
| 1 | `nb // ns > 8` 的 `num_split` 不入候选（每 CTA >8 块） | 8192 `ns=32`（16 块/CTA）65.79 µs vs `ns=64`（4 块）44.36 µs，慢 48%；16384 `ns=32` 116.56 vs 71.68 | 66 → 约 25 | 某些中间长度可能真的适合 12 块/CTA，六档暂无反例 |
| 2 | `ns < 32` 不入候选（CTA 覆盖 <1/3 AP） | 2048 的 `ns=8/16` 与 4096 的 `ns=16` 都是极端少 CTA，实测均显著更慢 | 与 #1 同时生效 | ctx < 1K 时可能需要 `ns=8` 这类值，届时需放开 |
| 3 | `combine_groups` 只在 `ns ≥ 256` 时加 | 8192/16384 的 `G=8` 反而慢（58.01 vs 44.36）；只有 64K 的 `G=16` 有效（204.13 → 197.03） | 25 → 约 43（含 G 的 18 个） | 32768 的 `ns=512, G=16` 未单独测过 |

对应改动点（各一行）：

```python
# 1 + 2：在 mla_autotune_configs 的 ns 候选处加过滤
if nb // num_split > 8 or num_split < 32:
    continue
# 3：combine_groups 候选
groups = [1] + [g for g in _TUNE_COMBINE_GROUPS
                if num_split >= 256 and num_split % g == 0]
```

> 之所以默认不启用：这三条是**用六档 shape 的实测反推**的，对 OJ 上出现的新长度未必成立；
> 而 autotune 是一次性成本（结果进磁盘缓存），省下的 7 分钟换来的是对未知 shape 的选型保险。

### 9.4 复现

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/mla_decode:$PYTHONPATH
cd /data/mla_decode

# 正确性 + 性能（对比独立 PyTorch 参考）
python scripts/check_mla.py --kv-ctx 2048 4096 8192 16384 32768 65536

# mcProfiler（本档报告即 mcprofiler_output/mla_decode/mla_best_16384/）
bash scripts/run_mcprofiler.sh 16384 0 mla_best_16384

# 解读
python scripts/summarize_report.py mcprofiler_output/mla_decode/mla_best_16384/output20261016161756
```

相关文档：脚本用法见 [`mla_decode_scripts_guide.md`](mla_decode_scripts_guide.md)；
选型与归因过程见 [`mla_decode_optimization_analysis.md`](mla_decode_optimization_analysis.md)；
`num_split` 的解析设计见 [`num_split_design.md`](num_split_design.md)。

---

## 10. 六档 profiler 实测与瓶颈迁移（最终配置）

采集条件：由 `scripts/tune_best.py` 逐配置实测选出的六档最优（下表），
`mcProfiler perf_exec --counts 5`，报告归档在
`mcprofiler_output/mla_decode/mla_best_<ctx>/output20261017*/`，
汇总见 `mla_profiler_summary.md` / `.json`（`scripts/collect_profiler.py` 生成）。

### 10.1 选型与延迟

| kv_ctx | `ns` | `spread` | `G` | 延迟 (µs) |
| ---: | ---: | :---: | ---: | ---: |
| 2048 | 64 | – | 1 | 22.34 |
| 4096 | 64 | – | 1 | 29.61 |
| 8192 | 64 | – | 1 | 44.43 |
| 16384 | 104 | ✓ | 1 | 68.43 |
| 32768 | 104 | ✓ | 1 | 114.20 |
| 65536 | 256 | – | 16 | 196.63 |

### 10.2 关键指标

| kv_ctx | WORKGROUPS | WAVES | wave_life | VL1 hit | L2C hit | Dnoc lat | MTE duty | MMA duty | compute duty | top stall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2048 | 6,377 | 12,708 | 10,465 | 92.07% | **96.72%** | 315.6 | 17.71% | 1.89% | 26.62% | vls 48.2% / wsm 33.8% |
| 4096 | 7,081 | 15,524 | 11,730 | 91.39% | 92.51% | 260.5 | 17.78% | 2.90% | 27.14% | vls 48.0% / wsm 37.9% |
| 8192 | 8,489 | 21,156 | 14,758 | 90.53% | 55.33% | 304.3 | 17.80% | 3.82% | 27.26% | vls 51.5% / wsm 38.6% |
| 16384 | 12,113 | 33,652 | 19,662 | 90.22% | 38.62% | 293.6 | 15.07% | 3.49% | 23.21% | **wsm 53.8%** / vls 33.8% |
| 32768 | 16,721 | 52,084 | 22,598 | 89.40% | 24.05% | 278.8 | 15.00% | 4.07% | 23.26% | **wsm 60.2%** / vls 28.9% |
| 65536 | 80,937 | 147,748 | 15,348 | 89.89% | 20.17% | 332.3 | **21.20%** | 5.35% | 32.98% | vls 55.8% / wsm 37.1% |

> WORKGROUPS / WAVES / 字节数是各 metric pass 的累加值（每 pass 的 replay 次数不同），
> 只用于看趋势；比率类指标（hit rate、duty、stall）可直接横向比。

### 10.3 三个单调迁移

1. **L2C 命中率 96.7% → 20.2%**：KV 工作集从 2.4 MB（2048）涨到 75.5 MB（65536），
   而 L2 只有 8 MB。2048/4096 几乎是纯 L2 计算，8192 开始漏，16K 以上基本走 DRAM。
2. **stall 主导从 `vls_pipeline` 切到 `wsm_stall`，再切回 `vls_pipeline`**：
   2048–8192 是 vls 48–52%（访存流水等待）；**16384/32768 变成 wsm 53.8%/60.2%**
   （wave 与 shared 调度）；65536 又回到 vls 55.8%。
3. **MMA duty 1.9% → 5.35%，compute duty 23%–33%**：全程算力未饱和，
   即使 64K 也只有 1/3 的指令槽在跑。

### 10.4 分档瓶颈判定

| 档位 | 证据 | 瓶颈 |
| --- | --- | --- |
| 2048 / 4096 | L2C 92–97%，带宽 159/200 GB/s，vls 48%，每 CTA 1–2 块 | **固定开销 + 单 CTA 串行依赖**（§2.2.9 已证明可压空间 ≤1 µs） |
| 8192 | L2C 55%，vls 51.5%，带宽 239 GB/s | 过渡区：L2 半失效，DRAM 延迟开始暴露 |
| **16384 / 32768** | L2C 24–39%，**wsm 54–60%**，wave_life 最长（19.7K/22.6K） | **wave / shared 调度**：1 波配置下每 CTA 4–10 块，`S_shared` 的“落 shared → 读回做 PV”往返被放大 |
| 65536 | vls 55.8%，MTE 21.2%，带宽 408 GB/s（22.1%） | **访存流水**：3 波 × 8 块把带宽推到全档最高，但离 1528 GB/s 实测峰值仍远 |

### 10.5 下一步（按证据强度排序）

1. ~~去掉 `S_shared` 中转~~：**已实测不可行**（§10.6）。`acc_s` 是 GEMM 的 C 布局、
   PV 的 A 操作数需要 MMA A 布局，TileLang 只允许经 shared 完成这次布局转换；
   `T.copy(acc_s, acc_s_cast)` 与逐元素 `T.Parallel` 赋值都会报
   `Layout infer conflict between acc_s and acc_s_cast`。当前实现已经是这条路径的最小代价
   （写一次 `S_shared`，PV 直接从 shared 读 A；baseline 是写 shared 再读回 fragment，共两次）。
2. **小 ctx 已见底**：2K/4K 的 vls 48% 来自每 CTA 只有 1–2 块，除固定开销外无并行度可加
   （§2.2.8/§2.2.9 已量化上限）。
3. **64K 的 22.1% 带宽**是全档最高，继续提升需要更多在途请求，但 `ns` 已扫到 256 且
   512/1024 更慢（§2.2.6），只剩 combine 侧还能动（G 已从 1 调到 16）。

### 10.6 已尝试：fragment 直喂 PV（不可行）

§10.4 把 16K/32K 的 `wsm_stall`（53.8%/60.2%）归因到 shared 往返，最直接的想法是把

```text
acc_s (fragment) ──T.copy──> S_shared ──> gemm 的 A 操作数
```

换成 `acc_s → acc_s_cast (fragment) → gemm 的 A 操作数`，完全绕开 shared。两种写法都编译失败：

| 写法 | 结果 |
| --- | --- |
| `for i, j in T.Parallel(block_H, block_N): acc_s_cast[i, j] = acc_s[i, j]` | `Layout infer conflict between acc_s and acc_s_cast in T.Parallel loop` |
| `T.copy(acc_s, acc_s_cast)`（让编译器做重排） | 同样报错（copy 被降级为 Parallel 循环，冲突不变） |

原因是布局语义不同：`acc_s` 是 `T.gemm(Q, KV)` 的输出，带 **MMA C 布局**；`acc_s_cast`
要作为下一次 GEMM 的 A 操作数，必须满足 **MMA A 布局**。用最小用例（`threads=128`、`16×32`）
把编译器推断出的 C 布局直接从报错信息里读了出来：

```text
Fragment((16, 32) -> (8,), replicate: 2, thread: 128,
         forward_thread: _rep * 64 + (_j % 16) // 4 * 16 + _i,
         forward_index:  (_j // 16 * 4 + _j % 4,))
```

即：行 `i` 落在 lane 的低 4 位、列 `j` 的 `(j%16)//4` 落在 bit4–5、`j//16` 决定 replicate
（两个 warp 各持一半列）；每线程 8 个槽位里，`N` 维按「4 列一组」稀疏分布。而 A 操作数
（ldmatrix 的 16×16 片段）是每 lane 8 个 fp16、按 `lane/4 + 8·(k/2)` 的方式分行——两者
在位域上完全不重叠，转换必须做**跨 lane 交换**。

这正是关键：TileLang 有 `T.shfl_xor/down/up`（`__shfl_*_sync`），但它们是**标量级 lane 交换**，
不承担布局转换；而 `T.Parallel` / `T.copy` 在两侧布局不一致时直接报冲突，**不会自动插入
shuffle 序列**。想在寄存器里手写，需要把值写进一个「已知是 A 布局」的 fragment，
而这次写入本身又要过布局推断 —— 仍然冲突（`T.alloc_local` 不携带布局，`T.gemm` 会拒收）。
`make_gemm_fragment_8x8()` 也救不了：它只适用于 8×8，套到 `16×32` 上直接报
`InputShape() = (8, 8) shape = (16, 32)` 不匹配。

⇒ **`S_shared` 不是冗余中转，而是布局转换的必要落点。** 当前实现已经是这条路径的最小代价：
只写一次 `S_shared`、PV 直接从 shared 读 A；baseline 是 `acc_s → S_shared → acc_s_cast`
（写 + 读共两次）。这也意味着 `wsm_stall` 不能靠“删掉 S_shared”消除，
只能靠减少块数（`ns` 选型）或提高每块的有效计算量。

> 注意：§6 的 profiler 表采集于 `ns=64` 的旧配置（16384 档），当前该档已改为整波
> `ns=104`，以本节数据为准。

---

## 11. 优化回顾与分歧点

### 11.1 上限账：98.3 分需要什么

按 `S = 100(T_b-T_h)/(T_b+T_k-2T_h)` 反解，每档 +1 分需要的降幅是**非线性**的：

| kv_ctx | 当前 \(T_k\) | 当前分 | +1 分需要 | 降幅 |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 22.34 | 94.49 | 18.3 µs | −18% |
| 4096 | 29.61 | 96.39 | 21.8 µs | −26% |
| 8192 | 44.43 | 97.41 | 29.0 µs | −35% |
| 16384 | 68.43 | 98.06 | 40.5 µs | −41% |
| 32768 | 114.20 | 98.43 | 54.2 µs | −53% |
| 65536 | 196.63 | 98.69 | 77.5 µs | −61% |

⇒ 98.3 大致等价于「六档全部提速 40%」，**不是靠某一档能补回来的**。

### 11.2 combine 的真实耗时（推翻了此前估算）

用编译期开关跳过第二个 `T.Kernel`，只跑主 kernel：

| kv_ctx | 端到端 | 仅主 kernel | combine | 占比 |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 21.62 | 13.68 | **7.94** | **36.7%** |
| 4096 | 29.09 | 21.10 | **7.99** | 27.5% |
| 8192 | 44.16 | 35.80 | **8.36** | 18.9% |
| 16384 | 68.24 | 57.55 | **10.69** | 15.7% |

此前 §2.2.4 把「固定开销 14–15 µs」估成 Q 装载 + prologue + combine + launch 的混合，
实测拆开是：**combine 8–10.7 µs，其余（Q 装载 0.7 + prologue + 收尾 + launch）约 6 µs**。
2048 档有 37% 的时间花在 combine 上，这是当前最大的单项可优化点。

### 11.3 三处「局部最优但上限不高」

| # | 当时的决策 | 局限 | 上限 |
| --- | --- | --- | --- |
| 1 | **combine 的 k 维分组（G）** | 为压 `ns` 轮串行，却把 **LSE 计算复制 G 倍**：`ns=64, G=8` 时 1024 个 CTA 各读 64 份 `glse` 做树形归约，LSE 成本反超串行节省 ⇒ 2048 的 G=8 实测无收益（21.92 vs 21.79） | 若按「组内 online 合并」实现（组内只算自己的 LSE），8 µs 可降到 3–4 µs |
| 2 | **`num_split` 的离散台阶**（64 / 104 / 256） | 步长 8/16 已扫过没有更优点（§2.2.6.1），但台阶本身由 64 KB shared 决定 | 再上一档需要 `num_stages=2` |
| 3 | **单 CTA 一个 tile = 16 heads × 32 tokens** | tile 形状被 shared 定死，`block_N` 上不去 | 同上 |

第 1 条是典型的「优化把瓶颈换了位置」：以为是串行轮数，实际加 G 后 LSE 变成了新瓶颈。

### 11.4 上限更高的分歧点

| 分歧点 | 现状 | 上限 | 状态 |
| --- | --- | --- | --- |
| **combine 两阶段 online 合并**（组内各算 LSE，组间再合并） | combine 8–10.7 µs | 3–4 µs（理论） | **❌ 已验证不可行**：总 load 次数 = `128G × 2ns/G = 256ns`，与 G 无关；G 越大波数越多，只增不减（详见 §11.6） |
| **combine 的 `dim_chunk_num` / `threads` 扫描** | 8 / 64 | 更大列宽、更长向量 | **❌ 已验证无收益**：dc=8 最优，dc=4 慢 2.5%、dc=2 慢 7%；ct=128/256 全部更慢（§11.6） |
| **combine 的 ⑤ 循环 4 路展开** | 64 轮串行 | 16 轮 × 4 路并行 | **❌ 已验证无收益**：21.86 vs 21.84，`T.copy` 之间没有形成在途重叠 |
| **单 kernel 融合 combine** | 两次 launch + partial 往返 | 再省 1–2 µs（2048 约 8%） | **❌ 已验证不可行**：小 `ns` 档慢 12%–32%（combine 并行度从 128 CTA 掉到 `ns` 个），`ns=256` 档直接 `mcErrorCooperativeLaunchTooLarge`（§11.7） |
| **`num_stages=2`** | 被 64 KB shared 锁死（§4.5.2） | 带宽 172–328 → 400–600 GB/s，全档 −30% | 低：dim 分块实验证明与 `Pipelined` 冲突且流量 ×2 |
| **fp8 KV** | 未做 | 流量减半 | 低：精度风险 + 题目精度约束 |

### 11.5 VL1 partition stall 为什么是空的

2026-10-17 六档报告的 `VL1 partition stall cycles layout` 全是 `pt0..pt3 = 0.0`，所以图是空白的
（**不是没测**：csv 里有该项，PNG 也生成了）。对照旧实现 `mla_ours_16384`（Q 放寄存器的 `frag_q`）：

| 版本 | pt0 | pt1 | pt2 | pt3 |
| --- | ---: | ---: | ---: | ---: |
| 旧 `frag_q`（Q→寄存器） | 1,664,788 | 4,672,848 | 2,064,956 | 5,742,507 |
| 现在（Q→shared） | 0 | 0 | 0 | 0 |

Q 走 `global→fragment` 时按 MMA 操作数布局散列落位（sector 利用率 ~25%），4 个 VL1 partition 被压偏；
改成 `global→shared` 的合并拷贝后负载均匀，stall 归零 —— 0 是「已经优化掉」的标志。

### 11.6 combine 的 8 µs 是两段式结构的下限

针对 §11.4 前三条做了实测，全部没有收益：

**① `dim_chunk_num` × `threads` 扫描**（2048 / 8192，单位 µs）：

| 配置 | dc=8,ct=64 | dc=4,ct=64 | dc=2,ct=64 | dc=4,ct=128 | dc=2,ct=128 | dc=2,ct=256 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | **21.84** | 22.38 | 23.37 | 22.22 | 22.46 | 22.42 |
| 8192 | **44.35** | 44.38 | 45.84 | 44.71 | 44.60 | 44.90 |

dc 变小 → CTA 数从 128 掉到 64/32，并行度下降抵消了「每次读更多字节」的收益；ct 变大 → 列数不够分，
线程空转。**默认的 8 / 64 已是最优。**

**② 4 路手动展开**（把 64 轮串行改成 16 轮 × 4 个 independent `T.copy`）：
2048 = 21.86（vs 21.84）、4096 = 29.48（vs 29.09）、8192 = 44.60（vs 44.35）、
16384 = 68.61（vs 68.24）、65536 = 196.28（vs 196.63）—— 全部在噪声内或略慢。
说明 `T.copy` 之间没有形成真正的在途重叠（编译器在每次 copy 后插入了同步）。

**③ 两阶段（组内各自算 LSE）为什么也不省**：combine 的总 load 次数是
`128 个 CTA × num_split 轮`。分成 G 组后变成 `128G 个 CTA × 2·num_split/G 轮`（LSE 也要读一遍），
**总次数从 `128·ns` 变成 `256·ns`，反而翻倍**；G 越大只是把同样的工作量摊到更多波次上，
关键路径缩短、总波数变长，净收益为 0（这就是 §11.3 第 1 条「把瓶颈换了位置」的完整解释）。

⇒ **在两段式结构下，combine 的 8 µs 已经是延迟下限**（128 CTA × 64 次 100 ns 级串行访问）。
要再降只有一个方向：**把 combine 融进主 kernel**，用 grid 级同步消除第二次 launch 与 partial 往返。
这条路的唯一前置风险（`T.sync_grid()` 会不会 hang）已在 §11.7 验证通过。

### 11.7 `T.sync_grid()` 可行性验证（通过）

最小用例（104 CTA × 128 线程，两个 kernel）：

```python
with T.Kernel(n, threads=128) as bx:
    O[bx] = A[bx]                 # 每个 CTA 先写自己的槽位
    T.sync_grid()                 # 网格级同步
    O[bx] = O[(bx + 1) % n] + 1000.0   # sync 之后读邻居的写
```

结果：

| 用例 | 编译 | 运行 | 语义 |
| --- | --- | --- | --- |
| 只调用 `sync_grid`（写→sync→再写） | OK | OK（`2i+1` 正确） | 不 hang |
| 跨 CTA 可见性（写→sync→读邻居） | OK | OK（`1000 + (i+1)%n` 全部正确） | **同步真实生效** |

说明 MACA 后端的 `this_grid().sync()` 是可用的。**但它是真 cooperative launch**（不是软件 barrier），
于是「全部 CTA 必须同时驻留」成为硬约束，而主 kernel 的 `dyn_smem = 57344 B`（56 KB）意味着
**每 AP 只能驻留 1 个 CTA** ⇒ cooperative grid 上限就是 104：

```text
mcErrorCooperativeLaunchTooLarge
grid=(1,1,256), block=(128,1,1) dyn_smem_bytes=57344
```

把 combine 融进主 kernel 后实测：

| kv_ctx | 两段式 | 融合 | 原因 |
| ---: | ---: | ---: | --- |
| 2048 | **21.63** | 28.52（+32%） | combine 并行度 128 CTA → 64 CTA，每个 CTA 还要多读一倍列 |
| 8192 | **44.15** | 49.38（+12%） | 同上 |
| 65536（ns=256） | 196.75 | **launch 失败** | 256 > 104，被 cooperative launch 拒绝 |

⇒ **融合的收益（省 1.5 µs launch）远小于并行度损失（combine 从 128 CTA 掉到 `ns`）。**
两段式里 combine 用独立 grid（128 CTA、1 wave），反而比融进 `ns` 个 CTA 更划算。
这条路关闭，§11.4 的结论回到「combine 的 8 µs 是当前结构的下限」。

### 11.8 内嵌 C 代码：可行，但绕不过 64 KB shared

两个最小用例已验证（`T.import_source(src)` + `T.call_extern`）：

| 用例 | 结果 |
| --- | --- |
| 注入 `__device__ float my_scale(float, float)`，在 kernel 里调用 | 编译运行正确 |
| 传 `T.address_of(shared[0])` 给 `__device__ void scale_buf(float*, int, float)`，原地改 shared | 结果正确（`p[i]*s+1`） |

⇒ 可以内嵌任意 `__device__` 函数，包括 `__shfl_*_sync`、inline asm、以及 MACA 头文件里的
`cp.async` 封装（`/opt/maca/include/cute/arch/copy_sm80.hpp` 存在）。

三个应用场景与它们各自的上限：

| 场景 | 想解决 | 上限 | 是否被别的限制挡住 |
| --- | --- | --- | --- |
| **手写 cp.async 双缓冲** | `num_stages=2`（带宽 22% → 40%） | 全档 −30%，正是 98.3 需要的量 | **是**：`2 × 36 KB + Q 18 KB = 90 KB > 64 KB`，shared 放不下 |
| **手写 fragment C→A 布局转换**（`__shfl_xor` 序列） | §10.6 的 `S_shared` 中转 | 2%–5% | 否，但需要精确的 A 布局公式 |
| **手写 combine 归约** | 64 轮串行的求和部分 | <1% | 瓶颈是 global 读延迟，不是求和 |

⇒ **内嵌 C 绕过了「抽象层」，但绕不过「64 KB shared」这个硬件配额。**
最大的一块收益（`num_stages=2`）依然不可达；能碰的只有布局转换的 2%–5%，
而它需要手写 MMA 的 A 操作数布局并与 `T.gemm` 协作，风险与收益不成比例。

---

## 附录 A：已实施优化手段总表

### A.1 算法结构

| # | 手段 | 代码位置 | 实测效果 |
| --- | --- | --- | --- |
| A1 | **split-KV 两段式**：主 kernel 算 partial + LSE，combine 归约 | `mla_decode.py` 两个 `T.Kernel` | 并行度从 1 个 CTA 提到 `ns` 个；h128 加速 5.4×–9.2× |
| A2 | online softmax 单遍，S 只在 shared 内流转 | 主循环 `acc_s → S_shared` | 不落 global，流量减半 |
| A3 | 全程 `log2/exp2`（scale 预乘 `log2(e)`） | `scale = softmax_scale * 1.44269504` | 省每次 `exp` 的乘法 |
| A4 | 倒数替代逐元素除法 | `inv_logsum` | 8192 次除法 → 16 次除法 + 8192 次乘 |
| A5 | `reduce_max(clear=False)` 复用运行最大值 | 主循环 | 每块省一次 `fill(-inf)` + 一次并行 max |

### A.2 分片与调度

| # | 手段 | 代码位置 | 实测效果 |
| --- | --- | --- | --- |
| A6 | `_split_plan`：`chunk` 对齐 `block_N`、`ns = ceil(ctx/chunk)`、无空 CTA | `_split_plan()` | 支持任意 ctx；旧 `ns=312` 的 56 个空 CTA 与漏算尾部消失 |
| A7 | 块级均分（`spread`）：前 `extra` 个 CTA 各多领一块 | `_split_plan(..., spread=True)` + 主循环 | 让 `ns` 可取 104 的倍数而不被「等长」绑住 |
| A8 | 整波族 `ns = 104 × w` | autotune 候选 | 16384 68.17、32768 113.97（+3.7% / +10.4%） |
| A9 | 尾部三态（整除 / clamp+列掩码 / 逐行兜底） | 主循环三个分支 | 非整除 ctx 的尾部开销 <0.5%（3000 vs 3008、4095 vs 4096） |

### A.3 访存与流水

| # | 手段 | 代码位置 | 实测效果 |
| --- | --- | --- | --- |
| A10 | `make_swizzled_layout`（KV / K_pe / S） | `T.annotate_layout` | shared 冲突 1.32 → 1.30 周期/指令，效率 70.95% |
| A11 | `coalesced_width=8`（16 B/线程） | 所有 global copy | `MTE_16CYCLES` 占比 36% → 66%，IPC 9.68 → 16.05 |
| A12 | L2 rasterization（`use_swizzle(panel=10)`） | 两个 kernel 开头 | L2C 命中 4.68% → 70.24%（16384 档） |
| A13 | 中间张量 padding（`dim_pad = glse_pad = 64`） | `output_partial` / `glse` | 打破 512 B 对齐，VL1 四个 partition 轮转 |
| A14 | Q 放 shared 而不是 fragment | `q_in_reg=False` | 全档比 frag 快 1.4–1.7×（sector 利用率 25% vs 合并访存） |
| A15 | prologue 重排：寄存器清零先于 Q 装载 | 主 kernel 开头 | 8192 44.36 → 43.99（−0.8%，接近噪声，§2.2.9） |

### A.4 phase2 归约

| # | 手段 | 代码位置 | 实测效果 |
| --- | --- | --- | --- |
| A16 | 串行标量链 → fragment + 两次树形归约 | combine 的 LSE 段 | `ns=64` 省 2.1–2.3 µs，`ns=256` 省 15.97 µs |
| A17 | ⑤ 循环 `T.copy` 到寄存器再 FMA | combine 累加段 | 避免逐元素 global 读的地址计算 |
| A18 | `glse` padding 区填 `-inf` | combine | 任意 `ns` 无需分支，`exp2(-inf-max)=0` |
| A19 | **k 维分组（两阶段归约）** | `partial2` + 第二个 `T.Kernel` | 串行轮 `ns → ns/G + G`；64K 204.13 → 197.03（−3.5%） |

### A.5 选型

| # | 手段 | 代码位置 | 说明 |
| --- | --- | --- | --- |
| A20 | autotune 维度：`num_split × spread × combine_groups`（`block_N/stages/q_in_reg/threads` 已被约束或证伪剪到唯一值） | `mla_autotune_configs()` | 逐 shape 实测，4–16 个候选（六档合计 66；可调项见 §9.3.1） |
| A21 | 候选裁剪：shared ≤ 64 KB、`block_N ≥ 16×threads/64`、`threads ≥ 128` | `_q_shared_bytes` / `_threads_feasible` | 先剪掉编译期必失败的组合 |

### A.6 已证伪（不要再试）

`block_H=8`（MMA 要求 16 的倍数，编译失败）、`block_N=16`（慢 6.55×）、`block_N=64`（shared 94.7 KB 超限）、
`threads=64`（慢 3.92×）、`threads=256`（编译期 `Divide by zero`）、`q_in_reg=True`（慢 13%–64%）、
`num_stages=2`（91 KB > 64 KB）、轮转分块（+26%–44%）、combine ⑤ 的 `T.Pipelined`（无收益）、
优化 Q 重读（整体 <1 µs）、降寄存器提占用率（没有 CTA 排队）。
