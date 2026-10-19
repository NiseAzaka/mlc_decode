# `num_split` 的设计：DeepSeek FlashMLA 的算法与我们的适配结论

> 本文单独讲一件事：**split-KV 的份数（`num_split` / `num_splits`）该怎么定**。
>
> - 上游参考：DeepSeek 官方 [FlashMLA](https://github.com/deepseek-ai/FlashMLA)（容器内装有其
>   MACA 版：`flash_mla 1.0.1+metax3.7.1.3torch2.8`，接口见
>   `/opt/conda/lib/python3.12/site-packages/flash_mla/flash_mla_interface.py`）
> - 本地实测数据：[`mla_decode_optimization_analysis.md`](mla_decode_optimization_analysis.md) §7.3、§8
> - 硬件口径：[`mxC500.md`](mxC500.md)（C500：104 AP、64 KB SMEM/AP、DRAM-L2 1843 GB/s）

---

## 1. 问题：`num_split` 为什么不是随便拍的

MLA Decode 里 `batch=1`、`heads=16`、`block_H=16` 时，**head 方向只能切出 1 个 CTA**，
整卡 104 个 AP 全靠 KV 方向切分（split-KV）来喂。切多少份是在两个相反的力之间取平衡：

| 力 | 增大 `num_split` 的效果 |
| --- | --- |
| ➕ **并行度** | main kernel 的 CTA 数 = `batch × (heads/block_H) × num_split` 增大 → 更多 AP 同时在跑、更多在途访存 → 更好隐藏延迟 |
| ➖ **每份固定开销** | 每份都要重读一遍 `Q/Q_pe`（`block_H×(dim+pe_dim)×2 = 18 KB`）；combine 阶段要串行归约 `num_split` 份 partial，耗时 ∝ `num_split`；`output_partial` 的写流量 ∝ `num_split` |

所以它存在最优值，且**最优值随 `kv_ctx` 变化**。问题是怎么找。

---

## 2. DeepSeek FlashMLA 的设计：运行时算，且逐序列算

### 2.1 产出物与调用方式

```python
tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens,
                                                       s_q * h_q // h_kv, h_kv)
out, lse = flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, dv,
                                  tile_scheduler_metadata, num_splits)
```

| 产出 | 形状 | 含义 |
| --- | --- | --- |
| `tile_scheduler_metadata` | `(num_sm_parts, 8)` int32 | 每个 part 负责的 `(seq, kv-block)` 区间；8 列里只用 5 列，凑 8 是为了 **int4(16 B) 对齐** |
| `num_splits` | `(batch_size + 1,)` int32 | **前缀和**形式；第 `b` 条序列被切成 `num_splits[b+1] - num_splits[b]` 份 |

5 个有效字段是：`begin_idx`（起始 seq）、`begin_seqlen`（起始 block×64）、`end_idx`、`end_seqlen`、
`n_split_idx`（这是该 seq 的第几份）。

计算它的 kernel 只有 **1 个 warp（32 线程）**：`get_mla_metadata_kernel<<<1, 32>>>` —— 说明这一步
本身不构成性能瓶颈。在 decode 循环里**每步算一次、各层复用**。

### 2.2 第一步：并行度上界由 SM 数决定（与 seqlen 无关）

`csrc/flash_api.cpp` 里是三个硬编码常量：

```cpp
static constexpr int block_size_m = 64;
static constexpr int block_size_n = 64;
static constexpr int fixed_overhead_num_blocks = 5;

num_sm_parts = sm_count / num_heads_k
             / cutlass::ceil_div(num_heads_per_head_k, block_size_m);
```

含义：M 方向按 64 切（`num_m_block`），乘以 `num_heads_k` 得到"每一条序列在 head/M 方向
能出多少个 CTA"，再用 `sm_count` 除它 → **一个 wave 能塞几个 KV 分片**。
即 `sm_count = num_sm_parts × num_heads_k × num_m_block`，正好铺满一波。

这个上限**与序列长度无关**：长度再长，也只是让每个 part 多吃几个 block，而不是继续加并行。

### 2.3 第二步：每个 part 的目标负载 = 均摊 + 固定开销

```cpp
int payload = cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks;
// total_num_blocks = Σ_b ceil(seqlens_k[b] / block_size_n)
// fixed_overhead_num_blocks = 5
```

- 第一项 `ceil(total / num_sm_parts)` 是**均摊**；
- 第二项 `+5` 是**每个 part 的冷启动开销**折算成的 KV block 数
  （上游注释与解读都指明它是"每个 SM 上至少要分到的算力块"，用来给 part 留余量、
  避免把负载压到刚好相等而在尾部互相等待）。

### 2.4 第三步：贪心打包，跨序列切分

从第 0 个 part 开始按 `payload` 向后消费 `(seq, block)`：

```text
now_idx, now_block = 0, 0
for each sm_part:
    remain_payload = payload
    while remain_payload >= now_remain_blocks + fixed_overhead_num_blocks:
        # 整个 sequence 的剩余 block 都装得下 → 吞掉，进入下一条序列
        remain_payload -= now_remain_blocks
        now_idx += 1
    # 装不下 → 在序列内部切一刀
    now_block = remain_payload - fixed_overhead_num_blocks
    n_split_idx += 1
    # 记下 (begin_idx, begin_seqlen, end_idx, end_seqlen, n_split_idx)
```

两条关键性质：

1. **切分点只落在 sequence 内部**（一个 part 可以吞掉若干完整序列 + 一条序列的一部分），
   于是某条序列跨了几个 part，就被切成几份 → 写进 `num_splits` 的前缀和；
2. **短序列不会被迫切分**：`ceil(total/num_sm_parts)` 小于一条序列的 block 数时就不会切，
   `actual_num_splits` 自然为 1。

### 2.5 计算侧：grid 与 combine 的配合

```cpp
// 主 kernel：z 维就是 num_sm_parts（=分片数）
flash_fwd_splitkv_mla_kernel<<<dim3(num_m_block, params.h, params.num_sm_parts),
                               kNThreads=256, smem_size=230400, stream>>>(params);
// combine：一维 grid = batch × heads × seqlen_q
MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, kMaxSplits, [&] {
    combine_kernel<<<grid_combine, 128, 0, stream>>>(params);
});
```

主 kernel 内部对分段做 for 循环（`batch_id` 从 `begin_idx` 到 `end_idx`），
用元数据里的 `n_block_min/n_block_max` 限定本次要扫的 KV 范围，
`NoSplit` 标记这条序列是否被切过。

combine kernel 里两个值得记的设计：

```cpp
const int split_offset      = num_splits_ptr[batch_idx];
const int actual_num_splits = num_splits_ptr[batch_idx + 1] - split_offset;
if (actual_num_splits == 1) return;          // ← 没切分就不付 combine 代价
```

用 warp shuffle（`__shfl_xor_sync`）在 32 个线程内做 LSE 的 max/sum 归约，
把每份的缩放系数 `sLseScale[split]` 写进 shared，再对 `output_partial` 做加权累加。

`MLA_NUM_SPLITS_SWITCH(num_sm_parts, kMaxSplits, ...)` 把 combine 的循环上界做成
**编译期模板特化**（例如 `num_sm_parts=78` → 取整到 `kMaxSplits=96`），
避免运行时的动态边界。

### 2.6 为什么非要"运行时逐序列算"

FlashMLA 面向的是 **varlen 服务**：batch 里几千条序列长度各异（举例：`b=128`，
`mean_sk` 在 4K–8K 之间浮动）。这时**不存在一个能同时适配的全局标量**：
短序列切多了纯亏固定开销，长序列切少了喂不饱 GPU。只能逐序列在运行时求解，
这也是 `num_splits` 为什么是 `batch+1` 的前缀和数组而不是一个整数。

---

## 3. 映射到我们的 kernel：结构对比

| 维度 | FlashMLA | 我们的 `mla_decode` |
| --- | --- | --- |
| batch / 序列 | 大 batch、varlen | `batch=1` 定长 |
| 主 kernel grid | `(num_m_block, h_kv, num_sm_parts)` | `(batch, heads/block_H, num_split)` |
| 分片数 | 运行时逐序列算（`num_splits[]`） | 候选集 + 实测选择（autotune） |
| combine grid | `batch × heads × seqlen_q`，**1 份不切就直接 return** | `heads × batch × dim_chunk_num(8)`，永远跑 |
| combine 循环上界 | 编译期模板 `kMaxSplits` | 运行时 `T.serial(num_split)` |
| SMEM / wave | 224 KB、双 warpgroup、compute-bound | **64 KB**、latency/并行度受限（带宽仅用 ~15%） |
| 硬约束 | block 粒度由 paged block 决定 | `num_split \| kv_ctx` 且 `chunk % block_N == 0`；`Q + num_stages×(KV+K_pe) + S ≤ 64 KB` |

**注意第 6 行**：FlashMLA 是 compute-bound 的大 kernel，一波就能喂饱；
我们**只用到 ~15% 带宽**（`mla_decode_optimization_analysis.md` §1.3），
长 ctx 下靠**多开 CTA 增加在途访存**更划算 —— 这一点后面会看到，正是它让最优分片数
**超过** AP 数（104），从根上偏离了 `num_sm_parts` 的假设。

---

## 4. 直接套用官方公式：准一半

把 §2.2 的公式套到我们的 shape（`batch=1, heads=16, kv_heads=1, dim=512, pe_dim=64, block_N=32`）：

```text
num_heads_per_head_k = s_q*h_q/h_kv = 16
n_m_block            = ceil(16/64)      = 1
num_sm_parts         = 104 / 1 / 1      = 104
```

| ctx | KV blocks（32） | `ceil(total/104)` | `payload`(+5) | FlashMLA 预测 | **我们实测最优** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 64 | 1 | 6 | 10.7 | **64** |
| 4096 | 128 | 2 | 7 | 18.3 | **64** |
| 8192 | 256 | 3 | 8 | 32.0 | **64** |
| 16384 | 512 | 5 | 10 | 51.2 | **64** |
| 32768 | 1024 | 10 | 15 | 68.3 | **128** |
| 65536 | 2048 | 20 | 25 | 81.9 | **256** |

16384 上预测 51 vs 实测 64，已经很接近；但短 ctx 明显偏小、长 ctx 明显偏小到 3 倍。
三个原因：

1. **`num_sm_parts` 钉死在"一个 wave"**。我们长 ctx 最优 256 > 104，说明多波更优；
2. **常数不可迁移**：`fixed_overhead_num_blocks=5`、`block_size=64` 是对
   "一份要重载 `Q(64×576)` 进 224 KB smem、还要 double buffer" 这种代价拟合的，
   我们在 64 KB smem 上、`Q` 只有 16×576 的代价结构完全不同；
3. 我们单序列定长，退化成标量后信息量本就比 varlen 少。

---

## 5. 试着拟合我们自己的闭式解 —— 留出验证失败

仿照"填满 + 每份固定开销"的思路，在 6 个训练 shape 上拟合：

```text
num_split* = max(NS_min, ceil(ctx / CHUNK*))          NS_min = 64, CHUNK* = 256
```

6 个点全部命中：

| ctx | `max(64, ctx/256)` | 实测最优 |
| ---: | ---: | ---: |
| 2048 | 64 | 64 ✅ |
| 4096 | 64 | 64 ✅ |
| 8192 | 64 | 64 ✅ |
| 16384 | 64 | 64 ✅ |
| 32768 | 128 | 128 ✅ |
| 65536 | 256 | 256 ✅ |

但用**未参与拟合**的 `ctx=12288` 验证时失败：

| `num_split` | chunk | 延迟 (µs) |
| ---: | ---: | ---: |
| 384 | 32 | 112.02 |
| 192 | 64 | 73.62 |
| 128 | 96 | 76.40 |
| **96** | **128** | **51.79 ← 实测最优** |
| 64 | 192 | 60.15 ← 模型预测（慢 16%） |
| 48 | 256 | 69.57 |
| 32 | 384 | 92.53 |

而且两个 shape 的偏好互相矛盾：
`ctx=12288` 偏好 `chunk=128`（51.79 < 192 的 60.15）；
`ctx=16384` 偏好 `chunk=256`（73.17 < 128 的 89.01）。

**根因**：最优分片数由两类离散效应对齐决定，两参数模型抓不到：

1. **wave 量化**：总 CTA 数 = 主 kernel `num_split` + combine `heads×batch×dim_chunk_num(=128)`，
   与 104 个 AP 的对齐关系随 `num_split` 跳变（如 192/224/256/384 个 CTA 分别是 1.85/2.15/2.46/3.7 波）；
2. **主/combine 此消彼长**：主 kernel 随 `num_split` 变快，combine 随 `num_split` 变慢（串行 `num_split` 次），
   两者的和构成一个带拐点的曲线。

（2 个自由参数拟合 6 个点再外推，本身就是过拟合；没有留出验证就会误以为模型成立。）

---

## 6. 结论

| 问题 | 结论 |
| --- | --- |
| DeepSeek 有没有"计算最优 `num_split`"的方法？ | **有**：`num_sm_parts`（SM 数定并行度上界）+ `payload = 均摊 + 固定开销` + 贪心打包 → 逐序列 `num_splits[]` |
| 能不能直接抄到我方？ | **不能**。bound 类型、smem 容量、序列形态（varlen vs 单定长）三处都不同 |
| 我方要不要改成闭式？ | **暂不改**。闭式在留出 shape 上失败（预测 64 vs 最优 96）；autotune 只有 ~8 个候选、有磁盘缓存，代价可接受 |
| 什么时候必须参考它？ | 一旦做 **varlen / paged**，逐序列 split 是刚需，应采用"运行时算 `num_splits[]`"的架构 |

---

## 7. 若将来要落地：给我方的设计草案

如果后续要支持 varlen / paged（那时 `num_split` 必须逐序列定），建议照搬架构：

```text
① 一个极小的 metadata kernel（1 warp），输入 cu_seqlens_k，输出：
     num_splits[batch+1]（int32 前缀和）        —— 逐序列分片数
     tile_scheduler_metadata[num_parts][N]      —— 每个 CTA 负责的 (seq, block 区间)
② 主 kernel grid 改为 (batch_or_m_block, heads/block_H, num_parts)，
   在 kernel 内用元数据决定 begin/end，而非用全局 num_split；
③ combine 改为逐 (batch, head) 一个 CTA（或按 dim 再切），
   读 num_splits[b..b+1] 得到实际份数，
   **actual == 1 时直接 return**（省掉 combine 的一次全局往返）；
④ 两个 kernel 都必须保留整除性校验：`chunk % block_N == 0`，
   否则尾部 block 会越界读 / 漏算（我们已踩过 `num_split=312` 的坑）。
```

`payload` 的两个常数需针对 C500 重新标定：

- `num_parts` 的上界：不要照抄 `sm_count`。我方长 ctx 最优分片数 > AP 数，
  说明上界应取 `sm_count × 2~3` 量级（或按 `sm_count / (heads/block_H)` 再乘一个波数因子）；
- `fixed_overhead`：按"重读 Q + 一份 partial 的写/读/combine"折算成 block 数，
  可用 §5 的实测矩阵反解标定。

---

## 8. 参考

| 来源 | 内容 |
| --- | --- |
| [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) | 官方仓库；`get_mla_metadata` / `flash_mla_with_kvcache` |
| 本地 `flash_mla/flash_mla_interface.py` | 接口签名与 `tile_scheduler_metadata` / `num_splits` 的形状说明 |
| `csrc/flash_api.cpp`（`block_size_m/n=64`、`fixed_overhead_num_blocks=5`、`num_sm_parts` 公式） | 常量与并行度上界 |
| [FlashMLA 源码分析（Zhao Dongyu）](https://zhao-dongyu.github.io/2025/02/24/117_FlashMLA/) | `payload` 公式、贪心打包装配过程、`tile_scheduler_metadata` 字段含义、combine kernel 的 LSE 归约 |
| [Flash MLA Kernel分析（悟剑阁）](https://sword865.github.io/posts/2025/2025-05-03-flashmla-kernel%E5%88%86%E6%9E%90/) | `num_sm_parts` 公式推导、grid 三维语义、block 到 SM 的分配流程 |
