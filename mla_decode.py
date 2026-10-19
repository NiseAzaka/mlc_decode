import os
import sys

import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune


def _split_plan(seqlen_kv, num_split, block_n, spread=False):
    """把 KV 的 `nb = ceil(kv_ctx/block_N)` 个块分配给 `ns` 个 CTA。

    返回 `(ns, blocks_per_cta, extra)`，`ns` 个 CTA 中前 `extra` 个各多领一块：

    - `spread=False`（等分族）：先把 `chunk` 向上对齐到 `block_N`，再令
      `ns = ceil(nb / blocks_per_cta)`。所有 CTA 块数相同，`ns` 会被**收敛**
      （传入 312 份 → 实际 47 份），不会产生 `bz * chunk ≥ kv_ctx` 的空 CTA；
    - `spread=True`（整波族）：`ns` 原样保留（可以是 104 的倍数），块级均分，
      前 `extra` 个 CTA 多领一块，负载差 ≤ 1 块，最后一个波次不会因余数空转。

    两种模式的所有块都是完整的 `block_N` 行，只有 `kv_ctx % block_N != 0` 时的全局
    最后一块偏短，由 kernel 内的列掩码处理。
    """
    nb = -(-seqlen_kv // block_n)
    if not spread:
        blocks_per_cta = max(1, -(-(-(-seqlen_kv // num_split)) // block_n))
        ns = max(1, -(-nb // blocks_per_cta))
        return ns, blocks_per_cta, 0
    ns = max(1, min(int(num_split), nb))
    blocks_per_cta, extra = divmod(nb, ns)
    return ns, blocks_per_cta, extra


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
)
def flashattn_shared_q(batch, heads, kv_head_num, seqlen_kv, dim, 
                        pe_dim, block_N, block_H, num_split, 
                        num_stages, dim_chunk_num, softmax_scale, threads=128,
                        spread=False, combine_groups=1, combine_threads=64):
    num_split, blocks_per_cta, extra = _split_plan(seqlen_kv, num_split, block_N, spread)
    # combine 的 k 维分组：串行轮数从 num_split 降到 num_split / combine_groups
    if combine_groups < 1 or num_split % combine_groups:
        combine_groups = 1
    # 尾部模式：0 = 每块都满；1 = 全局最后一块偏短（clamp + 列掩码）；2 = kv_ctx < block_N 兜底
    if seqlen_kv % block_N == 0:
        tail_mode = 0
    elif seqlen_kv >= block_N:
        tail_mode = 1
    else:
        tail_mode = 2
    scale = float(softmax_scale * 1.44269504)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    valid_block_h = min(block_H, kv_group_num)

    dim_per_chunk = dim // dim_chunk_num
    glse_size = (T.ceildiv(num_split, 64)) * 64
    # VL1(L1) 按地址 bit[8:7] 路由到 4 个 partition：128B 一个 partition，512B 一轮。
    # 中间张量若按 512B 对齐，每次访存都从同一个 partition 起步，oldest 请求会
    # 固定落在最后一个 partition 上，stall 全压在该 partition。给末维加 padding
    # 打破对齐（步长 mod 512 = 128B），使 k 循环逐次轮换 partition，摊平 stall。
    dim_pad = 64    # 64 * 2B = 128B = 恰好一个 partition
    glse_pad = 64
    @T.prim_func
    def main(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        glse = T.alloc_global([batch, heads, num_split + glse_pad], dtype)
        output_partial = T.alloc_global([batch, heads, num_split, dim + dim_pad], dtype)
        if combine_groups > 1:
            partial2 = T.alloc_global([batch, heads, combine_groups, dim + dim_pad], dtype)

        with T.Kernel(batch, heads // valid_block_h, num_split, threads=threads) as (bid, hid, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            # O_shared = T.alloc_shared([block_H, dim], dtype)

            # 将 T.alloc_shared 替换为 T.alloc_fragment
            # Q_frag = T.alloc_fragment([block_H, dim], dtype)
            # Q_pe_frag = T.alloc_fragment([block_H, pe_dim], dtype)
            
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)
            inv_logsum = T.alloc_fragment([block_H], accum_dtype)
            
            T.annotate_layout({
                KV_shared: tilelang.layout.make_swizzled_layout(KV_shared),
                K_pe_shared: tilelang.layout.make_swizzled_layout(K_pe_shared),
                S_shared:tilelang.layout.make_swizzled_layout(S_shared)
            })
            T.use_swizzle(panel_size=10, enable=True)
            
            # 先清寄存器：与 Q 的 global load 无依赖，可在 MTE 搬运期间并行执行
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.copy(Q[bid, hid * valid_block_h:(hid + 1) * valid_block_h, :], Q_shared, coalesced_width=8)
            T.copy(Q_pe[bid, hid * valid_block_h:(hid + 1) * valid_block_h, :], Q_pe_shared, coalesced_width=8)

            if extra == 0:
                # 整除：每个 CTA 恰好 blocks_per_cta 块，轮数是编译期常量
                kv_base = bz * (blocks_per_cta * block_N)
                loop_range = blocks_per_cta
            else:
                # 块级均分：前 extra 个 CTA 各多领一块，最后一波不会因余数空转
                kv_base = (bz * blocks_per_cta + T.min(bz, extra)) * block_N
                loop_range = T.if_then_else(bz < extra, blocks_per_cta + 1, blocks_per_cta)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                kv_start = kv_base + k * block_N
                if tail_mode == 1:
                    # 末块可能越过 kv_ctx：把整块起点 clamp 回界内（重复读少量已算过的
                    # token），再用列掩码只保留 [kv_start, kv_ctx) 对应的列。这样拷贝始终
                    # 是整块，能保留 coalesced_width=8，非末块与整档快路径完全一致。
                    src_start = T.min(kv_start, seqlen_kv - block_N)
                    T.copy(KV[bid, src_start:src_start + block_N, 0, :], KV_shared, coalesced_width=8)
                    T.copy(K_pe[bid, src_start:src_start + block_N, 0, :], K_pe_shared, coalesced_width=8)
                elif tail_mode == 2:
                    # kv_ctx 不足一个 block：逐行填，避免"短 region 拷贝"触发布局推断失败
                    kv_end = T.min(kv_start + block_N, seqlen_kv)
                    for r, c in T.Parallel(block_N, dim):
                        if kv_start + r < kv_end:
                            KV_shared[r, c] = KV[bid, kv_start + r, 0, c]
                        else:
                            KV_shared[r, c] = 0
                    for r, c in T.Parallel(block_N, pe_dim):
                        if kv_start + r < kv_end:
                            K_pe_shared[r, c] = K_pe[bid, kv_start + r, 0, c]
                        else:
                            K_pe_shared[r, c] = 0
                else:
                    kv_end = kv_start + block_N
                    T.copy(KV[bid, kv_start:kv_end, 0, :], KV_shared, coalesced_width=8)
                    T.copy(K_pe[bid, kv_start:kv_end, 0, :], K_pe_shared, coalesced_width=8)

                T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                if tail_mode == 1:
                    # 列掩码：只保留 [kv_start, kv_end) 对应的列（clamp 后可能从中间开始）。
                    # 无效列在 reduce_max 之前就被覆盖成 -inf，未初始化 shared 不影响有效列。
                    valid_start = kv_start - src_start
                    valid_end = T.min(kv_start + block_N, seqlen_kv) - src_start
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(
                            j >= valid_start,
                            T.if_then_else(j < valid_end, acc_s[i, j], -T.infinity(accum_dtype)),
                            -T.infinity(accum_dtype))
                elif tail_mode == 2:
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(
                            kv_start + j < kv_end, acc_s[i, j], -T.infinity(accum_dtype))

                T.copy(scores_max, scores_max_prev)
                # 删除了 T.fill(-inf)，直接利用 scores_max_prev 的遗留值进行无缝归约
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            # 先用 16 次除法求出倒数
            for i in T.Parallel(block_H):
                inv_logsum[i] = 1.0 / logsum[i]
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] *= inv_logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

            T.copy(logsum, glse[bid, hid * valid_block_h:(hid + 1) * valid_block_h, bz])
            T.copy(acc_o, output_partial[bid, hid * valid_block_h:(hid + 1) * valid_block_h, bz, 0:dim], coalesced_width=8)

        with T.Kernel(heads, batch, dim_chunk_num * combine_groups, threads=combine_threads) as (hid, bz, dim_grp):
            dim_start = (dim_grp % dim_chunk_num) * dim_per_chunk
            glse_shared = T.alloc_shared([glse_size], dtype)
            
            po_local = T.alloc_fragment([dim_per_chunk], dtype)
            o_accum_local = T.alloc_fragment([dim_per_chunk], accum_dtype)
            # 【phase2 优化】LSE 归约：原实现是 2×num_split 步的串行标量依赖链，
            # 期间 63/64 个线程全程空闲。改为「并行装载 + 并行 exp2 + 两次树形归约」。
            # 归约列数取 glse_size（已向上取整到 64 的倍数），padding 项填 -inf：
            # 对 max 无影响，且 exp2(-inf - max) = 0，对 sum 同样无影响。
            lse_frag = T.alloc_fragment([1, glse_size], accum_dtype)
            lse_max_local = T.alloc_fragment([1], accum_dtype)
            lse_sum_local = T.alloc_fragment([1], accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)

            T.clear(o_accum_local)

            T.copy(glse[bz, hid, 0:glse_size], glse_shared)
        
            for k in T.Parallel(glse_size):
                if k >= num_split:
                    glse_shared[k] = -T.infinity(dtype)
            for k in T.Parallel(glse_size):
                lse_frag[0, k] = glse_shared[k]
            T.reduce_max(lse_frag, lse_max_local, dim=1, clear=True)
            for k in T.Parallel(glse_size):
                lse_frag[0, k] = T.exp2(lse_frag[0, k] - lse_max_local[0])
            T.reduce_sum(lse_frag, lse_sum_local, dim=1)
            lse_logsum_local = T.log2(lse_sum_local[0]) + lse_max_local[0]
            # 融合循环：一次并行，0 寄存器碎片，直接 FMA
            group_chunk = num_split // combine_groups
            for k in T.serial(group_chunk):
                scale_local = T.exp2(
                    glse_shared[(dim_grp // dim_chunk_num) * group_chunk + k] - lse_logsum_local)

                # # 唯一的并行域：直接融合全局内存读取与缩放累加
                # for i in T.Parallel(dim):
                #     o_accum_local[i] += output_partial[bz, hid, k, i] * scale_local

                # 1. 直接用 T.copy 将 global 内存一口气倒进 po_local 寄存器
                T.copy(output_partial[bz, hid, (dim_grp // dim_chunk_num) * group_chunk + k,
                                     dim_start : dim_start + dim_per_chunk], po_local)
                
                # 2. 纯寄存器级别的 FMA 计算，0 延迟
                for i in T.Parallel(dim_per_chunk):
                    o_accum_local[i] += po_local[i] * scale_local
            if combine_groups > 1:
                # 第一阶段：各组只归约自己那 num_split/combine_groups 份
                T.copy(o_accum_local, partial2[bz, hid, dim_grp // dim_chunk_num,
                                              dim_start : dim_start + dim_per_chunk])
            else:
                T.copy(o_accum_local, Output[bz, hid, dim_start : dim_start + dim_per_chunk])

        if combine_groups > 1:
            # 第二阶段：把 combine_groups 份部分和加起来（LSE 权重已在第一阶段乘好）
            with T.Kernel(heads, batch, dim_chunk_num, threads=64) as (hid, bz, dim_idx):
                dim_start2 = dim_idx * dim_per_chunk
                po2 = T.alloc_fragment([dim_per_chunk], dtype)
                o_accum2 = T.alloc_fragment([dim_per_chunk], accum_dtype)
                T.clear(o_accum2)
                for cg2 in T.serial(combine_groups):
                    T.copy(partial2[bz, hid, cg2, dim_start2 : dim_start2 + dim_per_chunk], po2)
                    for i in T.Parallel(dim_per_chunk):
                        o_accum2[i] += po2[i]
                T.copy(o_accum2, Output[bz, hid, dim_start2 : dim_start2 + dim_per_chunk])

    return main

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn_frag_q(batch, heads, kv_head_num, seqlen_kv, dim, 
                        pe_dim, block_N, block_H, num_split, 
                        num_stages, dim_chunk_num, softmax_scale, threads=128,
                        spread=False, combine_groups=1, combine_threads=64):
    num_split, blocks_per_cta, extra = _split_plan(seqlen_kv, num_split, block_N, spread)
    # combine 的 k 维分组：串行轮数从 num_split 降到 num_split / combine_groups
    if combine_groups < 1 or num_split % combine_groups:
        combine_groups = 1
    # 尾部模式：0 = 每块都满；1 = 全局最后一块偏短（clamp + 列掩码）；2 = kv_ctx < block_N 兜底
    if seqlen_kv % block_N == 0:
        tail_mode = 0
    elif seqlen_kv >= block_N:
        tail_mode = 1
    else:
        tail_mode = 2
    scale = float(softmax_scale * 1.44269504)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    valid_block_h = min(block_H, kv_group_num)
    
    
    dim_per_chunk = dim // dim_chunk_num
    glse_size = (T.ceildiv(num_split, 64)) * 64
    # VL1(L1) 按地址 bit[8:7] 路由到 4 个 partition：128B 一个 partition，512B 一轮。
    # 中间张量若按 512B 对齐，每次访存都从同一个 partition 起步，oldest 请求会
    # 固定落在最后一个 partition 上，stall 全压在该 partition。给末维加 padding
    # 打破对齐（步长 mod 512 = 128B），使 k 循环逐次轮换 partition，摊平 stall。
    dim_pad = 64    # 64 * 2B = 128B = 恰好一个 partition
    glse_pad = 64
    @T.prim_func
    def main(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        glse = T.alloc_global([batch, heads, num_split + glse_pad], dtype)
        output_partial = T.alloc_global([batch, heads, num_split, dim + dim_pad], dtype)
        if combine_groups > 1:
            partial2 = T.alloc_global([batch, heads, combine_groups, dim + dim_pad], dtype)

        with T.Kernel(batch, heads // valid_block_h, num_split, threads=threads) as (bid, hid, bz):
            # Q_shared = T.alloc_shared([block_H, dim], dtype)
            # Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            # O_shared = T.alloc_shared([block_H, dim], dtype)

            # 将 T.alloc_shared 替换为 T.alloc_fragment
            Q_frag = T.alloc_fragment([block_H, dim], dtype)
            Q_pe_frag = T.alloc_fragment([block_H, pe_dim], dtype)
            
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)
            inv_logsum = T.alloc_fragment([block_H], accum_dtype)

            T.annotate_layout({
                KV_shared: tilelang.layout.make_swizzled_layout(KV_shared),
                K_pe_shared: tilelang.layout.make_swizzled_layout(K_pe_shared),
                S_shared:tilelang.layout.make_swizzled_layout(S_shared)
            })
            T.use_swizzle(panel_size=10, enable=True)
            
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.copy(Q[bid, hid * valid_block_h:(hid + 1) * valid_block_h, :], Q_frag, coalesced_width=8)
            T.copy(Q_pe[bid, hid * valid_block_h:(hid + 1) * valid_block_h, :], Q_pe_frag, coalesced_width=8)

            if extra == 0:
                # 整除：每个 CTA 恰好 blocks_per_cta 块，轮数是编译期常量
                kv_base = bz * (blocks_per_cta * block_N)
                loop_range = blocks_per_cta
            else:
                # 块级均分：前 extra 个 CTA 各多领一块，最后一波不会因余数空转
                kv_base = (bz * blocks_per_cta + T.min(bz, extra)) * block_N
                loop_range = T.if_then_else(bz < extra, blocks_per_cta + 1, blocks_per_cta)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                kv_start = kv_base + k * block_N
                if tail_mode == 1:
                    src_start = T.min(kv_start, seqlen_kv - block_N)
                    T.copy(KV[bid, src_start:src_start + block_N, 0, :], KV_shared, coalesced_width=8)
                    T.copy(K_pe[bid, src_start:src_start + block_N, 0, :], K_pe_shared, coalesced_width=8)
                elif tail_mode == 2:
                    # kv_ctx 不足一个 block：逐行填，避免"短 region 拷贝"触发布局推断失败
                    kv_end = T.min(kv_start + block_N, seqlen_kv)
                    for r, c in T.Parallel(block_N, dim):
                        if kv_start + r < kv_end:
                            KV_shared[r, c] = KV[bid, kv_start + r, 0, c]
                        else:
                            KV_shared[r, c] = 0
                    for r, c in T.Parallel(block_N, pe_dim):
                        if kv_start + r < kv_end:
                            K_pe_shared[r, c] = K_pe[bid, kv_start + r, 0, c]
                        else:
                            K_pe_shared[r, c] = 0
                else:
                    kv_end = kv_start + block_N
                    T.copy(KV[bid, kv_start:kv_end, 0, :], KV_shared, coalesced_width=8)
                    T.copy(K_pe[bid, kv_start:kv_end, 0, :], K_pe_shared, coalesced_width=8)

                T.clear(acc_s)
                T.gemm(Q_frag, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_frag, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                if tail_mode == 1:
                    valid_start = kv_start - src_start
                    valid_end = T.min(kv_start + block_N, seqlen_kv) - src_start
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(
                            j >= valid_start,
                            T.if_then_else(j < valid_end, acc_s[i, j], -T.infinity(accum_dtype)),
                            -T.infinity(accum_dtype))
                elif tail_mode == 2:
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(
                            kv_start + j < kv_end, acc_s[i, j], -T.infinity(accum_dtype))

                T.copy(scores_max, scores_max_prev)
                # 删除了 T.fill(-inf)，直接利用 scores_max_prev 的遗留值进行无缝归约
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            # 先用 16 次除法求出倒数
            for i in T.Parallel(block_H):
                inv_logsum[i] = 1.0 / logsum[i]
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] *= inv_logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

            T.copy(logsum, glse[bid, hid * valid_block_h:(hid + 1) * valid_block_h, bz])
            T.copy(acc_o, output_partial[bid, hid * valid_block_h:(hid + 1) * valid_block_h, bz, 0:dim], coalesced_width=8)

        with T.Kernel(heads, batch, dim_chunk_num * combine_groups, threads=combine_threads) as (hid, bz, dim_grp):
            dim_start = (dim_grp % dim_chunk_num) * dim_per_chunk
            glse_shared = T.alloc_shared([glse_size], dtype)
            
            po_local = T.alloc_fragment([dim_per_chunk], dtype)
            o_accum_local = T.alloc_fragment([dim_per_chunk], accum_dtype)
            # 【phase2 优化】LSE 归约：原实现是 2×num_split 步的串行标量依赖链，
            # 期间 63/64 个线程全程空闲。改为「并行装载 + 并行 exp2 + 两次树形归约」。
            # 归约列数取 glse_size（已向上取整到 64 的倍数），padding 项填 -inf：
            # 对 max 无影响，且 exp2(-inf - max) = 0，对 sum 同样无影响。
            lse_frag = T.alloc_fragment([1, glse_size], accum_dtype)
            lse_max_local = T.alloc_fragment([1], accum_dtype)
            lse_sum_local = T.alloc_fragment([1], accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)

            T.clear(o_accum_local)

            # 这里不用T.copy是因为 如果num_split = 312的话，128个线程不能平分
            for k in T.Parallel(glse_size):
                if k < num_split:
                    glse_shared[k] = glse[bz, hid, k]
        
            for k in T.Parallel(glse_size):
                if k >= num_split:
                    glse_shared[k] = -T.infinity(dtype)
            for k in T.Parallel(glse_size):
                lse_frag[0, k] = glse_shared[k]
            T.reduce_max(lse_frag, lse_max_local, dim=1, clear=True)
            for k in T.Parallel(glse_size):
                lse_frag[0, k] = T.exp2(lse_frag[0, k] - lse_max_local[0])
            T.reduce_sum(lse_frag, lse_sum_local, dim=1)
            lse_logsum_local = T.log2(lse_sum_local[0]) + lse_max_local[0]
            # 融合循环：一次并行，0 寄存器碎片，直接 FMA
            group_chunk = num_split // combine_groups
            for k in T.serial(group_chunk):
                scale_local = T.exp2(
                    glse_shared[(dim_grp // dim_chunk_num) * group_chunk + k] - lse_logsum_local)

                # # 唯一的并行域：直接融合全局内存读取与缩放累加
                # for i in T.Parallel(dim):
                #     o_accum_local[i] += output_partial[bz, hid, k, i] * scale_local

                # 1. 直接用 T.copy 将 global 内存一口气倒进 po_local 寄存器
                T.copy(output_partial[bz, hid, (dim_grp // dim_chunk_num) * group_chunk + k,
                                     dim_start : dim_start + dim_per_chunk], po_local)
                
                # 2. 纯寄存器级别的 FMA 计算，0 延迟
                for i in T.Parallel(dim_per_chunk):
                    o_accum_local[i] += po_local[i] * scale_local
            if combine_groups > 1:
                # 第一阶段：各组只归约自己那 num_split/combine_groups 份
                T.copy(o_accum_local, partial2[bz, hid, dim_grp // dim_chunk_num,
                                              dim_start : dim_start + dim_per_chunk])
            else:
                T.copy(o_accum_local, Output[bz, hid, dim_start : dim_start + dim_per_chunk])

        if combine_groups > 1:
            # 第二阶段：把 combine_groups 份部分和加起来（LSE 权重已在第一阶段乘好）
            with T.Kernel(heads, batch, dim_chunk_num, threads=64) as (hid, bz, dim_idx):
                dim_start2 = dim_idx * dim_per_chunk
                po2 = T.alloc_fragment([dim_per_chunk], dtype)
                o_accum2 = T.alloc_fragment([dim_per_chunk], accum_dtype)
                T.clear(o_accum2)
                for cg2 in T.serial(combine_groups):
                    T.copy(partial2[bz, hid, cg2, dim_start2 : dim_start2 + dim_per_chunk], po2)
                    for i in T.Parallel(dim_per_chunk):
                        o_accum2[i] += po2[i]
                T.copy(o_accum2, Output[bz, hid, dim_start2 : dim_start2 + dim_per_chunk])


    return main

# ---------------------------------------------------------------------------
# Autotune：自动选择 Q 放置（shared / 寄存器）与分片参数
# ---------------------------------------------------------------------------
# Q 放置与 num_split / block_N / num_stages 原先由 _get_kernel 里的 if/else 按 kv_ctx 硬编码。
# 实测「Q 入寄存器」在 C500 上是**劣化**而非优化（消融见
# docs/mla_decode_optimization_analysis.md §4.5），因此改为交给 autotune 逐 shape 实测选择。
#
# 使用 @autotune 的三个要点：
#   1. 可调参数必须带默认值——autotune 会用「只传 shape 参数」调用一次函数做校验，
#      缺省参数会直接 TypeError；
#   2. configs 可以传函数，会以调用方传入的 (batch, heads, kv_head_num, seqlen_kv, dim, pe_dim)
#      调用，因此能按 shape 裁剪候选；
#   3. 只要调用时显式给了某个可调参数，autotune 就会跳过调优直接编译（可用作旁路）。

_SMEM_LIMIT_BYTES = 64 * 1024       # C500 每 AP 的共享内存上限
_TUNE_CHUNKS = (32, 64, 128, 256)   # 等分族：每个 split 覆盖的 KV 长度候选
_WAVE_SIZE = 104                    # C500 AP 数；num_split 取它的倍数可消除波次量化
_TUNE_WAVES = (1, 2)                # 整波族：104 × w 个 CTA（3/4 波实测全档更差，见 §2.2.6）
_TUNE_COMBINE_GROUPS = (8, 16)      # combine 的 k 维分组数候选（需整除 num_split）
_TUNE_MAX_NUM_SPLIT = 1024
# 线程数候选：64 会慢 3.92×（acc_o/线程 128 → 306 reg，占用率腰斩，§4.5.3），
# 256 被 warp 映射约束剪掉（block_N 需 ≥64，而 shared 放不下），所以只剩 128 可行。
_TUNE_THREADS = (128,)


def _q_shared_bytes(block_n, block_h, dim, pe_dim, num_stages, q_in_reg):
    """估算主 kernel 的动态 shared 占用：Q + num_stages 份 KV/K_pe + S。"""
    q_bytes = 0 if q_in_reg else block_h * (dim + pe_dim) * 2
    kv_bytes = block_n * (dim + pe_dim) * 2 * num_stages
    s_bytes = block_h * block_n * 2
    return q_bytes + kv_bytes + s_bytes


def _threads_feasible(block_n, block_h, threads):
    """主 kernel 的 warp 映射约束。

    `acc_s[block_H, block_N]` 用 `T.GemmWarpPolicy.FullCol`：`threads/64` 个 warp
    沿 N 方向切分，每个 warp 至少要吃下一个 16x16 MMA tile 的宽度，
    即 `block_N >= 16 * threads/64`。这也解释了源码注释里"threads=128 需要 block_N>=32"。
    （实测：`block_N=32, threads=256` 会触发 MACA 版 TileLang 的
    `Check failed: pb->value != 0 ... Divide by zero`，本规则可提前剪掉。）
    """
    if threads < 64 or threads % 64:
        return False
    if block_h > 16:      # 超出 16 行时 block_H 也要能被 warp 切分
        return False
    return block_n >= 16 * (threads // 64)


def mla_autotune_configs(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim):
    """按 shape 生成合法配置。

    分片数候选有两族（去重后一起交给 autotune 实测）：

    1. **等分族**：由 `chunk ∈ {32,64,128,256}` 反推 `ns = ceil(kv_ctx/chunk)`，
       每 CTA 的块数基本相同；
    2. **整波族**：`ns = 104 × w`（104 = C500 的 AP 数），让 grid.z 正好铺满 w 个完整波次，
       避免最后一波只有 `ns % 104` 个 CTA 在跑。块级均分（`_split_plan`）保证各 CTA
       的块数差 ≤ 1，所以整波族不需要「每份等长」。

    约束：
      - 每块固定 `block_N` 行，缺少的尾部由 kernel 内的列掩码处理；
      - 主 kernel 的动态 shared 不得超过 `_SMEM_LIMIT_BYTES`；
      - `threads` 必须满足 `_threads_feasible` 的 warp 映射约束。
    """
    block_h = min(16, heads // kv_head_num)
    configs = []
    for block_n in (32, 64):
        nb = (seqlen_kv + block_n - 1) // block_n
        # 等分族：由 chunk 反推 ns（会被收敛，块数均匀）
        even_ns = set()
        for chunk in _TUNE_CHUNKS:
            if chunk % block_n:
                continue
            even_ns.add((seqlen_kv + chunk - 1) // chunk)
        # 整波族：ns = 104 × w，块级均分
        wave_ns = {_WAVE_SIZE * w for w in _TUNE_WAVES if _WAVE_SIZE * w <= nb}
        candidates = ([(ns, False) for ns in sorted(even_ns)]
                      + [(ns, True) for ns in sorted(wave_ns - even_ns)])
        for num_split, spread in candidates:
            if num_split < 1 or num_split > _TUNE_MAX_NUM_SPLIT:
                continue
            if nb // num_split > 256:      # 每 CTA 块数过多，等价于旧 chunk > 256
                continue
            # k 维分组候选：串行轮数 num_split → num_split/G，需要整除。
            # 只有 num_split ≥ 128 时分组才有正收益（小 ns 下多一次 launch 反而亏）
            groups = [1] + [g for g in _TUNE_COMBINE_GROUPS
                            if num_split >= 128 and num_split % g == 0]
            for combine_groups in groups:
                for num_stages in (1, 2):
                    # Q 入寄存器实测慢 13%–64%（§4.5），五档从未被选中，只保留 shared 路径
                    for q_in_reg in (False,):
                        if _q_shared_bytes(block_n, block_h, dim, pe_dim, num_stages,
                                           q_in_reg) > _SMEM_LIMIT_BYTES:
                            continue
                        for threads in _TUNE_THREADS:
                            if not _threads_feasible(block_n, block_h, threads):
                                continue
                            configs.append({
                                "block_N": block_n,
                                "block_H": block_h,
                                "num_split": num_split,
                                "spread": spread,
                                "combine_groups": combine_groups,
                                "num_stages": num_stages,
                                "dim_chunk_num": 8,
                                "q_in_reg": q_in_reg,
                                "threads": threads,
                            })
    if not configs:
        # 保底：shape 找不到合适分片时退回单 split
        configs.append({
            "block_N": 32, "block_H": block_h, "num_split": 1,
            "num_stages": 1, "dim_chunk_num": 8, "q_in_reg": False,
            "threads": 128,
        })
    return configs


@autotune(configs=mla_autotune_configs, warmup=25, rep=100)
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn_autotuned(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim,
                        block_N=32, block_H=16, num_split=64, num_stages=1,
                        dim_chunk_num=8, q_in_reg=False, threads=128, spread=False,
                        combine_groups=1):
    """统一构建入口：由 autotune 逐 shape 选择 Q 放置、分片参数与主 kernel 线程数。"""
    softmax_scale = (dim + pe_dim) ** -0.5
    factory = flashattn_frag_q if q_in_reg else flashattn_shared_q
    return factory.get_tir(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim,
                           block_N, block_H, num_split, num_stages, dim_chunk_num,
                           softmax_scale, threads, spread, combine_groups)


_CACHE_SHARED = {}
_CACHE_FRAG = {}
_TUNE_CACHE = {}
# MLA_NO_AUTOTUNE=1 时退回原来的 if/else 硬编码配置（便于对照与排错）
_AUTOTUNE_ENABLED = os.environ.get("MLA_NO_AUTOTUNE", "0") != "1"


def _legacy_kernel(batch, heads, kv_heads, kv_ctx, dim, pe_dim):
    """回退路径：沿用原来的 if/else 硬编码配置与 2Ki Q 放置规则。"""
    # 统一使用 16 保证 Q / Output 侧的极小碎片化
    block_h = min(16, heads // kv_heads)
    dim_chunk_num = 8
    # 动态 Block_N 调优：短序列加大吞吐，长序列死守缓存
    if kv_ctx == 2048:
        block_n = 32       # 【优化核心】2048 专用大吞吐模式，一口吞下 32 个 token
        chunk_size = 32    
        raw_split = kv_ctx // chunk_size
        num_split = max(1, raw_split) # 保证 64 份，维持 4.9 波次的完美 AP 占有率
        num_stages = 1
    elif kv_ctx == 4096:
        block_n = 32       # threads=128 需要 >=32，否则 16x16 MMA tile 无法映射 4 个 warp
        chunk_size = 64    # 需 >= 2*block_n，否则每个 split 只剩一次循环，流水线失效
        raw_split = kv_ctx // chunk_size
        num_split = max(1, raw_split) # 64
        num_stages = 1
    elif kv_ctx == 8192:
        block_n = 32
        chunk_size = 64
        raw_split = kv_ctx // chunk_size
        num_split = max(1, raw_split) # 128
        num_stages = 1
    elif kv_ctx == 16384:
        block_n = 32
        chunk_size = 128
        num_split = 256
        num_stages = 1
    elif kv_ctx == 32768:
        block_n = 32       # 同上；num_stages 降为 1 以压住 shared memory（C500 上限 64KB）
        num_split = 312    # 对齐 104 的 3 波次
        num_stages = 1
    else:
        # 64K
        block_n = 32
        num_split = 312    # 对齐 104 的 3 波次
        num_stages = 1
        
    softmax_scale = (dim + pe_dim) ** -0.5
    key = (batch, heads, kv_heads, kv_ctx, dim, pe_dim, block_n, block_h, num_split, num_stages, dim_chunk_num)
    # 动态路由：2Ki 边界跨界法则
    if kv_ctx <= 2048:
        kernel = _CACHE_SHARED.get(key)
        if kernel is None:
            kernel = flashattn_shared_q(batch, heads, kv_heads, kv_ctx, dim, 
                                        pe_dim, block_n, block_h, num_split, 
                                        num_stages, dim_chunk_num, softmax_scale)
            _CACHE_SHARED[key] = kernel
        return kernel
    else:
        kernel = _CACHE_FRAG.get(key)
        if kernel is None:
            kernel = flashattn_frag_q(batch, heads, kv_heads, kv_ctx, dim, 
                                    pe_dim, block_n, block_h, num_split, 
                                    num_stages, dim_chunk_num, softmax_scale)
            _CACHE_FRAG[key] = kernel
        return kernel


def _get_kernel(batch, heads, kv_heads, kv_ctx, dim, pe_dim):
    """按 shape 选取 kernel。

    默认走 autotune：由 `flashattn_autotuned` 逐 shape 实测选择 Q 放置
    （`q_in_reg`）与 `num_split / block_N / num_stages`。若 autotune 失败，
    或设置环境变量 `MLA_NO_AUTOTUNE=1`，则回退到 `_legacy_kernel` 的 if/else 配置。
    """
    key = (batch, heads, kv_heads, kv_ctx, dim, pe_dim)
    if _AUTOTUNE_ENABLED:
        kernel = _TUNE_CACHE.get(key)
        if kernel is not None:
            return kernel
        try:
            kernel = flashattn_autotuned(batch, heads, kv_heads, kv_ctx, dim, pe_dim)
        except Exception as exc:  # noqa: BLE001 - 任何失败都退回硬编码配置
            print(f"[mla_decode] autotune 失败（{type(exc).__name__}: {exc}），"
                  f"回退到内置配置", file=sys.stderr)
        else:
            _TUNE_CACHE[key] = kernel
            return kernel
    return _legacy_kernel(batch, heads, kv_heads, kv_ctx, dim, pe_dim)




def run_kernel(q, q_pe, kv, k_pe, output, batch, heads, kv_heads, kv_ctx, dim, pe_dim):
    kernel = _get_kernel(int(batch), int(heads), int(kv_heads), int(kv_ctx), int(dim), int(pe_dim))
    kernel(q, q_pe, kv, k_pe, output)


def build_kernel_from_config(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, cfg):
    """按显式配置直接构建 kernel（跳过 autotune），供 tuning / profiler 复现选型用。"""
    softmax_scale = (dim + pe_dim) ** -0.5
    factory = flashattn_frag_q if cfg.get("q_in_reg", False) else flashattn_shared_q
    return factory(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim,
                   cfg["block_N"], cfg["block_H"], cfg["num_split"],
                   cfg.get("num_stages", 1), cfg.get("dim_chunk_num", 8),
                   softmax_scale, cfg.get("threads", 128),
                   cfg.get("spread", False), cfg.get("combine_groups", 1),
                   cfg.get("combine_threads", 64))
