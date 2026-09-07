import tilelang
import tilelang.language as T

@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
)
def flashattn_shared_q(batch, heads, kv_head_num, seqlen_kv, dim, 
                        pe_dim, block_N, block_H, num_split, 
                        num_stages, dim_chunk_num, softmax_scale):
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

        with T.Kernel(batch, heads // valid_block_h, num_split, threads=128) as (bid, hid, bz):
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
            
            T.copy(Q[bid, hid * valid_block_h:(hid + 1) * valid_block_h, :], Q_shared, coalesced_width=8)
            T.copy(Q_pe[bid, hid * valid_block_h:(hid + 1) * valid_block_h, :], Q_pe_shared, coalesced_width=8)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seqlen_kv // num_split, block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                kv_start = (seqlen_kv // num_split) * bz + k * block_N
                kv_end = (seqlen_kv // num_split) * bz + (k + 1) * block_N
                T.copy(KV[bid, kv_start:kv_end, 0, :], KV_shared, coalesced_width=8)
                T.copy(K_pe[bid, kv_start:kv_end, 0, :], K_pe_shared, coalesced_width=8)

                T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

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

        with T.Kernel(heads, batch, dim_chunk_num, threads=64) as (hid, bz, dim_idx):
            dim_start = dim_idx * dim_per_chunk
            glse_shared = T.alloc_shared([glse_size], dtype)
            
            po_local = T.alloc_fragment([dim_per_chunk], dtype)
            o_accum_local = T.alloc_fragment([dim_per_chunk], accum_dtype)
            lse_local_split = T.alloc_var(accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            lse_max_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local = -T.infinity(accum_dtype)

            T.copy(glse[bz, hid, 0:glse_size], glse_shared)
        
            for k in T.serial(num_split):
                lse_max_local = T.max(lse_max_local, glse_shared[k])
            for k in T.serial(num_split):
                lse_local_split = glse_shared[k]
                lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
            lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
            # 融合循环：一次并行，0 寄存器碎片，直接 FMA
            for k in T.serial(num_split):
                scale_local = T.exp2(glse_shared[k] - lse_logsum_local)

                # # 唯一的并行域：直接融合全局内存读取与缩放累加
                # for i in T.Parallel(dim):
                #     o_accum_local[i] += output_partial[bz, hid, k, i] * scale_local

                # 1. 直接用 T.copy 将 global 内存一口气倒进 po_local 寄存器
                T.copy(output_partial[bz, hid, k, dim_start : dim_start + dim_per_chunk], po_local)
                
                # 2. 纯寄存器级别的 FMA 计算，0 延迟
                for i in T.Parallel(dim_per_chunk):
                    o_accum_local[i] += po_local[i] * scale_local
            T.copy(o_accum_local, Output[bz, hid, dim_start : dim_start + dim_per_chunk])

    return main

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn_frag_q(batch, heads, kv_head_num, seqlen_kv, dim, 
                        pe_dim, block_N, block_H, num_split, 
                        num_stages, dim_chunk_num,softmax_scale):
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

        with T.Kernel(batch, heads // valid_block_h, num_split, threads=128) as (bid, hid, bz):
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
            
            T.copy(Q[bid, hid * valid_block_h:(hid + 1) * valid_block_h, :], Q_frag, coalesced_width=8)
            T.copy(Q_pe[bid, hid * valid_block_h:(hid + 1) * valid_block_h, :], Q_pe_frag, coalesced_width=8)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seqlen_kv // num_split, block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                kv_start = (seqlen_kv // num_split) * bz + k * block_N
                kv_end = (seqlen_kv // num_split) * bz + (k + 1) * block_N
                T.copy(KV[bid, kv_start:kv_end, 0, :], KV_shared, coalesced_width=8)
                T.copy(K_pe[bid, kv_start:kv_end, 0, :], K_pe_shared, coalesced_width=8)

                T.clear(acc_s)
                T.gemm(Q_frag, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_frag, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

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

        with T.Kernel(heads, batch, dim_chunk_num, threads=64) as (hid, bz, dim_idx):
            dim_start = dim_idx * dim_per_chunk
            glse_shared = T.alloc_shared([glse_size], dtype)
            
            po_local = T.alloc_fragment([dim_per_chunk], dtype)
            o_accum_local = T.alloc_fragment([dim_per_chunk], accum_dtype)
            lse_local_split = T.alloc_var(accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            lse_max_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local = -T.infinity(accum_dtype)

            # 这里不用T.copy是因为 如果num_split = 312的话，128个线程不能平分
            for k in T.Parallel(glse_size):
                if k < num_split:
                    glse_shared[k] = glse[bz, hid, k]
        
            for k in T.serial(num_split):
                lse_max_local = T.max(lse_max_local, glse_shared[k])
            for k in T.serial(num_split):
                lse_local_split = glse_shared[k]
                lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
            lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
            # 融合循环：一次并行，0 寄存器碎片，直接 FMA
            for k in T.serial(num_split):
                scale_local = T.exp2(glse_shared[k] - lse_logsum_local)

                # # 唯一的并行域：直接融合全局内存读取与缩放累加
                # for i in T.Parallel(dim):
                #     o_accum_local[i] += output_partial[bz, hid, k, i] * scale_local

                # 1. 直接用 T.copy 将 global 内存一口气倒进 po_local 寄存器
                T.copy(output_partial[bz, hid, k, dim_start : dim_start + dim_per_chunk], po_local)
                
                # 2. 纯寄存器级别的 FMA 计算，0 延迟
                for i in T.Parallel(dim_per_chunk):
                    o_accum_local[i] += po_local[i] * scale_local
            T.copy(o_accum_local, Output[bz, hid, dim_start : dim_start + dim_per_chunk])


    return main

_CACHE_SHARED = {}
_CACHE_FRAG = {}


def _get_kernel(batch, heads, kv_heads, kv_ctx, dim, pe_dim):
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




def run_kernel(q, q_pe, kv, k_pe, output, batch, heads, kv_heads, kv_ctx, dim, pe_dim):
    kernel = _get_kernel(int(batch), int(heads), int(kv_heads), int(kv_ctx), int(dim), int(pe_dim))
    kernel(q, q_pe, kv, k_pe, output)
