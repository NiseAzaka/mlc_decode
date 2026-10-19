"""方案 A 探针：手写异步双缓冲（block_N=16, threads=64 + T.maca_async_copy）。

只实现主 kernel（写 glse / output_partial），用于和现状的「main-only」时间对比：
    现状 main-only：2048 = 13.68 µs，8192 = 35.80 µs，65536 ≈ 185 µs

设计：
    - 两份 KV buffer（KV_a / KV_b）+ 两份 K_pe（PA / PB），各自一个 maca barrier；
    - unroll 2：偶数块用 a、奇数块用 b，发下一块的同时计算当前块；
    - 要求 nb % (2*ns) == 0（六档在 ns=64 时都满足）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tilelang
import tilelang.language as T
import torch


def build_main(ctx, ns=64, block_N=16, block_H=16, threads=64, dim=512, pe_dim=64):
    softmax_scale = (dim + pe_dim) ** -0.5
    scale = float(softmax_scale * 1.44269504)
    dtype = T.float16
    accum_dtype = T.float32
    nb = (ctx + block_N - 1) // block_N
    assert nb % ns == 0, f"nb({nb}) % ns({ns}) != 0"
    per = nb // ns
    assert per % 2 == 0, f"per({per}) 必须为偶数（unroll 2）"
    glse_pad = 64
    dim_pad = 64

    @T.prim_func
    def main(
        Q: T.Tensor([1, 16, dim], dtype),
        Q_pe: T.Tensor([1, 16, pe_dim], dtype),
        KV: T.Tensor([1, ctx, 1, dim], dtype),
        K_pe: T.Tensor([1, ctx, 1, pe_dim], dtype),
    ):
        glse = T.alloc_global([1, 16, ns + glse_pad], dtype)
        output_partial = T.alloc_global([1, 16, ns, dim + dim_pad], dtype)

        with T.Kernel(1, 1, ns, threads=threads) as (bid, hid, bz):
            Q_s = T.alloc_shared([block_H, dim], dtype)
            Qp_s = T.alloc_shared([block_H, pe_dim], dtype)
            KV_a = T.alloc_shared([block_N, dim], dtype)
            KV_b = T.alloc_shared([block_N, dim], dtype)
            PA = T.alloc_shared([block_N, pe_dim], dtype)
            PB = T.alloc_shared([block_N, pe_dim], dtype)
            S_s = T.alloc_shared([block_H, block_N], dtype)
            bar = T.alloc_maca_barrier(2)

            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)
            inv_logsum = T.alloc_fragment([block_H], accum_dtype)

            T.annotate_layout({
                KV_a: tilelang.layout.make_swizzled_layout(KV_a),
                KV_b: tilelang.layout.make_swizzled_layout(KV_b),
                PA: tilelang.layout.make_swizzled_layout(PA),
                PB: tilelang.layout.make_swizzled_layout(PB),
                S_s: tilelang.layout.make_swizzled_layout(S_s),
            })
            T.use_swizzle(panel_size=10, enable=True)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.copy(Q[bid, hid * block_H:(hid + 1) * block_H, :], Q_s, coalesced_width=8)
            T.copy(Q_pe[bid, hid * block_H:(hid + 1) * block_H, :], Qp_s, coalesced_width=8)

            kv_base = bz * (per * block_N)

            # 预取第 0 块到 a
            T.maca_async_copy(KV[bid, kv_base:kv_base + block_N, 0, :], KV_a, barrier=bar[0])
            T.maca_async_copy(K_pe[bid, kv_base:kv_base + block_N, 0, :], PA, barrier=bar[0])

            for k2 in T.serial(per // 2):
                off1 = kv_base + (k2 * 2 + 1) * block_N
                # 预取奇数块到 b（最后一轮会重复读末块，结果不参与计算）
                T.maca_async_copy(KV[bid, off1:off1 + block_N, 0, :], KV_b, barrier=bar[1])
                T.maca_async_copy(K_pe[bid, off1:off1 + block_N, 0, :], PB, barrier=bar[1])
                T.maca_barrier_arrive_and_wait(bar[0])   # 等 a 就绪

                # ---- 计算 a ----
                T.clear(acc_s)
                T.gemm(Q_s, KV_a, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Qp_s, PA, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_s)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(S_s, KV_a, acc_o, policy=T.GemmWarpPolicy.FullCol)

                # 预取下一轮的偶数块到 a（越界 clamp，不会被计算）
                off2 = kv_base + (k2 * 2 + 2) * block_N
                off2c = T.min(off2, ctx - block_N)
                T.maca_async_copy(KV[bid, off2c:off2c + block_N, 0, :], KV_a, barrier=bar[0])
                T.maca_async_copy(K_pe[bid, off2c:off2c + block_N, 0, :], PA, barrier=bar[0])
                T.maca_barrier_arrive_and_wait(bar[1])   # 等 b 就绪

                # ---- 计算 b ----
                T.clear(acc_s)
                T.gemm(Q_s, KV_b, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Qp_s, PB, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_s)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(S_s, KV_b, acc_o, policy=T.GemmWarpPolicy.FullCol)

            for i in T.Parallel(block_H):
                inv_logsum[i] = 1.0 / logsum[i]
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] *= inv_logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

            T.copy(logsum, glse[bid, hid * block_H:(hid + 1) * block_H, bz])
            T.copy(acc_o, output_partial[bid, hid * block_H:(hid + 1) * block_H, bz, 0:dim],
                   coalesced_width=8)

    return main


def synth(ctx, dim=512, pe_dim=64):
    torch.manual_seed(0)
    q = torch.randn(1, 16, dim, dtype=torch.float16, device="cuda") * 0.1
    qp = torch.randn(1, 16, pe_dim, dtype=torch.float16, device="cuda") * 0.1
    kv = torch.randn(1, ctx, 1, dim, dtype=torch.float16, device="cuda") * 0.1
    kp = torch.randn(1, ctx, 1, pe_dim, dtype=torch.float16, device="cuda") * 0.1
    return q, qp, kv, kp


def bench(k, args, warmup=20, rep=50):
    for _ in range(warmup):
        k(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        k(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep * 1000


if __name__ == "__main__":
    for ctx in (2048, 8192, 65536):
        try:
            fn = build_main(ctx, ns=64, block_N=16, threads=64)
            k = tilelang.compile(fn, out_idx=None)
            args = synth(ctx)
            us = bench(k, args)
            print(f"[async-pipe] ctx={ctx:6d} us={us:8.2f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).replace("\n", " ")[:200]
            print(f"[async-pipe] ctx={ctx:6d} FAIL {type(exc).__name__}: {msg}", flush=True)
