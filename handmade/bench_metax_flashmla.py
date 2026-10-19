"""在 OJ 的 shape 上测沐曦官方 FlashMLA（flash_mla_cuda 预编译版）。

shape: b=1, s_q=1, h_q=16, h_kv=1, d=576, dv=512, block_size=64
对比基线：我们 TileLang 实现的 22.34 / 29.61 / 44.43 / 68.43 / 114.20 / 196.63 µs
"""

import os
import sys

import torch

sys.path.insert(0, "/data/flashmla_metax")

from flash_mla import flash_mla_with_kvcache, get_mla_metadata  # noqa: E402

OURS = {2048: 22.34, 4096: 29.61, 8192: 44.43, 16384: 68.43, 32768: 114.20, 65536: 196.63}


def run(ctx, block_size=64, rep=50, warmup=20):
    b, s_q, h_q, h_kv, d, dv = 1, 1, 16, 1, 576, 512
    max_seqlen_pad = ((ctx + 255) // 256) * 256
    num_blocks = b * max_seqlen_pad // block_size

    q = torch.randn(b, s_q, h_q, d, dtype=torch.float16, device="cuda") * 0.1
    blocked_k = torch.randn(num_blocks, block_size, h_kv, d,
                            dtype=torch.float16, device="cuda") * 0.1
    block_table = torch.arange(num_blocks, dtype=torch.int32,
                               device="cuda").view(b, -1)
    cache_seqlens = torch.full((b,), ctx, dtype=torch.int32, device="cuda")

    meta, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

    def call():
        return flash_mla_with_kvcache(q, blocked_k, block_table, cache_seqlens,
                                      dv, meta, num_splits, causal=True)

    out, lse = call()
    torch.cuda.synchronize()
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        call()
    e.record()
    torch.cuda.synchronize()
    us = s.elapsed_time(e) / rep * 1000

    # 数值校验：与 fp32 参考比 cos（只对第一档做，避免长 ctx 太慢）
    cos = None
    if ctx <= 8192:
        qf = q[0, 0].float()                      # [h_q, d]
        kf = blocked_k[:, :, 0, :].reshape(-1, d)[:ctx].float()   # [ctx, d]
        sc = qf @ kf.T * (d ** -0.5)
        at = torch.softmax(sc, -1)
        ref = at @ kf[:, :dv]
        cos = torch.nn.functional.cosine_similarity(
            out[0, 0].float().flatten(), ref.flatten(), dim=0).item()

    nsplit = int(num_splits.max().item()) if num_splits.numel() > 1 else int(num_splits[0].item())
    base = OURS.get(ctx)
    ratio = f"{us / base:.2f}x" if base else "-"
    cosstr = f"{cos:.6f}" if cos is not None else "-"
    print(f"[metax] ctx={ctx:6d} ns={nsplit:4d} us={us:8.2f}  (ours {base} µs, 比 {ratio})  cos={cosstr}",
          flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    for ctx in (2048, 4096, 8192, 16384, 32768, 65536):
        try:
            run(ctx)
        except Exception as exc:  # noqa: BLE001
            print(f"[metax] ctx={ctx} FAIL {type(exc).__name__}: {str(exc)[:180]}", flush=True)
