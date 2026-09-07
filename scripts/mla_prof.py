"""MLA Decode 单档 workload，供 mcProfiler 采集使用。

用法:
    python scripts/mla_prof.py --kv-ctx 16384 [--iters 20] [--warmup 5]

说明:
    - 只跑选定 kv_ctx 一档，保证 mcProfiler 多轮 replay 时 workload 稳定。
    - warmup 阶段的 kernel 也会被 mcProfiler 采到，需要排除时可用 --exclude。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mla_decode import _get_kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-ctx", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--pe-dim", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    b, h, kh, d, pd = args.batch, args.heads, args.kv_heads, args.dim, args.pe_dim
    ctx = args.kv_ctx

    torch.manual_seed(0)
    q = torch.randn(b, h, d, dtype=torch.float16, device="cuda") * 0.1
    q_pe = torch.randn(b, h, pd, dtype=torch.float16, device="cuda") * 0.1
    kv = torch.randn(b, ctx, kh, d, dtype=torch.float16, device="cuda") * 0.1
    k_pe = torch.randn(b, ctx, kh, pd, dtype=torch.float16, device="cuda") * 0.1
    out = torch.empty_like(q)

    kernel = _get_kernel(b, h, kh, ctx, d, pd)

    for _ in range(args.warmup):
        kernel(q, q_pe, kv, k_pe, out)
    torch.cuda.synchronize()

    for _ in range(args.iters):
        kernel(q, q_pe, kv, k_pe, out)
    torch.cuda.synchronize()

    print(f"[mla_prof] done: kv_ctx={ctx} warmup={args.warmup} iters={args.iters}")


if __name__ == "__main__":
    main()
