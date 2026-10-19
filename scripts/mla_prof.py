"""MLA Decode 单档 workload，供 mcProfiler 采集使用。

用法:
    python scripts/mla_prof.py --kv-ctx 16384 [--iters 20] [--warmup 5]
    python scripts/mla_prof.py --kv-ctx 16384 --impl baseline     # 采集基线

说明:
    - 只跑选定 kv_ctx 一档，保证 mcProfiler 多轮 replay 时 workload 稳定。
    - --impl ours 用 mla_decode.py（优化版），--impl baseline 用 baseline_mla.py，
      便于对同一 shape 分别采集、再用 compare_reports.py 对比。
    - warmup 阶段的 kernel 也会被 mcProfiler 采到，需要排除时可用 --exclude。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from baseline_mla import _get_kernel as _get_kernel_baseline
from mla_decode import _get_kernel, build_kernel_from_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_PATH = os.path.join(ROOT, "mla_best_configs.json")


def _load_best(ctx):
    """读取 tune_best.py 产出的最优配置；命中则跳过 autotune，保证 profiler 复现同一 kernel。"""
    if not os.path.exists(BEST_PATH):
        return None
    with open(BEST_PATH) as f:
        return json.load(f).get(str(ctx))


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
    ap.add_argument("--impl", choices=["ours", "baseline"], default="ours",
                    help="ours=mla_decode.py（优化版）；baseline=baseline_mla.py")
    args = ap.parse_args()

    b, h, kh, d, pd = args.batch, args.heads, args.kv_heads, args.dim, args.pe_dim
    ctx = args.kv_ctx

    torch.manual_seed(0)
    q = torch.randn(b, h, d, dtype=torch.float16, device="cuda") * 0.1
    q_pe = torch.randn(b, h, pd, dtype=torch.float16, device="cuda") * 0.1
    kv = torch.randn(b, ctx, kh, d, dtype=torch.float16, device="cuda") * 0.1
    k_pe = torch.randn(b, ctx, kh, pd, dtype=torch.float16, device="cuda") * 0.1
    out = torch.empty_like(q)

    if args.impl == "baseline":
        kernel = _get_kernel_baseline(b, h, kh, ctx, d, pd)
    else:
        cfg = _load_best(ctx)
        if cfg is not None:
            print(f"[mla_prof] 使用固化选型: {cfg}", flush=True)
            kernel = build_kernel_from_config(b, h, kh, ctx, d, pd, cfg)
        else:
            kernel = _get_kernel(b, h, kh, ctx, d, pd)

    for _ in range(args.warmup):
        kernel(q, q_pe, kv, k_pe, out)
    torch.cuda.synchronize()

    for _ in range(args.iters):
        kernel(q, q_pe, kv, k_pe, out)
    torch.cuda.synchronize()

    print(f"[mla_prof] done: kv_ctx={ctx} impl={args.impl} warmup={args.warmup} iters={args.iters}")


if __name__ == "__main__":
    main()
