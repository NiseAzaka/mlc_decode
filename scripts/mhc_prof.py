"""mhc_post 单 shape workload，供 mcProfiler 采集使用。

用法:
    python scripts/mhc_prof.py --batch 8192 --n-expand 4 --c-x 4096 \
        [--warmup 5] [--iters 20] [--tune]

说明:
    - 默认 tune=True，与 Benchmark 的口径一致（走 autotune 选出的配置）；
      采集单次 kernel 特征时可用 --no-tune 固定默认配置。
"""

import argparse
import os
import sys

MHC_ROOT = os.environ.get("MHC_ROOT", "/data/code/mhc_post")
sys.path.insert(0, MHC_ROOT)

import torch

from tileops.ops import MHCPostOp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--n-expand", type=int, default=4)
    ap.add_argument("--c-x", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tune", dest="tune", action="store_true", default=True)
    ap.add_argument("--no-tune", dest="tune", action="store_false")
    args = ap.parse_args()

    b, n, c = args.batch, args.n_expand, args.c_x

    torch.manual_seed(0)
    x_layer_out = torch.randn(b, c, dtype=torch.bfloat16, device="cuda")
    h_post = torch.randn(b, n, dtype=torch.float32, device="cuda")
    x_res = torch.randn(b, n * c, dtype=torch.bfloat16, device="cuda")

    op = MHCPostOp(tune=args.tune)

    for _ in range(args.warmup):
        op(x_layer_out, h_post, x_res)
    torch.cuda.synchronize()

    for _ in range(args.iters):
        op(x_layer_out, h_post, x_res)
    torch.cuda.synchronize()

    print(f"[mhc_prof] done: B={b} N={n} C={c} warmup={args.warmup} iters={args.iters} tune={args.tune}")


if __name__ == "__main__":
    main()
