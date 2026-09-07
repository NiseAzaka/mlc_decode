"""对指定 shape 逐个配置实测 mhc_post 延迟，用于核对 autotune 是否选对了配置。

用法:
    python scripts/mhc_config_probe.py --batch 32 --n-expand 4 --c-x 7168 \
        [--warmup 20] [--rep 100]

输出每个 (block_x_b, block_C, threads) 组合的延迟，按耗时升序排列。
"""

import argparse
import itertools
import os
import sys

MHC_ROOT = os.environ.get("MHC_ROOT", "/data/code/mhc_post")
sys.path.insert(0, MHC_ROOT)

import torch

from benchmarks.benchmark_base import BenchmarkBase
from tileops.kernels.mhc.mhc_post import MHCPostKernel
from workloads.mhc import MHCPostTest


class _ProbeBench(BenchmarkBase):
    """只借用 BenchmarkBase 的计时协议（CUPTI/kineto 纯 kernel 时间）。"""

    def calculate_flops(self):
        return None

    def calculate_memory(self):
        return None


def bench_kernel_only(shape, kernel):
    """用 BenchmarkBase 协议测纯 kernel 延迟，返回微秒。

    注意：不能用 CUDA event 直接包住 kernel 调用——那样量到的是
    Python→custom op→launch 的端到端开销（实测约 35 µs），
    比真实 kernel 时间（约 7 µs）大 5 倍，会完全淹没配置差异。
    """
    b, n, c = shape
    test = MHCPostTest(b, n, c, torch.bfloat16)
    bm = _ProbeBench(test)
    inputs = test.gen_inputs()
    res = bm.profile(kernel, *inputs)
    return res["latency_ms"] * 1000  # ms -> us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-expand", type=int, default=4)
    ap.add_argument("--c-x", type=int, default=7168)
    ap.add_argument("--block-x-b", type=int, nargs="+", default=[1, 8, 64])
    ap.add_argument("--block-C", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--threads", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rep", type=int, default=100)
    args = ap.parse_args()

    b, n, c = args.batch, args.n_expand, args.c_x

    results = []
    for bb, bc, th in itertools.product(args.block_x_b, args.block_C, args.threads):
        cfg = {"block_x_b": bb, "block_C": bc, "num_stages": 1, "threads": th}
        try:
            k = MHCPostKernel(b, n, c, torch.bfloat16, config=cfg, tune=False)
            us = bench_kernel_only((b, n, c), k)
            results.append((us, cfg))
            print(f"  bb={bb:<3} bC={bc:<4} th={th:<4} -> {us:8.2f} us", flush=True)
        except Exception as e:
            print(f"  bb={bb:<3} bC={bc:<4} th={th:<4} -> FAIL {type(e).__name__}: {str(e)[:60]}", flush=True)

    print(f"\n=== shape ({b},{n},{c}) 按耗时升序 ===")
    for us, cfg in sorted(results):
        print(f"  {us:8.2f} us  {cfg}")


if __name__ == "__main__":
    main()
