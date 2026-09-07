"""导出 mhc_post kernel 的 C-like 源码，用于向量化/访存分析。

用法:
    python scripts/dump_mhc_src.py --batch 8192 --n-expand 4 --c-x 4096 \
        [--block-x-b 1] [--block-C 64] [--threads 128] [--out /tmp/mhc_post.cu]

不指定 --out 时打印到 stdout。

注意: 需要从优化版目录导入 tileops，运行前需
    export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/code/mhc_post:$PYTHONPATH
"""

import argparse
import os
import sys

MHC_ROOT = os.environ.get("MHC_ROOT", "/data/code/mhc_post")
sys.path.insert(0, MHC_ROOT)

from tileops.kernels.mhc.mhc_post import _select_mhc_post_kernel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--n-expand", type=int, default=4)
    ap.add_argument("--c-x", type=int, default=4096)
    ap.add_argument("--block-x-b", type=int, default=1)
    ap.add_argument("--block-C", type=int, default=64)
    ap.add_argument("--num-stages", type=int, default=1)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    factory = _select_mhc_post_kernel(args.batch, args.n_expand, args.c_x, "bfloat16")
    kernel = factory(args.block_x_b, args.block_C, args.num_stages, args.threads)
    src = kernel.get_kernel_source()

    header = (
        f"// batch={args.batch} n_expand={args.n_expand} c_x={args.c_x} "
        f"block_x_b={args.block_x_b} block_C={args.block_C} threads={args.threads}\n"
    )
    src = header + src

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(src)
        print(f"[dump] 已写入 {args.out}  ({len(src.splitlines())} 行)", file=sys.stderr)
    else:
        print(src)


if __name__ == "__main__":
    main()
