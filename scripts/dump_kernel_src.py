"""导出 TileLang 编译出的 C-like kernel 源码，用于访存/partition 分析。

用法:
    python scripts/dump_kernel_src.py --kv-ctx 16384 [--out /tmp/kernel_16384.cu]

不指定 --out 时打印到 stdout。

注意: TileLang 是源码就地编译的 MACA 版，运行前需
    export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/code:$PYTHONPATH
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mla_decode import _get_kernel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-ctx", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--pe-dim", type=int, default=64)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    kernel = _get_kernel(args.batch, args.heads, args.kv_heads, args.kv_ctx,
                         args.dim, args.pe_dim)
    src = kernel.get_kernel_source()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(src)
        print(f"[dump] 已写入 {args.out}  ({len(src.splitlines())} 行)", file=sys.stderr)
    else:
        print(src)


if __name__ == "__main__":
    main()
