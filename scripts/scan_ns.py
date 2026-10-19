"""扫描 num_split 的细粒度取值（步长 16），找粗网格之间是否还有更优点。

用法:
    python scripts/scan_ns.py --kv-ctx 8192 16384
    python scripts/scan_ns.py --kv-ctx 8192 --ns 64,80,96,128 --groups 1

说明:
    - 固定 block_N=32 / block_H=16 / threads=128 / num_stages=1 / q_in_reg=False；
    - 用 `spread=True`（块级均分），因此任意 ns 都能落在「前 r 个 CTA 多领一块」的分配上；
    - ns 取到 nb 为止（nb = ceil(kv_ctx/32)），超过会被 _split_plan 收敛。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mla_decode import build_kernel_from_config, _split_plan


def bench(kernel, args, warmup=20, rep=50):
    for _ in range(warmup):
        kernel(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        kernel(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-ctx", type=int, nargs="+", required=True)
    ap.add_argument("--ns", type=str, default="64,80,96,112,128,144,160,176,192,208,224,240,256")
    ap.add_argument("--groups", type=int, default=1, help="combine_groups")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--pe-dim", type=int, default=64)
    args = ap.parse_args()

    b, h, kh, d, pd = args.batch, args.heads, args.kv_heads, args.dim, args.pe_dim
    ns_list = [int(x) for x in args.ns.split(",")]

    for ctx in args.kv_ctx:
        torch.manual_seed(0)
        q = torch.randn(b, h, d, dtype=torch.float16, device="cuda") * 0.1
        q_pe = torch.randn(b, h, pd, dtype=torch.float16, device="cuda") * 0.1
        kv = torch.randn(b, ctx, kh, d, dtype=torch.float16, device="cuda") * 0.1
        k_pe = torch.randn(b, ctx, kh, pd, dtype=torch.float16, device="cuda") * 0.1
        out = torch.zeros(b, h, d, dtype=torch.float16, device="cuda")
        call = (q, q_pe, kv, k_pe, out)

        nb = (ctx + 31) // 32
        print(f"\n=== kv_ctx={ctx}  nb={nb}  G={args.groups} ===", flush=True)
        best = None
        for ns_req in ns_list:
            cfg = {
                "block_N": 32, "block_H": 16, "num_split": ns_req,
                "spread": True, "combine_groups": args.groups,
                "num_stages": 1, "dim_chunk_num": 8,
                "q_in_reg": False, "threads": 128,
            }
            try:
                kernel = build_kernel_from_config(b, h, kh, ctx, d, pd, cfg)
                us = bench(kernel, call)
            except Exception as exc:  # noqa: BLE001
                print(f"  ns={ns_req:5d} FAIL {type(exc).__name__}: {exc}", flush=True)
                continue
            ns_eff, q_blk, r = _split_plan(ctx, ns_req, 32, True)
            tag = ""
            if best is None or us < best[1]:
                best = (ns_eff, us)
                tag = "  <-- best"
            print(f"  ns={ns_req:5d} -> eff={ns_eff:5d} blocks={q_blk}+{r:4d}  us={us:8.2f}{tag}", flush=True)
        print(f"  => best ns={best[0]}  {best[1]:.2f} us", flush=True)


if __name__ == "__main__":
    main()
