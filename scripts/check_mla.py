"""MLA Decode 正确性 + 性能校验。

用法:
    python scripts/check_mla.py --kv-ctx 16384
    python scripts/check_mla.py --kv-ctx 2048 4096 8192 16384 32768 65536

输出每档的 kernel 耗时、PyTorch 参考耗时、加速比与数值误差。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mla_decode import _get_kernel


def torch_ref(q, q_pe, kv, k_pe, batch, heads, kv_heads, kv_ctx, dim, pe_dim):
    group_num = heads // kv_heads
    q_main = q.reshape(batch, kv_heads, group_num, dim).permute(0, 2, 1, 3).float()
    q_pos = q_pe.reshape(batch, kv_heads, group_num, pe_dim).permute(0, 2, 1, 3).float()
    kv_main = kv.permute(0, 2, 1, 3).float()
    k_pos = k_pe.permute(0, 2, 1, 3).float()
    query = torch.cat([q_main, q_pos], dim=-1)
    key = torch.cat([kv_main, k_pos], dim=-1)
    scores = torch.einsum("bghd,bhsd->bghs", query, key)
    attention = torch.softmax(scores * ((dim + pe_dim) ** -0.5), dim=-1)
    out = torch.einsum("bghs,bhsd->bghd", attention, kv_main)
    return out.permute(0, 2, 1, 3).reshape(batch, heads, dim)


def bench(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-ctx", type=int, nargs="+", default=[16384])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--pe-dim", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=50)
    args = ap.parse_args()

    b, h, kh, d, pd = args.batch, args.heads, args.kv_heads, args.dim, args.pe_dim
    print(f"设备: {torch.cuda.get_device_name(0)}")
    print(f"{'kv_ctx':>8}  {'Our(μs)':>10}  {'Torch(μs)':>10}  {'vsTorch':>9}  {'max_abs_err':>12}  {'cos_sim':>9}")
    print("-" * 70)

    torch.manual_seed(0)
    for ctx in args.kv_ctx:
        q = torch.randn(b, h, d, dtype=torch.float16, device="cuda") * 0.1
        q_pe = torch.randn(b, h, pd, dtype=torch.float16, device="cuda") * 0.1
        kv = torch.randn(b, ctx, kh, d, dtype=torch.float16, device="cuda") * 0.1
        k_pe = torch.randn(b, ctx, kh, pd, dtype=torch.float16, device="cuda") * 0.1
        out = torch.empty_like(q)
        try:
            kernel = _get_kernel(b, h, kh, ctx, d, pd)
            our_us = bench(lambda: kernel(q, q_pe, kv, k_pe, out),
                           warmup=args.warmup, rep=args.rep)
            ref = torch_ref(q, q_pe, kv, k_pe, b, h, kh, ctx, d, pd)
            torch_us = bench(lambda: torch_ref(q, q_pe, kv, k_pe, b, h, kh, ctx, d, pd),
                             warmup=args.warmup, rep=args.rep)
            err = (out.float() - ref.float()).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                out.float().flatten(), ref.float().flatten(), dim=0).item()
            print(f"{ctx:>8}  {our_us:>10.2f}  {torch_us:>10.2f}  {torch_us/our_us:>8.2f}x  {err:>12.6f}  {cos:>9.6f}")
        except Exception as e:
            import traceback
            print(f"{ctx:>8}  FAIL {type(e).__name__}: {str(e)[:120]}")
            traceback.print_exc()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
