"""C500 HBM 实测带宽探针。

用于判断一个算子的"带宽利用率"到底该以谁为分母：理论峰值(1843 GB/s) 还是实测峰值。
跑法：用 PyTorch 的纯 elementwise / copy 操作逼近硬件上限，再与算子实测带宽对比。

用法:
    python scripts/bandwidth_probe.py [--elems 134217728] [--rep 50]

输出每种访问模式的流量与带宽（GB/s）。
"""

import argparse

import torch


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
    return s.elapsed_time(e) / rep * 1000  # us


def gb(nbytes):
    # 与 mxC500.md 的 1843 GB/s 峰值保持同一口径（10^9 字节），不要用 1024^3
    return nbytes / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elems", type=int, default=8192 * 16384)  # 268MB in bf16
    ap.add_argument("--rep", type=int, default=50)
    args = ap.parse_args()

    n = args.elems
    nbytes = n * 2  # bfloat16
    print(f"设备: {torch.cuda.get_device_name(0)}")
    print(f"单张量: {n:,} 元素 = {gb(nbytes):.3f} GiB (bf16)\n")

    x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    a = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x)

    cases = [
        ("write only  (y.fill_)", lambda: y.fill_(1.0), nbytes),
        ("read+write  (y.copy_(x))", lambda: y.copy_(x), 2 * nbytes),
        ("read+write  (mul x*2)", lambda: torch.mul(x, 2.0, out=out), 2 * nbytes),
        ("2R+1W       (a+b)", lambda: torch.add(a, b, out=out), 3 * nbytes),
        ("2R+1W       (a*b+c-ish)", lambda: torch.addcmul(a, b, a, out=out), 3 * nbytes),
    ]

    print(f"{'模式':<28}{'流量(GiB)':>12}{'耗时(µs)':>12}{'带宽(GB/s)':>14}")
    print("-" * 66)
    best = 0.0
    for name, fn, traffic in cases:
        us = bench(fn, rep=args.rep)
        bw = gb(traffic) / (us * 1e-6)
        best = max(best, bw)
        print(f"{name:<28}{gb(traffic):>12.3f}{us:>12.2f}{bw:>14.1f}")

    print(f"\n实测峰值（取最大）: {best:.1f} GB/s")
    print(f"理论峰值(1843 GB/s) 的 {100*best/1843:.1f}%")


if __name__ == "__main__":
    main()
