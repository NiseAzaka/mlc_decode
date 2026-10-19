"""逐档枚举配置并用 CUDA event 计时，挑出「与文档同一计时口径」的最优配置。

与 `@autotune` 的区别：官方 autotune 的 bench 口径偏保守（会选到明显更慢的配置），
这里用与 `check_mla.py` 相同的 warmup/rep + event 计时，结果可直接复现 §7.1。

用法:
    python scripts/tune_best.py --kv-ctx 2048 4096 8192
    python scripts/tune_best.py --kv-ctx 65536 --rep 50

输出:
    <repo>/mla_best_configs.json   # 每档最优配置，供 mla_prof.py / profiler 复现
    <repo>/mla_tune_all.json       # 全部候选的实测延迟（便于回看）
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mla_decode import build_kernel_from_config, mla_autotune_configs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_PATH = os.path.join(ROOT, "mla_best_configs.json")
ALL_PATH = os.path.join(ROOT, "mla_tune_all.json")


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
    return s.elapsed_time(e) / rep * 1000  # µs


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-ctx", type=int, nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--pe-dim", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rep", type=int, default=50)
    args = ap.parse_args()

    b, h, kh, d, pd = args.batch, args.heads, args.kv_heads, args.dim, args.pe_dim
    best_all = load_json(BEST_PATH)
    tune_all = load_json(ALL_PATH)

    for ctx in args.kv_ctx:
        torch.manual_seed(0)
        q = torch.randn(b, h, d, dtype=torch.float16, device="cuda") * 0.1
        q_pe = torch.randn(b, h, pd, dtype=torch.float16, device="cuda") * 0.1
        kv = torch.randn(b, ctx, kh, d, dtype=torch.float16, device="cuda") * 0.1
        k_pe = torch.randn(b, ctx, kh, pd, dtype=torch.float16, device="cuda") * 0.1
        out = torch.zeros(b, h, d, dtype=torch.float16, device="cuda")
        call_args = (q, q_pe, kv, k_pe, out)

        configs = mla_autotune_configs(b, h, kh, ctx, d, pd)
        print(f"\n=== kv_ctx={ctx}  {len(configs)} configs ===", flush=True)
        rows = []
        best = None
        for i, cfg in enumerate(configs):
            t0 = time.time()
            try:
                kernel = build_kernel_from_config(b, h, kh, ctx, d, pd, cfg)
                us = bench(kernel, call_args, args.warmup, args.rep)
            except Exception as exc:  # noqa: BLE001 - 个别配置可能编译失败
                print(f"  [{i:2d}] FAIL {cfg} ({type(exc).__name__})", flush=True)
                continue
            row = dict(cfg, latency_us=round(us, 3))
            rows.append(row)
            tag = ""
            if best is None or us < best["latency_us"]:
                best = row
                tag = "  <-- best"
            print(f"  [{i:2d}] ns={cfg['num_split']:5d} spread={int(cfg['spread'])} "
                  f"G={cfg['combine_groups']:2d} block_N={cfg['block_N']} "
                  f"qreg={int(cfg['q_in_reg'])} us={us:8.2f} "
                  f"({time.time() - t0:4.1f}s){tag}", flush=True)

        best_all[str(ctx)] = best
        tune_all[str(ctx)] = rows
        print(f"  => best: {best}", flush=True)

    with open(BEST_PATH, "w") as f:
        json.dump(best_all, f, indent=2, sort_keys=True)
    with open(ALL_PATH, "w") as f:
        json.dump(tune_all, f, indent=2, sort_keys=True)
    print(f"\nsaved {BEST_PATH}\nsaved {ALL_PATH}")


if __name__ == "__main__":
    main()
