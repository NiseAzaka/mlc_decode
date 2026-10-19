#!/usr/bin/env python3
"""MLA Decode 按 head 数扫描的 benchmark。

默认扫描 ``heads=128``（DeepSeek-V2 / V3 / R1 的真实配置，见 README 讨论），
可与 ``--heads 16``（本仓库 test_mla.py 的小配置，对应 DeepSeek-V2-Lite）对照。

用法::

    export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/mla_decode:$PYTHONPATH
    python scripts/bench_mla_heads.py                       # heads=128，ctx 2048..65536
    python scripts/bench_mla_heads.py --heads 16 128        # 双配置对照
    python scripts/bench_mla_heads.py --ctx 16384 --rep 30  # 单档快速
    python scripts/bench_mla_heads.py --no-baseline         # 跳过 baseline_mla

说明:
    - 显式给全部可调参数直接调用 ``flashattn_shared_q`` / ``flashattn_frag_q``，
      **绕过 autotune**：128 head 下 ``mla_autotune_configs`` 会生成上百个候选
      （autotune 实测 + 编译一次约 10s），全网格跑不动，这里改由脚本自己做
      小网格扫描（chunk × block_N × q_in_reg）。
    - 每个候选都跑一次数值校验（cos_sim vs PyTorch），只把通过的配置计入最优。
    - ``--heads`` 默认 KV head 数为 1（吸收后的 MLA 形态），即 GQA group == heads。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from baseline_mla import _get_kernel as _get_kernel_ref  # noqa: E402
from mla_decode import flashattn_frag_q, flashattn_shared_q  # noqa: E402

DIM = 512            # kv_lora_rank（吸收后的 K/V 维度）
PE_DIM = 64          # qk_rope_head_dim
KV_HEADS = 1         # 吸收后 KV 只剩一个共享 latent
DIM_CHUNK = 8
BLOCK_H_CAP = 16
COS_SIM_MIN = 0.99
SMEM_LIMIT_BYTES = 64 * 1024   # C500 每 AP 共享内存上限
_SWEEP_CHUNKS = (64, 128, 256)
_SWEEP_BLOCK_N = (32, 64)
_SWEEP_Q_PLACE = (False, True)


def torch_ref(q, q_pe, kv, k_pe, batch, heads, kv_heads, dim, pe_dim):
    """与 test_mla.py:baseline 一致的 PyTorch 参考实现（fp32，非吸收形式）。"""
    group = heads // kv_heads
    q_main = q.reshape(batch, kv_heads, group, dim).permute(0, 2, 1, 3).float()
    q_pos = q_pe.reshape(batch, kv_heads, group, pe_dim).permute(0, 2, 1, 3).float()
    kv_main = kv.permute(0, 2, 1, 3).float()
    k_pos = k_pe.permute(0, 2, 1, 3).float()
    query = torch.cat([q_main, q_pos], dim=-1)
    key = torch.cat([kv_main, k_pos], dim=-1)
    scores = torch.einsum("bghd,bhsd->bghs", query, key)
    attn = torch.softmax(scores * ((dim + pe_dim) ** -0.5), dim=-1)
    out = torch.einsum("bghs,bhsd->bghd", attn, kv_main)
    return out.permute(0, 2, 1, 3).reshape(batch, heads, dim)


def bench(kernel, q, q_pe, kv, k_pe, out, warmup, rep):
    for _ in range(warmup):
        kernel(q, q_pe, kv, k_pe, out)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        kernel(q, q_pe, kv, k_pe, out)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep * 1000.0


def _smem_bytes(block_n, block_h, num_stages, q_in_reg):
    """主 kernel 动态 shared 占用估算（与 mla_decode._q_shared_bytes 同口径）。"""
    q_bytes = 0 if q_in_reg else block_h * (DIM + PE_DIM) * 2
    kv_bytes = block_n * (DIM + PE_DIM) * 2 * num_stages
    s_bytes = block_h * block_n * 2
    return q_bytes + kv_bytes + s_bytes


def candidates(ctx):
    """小网格候选：保证 chunk 整除 ctx、chunk % block_N == 0 且 shared 不超限。

    smem=64KB/AP 的硬上限会直接砍掉 block_N=64（64*(512+64)*2 = 73728B 已超限，
    实测会以 ``mcModuleLaunchKernel: mcErrorInvalidValue`` 失败），预先过滤可省去
    无意义的编译。
    """
    cfgs, seen = [], set()
    block_h = BLOCK_H_CAP
    for chunk in _SWEEP_CHUNKS:
        if ctx % chunk:
            continue
        num_split = ctx // chunk
        if num_split > 1024:
            continue
        for block_n in _SWEEP_BLOCK_N:
            if chunk % block_n:
                continue
            for q_in_reg in _SWEEP_Q_PLACE:
                if _smem_bytes(block_n, block_h, 1, q_in_reg) > SMEM_LIMIT_BYTES:
                    continue
                key = (block_n, num_split, q_in_reg)
                if key in seen:
                    continue
                seen.add(key)
                cfgs.append({"block_N": block_n, "num_split": num_split,
                             "num_stages": 1, "q_in_reg": q_in_reg, "threads": 128})
    return cfgs


def sweep_one(heads, ctx, args):
    batch = args.batch
    scale = (DIM + PE_DIM) ** -0.5
    block_h = min(BLOCK_H_CAP, heads // KV_HEADS)

    q = torch.randn(batch, heads, DIM, dtype=torch.float16, device="cuda") * 0.1
    q_pe = torch.randn(batch, heads, PE_DIM, dtype=torch.float16, device="cuda") * 0.1
    kv = torch.randn(batch, ctx, KV_HEADS, DIM, dtype=torch.float16, device="cuda") * 0.1
    k_pe = torch.randn(batch, ctx, KV_HEADS, PE_DIM, dtype=torch.float16, device="cuda") * 0.1
    out = torch.empty_like(q)

    ref = torch_ref(q, q_pe, kv, k_pe, batch, heads, KV_HEADS, DIM, PE_DIM)

    rows = []
    for cfg in candidates(ctx):
        fac = flashattn_frag_q if cfg["q_in_reg"] else flashattn_shared_q
        tag = f"blk_n={cfg['block_N']:<3} split={cfg['num_split']:<4} q_in_reg={int(cfg['q_in_reg'])}"
        try:
            t0 = time.time()
            kernel = fac(batch, heads, KV_HEADS, ctx, DIM, PE_DIM, cfg["block_N"], block_h,
                         cfg["num_split"], cfg["num_stages"], DIM_CHUNK, scale,
                         threads=cfg["threads"])
            compile_s = time.time() - t0
            out.zero_()
            kernel(q, q_pe, kv, k_pe, out)
            torch.cuda.synchronize()
            cos = torch.nn.functional.cosine_similarity(
                out.float().flatten(), ref.flatten(), dim=0).item()
            if cos < COS_SIM_MIN:
                print(f"    [skip] {tag}  cos_sim={cos:.4f} < {COS_SIM_MIN}", flush=True)
                continue
            us = bench(kernel, q, q_pe, kv, k_pe, out, args.warmup, args.rep)
            rows.append({"us": us, "cfg": cfg, "cos": cos, "compile_s": compile_s})
            print(f"    [ok]   {tag}  {us:8.2f} us  cos_sim={cos:.5f}  "
                  f"(compile {compile_s:4.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001 - 配置不可用时跳过
            print(f"    [fail] {tag}  {type(exc).__name__}: {str(exc)[:90]}", flush=True)
        finally:
            torch.cuda.empty_cache()

    if not rows:
        print(f"  heads={heads:<4} ctx={ctx:<6} 全部候选失败", flush=True)
        return None

    rows.sort(key=lambda r: r["us"])
    best = rows[0]

    base_us = None
    if not args.no_baseline:
        try:
            k_ref = _get_kernel_ref(batch, heads, KV_HEADS, ctx, DIM, PE_DIM)
            base_us = bench(k_ref, q, q_pe, kv, k_pe, out, args.warmup,
                            max(5, args.rep // 4))
        except Exception as exc:  # noqa: BLE001
            print(f"    [fail] baseline {type(exc).__name__}: {str(exc)[:90]}", flush=True)

    rec = {"heads": heads, "ctx": ctx, "us": best["us"], "cfg": best["cfg"],
           "cos": best["cos"], "n_ok": len(rows), "base_us": base_us}
    speed = f"{base_us / best['us']:6.2f}x" if base_us else "   n/a"
    base_s = f"{base_us:9.2f}" if base_us else "      n/a"
    print(f"  heads={heads:<4} ctx={ctx:<6} our={best['us']:8.2f}us  base={base_s}us  "
          f"speedup={speed}  best: blk_n={best['cfg']['block_N']} "
          f"split={best['cfg']['num_split']} q_in_reg={int(best['cfg']['q_in_reg'])} "
          f"({len(rows)}/{len(candidates(ctx))} 候选通过)", flush=True)

    del q, q_pe, kv, k_pe, out, ref
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, nargs="+", default=[128])
    ap.add_argument("--ctx", type=int, nargs="+",
                    default=[2048, 4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=50)
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()

    print(f"device={torch.cuda.get_device_name(0)} dim={DIM} pe_dim={PE_DIM} "
          f"kv_heads={KV_HEADS}", flush=True)
    print(f"heads={args.heads}  ctx={args.ctx}  warmup={args.warmup} rep={args.rep}", flush=True)

    all_recs = []
    for heads in args.heads:
        print(f"\n===== heads={heads} (group={heads // KV_HEADS}, "
              f"block_H={min(BLOCK_H_CAP, heads // KV_HEADS)}) =====", flush=True)
        for ctx in args.ctx:
            rec = sweep_one(heads, ctx, args)
            if rec:
                all_recs.append(rec)

    if all_recs:
        print("\n===== 汇总 =====", flush=True)
        print(f"{'heads':>6}{'ctx':>8}{'our(us)':>11}{'base(us)':>11}{'speedup':>10}"
              f"{'blk_n':>7}{'split':>7}{'q_in_reg':>10}{'cos':>9}", flush=True)
        for r in all_recs:
            speed = f"{r['base_us'] / r['us']:.2f}x" if r["base_us"] else "n/a"
            base_s = f"{r['base_us']:.2f}" if r["base_us"] else "n/a"
            print(f"{r['heads']:>6}{r['ctx']:>8}{r['us']:>11.2f}{base_s:>11}{speed:>10}"
                  f"{r['cfg']['block_N']:>7}{r['cfg']['num_split']:>7}"
                  f"{int(r['cfg']['q_in_reg']):>10}{r['cos']:>9.5f}", flush=True)


if __name__ == "__main__":
    main()
