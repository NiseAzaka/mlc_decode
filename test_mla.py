"""MLA Decode 六测试点批量运行脚本 — 含 PyTorch 加速比

用法:
    python test_mla.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from mla_decode import _get_kernel
from baseline_mla import _get_kernel as _get_kernel_ref


def baseline(q, q_pe, kv, k_pe, output, batch, heads, kv_heads, kv_ctx, dim, pe_dim):
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
    output.copy_(out.permute(0, 2, 1, 3).reshape(batch, heads, dim).to(output.dtype))


def bench_kernel(kernel, q, q_pe, kv, k_pe, warmup=10, rep=50, out_idx=False):
    # 计时区外：预先分配好输出张量（Zero-Allocation Benchmark）
    out = torch.empty_like(q)
    
    # Warmup 阶段
    for _ in range(warmup):
        if out_idx:
            _ = kernel(q, q_pe, kv, k_pe)
        else:
            kernel(q, q_pe, kv, k_pe, out)
            
    torch.cuda.synchronize()
    
    # 正式计时阶段
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(rep):
        if out_idx:
            # 注意：如果 Reference 依然保留了 out_idx=[4]，这里仍会有隐式分配开销
            _ = kernel(q, q_pe, kv, k_pe)
        else:
            # 纯净的指针传递，无任何分配开销
            kernel(q, q_pe, kv, k_pe, out)
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep * 1000


def bench_torch(q, q_pe, kv, k_pe, batch, heads, kv_heads, kv_ctx, dim, pe_dim, warmup=10, rep=50):
    out = torch.empty_like(q)
    for _ in range(warmup):
        baseline(q, q_pe, kv, k_pe, out, batch, heads, kv_heads, kv_ctx, dim, pe_dim)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        baseline(q, q_pe, kv, k_pe, out, batch, heads, kv_heads, kv_ctx, dim, pe_dim)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep * 1000


TEST_CASES = [
    {"batch": 1, "heads": 16, "kv_heads": 1, "kv_ctx": 2048,  "dim": 512, "pe_dim": 64},
    {"batch": 1, "heads": 16, "kv_heads": 1, "kv_ctx": 4096,  "dim": 512, "pe_dim": 64},
    {"batch": 1, "heads": 16, "kv_heads": 1, "kv_ctx": 8192,  "dim": 512, "pe_dim": 64},
    {"batch": 1, "heads": 16, "kv_heads": 1, "kv_ctx": 16384, "dim": 512, "pe_dim": 64},
    {"batch": 1, "heads": 16, "kv_heads": 1, "kv_ctx": 32768, "dim": 512, "pe_dim": 64},
    {"batch": 1, "heads": 16, "kv_heads": 1, "kv_ctx": 65536, "dim": 512, "pe_dim": 64},
]

if __name__ == "__main__":
    print(f"设备: {torch.cuda.get_device_name(0)}")
    print(f"{'kv_ctx':>8}  {'Our(μs)':>10}  {'Ref(μs)':>10}  {'vsRef':>10}")
    print("-" * 45)

    for tc in TEST_CASES:
        try:
            b, h, kh, ctx, d, pd = tc["batch"], tc["heads"], tc["kv_heads"], tc["kv_ctx"], tc["dim"], tc["pe_dim"]
            q = torch.randn(b, h, d, dtype=torch.float16, device="cuda") * 0.1
            q_pe = torch.randn(b, h, pd, dtype=torch.float16, device="cuda") * 0.1
            kv = torch.randn(b, ctx, kh, d, dtype=torch.float16, device="cuda") * 0.1
            k_pe = torch.randn(b, ctx, kh, pd, dtype=torch.float16, device="cuda") * 0.1

            torch_us = bench_torch(q, q_pe, kv, k_pe, b, h, kh, ctx, d, pd)

            k_our = _get_kernel(b, h, kh, ctx, d, pd)
            our_us = bench_kernel(k_our, q, q_pe, kv, k_pe, out_idx=False)

            k_ref = _get_kernel_ref(b, h, kh, ctx, d, pd)
            ref_us = bench_kernel(k_ref, q, q_pe, kv, k_pe, out_idx=False)

            print(f"{ctx:>8}  {our_us:>10.2f}  {ref_us:>10.2f} {ref_us/our_us:>10.2f}x")
        except Exception as e:
            print(f"{tc['kv_ctx']:>8}  {'FAIL':>10}  {str(e)[:]}")
        torch.cuda.empty_cache()