"""手写 MLA kernel 统一入口：源码准备 → mxcc 编译 → ctypes 加载 → 对拍 → 计时。

背景（详见 docs/mla_decode_best_implementation.md §12）：
- 基底是 TileLang 生成的 kernel 源码（`mla_bn16_base.cu`，block_N=16 / threads=64 / ns=64 / ctx=8192）；
- MACA 上 `cudaLaunchKernel` 没有 cu-bridge 映射，必须用 `mcFuncSetAttribute` + `mcLaunchKernel`；
- 命令行里 `mxcc` 需要 `-x maca -D__FAST_HALF_CVT__ --offload-arch=xcore1000`
  （照抄 `tilelang/jit/adapter/libgen.py` 的 MACA 分支）。

当前状态：
- ✅ 编译通过、`cos = 1.000000`
- ❌ 基线性能 246.9 µs（TileLang 同配置 main-only 35.8 µs），差 7×，待查

用法：
    python handmade/mla_handmade.py --ctx 8192 --ns 64
    python handmade/mla_handmade.py --ctx 8192 --ns 64 --opt=-O0 --ptxas
"""

import argparse
import ctypes
import os
import re
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/handmade
ROOT = os.path.dirname(HERE)                                # <repo>
BASE_CU = os.path.join(HERE, "mla_bn16_base.cu")
BUILD_DIR = os.path.join(HERE, "build")

HOST_WRAPPER = r'''
// ---- 手写 host 侧：MACA 原生 launch（cu-bridge 不桥接 cudaLaunchKernel）----
extern "C" void mla_launch(const void* Q, const void* Qpe, const void* KV, const void* Kpe,
                           void* glse, void* partial, void* out, int ns) {
    static bool inited = false;
    const int smem_main = 37888;   // Q 16K + Q_pe 2K + K_pe 2K + KV 16K + S 0.5K + workspace
    const int smem_comb = 8192;
    if (!inited) {
        mcFuncSetAttribute((const void*)main_kernel,
                           mcFuncAttributeMaxDynamicSharedMemorySize, smem_main);
        mcFuncSetAttribute((const void*)main_kernel_1,
                           mcFuncAttributeMaxDynamicSharedMemorySize, smem_comb);
        inited = true;
    }
    void* args0[] = {(void*)&KV, (void*)&Kpe, (void*)&Q, (void*)&Qpe,
                     (void*)&glse, (void*)&partial};
    mcLaunchKernel((const void*)main_kernel, dim3(1, 1, ns), dim3(64), args0, smem_main, 0);

    void* args1[] = {(void*)&out, (void*)&glse, (void*)&partial};
    mcLaunchKernel((const void*)main_kernel_1, dim3(16, 1, 8), dim3(64), args1, smem_comb, 0);
}
'''


def prepare_source():
    """把生成的 kernel 源码 + 自写 host 包装拼成可编译的 .cu。"""
    if not os.path.exists(BASE_CU):
        raise SystemExit(f"缺少基底源码 {BASE_CU}，先运行 scripts/dump_kernel_src.py 生成")
    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(BASE_CU) as f:
        base = f.read()
    src = os.path.join(BUILD_DIR, "mla_hand.cu")
    with open(src, "w") as f:
        f.write(base + HOST_WRAPPER)
    return src


def compile_so(src, opt="-O3", ptxas=False, extra=None):
    so = os.path.join(BUILD_DIR, "mla_hand.so")
    cmd = [
        "mxcc", "-x", "maca", "-w", "-Wno-error=address-of-temporary",
        "-std=c++17", "-fPIC", "-D__FAST_HALF_CVT__",
        "--offload-arch=xcore1000", "-use-fast-math", opt,
        "--shared", "-I/opt/maca/include",
        "-I/opt/tilelang-metax-v0.1.10/src",
        "-o", so, src,
    ]
    if ptxas:
        # mxcc 不认识 --ptxas-options，改用 -V 由驱动侧打印（若支持）
        cmd.insert(cmd.index("--shared"), "-V")
    if extra:
        cmd[1:1] = extra
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[compile] FAILED", file=sys.stderr)
        print(proc.stderr[:1200], file=sys.stderr)
        raise SystemExit(1)
    if ptxas:
        for line in proc.stderr.splitlines():
            if re.search(r"registers|spill|Function properties", line):
                print("[ptxas]", line.strip())
    print(f"[compile] OK {so}  (opt={opt})", flush=True)
    return so


class Runner:
    def __init__(self, so):
        self.lib = ctypes.CDLL(so)
        self.fn = self.lib.mla_launch
        self.fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int]
        self.fn.restype = None

    def __call__(self, q, qp, kv, kp, glse, partial, out, ns):
        self.fn(q.data_ptr(), qp.data_ptr(), kv.data_ptr(), kp.data_ptr(),
                glse.data_ptr(), partial.data_ptr(), out.data_ptr(), ns)


def make_inputs(ctx, ns):
    torch.manual_seed(0)
    q = (torch.randn(1, 16, 512, dtype=torch.float16, device="cuda") * 0.1).contiguous()
    qp = (torch.randn(1, 16, 64, dtype=torch.float16, device="cuda") * 0.1).contiguous()
    kv = (torch.randn(1, ctx, 1, 512, dtype=torch.float16, device="cuda") * 0.1).contiguous()
    kp = (torch.randn(1, ctx, 1, 64, dtype=torch.float16, device="cuda") * 0.1).contiguous()
    glse = torch.zeros(1, 16, ns + 64, dtype=torch.float16, device="cuda")
    partial = torch.zeros(1, 16, ns, 512 + 64, dtype=torch.float16, device="cuda")
    out = torch.zeros(1, 16, 512, dtype=torch.float16, device="cuda")
    return q, qp, kv, kp, glse, partial, out


def reference(q, qp, kv, kp):
    g = 16
    qm = q.reshape(1, 1, g, 512).permute(0, 2, 1, 3).float()
    qs = qp.reshape(1, 1, g, 64).permute(0, 2, 1, 3).float()
    kvm = kv.permute(0, 2, 1, 3).float()
    kp2 = kp.permute(0, 2, 1, 3).float()
    sc = torch.einsum("bghd,bhsd->bghs", torch.cat([qm, qs], -1), torch.cat([kvm, kp2], -1))
    at = torch.softmax(sc * ((512 + 64) ** -0.5), -1)
    out = torch.einsum("bghs,bhsd->bghd", at, kvm)
    return out.permute(0, 2, 1, 3).reshape(1, 16, 512)


def bench(fn, args, warmup=20, rep=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--ns", type=int, default=64)
    ap.add_argument("--opt", type=str, default="-O3", help="-O0/-O1/-O2/-O3/-Ofast")
    ap.add_argument("--ptxas", action="store_true", help="打印寄存器/溢出信息")
    ap.add_argument("--check", action="store_true", default=True)
    args = ap.parse_args()

    src = prepare_source()
    so = compile_so(src, opt=args.opt, ptxas=args.ptxas)
    fn = Runner(so)

    inputs = make_inputs(args.ctx, args.ns)
    *kern_args, out = inputs
    fn(*kern_args, out, args.ns)
    torch.cuda.synchronize()

    ref = reference(inputs[0], inputs[1], inputs[2], inputs[3])
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0).item()
    us = bench(fn, (*kern_args, out, args.ns))
    print(f"[handmade] ctx={args.ctx} ns={args.ns} opt={args.opt} "
          f"us={us:.2f} cos={cos:.6f}", flush=True)


if __name__ == "__main__":
    main()
