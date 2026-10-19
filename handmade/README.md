# 手写 MLA kernel（从 TileLang 生成代码出发）

目标：以 TileLang 生成的 kernel 源码为基底，手工加入异步双缓冲（`cp.async`），
绕开 `num_stages=2` 被 64 KB shared 卡死的限制。

## 文件

| 文件 | 说明 |
| --- | --- |
| `mla_bn16_base.cu` | TileLang 生成的源码（`block_N=16, threads=64, ns=64, ctx=8192`，269 行，含 `main_kernel` + `main_kernel_1`） |
| `mla_handmade.py` | 统一入口：拼 host 包装 → `mxcc` 编译 → `ctypes` 加载 → 对拍 → 计时 |
| `build/` | 编译中间产物（`mla_hand.cu`、`mla_hand.so`），可随时删除 |

## 已打通的部分

### 1. 编译命令（照抄 `tilelang/jit/adapter/libgen.py` 的 MACA 分支）

```bash
mxcc -x maca -w -Wno-error=address-of-temporary \
     -std=c++17 -fPIC -D__FAST_HALF_CVT__ \
     --offload-arch=xcore1000 -use-fast-math -O3 \
     --shared -I/opt/maca/include -I/opt/tilelang-metax-v0.1.10/src \
     -o mla_hand.so mla_hand.cu
```

- `arch` 由 `tilelang/contrib/mxcc.py:get_maca_arch` 给出，固定 `xcore1000`；
- `-I<tilelang>/src` 是为了 `<tl_templates/maca/*.h>`；
- 不能用 `load_inline`：torch 的 `cpp_extension` 会强加
  `-D__CUDA_NO_BFLOAT16_CONVERSIONS__` 等宏，与 `tl_templates/maca/common.h` 冲突
  （`bfloat16_t → float` 转换被禁用，14 个 error）。

### 2. host 侧必须用 MACA 原生 API

cu-bridge 没有桥接 `cudaLaunchKernel` / `cudaFuncAttributeMaxDynamicSharedMemorySize`，
要换成：

```cpp
mcFuncSetAttribute((const void*)main_kernel,
                   mcFuncAttributeMaxDynamicSharedMemorySize, smem);
void* args[] = {...};
mcLaunchKernel((const void*)main_kernel, dim3(1, 1, ns), dim3(64), args, smem, 0);
```

### 3. 数值正确

`ctx=8192, ns=64` → `cos = 1.000000`（与 PyTorch 参考完全一致）。

## 基线已对齐（此前的「7 倍」是误比）

| 配置 | main-only 延迟 |
| --- | ---: |
| TileLang `block_N=16, threads=64`（与本文件基底同一配置） | **252.58 µs** |
| 自编译（同一份源码） | **255.26 µs**（差 1%） |
| TileLang `block_N=32, threads=128`（生产最优配置） | 35.8 µs |

之前拿来对比的 35.8 µs 是 **`block_N=32 + threads=128`** 的配置，而 `mla_bn16_base.cu` 是
**`block_N=16 + threads=64`**（1 个 warp）—— 后者在 §4.5.2 就已实测慢 6.55×。
所以 255 µs 是**同一配置的正确复现**，编译链路没有问题（`mcLaunchKernel`、`-O` 级别、
shared 大小都验证过，与 TileLang 一致）。

已排除：

| 假设 | 验证 | 结论 |
| --- | --- | --- |
| fast-math 缺失 | 加 `-use-fast-math` | 无变化 |
| shared 给多了 | 40960 → 精确 37888 | 无变化 |
| launch 方式（`<<<>>>` vs `mcLaunchKernel`） | 两种都测 | 完全相同 |
| 编译优化级别 | `-O3` vs `-O0` | **完全相同（255.6 vs 255.4）** ⇒ `-O` 对 device 代码没生效，或慢的原因不在代码优化 |

已排除的假设（都已验证）：

| 假设 | 验证 | 结论 |
| --- | --- | --- |
| fast-math 缺失 | 加 `-use-fast-math` | 无变化 |
| shared 给多了 | 40960 → 精确 37888 | 无变化 |
| launch 方式 | `<<<>>>` vs `mcFuncSetAttribute + mcLaunchKernel` | 完全相同 |
| 编译优化级别 | `-O3` vs `-O0` | 完全相同（255.6 vs 255.4） |
| 编译产物 | 与 TileLang 同配置（252.58）对比 | 差 1%，**已对齐** |

## 下一步：真正的瓶颈在「1 warp」

基线既已对齐，`block_N=16 + threads=64` 的 252 µs 就是 1 个 warp 的算力上限
（§4.5.2 实测慢 6.55×）。手写要突破它有两条路：

1. **把 MMA 从 `m16n16k16` 换成 `m16n8k16`**（§12.2：硬件支持 n8 粒度），
   让 `block_N=16` 能用 2 warp（每 warp 8 列），从而拿到 `block_N=32 + threads=128` 的算力，
   同时 shared 仍只有 59 KB、可以开双缓冲；
2. 保持 1 warp，靠 `cp.async` 双缓冲把访存完全隐藏 —— 但算力腰斩，估计只能到 ~130 µs，
   仍不及现状 44 µs。

⇒ 应该走第 1 条：改生成代码里的 `__builtin_mxc_mma_16x16x16f16` 为两个 `mma.sync.m16n8k16`，
并把线程数提到 128、按 8 列切 warp。这是后续主要工作量。
