# MLA Decode 脚本使用指南与工作流

本文说明 `mla_decode/scripts/` 下每个脚本的用途、参数与调用方式，并给出从
「正确性 → 性能基线 → mcProfiler 采集 → 结果分析」的完整工作流。

配套阅读：[`mla_decode_best_implementation.md`](mla_decode_best_implementation.md)（当前最优实现的最终结果）、
[`mcProfiler_guideline.md`](../mcProfiler_guideline.md)（mcProfiler CLI 原始说明）、
[`mxC500.md`](mxC500.md)（C500 硬件口径）、
[`mla_decode_optimization_analysis.md`](mla_decode_optimization_analysis.md)（优化点与分析结论）。

---

## 0. 目录总览

```text
mla_decode/
├── mla_decode.py              # 优化版 MLA Decode kernel（TileLang）
├── baseline_mla.py            # 未优化基线 kernel（对照实验用）
├── test_mla.py                # 优化版 vs 基线 的批量对比（torch 参考 + 基线 kernel）
├── mcProfiler_guideline.md    # mcProfiler 命令行使用说明
├── docs/                      # 分析文档
└── scripts/
    ├── check_mla.py           # 正确性 + 性能自检（对比 PyTorch 参考）
    ├── mla_prof.py            # MLA 单档 workload（供 mcProfiler 采集）
    ├── run_mla.sh             # MLA workload 入口（mcProfiler 的 --cmdline 用）
    ├── run_mla_16384.sh       # 固定 kv_ctx=16384 的 workload 入口
    ├── run_mcprofiler.sh      # 一键 mcProfiler 采集 + 报告归档
    ├── run_mhc.sh             # mhc_post workload 入口（同一套工具链）
    ├── mhc_prof.py            # mhc_post 单 shape workload（供 mcProfiler 采集）
    ├── mhc_config_probe.py    # 逐配置实测 mhc_post 延迟（CUPTI 口径）
    ├── summarize_report.py    # 汇总单份 mcProfiler 报告的关键指标
    ├── compare_reports.py     # 对比多份报告（重点看 VL1 partition stall）
    ├── dump_kernel_src.py     # 导出 MLA kernel 的 C-like 源码
    ├── bench_mla_heads.py     # 按 head 数扫描的 benchmark（heads=128 真实配置）
    ├── dump_mhc_src.py        # 导出 mhc_post kernel 的 C-like 源码
    └── bandwidth_probe.py     # C500 HBM 实测带宽探针
```

脚本分三组：

| 组别 | 脚本 | 作用 |
| --- | --- | --- |
| 正确性 / 性能 | `check_mla.py`、`test_mla.py`、`bench_mla_heads.py` | 输出延迟、加速比、数值误差 |
| Profiling 采集 | `mla_prof.py`+`run_mla.sh`、`run_mcprofiler.sh` | 驱动 mcProfiler 采样 |
| 结果分析 | `summarize_report.py`、`compare_reports.py`、`dump_*_src.py`、`bandwidth_probe.py` | 解读报告 / 定位瓶颈 |

---

## 1. 环境前提

### 1.1 每次新 shell 都要设置 PYTHONPATH

TileLang 是容器内**源码就地编译的 MACA 版**（`/opt/tilelang-metax-v0.1.10`），不是 pip 包，
必须显式加到 `PYTHONPATH`：

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/mla_decode:$PYTHONPATH
```

> ⚠️ 不要 `pip install tileops` / `make install` / 带 `-c constraints.txt` 安装，
> 否则会拉入官方 CUDA 版 wheel，遮蔽 MACA 编译产物，kernel 编译会走错后端。

### 1.2 shell 脚本自带完整环境

`run_*.sh` 已经把 `LD_LIBRARY_PATH`、`MACA_PATH`、`PATH`、`PYTHONPATH` 全部写好，
所以 `mcProfiler` 每轮 replay 直接用 `bash scripts/run_mla.sh` 即可，不需要外部再 export。

### 1.3 mcProfiler 必须在自身安装目录下运行

`mcProfiler` 从**当前工作目录**推断安装根（读 `config/.version`）。不在
`/opt/mcProfiler-ubuntu18.04` 下运行会报：

```text
ERROR: [Errno 2] No such file or directory: '/data/config/.version'
```

`run_mcprofiler.sh` 已经 `cd /opt/mcProfiler-ubuntu18.04`，无需手动处理。

### 1.4 运行模式：本地 vs 远端

mcProfiler 支持两种执行方式：

| 模式 | 需要的参数 | 适用场景 |
| --- | --- | --- |
| **本地模式**（默认） | 不传 `--remote_*` | mcProfiler 直接在本机 fork workload，**不需要 SSH** |
| 远端模式 | `--remote_ip/--remote_port/--remote_user/--remote_pwd` | 通过 SSH 到另一台机器执行 workload |

> 本容器的 root SSH 密码未知，**远端模式会一直卡在 `init` 状态**（服务端日志只有开头的
> `/perf_progress` 轮询，目标程序不启动）。因此本工作区默认走**本地模式**。
> 需要远端时：`USE_REMOTE=1 REMOTE_PWD=<密码> bash scripts/run_mcprofiler.sh ...`

### 1.5 kernel 选型：autotune（与 profiling 的相互影响）

`mla_decode._get_kernel()` 默认**走 autotune**：按 shape 自动选择 Q 放置（shared / 寄存器）
与 `num_split / block_N / num_stages`，结论见
[`mla_decode_optimization_analysis.md`](mla_decode_optimization_analysis.md) §7。

- **首次**调用某个 shape 会编译并实测全部候选（单 shape 约 1–2 min），之后走
  进程内 + TileLang 磁盘缓存，**同 shape 再调用约 0.1 s**；
- 旁路 autotune：`MLA_NO_AUTOTUNE=1`（走原 `if/else` 硬编码配置，便于对照）。

> ⚠️ **做 mcProfiler 采集前必须先“预热”选型**，否则 autotune 会编译/启动大量额外 kernel，
> 被 profiler 一并采到，污染报告。两种做法（任选其一）：
>
> ```bash
> # a) 先用同一 shape 跑一次普通调用，把选型写进缓存
> python3 -c "import sys;sys.path.insert(0,'/data/mla_decode');\
> from mla_decode import _get_kernel;_get_kernel(1,16,1,16384,512,64)"
> # b) 或者这次采集直接关掉 autotune，固定用内置配置
> MLA_NO_AUTOTUNE=1 bash scripts/run_mcprofiler.sh 16384 0 mla_16384
> ```
>
> 同理，`check_mla.py` / `test_mla.py` 首次跑到新 shape 时也会先花时间调优，
> 属于正常现象，不是卡死。

---

## 2. 脚本清单与用法

### 2.1 `check_mla.py` — 正确性 + 性能自检（对比 PyTorch）

跑选定 `kv_ctx` 的优化版 kernel，和**独立 PyTorch 参考实现**（`torch.einsum + softmax`）
对比，输出延迟、加速比与数值误差。

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/mla_decode:$PYTHONPATH
python scripts/check_mla.py --kv-ctx 2048 4096 8192 16384 32768 65536
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--kv-ctx` | `[16384]` | 可一次给多档 |
| `--batch` / `--heads` / `--kv-heads` | 1 / 16 / 1 | |
| `--dim` / `--pe-dim` | 512 / 64 | |
| `--warmup` / `--rep` | 10 / 50 | CUDA event 计时 |

输出：

```text
  kv_ctx     Our(μs)   Torch(μs)    vsTorch   max_abs_err    cos_sim
    2048       23.97      157.08      6.55x      0.000007   1.000000
```

> 注意：对比对象是 **PyTorch**，不是 `baseline_mla`。要看「相对基线 kernel」的加速比用 `test_mla.py`。

### 2.2 `test_mla.py`（仓库根）— 优化版 vs 基线 kernel

固定六档 `kv_ctx`（2048…65536），同一组输入分别跑 `mla_decode._get_kernel` 与
`baseline_mla._get_kernel`，计时用**预分配输出**（Zero-Allocation Benchmark），
把 Python/launch 开销排除在计时区外。

```bash
python test_mla.py
```

```text
  kv_ctx     Our(μs)     Ref(μs)       vsRef
    2048       24.17      362.28      14.99x
    ...
   65536      353.17    11783.93      33.37x
```

### 2.3 `mla_prof.py` — mcProfiler 的 workload

只跑**一档** `kv_ctx`，保证 mcProfiler 多轮 replay 时 workload 稳定。

```bash
python scripts/mla_prof.py --kv-ctx 16384 [--warmup 5] [--iters 20] [--impl ours|baseline]
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--kv-ctx` | 16384 | 单档 |
| `--warmup` / `--iters` | 5 / 20 | 正式计时前的预热与迭代次数 |
| `--impl` | `ours` | `ours`=优化版；`baseline`=基线（用于对比采集） |

结束时打印 `[mla_prof] done: ...`；mcProfiler 采集中只要日志里**周期性出现**这行，就说明 workload 在正常 replay。

### 2.4 `run_mla.sh` — workload 入口（供 mcProfiler 调用）

```bash
bash scripts/run_mla.sh [kv_ctx] [warmup] [iters] [impl]
```

例：`bash scripts/run_mla.sh 16384 5 20 baseline` 采集基线。

`run_mla_16384.sh` 是固定 `kv_ctx=16384` 的等价入口。

### 2.5 `run_mcprofiler.sh` — 一键采集 + 归档

```bash
bash scripts/run_mcprofiler.sh [kv_ctx] [counts] [casename]
#   kv_ctx   默认 16384
#   counts   默认 5（建议先 5 跑通，再 0 采集全量）
#   casename 默认 mla_<kv_ctx>
```

流程：
1. `cd /opt/mcProfiler-ubuntu18.04`，以本地模式启动 `mcProfiler perf_exec`；
2. `--cwd` = 仓库根，`--cmdline` = `bash <repo>/scripts/run_mla.sh <CMD_ARGS>`；
3. 从日志里 `sed` 出 `please check report file <dir>`，把报告**整目录归档**到
   `mcprofiler_output/<OP_NAME>/<casename>/`。

可覆盖的环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `CMD_ARGS` | `<kv_ctx>` | 传给 workload 脚本的参数（采集基线时设为 `"16384 5 20 baseline"`） |
| `WORKLOAD_SCRIPT` | `run_mla.sh` | 换 `run_mhc.sh` 即可采 mhc_post |
| `METRICS` | 空（全部） | 只采指定指标，显著缩短耗时 |
| `USE_REMOTE` / `REMOTE_PWD` | `0` / — | 远端模式开关与密码 |
| `OP_NAME` | `mla_decode` | 归档子目录名 |
| `ARCHIVE_DIR` | `<repo>/mcprofiler_output` | 归档根目录 |
| `LOG` | `/tmp/mcprof_<casename>.log` | 采集日志 |

例：

```bash
# 优化版 @16384，采集全量 kernel
bash scripts/run_mcprofiler.sh 16384 0 mla_ours_16384
# 基线 @16384
CMD_ARGS="16384 5 20 baseline" bash scripts/run_mcprofiler.sh 16384 0 mla_base_16384
# mhc_post @(8192,4,4096)
WORKLOAD_SCRIPT=run_mhc.sh OP_NAME=mhc_post CMD_ARGS="8192 4 4096" \
  bash scripts/run_mcprofiler.sh 8192_4096 5 mhc_post_8192_4096
```

> 耗时经验：全指标（默认）单档约 **10 分钟**，因为 mcProfiler 会**为每组 metric 重跑一遍
> workload**（默认约 10 轮）。想快就用 `METRICS="Total Cycles Memory Access per Second ..."` 限定。

### 2.6 `summarize_report.py` — 汇总单份报告

```bash
python scripts/summarize_report.py mcprofiler_output/mla_decode/mla_ours_16384/output20261016xxxxxx
```

从 `report.txt.csv` 里挑出关键指标，并解析 `RoofLine` / `ISU stall cycles layout` /
`Instructions Comparison` / `Compute Instructions Cycles Rate` 里的 JSON：

- 吞吐与效率：`Total Cycles`、`Memory Access per Second`、`AP busy Duty`、`instruction throughput efficiency`；
- 流量：`Global Memory Read/Write bytes`；
- 缓存 / 延迟：`VL1 Hit Rate`、`L2C Hit Rate`、`Dnoc Read Average Latency`；
- 指令构成：`MTE/STE/MISC/BSM/GVM/MMA`；
- RoofLine：`case_I`（计算强度）、`case_bandwith`、`case_flops`、`MAX_I`。

### 2.7 `compare_reports.py` — 对比多份报告

```bash
python scripts/compare_reports.py <报告目录1> <报告目录2> [...]
```

重点输出 **VL1 四个 partition 的 stall 分布**（`pt0..pt3`）与均衡度，再横向对比
`Total Cycles`、`Memory Access per Second`、`AP busy Duty`、`Global Memory Read/Write bytes`、
`WAVES` 等关键指标。用于验证「某次改动是否真的改善了 partition 均衡」。

### 2.8 `dump_kernel_src.py` — 导出 MLA kernel C-like 源码

```bash
python scripts/dump_kernel_src.py --kv-ctx 16384 --out /tmp/mla_16384.cu
```

用于确认访存宽度（`uint1/2/4`）、地址表达式、swizzle 等底层细节。

`--impl {ours,baseline}` 可导出两端；`--heads N` 指定 head 数（**判断 grid 时必须用真实 head 数**：
`heads=16` 时基线 grid extent = 1，TVM 会把 `blockIdx` 常量折叠掉，看起来像"没有并行度"）。

### 2.9 `bench_mla_heads.py` — 按 head 数扫描的 benchmark

```bash
python scripts/bench_mla_heads.py                        # heads=128（DeepSeek-V2/V3 真实配置）
python scripts/bench_mla_heads.py --heads 16 128         # 与 test_mla.py 的配置对照
python scripts/bench_mla_heads.py --ctx 16384 --rep 30   # 单档
python scripts/bench_mla_heads.py --no-baseline          # 只扫优化版
```

- **为什么需要有这个脚本**：`test_mla.py` 固定 `heads=16`（≈ DeepSeek-V2-Lite）。真实 DeepSeek-V2/V3
  是 128 个 query head，而**基线 grid 第一维 = `heads // block_H`**，head 数会显著改变基线并行度，
  从而改变加速比（`heads=16` 报 13.9×–53.7×，`heads=128` 只有 5.4×–9.2×）。
- **绕过 autotune**：128 head 下 `mla_autotune_configs` 会生成上百个候选，全网格跑不动；
  本脚本显式传参直接调用 kernel 工厂，自己做 `chunk × block_N × q_in_reg` 小网格扫描，
  并对每个候选做 cos_sim 校验（不通过的不计入最优）。
- **会自动剔除不可行配置**：`dim=512, pe_dim=64` 时 `block_N=64` 的单份 KV 就是
  `64×576×2 = 73728 B > 64 KB`，扫描前按 shared 上限过滤，避免无意义的编译。

### 2.10 `bandwidth_probe.py` — 实测 HBM 带宽

```bash
python scripts/bandwidth_probe.py --elems 134217728 --rep 50
```

用 PyTorch 纯 elementwise/copy 逼近硬件上限，判断算子的「带宽利用率」该以**理论峰值
1843 GB/s** 还是**实测峰值**为分母。

### 2.11 mhc_post 相关脚本

`run_mhc.sh` / `mhc_prof.py` / `mhc_config_probe.py` / `dump_mhc_src.py` 是同一套工具链
对 mhc_post 算子的版本，用法见脚本 docstring；`mhc_prof.py` 支持 `--tune/--no-tune`。

---

## 3. 典型工作流

```text
         ┌───────────────────────────┐
         │ 1. 正确性自检 check_mla.py │  确认 kernel 能编译、cos_sim≈1.0
         └────────────┬──────────────┘
                      ▼
      ┌───────────────────────────────────┐
      │ 2. 性能对比 test_mla.py            │  优化版 vs baseline_mla，得到加速比
      └────────────┬──────────────────────┘
                   ▼
   ┌──────────────────────────────────────────────┐
   │ 3. mcProfiler 采集 run_mcprofiler.sh          │  优化版 + 基线各采一份，自动归档
   └────────────┬─────────────────────────────────┘
                ▼
   ┌────────────────────────────────────────────┐
   │ 4. 解读 summarize_report.py / compare_reports.py │  找瓶颈、验证 partition 均衡
   └────────────┬───────────────────────────────┘
                ▼
   ┌────────────────────────────────────────────┐
   │ 5. 定位细节 dump_kernel_src.py / bandwidth_probe.py │  访存宽度、带宽分母
   └────────────────────────────────────────────┘
```

对应命令（一条龙）：

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/mla_decode:$PYTHONPATH
cd /data/mla_decode

# 1) 正确性
python scripts/check_mla.py --kv-ctx 16384
# 2) 性能对比
python test_mla.py
# 3) 采集（优化版 + 基线）
bash scripts/run_mcprofiler.sh 16384 0 mla_ours_16384
CMD_ARGS="16384 5 20 baseline" bash scripts/run_mcprofiler.sh 16384 0 mla_base_16384
# 4) 解读
python scripts/summarize_report.py mcprofiler_output/mla_decode/mla_ours_16384/<output_dir>
python scripts/compare_reports.py mcprofiler_output/mla_decode/mla_ours_16384/<output_dir> \
                                  mcprofiler_output/mla_decode/mla_base_16384/<output_dir>
```

产出的报告结构：

```text
mcprofiler_output/
└── mla_decode/
    ├── mla_ours_16384/
    │   └── output2026xxxxxxxxxx/
    │       ├── report.txt        # 人类可读
    │       ├── report.txt.csv    # summarize/compare 读取的格式
    │       ├── report.txt.json
    │       ├── report.pdf
    │       └── batches.json / event_batch_*.json / mcProfiler.json
    └── mla_base_16384/ ...
```

---

## 4. 本工作区的路径适配说明

原始脚本是为**单仓库布局** `/data/code`（含 `scripts/`、`mla_decode.py`、`mhc_post/`）
编写的。本工作区把 MLA 与 mhc_post 拆成两个目录，因此做了两处适配：

| 适配点 | 改动前 | 改动后 |
| --- | --- | --- |
| 仓库根路径 | 硬编码 `/data/code` | `run_*.sh` 由 `BASH_SOURCE` 推导 `<repo>`；`mla_prof.py` 用自身路径推导；mhc 脚本用 `MHC_ROOT`（默认 `/data/mhc_post`） |
| mcProfiler 执行方式 | 远端 SSH（`--remote_ip/_user/_pwd`） | 默认**本地模式**；`USE_REMOTE=1` + `REMOTE_PWD` 可切回远端 |

其余逻辑（采集流程、归档结构、报告解析）保持不变。

---

## 5. 故障排查速查

| 现象 | 原因 / 处理 |
| --- | --- |
| `ModuleNotFoundError: No module named 'tilelang'` | 未设 `PYTHONPATH`，见 1.1 |
| `No such file or directory: '/data/config/.version'` | 没在 `/opt/mcProfiler-ubuntu18.04` 下运行 mcProfiler |
| mcProfiler 长期停在 `init`、无 `[mla_prof] done` | 远端 SSH 密码错误；改用本地模式（默认） |
| 日志刷满 `get_srvrpc_port` / `MXKW` / `topology` | **正常**，只要周期性出现 `[mla_prof] done` 即可 |
| kernel 编译走错后端 / 报 ABI 不匹配 | 环境被 pip 装成官方 CUDA 版 tilelang，需恢复 `/opt/tilelang-metax-*` |
| 报告里没有 `report.txt.csv` | 采集未完成（被 kill）或失败；查 `/tmp/mcprof_<casename>.log` |
| `--kernelname` 与 `--kernelnames` 混淆 | `--kernelname` 只是占位必填（填 `all`）；筛选用 `--kernelnames` |
