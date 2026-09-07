# PR B：[mhc_post] optimize: 优化 MHC 后处理的泛化分派、访存效率与编译安全

## 小组课题信息

课题名称：面向 MetaX C500 的 MHC 后处理算子泛化与性能分析优化研究

课题完整简介：本课题面向 MHC（Manifold-Constrained Hyper-Connections）后处理链路中的 `mhc_post` 算子，在保持

```text
x_out[b, n, c] =
    h_post[b, n] * float(x_layer_out[b, c])
    + float(x_res[b, n, c])
```

语义不变的前提下，补全原实现对非整块通道、任意扩展分支数、空维度、输入契约和 `torch.compile(fullgraph=True)` 的支持，并针对 MetaX C500 设计通用逐分支 Kernel 与 `N=4` 合并分支 Kernel 的 shape-aware 分派。性能评估同时采用独立 PyTorch reference 和未优化 Kernel 两个 baseline：优化后的实现在 15 组 mHC/DeepSeek V4 workload 上相对 PyTorch reference 取得 `5.811×–14.171×` 加速，几何平均加速约 `9.998×`；相对未优化 Kernel 有 13 组取得加速，整体几何平均加速约 `1.385×`。

除 Kernel 优化外，本课题还修复了原 Benchmark 中会将 TFLOPS 夸大 `N*C+1` 倍的 FLOPs 公式错误，将性能矩阵从 3 个微型 shape 扩展到 15 个真实/泛化 workload，将定向验证从 3 个常规 case 扩展到 38 项测试，并交付“一键正确性检查 → Benchmark → Markdown/CSV 报告”和“15-case 最终 Review 矩阵”工具。成果不只是一个更快的 Kernel，也是一套可复现、可审阅、可继续扩展的 C500 算子性能验证闭环。

### PR 简要描述

算子名称：`mhc_post` / `MHCPostOp`

算子认领 Issue：`[提交前回填：GitLink 算子认领 Issue 编号与 URL]`

关联 Manifest PR（PR A）：`[提交前回填：PR A 编号与 URL]`

小组：第 4 组

成员：

认领类型：主算子

候选清单状态：待优化

候选清单难度：中

改动类型：

- [ ] `feat`：新增算子或功能
- [x] `optimize`：优化已有实现

来源与版本：

- 优化前源文件：`TileOPs-Metax/tileops/kernels/mhc/mhc_post.py`
- 优化前源提交 SHA：`181801d4be993b757008c857ccbc2befc7061105`
- 优化实现审阅快照：`feat/final-test@84b385cbe3418e41507e914443ec0efa990cb156`
- Benchmark/报告脚本叠加快照：`feat/mhc-post@d1fa7ffb54ccf8b9e99c6f4eeaac45e5c83cf852`
- 最终打包说明：审阅包是上述实现快照与 Benchmark 脚本快照组成的双来源快照，不应写成单一 Git 提交
- 最终 PR HEAD / 实测提交 SHA：`[提交前回填；bench_analysis.md 本身未嵌入被测 SHA]`

本 PR 只包含 `mhc_post` 相关改动；共享文件中的 `mhc_pre` 不属于本 PR 的实现、测试或性能结论范围。

### 算子接口与语义

| 张量          | 逻辑形状     | dtype      | 约束                   |
| ------------- | ------------ | ---------- | ---------------------- |
| `x_layer_out` | `[B, C]`     | `bfloat16` | C500/MACA 设备、连续   |
| `h_post`      | `[B, N]`     | `float32`  | 与其他输入同设备、连续 |
| `x_res`       | `[B, N * C]` | `bfloat16` | 与其他输入同设备、连续 |
| `x_out`       | `[B, N * C]` | `bfloat16` | 连续输出               |

每个输出元素执行一次 FP32 乘法和一次 FP32 加法，最终转换为 BF16。

### 核心成果与开源价值

1. **Kernel 性能与泛化同时完成。** 在保留任意 `N`、非整块 C tail 和完整接口契约的同时，15 组 workload 全部快于独立 PyTorch reference，相对未优化 Kernel 则有 13 组取得加速；相对 PyTorch 的最大加速为 `14.171×`，相对未优化 Kernel 的最大加速为 `3.113×`，大 workload 的估算有效带宽达到 `1.45–1.50 TB/s`。
2. **修复影响性能可信度的基础设施错误。** 原 `MHCPostBenchmark.calculate_flops()` 错用近似矩阵乘法复杂度，导致 TFLOPS 被夸大数千至数万倍；本 PR 将 Benchmark、Manifest 和 Roofline 统一到真实逐元素 FMA 口径。
3. **验证规模显著扩展。** Benchmark 从 3 个小 shape 扩展为 15 个 mHC/DeepSeek V4 decode、prefill 与 `N=2/8` 泛化 workload；定向测试从 3 个常规 case 扩展为 38 项，覆盖数值、边界、异常、分派、缓存和编译路径。
4. **把人工整理升级为一键、可审计流水线。** `scripts/mhc_bench_report.py` 可零参数依次执行正确性测试和 Benchmark，并从原始日志自动生成 Markdown/CSV；审阅归档中的 `mhc_post_final_matrix.py` 可生成 15-case dispatch、配置、源码哈希、barrier、输出契约与逐元素 oracle 结果。
5. **提供面向维护者的 Review 资产。** `AGENTS.md` 固化 `mhc_post` 审阅范围、正确性基准、性能口径、验证命令和问题报告格式，防止共享文件中的 `mhc_pre` 结果被误纳入本 PR。

## 1. 本次 PR 优化方案

### 优化前的主要问题

1. 原 Kernel 使用 `c_x // block_C` 构造 C 维 grid，`C` 不是 `block_C` 整数倍时尾部元素无法覆盖。
2. 原 Kernel 的一个 program 在内部处理所有 `N`，并为 `x_res`、`x_out` 分配 shared memory；资源开销随 `N` 增长，且单次使用的数据存在冗余暂存。
3. 原实现缺少任意 `N`、最小 shape、空维度、跨 shape 缓存复用等完整契约。
4. Op 层未完整拒绝错误 dtype、CPU/跨设备输入和非连续输入。
5. FakeTensor 输出沿用了 `x_layer_out` 的 `[B,C]` shape，与真实 `[B,N*C]` 输出不一致；cold `torch.compile(fullgraph=True)` 路径不安全。
6. 原 Benchmark 只有 3 个小 shape；更严重的是 FLOPs 误写成 `2*B*(N²*C²+N*C)`，而真实计算只有 `2*B*N*C`，会把 TFLOPS 夸大 `N*C+1` 倍。原 memory 统计也漏掉 batch、dtype 字节数与 `h_post` FP32 流量。
7. 原始日志需要人工换算和拼表，缺少从正确性门禁、Benchmark 到资源分析和 Review 证据的自动化闭环，容易产生手抄错误和统计口径漂移。

### 采用的优化方案

1. **Tail-safe 通用 Kernel。** 使用 `T.ceildiv` 与边界 mask 覆盖任意 `B/N/C`；整块 shape 使用 `T.copy`，不规则 shape 使用带 mask 的加载。
2. **保持 C 维 CTA 并行度。** C tile 继续作为 grid 维度，避免把 C 维改成 program 内循环后损失 block 级并行度。
3. **按 shape 分派。**
   - `N == 4 && B >= 8`：选择 all-N4 2D Kernel，一个 program 合并四个分支并复用 `x_layer_out`。
   - 其他 shape：选择任意 `N` 的通用 3D Kernel。
4. **精简 shared-memory 路径。** `h_post` 与可复用的 `x_layer_out` 保留在 shared memory；只使用一次的 `x_res` 直接从 global memory 读取，`x_out` 直接写回。
5. **可验证的配置空间。** 搜索 `block_x_b={1,8,64}`、`block_C={64,128}`、`threads={128,256}`；`num_stages=1` 仅为配置与 ABI 兼容保留。
6. **补全 Op 契约。** 增加 dtype、rank、shape、device 与 contiguous 检查；空维度返回正确空输出；Kernel cache key 包含 shape、dtype、device 与 tune 状态。
7. **补全编译路径。** 修正 custom op 的 FakeTensor 输出 metadata，并在 tracing 时通过固定安全配置进入注册的 custom op，避免在 Dynamo 图内构造 TileLang JIT factory。
8. **修复性能统计并扩展性能契约。** 将 FLOPs 修正为真实的 `2*B*N*C`，按实际 dtype 字节数统计 `x_layer_out/h_post/x_res/x_out`，并将 Benchmark/Manifest 扩展到 15 个 mHC/DeepSeek V4 decode、prefill 与 `N=2/8` 泛化 workload。
9. **交付一键报告与 Review 工具。** 报告工具直接解析 `profile_run.log`，同一份结构化数据同时驱动延迟换算、两个 baseline 对比、block/shared/HBM/FLOPs/带宽利用率分析和可选 CSV，避免硬编码 15 行数据；Review 矩阵工具记录每个 case 的 dispatch、配置、生成源码 SHA256、barrier 数、输出 metadata、最大误差和逐元素 PyTorch oracle 结论。

### 关键代码或配置变更

| 文件                                         | 改动                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| `tileops/kernels/mhc/mhc_post.py`            | tail-safe 通用 Kernel、all-N4 Kernel、shape-aware selector、直接 residual/output I/O、custom op/fake/compile 路径与配置空间 |
| `tileops/ops/mhc.py`                         | 输入契约、空维度、跨设备/非连续检查、Kernel cache 与 compile 分派 |
| `tests/ops/test_mhc.py`                      | 正确性、边界、异常、固定配置、分派边界、cache 复用、FakeTensor 与 fullgraph 测试 |
| `benchmarks/ops/bench_mhc.py`                | 15 组 workload、独立 PyTorch 与未优化 Kernel 两个 baseline、FLOPs 与 memory 统计修正 |
| `tileops/manifest/sequence_modeling.yaml`    | 语义、shape/dtype、15 组 workload 与 Roofline 公式           |
| `scripts/mhc_bench_report.py`                | 一键执行正确性与 Benchmark；仅提取 `MHCPostOp`，自动生成 Markdown/CSV、双 baseline 对比、资源、访存与带宽分析 |
| `mhc_post_final_matrix.py`（审阅归档根目录） | 一键生成 15-case Review 矩阵：dispatch、配置、源码 SHA256/大小/barrier、输出契约、最大误差与逐元素 oracle |
| `AGENTS.md`                                  | 固化最终 Review 范围、正确性基准、性能原则、验证命令和问题报告格式 |

### 优化实验结论

- 单独删除 shared staging 等候选在早期固定配置实验中没有稳定超过 3% 噪声门槛；最终方案以完整 Benchmark 与 Profiler 证据为准。
- all-N4 的主要作用是合并四个 `N` 分支、复用 C tile，并减少 program/wave 与重复访存指令。
- 通用 Kernel 保留任意 `N` 与非整块 tail 的正确性；最终 `bench_analysis.md` 中 `N=2`、`N=8` workload 相对 PyTorch reference 分别达到 `10.596×` 与 `8.329×`。
- Benchmark 的 TFLOPS 修复不是展示层调整，而是把计算量从错误的 `O(B*N²*C²)` 恢复为与 Kernel 语义一致的 `O(B*N*C)`；这使性能数字、Profiler 解释和 Roofline 判断第一次处于同一可信口径。
- 报告生成器不在父进程导入 TileLang，正确性与 Benchmark 使用独立 pytest 子进程，规避 C500 环境中父子进程重复导入 TileLang 可能触发的 `exit 137`。
- 本报告不再引用上一版 PRB 中 v09/v11 的过期延迟或回归结论；第 3 节性能表统一呈现最终优化后 `MHCPostOp/tileops` 数据，并同时与独立 PyTorch reference、未优化 Kernel 两个 baseline 比较。

## 2. 精度验证

精度对比基准实现：独立 PyTorch reference。未优化 Kernel 仅作为第 3 节的性能 baseline，不作为正确性 oracle。

```python
(
    h_post.unsqueeze(2).float()
    @ x_layer_out.unsqueeze(1).float()
).reshape(B, N * C) + x_res.float()
```

参考结果最终转换为 `bfloat16`，不复用 TileLang Kernel 的实现逻辑。

测试命令：

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:$PWD:${PYTHONPATH:-}
python -m pytest -q tests/ops/test_mhc.py -k "mhc_post"
```

归档 C500 运行使用的等价独立测试文件命令：

```bash
PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/TileOPs-Metax \
python -m pytest -q tests/ops/test_mhc.py
```

测试矩阵：

- 原始版本只有 3 个常规 shape；本次扩展后的定向套件共 38 项测试节点。
- 常规 shape：`(B,N,C)=(1,4,1280)`、`(2,4,1920)`、`(4,4,2560)`。
- 最小与泛化：`(1,1,1)`、`(2,4,1024)`、`(1,3,128)`。
- 非整块 C tail：`C=65/127/129`。
- 分派边界：`B=7 -> 8 -> 7`、`N=4`，并验证 `N!=4` 回退。
- 空维度：`B=0`、`N=0`、`C=0`。
- 配置矩阵：通用 Kernel 与 all-N4 Kernel 均覆盖 `C64/C128 × T128/T256`。
- 异常输入：错误 dtype/rank/batch/residual width、CPU、跨设备、非连续输入。
- 编译契约：FakeTensor metadata 与 cold `torch.compile(fullgraph=True)`。
- dtype：`x_layer_out/x_res=bfloat16`、`h_post=float32`、`x_out=bfloat16`。

误差范围：`rtol=1.6e-2`，`atol=1.6e-2`。

归档验证结果：

| 验证项                                              |                                                      结果 | 退出码 |
| --------------------------------------------------- | --------------------------------------------------------: | -----: |
| 定向正确性                                          |                                     `38 passed in 30.24s` |    `0` |
| PR Benchmark 门禁                                   |                                 `3 passed, 12 deselected` |    `0` |
| Short Benchmark 门禁                                |                                  `7 passed, 8 deselected` |    `0` |
| Full Benchmark 门禁                                 |                                 `13 passed, 2 deselected` |    `0` |
| Nightly Benchmark 门禁                              |                                               `15 passed` |    `0` |
| `benchmarks/tests`                                  |                                               `18 passed` |    `0` |
| `tests/test_ops_manifest.py`                        |                                                `7 passed` |    `0` |
| `scripts/validate_manifest.py --check-op MHCPostOp` | 通过，含 1 条 synthetic-input shape precondition advisory |    `0` |
| `git diff --check`                                  |                                                      通过 |    `0` |
| `pre-commit run --all-files`                        |                              未运行：归档环境中命令不可用 |  `127` |

提交前完整校验命令：

```bash
git diff --check
python scripts/validate_manifest.py
python -m pytest -q tests/ops/test_mhc.py -k "mhc_post"
python -m pytest -q benchmarks/tests
python -m pytest -q tests/test_ops_manifest.py
pre-commit run --all-files
```

证据边界：

- 上述 C500 正确性与门禁结果来自最终打包归档中的 v10/final 审计日志。
- 归档结果文件以短 SHA `8ba899b` 标识该轮测试源码；日志没有记录其完整 SHA，不能自行补全。
- 最终实现审阅快照为 `84b385cbe3418e41507e914443ec0efa990cb156`；Benchmark 脚本来自 `d1fa7ffb54ccf8b9e99c6f4eeaac45e5c83cf852`。
- 打包过程本身没有重新运行测试或 GPU 作业；提交 PR 前应在最终单一 PR HEAD 上重跑并回填该 SHA。

## 3. 性能数据【必填】

### 测试环境

| 项目                          | 配置                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| GPU                           | MetaX C500，完整物理卡                                       |
| sGPU                          | Disabled；不是切片                                           |
| 显存                          | 65536 MiB                                                    |
| Python                        | 3.12.11                                                      |
| PyTorch                       | `2.8.0+metax3.7.1.3`                                         |
| `torch.version.cuda` 兼容字段 | `11.6`                                                       |
| TileLang                      | `0.1.10+cuda.gitf549117c`，路径 `/opt/tilelang-metax-v0.1.10` |
| TileLang target               | `maca`                                                       |
| MACA                          | 3.7.1.5                                                      |
| mx-smi                        | 2.3.1                                                        |
| Kernel mode driver            | 3.8.30                                                       |
| L2 cache                      | 8 MiB                                                        |

`torch.cuda.get_device_capability()` 的兼容值为 `(8,0)`，这里只用于仓库门禁，不代表 NVIDIA Ampere。

### 性能协议与复现命令

`BenchmarkBase` 协议：

- 10 次 warmup；
- 每个 trial 测量 50 次，共 3 个 trial；
- 默认使用 CUPTI/Kineto 统计纯 Kernel 时间；trace 不可信时才回退 CUDA/MACA event；
- 每次测量前写满实际 L2 大小的扰动缓冲区并同步；
- 输入不超过 1 GiB 时预生成 3 组 clone 并轮换地址；
- 报告 3 个 trial mean 的中位数；
- 编译、autotune 与首次缓存开销不计入稳定态 Kernel latency。

最终性能数据来源：`/data/logs/mhc_post_202608051533.log`，报告生成时间 `2026-08-05 07:36:10`。报告行没有出现 event fallback 标记。

本节设置两个性能对比 baseline：

1. **独立 PyTorch reference**：用于衡量 TileOps Kernel 相对框架参考实现的加速效果。
2. **未优化 Kernel**：用于衡量本 PR 的 Kernel 优化相对原始实现带来的直接收益。

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:$PWD:${PYTHONPATH:-}

# PR：3 个正式小 shape
python -m pytest -q benchmarks/ops/bench_mhc.py \
  -k "mhc_post and (small or medium or large)" -s

# Short：7 个 shape
python -m pytest -q benchmarks/ops/bench_mhc.py \
  -k "mhc_post and (small or medium or large or decode-s or expand)" -s

# Full/Nightly：15 个 workload
python -m pytest -q benchmarks/ops/bench_mhc.py -k "mhc_post" -s
```

### 一键报告与 Review 证据生成

提交的报告脚本不需要手工拼接表格。零参数运行时，它会先执行 `mhc_post` 正确性门禁，只有测试通过才执行完整 Benchmark，随后解析 `profile_run.log` 并生成统一口径的 Markdown 报告：

```bash
python scripts/mhc_bench_report.py
```

也可以对已有日志离线生成 Markdown 与 CSV 宽表：

```bash
python scripts/mhc_bench_report.py \
  --input profile_run.log \
  --output mhc_post_bench_report.md \
  --csv mhc_post_bench_report.csv
```

审阅归档还提供最终矩阵工具，在真实 C500 上逐一检查 15 个 workload，并生成机器可读 Review 证据：

```bash
python mhc_post_final_matrix.py --output mhc_post_final_matrix.json
```

该矩阵不仅记录 pass/fail，还记录 public selector 实际选择的 mapping、配置、生成源码 SHA256/大小/barrier 数、输出 shape/dtype、最大绝对误差，以及全元素独立 PyTorch oracle 结论。这样维护者可以复核“运行的是哪份设备代码”，而不只看到一张手工整理的性能表。

### Benchmark TFLOPS 口径修复

原 Benchmark 的 FLOPs 公式为：

```text
FLOPs_old = 2 * B * (N² * C² + N * C)
```

但 `mhc_post` 的每个输出元素只执行一次乘法和一次加法，真实公式应为：

```text
FLOPs_new = 2 * B * N * C
```

因此：

```text
FLOPs_old / FLOPs_new = N * C + 1
```

在本次 15 组 workload 中，旧公式会将 TFLOPS 夸大约 `4,097–28,673` 倍。该问题会直接扭曲算力利用率和 Roofline 判断，并非单纯的显示误差。本 PR 同时完成：

- 将 Benchmark FLOPs 修正为 `2*B*N*C`，与 Kernel 逐元素 FMA 语义及 Manifest 完全一致；
- 将 memory 从未乘 batch/字节数的元素计数，修正为包含 BF16 `x_layer_out/x_res/x_out` 与 FP32 `h_post` 的真实字节口径；
- 让 Benchmark、自动报告、Profiler 分析和 Roofline 使用同一组公式，防止后续贡献者再次引用失真的 TFLOPS。

### 两个 baseline 与优化后 `mhc_post` 的 15 组结果

下表同时给出两个比较对象。“优化后”是 `tileops` 行；“独立 PyTorch baseline”是 `torch-ref` 行，其加速比定义为 `torch-ref_us / tileops_us`；“未优化 Kernel baseline”是优化前实现，其加速比定义为 `base_us / tileops_us`。因此，加速比大于 `1×` 表示优化后实现更快，小于 `1×` 表示优化后实现存在回退。

| case `(B,N,C)`                     | 优化后 mhc_post (µs) | 独立 PyTorch baseline (µs) | 相对 PyTorch 加速比 | 未优化 Kernel baseline (µs) | 相对未优化 Kernel 加速比 | 估算有效带宽 (GB/s) | C500 1843 GB/s 利用率 |
| ---------------------------------- | -------------------: | -------------------------: | ------------------: | --------------------------: | -----------------------: | ------------------: | --------------------: |
| small `(1,4,1280)`                 |                 3.50 |                      27.40 |              7.829× |                        3.70 |                   1.057× |                 8.8 |                  0.5% |
| medium `(2,4,1920)`                |                 3.60 |                      22.80 |              6.333× |                        3.50 |                   0.972× |                25.9 |                  1.4% |
| large `(4,4,2560)`                 |                 3.70 |                      21.50 |              5.811× |                        4.00 |                   1.081× |                66.8 |                  3.6% |
| v4-flash-decode-s `(32,4,4096)`    |                 6.00 |                      44.80 |              7.467× |                        5.50 |                   0.917× |               395.9 |                 21.5% |
| v4-pro-decode-s `(32,4,7168)`      |                 6.30 |                      59.20 |              9.397× |                        7.30 |                   1.159× |               659.9 |                 35.8% |
| v4-flash-decode-m `(128,4,4096)`   |                11.70 |                     111.20 |              9.504× |                       12.10 |                   1.034× |               812.2 |                 44.1% |
| v4-pro-decode-m `(128,4,7168)`     |                14.80 |                     178.20 |             12.041× |                       19.00 |                   1.284× |              1123.6 |                 61.0% |
| v4-flash-decode-l `(512,4,4096)`   |                31.60 |                     377.60 |             11.949× |                       38.90 |                   1.231× |              1202.9 |                 65.3% |
| v4-pro-decode-l `(512,4,7168)`     |                51.60 |                     645.00 |             12.500× |                       69.20 |                   1.341× |              1289.1 |                 69.9% |
| v4-flash-prefill-s `(1024,4,4096)` |                56.80 |                     734.20 |             12.926× |                       90.30 |                   1.590× |              1338.4 |                 72.6% |
| v4-pro-prefill-s `(1024,4,7168)`   |                97.00 |                    1267.30 |             13.065× |                      145.50 |                   1.500× |              1371.5 |                 74.4% |
| v4-flash-prefill-l `(8192,4,4096)` |               405.60 |                    5747.60 |             14.171× |                     1262.70 |                   3.113× |              1499.4 |                 81.4% |
| v4-pro-prefill-l `(4096,4,7168)`   |               367.60 |                    5024.20 |             13.668× |                     1029.70 |                   2.801× |              1447.6 |                 78.5% |
| expand2-c2048 `(128,2,2048)`       |                 5.70 |                      60.40 |             10.596× |                       11.60 |                   2.035× |               554.8 |                 30.1% |
| expand8-c1024 `(128,8,1024)`       |                 7.00 |                      58.30 |              8.329× |                        8.50 |                   1.214× |               903.5 |                 49.0% |

汇总结论：

- 相对独立 PyTorch baseline，15 组 workload 全部取得加速；加速范围为 `5.811×–14.171×`，中位数 `10.596×`，几何平均约 `9.998×`。
- 相对未优化 Kernel baseline，15 组中有 13 组取得加速；加速比范围为 `0.917×–3.113×`，中位数 `1.231×`，几何平均约 `1.385×`。`medium` 和 `v4-flash-decode-s` 两组分别为 `0.972×` 和 `0.917×`，即存在约 `2.8%` 和 `8.3%` 的轻微回退。
- 大 workload 的估算有效带宽达到 `1.45–1.50 TB/s`，约为完整 C500 `1843 GB/s` 理论参考带宽的 `78.5%–81.4%`。
- 最终报告中的 `N=2/8` 泛化 workload 已分别达到 `5.70 µs` 和 `7.00 µs`，不能继续使用上一版 PRB 的 `20.0 µs`、`35.5 µs` 数据。

带宽口径说明：

- 上表带宽采用 `bench_analysis.md`“资源与性能分析/表 3”的估算 HBM 流量除以优化后延迟，包含 Kernel mapping 导致的 `x_layer_out` 读放大，未扣除 L2 命中。
- `bench_analysis.md` 前部原始 `bandwidth_tbs` 列与后部按字节重算结果明显不一致（all-N4 路径约 2 倍，通用路径还叠加了 `x_layer_out` 读放大）；本 PR 不使用该原始列作为 Roofline 证据。
- 未优化 Kernel baseline 与优化后数据使用相同的 15 组 shape 和微秒单位；表中的“相对未优化 Kernel 加速比”统一按 `base_us / tileops_us` 计算，并与 PyTorch reference 对比分列展示，避免混淆两种 baseline。

### Roofline 公式与实测

Manifest 公式：

```text
FLOPs = 2 * B * N * C

Bytes =
    (B * C + 2 * B * N * C) * elem_bytes
    + B * N * 4

elem_bytes = 2
```

公式分别计入 `x_layer_out` 读取、`h_post` FP32 读取、`x_res` 读取和 `x_out` 写回。典型 `N=4` shape 的算术强度约为 `0.444 FLOP/Byte`，属于低算术强度算子。

本次使用完整 C500 的 `1843 GB/s` 理论 DRAM-L2 参考带宽，不进行 sGPU 折算。打包证据没有建立同一软件栈下的实测 `BW_peak` 微基准或 FP32 vector `P_peak`，因此下表报告的是 `AI × 1843 GB/s` 的**内存参考 Roofline**，不将其包装成已经校准的完整 Roofline。

| case `(B,N,C)`  |       FLOPs |      逻辑字节 | AI (FLOP/B) | 优化后延迟 |       Achieved |   内存参考上限 | Achieved / 参考上限 |
| --------------- | ----------: | ------------: | ----------: | ---------: | -------------: | -------------: | ------------------: |
| `(128,4,4096)`  |   4,194,304 |   9,439,232 B |     0.44435 |   11.70 µs | 0.3585 TFLOP/s | 0.8189 TFLOP/s |               43.8% |
| `(8192,4,4096)` | 268,435,456 | 604,110,848 B |     0.44435 |  405.60 µs | 0.6618 TFLOP/s | 0.8189 TFLOP/s |               80.8% |
| `(4096,4,7168)` | 234,881,024 | 528,547,840 B |     0.44439 |  367.60 µs | 0.6390 TFLOP/s | 0.8190 TFLOP/s |               78.0% |
| `(128,2,2048)`  |   1,048,576 |   2,622,464 B |     0.39984 |    5.70 µs | 0.1840 TFLOP/s | 0.7369 TFLOP/s |               25.0% |
| `(128,8,1024)`  |   2,097,152 |   4,460,544 B |     0.47016 |    7.00 µs | 0.2996 TFLOP/s | 0.8665 TFLOP/s |               34.6% |

### mcProfiler 瓶颈分析

Profiler 证据来自最终打包目录 `mcProfiler-v4 flash and pro`，覆盖 DeepSeek V4 Flash/Pro 的 10 个 decode/prefill workload。每个目录均包含 raw JSON、`report.txt/json/csv`、Roofline 图、访存图和 PDF。

- `report.txt` 本身没有记录工具版本；同一最终打包归档中的 profiler metadata 记录为 mcProfiler `3.8.1.4`、build `20260715214228`。
- mcProfiler 只用于计数器和瓶颈归因，稳定态延迟仍以第 3 节 BenchmarkBase 结果为准。
- 报告中的 workgroup/wave 是 Profiler 驱动多次执行后的聚合值，不与单次 analytical grid 直接相除。

代表性计数器：

| workload           | L2C hit | VL1 hit | AP MTE duty | AP STE duty | shared efficiency | Profiler 报告带宽 |
| ------------------ | ------: | ------: | ----------: | ----------: | ----------------: | ----------------: |
| v4-flash-decode-s  |  99.40% |  93.02% |      35.89% |       4.09% |              100% |         6.05 GB/s |
| v4-pro-decode-s    |  99.03% |  91.04% |      47.38% |       2.20% |              100% |        26.42 GB/s |
| v4-flash-decode-m  |  54.88% |  91.04% |      49.61% |       2.28% |              100% |       831.39 GB/s |
| v4-pro-decode-m    |   6.59% |  91.04% |      43.64% |       1.07% |              100% |      1207.23 GB/s |
| v4-flash-decode-l  |   2.02% |  91.04% |      46.58% |       1.12% |              100% |      1328.60 GB/s |
| v4-pro-decode-l    |   1.44% |  91.04% |      45.84% |       1.10% |              100% |      1308.14 GB/s |
| v4-flash-prefill-s |   1.12% |  91.04% |      50.42% |       2.27% |              100% |      1362.05 GB/s |
| v4-pro-prefill-s   |   0.93% |  91.04% |      49.46% |       2.22% |              100% |      1325.39 GB/s |
| v4-flash-prefill-l |   0.06% |  91.04% |      53.22% |       2.39% |              100% |      1456.10 GB/s |
| v4-pro-prefill-l   |   0.27% |  91.04% |      48.96% |       1.16% |              100% |      1414.07 GB/s |

瓶颈判断与优化对应关系：

1. **小 workload 主要受固定开销和并行度不足影响。** decode-s 的数据高度命中 L2，但延迟仍为数微秒，理论带宽利用率只有 `21.5%–35.8%`；继续减少 launch、workgroup 和指令开销比增加计算吞吐更重要。
2. **规模增大后转为明显的访存瓶颈。** L2C hit 从 decode-s 的约 `99%` 降到大 decode/prefill 的 `0.06%–2.02%`，Profiler 带宽升到 `1.31–1.46 TB/s`，Benchmark 估算带宽达到 `1.45–1.50 TB/s`。
3. **MTE 明显高于 STE，且没有 MMA 工作。** AP MTE duty 为 `35.89%–53.22%`，STE 只有 `1.07%–4.09%`；这与逐元素 FMA、低算术强度和 load/store 主导的静态判断一致。
4. **shared memory 不是当前冲突瓶颈。** 10 组报告的 shared-memory access efficiency 均为 `100%`，当前 shared/block 仅约 `0.5–4.1 KiB`，远低于 C500 的 `64 KiB/block` 上限。
5. **all-N4 分派与瓶颈相匹配。** 合并四个 N 分支可复用 `x_layer_out`，减少重复 program/wave 和指令；`x_res` 直接读取、`x_out` 直接写回则避免无复用价值的 shared staging。

## 4. 提交自检清单

- [x] 已使用独立 PyTorch reference 验证精度
- [x] 已覆盖常规、最小、非整块 tail、任意 N、空维度和异常输入
- [x] 已覆盖 FakeTensor 与 cold `torch.compile(fullgraph=True)`
- [x] 已在真实 MetaX C500 完成定向正确性和 15 组 Benchmark
- [x] 已记录预热、测量、同步、统计方式、shape、dtype、设备和 sGPU 状态
- [x] 已提供优化后延迟、独立 PyTorch baseline、未优化 Kernel baseline、两组加速比、Roofline 公式和 mcProfiler 瓶颈分析
- [x] 已修复原 Benchmark TFLOPS 公式及 memory 字节统计错误
- [x] 已将 Benchmark 从 3 个 shape 扩展到 15 个 workload，将定向测试扩展到 38 项
- [x] 已提供一键正确性/Benchmark/Markdown/CSV 报告工具与最终 Review 矩阵工具
- [x] `git diff --check`、Manifest 校验、Benchmark tests 与 Manifest tests 已有归档通过证据
- [ ] 回填算子认领 Issue、PR A 链接和最终单一 PR HEAD 完整 SHA
- [ ] 在最终 PR HEAD 上重跑 `tests/ops/test_mhc.py -k "mhc_post"` 与 15 组 Benchmark
- [ ] 安装/提供 `pre-commit` 后执行 `pre-commit run --all-files`
- [ ] 将最终 PR 中 `MHCPostOp` Manifest 状态由当前审阅快照的 `spec-only` 更新为 `implemented`
- [ ] 在 PR 提交前检查 Issue #4 中完成最终自查