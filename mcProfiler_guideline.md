# mcProfiler 命令行使用说明

本文基于 `mcProfiler -h` 输出整理，适用于 沐曦服务器 GUI 不可用、只能在命令行运行 mcProfiler 的场景。

---

## 1. 基本命令结构

```bash
mcProfiler <子命令> [参数]
```

常见子命令：

| 子命令         | 作用                   |
| -------------- | ---------------------- |
| `show_metrics` | 查看当前支持的性能指标 |
| `perf_exec`    | 启动一次性能分析任务   |
| `version`      | 查看版本               |
| `help` / `-h`  | 查看帮助               |

最常用的是：

```bash
mcProfiler show_metrics
mcProfiler perf_exec ...
```

---

## 2. 推荐使用流程

### 第一步：确认程序本身能运行

先直接运行你的程序，确保没有环境、路径、动态库问题。

```bash
cd /path/to/project
python3 infer.py --batch-size 8
```

### 第二步：查看可用指标

首先进入 mcProfiler 工具所在目录：

```bash
cd /opt/mcProfiler-ubuntu18.04/
```

查看可用指标

```bash
mcProfiler show_metrics
```

后续 `--metrics` 中填写的指标，应来自这个命令的输出。

### 第三步：执行一次最小 profiling

```bash
mcProfiler perf_exec \
  --casename 1 \
  --kernelname all \
  --cwd /data/TileOPs-Metax \
  --remote_ip 127.0.0.1 \
  --remote_port 22 \
  --remote_user root \
  --remote_pwd "1234Qwerasdf" \
  --cmdline "bash ../scripts/run.sh" \
  --counts 5
```

说明：第一次建议加 `--counts 5`，只采少量 kernel，先确认流程能跑通。

### 第四步：使用常用模板执行 profiling

```bash
mcProfiler perf_exec \
  --casename casename \
  --kernelname all \
  --cwd /path/to/project \
  --remote_ip 127.0.0.1 \
  --remote_port 22 \
  --remote_user root \
  --remote_pwd [沐曦服务器密码] \
  --cmdline "bash ./run.sh"
```

注意 `run.sh` 中必须配置环境变量：

```bash
#!/bin/bash

export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH
export MACA_PATH=/opt/maca
export PATH=/opt/conda/bin:/opt/conda/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/ompi/bin:/opt/maca/ucx/bin:/opt/mxdriver/bin:/root/.opencode/bin:/root/.nvm/versions/node/v24.16.0/bin:/opt/conda/bin:/opt/conda/condabin:/opt/conda/bin:/opt/conda/condabin:/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/ompi/bin:/opt/maca/ucx/bin:/opt/mxdriver/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

python3 infer.py --batch-size 8
```

> 注意：在整个任务执行时，当选择多个度量指标（metrics）时，mcProfiler会执行多轮测量，即 bash ./run.sh 会被执行多次。

---

## 3. 核心参数说明

| 参数                     | 作用                                                         |
| ------------------------ | ------------------------------------------------------------ |
| `--cmdline`              | 要被分析的程序命令                                           |
| `--casename`             | 本次分析任务名称                                             |
| `--kernelname`           | help 中标注为 `perf_exec` 必填，通常可先填 `all`             |
| `--cwd`                  | 程序运行的工作目录                                           |
| `--metrics`              | 指定要采集的具体指标；不填则默认采集全部                     |
| `--kernelnames`          | 指定要分析的 kernel 名称列表                                 |
| `--exclude`              | 与 `--kernelnames` 配合使用，表示排除这些 kernel             |
| `--counts`               | 限制采集 kernel 数量，默认 50；`0` 表示不限制                |
| `--per-kernel`           | 对每个 kernel 分别分析，信息更全但更耗时                     |
| `--custom`               | 只采集代码中 `mcProfilerStart` / `mcProfilerStop` 包住的区域 |
| `--profile-from-start 0` | 效果等价于 `--custom`                                        |
| `--single-pass`          | 减少采样轮次，但要求 workload 足够大、足够稳定               |
| `--multi-device`         | 对所有设备进行 profiling                                     |
| `--container_id`         | 在 Docker 容器中执行任务时填写容器名或 ID                    |

注意：`--kernelname` 和 `--kernelnames` 容易混淆。一般可以把 `--kernelname all` 当作任务必填项；真正筛选目标 kernel 时使用 `--kernelnames`。

---

## 4. 常用命令模板

### 4.1 本机分析 Python 程序

```bash
mcProfiler perf_exec \
  --casename infer_bs8 \
  --kernelname all \
  --cwd /root/project \
  --cmdline "python3 infer.py --batch-size 8" \
  --counts 10
```

### 4.2 本机分析 C/C++ 程序

```bash
mcProfiler perf_exec \
  --casename cpp_main \
  --kernelname all \
  --cwd /root/demo \
  --cmdline "./main" \
  --counts 10
```

### 4.3 只分析指定 kernel

```bash
mcProfiler perf_exec \
  --casename target_kernel \
  --kernelname all \
  --cwd /root/project \
  --cmdline "python3 infer.py" \
  --kernelnames target_kernel_name \
  --counts 0
```

多个 kernel 可直接追加：

```bash
--kernelnames kernel_a kernel_b kernel_c
```

### 4.4 排除无关 kernel

```bash
mcProfiler perf_exec \
  --casename exclude_warmup \
  --kernelname all \
  --cwd /root/project \
  --cmdline "python3 infer.py" \
  --exclude \
  --kernelnames init_kernel warmup_kernel
```

含义：不分析 `init_kernel` 和 `warmup_kernel`，分析其他 kernel。

### 4.5 指定部分指标

先查看指标名：

```bash
mcProfiler show_metrics
```

再执行：

```bash
mcProfiler perf_exec \
  --casename selected_metrics \
  --kernelname all \
  --cwd /root/project \
  --cmdline "python3 infer.py" \
  --metrics metric_a metric_b metric_c
```

### 4.6 只采自定义代码区间

前提：程序中已经调用 `mcProfilerStart()` 和 `mcProfilerStop()`。

```bash
mcProfiler perf_exec \
  --casename custom_region \
  --kernelname all \
  --cwd /root/project \
  --cmdline "./main" \
  --custom
```

也可以使用：

```bash
--profile-from-start 0
```

### 4.7 在 Docker 容器中执行

```bash
mcProfiler perf_exec \
  --casename docker_case \
  --kernelname all \
  --cwd /workspace/project \
  --cmdline "python3 infer.py" \
  --container_id container_name_or_id \
  --counts 10
```

注意：`--cwd` 应填写容器内部路径。

---

## 5. 环境变量写法

CLI 参数中没有单独的环境变量参数。复杂环境建议写成脚本。

例如创建 `run.sh`：

```bash
#!/usr/bin/env bash
set -e

export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH
export MACA_PATH=/opt/maca

python3 infer.py --batch-size 8
```

加执行权限：

```bash
chmod +x run.sh
```

然后执行：

```bash
mcProfiler perf_exec \
  --casename script_case \
  --kernelname all \
  --cwd /root/project \
  --cmdline "bash ./run.sh" \
  --counts 10
```

---

## 6. 状态与报告说明

help 中出现了以下状态：

| 状态              | 含义                          |
| ----------------- | ----------------------------- |
| `init`            | 任务初始化                    |
| `init_failed`     | 初始化失败                    |
| `executing`       | 正在运行目标程序              |
| `execute_failed`  | 目标程序运行失败              |
| `profiling`       | 正在采集或分析 profiling 数据 |
| `profiler_failed` | profiling 失败                |
| `done`            | 任务完成                      |

help 中还提到 `perf_progress`、`perf_report`、`perf_wait`、`list_perf`，但当前子命令列表只显示 `show_metrics`、`perf_exec`、`version`、`help`。如需查询进度或导出报告，先执行：

```bash
mcProfiler help
profiler_client help
```

确认当前版本实际支持的命令格式。

---

## 7. 常见问题排查

### 7.1 报缺少参数

优先检查是否填写：

```bash
--cmdline
--casename
--kernelname
```

### 7.2 找不到程序或文件

检查 `--cwd` 是否正确。推荐使用绝对路径。

### 7.3 找不到动态库

把 `LD_LIBRARY_PATH`、`MACA_PATH` 等环境变量写进 `run.sh`，再用 `--cmdline "bash ./run.sh"` 执行。

### 7.4 跑得太慢

常见原因：默认采集全部 metrics、kernel 太多、使用了 `--per-kernel`、`--counts 0`。建议先用：

```bash
--counts 5
```

跑通后再扩大采集范围。

---

