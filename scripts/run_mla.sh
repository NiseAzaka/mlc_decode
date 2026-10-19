#!/usr/bin/env bash
# MLA Decode workload 通用入口（mcProfiler 会对每个 metric 各执行一轮本脚本）。
# 用法: bash scripts/run_mla.sh [kv_ctx] [warmup] [iters]
#
# 仓库根不再硬编码：由本脚本所在目录的上一级推导（即含 mla_decode.py 的目录），
# 因此仓库可放在任意路径下运行。
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH
export MACA_PATH=/opt/maca
export PATH=/opt/conda/bin:/opt/conda/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/ompi/bin:/opt/maca/ucx/bin:/opt/mxdriver/bin:/opt/maca/bin
# TileLang 为源码就地编译的 MACA 版，必须显式指到源码目录
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:${ROOT_DIR}:$PYTHONPATH

KV_CTX=${1:-16384}
WARMUP=${2:-5}
ITERS=${3:-20}
# 第 4 个位置参数或 IMPL 环境变量指定实现：baseline=baseline_mla.py，默认 ours
IMPL=${4:-${IMPL:-ours}}

python3 "${SCRIPT_DIR}/mla_prof.py" --kv-ctx "$KV_CTX" --warmup "$WARMUP" --iters "$ITERS" --impl "$IMPL"
