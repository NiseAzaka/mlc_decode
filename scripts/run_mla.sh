#!/usr/bin/env bash
# MLA Decode workload 通用入口（mcProfiler 会对每个 metric 各执行一轮本脚本）。
# 用法: bash scripts/run_mla.sh [kv_ctx] [warmup] [iters]
set -e

export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH
export MACA_PATH=/opt/maca
export PATH=/opt/conda/bin:/opt/conda/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/ompi/bin:/opt/maca/ucx/bin:/opt/mxdriver/bin:/opt/maca/bin
# TileLang 为源码就地编译的 MACA 版，必须显式指到源码目录
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/code:$PYTHONPATH

KV_CTX=${1:-16384}
WARMUP=${2:-5}
ITERS=${3:-20}

python3 /data/code/scripts/mla_prof.py --kv-ctx "$KV_CTX" --warmup "$WARMUP" --iters "$ITERS"
