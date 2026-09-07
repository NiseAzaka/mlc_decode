#!/usr/bin/env bash
# MLA Decode kv_ctx=16384 的 mcProfiler workload 入口。
# mcProfiler 会对每个 metric 各执行一轮本脚本，因此脚本必须自带完整环境变量。
set -e

export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH
export MACA_PATH=/opt/maca
export PATH=/opt/conda/bin:/opt/conda/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/ompi/bin:/opt/maca/ucx/bin:/opt/mxdriver/bin:/opt/maca/bin
# TileLang 为源码就地编译的 MACA 版，必须显式指到源码目录
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/code:$PYTHONPATH

python3 /data/code/scripts/mla_prof.py --kv-ctx 16384 --warmup 5 --iters 20
