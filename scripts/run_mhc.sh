#!/usr/bin/env bash
# mhc_post workload 通用入口（mcProfiler 会对每个 metric 各执行一轮本脚本）。
# 用法: bash scripts/run_mhc.sh [batch] [n_expand] [c_x]
#
# mhc_post 是独立的 TileOPs 仓库，用 MHC_ROOT 指定其路径（默认 /data/mhc_post）。
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH
export MACA_PATH=/opt/maca
export PATH=/opt/conda/bin:/opt/conda/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/ompi/bin:/opt/maca/ucx/bin:/opt/mxdriver/bin:/opt/maca/bin

export MHC_ROOT=${MHC_ROOT:-/data/mhc_post}
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:${MHC_ROOT}:$PYTHONPATH

BATCH=${1:-8192}
N_EXPAND=${2:-4}
C_X=${3:-4096}

python3 "${SCRIPT_DIR}/mhc_prof.py" \
  --batch "${BATCH}" --n-expand "${N_EXPAND}" --c-x "${C_X}" \
  --warmup 5 --iters 20 --tune
