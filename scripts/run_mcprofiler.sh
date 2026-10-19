#!/usr/bin/env bash
# 对 MLA Decode 指定 kv_ctx 执行一次 mcProfiler 采集。
#
# 用法: bash scripts/run_mcprofiler.sh [kv_ctx] [counts] [casename]
#   kv_ctx   默认 16384
#   counts   默认 5（guideline 建议先小量跑通，再放大；0 表示不限制）
#   casename 默认 mla_<kv_ctx>
#
# 运行模式:
#   默认 **本地模式**（不传 --remote_*），mcProfiler 直接在本机 fork 出 workload，
#   不需要 SSH。本容器 root 的 SSH 密码未知，远端模式会一直卡在 init 状态。
#   需要远端模式时设 USE_REMOTE=1，并用 REMOTE_IP/REMOTE_PORT/REMOTE_USER/REMOTE_PWD 覆盖。
#
# 其它可覆盖的环境变量:
#   WORKLOAD_SCRIPT   scripts/ 下的 workload 入口，默认 run_mla.sh
#   CMD_ARGS          传给 workload 脚本的参数，默认 <kv_ctx>
#   METRICS           只采集指定指标（空格分隔），默认全部
#   EXTRA_ARGS        追加给 mcProfiler 的额外参数
#   OP_NAME           归档子目录名，默认 mla_decode
#   ARCHIVE_DIR       归档根目录，默认 <仓库根>/mcprofiler_output
#   LOG               采集日志路径，默认 /tmp/mcprof_<casename>.log
#
# 报告输出目录会在结束时打印：/opt/mcProfiler-ubuntu18.04/output<时间戳>
#
# 注意：运行期间会刷大量 [get_srvrpc_port] / MXKW / topology 日志，属正常现象；
#       只要日志里周期性出现 `[mla_prof] done` 即说明目标程序在正常 replay，
#       若长时间没有该输出则直接 kill 本次采集。
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

KV_CTX=${1:-16384}
COUNTS=${2:-5}
CASENAME=${3:-mla_${KV_CTX}}
LOG=${LOG:-/tmp/mcprof_${CASENAME}.log}
WORKLOAD_SCRIPT=${WORKLOAD_SCRIPT:-run_mla.sh}
CMD_ARGS=${CMD_ARGS:-${KV_CTX}}

REMOTE_ARGS=()
if [ "${USE_REMOTE:-0}" = "1" ]; then
  REMOTE_ARGS=(
    --remote_ip "${REMOTE_IP:-127.0.0.1}"
    --remote_port "${REMOTE_PORT:-22}"
    --remote_user "${REMOTE_USER:-root}"
    --remote_pwd "${REMOTE_PWD:?USE_REMOTE=1 时必须设置 REMOTE_PWD}"
  )
fi

METRIC_ARGS=()
if [ -n "${METRICS:-}" ]; then
  # shellcheck disable=SC2206
  METRIC_ARGS=(--metrics ${METRICS})
fi

# mcProfiler 从当前目录推断自身安装根（config/.version），必须在安装目录下运行。
cd /opt/mcProfiler-ubuntu18.04

echo "[mcprof] casename=${CASENAME} kv_ctx=${KV_CTX} counts=${COUNTS} remote=${USE_REMOTE:-0} log=${LOG}"

mcProfiler perf_exec \
  --casename "${CASENAME}" \
  --kernelname all \
  --cwd "${ROOT_DIR}" \
  "${REMOTE_ARGS[@]}" \
  "${METRIC_ARGS[@]}" \
  ${EXTRA_ARGS:-} \
  --cmdline "bash ${SCRIPT_DIR}/${WORKLOAD_SCRIPT} ${CMD_ARGS}" \
  --counts "${COUNTS}" 2>&1 | tee "${LOG}"

echo "[mcprof] finished, log=${LOG}"

# 把原始报告归档到仓库目录，避免留在 /opt/mcProfiler-ubuntu18.04 下被清理。
# 归档结构统一为 <算子名>/<shape>：
#   mla_decode/16384、mhc_post/8192_4096 ...
ARCHIVE_DIR=${ARCHIVE_DIR:-${ROOT_DIR}/mcprofiler_output}
OP_NAME=${OP_NAME:-mla_decode}
REPORT_DIR=$(sed -n 's/.*please check report file \(\/[^ ]*\).*/\1/p' "${LOG}" | tail -1)
if [ -n "${REPORT_DIR}" ] && [ -d "${REPORT_DIR}" ]; then
  DEST="${ARCHIVE_DIR}/${OP_NAME}/${CASENAME}"
  mkdir -p "${DEST}"
  cp -r "${REPORT_DIR}" "${DEST}/"
  echo "[mcprof] 报告已归档: ${DEST}/$(basename "${REPORT_DIR}")"
else
  echo "[mcprof] 未能从日志中定位报告目录，请检查 ${LOG}"
fi
