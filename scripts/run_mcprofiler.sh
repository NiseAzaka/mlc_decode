#!/usr/bin/env bash
# 对 MLA Decode 指定 kv_ctx 执行一次 mcProfiler 采集。
#
# 用法: bash scripts/run_mcprofiler.sh [kv_ctx] [counts] [casename]
#   kv_ctx   默认 16384
#   counts   默认 5（guideline 建议先小量跑通，再放大；0 表示不限制）
#   casename 默认 mla_<kv_ctx>
#
# 报告输出目录会在结束时打印：/opt/mcProfiler-ubuntu18.04/output<时间戳>
#
# 注意：运行期间会刷大量 [warning] get_srvrpc_port ...，属正常现象；
#       只要日志里周期性出现 `[mla_prof] done` 即说明目标程序在正常 replay，
#       若长时间没有该输出则直接 kill 本次采集。
set -e

KV_CTX=${1:-16384}
COUNTS=${2:-5}
CASENAME=${3:-mla_${KV_CTX}}
REMOTE_PWD="1234Qwerasdf"
LOG=${LOG:-/tmp/mcprof_${CASENAME}.log}

cd /opt/mcProfiler-ubuntu18.04

echo "[mcprof] casename=${CASENAME} kv_ctx=${KV_CTX} counts=${COUNTS} log=${LOG}"

mcProfiler perf_exec \
  --casename "${CASENAME}" \
  --kernelname all \
  --cwd /data/code \
  --remote_ip 127.0.0.1 \
  --remote_port 22 \
  --remote_user root \
  --remote_pwd "${REMOTE_PWD}" \
  --cmdline "bash scripts/run_mla.sh ${KV_CTX}" \
  --counts "${COUNTS}" 2>&1 | tee "${LOG}"

echo "[mcprof] finished, log=${LOG}"

# 把原始报告归档到仓库目录，避免留在 /opt/mcProfiler-ubuntu18.04 下被清理
ARCHIVE_DIR=${ARCHIVE_DIR:-/data/code/mcprofiler_output}
REPORT_DIR=$(sed -n 's/.*please check report file \(\/[^ ]*\).*/\1/p' "${LOG}" | tail -1)
if [ -n "${REPORT_DIR}" ] && [ -d "${REPORT_DIR}" ]; then
  mkdir -p "${ARCHIVE_DIR}"
  cp -r "${REPORT_DIR}" "${ARCHIVE_DIR}/"
  echo "[mcprof] 报告已归档: ${ARCHIVE_DIR}/$(basename "${REPORT_DIR}")"
else
  echo "[mcprof] 未能从日志中定位报告目录，请检查 ${LOG}"
fi
