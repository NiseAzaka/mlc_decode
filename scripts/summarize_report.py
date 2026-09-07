"""汇总 mcProfiler 报告目录中的关键指标。

用法:
    python scripts/summarize_report.py /opt/mcProfiler-ubuntu18.04/output20260907050820

从 report.txt.csv（Name,Description,Value）中挑出对 kernel 调优最关键的几项，
并解析 RoofLine / ISU stall / 指令构成中的 JSON 数据。
"""

import ast
import csv
import json
import os
import sys

KEY_METRICS = [
    "Total Instructions",
    "Compute Instructions",
    "Memory Instructions",
    "Total Cycles",
    "Memory Access per Second",
    "Average AP busy cycles",
    "AP busy Duty",
    "Compute Instructions busy Duty",
    "WORKGROUPS",
    "WAVES",
    "Average Wave life cycles",
    "Global Memory Read bytes",
    "Global Memory Write bytes",
    "VL1 Hit Rate",
    "L2C Hit Rate",
    "Constant SL1 Hit Rate",
    "Dnoc Read Average Latency",
]


def load_rows(report_csv):
    with open(report_csv, newline="") as f:
        return list(csv.DictReader(f))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    out_dir = sys.argv[1]
    report_csv = os.path.join(out_dir, "report.txt.csv")
    if not os.path.exists(report_csv):
        print(f"找不到 {report_csv}")
        return 1

    rows = load_rows(report_csv)
    values = {r["Name"]: r["Value"] for r in rows}

    print(f"报告目录: {out_dir}\n")
    print("=== 关键指标 ===")
    for k in KEY_METRICS:
        if k in values:
            print(f"  {k:<32} {values[k].splitlines()[0]}")

    # RoofLine
    if "RoofLine" in values:
        try:
            rl = json.loads(values["RoofLine"])["data"]
            print("\n=== RoofLine ===")
            print(f"  block_num        {rl.get('block_num')}")
            print(f"  计算强度 case_I  {rl.get('case_I'):.2f}  (峰值拐点 MAX_I={rl.get('MAX_I')})")
            print(f"  带宽 case_bw     {rl.get('case_bandwith'):.2f} GB/s  (峰值 {rl.get('MAX_Bandwith')} GB/s)")
            print(f"  算力 case_flops  {rl.get('case_flops'):.2f} TFLOPS")
            print(f"  实测时长 during  {rl.get('during')} (ns 或 cycles，取决于版本)")
        except Exception as e:
            print(f"\nRoofLine 解析失败: {e}")

    # 指令构成 / stall
    for name in ("Instructions Comparison", "ISU stall cycles layout",
                 "Compute Instructions Cycles Rate"):
        if name in values:
            try:
                data = json.loads(values[name])["data"]
            except Exception:
                try:
                    data = ast.literal_eval(values[name])["data"]
                except Exception:
                    continue
            print(f"\n=== {name} ===")
            total = sum(v for v in data.values() if isinstance(v, (int, float))) or 1
            for k, v in sorted(data.items(), key=lambda kv: -kv[1]):
                pct = 100.0 * v / total
                print(f"  {k:<24} {v:>16,.0f}  {pct:>6.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
