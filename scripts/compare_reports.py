"""对比多份 mcProfiler 报告，重点看 VL1 四个 partition 的 stall 是否均衡。

用法:
    python scripts/compare_reports.py <报告目录1> <报告目录2> [...]

例:
    python scripts/compare_reports.py \
        mcprofiler_output/output20260907050820 \
        mcprofiler_output/output20260907060055
"""

import csv
import json
import os
import sys

PARTITION_KEY = "VL1 partition stall cycles layout"
KEY_METRICS = [
    "Total Cycles",
    "Memory Access per Second",
    "AP busy Duty",
    "Compute Instructions busy Duty",
    "Global Memory Read bytes",
    "Global Memory Write bytes",
    "Total Instructions",
    "WAVES",
]


def load(d):
    with open(os.path.join(d, "report.txt.csv"), newline="") as f:
        return {r["Name"]: r["Value"] for r in csv.DictReader(f)}


def jget(vals, key, field=None):
    if key not in vals:
        return None
    try:
        data = json.loads(vals[key]).get("data", {})
    except Exception:
        return None
    return data if field is None else data.get(field)


def main():
    dirs = sys.argv[1:]
    if len(dirs) < 2:
        print(__doc__)
        return 1
    reports = [load(d) for d in dirs]
    labels = [os.path.basename(d.rstrip("/")) for d in dirs]

    pts = [jget(r, PARTITION_KEY) or {} for r in reports]
    print("=== VL1 partition stall 分布 ===")
    header = f"{'':<8}" + "".join(f"{l[-14:]:>20}" for l in labels)
    print(header)
    for k in ("pt0", "pt1", "pt2", "pt3"):
        row = f"{k:<8}"
        for p in pts:
            v = p.get(k, 0)
            tot = sum(p.values()) or 1
            row += f"{v:>13,.0f} ({100*v/tot:>4.1f}%)"
        print(row)
    row = f"{'合计':<8}"
    for p in pts:
        row += f"{sum(p.values()):>20,.0f}"
    print(row)

    print("\n均衡度（越大越差）")
    for l, p in zip(labels, pts):
        nz = [v for v in p.values() if v]
        ratio = (max(p.values()) / min(p.values())) if min(p.values()) else float("inf")
        print(f"  {l[-14:]:>20}  非零 partition: {len(nz)}/4   最大/非零最小 = {ratio:.2f}")

    print("\n=== 关键指标 ===")
    print(f"{'指标':<32}" + "".join(f"{l[-14:]:>20}" for l in labels))
    for k in KEY_METRICS:
        if k in reports[0]:
            row = f"{k:<32}"
            for r in reports:
                row += f"{r.get(k, '-').splitlines()[0][:19]:>20}"
            print(row)
    for name, field in (("RoofLine 带宽(GB/s)", "case_bandwith"),
                        ("RoofLine 计算强度", "case_I"),
                        ("RoofLine 算力(TFLOPS)", "case_flops")):
        row = f"{name:<32}"
        for r in reports:
            v = jget(r, "RoofLine", field)
            row += f"{(f'{v:.2f}' if v is not None else '-'):>20}"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
