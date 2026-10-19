"""汇总六档 mcProfiler 报告为一张横向对比表（Markdown + JSON）。

用法:
    python scripts/collect_profiler.py [报告根目录]

默认扫描 <repo>/mcprofiler_output/mla_decode/mla_best_*/output*/，每档取最新一份。
"""

import ast
import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(ROOT, "mcprofiler_output", "mla_decode")

KEYS = {
    "Total Instructions": "inst_total",
    "Compute Instructions": "inst_compute",
    "Memory Instructions": "inst_mem",
    "Average Wave life cycles": "wave_life",
    "WORKGROUPS": "workgroups",
    "WAVES": "waves",
    "Global Memory Read bytes": "gmem_read",
    "Global Memory Write bytes": "gmem_write",
    "VL1 Hit Rate": "vl1_hit",
    "L2C Hit Rate": "l2c_hit",
    "Dnoc Read Average Latency": "dnoc_lat",
    "Compute Instructions busy Duty": "compute_duty",
}


def parse_report(out_dir):
    rows = {}
    with open(os.path.join(out_dir, "report.txt.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows[r["Name"].strip()] = r["Value"].strip()
    data = {v: rows.get(k, "-") for k, v in KEYS.items()}

    def _num(v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            for k in ("cycles", "count", "value"):
                if k in v:
                    return _num(v[k])
        return 0.0

    for r_name, val in rows.items():
        if "stall" in r_name.lower():
            parsed = None
            if val.startswith("{"):
                try:
                    obj = json.loads(val)
                except ValueError:
                    try:
                        obj = ast.literal_eval(val)
                    except (ValueError, SyntaxError):
                        obj = None
                if isinstance(obj, dict):
                    parsed = obj.get("data", obj)
            if not isinstance(parsed, dict):
                continue
            nums = {k: _num(v) for k, v in parsed.items() if _num(v) > 0}
            total = sum(nums.values())
            if not total:
                continue
            top = sorted(nums.items(), key=lambda kv: -kv[1])[:2]
            data["stall_top"] = "; ".join(
                f"{k}={v / total * 100:.1f}%" for k, v in top)
            break
    return data


def latest_output(case_dir):
    outs = sorted(glob.glob(os.path.join(case_dir, "output*")))
    return outs[-1] if outs else None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    result = {}
    for case_dir in sorted(glob.glob(os.path.join(root, "mla_best_*"))):
        ctx = os.path.basename(case_dir).replace("mla_best_", "")
        out = latest_output(case_dir)
        if not out or not os.path.exists(os.path.join(out, "report.txt.csv")):
            print(f"[skip] {case_dir}: 无 report.txt.csv")
            continue
        result[ctx] = dict(parse_report(out), report_dir=os.path.relpath(out, ROOT))

    order = sorted(result, key=int)
    cols = [
        ("workgroups", "WORKGROUPS"),
        ("waves", "WAVES"),
        ("wave_life", "wave_life"),
        ("gmem_read", "Read B"),
        ("gmem_write", "Write B"),
        ("vl1_hit", "VL1 hit"),
        ("l2c_hit", "L2C hit"),
        ("dnoc_lat", "Dnoc lat"),
        ("compute_duty", "compute duty"),
        ("stall_top", "top stall"),
    ]
    header = "| kv_ctx | " + " | ".join(c[1] for c in cols) + " |"
    sep = "| ---: | " + " | ".join("---:" for _ in cols) + " |"
    lines = [header, sep]
    for ctx in order:
        d = result[ctx]
        lines.append("| " + ctx + " | " + " | ".join(str(d.get(k, "-")) for k, _ in cols) + " |")
    table = "\n".join(lines)

    print(table)
    out_json = os.path.join(ROOT, "mla_profiler_summary.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"\nsaved {out_json}")

    out_md = os.path.join(ROOT, "mla_profiler_summary.md")
    with open(out_md, "w") as f:
        f.write("# MLA Decode 六档 mcProfiler 汇总\n\n" + table + "\n")
    print(f"saved {out_md}")


if __name__ == "__main__":
    main()
