#!/usr/bin/env python3
import os
import re
import sys
import argparse
import pandas as pd

def summarize_results(root_dir: str, out_path: str = None, verbose: bool = False):
    """
    Scans all subfolders in `root_dir` that look like M4_N30_L4_K1_P15.0,
    detects which variable changes (e.g., P), extracts metrics from
    run0/final_eval/final_eval_run0.txt, and writes summary_<var>.xlsx.
    """

    if not os.path.exists(root_dir):
        print(f"[ERROR] Folder '{root_dir}' does not exist.")
        sys.exit(1)

    pattern = re.compile(r"([A-Za-z]+)([-+]?\d*\.?\d+)")
    data = {}
    all_folder_vars = []

    # 1️⃣ Parse all subfolders
    for subdir in os.listdir(root_dir):
        full_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(full_path):
            continue

        pairs = dict(pattern.findall(subdir))
        if not pairs:
            continue
        all_folder_vars.append(pairs)

        eval_path = os.path.join(full_path, "run0", "final_eval", "final_eval_run0.txt")
        if not os.path.exists(eval_path):
            if verbose:
                print(f"[WARN] Missing {eval_path}")
            continue

        with open(eval_path, "r") as f:
            lines = f.read().strip().splitlines()

        metrics = {}
        for line in lines:
            if ":" in line:
                k, v = line.split(":")
                metrics[k.strip()] = float(v.strip())
        data[subdir] = metrics

    if not all_folder_vars:
        print("[ERROR] No valid subfolders found.")
        sys.exit(1)

    # 2️⃣ Detect varying variable
    keys = list(all_folder_vars[0].keys())
    varying_key = None
    for k in keys:
        vals = {float(f[k]) for f in all_folder_vars if k in f}
        if len(vals) > 1:
            varying_key = k
            break

    if varying_key is None:
        print("[ERROR] Could not determine varying variable.")
        sys.exit(1)

    print(f"[INFO] Detected varying variable: {varying_key}")

    # 3️⃣ Build data table
    val_to_metrics = {}
    for folder, metrics in data.items():
        pairs = dict(pattern.findall(folder))
        if varying_key in pairs:
            val_to_metrics[float(pairs[varying_key])] = metrics

    all_vals = sorted(val_to_metrics.keys())
    metric_names = sorted({m for d in val_to_metrics.values() for m in d.keys()})
    table = []
    for metric in metric_names:
        row = [metric]
        for v in all_vals:
            row.append(val_to_metrics[v].get(metric, None))
        table.append(row)

    columns = [""] + [f"{v}" for v in all_vals]
    df = pd.DataFrame(table, columns=columns)

    # 4️⃣ Output
    if out_path is None:
        out_path = os.path.join(root_dir, f"summary_{varying_key}.xlsx")

    df.to_excel(out_path, index=False)
    print(f"[INFO] Summary saved to {out_path}")

# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize results across parameter sweeps into an Excel file."
    )
    parser.add_argument(
        "--root", "-r", required=True,
        help="Path to the root results folder (e.g. 'results')"
    )
    parser.add_argument(
        "--out", "-o", default=None,
        help="Optional output Excel file path (default: summary_<var>.xlsx in root)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print additional debug info"
    )

    args = parser.parse_args()
    summarize_results(args.root, args.out, args.verbose)
