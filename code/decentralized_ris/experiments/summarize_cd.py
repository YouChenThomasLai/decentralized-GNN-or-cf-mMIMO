"""Collect the discrete coordinate-descent baseline summaries into one table."""

import argparse
import glob
import json
import os

import numpy as np


def paired_stats(npz_path, a, b):
    data = np.load(npz_path)
    diff = data[a] - data[b]
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return diff.mean(), se, float((diff > 0).mean()), n


def collect(root):
    rows = []
    for summary_path in sorted(glob.glob(os.path.join(root, "**", "summary.json"), recursive=True)):
        cell = os.path.relpath(os.path.dirname(summary_path), root)
        with open(summary_path) as fh:
            s = json.load(fh)
        npz = os.path.join(os.path.dirname(summary_path), "paired_sum_rates.npz")
        row = {"cell": cell, "samples": s["samples"], "means": s["means"], "deltas": s["deltas"]}
        if os.path.exists(npz):
            for scheme in ("C", "D"):
                mean, se, win, n = paired_stats(npz, f"{scheme}_cd", f"{scheme}_cont")
                row[f"{scheme}_cd_minus_cont_se"] = se
                row[f"{scheme}_cd_win_vs_cont"] = win
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", default="../../artifacts/decentralized_ris/e03_phase_headroom/discrete_cd"
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = collect(args.root)
    header = (f"{'cell':26s} {'n':>5s} {'C_cont':>8s} {'C_round':>8s} {'C_cd':>8s} "
              f"{'C_rand':>8s} {'quant_gap':>10s} {'cd-cont':>9s} {'±SE':>7s} {'win%':>6s}")
    lines = [header, "-" * len(header)]
    for r in rows:
        m, d = r["means"], r["deltas"]
        lines.append(
            f"{r['cell']:26s} {r['samples']:5d} {m['C_cont']:8.4f} {m['C_round']:8.4f} "
            f"{m['C_cd']:8.4f} {m['C_random']:8.4f} {d['C_quantization_gap']:10.4f} "
            f"{d['C_cd_vs_continuous']:9.4f} {r.get('C_cd_minus_cont_se', float('nan')):7.4f} "
            f"{100 * r.get('C_cd_win_vs_cont', float('nan')):5.1f}%")

    lines.append("")
    header_d = (f"{'cell':26s} {'n':>5s} {'D_cont':>8s} {'D_round':>8s} {'D_cd':>8s} "
                f"{'D_rand':>8s} {'quant_gap':>10s} {'cd-cont':>9s} {'±SE':>7s} {'win%':>6s}")
    lines += [header_d, "-" * len(header_d)]
    for r in rows:
        m, d = r["means"], r["deltas"]
        lines.append(
            f"{r['cell']:26s} {r['samples']:5d} {m['D_cont']:8.4f} {m['D_round']:8.4f} "
            f"{m['D_cd']:8.4f} {m['D_random']:8.4f} {d['D_quantization_gap']:10.4f} "
            f"{d['D_cd_vs_continuous']:9.4f} {r.get('D_cd_minus_cont_se', float('nan')):7.4f} "
            f"{100 * r.get('D_cd_win_vs_cont', float('nan')):5.1f}%")

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
