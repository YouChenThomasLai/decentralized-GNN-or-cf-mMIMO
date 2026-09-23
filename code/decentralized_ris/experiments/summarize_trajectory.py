"""Collect the per-checkpoint visibility-mode evaluations into one trajectory.

`scripts/e02_mode_trajectory.sh` writes one `experiments.local_csi` output directory
per checkpoint, named `iter<iteration>`. This script reads those directories,
recovers the training iteration from the name, and reports every mode mean and
every paired mode gap as a function of training steps. Statistics use the batch
as the independent unit, because `ChannelSimulator` draws one set of UE
locations per batch.
"""

import argparse
import glob
import json
import os
import re

import numpy as np


MODES = ("cen", "dec_paper", "dec_own_only")
GAPS = (
    ("cen", "dec_paper"),
    ("cen", "dec_own_only"),
    ("dec_paper", "dec_own_only"),
)


def clustered_stats(values, batch_size):
    """Mean and standard error with the batch as the independent unit."""
    groups = values.reshape(-1, batch_size).mean(axis=1)
    return (
        float(values.mean()),
        float(groups.std(ddof=1) / np.sqrt(len(groups))),
        groups,
    )


def load_cell(summary_path):
    cell_dir = os.path.dirname(summary_path)
    match = re.search(r"iter0*(\d+)", os.path.basename(cell_dir))
    if not match:
        return None
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    paired = np.load(os.path.join(cell_dir, "paired_sum_rates.npz"))
    batch_size = int(summary.get("batch_size", 8))

    row = {
        "iteration": int(match.group(1)),
        "cell": os.path.basename(cell_dir),
        "checkpoint": summary["checkpoint"],
        "samples": summary["samples"],
        "batch_size": batch_size,
        "visible_nodes": summary.get("visible_nodes", {}),
        "verification": summary.get("verification"),
        "means": {},
        "gaps": {},
    }
    schemes = [name for name in ("cont", "round", "greedy") if f"cen_{name}" in paired]
    row["schemes"] = schemes
    row["_paired"] = {key: paired[key] for key in paired.files}
    for scheme in schemes:
        for mode in MODES:
            mean, se, _ = clustered_stats(paired[f"{mode}_{scheme}"], batch_size)
            row["means"][f"{mode}_{scheme}"] = {"mean": mean, "clustered_se": se}
        for left, right in GAPS:
            diff = paired[f"{left}_{scheme}"] - paired[f"{right}_{scheme}"]
            mean, se, groups = clustered_stats(diff, batch_size)
            row["gaps"][f"{left}_minus_{right}_{scheme}"] = {
                "mean": mean,
                "clustered_se": se,
                "t": mean / se if se else float("nan"),
                "batch_win_fraction": float((groups > 0).mean()),
            }
    return row


def collect(root):
    rows = [
        row
        for path in sorted(glob.glob(os.path.join(root, "*", "summary.json")))
        if (row := load_cell(path))
    ]
    if not rows:
        raise SystemExit(f"no iter*/summary.json cells under {root}")
    return sorted(rows, key=lambda row: row["iteration"])


def render(rows, scheme):
    header = (
        f"{'iteration':>10s} {'n':>5s} {'centralized':>12s} {'decentralized':>14s} "
        f"{'own-only':>10s} {'cen-dec':>9s} {'±SE':>6s} {'dec-own':>9s} {'±SE':>6s} "
        f"{'cen-own':>9s} {'±SE':>6s}"
    )
    lines = [f"[{scheme} phase]", header, "-" * len(header)]
    for row in rows:
        if scheme not in row["schemes"]:
            continue
        mean = row["means"]
        gap = row["gaps"]
        lines.append(
            f"{row['iteration']:10d} {row['samples']:5d} "
            f"{mean[f'cen_{scheme}']['mean']:12.4f} "
            f"{mean[f'dec_paper_{scheme}']['mean']:14.4f} "
            f"{mean[f'dec_own_only_{scheme}']['mean']:10.4f} "
            f"{gap[f'cen_minus_dec_paper_{scheme}']['mean']:9.4f} "
            f"{gap[f'cen_minus_dec_paper_{scheme}']['clustered_se']:6.4f} "
            f"{gap[f'dec_paper_minus_dec_own_only_{scheme}']['mean']:9.4f} "
            f"{gap[f'dec_paper_minus_dec_own_only_{scheme}']['clustered_se']:6.4f} "
            f"{gap[f'cen_minus_dec_own_only_{scheme}']['mean']:9.4f} "
            f"{gap[f'cen_minus_dec_own_only_{scheme}']['clustered_se']:6.4f}"
        )
    return lines


def same_evaluation_data(rows):
    """True when every cell drew the same channels, so cells can be differenced.

    `local_csi` reports the mean number of AP-UE nodes each AP sees, which is a
    fingerprint of the association masks and therefore of the sampled topology.
    """
    fingerprints = {json.dumps(row["visible_nodes"], sort_keys=True) for row in rows}
    return len(fingerprints) == 1


def render_windows(rows, scheme):
    """Paired change between consecutive checkpoints.

    Each checkpoint is scored on the same channel realizations, so differencing
    two cells removes the channel variance that dominates the absolute means and
    answers whether a segment of the trajectory still moves.
    """
    if not same_evaluation_data(rows):
        return ["[windows] cells used different evaluation data; not differenced"]

    header = f"{'window':>16s} " + " ".join(f"{mode:>22s}" for mode in MODES)
    lines = [f"[{scheme} phase, paired change between checkpoints]", header,
             "-" * len(header)]
    for earlier, later in zip(rows, rows[1:]):
        if scheme not in earlier["schemes"] or scheme not in later["schemes"]:
            continue
        cells = []
        for mode in MODES:
            key = f"{mode}_{scheme}"
            mean, se, _ = clustered_stats(
                later["_paired"][key] - earlier["_paired"][key], later["batch_size"])
            cells.append(f"{mean:+7.3f}±{se:5.3f} t={mean / se:6.2f}" if se
                         else f"{mean:+7.3f}")
        window = f"{earlier['iteration'] // 1000}k->{later['iteration'] // 1000}k"
        lines.append(f"{window:>16s} " + " ".join(f"{cell:>22s}" for cell in cells))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="../../artifacts/decentralized_ris/e02_input_modes")
    parser.add_argument("--out", default=None, help="text table path")
    parser.add_argument("--json", default=None, help="machine-readable trajectory path")
    args = parser.parse_args()

    rows = collect(args.root)
    schemes = sorted({scheme for row in rows for scheme in row["schemes"]},
                     key=("cont", "round", "greedy").index)
    lines = []
    for scheme in schemes:
        lines += render(rows, scheme) + [""]
    lines += render_windows(rows, "cont") + [""]
    lines.append(
        "visible AP-UE nodes per AP: "
        + " ".join(
            f"{mode}={rows[-1]['visible_nodes'].get(mode, float('nan')):.3f}"
            for mode in ("dec_paper", "dec_own_only")
        )
    )
    text = "\n".join(lines)
    print(text)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"[done] {args.out}")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        payload = [{k: v for k, v in row.items() if k != "_paired"} for row in rows]
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"root": args.root, "points": payload}, handle, indent=2)
        print(f"[done] {args.json}")


if __name__ == "__main__":
    main()
