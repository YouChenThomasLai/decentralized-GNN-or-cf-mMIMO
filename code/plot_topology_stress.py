"""E14 topology panel from the paired comparison JSON."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.comparison.read_text())
    topologies = ("T0", "T1", "T2")
    series = (
        ("G2", "#0072B2", "o", lambda r: r["learned"]["g2"]["decentralized"]),
        ("R0", "#009E73", "s", lambda r: r["learned"]["r0"]["decentralized"]),
        ("Centralized fixed budget", "#D55E00", "^",
         lambda r: r["solver"]["centralized"]["continuous"]),
        ("Distributed fixed budget", "#CC79A7", "D",
         lambda r: r["solver"]["distributed"]["continuous"]),
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.1))
    for label, color, marker, select in series:
        cells = [select(result[t]) for t in topologies]
        axis.errorbar(range(3), [cell["mean"] for cell in cells],
                      yerr=[1.96 * cell.get("clustered_se", cell.get("clustered_sem", 0))
                            for cell in cells],
                      label=label, color=color, marker=marker, markersize=6,
                      linewidth=1.8, capsize=3)
    axis.set_xticks(range(3), topologies)
    axis.set_ylabel("Paper-decentralized or deployed sum rate (bps/Hz)")
    axis.set_xlabel("Fixed AP/RIS layout")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=220)
    figure.savefig(args.out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
