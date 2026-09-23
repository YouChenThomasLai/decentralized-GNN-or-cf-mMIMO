#!/usr/bin/env python3
"""Plot the E12 model-based pair: convergence, consensus, and signaling cost.

Three panels, because the model-based family is judged on three things and
collapsing them into a rate bar would hide two of them.

Left: sum rate against iteration for both arms, on their own natural unit — one
centralized iteration, one distributed sweep of L activations — so the curves
are comparable per round of coordination rather than per local update.

Middle: the distributed arm's primal consensus residual against sweep, on a log
axis. A rate curve alone cannot say whether the per-AP copies ever agreed, and
the deployed phase is a projection of those copies, so this is the panel that
decides whether the reported rate means anything. The curve is the mean over
samples while the declared gate applies to the worst sample, so the gate line
crossing this curve does not mean the gate passed.

Right: rate against cumulative online signaling, which is where the two arms
actually differ. The centralized arm pays a large fixed CSI upload once; the
distributed arm pays a small ring message every activation, so its cost grows
with the iteration count and the two curves cross.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, and each arm also carries its own marker and dash pattern so the
# panels stay readable without colour.
ARM_STYLES = (
    ("centralized", "Centralized, full CSI", "#0072B2", "o", "-"),
    ("distributed", "Distributed, AP-local CSI", "#D55E00", "s", "--"),
)
INK = "#222222"
MUTED = "#6A6A6A"


def style(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.grid(True, alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        default="../artifacts/decentralized_ris/e12_model_based_optimization",
    )
    parser.add_argument("--role", default="primary")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.result_dir)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    evaluation = summary["evaluations"][args.role]
    solvers = evaluation["solvers"]
    ledger = summary["signaling"]

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    axis = axes[0]
    for arm, label, colour, marker, dash in ARM_STYLES:
        trace = np.asarray(solvers[arm]["trace"])
        steps = np.arange(1, len(trace) + 1)
        every = max(1, len(trace) // 12)
        axis.plot(
            steps, trace, color=colour, linestyle=dash, linewidth=1.8,
            marker=marker, markevery=every, markersize=5, label=label,
        )
    axis.set_xlabel("centralized iteration / distributed sweep", fontsize=10)
    axis.set_ylabel("sum rate (bps/Hz)", fontsize=10)
    axis.set_title("Convergence", fontsize=11, color=INK)
    axis.legend(frameon=False, fontsize=9, loc="lower right")
    style(axis)

    axis = axes[1]
    residual = np.asarray(solvers["distributed"]["primal_trace"])
    axis.semilogy(
        np.arange(1, len(residual) + 1), residual,
        color="#D55E00", linestyle="--", linewidth=1.8, marker="s",
        markevery=max(1, len(residual) // 12), markersize=5,
    )
    threshold = summary["decision"]["criteria"]["max_consensus_residual"]
    axis.axhline(threshold, color=MUTED, linewidth=1.0, linestyle=":")
    axis.annotate(
        f"declared gate {threshold:g}, applied to the worst sample",
        xy=(0.98, threshold), xycoords=("axes fraction", "data"),
        ha="right", va="bottom", fontsize=8, color=MUTED,
    )
    axis.set_xlabel("distributed sweep", fontsize=10)
    axis.set_ylabel(r"primal residual $\|t\|$, mean over samples", fontsize=10)
    axis.set_title("Consensus of the per-AP RIS copies", fontsize=11, color=INK)
    style(axis)

    axis = axes[2]
    # The centralized arm pays its whole bill before it starts iterating, so its
    # trajectory is a vertical segment: every rate on it costs the same.  Drawing
    # it as a guide with one endpoint marker says that, where a line of markers
    # would suggest a trade-off curve that does not exist.
    centralized = np.asarray(solvers["centralized"]["trace"])
    upfront = float(ledger["centralized"]["total_online_values"])
    axis.plot(
        [upfront, upfront], [centralized.min(), centralized.max()],
        color="#0072B2", linewidth=1.2, alpha=0.45,
    )
    axis.plot(
        [upfront], [centralized[-1]], color="#0072B2", marker="o", markersize=8,
        linestyle="none", label="Centralized, full CSI (cost paid up front)",
    )

    distributed = np.asarray(solvers["distributed"]["trace"])
    steps = np.arange(1, len(distributed) + 1)
    # One sweep is L activations, and each activation sends one ring message, so
    # the cost grows linearly with the sweep count.
    per_sweep = (
        ledger["distributed"]["ap_to_ap_values_per_activation"] * summary["config"]["AP"]
    )
    cost = steps * per_sweep + ledger["distributed"]["cpu_to_ris_values"]
    axis.plot(
        cost, distributed, color="#D55E00", linestyle="--", linewidth=1.8,
        marker="s", markevery=max(1, len(distributed) // 12), markersize=5,
        label="Distributed, AP-local CSI",
    )
    axis.set_xscale("log")
    axis.set_xlabel("cumulative online signaling (real values per sample)", fontsize=10)
    axis.set_ylabel("sum rate (bps/Hz)", fontsize=10)
    axis.set_title("Rate against counted signaling", fontsize=11, color=INK)
    axis.legend(frameon=False, fontsize=9, loc="lower right")
    style(axis)

    figure.suptitle(
        f"E12 model-based pair — {summary['samples']} paired samples, "
        f"evaluation seed {evaluation['eval_seed']}",
        fontsize=12, color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    target = Path(args.out) if args.out else root / "plots" / "model_based_joint"
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(f"{target}.{suffix}", dpi=200, bbox_inches="tight")
    print(f"[done] {target}.png and {target}.pdf")


if __name__ == "__main__":
    main()
