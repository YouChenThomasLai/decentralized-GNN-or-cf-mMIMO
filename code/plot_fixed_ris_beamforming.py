#!/usr/bin/env python3
"""Plot the E13 fixed-G2-RIS active-beamforming comparison.

Left panel: the absolute paper-decentralized sum rate of every active design
under the same frozen G2 RIS phase, at continuous and rounded 2-bit phase.  Its
error bars are dominated by between-cluster channel variability that all arms
share, so the right panel shows the decision-relevant quantity instead: the
*paired* difference against the native G2 beamformer on the same batch clusters,
where that common variability cancels.

`_full` arms spend the whole per-AP budget; `_matched` arms reuse G2's learned
per-AP power fractions, so the pair isolates the beamforming direction from the
power control.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito for the active design; the phase setting carries its own hatch and
# marker as well as its alpha, so neither distinction rests on hue alone.
ARM_STYLES = (
    ("g2", "G2 readout", "#0072B2"),
    ("g2_full", "G2 direction, full power", "#56B4E9"),
    ("mrt_full", "Local MRT, full power", "#D55E00"),
    ("mrt_matched", "Local MRT, G2 power", "#E69F00"),
    ("rzf_full", "Local RZF, full power", "#009E73"),
    ("rzf_matched", "Local RZF, G2 power", "#CC79A7"),
)
PHASE_STYLES = (
    ("continuous", "Continuous phase", 1.0, "", "o"),
    ("2bit", "Rounded 2-bit phase", 0.55, "//", "s"),
)
INK = "#222222"
MUTED = "#6A6A6A"


def clustered_interval(values, batch_size):
    clusters = values.reshape(-1, batch_size).mean(axis=1)
    return clusters.mean(), 1.96 * clusters.std(ddof=1) / np.sqrt(len(clusters))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        default="../artifacts/decentralized_ris/e13_fixed_ris_beamforming",
    )
    parser.add_argument("--role", default="primary")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.result_dir)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    raw = np.load(root / "paired_rates.npz")
    batch_size = summary["config"]["batch_size"]

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    width = 0.38
    offsets = {"continuous": -width / 2, "2bit": width / 2}
    positions = np.arange(len(ARM_STYLES))

    for phase, phase_label, alpha, hatch, _ in PHASE_STYLES:
        heights, errors = [], []
        for arm, _, _ in ARM_STYLES:
            values = raw[f"{args.role}__{arm}_{phase}"]
            mean, half = clustered_interval(values, batch_size)
            heights.append(mean)
            errors.append(half)
        axes[0].bar(
            positions + offsets[phase], heights, width, yerr=errors, capsize=3,
            color=[style[2] for style in ARM_STYLES], alpha=alpha, hatch=hatch,
            edgecolor=INK, linewidth=0.6, label=phase_label,
        )

    axes[0].set_ylabel("Paper-decentralized sum rate (bps/Hz)")
    axes[0].set_title("Same frozen G2 RIS phase, different active design", color=INK)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([style[1] for style in ARM_STYLES], rotation=25, ha="right")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, alpha=alpha, hatch=hatch,
                      edgecolor=INK, linewidth=0.6)
        for _, _, alpha, hatch, _ in PHASE_STYLES
    ]
    axes[0].legend(
        handles, [style[1] for style in PHASE_STYLES],
        frameon=False, loc="lower left", fontsize=9,
    )
    axes[0].grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.6)
    axes[0].set_axisbelow(True)

    conventional = [style for style in ARM_STYLES if style[0] != "g2"]
    for index, (arm, _, color) in enumerate(conventional):
        for phase, _, _, _, marker in PHASE_STYLES:
            difference = (
                raw[f"{args.role}__{arm}_{phase}"] - raw[f"{args.role}__g2_{phase}"]
            )
            mean, half = clustered_interval(difference, batch_size)
            axes[1].errorbar(
                index + (-0.12 if phase == "continuous" else 0.12), mean, yerr=half,
                marker=marker, color=color, capsize=3, markersize=7, linewidth=1.4,
                markeredgecolor=INK, markeredgewidth=0.5,
            )

    axes[1].axhline(0.0, color=INK, linewidth=1.0)
    axes[1].set_ylabel("Paired difference from the G2 readout (bps/Hz)", labelpad=8)
    axes[1].set_title("Paired on the same channel clusters", color=INK)
    axes[1].set_xticks(np.arange(len(conventional)))
    axes[1].set_xticklabels([style[1] for style in conventional], rotation=25, ha="right")
    axes[1].grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.6)
    axes[1].set_axisbelow(True)
    for _, phase_label, _, _, marker in PHASE_STYLES:
        axes[1].plot([], [], marker, color=MUTED, linestyle="none", label=phase_label)
    axes[1].legend(frameon=False, loc="lower left", fontsize=9)
    axes[1].margins(x=0.12, y=0.18)

    seed = summary["evaluations"][args.role]["eval_seed"]
    figure.suptitle(
        f"E13 fixed-G2-RIS active-beamforming controls "
        f"({summary['samples']} paired samples, evaluation seed {seed})",
        color=INK, fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.subplots_adjust(wspace=0.32)

    out = Path(args.out) if args.out else root / "plots" / "fixed_ris_beamforming"
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=200)
        print(f"[write] {out.with_suffix(suffix)}")


if __name__ == "__main__":
    main()
