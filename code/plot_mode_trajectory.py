#!/usr/bin/env python3
"""Plot sum rate versus training steps for the three CSI visibility modes."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODE_STYLES = (
    ("cen", "Centralized", "#D55E00", "o"),
    ("dec_paper", "Decentralized (paper eq. 10)", "#0072B2", "s"),
    ("dec_own_only", "Own-only", "#009E73", "^"),
)
GAP_STYLES = (
    ("cen_minus_dec_paper", "Centralized − decentralized", "#CC79A7", "o"),
    ("dec_paper_minus_dec_own_only", "Decentralized − own-only", "#E69F00", "s"),
    ("cen_minus_dec_own_only", "Centralized − own-only", "#6F4E45", "^"),
)
SCHEME_LINESTYLE = {"cont": "-", "round": "--"}


def apply_style():
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def series(points, table, key):
    x, y = [], []
    for point in points:
        entry = point[table].get(key)
        if entry is None:
            continue
        x.append(point["iteration"])
        y.append(entry["mean"])
    return x, y


# A log axis cannot label every point of an uneven trajectory without the late
# ticks colliding, so label a readable subset and let the markers show the rest.
LOG_TICKS = (2000, 10000, 40000, 100000, 200000, 500000)


def decorate(axis, x_values, xscale):
    axis.set_xlabel("Training steps")
    if xscale == "log":
        axis.set_xscale("log")
        ticks = [tick for tick in LOG_TICKS if min(x_values) <= tick <= max(x_values)]
    else:
        step = 100000
        ticks = list(range(0, max(x_values) + step, step))
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{tick // 1000:d}k" for tick in ticks], fontsize=10)
    axis.minorticks_off()
    axis.grid(axis="y", color="0.88", linewidth=0.8)


def plot(trajectory_path, output_base, schemes, xscale):
    with open(trajectory_path, encoding="utf-8") as handle:
        points = json.load(handle)["points"]
    iterations = [point["iteration"] for point in points]

    apply_style()
    figure, (rate_axis, gap_axis) = plt.subplots(
        1, 2, figsize=(13.0, 4.8), constrained_layout=True
    )

    for scheme in schemes:
        for key, label, color, marker in MODE_STYLES:
            x, y = series(points, "means", f"{key}_{scheme}")
            if not x:
                continue
            rate_axis.plot(
                x, y, color=color, marker=marker,
                linestyle=SCHEME_LINESTYLE.get(scheme, "-"),
                linewidth=2.2, markersize=5.5,
                label=label if scheme == schemes[0] else "_nolegend_",
            )
    rate_axis.set_ylabel("Sum rate (bps/Hz)")
    decorate(rate_axis, iterations, xscale)
    mode_legend = rate_axis.legend(loc="upper left", fontsize=9, frameon=False)
    rate_axis.add_artist(mode_legend)
    rate_axis.legend(
        handles=[
            Line2D([], [], color="0.25", linewidth=2.2, linestyle="-",
                   label="Continuous phase"),
            Line2D([], [], color="0.25", linewidth=2.2, linestyle="--",
                   label="2-bit phase"),
        ],
        loc="lower right", fontsize=9, frameon=False,
    )

    for key, label, color, marker in GAP_STYLES:
        x, y = series(points, "gaps", f"{key}_cont")
        if not x:
            continue
        gap_axis.plot(
            x, y, color=color, marker=marker, linestyle="-",
            linewidth=2.2, markersize=5.5, label=label,
        )
    gap_axis.axhline(0.0, color="0.35", linewidth=0.8)
    gap_axis.set_ylabel("Paired gap (bps/Hz)")
    decorate(gap_axis, iterations, xscale)
    gap_axis.legend(loc="upper left", fontsize=9, frameon=False)

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = output_base.with_suffix(suffix)
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        default="../artifacts/decentralized_ris/e02_input_modes/trajectory.json",
    )
    parser.add_argument(
        "--output-base",
        default="../artifacts/decentralized_ris/e02_input_modes/mode_trajectory",
    )
    parser.add_argument("--schemes", nargs="+", default=["cont", "round"],
                        choices=["cont", "round"])
    parser.add_argument("--xscale", default="log", choices=["log", "linear"])
    args = parser.parse_args()
    plot(args.trajectory, args.output_base, args.schemes, args.xscale)
