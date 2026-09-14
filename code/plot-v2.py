#!/usr/bin/env python3
"""Plot a parameter sweep directly from JSON result summaries."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


STYLES = {
    "centralized_gnn": ("Centralized GNN", "tab:red", "-", "o"),
    "decentralized_gnn": ("Decentralized GNN", "tab:blue", "-", "s"),
    "centralized": ("Centralized", "tab:red", "-", "o"),
    "centralized_discrete": ("Centralized (2-bit)", "tab:red", "--", "o"),
    "centralized_random_phase": (
        "Centralized (random phase)", "tab:red", ":", "o"
    ),
    "centralized_random_phase_discrete": (
        "Centralized (random 2-bit)", "tab:red", "-.", "o"
    ),
    "decentralized": ("Decentralized", "tab:blue", "-", "s"),
    "decentralized_discrete": ("Decentralized (2-bit)", "tab:blue", "--", "s"),
    "decentralized_random_phase": (
        "Decentralized (random phase)", "tab:blue", ":", "s"
    ),
    "decentralized_random_phase_discrete": (
        "Decentralized (random 2-bit)", "tab:blue", "-.", "s"
    ),
}


def _new_run(summary_path):
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    metrics = summary.get("final_eval")
    if not metrics:
        return None
    return summary["config"], metrics


def _scaling_run(summary_path):
    pattern = re.compile(r"([A-Za-z]+)([-+]?\d*\.?\d+)")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    return dict(pattern.findall(summary_path.parent.name)), summary


def _x_key(experiments, requested):
    if requested:
        if not all(requested in parameters for parameters, _ in experiments):
            raise KeyError(f"x key {requested!r} is absent from at least one run")
        return requested

    shared = set(experiments[0][0])
    for parameters, _ in experiments[1:]:
        shared.intersection_update(parameters)
    varying = []
    for key in shared:
        try:
            values = {float(parameters[key]) for parameters, _ in experiments}
        except (TypeError, ValueError):
            continue
        if len(values) > 1:
            varying.append(key)
    if len(varying) != 1:
        raise ValueError(
            f"expected one varying numeric parameter, found {varying}; pass --x-key"
        )
    return varying[0]


def load_result_summaries(results_root, x_key=None):
    root = Path(results_root)
    summary_paths = sorted(root.glob("*/summary.json"))
    loader = _new_run
    if not summary_paths:
        summary_paths = sorted(root.glob("*/final_summary.json"))
        loader = _scaling_run

    experiments = [loaded for path in summary_paths if (loaded := loader(path))]
    if not experiments:
        raise FileNotFoundError(f"no usable JSON summaries found in {results_root}")

    key = _x_key(experiments, x_key)
    series = {}
    for parameters, metrics in experiments:
        x = float(parameters[key])
        for method in STYLES:
            value = metrics.get(method)
            if isinstance(value, dict):
                value = value.get("mean")
            if value is not None:
                series.setdefault(method, {})[x] = float(value)
    if not series:
        raise ValueError("the summaries contain no plottable metrics")
    return key, series


def plot(results_root, x_label, output_base, x_key=None):
    _, series = load_result_summaries(results_root, x_key)

    plt.rcParams.update(
        {
            "font.size": 13,
            "font.family": "serif",
            "axes.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    all_x = set()
    for key, (label, color, linestyle, marker) in STYLES.items():
        points = series.get(key)
        if not points:
            continue
        x = sorted(points)
        all_x.update(x)
        ax.plot(
            x,
            [points[value] for value in x],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.2,
            markersize=8,
            markerfacecolor="none",
            markeredgewidth=1.6,
            label=label,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Sum Rate (bps/Hz)")
    ax.set_xticks(sorted(all_x))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", linestyle=":", linewidth=0.9)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6)

    legend = ax.legend(
        loc="best", fontsize=9, frameon=True, fancybox=False, framealpha=1.0
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor("white")

    fig.tight_layout()
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        output_path = output_base.with_suffix(suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", required=True, help="Sweep directory containing run summaries"
    )
    parser.add_argument(
        "--x-key", help="Config key on the x axis; inferred for a one-variable sweep"
    )
    parser.add_argument("--x-label", required=True, help="X-axis label")
    parser.add_argument("--output-base", required=True, help="Output path without suffix")
    args = parser.parse_args()
    plot(args.results_root, args.x_label, args.output_base, args.x_key)
