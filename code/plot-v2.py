#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator


STYLES = {
    "centralized_gnn": ("Centralized GNN", "tab:red", "-", "o"),
    "decentralized_gnn": ("Decentralized GNN", "tab:blue", "-", "s"),
    "centralized": ("Centralized", "tab:red", "-", "o"),
    "centralized_discrete": ("Centralized (2-bit)", "tab:red", "--", "o"),
    "centralized_random_phase_discrete": (
        "Centralized (random 2-bit)", "tab:red", ":", "o"
    ),
    "decentralized": ("Decentralized", "tab:blue", "-", "s"),
    "decentralized_discrete": ("Decentralized (2-bit)", "tab:blue", "--", "s"),
    "decentralized_random_phase_discrete": (
        "Decentralized (random 2-bit)", "tab:blue", ":", "s"
    ),
}


def load_stage1_results(results_root):
    pattern = re.compile(r"([A-Za-z]+)([-+]?\d*\.?\d+)")
    experiments = []
    for summary_path in sorted(Path(results_root).glob("*/final_summary.npy")):
        variables = dict(pattern.findall(summary_path.parent.name))
        summary = np.load(summary_path, allow_pickle=True).item()
        experiments.append((variables, summary))

    if not experiments:
        raise FileNotFoundError(f"No final summaries found in {results_root}")

    varying_key = next(
        key
        for key in experiments[0][0]
        if len({float(variables[key]) for variables, _ in experiments}) > 1
    )
    data = {}
    for variables, summary in experiments:
        x = float(variables[varying_key])
        for method in ("centralized_gnn", "decentralized_gnn"):
            data.setdefault(method, {})[x] = summary[method]["mean"]
    return pd.DataFrame.from_dict(data, orient="index").sort_index(axis=1)


def plot(excel_file, x_label, output_base, results_root=None):
    if results_root:
        df = load_stage1_results(results_root)
    else:
        df = pd.read_excel(excel_file, index_col=0)
    x = df.columns.astype(float)

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
    for key, (label, color, linestyle, marker) in STYLES.items():
        if key not in df.index:
            continue
        ax.plot(
            x,
            df.loc[key].values,
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
    ax.set_xticks(x)
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
    parser = argparse.ArgumentParser(description="Plot a Stage 0 or Stage 1 sweep.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--excel", help="Input Stage 0 summary workbook")
    source.add_argument(
        "--results-root", help="Stage 1 sweep directory containing final summaries"
    )
    parser.add_argument("--x-label", required=True, help="X-axis label")
    parser.add_argument("--output-base", required=True, help="Output path without suffix")
    args = parser.parse_args()
    plot(args.excel, args.x_label, args.output_base, args.results_root)
