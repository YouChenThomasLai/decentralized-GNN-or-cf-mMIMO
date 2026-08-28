#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCALE_PATTERN = re.compile(
    r"scale(?P<scale>\d+)_A(?P<ap>\d+)_K(?P<ue>\d+)_L(?P<ris>\d+)"
)
COLORS = {"centralized": "tab:red", "decentralized": "tab:blue"}


def load_records(results_root):
    records = []
    pattern = "stage0_ris/scale*/seed*/final_eval/metrics_run*.json"
    for path in sorted(Path(results_root).glob(pattern)):
        match = SCALE_PATTERN.fullmatch(path.parents[2].name)
        if match is None:
            continue
        with open(path) as metrics_file:
            metrics = json.load(metrics_file)
        records.append(
            {
                "scale": int(match["scale"]),
                "num_ap": int(match["ap"]),
                "num_ue": int(match["ue"]),
                "num_ris": int(match["ris"]),
                "seed": int(path.parents[1].name.removeprefix("seed")),
                "metrics": metrics,
            }
        )
    if not records:
        raise FileNotFoundError(
            f"No completed v2 per-seed metrics found under {results_root}"
        )
    return records


def write_csv(records, output_dir):
    metric_names = sorted(
        {name for record in records for name in record["metrics"]}
    )
    output_path = Path(output_dir) / "scaling_metrics.csv"
    fields = ("scale", "num_ap", "num_ue", "num_ris", "seed", *metric_names)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{name: record[name] for name in fields[:5]},
                    **record["metrics"],
                }
            )


def metric_values(records, metric, x_key="scale"):
    grouped = {}
    for record in records:
        if metric not in record["metrics"]:
            continue
        x = (
            record[x_key]
            if x_key in record
            else record["num_ap"] * record["num_ue"]
        )
        grouped.setdefault(x, []).append(record["metrics"][metric])
    return grouped


def plot_series(ax, grouped, label, color, marker="o"):
    if not grouped:
        return
    x = np.asarray(sorted(grouped), dtype=float)
    values = [np.asarray(grouped[value], dtype=float) for value in x]
    means = np.asarray([value.mean() for value in values])
    lows = np.asarray([value.min() for value in values])
    highs = np.asarray([value.max() for value in values])
    for coordinate, seed_values in zip(x, values):
        ax.scatter(
            np.full(seed_values.shape, coordinate),
            seed_values,
            color=color,
            marker=marker,
            facecolors="none",
            zorder=3,
        )
    ax.plot(x, means, color=color, marker=marker, linewidth=2, label=label)
    ax.fill_between(x, lows, highs, color=color, alpha=0.12)


def finish(fig, output_dir, name):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            Path(output_dir) / f"{name}.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_topology_gate(results_root, output_dir):
    with open(Path(results_root) / "topology_gate" / "summary.json") as source:
        summary = json.load(source)
    scales = sorted(int(scale) for scale in summary["scales"])
    values = [summary["scales"][str(scale)] for scale in scales]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    axes[0, 0].plot(
        scales,
        [value["ap_nearest_neighbor_distance"]["median"] for value in values],
        "o-",
        label="AP nearest neighbor",
    )
    axes[0, 0].plot(
        scales,
        [value["ue_nearest_ap_distance"]["median"] for value in values],
        "s-",
        label="UE nearest AP",
    )
    axes[0, 0].set_ylabel("Wrapped distance (m)")

    axes[0, 1].plot(
        scales,
        [value["serving_ap_count"]["mean"] for value in values],
        "o-",
        label="Serving APs / UE",
    )
    axes[0, 1].plot(
        scales,
        [value["associated_ue_count"]["mean"] for value in values],
        "s-",
        label="UEs / AP",
    )
    axes[0, 1].set_ylabel("Association load")

    axes[1, 0].plot(
        scales,
        [value["visible_links_per_ap"]["mean"] for value in values],
        "o-",
        label="Visible links / AP",
    )
    axes[1, 0].set_ylabel("Absolute visible AP–UE links")
    ratio_axis = axes[1, 0].twinx()
    ratio_axis.plot(
        scales,
        [
            value["local_to_global_visibility_ratio"]["mean"]
            for value in values
        ],
        "s--",
        color="tab:purple",
        label="Visibility ratio",
    )
    ratio_axis.set_ylabel("Local / global visibility")
    lines = axes[1, 0].lines + ratio_axis.lines
    axes[1, 0].legend(lines, [line.get_label() for line in lines])

    axes[1, 1].plot(
        scales,
        [value["ap_pair_shared_ue_fraction"] for value in values],
        "o-",
        label="AP pair shares a UE",
    )
    axes[1, 1].plot(
        scales,
        [value["bipartite_fully_connected_fraction"] for value in values],
        "s-",
        label="Bipartite graph connected",
    )
    axes[1, 1].set_ylabel("Snapshot / pair fraction")
    for ax in axes.flat:
        ax.set_xlabel("Scale")
        ax.set_xticks(scales)
        ax.grid(True, linestyle=":", alpha=0.7)
        if ax is not axes[1, 0]:
            ax.legend()
    fig.suptitle(f"Stage 0 v2 topology gate: {summary['status']}")
    finish(fig, output_dir, "topology_stationarity")


def plot_performance(records, output_dir):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plot_series(
        ax,
        metric_values(records, "relative_gap_percent"),
        "Centralized − decentralized",
        "tab:purple",
    )
    ax.set_xlabel("Scale")
    ax.set_ylabel("Relative C/D gap (%)")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()
    finish(fig, output_dir, "relative_gap")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, suffix, title in (
        (axes[0], "mean_per_ue_rate", "Mean per-UE rate"),
        (axes[1], "fifth_percentile_ue_rate", "5th-percentile UE rate"),
    ):
        for method, label, marker in (
            ("centralized", "Centralized", "o"),
            ("decentralized", "Decentralized", "s"),
        ):
            plot_series(
                ax,
                metric_values(records, f"{method}_{suffix}"),
                label,
                COLORS[method],
                marker,
            )
        ax.set_title(title)
        ax.set_xlabel("Scale")
        ax.set_ylabel("Rate (bps/Hz)")
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend()
    finish(fig, output_dir, "per_ue_rates")


def plot_mechanisms(records, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for metric, label, color, marker in (
        ("serving_ap_count_mean", "Serving APs / UE", "tab:orange", "o"),
        ("associated_ue_count_mean", "UEs / AP", "tab:green", "s"),
        ("visible_links_per_ap_mean", "Visible links / AP", "tab:cyan", "^"),
    ):
        plot_series(axes[0], metric_values(records, metric), label, color, marker)
    plot_series(
        axes[1],
        metric_values(records, "local_to_global_visibility_ratio_mean"),
        "Visibility ratio",
        "tab:purple",
    )
    axes[0].set_ylabel("Count")
    axes[1].set_ylabel("Local / global visibility")
    for ax in axes:
        ax.set_xlabel("Scale")
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend()
    finish(fig, output_dir, "association_visibility")


def plot_scalability(records, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_series(
        axes[0, 0],
        metric_values(records, "training_seconds", "network_size"),
        "Training",
        "tab:green",
    )
    for method, label, marker in (
        ("centralized", "Centralized", "o"),
        ("decentralized", "Decentralized", "s"),
    ):
        plot_series(
            axes[0, 1],
            metric_values(
                records, f"{method}_seconds_per_sample", "network_size"
            ),
            label,
            COLORS[method],
            marker,
        )
    plot_series(
        axes[1, 0],
        metric_values(records, "peak_gpu_memory_bytes", "network_size"),
        "Peak GPU memory",
        "tab:purple",
    )
    plot_series(
        axes[1, 1],
        metric_values(records, "parameter_count", "network_size"),
        "Parameters",
        "tab:brown",
    )
    ylabels = ("Seconds", "Seconds / sample", "Bytes", "Count")
    for ax, ylabel in zip(axes.flat, ylabels):
        ax.set_xlabel("AP × UE")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend()
    finish(fig, output_dir, "runtime_memory_parameters")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Stage 0 v2 BPP scaling artifacts"
    )
    parser.add_argument(
        "--results-root", default="../results_snapshot_scaling_v2_bpp"
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    output_dir = Path(args.output_dir or Path(args.results_root) / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.results_root)
    write_csv(records, output_dir)
    plot_topology_gate(args.results_root, output_dir)
    plot_performance(records, output_dir)
    plot_mechanisms(records, output_dir)
    plot_scalability(records, output_dir)


if __name__ == "__main__":
    main()
