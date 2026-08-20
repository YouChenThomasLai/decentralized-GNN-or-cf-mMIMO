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


SCALE_PATTERN = re.compile(r"scale(?P<scale>\d+)_A(?P<ap>\d+)_K(?P<ue>\d+)")
STAGES = (
    ("stage0_ris", "Stage 0 RIS"),
    ("stage1_no_ris", "Stage 1 no-RIS"),
)
COLORS = {"centralized": "tab:red", "decentralized": "tab:blue"}


def load_records(results_root):
    records = []
    for stage, _ in STAGES:
        for path in sorted(
            (Path(results_root) / stage).glob(
                "scale*/seed*/final_eval/metrics_run*.json"
            )
        ):
            match = SCALE_PATTERN.match(path.parents[2].name)
            if match is None:
                continue
            with open(path) as metrics_file:
                metrics = json.load(metrics_file)
            records.append(
                {
                    "stage": stage,
                    "scale": int(match["scale"]),
                    "num_ap": int(match["ap"]),
                    "num_ue": int(match["ue"]),
                    "seed": int(path.parents[1].name.removeprefix("seed")),
                    "metrics": metrics,
                }
            )
    if not records:
        raise FileNotFoundError(f"No per-seed metrics found under {results_root}")
    return records


def write_csv(records, output_dir):
    metric_names = sorted(
        {name for record in records for name in record["metrics"]}
    )
    output_path = Path(output_dir) / "scaling_metrics.csv"
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("stage", "scale", "num_ap", "num_ue", "seed", *metric_names),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{key: record[key] for key in writer.fieldnames[:5]},
                    **record["metrics"],
                }
            )
    print(f"Saved: {output_path}")


def metric_values(records, stage, metric, x_key="scale"):
    grouped = {}
    for record in records:
        if record["stage"] != stage or metric not in record["metrics"]:
            continue
        x = (
            record[x_key]
            if x_key in record
            else record["num_ap"] * record["num_ue"]
        )
        grouped.setdefault(x, []).append(record["metrics"][metric])
    return grouped


def plot_series(ax, grouped, label, color, marker):
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
            alpha=0.7,
            zorder=3,
        )
    ax.plot(x, means, color=color, marker=marker, linewidth=2, label=label)
    ax.fill_between(x, lows, highs, color=color, alpha=0.12)


def save_figure(fig, output_dir, name):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = Path(output_dir) / f"{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_stage_panels(records, output_dir, name, ylabel, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=False)
    for ax, (stage, title) in zip(axes, STAGES):
        for metric, label, color, marker in metrics(stage):
            plot_series(
                ax,
                metric_values(records, stage, metric),
                label,
                color,
                marker,
            )
        ax.set_title(title)
        ax.set_xlabel("Scale")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted({r["scale"] for r in records if r["stage"] == stage}))
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend()
    save_figure(fig, output_dir, name)


def plot_runtime_memory(records, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, (stage, title) in enumerate(STAGES):
        central = "centralized" if stage == "stage0_ris" else "centralized_gnn"
        decentral = "decentralized" if stage == "stage0_ris" else "decentralized_gnn"
        plot_series(
            axes[row, 0],
            metric_values(
                records, stage, f"{central}_seconds_per_sample", "network_size"
            ),
            "Centralized",
            COLORS["centralized"],
            "o",
        )
        plot_series(
            axes[row, 0],
            metric_values(
                records, stage, f"{decentral}_seconds_per_sample", "network_size"
            ),
            "Decentralized",
            COLORS["decentralized"],
            "s",
        )
        plot_series(
            axes[row, 1],
            metric_values(records, stage, "peak_gpu_memory_bytes", "network_size"),
            "Peak GPU memory",
            "tab:purple",
            "^",
        )
        axes[row, 0].set_ylabel(f"{title}\nSeconds / sample")
        axes[row, 1].set_ylabel(f"{title}\nBytes")
        for ax in axes[row]:
            ax.set_xlabel("AP × UE")
            ax.grid(True, linestyle=":", alpha=0.7)
            ax.legend()
    save_figure(fig, output_dir, "runtime_memory")


def plot_association_visibility(records, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, (stage, title) in enumerate(STAGES):
        plot_series(
            axes[row, 0],
            metric_values(records, stage, "associated_ue_count_mean"),
            "UEs / AP",
            "tab:green",
            "o",
        )
        plot_series(
            axes[row, 0],
            metric_values(records, stage, "serving_ap_count_mean"),
            "APs / UE",
            "tab:orange",
            "s",
        )
        plot_series(
            axes[row, 1],
            metric_values(
                records, stage, "local_to_global_visibility_ratio_mean"
            ),
            "Visibility ratio",
            "tab:cyan",
            "^",
        )
        axes[row, 0].set_ylabel(title)
        axes[row, 1].set_ylabel(title)
        for ax in axes[row]:
            ax.set_xlabel("Scale")
            ax.grid(True, linestyle=":", alpha=0.7)
            ax.legend()
    axes[0, 0].set_title("Association load")
    axes[0, 1].set_title("Local / global CSI visibility")
    save_figure(fig, output_dir, "association_visibility")


def main():
    parser = argparse.ArgumentParser(
        description="Plot snapshot network-scaling per-seed metrics"
    )
    parser.add_argument("--results-root", default="results_snapshot_scaling")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    output_dir = Path(args.output_dir or Path(args.results_root) / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.results_root)
    write_csv(records, output_dir)

    plot_stage_panels(
        records,
        output_dir,
        "relative_gap",
        "Relative C/D gap (%)",
        lambda stage: (("relative_gap_percent", "C−D", "tab:purple", "o"),),
    )
    plot_stage_panels(
        records,
        output_dir,
        "mean_per_ue_rate",
        "Mean per-UE rate (bps/Hz)",
        lambda stage: (
            (
                "centralized_mean_per_ue_rate"
                if stage == "stage0_ris"
                else "centralized_gnn_mean_per_ue_rate",
                "Centralized",
                COLORS["centralized"],
                "o",
            ),
            (
                "decentralized_mean_per_ue_rate"
                if stage == "stage0_ris"
                else "decentralized_gnn_mean_per_ue_rate",
                "Decentralized",
                COLORS["decentralized"],
                "s",
            ),
        ),
    )
    plot_stage_panels(
        records,
        output_dir,
        "fifth_percentile_ue_rate",
        "5th-percentile UE rate (bps/Hz)",
        lambda stage: (
            (
                "centralized_fifth_percentile_ue_rate"
                if stage == "stage0_ris"
                else "centralized_gnn_fifth_percentile_ue_rate",
                "Centralized",
                COLORS["centralized"],
                "o",
            ),
            (
                "decentralized_fifth_percentile_ue_rate"
                if stage == "stage0_ris"
                else "decentralized_gnn_fifth_percentile_ue_rate",
                "Decentralized",
                COLORS["decentralized"],
                "s",
            ),
        ),
    )
    plot_runtime_memory(records, output_dir)
    plot_association_visibility(records, output_dir)


if __name__ == "__main__":
    main()
