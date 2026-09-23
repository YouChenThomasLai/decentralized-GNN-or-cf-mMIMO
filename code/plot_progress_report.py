"""Render the E01 and E06 training figures from stored evaluation results."""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/decentralized_ris"
OUTPUT = ROOT / "doc/figures"
INK = "#263238"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#777777"


E01 = ARTIFACTS / "e01_baseline_training"
R0_MILESTONES = (2000, 40000, 100000, 150000, 500000)
VALIDATION_WINDOW = 9


def r0_run_dir(step):
    paths = list((E01 / f"iter{step:06d}").glob("*/run0"))
    assert len(paths) == 1, (step, paths)
    return paths[0]


def r0_rate(step, mode):
    lines = (r0_run_dir(step) / "final_eval/final_eval_run0.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    prefix = f"{mode}:"
    return float(next(line.split(":", 1)[1] for line in lines if line.startswith(prefix)))


def r0_validation_curve(mode):
    """Stitch the per-segment validation logs into one step-ordered trajectory.

    Each milestone directory stores one resume segment: the per-step loss array gives its
    length, so its start iteration and validation interval follow from the stored arrays.
    The 2k segment is a separate from-scratch run of the same seed; it is plotted as part
    of the same trajectory, and the two runs' overlapping steps are both kept.
    """
    steps, rates = [], []
    for milestone in R0_MILESTONES:
        arrays = r0_run_dir(milestone) / "arrays"
        length = len(np.load(arrays / "losses_run0.npy"))
        values = np.load(arrays / f"val_sum_rate_{mode}_run0.npy")
        interval = length // len(values)
        start = milestone - length
        steps.extend(start + interval * np.arange(1, len(values) + 1))
        rates.extend(values)
    order = np.argsort(np.asarray(steps), kind="stable")
    return np.asarray(steps)[order], np.asarray(rates)[order]


def rolling_mean(values, window):
    """Centred mean that shrinks the window at both ends instead of trimming the curve."""
    half = window // 2
    return np.asarray([
        values[max(0, index - half):index + half + 1].mean()
        for index in range(len(values))
    ])


def g2_validation_curve(mode):
    root = ARTIFACTS / "e06_graph_energy_training"
    paths = [
        *sorted((root / "training_10k").glob("graph-energy-g2-*/metrics.npz")),
        *sorted((root / "g2_long_training/g2").glob("iter*/metrics.npz")),
    ]
    steps, rates = [], []
    for path in paths:
        with np.load(path) as arrays:
            steps.extend(arrays["validation_iteration"])
            rates.extend(arrays[f"validation_{mode}"])
    order = np.argsort(np.asarray(steps), kind="stable")
    return np.asarray(steps)[order], np.asarray(rates)[order]


def save(figure, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / name, format="svg", bbox_inches="tight")
    plt.close(figure)


def main():
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    figure, axis = plt.subplots(figsize=(8.0, 4.3), layout="constrained")
    milestone_x = [step / 1000 for step in R0_MILESTONES]
    for mode, label, color in (("centralized", "Centralized", ORANGE),
                               ("decentralized", "Decentralized", BLUE)):
        curve_steps, curve_rates = r0_validation_curve(mode)
        axis.plot(curve_steps / 1000, rolling_mean(curve_rates, VALIDATION_WINDOW), color=color,
                  linewidth=2.0, label=label)
        rates = [r0_rate(step, mode) for step in R0_MILESTONES]
        axis.plot(milestone_x, rates, linestyle="none", marker="o", markersize=7,
                  markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6)
        axis.annotate(f"{rates[-1]:.2f}", (milestone_x[-1], rates[-1]), xytext=(-6, 9),
                      textcoords="offset points", ha="right", color=color)
    axis.plot([], [], linestyle="none", marker="o", markersize=7, markerfacecolor="white",
              markeredgecolor=INK, markeredgewidth=1.6, label="Milestone evaluation")
    axis.set(xlabel="Training steps (thousands)", ylabel="Sum rate (bps/Hz)",
             ylim=(4, 26), xlim=(-15, 525))
    axis.set_xticks([2, 40, 100, 150, 300, 500])
    axis.grid(axis="y", color="0.88")
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, fontsize=9.5)
    save(figure, "r0_training_milestones.svg")

    figure, axis = plt.subplots(figsize=(8.0, 4.3), layout="constrained")
    for model, color, curve in (
        ("R0", GRAY, r0_validation_curve),
        ("G2", BLUE, g2_validation_curve),
    ):
        for mode, linestyle in (("centralized", "-"), ("decentralized", "--")):
            steps, rates = curve(mode)
            window = VALIDATION_WINDOW if model == "R0" else 5
            axis.plot(
                steps / 1000,
                rolling_mean(rates, window),
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=f"{model} {mode}",
            )
    axis.set_xscale("log")
    axis.set(
        xlabel="Training steps (thousands; log scale)",
        ylabel="Sum rate (bps/Hz)",
        ylim=(4, 27),
        xlim=(0.85, 550),
    )
    axis.set_xticks(
        [1, 2, 5, 10, 20, 40, 100, 200, 500],
        labels=["1", "2", "5", "10", "20", "40", "100", "200", "500"],
    )
    axis.grid(axis="y", color="0.88")
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, ncol=2, fontsize=9.5)
    save(figure, "r0_g2_training.svg")


if __name__ == "__main__":
    main()
