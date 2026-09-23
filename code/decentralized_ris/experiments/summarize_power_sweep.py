"""Render the frozen G2/R0 transmit-power sweep as a Markdown table and a figure.

Line conventions follow `code/plot_progress_report.py`: G2 blue, R0 grey,
centralized solid, paper-decentralized dashed.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INK = "#263238"
BLUE = "#0072B2"
GRAY = "#777777"


def rows(summary):
    for power in (f"{value:g}" for value in summary["decision"]["powers_dbm"]):
        g2 = summary["models"]["g2"]["per_power"][power]
        r0 = summary["models"]["r0"]["per_power"][power]
        contrast = summary["contrasts"][power]["g2_minus_r0_decentralized"]
        yield power, g2, r0, contrast


def markdown(summary):
    lines = [
        "| $P_{\\max}$ dBm | G2 centralized | G2 paper-dec. | G2 2-bit dec. | G2 rel. gap |"
        " R0 centralized | R0 paper-dec. | R0 2-bit dec. | R0 rel. gap |"
        " G2−R0 paper-dec. [95% CI] |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for power, g2, r0, contrast in rows(summary):
        lines.append(
            f"| {power} | {g2['centralized']['mean']:.4f} | {g2['decentralized']['mean']:.4f} |"
            f" {g2['decentralized_2bit']['mean']:.4f} |"
            f" {100 * g2['relative_decentralization_gap']['mean']:.2f}% |"
            f" {r0['centralized']['mean']:.4f} | {r0['decentralized']['mean']:.4f} |"
            f" {r0['decentralized_2bit']['mean']:.4f} |"
            f" {100 * r0['relative_decentralization_gap']['mean']:.2f}% |"
            f" ${contrast['mean']:+.4f}$ $[{contrast['ci_low']:+.4f},{contrast['ci_high']:+.4f}]$ |"
        )
    return "\n".join(lines)


def figure(summary, path):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    powers = [float(power) for power, *_ in rows(summary)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")

    for name, color in (("g2", BLUE), ("r0", GRAY)):
        model = summary["models"][name]
        label = "G2-150k" if name == "g2" else "R0-500k"
        for mode, style in (("centralized", "-"), ("decentralized", "--")):
            values = [model["per_power"][f"{power:g}"][mode]["mean"] for power in powers]
            errors = [
                model["per_power"][f"{power:g}"][mode]["mean"]
                - model["per_power"][f"{power:g}"][mode]["ci_low"]
                for power in powers
            ]
            name_mode = "centralized" if mode == "centralized" else "paper-dec."
            left.errorbar(powers, values, yerr=errors, color=color, linestyle=style,
                          linewidth=2.0, marker="o", markersize=5,
                          markerfacecolor="white", capsize=3,
                          label=f"{label} {name_mode}")
        ceiling = model["interference_limited_ceiling"]["decentralized"]["mean"]
        left.axhline(ceiling, color=color, linewidth=1.0, linestyle=":")
        left.annotate(f"{label} interference ceiling {ceiling:.1f}",
                      (powers[0], ceiling), xytext=(2, 3), textcoords="offset points",
                      color=color, fontsize=8.5)
        gaps = [100 * model["per_power"][f"{power:g}"]["relative_decentralization_gap"]["mean"]
                for power in powers]
        right.plot(powers, gaps, color=color, linewidth=2.0, marker="o", markersize=5,
                   markerfacecolor="white", label=label)

    left.set(xlabel="Per-AP transmit power budget $P_{\\max}$ (dBm)",
             ylabel="Sum rate (bps/Hz)")
    left.grid(axis="y", color="0.88")
    left.set_axisbelow(True)
    left.legend(loc="lower right", frameon=False, fontsize=9.0)
    right.set(xlabel="Per-AP transmit power budget $P_{\\max}$ (dBm)",
              ylabel="Relative decentralization gap $(C-D)/C$ (%)")
    right.axhline(0.0, color=INK, linewidth=0.8)
    right.grid(axis="y", color="0.88")
    right.set_axisbelow(True)
    right.legend(loc="center right", frameon=False, fontsize=9.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    table = markdown(summary)
    print(table)
    args.summary.with_suffix(".md").write_text(table + "\n", encoding="utf-8")
    if args.figure:
        figure(summary, args.figure)


if __name__ == "__main__":
    main()
