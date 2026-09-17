#!/usr/bin/env python3
"""Plot the AP->CPU RIS message bits vs decentralized sum-rate frontier.

Left panel: absolute decentralized sum rate against the per-AP-RIS-pair payload
for every codec operating point, with the Pareto frontier.  Its error bars are
dominated by between-cluster channel variability, which is common to every arm,
so the right panel shows the decision-relevant quantity instead: the *paired*
difference against the fp32 R0c interface on the same 50 batch clusters, where
that common variability cancels.

`progressive` points with b_eps = 0 send no residual and are bit-identical to
the corresponding `pair_mag` point, so they are drawn once, as pair magnitude.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Okabe-Ito, ordered so adjacent pairs clear the CVD separation check; every
# family also carries its own marker, so identity never rests on hue alone.
FAMILY_STYLES = (
    ("phase_only", "Phase-only  $N b_p$", "#0072B2", "o"),
    ("pair_mag", "Pair magnitude  $N b_p + b_s$", "#D55E00", "s"),
    ("polar", "R0c polar  $N(b_p + b_m)$", "#009E73", "^"),
    ("progressive", "Progressive residual  $N b_p + b_s + N b_\\epsilon$", "#E69F00", "D"),
    ("vq", "Cartesian 2-D VQ  $N b_{vq}$", "#CC79A7", "v"),
    ("vq_gain", "Cartesian 2-D VQ + gain  $N b_{vq} + b_s$", "#56B4E9", "P"),
)
REFERENCE_COLOR = "#3A3A3A"
INK = "#222222"
MUTED = "#6A6A6A"

SHORT = {"r0_fp32": "R0 fp32", "r0c_fp32": "R0c fp32",
         "r1_shared_fp32": "Shared fp32", "r1_ap_ris_mag_fp32": "pair-mag fp32"}
TICKS = (30, 68, 128, 300, 960, 1920, 3840)
# Labelling all 16 frontier points collides; these are the decision-relevant ones.
LABELLED = ("phase_only_bp1", "pair_mag_bp2_bs2", "pair_mag_bp2_bs4",
            "pair_mag_bp3_bs8", "vq_cart_gain_b4_bs8", "vq_cart_gain_b6_bs8",
            "pair_mag_bpinf_bs8")


def apply_style():
    plt.rcParams.update({
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "0.55", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def short_label(name):
    if name in SHORT:
        return SHORT[name]
    for prefix, replacement in (("phase_only_", ""), ("pair_mag_", ""),
                                ("polar_", ""), ("progressive_", ""),
                                ("vq_cart_gain_", "vq+g "), ("vq_cart_", "vq ")):
        if name.startswith(prefix):
            return name[len(prefix):] if not replacement else replacement + name[len(prefix):]
    return name


def visible(arms):
    """Drop the progressive b_eps = 0 twins: they are the pair_mag points."""
    return [n for n, a in arms.items()
            if not (a["family"] == "progressive" and not a.get("b_resid"))]


def decorate(axis, pairs_count):
    axis.set_xscale("log", base=2)
    axis.set_xticks(TICKS)
    axis.set_xticklabels([f"{t}\n{t * pairs_count}" for t in TICKS], fontsize=8.5)
    axis.minorticks_off()
    axis.grid(axis="y", color="0.90", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_xlabel("AP$\\to$CPU payload: bits per AP-RIS pair (top) "
                    f"/ whole system, {pairs_count} pairs (bottom)")


def scatter(axis, arms, names, value, error, size=6.0):
    for family, _, color, marker in FAMILY_STYLES:
        picked = [n for n in names if arms[n]["family"] == family]
        if not picked:
            continue
        axis.errorbar([arms[n]["bits_per_pair"] for n in picked],
                      [value(arms[n]) for n in picked],
                      yerr=[error(arms[n]) for n in picked],
                      fmt=marker, markersize=size, color=color, linestyle="none",
                      elinewidth=1.0, capsize=2.0, markeredgecolor="white",
                      markeredgewidth=0.6, alpha=0.95, zorder=3)
    picked = [n for n in names if arms[n]["family"] == "reference"]
    if picked:
        axis.errorbar([arms[n]["bits_per_pair"] for n in picked],
                      [value(arms[n]) for n in picked],
                      yerr=[error(arms[n]) for n in picked],
                      fmt="*", markersize=13, color=REFERENCE_COLOR,
                      linestyle="none", elinewidth=1.0, capsize=2.0,
                      markeredgecolor="white", markeredgewidth=0.6, zorder=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="artifacts/decentralized_ris/e07_message_codec")
    args = parser.parse_args()

    root = Path(args.root)
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    arms = results["arms"]
    pairs_count = config["topology"]["ap_ris_pairs"]
    clusters = results["clusters"]
    names = visible(arms)

    apply_style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.6, 6.0))

    scatter(left, arms, names,
            lambda a: a["decentralized_sum_rate"],
            lambda a: a["decentralized_sum_rate"] - a["ci95_low"])
    front = sorted((n for n in names if arms[n]["on_pareto_frontier"]),
                   key=lambda n: arms[n]["bits_per_pair"])
    left.step([arms[n]["bits_per_pair"] for n in front],
              [arms[n]["decentralized_sum_rate"] for n in front],
              where="post", color=INK, linewidth=1.6, alpha=0.75, zorder=2)
    left.set_ylabel("Decentralized sum rate (bps/Hz)")
    left.set_title("Absolute rate — error bars are dominated by\n"
                   "channel variability shared by every arm",
                   fontsize=10.5, loc="left", color=MUTED)
    decorate(left, pairs_count)

    scatter(right, arms, names,
            lambda a: a["vs_r0c_fp32"]["mean"],
            lambda a: a["vs_r0c_fp32"]["mean"] - a["vs_r0c_fp32"]["ci95_low"])
    right.axhline(0.0, color=REFERENCE_COLOR, linewidth=1.2, zorder=1)
    right.annotate("fp32 R0c interface", (30, 0.0), textcoords="offset points",
                   xytext=(2, 7), fontsize=9, color=REFERENCE_COLOR, ha="left")
    right.step([arms[n]["bits_per_pair"] for n in front],
               [arms[n]["vs_r0c_fp32"]["mean"] for n in front],
               where="post", color=INK, linewidth=1.6, alpha=0.75, zorder=2)
    for index, name in enumerate(n for n in front if n in LABELLED):
        arm = arms[name]
        right.annotate(short_label(name),
                       (arm["bits_per_pair"], arm["vs_r0c_fp32"]["mean"]),
                       textcoords="offset points",
                       xytext=(8, 6 if index % 2 else -17), fontsize=8.5,
                       color=INK, rotation=24, rotation_mode="anchor")
    right.set_ylabel("Paired difference vs fp32 R0c (bps/Hz)")
    right.set_title("Paired difference — the decision-relevant view;\n"
                    "labels mark the Pareto frontier",
                    fontsize=10.5, loc="left", color=MUTED)
    right.set_ylim(-9.4, 1.5)
    decorate(right, pairs_count)

    handles = [Line2D([], [], color=color, marker=marker, linestyle="none",
                      markersize=6.5, markeredgecolor="white", label=label)
               for _, label, color, marker in FAMILY_STYLES]
    handles.append(Line2D([], [], color=REFERENCE_COLOR, marker="*", linestyle="none",
                          markersize=11, label="fp32 reference interface"))
    handles.append(Line2D([], [], color=INK, linewidth=1.6, alpha=0.75,
                          label="Pareto frontier"))
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                  fontsize=9.5, bbox_to_anchor=(0.5, -0.03))
    status = ("controls PASSED" if results["controls"]["passed"]
              else "PROVISIONAL — a declared control did not pass, see report.md")
    figure.suptitle(
        "AP$\\to$CPU RIS message: bits vs decentralized sum rate "
        f"(fixed 500k R0 checkpoint, {clusters} batch clusters, 95% CI)\n{status}",
        fontsize=12.5, y=1.0)
    figure.tight_layout(rect=(0, 0.09, 1, 0.94))

    for suffix in ("png", "pdf"):
        path = root / f"frontier.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[plot] {path}")


if __name__ == "__main__":
    main()
