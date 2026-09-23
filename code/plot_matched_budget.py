#!/usr/bin/env python3
"""Plot the matched-bit-budget frontier of the G2 and R0 AP->CPU messages.

Left panel: paper-decentralized sum rate against the payload each wire format
actually costs, for both frozen policies on the same channel draws.  Colour
carries the only comparison that matters here - which message is being
compressed - and marker shape carries the codec family, so identity never rests
on hue alone.

Right panel: the paired difference against the *uncompressed* R0 anchor on the
same 50 batch clusters, where the between-cluster channel variability that
dominates the absolute error bars cancels.  A point above zero is a wire format
that beats the anchor while sending fewer bits.

R0 payloads use the generous accounting: the last 2N latent coordinates are the
RIS-independent fe_AP block, so they are charged once per AP rather than once
per AP-RIS pair.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Two Okabe-Ito hues, validated for CVD separation against the chart surface.
G2_COLOR = "#0072B2"
R0_COLOR = "#D55E00"
REFERENCE_COLOR = "#3A3A3A"
INK = "#222222"
MUTED = "#6A6A6A"

MARKERS = {
    "g2_energy": ("o", "G2: phase + energy scalar"),
    "g2_equal": ("v", "G2: phase only, equal weights"),
    "r0_latent": ("s", "R0: 4N latent, learned reduction"),
    "r0c_logit": ("D", "R0c: 2N logits"),
    "r0c_vq": ("X", "R0c: Cartesian 2-D VQ"),
    "r0c_vqgain": ("P", "R0c: 2-D VQ + gain"),
    "r0_phase": ("^", "R0 backbone: phase only"),
    "r0_pairmag": ("<", "R0 backbone: phase + pair scale"),
    "r0c_polar": (">", "R0c: polar, per-element magnitude"),
    "r0c_progressive": ("p", "R0c: progressive residual"),
}
TICKS = (30, 68, 128, 240, 480, 992, 2400)


def apply_style():
    plt.rcParams.update({
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "0.55", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def decorate(axis, pairs_count, budget):
    axis.set_xscale("log", base=2)
    axis.set_xticks(TICKS)
    axis.set_xticklabels([f"{t}\n{t * pairs_count}" for t in TICKS], fontsize=8.5)
    axis.minorticks_off()
    axis.grid(axis="y", color="0.90", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.axvline(budget, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                 zorder=1)
    axis.set_xlabel("AP$\\to$CPU payload: bits per AP-RIS pair (top) "
                    f"/ whole system, {pairs_count} pairs (bottom)")


def scatter(axis, arms, value, error):
    for family, (marker, _) in MARKERS.items():
        picked = [n for n, a in arms.items() if a["family"] == family]
        if not picked:
            continue
        color = G2_COLOR if arms[picked[0]]["backbone"] == "g2" else R0_COLOR
        axis.errorbar([arms[n]["bits_per_pair_generous"] for n in picked],
                      [value(arms[n]) for n in picked],
                      yerr=[error(arms[n]) for n in picked],
                      fmt=marker, markersize=6.0, color=color, linestyle="none",
                      elinewidth=0.9, capsize=1.8, markeredgecolor="white",
                      markeredgewidth=0.6, alpha=0.95, zorder=3)
    for name, arm in arms.items():
        if arm["family"] != "reference":
            continue
        color = G2_COLOR if arm["backbone"] == "g2" else R0_COLOR
        axis.errorbar([arm["bits_per_pair_generous"]], [value(arm)],
                      yerr=[error(arm)], fmt="*", markersize=14, color=color,
                      linestyle="none", elinewidth=0.9, capsize=1.8,
                      markeredgecolor="white", markeredgewidth=0.6, zorder=4)


def frontier(axis, arms, backbone, value):
    picked = sorted((n for n, a in arms.items()
                     if a["backbone"] == backbone and a["on_pareto_frontier"]),
                    key=lambda n: arms[n]["bits_per_pair_generous"])
    color = G2_COLOR if backbone == "g2" else R0_COLOR
    axis.step([arms[n]["bits_per_pair_generous"] for n in picked],
              [value(arms[n]) for n in picked], where="post", color=color,
              linewidth=1.6, alpha=0.6, zorder=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="artifacts/decentralized_ris/e07_message_codec")
    args = parser.parse_args()

    root = Path(args.root)
    results = json.loads(
        (root / "matched_budget_results.json").read_text(encoding="utf-8"))
    config = json.loads(
        (root / "matched_budget_config.json").read_text(encoding="utf-8"))
    discovery = results["discovery"]
    arms = discovery["arms"]
    decisions = discovery["decisions"]
    pairs_count = config["topology"]["ap_ris_pairs"]
    budget = decisions["primary_budget_bits_per_pair"]
    primary = decisions["primary_arm"]

    apply_style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.8, 6.1))

    scatter(left, arms, lambda a: a["continuous"]["mean"],
            lambda a: a["continuous"]["mean"] - a["continuous"]["ci95_low"])
    for backbone in ("g2", "r0"):
        frontier(left, arms, backbone, lambda a: a["continuous"]["mean"])
    left.set_ylabel("Paper-decentralized sum rate (bps/Hz)")
    left.set_title("Absolute rate — error bars are dominated by the\n"
                   "channel variability every arm shares",
                   fontsize=10.5, loc="left", color=MUTED)
    decorate(left, pairs_count, budget)

    scatter(right, arms,
            lambda a: a["vs_g2_native_continuous"]["mean"],
            lambda a: (a["vs_g2_native_continuous"]["mean"]
                       - a["vs_g2_native_continuous"]["ci95_low"]))
    anchor = arms["r0_native_fp32"]["vs_g2_native_continuous"]["mean"]
    right.axhline(0.0, color=G2_COLOR, linewidth=1.2, alpha=0.8, zorder=1)
    right.axhline(anchor, color=R0_COLOR, linewidth=1.2, alpha=0.8, zorder=1)
    right.annotate("uncompressed G2 message (992 bits)", (30, 0.0),
                   textcoords="offset points", xytext=(2, 5), fontsize=9,
                   color=G2_COLOR, ha="left")
    right.annotate("uncompressed R0 anchor", (30, anchor),
                   textcoords="offset points", xytext=(2, 5), fontsize=9,
                   color=R0_COLOR, ha="left")
    for backbone in ("g2", "r0"):
        frontier(right, arms, backbone,
                 lambda a: a["vs_g2_native_continuous"]["mean"])
    point = arms[primary]
    right.annotate(f"declared operating point, {budget} bits/pair",
                   (point["bits_per_pair_generous"],
                    point["vs_g2_native_continuous"]["mean"]),
                   textcoords="offset points", xytext=(52, -74), fontsize=9,
                   color=INK, ha="left",
                   arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9))
    right.set_ylabel("Paired difference vs the uncompressed G2 message (bps/Hz)")
    right.set_title("Paired difference — the decision-relevant view;\n"
                    "the common channel variability cancels",
                    fontsize=10.5, loc="left", color=MUTED)
    decorate(right, pairs_count, budget)

    handles = [Line2D([], [], color=(G2_COLOR if family.startswith("g2")
                                     else R0_COLOR),
                      marker=marker, linestyle="none", markersize=6.5,
                      markeredgecolor="white", label=label)
               for family, (marker, label) in MARKERS.items()]
    handles.append(Line2D([], [], color=REFERENCE_COLOR, marker="*",
                          linestyle="none", markersize=11,
                          label="uncompressed fp32 message (per backbone colour)"))
    handles.append(Line2D([], [], color=INK, linewidth=1.6, alpha=0.6,
                          label="per-backbone Pareto frontier"))
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                  fontsize=9.5, bbox_to_anchor=(0.5, -0.05))
    controls = "controls PASSED" if discovery["controls"]["passed"] else \
        "CONTROL FAILURE — see the E07 report"
    figure.suptitle(
        "Matched bit budget: G2-150k executable-phase message vs R0-500k latent "
        f"message\nfrozen checkpoints, identical draws, "
        f"{discovery['arms'][primary]['continuous']['clusters']} batch clusters, "
        f"95% CI — {controls}", fontsize=12.5, y=1.0)
    figure.tight_layout(rect=(0, 0.11, 1, 0.93))

    for suffix in ("png", "pdf"):
        path = root / f"matched_budget_frontier.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[plot] {path}")


if __name__ == "__main__":
    main()
