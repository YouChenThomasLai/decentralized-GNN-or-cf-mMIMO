"""E14: pre-declared statistics for the rate-matched visibility pair."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def welch(a, b):
    diff = float(a.mean() - b.mean())
    se = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return {"difference": diff, "se": se,
            "ci95": [diff - 1.96 * se, diff + 1.96 * se],
            "z": diff / se if se else float("nan")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--match_tolerance", type=float, default=1.5,
                        help="declared holdout centralized-rate control")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    docs, gaps, paired = {}, {}, {}
    for layout in ("mlow", "mhigh"):
        docs[layout] = json.loads((root / f"{layout}.json").read_text())
        arrays = np.load(root / f"{layout}.npz")
        gaps[layout] = {
            model: 1.0 - arrays[f"{model}_decentralized"] / arrays[f"{model}_centralized"]
            for model in ("g2", "r0")
        }
        paired[layout] = gaps[layout]["r0"] - gaps[layout]["g2"]

    control = {}
    for model in ("g2", "r0"):
        rates = [docs[layout]["methods"][model]["rates"]["centralized"]["mean"]
                 for layout in ("mlow", "mhigh")]
        control[model] = {"mlow": rates[0], "mhigh": rates[1],
                          "difference": rates[1] - rates[0]}
    worst = max(abs(value["difference"]) for value in control.values())
    control["worst_absolute_difference"] = worst
    control["tolerance"] = args.match_tolerance
    control["passed"] = bool(worst <= args.match_tolerance)

    summary = {
        "experiment": "E14 rate-matched visibility continuation",
        "eval_seed": docs["mlow"]["eval_seed"],
        "samples": docs["mlow"]["samples"],
        "holdout_match_control": control,
        "visibility": {layout: docs[layout]["visibility"]
                       ["paper_visible_fraction_of_global"]["mean"]
                       for layout in docs},
        "gap_mean": {layout: {model: float(value.mean())
                              for model, value in gaps[layout].items()}
                     for layout in gaps},
        "primary_gap_increase_at_low_visibility": {
            model: welch(gaps["mlow"][model], gaps["mhigh"][model])
            for model in ("g2", "r0")
        },
        "secondary_r0_minus_g2_gap_increase": welch(paired["mlow"], paired["mhigh"]),
        "within_layout_paired_g2_minus_r0_retention": {
            layout: docs[layout]["contrasts"]["g2_minus_r0_retention_per_batch"]
            for layout in docs
        },
    }
    if not control["passed"]:
        summary["status"] = ("FAILED declared holdout match control; the rate-matched "
                             "claim does not hold and no reselection is permitted")
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"holdout match control: worst |dC| = {worst:.4f} bps/Hz, "
          f"tolerance {args.match_tolerance} -> {'pass' if control['passed'] else 'FAIL'}")
    for layout in ("mlow", "mhigh"):
        print(f"{layout.upper():6} V={summary['visibility'][layout]:.4f} "
              f"gap_g2={summary['gap_mean'][layout]['g2']*100:5.2f}% "
              f"gap_r0={summary['gap_mean'][layout]['r0']*100:5.2f}%")
    for model in ("g2", "r0"):
        item = summary["primary_gap_increase_at_low_visibility"][model]
        print(f"{model.upper():3} gap(MLOW)-gap(MHIGH) = {item['difference']*100:+.2f} pp "
              f"[{item['ci95'][0]*100:+.2f},{item['ci95'][1]*100:+.2f}] z={item['z']:+.2f}")
    item = summary["secondary_r0_minus_g2_gap_increase"]
    print(f"R0 minus G2 increase = {item['difference']*100:+.2f} pp "
          f"[{item['ci95'][0]*100:+.2f},{item['ci95'][1]*100:+.2f}] z={item['z']:+.2f}")
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
