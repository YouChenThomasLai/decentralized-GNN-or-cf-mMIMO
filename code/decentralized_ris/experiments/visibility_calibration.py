"""E14: mask-only AP-ring calibration for the shared-UE visibility pair."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evaluate import seed_everything, temporary_seed
from experiments.topology_screen import RUNS, csi_values
from rates import ring_locations
from simulation import ChannelSimulator


def ring_layout(identifier, radius, n_ap, n_ris, ris_radius):
    return {
        "id": identifier,
        "ap_locations": ring_locations(n_ap, radius).tolist(),
        "ris_locations": ring_locations(n_ris, ris_radius).tolist(),
    }


def measure(config, layout, samples, seed):
    """Association masks only; no model, channel realization use, or rate."""
    seed_everything(config["seed"])
    simulator = ChannelSimulator(
        config["M"], config["N"], config["L"], config["batch_size"],
        n_ap=config["AP"], layout=layout,
    )
    paper, own, copies = [], [], []
    with temporary_seed(seed):
        for _ in range(samples // config["batch_size"]):
            simulator.training_batch(
                config["K"], config["assoc_threshold"], config["assoc_threshold"],
            )
            masks = np.stack([station.user_mask for station in
                              simulator.base_stations], axis=1)
            served = masks.sum(axis=1)
            paper.extend((served * served).sum(axis=1) / config["AP"])
            own.extend(served.sum(axis=1) / config["AP"])
            copies.extend(csi_values(masks, config["M"], config["L"], config["N"])[0])
    paper = np.asarray(paper, dtype=float)
    return {
        "visible_fraction": float(paper.mean() / (config["AP"] * config["K"])),
        "paper_visible_links_per_ap": float(paper.mean()),
        "own_visible_links_per_ap": float(np.mean(own)),
        "cross_ap_copies": float(np.mean(copies)),
        "layout": layout,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radii", type=float, nargs="+",
                        default=[140, 170, 200, 280, 350])
    parser.add_argument("--reference_radius", type=float, default=200,
                        help="canonical T0 AP ring, excluded from selection")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--min_span", type=float, default=0.10)
    parser.add_argument("--holdout_seed", type=int, default=20260924,
                        help="recorded for provenance; no rate is evaluated here")
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--low_layout", required=True)
    parser.add_argument("--high_layout", required=True)
    args = parser.parse_args()

    run, _ = RUNS["g2"]
    with open(run / "summary.json", encoding="utf-8") as handle:
        config = json.load(handle)["config"]
    ris_radius = float(np.linalg.norm(
        ChannelSimulator(config["M"], config["N"], config["L"],
                         config["batch_size"], n_ap=config["AP"]).ris_locations[0]
    ))

    results = {}
    for radius in args.radii:
        layout = ring_layout(f"R{radius:g}", radius, config["AP"], config["L"],
                             ris_radius)
        results[f"{radius:g}"] = measure(config, layout, args.samples, args.seed)
        print(f"[calibration] AP ring {radius:g} m: "
              f"V={results[f'{radius:g}']['visible_fraction']:.5f}", flush=True)

    candidates = sorted(
        ((value["visible_fraction"], float(key)) for key, value in results.items()
         if float(key) != args.reference_radius)
    )
    if not candidates:
        raise ValueError("selection needs at least one non-reference radius")
    low, high = candidates[0], candidates[-1]
    span = high[0] - low[0]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "seed": args.seed, "samples": args.samples,
        "candidate_ap_ring_radii_m": [float(r) for r in args.radii],
        "command": sys.argv, "results": results,
    }, indent=2) + "\n", encoding="utf-8")

    selection = {
        "selection_rule": "minimum/maximum mask-only V among non-T0 candidates",
        "calibration_seed": args.seed, "calibration_samples": args.samples,
        "low_radius_m": low[1], "high_radius_m": high[1],
        "low_visible_fraction": low[0], "high_visible_fraction": high[0],
        "span": span, "threshold_met": bool(span >= args.min_span),
        "holdout_seed": args.holdout_seed,
        "layouts": [args.low_layout, args.high_layout],
    }
    Path(args.selection).write_text(json.dumps(selection, indent=2) + "\n",
                                    encoding="utf-8")
    if not selection["threshold_met"]:
        print(f"[stop] visibility span {span:.4f} below {args.min_span}; "
              "report insufficient span instead of evaluating rates")
        return

    for name, (radius, path) in (("VLOW", (low[1], args.low_layout)),
                                 ("VHIGH", (high[1], args.high_layout))):
        layout = ring_layout(name, radius, config["AP"], config["L"], ris_radius)
        layout["description"] = (
            f"Mask-only calibration selected the {radius:g} m AP ring for "
            f"{'low' if name == 'VLOW' else 'high'} shared-UE visibility"
        )
        target = Path(path)
        if target.exists() and json.loads(target.read_text()) == layout:
            print(f"[keep] {target} already holds the selected layout")
            continue
        target.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
