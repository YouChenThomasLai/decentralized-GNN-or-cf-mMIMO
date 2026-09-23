"""E14: select two layouts with matched centralized rate but different shared-UE visibility."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evaluate import build_model, resolve_device, seed_everything, temporary_seed
from experiments.topology_screen import RUNS
from experiments.visibility_calibration import ring_layout
from model import load_checkpoint
from simulation import ChannelSimulator


def load_models(device):
    """Each checkpoint is built from its own run config, as in the topology screen."""
    models, shared = {}, None
    for name, (run, checkpoint_name) in RUNS.items():
        with open(run / "summary.json", encoding="utf-8") as handle:
            config = json.load(handle)["config"]
        if shared is not None:
            for key in ("M", "N", "L", "K", "AP", "batch_size",
                        "assoc_threshold", "pmax_dbm", "seed"):
                if config[key] != shared[key]:
                    raise ValueError(f"{name} has different {key}")
        shared = config
        seed_everything(config["seed"])
        simulator = ChannelSimulator(
            config["M"], config["N"], config["L"], config["batch_size"],
            n_ap=config["AP"],
        )
        model = build_model(config, simulator, device)
        checkpoint = run / "checkpoints" / checkpoint_name
        if not checkpoint.exists():
            checkpoint = run / "models" / checkpoint_name
        load_checkpoint(model, str(checkpoint), device)
        model.eval()
        models[name] = model
    return models, shared


def probe(config, layout, models, samples, seed, device):
    """Mask-only visibility and centralized-only rate on the calibration seed."""
    seed_everything(config["seed"])
    simulator = ChannelSimulator(
        config["M"], config["N"], config["L"], config["batch_size"],
        n_ap=config["AP"], layout=layout,
    )
    visible, rates = [], {name: [] for name in models}
    with temporary_seed(seed), torch.no_grad():
        for _ in range(samples // config["batch_size"]):
            features, edges, masks, direct, _ = simulator.training_batch(
                config["K"], config["assoc_threshold"], config["assoc_threshold"],
            )
            served = np.stack([station.user_mask for station
                               in simulator.base_stations], axis=1).sum(axis=1)
            visible.extend((served * served).sum(axis=1) / config["AP"])
            for name, model in models.items():
                beamformer, phase = model.centralized(
                    features.to(device), edges.to(device), masks, direct.to(device)
                )
                rates[name].append(float(simulator.loss(beamformer, phase, device)[1]))
    visible = np.asarray(visible, dtype=float)
    return {
        "visible_fraction": float(visible.mean() / (config["AP"] * config["K"])),
        "centralized_rate": {name: float(np.mean(value))
                             for name, value in rates.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radii", type=float, nargs="+",
                        default=[120, 140, 170, 200, 240, 280, 350])
    parser.add_argument("--user_radii", type=float, nargs="+",
                        default=[20, 40, 60, 80, 100, 140, 180])
    parser.add_argument("--max_user_ratio", type=float, default=0.9,
                        help="keep users inside the AP ring")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--max_rate_gap", type=float, default=1.0,
                        help="largest accepted centralized-rate mismatch, bps/Hz")
    parser.add_argument("--min_span", type=float, default=0.20)
    parser.add_argument("--holdout_seed", type=int, default=20260927)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--low_layout", required=True)
    parser.add_argument("--high_layout", required=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    models, config = load_models(device)
    ris_radius = float(np.linalg.norm(
        ChannelSimulator(config["M"], config["N"], config["L"],
                         config["batch_size"], n_ap=config["AP"]).ris_locations[0]
    ))

    points = {}
    for radius, user_radius in itertools.product(args.radii, args.user_radii):
        if user_radius > args.max_user_ratio * radius:
            continue
        key = f"r{radius:g}_u{user_radius:g}"
        layout = ring_layout(key.upper(), radius, config["AP"], config["L"],
                             ris_radius)
        layout["user_radius"] = user_radius
        points[key] = probe(config, layout, models, args.samples, args.seed, device)
        points[key].update(ap_ring_radius_m=radius, user_radius_m=user_radius)
        print(f"[probe] {key}: V={points[key]['visible_fraction']:.4f} "
              f"C_g2={points[key]['centralized_rate']['g2']:.4f} "
              f"C_r0={points[key]['centralized_rate']['r0']:.4f}", flush=True)

    eligible = []
    for low, high in itertools.permutations(points, 2):
        span = points[high]["visible_fraction"] - points[low]["visible_fraction"]
        gaps = [abs(points[high]["centralized_rate"][name]
                    - points[low]["centralized_rate"][name]) for name in models]
        if span >= args.min_span and max(gaps) <= args.max_rate_gap:
            eligible.append((span, -max(gaps), -points[low]["ap_ring_radius_m"],
                             low, high))
    eligible.sort(reverse=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "seed": args.seed, "samples": args.samples,
        "ap_ring_radii_m": [float(r) for r in args.radii],
        "user_radii_m": [float(u) for u in args.user_radii],
        "max_user_ratio": args.max_user_ratio,
        "device": str(device), "command": sys.argv, "points": points,
        "eligible_pair_count": len(eligible),
    }, indent=2) + "\n", encoding="utf-8")

    selection = {
        "selection_rule": "maximum visibility span among pairs whose centralized rate "
                          "matches within the declared tolerance for both models",
        "calibration_seed": args.seed, "calibration_samples": args.samples,
        "max_rate_gap_bps_hz": args.max_rate_gap, "min_span": args.min_span,
        "holdout_seed": args.holdout_seed,
        "eligible_pair_count": len(eligible),
        "threshold_met": bool(eligible),
    }
    if not eligible:
        Path(args.selection).write_text(json.dumps(selection, indent=2) + "\n",
                                        encoding="utf-8")
        print("[stop] no rate-matched pair in the declared grid; report the failure "
              "instead of widening the grid or loosening the tolerance")
        return

    span, neg_gap, _, low, high = eligible[0]
    selection.update(
        low_point=low, high_point=high, span=span, rate_gap=-neg_gap,
        low_visible_fraction=points[low]["visible_fraction"],
        high_visible_fraction=points[high]["visible_fraction"],
        low_centralized_rate=points[low]["centralized_rate"],
        high_centralized_rate=points[high]["centralized_rate"],
        layouts=[args.low_layout, args.high_layout],
    )
    Path(args.selection).write_text(json.dumps(selection, indent=2) + "\n",
                                    encoding="utf-8")

    for name, key, path in (("MLOW", low, args.low_layout),
                            ("MHIGH", high, args.high_layout)):
        point = points[key]
        layout = ring_layout(name, point["ap_ring_radius_m"], config["AP"],
                             config["L"], ris_radius)
        layout["user_radius"] = point["user_radius_m"]
        layout["description"] = (
            f"Rate-matched calibration selected AP ring {point['ap_ring_radius_m']:g} m "
            f"with user radius {point['user_radius_m']:g} m for "
            f"{'low' if name == 'MLOW' else 'high'} shared-UE visibility"
        )
        target = Path(path)
        if target.exists() and json.loads(target.read_text()) == layout:
            print(f"[keep] {target} already holds the selected layout")
            continue
        target.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
