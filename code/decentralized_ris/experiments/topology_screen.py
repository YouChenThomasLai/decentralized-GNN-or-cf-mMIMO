"""Frozen G2/R0 topology screen and shared-UE CSI delivery ledger."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evaluate import (build_model, evaluate_model, resolve_device,
                      seed_everything, temporary_seed)
from model import load_checkpoint
from simulation import ChannelSimulator


ROOT = Path(__file__).resolve().parents[3]
RUNS = {
    "g2": (ROOT / "artifacts/decentralized_ris/e06_graph_energy_training/"
            "g2_long_training/g2/iter150000", "iter150000.pt"),
    "r0": (ROOT / "artifacts/decentralized_ris/e01_baseline_training/"
            "iter500000/M2_N30_L4_K8_P15.0_iter350000_seed0/run0",
            "model_final_run0.pt"),
}


def clustered(values, batch_size):
    values = np.asarray(values, dtype=float).reshape(-1)
    groups = values.reshape(-1, batch_size).mean(axis=1)
    se = groups.std(ddof=1) / np.sqrt(groups.size) if groups.size > 1 else 0.0
    return {"mean": float(values.mean()), "clustered_se": float(se),
            "ci95": [float(values.mean() - 1.96 * se),
                     float(values.mean() + 1.96 * se)]}


def csi_values(masks, n_antennas, n_ris, n_elements):
    """One AP-to-AP copy per ordered serving-AP pair and shared UE."""
    served = masks.sum(axis=1)
    copies = (served * (served - 1)).sum(axis=1)
    values = copies * 2 * n_antennas * (n_ris * n_elements + 1)
    return copies, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True)
    parser.add_argument("--layout", help="fixed JSON coordinates; omitted for T0")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--eval_seed", type=int, default=20260920)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if (args.topology == "T0") != (args.layout is None):
        parser.error("T0 uses the canonical layout; other topologies require --layout")
    layout = None
    if args.layout:
        with open(args.layout, encoding="utf-8") as handle:
            layout = json.load(handle)
        if layout["id"] != args.topology:
            parser.error("layout id and --topology disagree")

    device = resolve_device(args.device)
    config = None
    results, arrays = {}, {}
    for name, (run, checkpoint_name) in RUNS.items():
        with open(run / "summary.json", encoding="utf-8") as handle:
            current = json.load(handle)["config"]
        if config is not None:
            for key in ("M", "N", "L", "K", "AP", "batch_size",
                        "assoc_threshold", "pmax_dbm", "seed"):
                if current[key] != config[key]:
                    raise ValueError(f"{name} has different {key}")
        config = current
        seed_everything(config["seed"])
        simulator = ChannelSimulator(
            config["M"], config["N"], config["L"], config["batch_size"],
            n_ap=config["AP"], layout=layout,
        )
        model = build_model(config, simulator, device)
        checkpoint = run / "checkpoints" / checkpoint_name
        if not checkpoint.exists():
            checkpoint = run / "models" / checkpoint_name
        load_checkpoint(model, str(checkpoint), device)
        with temporary_seed(args.eval_seed):
            batches, unit_error, _ = evaluate_model(
                model, simulator, config["K"], config["assoc_threshold"],
                device, args.samples, config["batch_size"],
            )
        results[name] = {
            "checkpoint": str(checkpoint), "unit_modulus_error": unit_error,
            "rates": {key: clustered(values, 1) for key, values in batches.items()},
            "retention_ratio_of_means": float(
                batches["decentralized"].mean() / batches["centralized"].mean()
            ),
            "retention_per_batch": clustered(
                batches["decentralized"] / batches["centralized"], 1
            ),
        }
        arrays.update({f"{name}_{key}": values for key, values in batches.items()})
        print(f"[{args.topology}] {name}: "
              f"paper={batches['decentralized'].mean():.4f}, "
              f"2-bit={batches['decentralized_discrete'].mean():.4f}", flush=True)

    seed_everything(config["seed"])
    simulator = ChannelSimulator(
        config["M"], config["N"], config["L"], config["batch_size"],
        n_ap=config["AP"], layout=layout,
    )
    copies, values, paper_visible, own_visible = [], [], [], []
    with temporary_seed(args.eval_seed):
        for _ in range(args.samples // config["batch_size"]):
            simulator.training_batch(
                config["K"], config["assoc_threshold"],
                config["assoc_threshold"],
            )
            masks = np.stack([station.user_mask for station in
                              simulator.base_stations], axis=1)
            block_copies, block_values = csi_values(
                masks, config["M"], config["L"], config["N"]
            )
            served = masks.sum(axis=1)
            paper_visible.extend((served * served).sum(axis=1) / config["AP"])
            own_visible.extend(served.sum(axis=1) / config["AP"])
            copies.extend(block_copies)
            values.extend(block_values)
    copies, values = np.asarray(copies), np.asarray(values)
    paper_visible, own_visible = np.asarray(paper_visible), np.asarray(own_visible)
    proposal_values = config["AP"] * config["L"] * (config["N"] + 1)
    r0_proposal_values = config["AP"] * config["L"] * 4 * config["N"]
    processed_values_per_copy = (
        config["L"] * (2 * config["M"] * (config["N"] + 1) + 1) + 1
    )
    csi = {
        "definition": "ordered source-AP to target-AP copy for each jointly served UE",
        "values_per_copy": 2 * config["M"] * (config["L"] * config["N"] + 1),
        "copies": clustered(copies, config["batch_size"]),
        "copies_max": int(copies.max()),
        "zero_copy_fraction": float((copies == 0).mean()),
        "ap_to_ap_values": clustered(values, config["batch_size"]),
        "ap_to_ap_bits_fp32": clustered(values * 32, config["batch_size"]),
        "proposal_bits_fp32": proposal_values * 32,
        "r0_proposal_bits_fp32": r0_proposal_values * 32,
        "g2_total_bits_fp32": clustered((values + proposal_values) * 32,
                                           config["batch_size"]),
        "r0_total_bits_fp32": clustered((values + r0_proposal_values) * 32,
                                           config["batch_size"]),
        "processed_input_values_per_copy": processed_values_per_copy,
        "processed_input_g2_total_bits_fp32": clustered(
            (copies * processed_values_per_copy + proposal_values) * 32,
            config["batch_size"],
        ),
        "processed_input_note": (
            "descriptive exact model-input tensor alternative: per RIS, "
            "2M(N+1) normalized features plus one edge, and one direct edge "
            "per AP-UE link; excludes mask/routing metadata"
        ),
        "direct_exchange_rounds": 2,
        "cpu_relay_total_bits_fp32": clustered(
            (2 * values + proposal_values) * 32, config["batch_size"]),
        "cpu_relay_rounds": 3,
        "boundary": "AP-to-AP shared-UE CSI and AP-to-CPU G2 proposals; "
                    "UE-to-AP acquisition, association metadata, and CPU-to-RIS "
                    "actuation excluded",
    }
    if not np.any(copies):
        csi["direct_exchange_rounds"] = 1
        csi["cpu_relay_rounds"] = 1
    contrasts = {}
    for phase in ("decentralized", "decentralized_discrete"):
        contrasts[f"g2_minus_r0_{phase}"] = clustered(
            arrays[f"g2_{phase}"] - arrays[f"r0_{phase}"], 1
        )
    contrasts["g2_minus_r0_retention_per_batch"] = clustered(
        arrays["g2_decentralized"] / arrays["g2_centralized"]
        - arrays["r0_decentralized"] / arrays["r0_centralized"], 1
    )
    visibility = {
        "paper_visible_ap_ue_links_per_ap": clustered(
            paper_visible, config["batch_size"]
        ),
        "own_visible_ap_ue_links_per_ap": clustered(
            own_visible, config["batch_size"]
        ),
        "paper_visible_fraction_of_global": clustered(
            paper_visible / (config["AP"] * config["K"]),
            config["batch_size"],
        ),
    }
    output = {
        "experiment": "E14", "topology": args.topology,
        "layout": layout or {"ap_locations": simulator.ap_locations.tolist(),
                             "ris_locations": simulator.ris_locations.tolist()},
        "config": {key: config[key] for key in ("M", "N", "L", "K", "AP",
                                              "batch_size", "assoc_threshold",
                                              "pmax_dbm", "seed")},
        "eval_seed": args.eval_seed, "samples": args.samples,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda"
                       else "CPU",
        "command": sys.argv,
        "methods": results, "contrasts": contrasts, "visibility": visibility,
        "csi_delivery": csi,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    arrays.update(csi_copies=copies, csi_values=values)
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
