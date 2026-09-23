"""E02: paired G2/R0 response to removing shared-UE cross-AP CSI."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import build_model, resolve_device, seed_everything, temporary_seed
from experiments.topology_screen import RUNS, clustered
from model import load_checkpoint
from simulation import ChannelSimulator


def evaluate(run, checkpoint_name, samples, eval_seed, device):
    with open(run / "summary.json", encoding="utf-8") as handle:
        config = json.load(handle)["config"]
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
    rates = {name: [] for name in ("centralized", "paper", "own")}
    visible = {name: [] for name in ("paper", "own")}
    with temporary_seed(eval_seed), torch.no_grad():
        for _ in range(samples // config["batch_size"]):
            features, edges, masks, direct, _ = simulator.training_batch(
                config["K"], config["assoc_threshold"],
                config["assoc_threshold"],
            )
            beamformer, phase = model.centralized(
                features.to(device), edges.to(device), masks, direct.to(device)
            )
            rates["centralized"].append(float(simulator.loss(beamformer, phase, device)[1]))
            features, edges, masks, direct = simulator.decentralized_batch(
                config["K"], config["assoc_threshold"],
                config["assoc_threshold"], regenerate_channels=False,
            )
            features = [value.to(device) for value in features]
            edges = [value.to(device) for value in edges]
            direct = [value.to(device) for value in direct]
            service = np.stack(masks, axis=1).astype(int)
            count = service.sum(axis=1)
            visible["paper"].extend((count * count).sum(axis=1) / config["AP"])
            visible["own"].extend(count.sum(axis=1) / config["AP"])
            for name, cross in (("paper", True), ("own", False)):
                beamformer, phase = model.decentralized(
                    features, edges, masks, direct, include_cross_ap_csi=cross
                )
                rates[name].append(float(simulator.loss(beamformer, phase, device)[1]))
    return (
        {key: np.asarray(value) for key, value in rates.items()},
        {key: float(np.mean(value)) for key, value in visible.items()},
        str(checkpoint),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--eval_seed", type=int, default=20260920)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.samples < 16 or args.samples % 8:
        parser.error("samples must be at least two batches and divisible by eight")
    device = resolve_device(args.device)
    raw, summary = {}, {}
    for name, (run, checkpoint_name) in RUNS.items():
        rates, visible, checkpoint = evaluate(
            run, checkpoint_name, args.samples, args.eval_seed, device
        )
        raw[name] = rates
        summary[name] = {
            "checkpoint": checkpoint, "visible_ap_ue_nodes_per_ap": visible,
            "rates": {key: clustered(value, 1) for key, value in rates.items()},
            "centralized_minus_paper": clustered(
                rates["centralized"] - rates["paper"], 1
            ),
            "centralized_minus_own": clustered(
                rates["centralized"] - rates["own"], 1
            ),
            "paper_minus_own": clustered(rates["paper"] - rates["own"], 1),
        }
        print(name, {key: round(value.mean(), 4) for key, value in rates.items()},
              flush=True)
    contrast = clustered(
        raw["g2"]["paper"] - raw["g2"]["own"]
        - raw["r0"]["paper"] + raw["r0"]["own"], 1
    )
    output = {
        "experiment": "E02 continuation", "topology": "T0",
        "eval_seed": args.eval_seed, "samples": args.samples, "batch_size": 8,
        "device": str(device), "models": summary,
        "g2_minus_r0_visibility_penalty": contrast,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    np.savez_compressed(args.out.with_suffix(".npz"), **{
        f"{name}_{mode}": values for name, rates in raw.items()
        for mode, values in rates.items()
    })
    print("contrast", contrast)


if __name__ == "__main__":
    main()
