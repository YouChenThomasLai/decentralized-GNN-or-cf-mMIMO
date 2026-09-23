"""Matched short R0 training-step benchmark for the E01 progress report."""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/decentralized_ris"))
from model import BaselineNet
from simulation import ChannelSimulator


LEGACY = ROOT / "code/snapshot_network_scaling/stage0"
sys.path.insert(0, str(LEGACY))
from model_2 import node_update  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_step(model, simulator, optimizer, device):
    features, edges, masks, direct, _ = simulator.training_batch(8, 0.1, 0.1)
    features, edges, direct = (tensor.to(device) for tensor in (features, edges, direct))
    optimizer.zero_grad(set_to_none=True)
    beamformer, phase = model(features, edges, masks, direct, training=True)
    loss, _, _ = simulator.loss(beamformer, phase, device)
    loss.backward()
    optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps must be positive and warmup nonnegative")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)
    pmax = 10 ** ((15 - 30) / 10)
    current = BaselineNet(2, 30, 4, 6, pmax, 64, 5, device).to(device)
    legacy = node_update(2, 30, 4, 6, pmax, 2, 64, 5, device).to(device)
    mismatch = legacy.load_state_dict(current.state_dict(), strict=False)
    assert not mismatch.unexpected_keys
    assert all(".fc." in key or ".edge_update." in key for key in mismatch.missing_keys)

    simulator = ChannelSimulator(2, 30, 4, 8)
    features, edges, masks, direct, _ = simulator.training_batch(8, 0.1, 0.1)
    features, edges, direct = (tensor.to(device) for tensor in (features, edges, direct))
    with torch.no_grad():
        old = legacy(features.clone(), edges.clone(), masks.copy(), direct.clone())
        new = current(features.clone(), edges.clone(), masks.copy(), direct.clone())
    errors = [float((a - b).abs().max()) for a, b in zip(old, new)]
    assert errors[0] < 1e-5 and errors[1] < 1e-5, errors

    timings = {}
    for label, model in (("legacy", legacy), ("current", current)):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        np.random.seed(1234)
        torch.manual_seed(1234)
        for _ in range(args.warmup):
            training_step(model, simulator, optimizer, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.steps):
            training_step(model, simulator, optimizer, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings[label] = (time.perf_counter() - started) / args.steps
        print(f"{label}: {timings[label]:.6f} seconds/step", flush=True)

    result = {
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "host": platform.node(),
        "torch": torch.__version__,
        "batch_size": 8,
        "training_seed": 1234,
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "scope": "training_batch + forward + sum_rate_loss + backward + Adam; no evaluation or checkpoint I/O",
        "max_abs_output_error": {"beamformer": errors[0], "phase": errors[1]},
        "seconds_per_step": timings,
        "relative_speed": timings["legacy"] / timings["current"],
        "source_sha256": {
            "legacy_model": sha256(LEGACY / "model_2.py"),
            "legacy_data": sha256(LEGACY / "data.py"),
            "legacy_channels": sha256(LEGACY / "utils_return_indivial_rates.py"),
            "current_model": sha256(ROOT / "code/decentralized_ris/model.py"),
            "current_simulation": sha256(ROOT / "code/decentralized_ris/simulation.py"),
            "current_rates": sha256(ROOT / "code/decentralized_ris/rates.py"),
            "benchmark": sha256(Path(__file__)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
