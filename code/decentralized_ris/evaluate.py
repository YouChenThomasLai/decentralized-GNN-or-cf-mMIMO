"""Shared paired evaluation for baseline and RIS-action variants."""

import argparse
from contextlib import contextmanager
import glob
import json
import os
import random

import numpy as np
import torch

from model import load_checkpoint
from rates import quantize_phase, random_phase_like
from simulation import ChannelSimulator
from variants import VariantNet


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


@contextmanager
def temporary_seed(seed):
    """Use deterministic evaluation data without perturbing training RNG state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seed_everything(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def resolve_device(name):
    return torch.device(name if torch.cuda.is_available() else "cpu")


def evaluate_model(
    model,
    simulator,
    users_per_ap,
    threshold,
    device,
    samples,
    batch_size,
    num_bits=2,
):
    if samples <= 0 or samples % batch_size:
        raise ValueError("samples must be a positive multiple of batch_size")

    keys = (
        "centralized",
        "centralized_discrete",
        "centralized_random_phase",
        "decentralized",
        "decentralized_discrete",
        "decentralized_random_phase",
    )
    records = {key: [] for key in keys}
    unit_error = {"centralized": 0.0, "decentralized": 0.0}
    per_user = []
    model.eval()

    with torch.no_grad():
        for _ in range(samples // batch_size):
            features, edges, masks, direct, _ = simulator.training_batch(
                users_per_ap, threshold, threshold
            )
            features, edges, direct = (
                features.to(device), edges.to(device), direct.to(device)
            )
            beamformer, phase = model.centralized(features, edges, masks, direct)
            unit_error["centralized"] = max(
                unit_error["centralized"],
                float((phase.norm(dim=-1) - 1).abs().max()),
            )
            _, rate, _ = simulator.loss(beamformer, phase, device)
            records["centralized"].append(float(rate))
            _, rate, _ = simulator.loss(
                beamformer, quantize_phase(phase, num_bits), device
            )
            records["centralized_discrete"].append(float(rate))
            _, rate, _ = simulator.loss(
                beamformer, random_phase_like(phase), device
            )
            records["centralized_random_phase"].append(float(rate))

            features, edges, masks, direct = simulator.decentralized_batch(
                users_per_ap, threshold, threshold, regenerate_channels=False
            )
            features = [tensor.to(device) for tensor in features]
            edges = [tensor.to(device) for tensor in edges]
            direct = [tensor.to(device) for tensor in direct]
            beamformer, phase = model.decentralized(features, edges, masks, direct)
            unit_error["decentralized"] = max(
                unit_error["decentralized"],
                float((phase.norm(dim=-1) - 1).abs().max()),
            )
            _, rate, user_rates = simulator.loss(beamformer, phase, device)
            records["decentralized"].append(float(rate))
            per_user.append(user_rates.detach().cpu().numpy())
            _, rate, _ = simulator.loss(
                beamformer, quantize_phase(phase, num_bits), device
            )
            records["decentralized_discrete"].append(float(rate))
            _, rate, _ = simulator.loss(
                beamformer, random_phase_like(phase), device
            )
            records["decentralized_random_phase"].append(float(rate))

    return (
        {key: np.asarray(values) for key, values in records.items()},
        unit_error,
        np.mean(np.stack(per_user), axis=0),
    )


def build_model(config, simulator, device):
    pmax = 10 ** ((config["pmax_dbm"] - 30) / 10)
    consensus = config.get("consensus") or (
        "wreduce" if config["arch"] == "r0" else "equal"
    )
    return VariantNet(
        config["M"],
        config["N"],
        config["L"],
        config["D"],
        pmax,
        config["ch"],
        config["AP"],
        device,
        arch=config["arch"],
        identity=config["identity"],
        consensus=consensus,
        tau=config["tau"],
        ris_loc=simulator.ris_locations,
        users_per_ap=config["K"],
    ).to(device)


def checkpoint_path(run_dir, name):
    candidates = (
        os.path.join(run_dir, "checkpoints", name),
        os.path.join(run_dir, "models", name),
    )
    return next((path for path in candidates if os.path.exists(path)), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--samples", type=int, default=3200)
    parser.add_argument("--eval_seed", type=int, default=20260914)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="results_ris_action/final_screening.json")
    args = parser.parse_args()

    run_dirs = sorted(
        {path for pattern in args.runs for path in glob.glob(pattern) if os.path.isdir(path)}
    )
    if not run_dirs:
        raise SystemExit(f"no run directories matched {args.runs}")

    device = resolve_device(args.device)
    results = {}
    paired = {}
    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as handle:
            summary = json.load(handle)
        config = summary["config"]
        checkpoint = checkpoint_path(run_dir, args.checkpoint)
        if checkpoint is None:
            print(f"[skip] {run_dir}: no {args.checkpoint}")
            continue

        seed_everything(config["seed"])
        simulator = ChannelSimulator(
            config["M"],
            config["N"],
            config["L"],
            config["batch_size"],
            n_ap=config["AP"],
        )
        model = build_model(config, simulator, device)
        load_checkpoint(model, checkpoint, device)
        with temporary_seed(args.eval_seed):
            batches, unit_error, per_user = evaluate_model(
                model,
                simulator,
                config["K"],
                config.get("assoc_threshold", 0.1),
                device,
                args.samples,
                config["batch_size"],
            )

        tag = summary["tag"]
        results[tag] = {
            "run_dir": run_dir,
            "checkpoint": args.checkpoint,
            "samples": args.samples,
            "eval_seed": args.eval_seed,
            "unit_modulus_error": unit_error,
            "model": summary["model"],
            "best_val_iteration": summary["best_val"].get("iteration"),
            "per_user_rate": per_user.tolist(),
            **{key: float(values.mean()) for key, values in batches.items()},
            **{
                f"{key}_sem": float(values.std(ddof=1) / np.sqrt(len(values)))
                for key, values in batches.items()
            },
        }
        paired[tag] = batches
        print(
            f"[eval] {tag:28s} cen={batches['centralized'].mean():.5f} "
            f"dec={batches['decentralized'].mean():.5f} "
            f"dec_2bit={batches['decentralized_discrete'].mean():.5f}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    paired_path = os.path.splitext(args.out)[0] + "_paired.npz"
    np.savez(
        paired_path,
        **{
            f"{tag}__{key}": values
            for tag, batches in paired.items()
            for key, values in batches.items()
        },
    )
    print(f"[done] {args.out} and {paired_path}")


if __name__ == "__main__":
    main()
