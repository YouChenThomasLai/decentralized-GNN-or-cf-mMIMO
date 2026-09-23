"""E05 extension: how close is the local-energy weight to a strong fitted reference?

The deployed rule fuses the AP phase proposals with

    theta_{r,n} = Pi( sum_l Ebar_{l,r} p_{l,r,n} ),
    E_{l,r}     = sum_{k in K_l} ||H_{(l,r,k)}||_F^2 ,

and the method report argues that this is the maximum-likelihood fusion when each
AP's proposal concentration is proportional to E_{l,r}.  That argument is a
modeling assumption about proposals produced by a trained network, not a theorem
about the network, so this script measures the assumption's consequence directly.

Beamformer and proposals are frozen.  The *only* thing that changes between arms
is the consensus weight tensor:

    equal    w_{l,r} = 1                       (no information)
    energy   w_{l,r} = E_{l,r}                 (the deployed parameter-free rule)
    oracle   w_{l,r} = fitted against sum rate (non-deployable reference)

The oracle-style arm optimizes L*R non-negative scalars per channel sample against
the true sum rate, so it sees the realized rate and is not a method.  Adam is a
finite-budget local search, not a certified global optimizer; the arm therefore
reports the best value found under the declared search rather than an upper bound.

Two questions follow.  How much of the equal-to-best-found gap does the parameter-free
energy weight already recover, and does E_{l,r} rank the APs the way the oracle
does?  Nothing here trains the network; one frozen checkpoint, training seed 0,
fixed evaluation seeds.

Run from `code/decentralized_ris/`:

    python -m experiments.consensus_weight_oracle \
        --run ../../artifacts/decentralized_ris/e06_graph_energy_training/g2_long_training/g2/iter150000 \
        --checkpoint iter150000.pt --samples 400 --eval_seed 20260915
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import build_model, checkpoint_path, resolve_device, seed_everything
from experiments.mrc_proxy_diagnostic import (
    ControlFailure,
    cluster_summary,
    weighted_consensus,
)
from model import load_checkpoint
from rates import RatePrecompute, quantize_phase
from simulation import ChannelSimulator


ARMS = ("equal", "energy", "oracle")
PHASES = ("continuous", "2bit")

# Declared before the locked evaluation.  The repository history does not contain
# an immutable pre-run commit, so these are declared controls rather than an
# independently auditable pre-registration; a failed gate is reported, never
# retuned (`doc/nextstep.md`, item C).
DECISION = {
    "primary_metric": "continuous",
    "gap_recovery": "(energy - equal) / (oracle - equal), computed per batch cluster",
    "proxy_supported_if": (
        "mean gap recovery >= 0.50 and the 95% clustered interval excludes zero "
        "on the primary metric"
    ),
    "gap_recovery_threshold": 0.50,
    "rank_agreement_supported_if": (
        "mean tie-corrected Spearman correlation between E_{l,r} and the fitted "
        "weight across active APs is >= 0.30"
    ),
    "rank_correlation_threshold": 0.30,
    "oracle_steps": 300,
    "oracle_lr": 0.05,
    "oracle_inits": ["equal", "energy"],
    "oracle_parameterization": "w = softplus(u), best per-sample iterate retained",
    "oracle_objective": (
        "continuous sum rate; the 2-bit row evaluates the same fitted weights after "
        "actuation quantization and is not a separately optimized 2-bit oracle"
    ),
    "rank_correlation": "tie-corrected Spearman over active APs only",
    "max_unit_modulus_error": 1e-5,
    "max_deployed_theta_deviation": 1e-5,
    "max_rate_path_deviation": 2e-5,
    "oracle_regression_tolerance_bps_hz": 1e-4,
    "min_gradient_norm": 1e-8,
}


def inverse_softplus(x):
    return torch.log(torch.expm1(x.clamp(min=1e-6)))


def normalize_over_ap(weights, active):
    """w / sum_l w on the AP axis, with inactive APs removed first."""
    masked = weights * active.unsqueeze(2)
    return masked / masked.sum(dim=1, keepdim=True).clamp(min=1e-12)


def _average_ranks(values):
    """Average ranks for one small vector, including exact ties."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def active_spearman(a, b, active):
    """Mean tie-corrected Spearman correlation over sample/RIS slices."""
    values = []
    for sample in range(a.shape[0]):
        mask = active[sample].astype(bool)
        if mask.sum() < 2:
            continue
        for ris in range(a.shape[2]):
            rank_a = _average_ranks(a[sample, mask, ris])
            rank_b = _average_ranks(b[sample, mask, ris])
            rank_a -= rank_a.mean()
            rank_b -= rank_b.mean()
            denominator = np.linalg.norm(rank_a) * np.linalg.norm(rank_b)
            if denominator > 0:
                values.append(float(rank_a @ rank_b / denominator))
    return float(np.mean(values)) if values else math.nan


def oracle_weights(proposals, active, beamformer, precompute, energy, steps, lr):
    """Per-sample weights that maximize the realized sum rate, proposals frozen.

    Returns the best weights, their rate, and the gradient norm observed at the
    first step of the energy initialization (a control against a broken autograd
    path through the rate).
    """
    act = active.unsqueeze(2)
    starts = {
        "equal": torch.ones_like(energy),
        "energy": normalize_over_ap(energy, active) * energy.shape[1],
    }
    best_rate, best_w, first_grad = None, None, None

    for name, w0 in starts.items():
        u = inverse_softplus(w0).detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([u], lr=lr)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            w = F.softplus(u) * act
            rate = precompute.sum_rate(beamformer, weighted_consensus(proposals, w))
            (-rate.sum()).backward()
            if name == "energy" and step == 0:
                first_grad = float(u.grad.norm())
            with torch.no_grad():
                # Track the best iterate per sample so the reference can never be
                # worse than its own initialization, whatever Adam does later.
                current = rate.detach()
                if best_rate is None:
                    best_rate, best_w = current.clone(), w.detach().clone()
                else:
                    better = current > best_rate
                    best_rate = torch.where(better, current, best_rate)
                    best_w = torch.where(better[:, None, None], w.detach(), best_w)
            optimizer.step()

        with torch.no_grad():
            w = F.softplus(u) * act
            rate = precompute.sum_rate(beamformer, weighted_consensus(proposals, w))
            better = rate > best_rate
            best_rate = torch.where(better, rate, best_rate)
            best_w = torch.where(better[:, None, None], w, best_w)

    return best_w, best_rate, first_grad


def evaluate(model, simulator, config, device, samples, eval_seed, num_bits, steps, lr):
    batch_size = config["batch_size"]
    if samples <= 0 or samples % batch_size:
        raise ValueError("samples must be a positive multiple of the batch size")
    users = config["K"]
    threshold = config.get("assoc_threshold", 0.1)

    seed_everything(eval_seed)
    per_sample = {f"{arm}_{phase}": [] for arm in ARMS for phase in PHASES}
    rank_corr = {"energy_vs_oracle": []}
    weight_stats = {"energy_share_max": [], "oracle_share_max": [], "active_aps": []}
    controls = {
        "max_unit_modulus_error": 0.0,
        "max_deployed_theta_deviation": 0.0,
        "max_rate_path_deviation": 0.0,
        "min_oracle_minus_energy": math.inf,
        "min_gradient_norm": math.inf,
        "nonfinite_values": 0,
    }

    for _ in range(samples // batch_size):
        simulator.training_batch(users, threshold, threshold)
        features, edges, masks, direct = simulator.decentralized_batch(
            users, threshold, threshold, regenerate_channels=False
        )
        features = [tensor.to(device) for tensor in features]
        edges = [tensor.to(device) for tensor in edges]
        direct = [tensor.to(device) for tensor in direct]
        precompute = RatePrecompute(simulator, device)

        trace = {}
        with torch.no_grad():
            beamformer, deployed = model.decentralized(
                features, edges, masks, direct, trace
            )
        proposals = trace["proposals"].detach()                     # (B, A, R, N, 2)
        energy = trace["energy"].detach()                           # (B, A, R)
        active = trace["active"].detach()                           # (B, A)
        beamformer = beamformer.detach()

        weights = {
            "equal": torch.ones_like(energy) * active.unsqueeze(2),
            "energy": energy * active.unsqueeze(2),
        }
        oracle_w, oracle_rate, grad_norm = oracle_weights(
            proposals, active, beamformer, precompute, energy, steps, lr
        )
        weights["oracle"] = oracle_w
        if grad_norm is not None:
            controls["min_gradient_norm"] = min(controls["min_gradient_norm"], grad_norm)

        with torch.no_grad():
            for arm, weight in weights.items():
                theta = weighted_consensus(proposals, weight)
                controls["max_unit_modulus_error"] = max(
                    controls["max_unit_modulus_error"],
                    float((theta.norm(dim=-1) - 1.0).abs().max()),
                )
                if arm == "energy":
                    controls["max_deployed_theta_deviation"] = max(
                        controls["max_deployed_theta_deviation"],
                        float((theta - deployed).abs().max()),
                    )
                    _, reference, _ = simulator.loss(beamformer, theta, device)
                    controls["max_rate_path_deviation"] = max(
                        controls["max_rate_path_deviation"],
                        float(abs(
                            float(precompute.sum_rate(beamformer, theta).mean())
                            - float(reference)
                        )),
                    )
                continuous = precompute.sum_rate(beamformer, theta)
                discrete = precompute.sum_rate(
                    beamformer, quantize_phase(theta, num_bits)
                )
                per_sample[f"{arm}_continuous"].append(continuous.cpu().numpy())
                per_sample[f"{arm}_2bit"].append(discrete.cpu().numpy())
                if not (torch.isfinite(continuous).all() and torch.isfinite(discrete).all()):
                    controls["nonfinite_values"] += 1

            energy_rate = precompute.sum_rate(
                beamformer, weighted_consensus(proposals, weights["energy"])
            )
            controls["min_oracle_minus_energy"] = min(
                controls["min_oracle_minus_energy"],
                float((oracle_rate - energy_rate).min()),
            )

            share_energy = normalize_over_ap(energy, active).cpu().numpy()
            share_oracle = normalize_over_ap(oracle_w, active).cpu().numpy()
            rank_corr["energy_vs_oracle"].append(
                active_spearman(
                    share_energy,
                    share_oracle,
                    active.cpu().numpy(),
                )
            )
            weight_stats["energy_share_max"].append(float(share_energy.max(axis=1).mean()))
            weight_stats["oracle_share_max"].append(float(share_oracle.max(axis=1).mean()))
            weight_stats["active_aps"].append(float(active.sum(dim=1).float().mean()))

    raw = {key: np.concatenate(value) for key, value in per_sample.items()}
    return raw, rank_corr, weight_stats, controls


def gap_recovery(raw, batch_size, phase):
    """Per-cluster (energy - equal) / (oracle - equal) on one phase setting."""
    def clusters(arm):
        return raw[f"{arm}_{phase}"].reshape(-1, batch_size).mean(axis=1)

    equal, energy, oracle = clusters("equal"), clusters("energy"), clusters("oracle")
    span = oracle - equal
    valid = span > 1e-9
    ratio = np.full(span.shape, np.nan)
    ratio[valid] = (energy - equal)[valid] / span[valid]
    return ratio, equal, energy, oracle


def summarize(raw, rank_corr, weight_stats, controls, batch_size):
    metrics = {}
    for arm in ARMS:
        for phase in PHASES:
            values = raw[f"{arm}_{phase}"]
            metrics[f"{arm}_{phase}"] = cluster_summary(
                values.reshape(-1, batch_size).mean(axis=1)
            )

    recovery = {}
    for phase in PHASES:
        ratio, equal, energy, oracle = gap_recovery(raw, batch_size, phase)
        recovery[phase] = {
            "gap_recovery": cluster_summary(ratio),
            "energy_minus_equal": cluster_summary(energy - equal),
            "oracle_minus_energy": cluster_summary(oracle - energy),
            "oracle_minus_equal": cluster_summary(oracle - equal),
            "degenerate_clusters": int((~np.isfinite(ratio)).sum()),
        }

    primary = DECISION["primary_metric"]
    ratio_summary = recovery[primary]["gap_recovery"]
    rho = cluster_summary(np.asarray(rank_corr["energy_vs_oracle"]))
    gates = {
        "unit_modulus": controls["max_unit_modulus_error"]
        <= DECISION["max_unit_modulus_error"],
        "deployed_theta_match": controls["max_deployed_theta_deviation"]
        <= DECISION["max_deployed_theta_deviation"],
        "rate_path_equivalence": controls["max_rate_path_deviation"]
        <= DECISION["max_rate_path_deviation"],
        "oracle_not_below_energy": controls["min_oracle_minus_energy"]
        >= -DECISION["oracle_regression_tolerance_bps_hz"],
        "gradient_flow": controls["min_gradient_norm"] >= DECISION["min_gradient_norm"],
        "finite": controls["nonfinite_values"] == 0,
    }
    supported = bool(
        ratio_summary["mean"] is not None
        and ratio_summary["mean"] >= DECISION["gap_recovery_threshold"]
        and ratio_summary.get("ci_low", -math.inf) > 0.0
    )
    rank_supported = bool(
        rho["mean"] is not None and rho["mean"] >= DECISION["rank_correlation_threshold"]
    )
    return {
        "criteria": DECISION,
        "metrics": metrics,
        "recovery": recovery,
        "rank_correlation": {
            "energy_vs_oracle": rho,
        },
        "weight_stats": {k: cluster_summary(np.asarray(v)) for k, v in weight_stats.items()},
        "controls": controls,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "proxy_supported": supported,
        "rank_agreement_supported": rank_supported,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="frozen G2 run directory")
    parser.add_argument("--checkpoint", default="iter150000.pt")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--eval_seed", type=int, default=20260915)
    parser.add_argument("--replication_seed", type=int, default=20260918)
    parser.add_argument("--num_bits", type=int, default=2)
    parser.add_argument("--steps", type=int, default=DECISION["oracle_steps"])
    parser.add_argument("--lr", type=float, default=DECISION["oracle_lr"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out_dir",
        default="../../artifacts/decentralized_ris/e05_energy_consensus/consensus_weight_oracle",
    )
    args = parser.parse_args()

    if args.steps != DECISION["oracle_steps"] or args.lr != DECISION["oracle_lr"]:
        raise SystemExit("--steps and --lr must match the declared oracle budget")
    if args.num_bits != 2:
        raise SystemExit("--num_bits must be 2 because the recorded phase label is '2bit'")

    device = resolve_device(args.device)
    with open(os.path.join(args.run, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    config = summary["config"]

    seed_everything(config["seed"])
    simulator = ChannelSimulator(
        config["M"], config["N"], config["L"], config["batch_size"], n_ap=config["AP"]
    )
    model = build_model(config, simulator, device)
    path = checkpoint_path(args.run, args.checkpoint)
    if path is None:
        raise SystemExit(f"missing {args.checkpoint} under {args.run}")
    load_checkpoint(model, path, device)
    model.eval()

    seeds = {"primary": args.eval_seed}
    if args.replication_seed and args.replication_seed != args.eval_seed:
        seeds["replication"] = args.replication_seed

    os.makedirs(args.out_dir, exist_ok=True)
    payload = {
        "run": args.run,
        "checkpoint": args.checkpoint,
        "config": config,
        "samples": args.samples,
        "num_bits": args.num_bits,
        "oracle_steps": args.steps,
        "oracle_lr": args.lr,
        "evaluations": {},
    }
    for role, seed in seeds.items():
        started = time.time()
        print(f"[oracle] {role} seed={seed}", flush=True)
        raw, rank_corr, weight_stats, controls = evaluate(
            model, simulator, config, device, args.samples, seed,
            args.num_bits, args.steps, args.lr,
        )
        report = summarize(raw, rank_corr, weight_stats, controls, config["batch_size"])
        report["eval_seed"] = seed
        report["wall_seconds"] = round(time.time() - started, 1)
        payload["evaluations"][role] = report
        np.savez(
            os.path.join(args.out_dir, f"raw_{role}.npz"),
            **raw,
            rank_energy_vs_oracle=np.asarray(rank_corr["energy_vs_oracle"]),
        )
        print(
            "  " + "  ".join(
                f"{arm}={report['metrics'][f'{arm}_continuous']['mean']:.4f}"
                for arm in ARMS
            ),
            flush=True,
        )
        print(
            f"  gap_recovery={report['recovery']['continuous']['gap_recovery']['mean']:.4f}"
            f"  rho={report['rank_correlation']['energy_vs_oracle']['mean']:.4f}"
            f"  gates_pass={report['all_gates_pass']}",
            flush=True,
        )

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    failed = {
        role: [name for name, ok in report["gates"].items() if not ok]
        for role, report in payload["evaluations"].items()
        if not report["all_gates_pass"]
    }
    if failed:
        details = "; ".join(f"{role}: {', '.join(names)}" for role, names in failed.items())
        raise ControlFailure(f"declared controls failed: {details}")


if __name__ == "__main__":
    main()
