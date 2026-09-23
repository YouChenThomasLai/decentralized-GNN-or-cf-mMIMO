"""E13: active-beamforming controls under a frozen G2 RIS phase.

Benchmark item 7 of the plan in `doc/research_positioning.md`.  The RIS action is
held fixed: every arm receives the *same* paper-decentralized G2 phase proposals
and the same parameter-free local-energy consensus, so the only quantity that
changes is how each AP forms its active beamformer.

    arm            active design at AP l                      per-AP power
    -------------------------------------------------------------------------
    g2             native G2 readout                          learned alpha_l
    g2_full        native G2 direction                        full Pmax
    mrt_full       RIS-aware local MRT                        full Pmax
    rzf_full       RIS-aware local RZF                        full Pmax
    mrt_matched    RIS-aware local MRT                        G2's alpha_l
    rzf_matched    RIS-aware local RZF                        G2's alpha_l

MRT and RZF are two-stage controls: the CPU must broadcast the fused phase back
before an AP can form its effective channels, so they add one CPU->AP round that
the native one-shot G2 path does not need.  The script counts that round.

The signal model in `rates` pairs the stored channel with the beamformer as
h^T w, so the conjugated channel g = conj(h) is the one that satisfies the usual
g^H w convention.  Every precoder below is written in terms of g, and
H^eff in the report denotes the matrix whose columns are the served g_{l,k}.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from evaluate import build_model, checkpoint_path, resolve_device, seed_everything
from model import load_checkpoint
from rates import RatePrecompute, phase_levels, quantize_phase
from simulation import ChannelSimulator


ARMS = ("g2", "g2_full", "mrt_full", "rzf_full", "mrt_matched", "rzf_matched")
CONVENTIONAL = ("mrt_full", "rzf_full", "mrt_matched", "rzf_matched")
PHASES = ("continuous", "2bit")

# Declared before the locked evaluation.  The current history does not contain
# an immutable pre-run commit, so the artifact records the declaration and a
# failed gate is reported, never retuned.
PRIMARY_METRIC = "continuous"
DECISION = {
    "primary_phase": PRIMARY_METRIC,
    "advantage_supported_if": (
        "g2 minus the best conventional arm is positive and its 95% clustered "
        "interval excludes zero on the primary phase setting"
    ),
    "advance_to_stage_b_if_deficit_below_bps_hz": 0.5,
    "rzf_lambda_multipliers": [0.1, 10.0],
    "rzf_lambda_sensitivity_samples": 80,
    "power_feasibility_tolerance": 1e-6,
    "max_solve_relative_residual": 1e-4,
    "max_locality_deviation": 0.0,
    "max_unit_modulus_error": 1e-5,
    "max_quantization_grid_error": 1e-6,
    "max_rate_path_deviation": 2e-5,
}


def clustered(values, batch_size):
    """Mean and batch-clustered standard error over per-sample sum rates."""
    values = np.asarray(values).ravel()
    clusters = values.reshape(-1, batch_size).mean(axis=1)
    sem = clusters.std(ddof=1) / math.sqrt(len(clusters)) if len(clusters) > 1 else 0.0
    return {
        "mean": float(values.mean()),
        "clustered_sem": float(sem),
        "samples": int(values.size),
        "clusters": int(clusters.size),
    }


def paired(left, right, batch_size):
    """Paired contrast on identical channel samples, with a cluster interval."""
    difference = np.asarray(left) - np.asarray(right)
    stats = clustered(difference, batch_size)
    half = 1.96 * stats["clustered_sem"]
    clusters = difference.reshape(-1, batch_size).mean(axis=1)
    stats.update(
        ci95=[stats["mean"] - half, stats["mean"] + half],
        cluster_win_fraction=float((clusters > 0).mean()),
        sample_win_fraction=float((difference > 0).mean()),
    )
    return stats


def complex_channels(effective, n_ap, n_antennas):
    """(B, A*K, 2M) real embedding -> (B, A, K, M) complex per-AP channels."""
    batch = effective.shape[0]
    k_user = effective.shape[1] // n_ap
    real = effective[..., :n_antennas].reshape(batch, n_ap, k_user, n_antennas)
    imag = effective[..., n_antennas:].reshape(batch, n_ap, k_user, n_antennas)
    return torch.complex(real, imag)


def embed_beamformer(weights):
    """(B, A, K, M) complex precoders -> the (B, 2M, A*K) layout `rates` expects."""
    batch, n_ap, k_user, n_antennas = weights.shape
    real = weights.real.reshape(batch, n_ap * k_user, n_antennas)
    imag = weights.imag.reshape(batch, n_ap * k_user, n_antennas)
    return torch.cat((real, imag), dim=2).transpose(2, 1).contiguous()


def per_ap_power(beamformer, n_ap):
    """||W_l||_F^2 for the (B, 2M, A*K) real layout."""
    batch, width, k_tot = beamformer.shape
    blocks = beamformer.reshape(batch, width, n_ap, k_tot // n_ap)
    return blocks.permute(0, 2, 1, 3).reshape(batch, n_ap, -1).pow(2).sum(dim=2)


def apply_power(weights, pmax, alpha):
    """Scale each AP block to ||W_l||_F^2 = Pmax * alpha_l."""
    batch, n_ap = weights.shape[0], weights.shape[1]
    norm = weights.reshape(batch, n_ap, -1).abs().pow(2).sum(dim=2).sqrt()
    target = torch.sqrt(pmax * alpha)
    scale = torch.where(norm > 0, target / norm.clamp(min=1e-12), torch.zeros_like(norm))
    return weights * scale[..., None, None]


def local_precoder(effective, served, pmax, kind, alpha=None, sigma=RatePrecompute.SIGMA):
    """RIS-aware local MRT or RZF from each AP's own effective channels.

    `effective` is the (B, A*K, 2M) embedding returned by
    `RatePrecompute.effective_channel`, i.e. the channel *after* the fused RIS
    phase has been applied, and `served` is the (B, A, K) association mask.  AP l
    reads only its own block, so the precoder is strictly AP-local given the
    broadcast phase; `locality_deviation` verifies that on real data.
    """
    n_ap = served.shape[1]
    n_antennas = effective.shape[2] // 2
    gain = complex_channels(effective, n_ap, n_antennas).conj()
    mask = served.to(gain.real.dtype)
    gain = gain * mask[..., None]
    counts = mask.sum(dim=2)

    residual = 0.0
    if kind == "mrt":
        weights = gain
    elif kind == "rzf":
        gram = torch.einsum("bakm,bakn->bamn", gain, gain.conj())
        # lambda_l = |K_l| sigma^2 / P_l, declared rather than tuned.  An
        # AP that serves nobody has an all-zero G_l, so any positive lambda
        # leaves its block at zero and only keeps the solve well posed.
        lam = torch.where(counts > 0, counts * sigma / pmax, torch.ones_like(counts))
        eye = torch.eye(n_antennas, device=gain.device, dtype=gain.dtype)
        system = gram + lam[..., None, None] * eye
        matrix = gain.transpose(-1, -2)
        solution = torch.linalg.solve(system, matrix)
        weights = solution.transpose(-1, -2)
        error = (system @ solution - matrix).abs().pow(2).sum(dim=(-1, -2)).sqrt()
        scale = matrix.abs().pow(2).sum(dim=(-1, -2)).sqrt().clamp(min=1e-30)
        residual = float((error / scale).max())
    else:
        raise ValueError(f"unknown precoder {kind}")

    weights = weights * mask[..., None]
    if alpha is None:
        alpha = torch.ones_like(counts)
    return apply_power(weights, pmax, alpha), residual


def rescale_to_full_power(beamformer, served, pmax, n_ap):
    """Keep a beamformer's direction but spend the whole per-AP budget."""
    batch, width, k_tot = beamformer.shape
    k_user = k_tot // n_ap
    blocks = beamformer.reshape(batch, width, n_ap, k_user).permute(0, 2, 1, 3)
    norm = blocks.reshape(batch, n_ap, -1).pow(2).sum(dim=2).sqrt()
    active = (served.to(norm.dtype).sum(dim=2) > 0).to(norm.dtype)
    scale = torch.where(
        norm > 0, math.sqrt(pmax) * active / norm.clamp(min=1e-12), torch.zeros_like(norm)
    )
    blocks = blocks * scale[:, :, None, None]
    return blocks.permute(0, 2, 1, 3).reshape(batch, width, k_tot)


def stack_masks(masks, device):
    """List of A (B, K) arrays -> one (B, A, K) boolean tensor."""
    return torch.stack(
        [torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=device) for mask in masks],
        dim=1,
    )


def signaling_ledger(n_ris, n_elements, n_ap, num_bits):
    """Counted online payloads, in real values and in bits, per topology sample.

    fp32 is the nominal wire format for continuous phase; the rounded arm sends
    `num_bits` per element.  UE->AP CSI acquisition is outside the boundary and
    is identical for every arm, so it is listed as excluded rather than zero.
    """
    proposal_values = n_ap * n_ris * (n_elements + 1)
    broadcast_values = n_ap * n_ris * n_elements
    actuation_values = n_ris * n_elements
    return {
        "accounting_boundary": "AP->CPU proposals, CPU->AP phase broadcast, CPU->RIS actuation; UE->AP CSI acquisition excluded and identical across arms",
        "ap_to_cpu_values": proposal_values,
        "ap_to_cpu_bits_fp32": 32 * proposal_values,
        "cpu_to_ap_values": {"one_shot": 0, "two_stage": broadcast_values},
        "cpu_to_ap_bits": {
            "one_shot": 0,
            "two_stage_fp32": 32 * broadcast_values,
            f"two_stage_{num_bits}bit": num_bits * broadcast_values,
        },
        "cpu_to_ris_values": actuation_values,
        "cpu_to_ris_bits": {
            "fp32": 32 * actuation_values,
            f"{num_bits}bit": num_bits * actuation_values,
        },
        "online_rounds": {"one_shot": 1, "two_stage": 2},
        "arms_by_interface": {
            "one_shot": ["g2", "g2_full"],
            "two_stage": list(CONVENTIONAL),
        },
    }


def timing(model, simulator, config, device, samples, repeats, num_bits):
    """Batch-1 inference and solve time after warm-up, on one named device."""
    users = config["K"]
    threshold = config.get("assoc_threshold", 0.1)
    pmax = 10 ** ((config["pmax_dbm"] - 30) / 10)
    records = {key: [] for key in ("g2_inference", "mrt_stage", "rzf_stage")}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    collected = 0
    while collected < samples:
        simulator.training_batch(users, threshold, threshold)
        features, edges, masks, direct = simulator.decentralized_batch(
            users, threshold, threshold, regenerate_channels=False
        )
        features = [tensor[:1].to(device) for tensor in features]
        edges = [tensor[:1].to(device) for tensor in edges]
        direct = [tensor[:1].to(device) for tensor in direct]
        masks = [mask[:1] for mask in masks]
        served = stack_masks(masks, device)
        precompute = RatePrecompute(simulator, device)

        with torch.no_grad():
            for _ in range(repeats):                                     # warm-up
                beamformer, phase = model.decentralized(features, edges, masks, direct)
            sync()
            start = time.perf_counter()
            for _ in range(repeats):
                beamformer, phase = model.decentralized(features, edges, masks, direct)
            sync()
            records["g2_inference"].append((time.perf_counter() - start) / repeats)

            rounded = quantize_phase(phase, num_bits)
            effective = precompute.effective_channel(rounded)[:1]
            for kind, key in (("mrt", "mrt_stage"), ("rzf", "rzf_stage")):
                for _ in range(repeats):
                    local_precoder(effective, served, pmax, kind)
                sync()
                start = time.perf_counter()
                for _ in range(repeats):
                    local_precoder(effective, served, pmax, kind)
                sync()
                records[key].append((time.perf_counter() - start) / repeats)
        collected += 1

    report = {
        key: {
            "median_ms": float(np.median(value) * 1e3),
            "p95_ms": float(np.quantile(value, 0.95) * 1e3),
            "samples": len(value),
        }
        for key, value in records.items()
    }
    for kind in ("mrt", "rzf"):
        report[f"{kind}_end_to_end"] = {
            "median_ms": report["g2_inference"]["median_ms"] + report[f"{kind}_stage"]["median_ms"],
            "note": "one-shot G2 stage plus the second local-precoding stage; the CPU->AP broadcast latency is not modeled",
        }
    return report


def evaluate(model, simulator, config, device, samples, eval_seed, num_bits):
    batch_size = config["batch_size"]
    if samples <= 0 or samples % batch_size:
        raise ValueError("samples must be a positive multiple of the batch size")
    users = config["K"]
    threshold = config.get("assoc_threshold", 0.1)
    pmax = 10 ** ((config["pmax_dbm"] - 30) / 10)
    n_ap = config["AP"]

    seed_everything(eval_seed)
    records = {f"{arm}_{phase}": [] for arm in ARMS for phase in PHASES}
    served_counts = []
    controls = {
        "max_unit_modulus_error": 0.0,
        "max_quantization_grid_error": 0.0,
        "max_solve_relative_residual": 0.0,
        "max_power_overshoot": 0.0,
        "max_locality_deviation": None,
        "max_rate_path_deviation": None,
        "unserved_column_leakage": 0.0,
        "nonfinite_values": 0,
    }
    levels = phase_levels(num_bits, device)
    # Declared before the run as a sensitivity diagnostic only: the primary RZF
    # arm keeps the declared lambda, which is never tuned on the holdout.
    sensitivity_batches = DECISION["rzf_lambda_sensitivity_samples"] // batch_size
    sensitivity = {f"rzf_lambda_x{factor:g}": [] for factor in DECISION["rzf_lambda_multipliers"]}

    for index in range(samples // batch_size):
        simulator.training_batch(users, threshold, threshold)
        features, edges, masks, direct = simulator.decentralized_batch(
            users, threshold, threshold, regenerate_channels=False
        )
        features = [tensor.to(device) for tensor in features]
        edges = [tensor.to(device) for tensor in edges]
        direct = [tensor.to(device) for tensor in direct]
        served = stack_masks(masks, device)
        served_counts.append(served.sum(dim=2).cpu().numpy())
        precompute = RatePrecompute(simulator, device)

        with torch.no_grad():
            native, phase = model.decentralized(features, edges, masks, direct)
            alpha = per_ap_power(native, n_ap) / pmax
            native_full = rescale_to_full_power(native, served, pmax, n_ap)

            for name, theta in (
                ("continuous", phase), ("2bit", quantize_phase(phase, num_bits))
            ):
                effective = precompute.effective_channel(theta)
                beamformers = {"g2": native, "g2_full": native_full}
                for kind in ("mrt", "rzf"):
                    for suffix, scale in (("full", None), ("matched", alpha)):
                        weights, residual = local_precoder(
                            effective, served, pmax, kind, alpha=scale
                        )
                        beamformers[f"{kind}_{suffix}"] = embed_beamformer(weights)
                        controls["max_solve_relative_residual"] = max(
                            controls["max_solve_relative_residual"], residual
                        )

                for arm, beamformer in beamformers.items():
                    rate = precompute.sum_rate_from_effective(beamformer, effective)
                    records[f"{arm}_{name}"].append(rate.cpu().numpy())
                    controls["nonfinite_values"] += int((~torch.isfinite(rate)).sum())
                    controls["nonfinite_values"] += int((~torch.isfinite(beamformer)).sum())
                    power = per_ap_power(beamformer, n_ap)
                    controls["max_power_overshoot"] = max(
                        controls["max_power_overshoot"],
                        float((power / pmax - 1.0).max()),
                    )
                    idle = (~served).reshape(served.shape[0], -1).to(beamformer.dtype)
                    controls["unserved_column_leakage"] = max(
                        controls["unserved_column_leakage"],
                        float((beamformer.abs() * idle[:, None, :]).max()),
                    )

                controls["max_unit_modulus_error"] = max(
                    controls["max_unit_modulus_error"],
                    float((theta.norm(dim=-1) - 1.0).abs().max()),
                )
                if name == "2bit":
                    grid = (theta.unsqueeze(-2) - levels).norm(dim=-1).min(dim=-1).values
                    controls["max_quantization_grid_error"] = max(
                        controls["max_quantization_grid_error"], float(grid.max())
                    )

                if index == 0:
                    # Independent scalar rate path, for every arm and not only
                    # for G2, so a shared-phase or layout error cannot hide.
                    deviation = controls["max_rate_path_deviation"] or 0.0
                    for arm, beamformer in beamformers.items():
                        _, reference, _ = simulator.loss(beamformer, theta, device)
                        vectorized = precompute.sum_rate_from_effective(
                            beamformer, effective
                        ).mean()
                        deviation = max(deviation, float((reference - vectorized).abs()))
                    controls["max_rate_path_deviation"] = deviation

            if index < sensitivity_batches:
                reference = precompute.effective_channel(phase)
                for factor in DECISION["rzf_lambda_multipliers"]:
                    weights, _ = local_precoder(
                        reference, served, pmax, "rzf",
                        sigma=RatePrecompute.SIGMA * factor,
                    )
                    sensitivity[f"rzf_lambda_x{factor:g}"].append(
                        precompute.sum_rate_from_effective(
                            embed_beamformer(weights), reference
                        ).cpu().numpy()
                    )

            if index == 0:
                controls["max_locality_deviation"] = locality_control(
                    precompute.effective_channel(phase), served, pmax
                )

    served_counts = np.concatenate(served_counts, axis=0)
    metrics = {
        key: clustered(np.concatenate(value), batch_size) for key, value in records.items()
    }
    raw = {key: np.concatenate(value) for key, value in records.items()}
    lambda_sensitivity = {
        key: clustered(np.concatenate(value), batch_size)
        for key, value in sensitivity.items() if value
    }
    return metrics, raw, controls, served_counts, lambda_sensitivity


def locality_control(effective, served, pmax):
    """Replace every other AP's effective channel and re-derive AP l's precoder.

    Mirrors the E05 locality control: if the conventional precoders were reading
    anything beyond AP l's own block, the recomputed block would move.
    """
    deviation = 0.0
    generator = torch.Generator(device="cpu").manual_seed(0)
    for kind in ("mrt", "rzf"):
        reference, _ = local_precoder(effective, served, pmax, kind)
        for ap_index in range(served.shape[1]):
            perturbed = effective.clone()
            k_user = effective.shape[1] // served.shape[1]
            noise = torch.randn(
                perturbed.shape, generator=generator, dtype=perturbed.dtype
            ).to(perturbed.device)
            keep = slice(ap_index * k_user, (ap_index + 1) * k_user)
            noise[:, keep, :] = perturbed[:, keep, :]
            candidate, _ = local_precoder(noise, served, pmax, kind)
            deviation = max(
                deviation,
                float((candidate[:, ap_index] - reference[:, ap_index]).abs().max()),
            )
    return deviation


def contrasts(raw, batch_size):
    table = {}
    for phase in PHASES:
        for arm in ARMS:
            if arm == "g2":
                continue
            table[f"g2_minus_{arm}_{phase}"] = paired(
                raw[f"g2_{phase}"], raw[f"{arm}_{phase}"], batch_size
            )
        table[f"rzf_full_minus_mrt_full_{phase}"] = paired(
            raw[f"rzf_full_{phase}"], raw[f"mrt_full_{phase}"], batch_size
        )
        table[f"rzf_matched_minus_rzf_full_{phase}"] = paired(
            raw[f"rzf_matched_{phase}"], raw[f"rzf_full_{phase}"], batch_size
        )
    return table


def decide(metrics, table, controls):
    """Apply the declared rules without re-reading them from the results."""
    phase = DECISION["primary_phase"]
    best = max(CONVENTIONAL, key=lambda arm: metrics[f"{arm}_{phase}"]["mean"])
    contrast = table[f"g2_minus_{best}_{phase}"]
    # How far the best conventional arm trails G2; negative if it wins.
    deficit = contrast["mean"]
    gates = {
        "power_feasibility": controls["max_power_overshoot"]
        <= DECISION["power_feasibility_tolerance"],
        "solve_residual": controls["max_solve_relative_residual"]
        <= DECISION["max_solve_relative_residual"],
        "locality": controls["max_locality_deviation"] is not None
        and controls["max_locality_deviation"] <= DECISION["max_locality_deviation"],
        "unit_modulus": controls["max_unit_modulus_error"]
        <= DECISION["max_unit_modulus_error"],
        "quantization_grid": controls["max_quantization_grid_error"]
        <= DECISION["max_quantization_grid_error"],
        "rate_path_equivalence": controls["max_rate_path_deviation"] is not None
        and controls["max_rate_path_deviation"] <= DECISION["max_rate_path_deviation"],
        "no_unserved_column_leakage": controls["unserved_column_leakage"] == 0.0,
        "finite": controls["nonfinite_values"] == 0,
    }
    supported = bool(contrast["mean"] > 0 and contrast["ci95"][0] > 0)
    return {
        "criteria": DECISION,
        "best_conventional_arm": best,
        "primary_contrast": contrast,
        "g2_advantage_supported": supported,
        "best_conventional_deficit_bps_hz": float(deficit),
        "advance_to_stage_b": bool(
            deficit < DECISION["advance_to_stage_b_if_deficit_below_bps_hz"]
        ),
        "numerical_gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def render(payload):
    lines = [
        "# E13 fixed-G2-RIS active-beamforming controls",
        "",
        f"Checkpoint: `{payload['checkpoint']}`",
        "",
        f"Holdout: {payload['samples']} paired samples, evaluation seed "
        f"{payload['eval_seed']}, device {payload['device']}.",
        "",
        "| arm | interface | continuous | clustered SE | 2-bit | clustered SE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    interfaces = {arm: ("two-stage" if arm in CONVENTIONAL else "one-shot") for arm in ARMS}
    for arm in ARMS:
        cont = payload["metrics"][f"{arm}_continuous"]
        disc = payload["metrics"][f"{arm}_2bit"]
        lines.append(
            f"| {arm} | {interfaces[arm]} | {cont['mean']:.4f} | {cont['clustered_sem']:.4f} | "
            f"{disc['mean']:.4f} | {disc['clustered_sem']:.4f} |"
        )

    lines += [
        "",
        "## Paired contrasts",
        "",
        "| contrast | mean | clustered SE | 95% CI | cluster wins |",
        "|---|---:|---:|---|---:|",
    ]
    for name, value in payload["paired_contrasts"].items():
        lines.append(
            f"| {name} | {value['mean']:+.4f} | {value['clustered_sem']:.4f} | "
            f"[{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}] | "
            f"{100 * value['cluster_win_fraction']:.1f}% |"
        )

    lines += [
        "",
        "## Pre-registered gates",
        "",
        "| gate | passed |",
        "|---|:-:|",
    ]
    for gate, passed in payload["decision"]["numerical_gates"].items():
        lines.append(f"| {gate} | {'yes' if passed else 'NO'} |")

    served = payload["association"]
    lines += [
        "",
        "## Association and timing",
        "",
        f"Served users per AP: mean {served['mean']:.3f}, max {served['max']}, "
        f"fraction of AP blocks with more users than antennas "
        f"{served['fraction_above_antennas']:.3f}.",
        "",
        "| stage | median (ms) | p95 (ms) |",
        "|---|---:|---:|",
    ]
    for stage in ("g2_inference", "mrt_stage", "rzf_stage"):
        row = payload["timing"][stage]
        lines.append(f"| {stage} | {row['median_ms']:.3f} | {row['p95_ms']:.3f} |")

    lines += [
        "",
        "## RZF regularizer sensitivity (diagnostic, excluded from the decision)",
        "",
        f"Pre-registered lambda_l = |K_l| sigma^2 / P_l on the first "
        f"{DECISION['rzf_lambda_sensitivity_samples']} samples, continuous phase.",
        "",
        "| multiplier | continuous | clustered SE |",
        "|---|---:|---:|",
    ]
    for key, row in payload["rzf_lambda_sensitivity"].items():
        lines.append(f"| {key} | {row['mean']:.4f} | {row['clustered_sem']:.4f} |")

    decision = payload["decision"]
    lines += [
        "",
        "## Decision",
        "",
        f"- Best conventional arm: `{decision['best_conventional_arm']}`",
        f"- G2 advantage supported: {decision['g2_advantage_supported']}",
        f"- Best conventional arm trails G2 by "
        f"{decision['best_conventional_deficit_bps_hz']:+.4f} bps/Hz",
        f"- Advance to Stage B: {decision['advance_to_stage_b']}",
        f"- All numerical gates passed: {decision['all_gates_passed']}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="frozen G2 run directory")
    parser.add_argument("--checkpoint", default="iter150000.pt")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--eval_seed", type=int, default=20260918)
    parser.add_argument(
        "--replication_seed",
        type=int,
        default=20260915,
        help="second evaluation seed kept as a cross-experiment consistency check",
    )
    parser.add_argument("--num_bits", type=int, default=2)
    parser.add_argument("--timing_samples", type=int, default=50)
    parser.add_argument("--timing_repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out_dir",
        default="../../artifacts/decentralized_ris/e13_fixed_ris_beamforming",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    with open(os.path.join(args.run, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    config = summary["config"]

    # Reset before construction so the AP-RIS LoS draw reproduces the training
    # topology, exactly as the E06 evaluator does.
    seed_everything(config["seed"])
    simulator = ChannelSimulator(
        config["M"], config["N"], config["L"], config["batch_size"], n_ap=config["AP"]
    )
    model = build_model(config, simulator, device)
    checkpoint = checkpoint_path(args.run, args.checkpoint)
    if checkpoint is None:
        raise SystemExit(f"missing {args.checkpoint} under {args.run}")
    load_checkpoint(model, checkpoint, device)
    model.eval()

    seeds = {"primary": args.eval_seed}
    if args.replication_seed and args.replication_seed != args.eval_seed:
        seeds["replication"] = args.replication_seed

    evaluations = {}
    raw_all = {}
    for role, seed in seeds.items():
        print(f"[evaluate] {role} seed={seed}", flush=True)
        metrics, raw, controls, served, lambda_sensitivity = evaluate(
            model, simulator, config, device, args.samples, seed, args.num_bits
        )
        table = contrasts(raw, config["batch_size"])
        counts = served.ravel()
        evaluations[role] = {
            "eval_seed": seed,
            "metrics": metrics,
            "paired_contrasts": table,
            "controls": controls,
            "association": {
                "mean": float(counts.mean()),
                "max": int(counts.max()),
                "fraction_above_antennas": float((counts > config["M"]).mean()),
            },
            "rzf_lambda_sensitivity": lambda_sensitivity,
            "decision": decide(metrics, table, controls),
        }
        raw_all[role] = raw
        print(
            "  " + "  ".join(
                f"{arm}={metrics[f'{arm}_continuous']['mean']:.4f}" for arm in ARMS
            ),
            flush=True,
        )

    seed_everything(args.eval_seed)
    timings = timing(
        model, simulator, config, device,
        args.timing_samples, args.timing_repeats, args.num_bits,
    )

    payload = {
        "experiment": "E13",
        "question": "under the frozen G2 RIS phase, does the learned active beamformer beat RIS-aware local MRT and RZF?",
        "run_dir": args.run,
        "checkpoint": checkpoint,
        "config": config,
        "model": model.describe(),
        "samples": args.samples,
        "num_bits": args.num_bits,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "eval_seed": args.eval_seed,
        "seeds": seeds,
        "signaling": signaling_ledger(
            config["L"], config["N"], config["AP"], args.num_bits
        ),
        "timing": timings,
        "evaluations": evaluations,
    }
    primary = evaluations["primary"]
    payload.update(
        metrics=primary["metrics"],
        paired_contrasts=primary["paired_contrasts"],
        controls=primary["controls"],
        association=primary["association"],
        rzf_lambda_sensitivity=primary["rzf_lambda_sensitivity"],
        decision=primary["decision"],
    )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    np.savez(
        os.path.join(args.out_dir, "paired_rates.npz"),
        **{
            f"{role}__{key}": values
            for role, raw in raw_all.items()
            for key, values in raw.items()
        },
    )
    report = render(payload)
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(report + "\n")
    print(report)
    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
