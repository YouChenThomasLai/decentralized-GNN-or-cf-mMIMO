"""Frozen G2/R0 sum rate against the common per-AP transmit power budget.

The paper plots sum rate against `P_max,l = P_max` for all APs.  This program
runs that sweep for the two shortlisted frozen policies, G2-150k and R0-500k,
on paired channel draws, so every power point compares the same realizations.

Both checkpoints were trained at 15 dBm and are *not* retrained here.  That is
a deliberate scope limit and it is exact rather than approximate: in both
architectures `Pmax` enters only as the scale of an already L2-normalized
per-AP block,

    W_l = sqrt(Pmax * alpha_l) * normalize(Wtilde_l),

and neither `alpha_l` (PowerControl) nor the RIS phase depends on `Pmax`.  A
frozen checkpoint evaluated at `P` therefore emits exactly
`sqrt(P / P_ref) * W_l(P_ref)` with an unchanged phase, so the sweep measures
one fixed spatial solution carried across budgets.  Every AP scales by the same
factor, so signal and interference scale together and each user's SINR
`P*S_k / (P*I_k + sigma^2)` is monotone in `P` with the interference-limited
ceiling `log2(1 + S_k/I_k)`.  That ceiling is reported as the `sigma^2 -> 0`
limit of the same frozen solution.

Because of this identity a single forward pass per batch serves all power
points.  The declared equivalence control rebuilds each model at the extreme
budgets and checks the fast path against it before any result is used.

Run from `code/decentralized_ris/`:

    python -m experiments.power_sweep --samples 1600 --eval_seed 20261003 \
        --out ../../artifacts/decentralized_ris/e15_transmit_power_sweep/sweep.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import build_model, resolve_device, seed_everything, temporary_seed
from experiments.mrc_proxy_diagnostic import ControlFailure, cluster_summary, paired_difference
from experiments.topology_screen import RUNS
from model import load_checkpoint
from rates import RatePrecompute, quantize_phase, random_phase_like
from simulation import ChannelSimulator


POWERS_DBM = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0)
MODES = ("centralized", "centralized_2bit", "decentralized", "decentralized_2bit")

# Declared before the locked evaluation; a failed gate is reported, not retuned.
DECISION = {
    "question": (
        "does the frozen G2-150k advantage over frozen R0-500k in paper-decentralized "
        "sum rate persist across the common per-AP transmit power budget?"
    ),
    "primary_metric": "paper-decentralized continuous sum rate, G2 minus R0, per power point",
    "secondary_metric": "the same contrast under rounded 2-bit RIS actuation",
    "persists_if": (
        "the 95% batch-clustered interval of G2 minus R0 excludes zero in both phase "
        "settings at every swept power"
    ),
    "powers_dbm": list(POWERS_DBM),
    "reference_pmax_dbm": 15.0,
    "training_budget": "no retraining; both checkpoints were trained at 15 dBm only",
    "max_unit_modulus_error": 1e-5,
    "max_rate_path_deviation": 2e-5,
    "max_rebuild_deviation_bps_hz": 1e-3,
    "rebuild_check_powers_dbm": [5.0, 35.0],
    "rebuild_check_batches": 2,
    "max_power_excess_ratio": 1.0 + 1e-5,
    "monotonicity_tolerance_bps_hz": 1e-4,
}


def block_power(beamformer, n_ap):
    """Per-AP squared Frobenius norm of the deployed beamformer blocks."""
    batch, width, total = beamformer.shape
    blocks = beamformer.reshape(batch, width, n_ap, total // n_ap)
    return (blocks ** 2).sum(dim=(1, 3))


def sweep_rates(beamformer, phase, precompute, scales, num_bits):
    """Continuous and rounded rates for one frozen solution at every scale."""
    discrete = quantize_phase(phase, num_bits)
    continuous_channel = precompute.effective_channel(phase)
    discrete_channel = precompute.effective_channel(discrete)
    out = {}
    for power, scale in scales.items():
        scaled = beamformer * scale
        out[(power, "continuous")] = precompute.sum_rate_from_effective(
            scaled, continuous_channel
        )
        out[(power, "2bit")] = precompute.sum_rate_from_effective(
            scaled, discrete_channel
        )
    return out


def ceiling_rate(beamformer, phase, precompute):
    """Interference-limited limit of the same frozen solution as sigma^2 -> 0."""
    sigma = precompute.SIGMA
    try:
        precompute.SIGMA = 0.0
        return precompute.sum_rate(beamformer, phase)
    finally:
        precompute.SIGMA = sigma


def run(samples, eval_seed, device, num_bits):
    config = None
    scales = {power: 10.0 ** ((power - DECISION["reference_pmax_dbm"]) / 20.0)
              for power in POWERS_DBM}
    raw = {name: {} for name in RUNS}
    controls = {
        "max_unit_modulus_error": 0.0,
        "max_rate_path_deviation": 0.0,
        "max_rebuild_deviation_bps_hz": 0.0,
        "max_power_excess_ratio": 0.0,
        "min_monotone_step_bps_hz": float("inf"),
        "nonfinite_values": 0,
        "nonfinite_ceilings": 0,
    }

    for name, (run_dir, checkpoint_name) in RUNS.items():
        with open(run_dir / "summary.json", encoding="utf-8") as handle:
            current = json.load(handle)["config"]
        if config is not None:
            for key in ("M", "N", "L", "K", "AP", "batch_size", "assoc_threshold",
                        "pmax_dbm", "seed"):
                if current[key] != config[key]:
                    raise ValueError(f"{name} has a different {key}")
        config = current
        if float(config["pmax_dbm"]) != DECISION["reference_pmax_dbm"]:
            raise ValueError("checkpoints were not trained at the declared reference power")

        seed_everything(config["seed"])
        simulator = ChannelSimulator(
            config["M"], config["N"], config["L"], config["batch_size"],
            n_ap=config["AP"],
        )
        model = build_model(config, simulator, device)
        checkpoint = run_dir / "checkpoints" / checkpoint_name
        if not checkpoint.exists():
            checkpoint = run_dir / "models" / checkpoint_name
        load_checkpoint(model, str(checkpoint), device)
        model.eval()

        rebuilt = {}
        for power in DECISION["rebuild_check_powers_dbm"]:
            reference = dict(config, pmax_dbm=power)
            other = build_model(reference, simulator, device)
            load_checkpoint(other, str(checkpoint), device)
            other.eval()
            rebuilt[power] = other

        records = {f"{power:g}_{mode}": [] for power in POWERS_DBM for mode in MODES}
        records["ceiling_centralized"] = []
        records["ceiling_decentralized"] = []
        records["decentralized_random_phase"] = []

        pmax_linear = 10.0 ** ((DECISION["reference_pmax_dbm"] - 30.0) / 10.0)
        with temporary_seed(eval_seed), torch.no_grad():
            for index in range(samples // config["batch_size"]):
                features, edges, masks, direct, _ = simulator.training_batch(
                    config["K"], config["assoc_threshold"], config["assoc_threshold"]
                )
                local = simulator.decentralized_batch(
                    config["K"], config["assoc_threshold"], config["assoc_threshold"],
                    regenerate_channels=False,
                )
                precompute = RatePrecompute(simulator, device)

                solutions = {}
                solutions["centralized"] = model.centralized(
                    features.to(device), edges.to(device), masks, direct.to(device)
                )
                solutions["decentralized"] = model.decentralized(
                    [tensor.to(device) for tensor in local[0]],
                    [tensor.to(device) for tensor in local[1]],
                    local[2],
                    [tensor.to(device) for tensor in local[3]],
                )

                for arm, (beamformer, phase) in solutions.items():
                    controls["max_unit_modulus_error"] = max(
                        controls["max_unit_modulus_error"],
                        float((phase.norm(dim=-1) - 1.0).abs().max()),
                    )
                    controls["max_power_excess_ratio"] = max(
                        controls["max_power_excess_ratio"],
                        float((block_power(beamformer, config["AP"]) / pmax_linear).max()),
                    )
                    swept = sweep_rates(beamformer, phase, precompute, scales, num_bits)
                    for power in POWERS_DBM:
                        continuous = swept[(power, "continuous")]
                        discrete = swept[(power, "2bit")]
                        records[f"{power:g}_{arm}"].append(continuous.cpu().numpy())
                        records[f"{power:g}_{arm}_2bit"].append(discrete.cpu().numpy())
                        if not (torch.isfinite(continuous).all()
                                and torch.isfinite(discrete).all()):
                            controls["nonfinite_values"] += 1
                    series = np.stack(
                        [swept[(power, "continuous")].cpu().numpy() for power in POWERS_DBM]
                    )
                    controls["min_monotone_step_bps_hz"] = min(
                        controls["min_monotone_step_bps_hz"], float(np.diff(series, axis=0).min())
                    )
                    limit = ceiling_rate(beamformer, phase, precompute)
                    if not torch.isfinite(limit).all():
                        controls["nonfinite_ceilings"] += 1
                    records[f"ceiling_{arm}"].append(limit.cpu().numpy())

                beamformer, phase = solutions["decentralized"]
                records["decentralized_random_phase"].append(
                    precompute.sum_rate(beamformer, random_phase_like(phase)).cpu().numpy()
                )

                # The cached evaluator must agree with the deployed rate path.
                _, reference, _ = simulator.loss(beamformer, phase, device)
                controls["max_rate_path_deviation"] = max(
                    controls["max_rate_path_deviation"],
                    abs(float(precompute.sum_rate(beamformer, phase).mean())
                        - float(reference)),
                )

                # The scaling identity must reproduce a genuine rebuild at the extremes.
                if index < DECISION["rebuild_check_batches"]:
                    for power, other in rebuilt.items():
                        for arm in ("centralized", "decentralized"):
                            if arm == "centralized":
                                check = other.centralized(
                                    features.to(device), edges.to(device), masks,
                                    direct.to(device),
                                )
                            else:
                                check = other.decentralized(
                                    [tensor.to(device) for tensor in local[0]],
                                    [tensor.to(device) for tensor in local[1]],
                                    local[2],
                                    [tensor.to(device) for tensor in local[3]],
                                )
                            fast = solutions[arm][0] * scales[power]
                            controls["max_rebuild_deviation_bps_hz"] = max(
                                controls["max_rebuild_deviation_bps_hz"],
                                float((precompute.sum_rate(check[0], check[1])
                                       - precompute.sum_rate(fast, solutions[arm][1]))
                                      .abs().max()),
                            )

        raw[name] = {
            "checkpoint": str(checkpoint),
            "arrays": {key: np.concatenate(value) for key, value in records.items()},
        }
        print(f"[{name}] "
              + ", ".join(f"{power:g}dBm={raw[name]['arrays'][f'{power:g}_decentralized'].mean():.4f}"
                          for power in POWERS_DBM), flush=True)

    return raw, controls, config


def summarize(raw, controls, config, batch_size):
    def clusters(values):
        return np.asarray(values, dtype=np.float64).reshape(-1, batch_size).mean(axis=1)

    models = {}
    for name, entry in raw.items():
        per_power = {}
        for power in POWERS_DBM:
            arrays = entry["arrays"]
            centralized = clusters(arrays[f"{power:g}_centralized"])
            decentralized = clusters(arrays[f"{power:g}_decentralized"])
            per_power[f"{power:g}"] = {
                "centralized": cluster_summary(centralized),
                "centralized_2bit": cluster_summary(clusters(arrays[f"{power:g}_centralized_2bit"])),
                "decentralized": cluster_summary(decentralized),
                "decentralized_2bit": cluster_summary(clusters(arrays[f"{power:g}_decentralized_2bit"])),
                "centralized_minus_decentralized": cluster_summary(centralized - decentralized),
                "relative_decentralization_gap": cluster_summary(
                    (centralized - decentralized) / centralized
                ),
            }
        models[name] = {
            "checkpoint": entry["checkpoint"],
            "per_power": per_power,
            "interference_limited_ceiling": {
                "centralized": cluster_summary(clusters(entry["arrays"]["ceiling_centralized"])),
                "decentralized": cluster_summary(clusters(entry["arrays"]["ceiling_decentralized"])),
            },
            "decentralized_random_phase_at_reference": cluster_summary(
                clusters(entry["arrays"]["decentralized_random_phase"])
            ),
        }

    contrasts = {}
    for power in POWERS_DBM:
        entry = {}
        for mode in ("decentralized", "decentralized_2bit", "centralized"):
            g2 = clusters(raw["g2"]["arrays"][f"{power:g}_{mode}"])
            r0 = clusters(raw["r0"]["arrays"][f"{power:g}_{mode}"])
            entry[f"g2_minus_r0_{mode}"] = paired_difference(g2, r0)
        contrasts[f"{power:g}"] = entry

    gates = {
        "unit_modulus": controls["max_unit_modulus_error"] <= DECISION["max_unit_modulus_error"],
        "rate_path_equivalence": (
            controls["max_rate_path_deviation"] <= DECISION["max_rate_path_deviation"]
        ),
        "scaling_identity": (
            controls["max_rebuild_deviation_bps_hz"]
            <= DECISION["max_rebuild_deviation_bps_hz"]
        ),
        "per_ap_power_feasible": (
            controls["max_power_excess_ratio"] <= DECISION["max_power_excess_ratio"]
        ),
        "monotone_in_power": (
            controls["min_monotone_step_bps_hz"] >= -DECISION["monotonicity_tolerance_bps_hz"]
        ),
        # The sigma^2 -> 0 ceiling is a descriptive limit, so its finiteness is
        # recorded as a diagnostic count rather than gating the swept results.
        "finite_values": controls["nonfinite_values"] == 0,
    }
    persists = all(
        contrasts[f"{power:g}"][f"g2_minus_r0_{mode}"]["ci_low"] > 0.0
        for power in POWERS_DBM
        for mode in ("decentralized", "decentralized_2bit")
    )
    return {
        "models": models,
        "contrasts": contrasts,
        "controls": controls,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "advantage_persists_at_every_power": persists,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--eval_seed", type=int, default=20261002)
    parser.add_argument("--num_bits", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.samples < 16 or args.samples % 8:
        parser.error("samples must be at least two batches and divisible by eight")

    device = resolve_device(args.device)
    raw, controls, config = run(args.samples, args.eval_seed, device, args.num_bits)
    summary = summarize(raw, controls, config, config["batch_size"])
    summary.update({
        "experiment": "sum rate versus the common per-AP transmit power budget",
        "layout": "T0",
        "decision": DECISION,
        "eval_seed": args.eval_seed,
        "samples": args.samples,
        "batch_size": config["batch_size"],
        "num_bits": args.num_bits,
        "device": str(device),
        "config": config,
    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(args.out.with_suffix(".npz"), **{
        f"{name}_{key}": values
        for name, entry in raw.items() for key, values in entry["arrays"].items()
    })
    if not summary["all_gates_pass"]:
        raise ControlFailure(f"declared control failed: {summary['gates']}")
    print(json.dumps(summary["gates"], indent=2))
    for power in POWERS_DBM:
        contrast = summary["contrasts"][f"{power:g}"]["g2_minus_r0_decentralized"]
        print(f"{power:g} dBm: G2-R0 = {contrast['mean']:+.4f} "
              f"[{contrast['ci_low']:+.4f},{contrast['ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
