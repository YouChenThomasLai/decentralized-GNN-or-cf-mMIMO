"""E12: the mandatory centralized/distributed model-based joint-optimization pair.

Benchmark item 5 of the plan in `doc/research_positioning.md`.  Two complete
model-based methods solve the *same* unweighted sum-rate problem as the learned
arms, on the same paired holdout channels, under the same per-AP power budget,
unit-modulus constraint and association mask:

    arm            information                       coordination
    ---------------------------------------------------------------------------
    centralized    full CSI at the CPU               one upload, one download
    distributed    AP-local CSI only                 incremental ring ADMM

Both follow Huang et al. [7]: the same fractional-programming surrogate, the
same eq. (11) precoder with a bisection on the power multiplier, and the same MM
reflection update, so the two arms differ in information and coordination and
not in their solver.  `model_based` holds the mathematics; this program owns the
evaluation protocol, the declared controls, and the signaling ledger.

No cross-method comparison table is produced here.  This report establishes the
model-based arms themselves; the Stage-A table is assembled only once every
complete-method group exists.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import model_based as mb
from evaluate import resolve_device, seed_everything
from rates import RatePrecompute, phase_levels, quantize_phase
from simulation import ChannelSimulator


ARMS = ("centralized", "distributed")
PHASES = ("continuous", "2bit")
PRIMARY_PHASE = "continuous"

# Fixed before the holdout run.  The penalty scale and the multiplier schedule
# are the two quantities Huang et al. leave unspecified; both were selected on a
# calibration seed that is disjoint from every evaluation seed, by the rule
# recorded in `rho_selection_rule`.  A run records the exact source revision;
# the repository history, rather than this comment, is the audit trail.
DECISION = {
    "primary_phase": PRIMARY_PHASE,
    "source_algorithm": "Huang et al. [7], Algorithm 1, and its centralized counterpart",
    "source_fallback_reason": (
        "the declared first choice, Xu et al. [8], solves its reflection "
        "subproblem with a learned CNN block and gives no conventional solver "
        "for it, so a non-unrolled version of that paper does not exist"
    ),
    "rho_selection_rule": (
        "smallest scale in the declared grid whose worst-case consensus residual "
        "on the calibration seed is at most max_consensus_residual; ties resolved "
        "towards the smaller penalty"
    ),
    "rho_grid": [0.05, 0.1, 0.2, 0.4, 0.8],
    "dual_schedule": "sweep",
    "dual_schedule_note": (
        "one multiplier update per full pass over the APs.  Updating it at every "
        "activation, as eq. (8e) reads literally, multiplies the dual step by L "
        "and diverged in the original calibration; the corrected calibration "
        "therefore evaluates only the maintained per-sweep schedule"
    ),
    "initializations": ["ones", "random", "matched"],
    "primary_initialization": "matched",
    "max_consensus_residual": 0.05,
    "max_power_overshoot": 1e-6,
    "max_unit_modulus_error": 1e-5,
    "max_quantization_grid_error": 1e-6,
    "max_rate_path_deviation": 2e-5,
    "max_monotonicity_violation": 1e-6,
    "max_phase_block_violation": 1e-6,
    "max_locality_deviation": 0.0,
    "max_precision_deviation": 1e-2,
    "divergence_rule": (
        "a sample diverged if its final primal residual exceeds the residual "
        "after its first sweep"
    ),
    "max_divergence_rate": 0.0,
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


def collect(simulator, config, samples, eval_seed, device, dtype, control_batches=1):
    """Draw the shared holdout and keep it as complex tensors.

    The RNG is advanced exactly as `evaluate.evaluate_model` advances it — one
    `training_batch` per batch and nothing else — so the channel samples are the
    same tensors every learned arm is scored on at this seed.  The first batches
    keep a `RatePrecompute` snapshot so the complex rate path used by the solvers
    can be checked against the maintained one.
    """
    batch_size = config["batch_size"]
    if samples <= 0 or samples % batch_size:
        raise ValueError("samples must be a positive multiple of the batch size")
    users = config["K"]
    threshold = config.get("assoc_threshold", 0.1)

    seed_everything(eval_seed)
    ris, direct, served, references = [], [], [], []
    for index in range(samples // batch_size):
        simulator.training_batch(users, threshold, threshold)
        block = mb.channel_tensors(simulator, device, dtype)
        ris.append(block[0])
        direct.append(block[1])
        served.append(block[2])
        if index < control_batches:
            references.append(RatePrecompute(simulator, device))
    return (
        torch.cat(ris, dim=0),
        torch.cat(direct, dim=0),
        torch.cat(served, dim=0),
        references,
    )


def solve(arm, ris, direct, served, pmax, args, init=None, dtype=None, **overrides):
    """Run one arm on one chunk and return its solver record."""
    if dtype is not None and dtype != ris.dtype:
        ris, direct = ris.to(dtype), direct.to(dtype)
    settings = dict(
        pmax=pmax,
        tolerance=args.tolerance,
        bisect_iters=args.bisect_iters,
        mm_iterations=args.mm_iterations,
        exact_curvature=args.exact_curvature,
        init=init or args.init,
        trace=True,
    )
    settings.update(overrides)
    if arm == "centralized":
        settings.setdefault("max_iterations", args.max_iterations)
        settings.setdefault("inner_cycles", args.inner_cycles)
        return mb.centralized_fp(ris, direct, served, **settings)
    settings.setdefault("max_sweeps", args.max_sweeps)
    settings.setdefault("inner_cycles", 1)
    settings.setdefault("rho_scale", args.rho_scale)
    settings.setdefault("dual_period", args.dual_period)
    return mb.incremental_admm(ris, direct, served, **settings)


def chunked(arm, ris, direct, served, pmax, args, chunk, **kwargs):
    """Solve in fixed-size chunks and concatenate the per-sample records."""
    records = []
    for start in range(0, ris.shape[0], chunk):
        stop = min(start + chunk, ris.shape[0])
        records.append(
            solve(
                arm,
                ris[start:stop],
                direct[start:stop],
                served[start:stop],
                pmax,
                args,
                **kwargs,
            )
        )
    merged = {}
    width = min(record["trace"].shape[1] for record in records)
    for key in records[0]:
        values = [record[key] for record in records]
        if key == "activations":
            merged[key] = max(values)
        elif isinstance(values[0], torch.Tensor):
            if values[0].dim() > 1:
                values = [value[:, :width] for value in values]
            merged[key] = torch.cat(values, dim=0)
        else:
            merged[key] = max(values)
    return merged


def deterministic_phase(shape, device, dtype, seed=0):
    """Fixed random unit-modulus phase, used only as a smoke control."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    angles = 2 * math.pi * torch.rand(shape, generator=generator)
    return torch.complex(angles.cos(), angles.sin()).to(device=device, dtype=dtype)


def score(record, ris, direct, served, pmax, num_bits, args, controls, levels):
    """Rates and feasibility for one solved arm, in every reported phase setting."""
    weights = record["weights"]
    rates, phases = {}, {}
    phases["continuous"] = record["v"]
    theta = mb.to_theta(record["v"])
    rounded = quantize_phase(theta, num_bits)
    phases["2bit"] = mb.from_theta(rounded).to(record["v"].dtype)

    for name, v in phases.items():
        gain = mb.effective_channels(ris, direct, v)
        rates[name] = mb.sum_rate(gain, weights)
        controls["max_unit_modulus_error"] = max(
            controls["max_unit_modulus_error"], float((v.abs() - 1.0).abs().max())
        )
        controls["nonfinite_values"] += int((~torch.isfinite(rates[name])).sum())

    grid = (rounded.unsqueeze(-2) - levels).norm(dim=-1).min(dim=-1).values
    controls["max_quantization_grid_error"] = max(
        controls["max_quantization_grid_error"], float(grid.max())
    )

    # Secondary, clearly separated from the maintained 2-bit rule: an optimizer
    # can re-derive its beamformer against the phase it will actually apply.
    resolved = mb.resolve_beamformer(
        ris, direct, served, phases["2bit"], pmax,
        rounds=args.resolve_rounds, inner_cycles=args.inner_cycles,
        bisect_iters=args.bisect_iters,
    )
    rates["2bit_resolved"] = mb.sum_rate(
        mb.effective_channels(ris, direct, phases["2bit"]), resolved
    )

    random_v = deterministic_phase(record["v"].shape, ris.device, record["v"].dtype)
    rates["random_phase"] = mb.sum_rate(
        mb.effective_channels(ris, direct, random_v), weights
    )

    power = mb.per_ap_power(weights)
    controls["max_power_overshoot"] = max(
        controls["max_power_overshoot"], float((power / pmax - 1.0).max())
    )
    idle = ~served
    controls["unserved_column_leakage"] = max(
        controls["unserved_column_leakage"],
        float((weights.abs() * idle.unsqueeze(-1)).max()) if bool(idle.any()) else 0.0,
    )
    controls["nonfinite_values"] += int((~torch.isfinite(weights)).sum())
    return rates, phases


def rate_path_deviation(record, phases, references, batch_size, ris, direct):
    """Complex solver rate vs. the maintained `rates` path, on the first batches.

    A transposed block, a dropped conjugate, or a mismatched phase convention
    would still produce plausible numbers inside `model_based`; this compares
    against the same code every learned arm is scored with.
    """
    deviation = 0.0
    embedded = mb.embed_beamformer(record["weights"])
    for index, precompute in enumerate(references):
        start, stop = index * batch_size, (index + 1) * batch_size
        for v in phases.values():
            theta = mb.to_theta(v[start:stop])
            reference = precompute.sum_rate_from_effective(
                embedded[start:stop], precompute.effective_channel(theta)
            )
            mine = mb.sum_rate(
                mb.effective_channels(
                    ris[start:stop], direct[start:stop], v[start:stop]
                ),
                record["weights"][start:stop],
            )
            deviation = max(deviation, float((reference - mine).abs().max()))
    return deviation


def locality_deviation(ris, direct, served, pmax, args):
    """An AP's local update must depend on its peers only through the aggregate.

    Peer channel matrices are replaced by noise while the published aggregate is
    held fixed.  If either local block read anything beyond its own channels and
    that aggregate, the recomputed beamformer or reflection copy would move.
    """
    batch, n_ap = served.shape[0], served.shape[1]
    n_ris, n_elements = ris.shape[3], ris.shape[5]
    v = mb.initial_phase(ris, direct, served, args.init)
    gain = mb.effective_channels(ris, direct, v)
    weights = mb.mrt_initial(gain, served, pmax)
    blocks = mb.contributions(gain, weights)
    generator = torch.Generator(device="cpu").manual_seed(0)

    deviation = 0.0
    for index in range(n_ap):
        rest = blocks.sum(dim=1) - blocks[:, index]

        def local(source_ris, source_direct):
            own_gain = mb.effective_channels(
                source_ris[:, index : index + 1],
                source_direct[:, index : index + 1],
                v,
            )
            gamma, xi = mb.fp_auxiliaries(rest + mb.aggregate(own_gain, weights[:, index : index + 1]))
            block = mb.beamformer_step(
                own_gain, weights[:, index : index + 1], gamma, xi,
                served[:, index : index + 1], pmax, 1, args.bisect_iters, rest=rest,
            )
            system, cross = mb.phase_quadratic(
                source_ris[:, index : index + 1],
                source_direct[:, index : index + 1],
                block, gamma, xi, rest=rest,
            )
            return block, mb.mm_phase_step(
                system, cross, v.reshape(batch, -1), args.mm_iterations
            )

        reference_block, reference_phase = local(ris, direct)
        noisy_ris, noisy_direct = ris.clone(), direct.clone()
        for other in range(n_ap):
            if other == index:
                continue
            shape = noisy_ris[:, other].shape
            noise = torch.randn(shape + (2,), generator=generator)
            noisy_ris[:, other] = torch.complex(noise[..., 0], noise[..., 1]).to(
                device=ris.device, dtype=ris.dtype
            )
            shape = noisy_direct[:, other].shape
            noise = torch.randn(shape + (2,), generator=generator)
            noisy_direct[:, other] = torch.complex(noise[..., 0], noise[..., 1]).to(
                device=ris.device, dtype=ris.dtype
            )
        candidate_block, candidate_phase = local(noisy_ris, noisy_direct)
        deviation = max(
            deviation,
            float((candidate_block - reference_block).abs().max()),
            float((candidate_phase - reference_phase).abs().max()),
        )
    return deviation


def signaling_ledger(config, num_bits, sweeps, n_edges):
    """Counted online payloads per topology sample, in real values and in bits.

    UE->AP CSI acquisition sits outside the boundary and is listed as excluded
    rather than zero. G2's paper-decentralized view can additionally require
    delivery of shared-UE cross-AP CSI; that delivery is also outside this ledger.
    The centralized arm pays
    one CSI upload and one precoder download; the distributed arm pays no CSI
    upload but one ring message per activation, and one AP->CPU delivery of the
    agreed phase so the controller can actuate the surfaces.
    """
    n_ap, n_ris = config["AP"], config["L"]
    n_elements, n_antennas, users = config["N"], config["M"], config["K"]
    cascade = n_ris * users * n_antennas * n_elements
    upload = 2 * n_ap * (cascade + users * n_antennas)
    download = 2 * n_ap * users * n_antennas
    actuation = n_ris * n_elements
    ring = 2 * (n_edges * n_ris * n_elements + 2 * users * users)
    activations = n_ap * sweeps
    return {
        "accounting_boundary": (
            "AP->CPU CSI upload, CPU->AP precoder download, AP->AP ring messages, "
            "AP->CPU agreed phase, CPU->RIS actuation; UE->AP CSI acquisition "
            "and G2 shared-UE cross-AP CSI delivery excluded"
        ),
        "units": "real values per sample; fp32 unless a bit width is named",
        "centralized": {
            "ap_to_cpu_values": upload,
            "cpu_to_ap_values": download,
            "ap_to_ap_values": 0,
            "cpu_to_ris_values": actuation,
            "online_rounds": 2,
            "total_online_values": upload + download + actuation,
            "total_online_bits_fp32": 32 * (upload + download + actuation),
            "cpu_to_ris_bits": {
                "fp32": 32 * actuation, f"{num_bits}bit": num_bits * actuation
            },
        },
        "distributed": {
            "ap_to_cpu_values": n_ris * n_elements,
            "cpu_to_ap_values": 0,
            "ap_to_ap_values_per_activation": ring,
            "activations": activations,
            "ap_to_ap_values": ring * activations,
            "cpu_to_ris_values": actuation,
            "online_rounds": activations + 1,
            "total_online_values": ring * activations + n_ris * n_elements + actuation,
            "total_online_bits_fp32": 32 * (
                ring * activations + n_ris * n_elements + actuation
            ),
            "message_composition": {
                "consensus_residual_t": 2 * n_edges * n_ris * n_elements,
                "direct_aggregate_phi": 2 * users * users,
                "reflected_aggregate_psi": 2 * users * users,
            },
        },
    }


def timing(simulator, config, device, dtype, args, samples, realized=None):
    """Batch-1 per-iteration solve cost after warm-up, on one named device.

    A converged batch-1 solve takes minutes here, so timing hundreds of them
    would spend more device time than the experiment itself.  What transfers is
    the cost of one outer iteration, and the realized iteration count, so this
    measures a short fixed prefix of each solve and reports the per-unit cost;
    `end_to_end_estimate_ms` multiplies it by the realized count from the
    holdout.  The estimate is an estimate, and is labelled as one.
    """
    users = config["K"]
    threshold = config.get("assoc_threshold", 0.1)
    pmax = 10 ** ((config["pmax_dbm"] - 30) / 10)
    units = {"centralized": args.timing_iterations, "distributed": args.timing_sweeps}
    records = {arm: [] for arm in ARMS}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for index in range(samples + 1):
        simulator.training_batch(users, threshold, threshold)
        ris, direct, served = mb.channel_tensors(simulator, device, dtype)
        ris, direct, served = ris[:1], direct[:1], served[:1]
        for arm in ARMS:
            budget = (
                {"max_iterations": units[arm]}
                if arm == "centralized"
                else {"max_sweeps": units[arm]}
            )
            sync()
            start = time.perf_counter()
            # tolerance 0 disables early stopping, so every sample times the same
            # number of iterations and the per-unit cost is comparable.
            solve(arm, ris, direct, served, pmax, args, tolerance=0.0, **budget)
            sync()
            if index == 0:                                               # warm-up
                continue
            records[arm].append((time.perf_counter() - start) / units[arm])

    report = {}
    for arm, value in records.items():
        unit = "iteration" if arm == "centralized" else "sweep"
        report[arm] = {
            "unit": unit,
            f"median_ms_per_{unit}": float(np.median(value) * 1e3),
            f"p95_ms_per_{unit}": float(np.quantile(value, 0.95) * 1e3),
            "timed_units_per_sample": units[arm],
            "samples": len(value),
        }
        if realized and arm in realized:
            report[arm]["realized_units"] = realized[arm]
            report[arm]["end_to_end_estimate_ms"] = float(
                np.median(value) * 1e3 * realized[arm]
            )
    return report


def calibrate(simulator, config, device, dtype, args):
    """Choose the penalty scale and the multiplier schedule off the holdout.

    Huang et al. specify neither.  The grid is declared in `DECISION`, the seed
    here is disjoint from every evaluation seed, and the selected value is written
    to calibration.json before the holdout run.
    """
    pmax = 10 ** ((config["pmax_dbm"] - 30) / 10)
    ris, direct, served, _ = collect(
        simulator, config, args.calibration_samples, args.calibration_seed,
        device, dtype, control_batches=0,
    )
    reference = solve("centralized", ris, direct, served, pmax, args)
    table = {"centralized_rate": float(reference["rate"].mean())}
    for schedule in (DECISION["dual_schedule"],):
        for scale in DECISION["rho_grid"]:
            record = solve(
                "distributed", ris, direct, served, pmax, args,
                rho_scale=scale, dual_period=schedule,
            )
            first = record["primal_trace"][:, 0]
            table[f"{schedule}_rho{scale:g}"] = {
                "rate": float(record["rate"].mean()),
                "retention": float(record["rate"].mean() / reference["rate"].mean()),
                "primal_residual_max": float(record["primal_residual"].max()),
                "primal_residual_median": float(record["primal_residual"].median()),
                "consensus_residual_max": float(record["consensus_residual"].max()),
                "divergence_rate": float(
                    (record["primal_residual"] > first).to(torch.float32).mean()
                ),
            }
            print(
                f"[calibrate] {schedule:10s} rho={scale:<5g} "
                f"rate={table[f'{schedule}_rho{scale:g}']['rate']:.4f} "
                f"consensus={table[f'{schedule}_rho{scale:g}']['consensus_residual_max']:.4f}",
                flush=True,
            )

    # No fallback is defined on purpose.  If no grid point reaches the consensus
    # gate the selection returns None and the maintained script stops, so the
    # owner decides what to do instead of the threshold quietly widening.
    schedule = DECISION["dual_schedule"]
    feasible = [
        scale for scale in DECISION["rho_grid"]
        if table[f"{schedule}_rho{scale:g}"]["consensus_residual_max"]
        <= DECISION["max_consensus_residual"]
        and table[f"{schedule}_rho{scale:g}"]["divergence_rate"] == 0.0
    ]
    table["selected"] = {
        "dual_period": schedule,
        "rho_scale": min(feasible) if feasible else None,
        "rule": DECISION["rho_selection_rule"],
        "calibration_seed": args.calibration_seed,
        "calibration_samples": args.calibration_samples,
        "init": args.init,
    }
    return table


def evaluate(simulator, config, device, dtype, args):
    """Solve both arms on the shared holdout and apply the declared controls."""
    pmax = 10 ** ((config["pmax_dbm"] - 30) / 10)
    batch_size = config["batch_size"]
    levels = phase_levels(args.num_bits, device)
    ris, direct, served, references = collect(
        simulator, config, args.samples, args.eval_seed, device, dtype
    )

    controls = {
        "max_unit_modulus_error": 0.0,
        "max_quantization_grid_error": 0.0,
        "max_power_overshoot": 0.0,
        "max_rate_path_deviation": 0.0,
        "max_locality_deviation": None,
        "max_precision_deviation": None,
        "unserved_column_leakage": 0.0,
        "nonfinite_values": 0,
    }
    raw, solvers = {}, {}
    for arm in ARMS:
        print(f"[solve] {arm}", flush=True)
        start = time.perf_counter()
        record = chunked(arm, ris, direct, served, pmax, args, args.chunk)
        elapsed = time.perf_counter() - start
        rates, phases = score(
            record, ris, direct, served, pmax, args.num_bits, args, controls, levels
        )
        controls["max_rate_path_deviation"] = max(
            controls["max_rate_path_deviation"],
            rate_path_deviation(
                record, phases, references, batch_size, ris, direct
            ),
        )
        for name, value in rates.items():
            raw[f"{arm}_{name}"] = value.detach().cpu().numpy()

        key = "iterations" if arm == "centralized" else "sweeps"
        summary = {
            "wall_clock_s": elapsed,
            "realized_iterations": {
                "median": float(record[key].to(torch.float32).median()),
                "max": int(record[key].max()),
                "capped_fraction": float(
                    (~record["converged"]).to(torch.float32).mean()
                ),
            },
            "trace": record["trace"].mean(dim=0).detach().cpu().numpy().tolist(),
        }
        if arm == "centralized":
            summary.update(
                monotonicity_violation=record["monotonicity_violation"],
                phase_block_violation=record["phase_block_violation"],
            )
        else:
            first = record["primal_trace"][:, 0]
            summary.update(
                penalty_scale=args.rho_scale,
                dual_period=args.dual_period,
                penalty_median=float(record["penalty"].median()),
                primal_residual={
                    "median": float(record["primal_residual"].median()),
                    "max": float(record["primal_residual"].max()),
                },
                dual_residual={
                    "median": float(record["dual_residual"].median()),
                    "max": float(record["dual_residual"].max()),
                },
                consensus_residual={
                    "median": float(record["consensus_residual"].median()),
                    "max": float(record["consensus_residual"].max()),
                },
                divergence_rate=float(
                    (record["primal_residual"] > first).to(torch.float32).mean()
                ),
                activations=int(record["activations"]),
                primal_trace=record["primal_trace"].mean(dim=0).detach().cpu().numpy().tolist(),
            )
        solvers[arm] = summary

    controls["max_locality_deviation"] = locality_deviation(
        ris[:batch_size], direct[:batch_size], served[:batch_size], pmax, args
    )
    if args.precision_samples:
        # Re-solve a subset in the other working precision.  The solvers are
        # iterative and the phase updates are sign-like, so a result that moved
        # would mean the reported rate is a rounding artefact rather than a
        # property of the algorithm.
        stop = args.precision_samples
        other = torch.complex64 if dtype == torch.complex128 else torch.complex128
        deviation = 0.0
        for arm in ARMS:
            alternate = solve(
                arm, ris[:stop], direct[:stop], served[:stop], pmax, args,
                dtype=other,
            )
            rate = mb.sum_rate(
                mb.effective_channels(
                    ris[:stop].to(other), direct[:stop].to(other), alternate["v"]
                ),
                alternate["weights"],
            )
            deviation = max(
                deviation,
                float((rate.to(torch.float32) - torch.as_tensor(
                    raw[f"{arm}_continuous"][:stop], device=rate.device
                )).abs().max()),
            )
        controls["max_precision_deviation"] = deviation

    metrics = {key: clustered(value, batch_size) for key, value in raw.items()}
    served_counts = served.sum(dim=2).detach().cpu().numpy().ravel()
    return metrics, raw, controls, solvers, served_counts


def sensitivity(simulator, config, device, dtype, args):
    """Declared initialization sensitivity, on a subset and never on the decision."""
    pmax = 10 ** ((config["pmax_dbm"] - 30) / 10)
    ris, direct, served, _ = collect(
        simulator, config, args.init_samples, args.eval_seed, device, dtype,
        control_batches=0,
    )
    table = {}
    for init in DECISION["initializations"]:
        for arm in ARMS:
            record = solve(arm, ris, direct, served, pmax, args, init=init)
            table[f"{arm}_{init}"] = {
                "rate": float(record["rate"].mean()),
                "samples": int(ris.shape[0]),
            }
    return table


def decide(metrics, controls, solvers):
    """Apply the declared gates without re-reading them from the results."""
    gates = {
        "power_feasibility": controls["max_power_overshoot"]
        <= DECISION["max_power_overshoot"],
        "unit_modulus": controls["max_unit_modulus_error"]
        <= DECISION["max_unit_modulus_error"],
        "quantization_grid": controls["max_quantization_grid_error"]
        <= DECISION["max_quantization_grid_error"],
        "rate_path_equivalence": controls["max_rate_path_deviation"]
        <= DECISION["max_rate_path_deviation"],
        "locality": controls["max_locality_deviation"] is not None
        and controls["max_locality_deviation"] <= DECISION["max_locality_deviation"],
        "precision": controls["max_precision_deviation"] is None
        or controls["max_precision_deviation"] <= DECISION["max_precision_deviation"],
        "centralized_monotone": solvers["centralized"]["monotonicity_violation"]
        <= DECISION["max_monotonicity_violation"],
        "phase_block_monotone": solvers["centralized"]["phase_block_violation"]
        <= DECISION["max_phase_block_violation"],
        "consensus_feasible": solvers["distributed"]["consensus_residual"]["max"]
        <= DECISION["max_consensus_residual"],
        "no_divergence": solvers["distributed"]["divergence_rate"]
        <= DECISION["max_divergence_rate"],
        "no_unserved_column_leakage": controls["unserved_column_leakage"] == 0.0,
        "finite": controls["nonfinite_values"] == 0,
        "beats_random_phase": all(
            metrics[f"{arm}_{PRIMARY_PHASE}"]["mean"]
            > metrics[f"{arm}_random_phase"]["mean"]
            for arm in ARMS
        ),
    }
    centralized = metrics[f"centralized_{PRIMARY_PHASE}"]["mean"]
    distributed = metrics[f"distributed_{PRIMARY_PHASE}"]["mean"]
    return {
        "criteria": DECISION,
        "centralized_rate": centralized,
        "distributed_rate": distributed,
        "decentralization_gap": centralized - distributed,
        "decentralization_retention": distributed / centralized,
        "numerical_gates": gates,
        "all_gates_passed": all(gates.values()),
        "stage_a_model_based_pair_executed": True,
    }


def render(payload):
    lines = [
        "# E12 model-based joint-optimization pair",
        "",
        f"Holdout: {payload['samples']} paired samples, evaluation seed "
        f"{payload['eval_seed']}, device {payload['device_name']}.",
        "",
        "| arm | continuous | clustered SE | 2-bit | clustered SE | 2-bit resolved | random phase |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        cont = payload["metrics"][f"{arm}_continuous"]
        disc = payload["metrics"][f"{arm}_2bit"]
        lines.append(
            f"| {arm} | {cont['mean']:.4f} | {cont['clustered_sem']:.4f} | "
            f"{disc['mean']:.4f} | {disc['clustered_sem']:.4f} | "
            f"{payload['metrics'][f'{arm}_2bit_resolved']['mean']:.4f} | "
            f"{payload['metrics'][f'{arm}_random_phase']['mean']:.4f} |"
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

    solvers = payload["solvers"]
    lines += [
        "",
        "## Solver behaviour",
        "",
        f"- centralized: median {solvers['centralized']['realized_iterations']['median']:.0f} "
        f"iterations, {100 * solvers['centralized']['realized_iterations']['capped_fraction']:.1f}% "
        f"hit the cap, monotonicity violation "
        f"{solvers['centralized']['monotonicity_violation']:.3e}",
        f"- distributed: median {solvers['distributed']['realized_iterations']['median']:.0f} "
        f"sweeps ({solvers['distributed']['activations']} activations), primal residual "
        f"median {solvers['distributed']['primal_residual']['median']:.3e} / max "
        f"{solvers['distributed']['primal_residual']['max']:.3e}, consensus residual max "
        f"{solvers['distributed']['consensus_residual']['max']:.3e}, divergence rate "
        f"{solvers['distributed']['divergence_rate']:.3f}",
        "",
        "## Declared gates",
        "",
        "| gate | passed |",
        "|---|:-:|",
    ]
    for gate, passed in payload["decision"]["numerical_gates"].items():
        lines.append(f"| {gate} | {'yes' if passed else 'NO'} |")

    ledger = payload["signaling"]
    lines += [
        "",
        "## Signaling ledger, real values per sample",
        "",
        "| arm | AP→CPU | CPU→AP | AP→AP | CPU→RIS | rounds | total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = ledger[arm]
        lines.append(
            f"| {arm} | {row['ap_to_cpu_values']} | {row['cpu_to_ap_values']} | "
            f"{row['ap_to_ap_values']} | {row['cpu_to_ris_values']} | "
            f"{row['online_rounds']} | {row['total_online_values']} |"
        )

    lines += [
        "",
        "## Timing, batch 1 after warm-up",
        "",
        "| arm | unit | median (ms/unit) | p95 (ms/unit) | realized units | end-to-end estimate (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = payload["timing"][arm]
        unit = row["unit"]
        lines.append(
            f"| {arm} | {unit} | {row[f'median_ms_per_{unit}']:.1f} | "
            f"{row[f'p95_ms_per_{unit}']:.1f} | {row.get('realized_units', 0):.0f} | "
            f"{row.get('end_to_end_estimate_ms', 0) / 1e3:.1f} |"
        )

    lines += [
        "",
        "## Initialization sensitivity (diagnostic, excluded from the decision)",
        "",
        "| arm and initialization | rate |",
        "|---|---:|",
    ]
    for name, row in payload["initialization_sensitivity"].items():
        lines.append(f"| {name} | {row['rate']:.4f} |")

    decision = payload["decision"]
    lines += [
        "",
        "## Summary",
        "",
        f"- Centralized: {decision['centralized_rate']:.4f} bps/Hz",
        f"- Distributed: {decision['distributed_rate']:.4f} bps/Hz",
        f"- Gap: {decision['decentralization_gap']:+.4f}, retention "
        f"{decision['decentralization_retention']:.4f}",
        f"- All numerical gates passed: {decision['all_gates_passed']}",
        "",
    ]
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", help="JSON layout with AP and RIS coordinates")
    parser.add_argument(
        "--run",
        required=True,
        help="run directory whose summary.json supplies the system configuration; "
             "its checkpoint is never loaded",
    )
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--eval_seed", type=int, default=20260915)
    parser.add_argument("--replication_seed", type=int, default=20260918)
    parser.add_argument("--chunk", type=int, default=400)
    parser.add_argument("--num_bits", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--max_sweeps", type=int, default=120)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--inner_cycles", type=int, default=2)
    parser.add_argument("--mm_iterations", type=int, default=30)
    parser.add_argument("--bisect_iters", type=int, default=30)
    parser.add_argument(
        "--exact_curvature",
        action="store_true",
        help="use lambda_max(Z) instead of the certified row-sum bound in the MM step",
    )
    parser.add_argument("--resolve_rounds", type=int, default=8)
    parser.add_argument("--rho_scale", type=float, default=0.4)
    parser.add_argument("--dual_period", default="sweep", choices=("sweep", "activation"))
    parser.add_argument(
        "--init",
        default=DECISION["primary_initialization"],
        choices=("ones", "random", "matched"),
    )
    parser.add_argument("--init_samples", type=int, default=80)
    parser.add_argument("--precision_samples", type=int, default=40)
    parser.add_argument("--timing_samples", type=int, default=20)
    parser.add_argument("--timing_iterations", type=int, default=20)
    parser.add_argument("--timing_sweeps", type=int, default=4)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration_seed", type=int, default=20260912)
    parser.add_argument("--calibration_samples", type=int, default=80)
    parser.add_argument("--precision", default="double", choices=("single", "double"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out_dir",
        default="../../artifacts/decentralized_ris/e12_model_based_optimization",
    )
    return parser


def main():
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    dtype = torch.complex64 if args.precision == "single" else torch.complex128
    with open(os.path.join(args.run, "summary.json"), encoding="utf-8") as handle:
        config = json.load(handle)["config"]

    # Reset before construction so the AP-RIS LoS draw reproduces the training
    # topology, exactly as every other evaluator does.
    seed_everything(config["seed"])
    layout = None
    if args.layout:
        with open(args.layout, encoding="utf-8") as handle:
            layout = json.load(handle)
    simulator = ChannelSimulator(
        config["M"], config["N"], config["L"], config["batch_size"],
        n_ap=config["AP"], layout=layout,
    )
    os.makedirs(args.out_dir, exist_ok=True)

    if args.calibrate:
        table = calibrate(simulator, config, device, dtype, args)
        path = os.path.join(args.out_dir, "calibration.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(table, handle, indent=2)
        print(json.dumps(table["selected"], indent=2))
        print(f"[done] {path}")
        return

    seeds = {"primary": args.eval_seed}
    if args.replication_seed and args.replication_seed != args.eval_seed:
        seeds["replication"] = args.replication_seed

    evaluations, raw_all = {}, {}
    for role, seed in seeds.items():
        print(f"[evaluate] {role} seed={seed}", flush=True)
        scoped = argparse.Namespace(**vars(args))
        scoped.eval_seed = seed
        metrics, raw, controls, solvers, served = evaluate(
            simulator, config, device, dtype, scoped
        )
        table = {
            f"centralized_minus_distributed_{phase}": paired(
                raw[f"centralized_{phase}"], raw[f"distributed_{phase}"],
                config["batch_size"],
            )
            for phase in PHASES
        }
        for arm in ARMS:
            table[f"{arm}_continuous_minus_2bit"] = paired(
                raw[f"{arm}_continuous"], raw[f"{arm}_2bit"], config["batch_size"]
            )
        evaluations[role] = {
            "eval_seed": seed,
            "metrics": metrics,
            "paired_contrasts": table,
            "controls": controls,
            "solvers": solvers,
            "association": {
                "mean": float(served.mean()),
                "max": int(served.max()),
                "fraction_above_antennas": float((served > config["M"]).mean()),
            },
            "decision": decide(metrics, controls, solvers),
        }
        raw_all[role] = raw
        print(
            "  " + "  ".join(
                f"{arm}={metrics[f'{arm}_continuous']['mean']:.4f}" for arm in ARMS
            ),
            flush=True,
        )

    primary = argparse.Namespace(**vars(args))
    primary.eval_seed = args.eval_seed
    initialization = sensitivity(simulator, config, device, dtype, primary)
    seed_everything(args.eval_seed)
    realized = {
        arm: evaluations["primary"]["solvers"][arm]["realized_iterations"]["median"]
        for arm in ARMS
    }
    timings = timing(
        simulator, config, device, dtype, args, args.timing_samples, realized
    )

    sweeps = evaluations["primary"]["solvers"]["distributed"]["realized_iterations"]["max"]
    payload = {
        "experiment": "E12",
        "question": (
            "what do a centralized and a distributed model-based joint "
            "active/passive design achieve on the shared holdout?"
        ),
        "config_source": args.run,
        "config": config,
        "layout": layout,
        "samples": args.samples,
        "num_bits": args.num_bits,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "precision": args.precision,
        "solver_settings": {
            key: getattr(args, key)
            for key in (
                "max_iterations", "max_sweeps", "tolerance", "inner_cycles",
                "mm_iterations", "bisect_iters", "rho_scale", "dual_period", "init",
                "exact_curvature",
            )
        },
        "eval_seed": args.eval_seed,
        "seeds": seeds,
        "signaling": signaling_ledger(
            config, args.num_bits, sweeps, config["AP"]
        ),
        "timing": timings,
        "initialization_sensitivity": initialization,
        "evaluations": evaluations,
    }
    payload.update(
        metrics=evaluations["primary"]["metrics"],
        paired_contrasts=evaluations["primary"]["paired_contrasts"],
        controls=evaluations["primary"]["controls"],
        solvers=evaluations["primary"]["solvers"],
        association=evaluations["primary"]["association"],
        decision=evaluations["primary"]["decision"],
    )

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
