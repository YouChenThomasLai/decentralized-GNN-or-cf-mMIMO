import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from environment import MOBILITY_STRAIGHT, TOPOLOGY_TYPE, WRAP_AROUND, MobilityEnvironment
from feedback import SCHEDULERS, FeedbackState
from utils_return_indivial_rates import (
    HEIGHT_DIFFERENCE,
    SQUARE_SIDE,
    calculate_rates,
    rzf_beamforming,
)


NUM_AP = 5
SOURCE_FILES = (
    "environment.py",
    "utils_return_indivial_rates.py",
    "feedback.py",
    "evaluate.py",
    "test_stage4.py",
    "run_exp-v4.sh",
)
EXPECTED_STAGE2_SOURCE_SHA256 = {
    "data.py": "2e34c6c5f343b645a839f704c351ca715114fa55263ca84840b5ca712aa37de4",
    "environment.py": "1416de41ccd87983c8d9f0a9d0d3c9b22c7b29c026a3830523701c6726c2ba6e",
    "evaluate.py": "b67ed39304f9c34280373731868f6b12d47e0fb81f6b5edd009cabf2b983fe27",
    "model_2.py": "d916ebd5bcf4077b0ff6c9887d84aaf0d1eee30088d5720f8e770b9036225869",
    "run_exp-v2.sh": "5f4b4f25a7200155650712a29ded689b8beb3ab8bf387dd0b119316acaa9d3b0",
    "test_stage2.py": "7b5269fb80e753e178ad1aba702cc828122da223fc42643fcc3ba7b3451cc13a",
    "trainer_2.py": "ea1016a20afb09581cb65eead533b4ef1d5ce1474f8bd03683e87d60c2c8e045",
    "utils_return_indivial_rates.py": "530549069179451187843158068cad4f1ea218c44d5475a44ffc8e51698346a2",
}
EXPECTED_STAGE1C_CHECKPOINT_SHA256 = (
    "16dba87572cd9f9dfc3272b4faeda2d7127dc414945450b856758efdba7bba45"
)
EXPECTED_STAGE1C_AP_SHA256 = (
    "421b817e0e1e70b88db4e2ef685954b35f2e65432ffcb056190adb881df8b92d"
)
EXPECTED_STAGE1C_CONFIG_SHA256 = (
    "ace7fbe95ab86dab070de0a04b7517edd1924af1271446fc69d94e72cd98cf0b"
)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def configure_logging(out_dir):
    logger = logging.getLogger("stage4")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(out_dir / "run.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def save_provenance(out_dir, args):
    source_dir = Path(__file__).resolve().parent
    snapshot_dir = out_dir / "source_snapshot"
    snapshot_dir.mkdir()
    source_hashes = {}
    for filename in SOURCE_FILES:
        source = source_dir / filename
        shutil.copy2(source, snapshot_dir / filename)
        source_hashes[filename] = sha256(source)
    try:
        repository = source_dir.parents[1]
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=repository, text=True
        ).strip()
        dirty = subprocess.check_output(
            ("git", "status", "--short"), cwd=repository, text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = ["git status unavailable"]
    provenance = {
        "git_commit": commit,
        "git_dirty": bool(dirty),
        "git_status_short": dirty,
        "source_sha256": source_hashes,
        "stage2_source_sha256": EXPECTED_STAGE2_SOURCE_SHA256,
        "stage2_reference": args.stage2_reference,
        "stage2_reference_sha256": {
            filename: sha256(Path(args.stage2_reference) / filename)
            for filename in ("config.json", "provenance.json", "raw_metrics.npz")
        },
        "ap_coordinates_sha256": sha256(args.ap_coordinates),
        "stage1_config_sha256": sha256(args.stage1_config),
    }
    write_json(out_dir / "provenance.json", provenance)
    return provenance


def validate_stage2_handoff(args, provenance):
    reference = Path(args.stage2_reference)
    completion = json.loads((reference / "completion.json").read_text())
    config = json.loads((reference / "config.json").read_text())
    reference_provenance = json.loads((reference / "provenance.json").read_text())
    if completion.get("status") != "complete":
        raise RuntimeError("Stage 2 reference is not complete")
    for filename, expected in EXPECTED_STAGE2_SOURCE_SHA256.items():
        if reference_provenance["source_sha256"].get(filename) != expected:
            raise RuntimeError(f"Stage 2 source hash mismatch: {filename}")
    for filename in ("environment.py", "utils_return_indivial_rates.py"):
        if provenance["source_sha256"].get(filename) != EXPECTED_STAGE2_SOURCE_SHA256[filename]:
            raise RuntimeError(f"Frozen Stage 4 source copy changed: {filename}")
    expected_artifacts = {
        "checkpoint_sha256": EXPECTED_STAGE1C_CHECKPOINT_SHA256,
        "ap_coordinates_sha256": EXPECTED_STAGE1C_AP_SHA256,
        "stage1_config_sha256": EXPECTED_STAGE1C_CONFIG_SHA256,
    }
    for name, expected in expected_artifacts.items():
        if reference_provenance.get(name) != expected:
            raise RuntimeError(f"Stage 2 frozen artifact hash mismatch: {name}")
    for name in ("ap_coordinates_sha256", "stage1_config_sha256"):
        if provenance[name] != expected_artifacts[name]:
            raise RuntimeError(f"Stage 4 frozen artifact hash mismatch: {name}")

    reference_cli = config["cli"]
    fixed_fields = {
        "M": args.M,
        "K": args.K,
        "pmax_dbm": args.pmax_dbm,
        "noise_power": args.noise_power,
        "seed": args.seed,
        "speed_kmh": args.speed_kmh,
        "mobility_model": MOBILITY_STRAIGHT,
        "decision_period_s": args.decision_period_s,
        "carrier_frequency_hz": args.carrier_frequency_hz,
    }
    for name, expected in fixed_fields.items():
        actual = reference_cli.get(name)
        equal = actual == expected
        if isinstance(expected, float):
            equal = np.isclose(actual, expected, rtol=0, atol=0)
        if not equal:
            raise RuntimeError(
                f"Stage 2 reference config mismatch for {name}: {actual} != {expected}"
            )
    compatible = (
        reference_cli["episode_steps"] == args.episode_steps
        and reference_cli["trajectories"] == args.trajectories
    )
    return config, compatible


def save_environment(loader, out_dir):
    np.savez_compressed(
        out_dir / "environment.npz",
        mobility_model=np.asarray(loader.mobility_model),
        ap_coordinates=loader.BS_Loc_array,
        ue_positions=loader.ue_positions,
        ue_speeds_mps=loader.ue_speeds_mps,
        ue_directions_rad=loader.ue_directions_rad,
        instantaneous_speeds_mps=loader.instantaneous_speeds_mps,
        distances=loader.distances,
        path_loss_factors=loader.path_loss_factors,
        stage2_t0_association_mask=loader.association_mask,
        rhos=loader.rhos,
    )


def environment_matches_reference(loader, reference):
    names = {
        "ap_coordinates": loader.BS_Loc_array,
        "ue_positions": loader.ue_positions,
        "ue_speeds_mps": loader.ue_speeds_mps,
        "ue_directions_rad": loader.ue_directions_rad,
        "instantaneous_speeds_mps": loader.instantaneous_speeds_mps,
        "distances": loader.distances,
        "path_loss_factors": loader.path_loss_factors,
        "association_mask": loader.association_mask,
        "rhos": loader.rhos,
    }
    with np.load(Path(reference) / "environment.npz") as frozen:
        return all(np.array_equal(frozen[name], value) for name, value in names.items())


def build_cells(budgets, schedulers, num_users):
    budgets = tuple(dict.fromkeys(budgets))
    cells = []
    if num_users in budgets:
        cells.append({"label": f"full_current__B{num_users}", "budget": num_users,
                      "scheduler": "round_robin", "kind": "full_current"})
    if 0 in budgets:
        cells.append({"label": "hold_t0__B0", "budget": 0,
                      "scheduler": "round_robin", "kind": "hold_t0"})
    for budget in budgets:
        if budget in (0, num_users):
            continue
        for scheduler in schedulers:
            cells.append({"label": f"{scheduler}__B{budget}", "budget": budget,
                          "scheduler": scheduler, "kind": scheduler})
    return cells


def beamformer_checks(weights, association_mask, num_users, pmax_w):
    finite = bool(torch.isfinite(weights).all())
    flat = association_mask.transpose(0, 2, 1).reshape(
        len(association_mask), NUM_AP * num_users
    )
    mask = torch.as_tensor(flat, dtype=torch.bool, device=weights.device)
    mask = mask.unsqueeze(1).expand_as(weights)
    unassociated = weights[~mask]
    max_unassociated = (
        float(unassociated.abs().max().item()) if unassociated.numel() else 0.0
    )
    max_power = 0.0
    for ap in range(NUM_AP):
        block = weights[:, :, ap * num_users:(ap + 1) * num_users]
        max_power = max(
            max_power, float(block.square().sum(dim=(1, 2)).max().item())
        )
    return finite, max_unassociated, max_power


def evaluate_rates(
    stored_channels,
    true_channels,
    association_mask,
    time_indices,
    *,
    num_users,
    pmax_w,
    noise_power,
    device,
    frame_batch_size,
):
    trajectories = stored_channels.shape[0]
    trajectory_indices = np.repeat(np.arange(trajectories), len(time_indices))
    frame_indices = np.tile(time_indices, trajectories)
    rates = np.empty((len(trajectory_indices), num_users), dtype=np.float64)
    finite = True
    max_unassociated = 0.0
    max_power = 0.0
    construction_s = 0.0
    total_start = time.perf_counter()
    weight_digest = hashlib.sha256()
    with torch.inference_mode():
        for start in range(0, len(trajectory_indices), frame_batch_size):
            stop = min(start + frame_batch_size, len(trajectory_indices))
            trajectory_batch = trajectory_indices[start:stop]
            time_batch = frame_indices[start:stop]
            stored = stored_channels[trajectory_batch, time_batch]
            current = true_channels[trajectory_batch, time_batch]
            masks = association_mask[trajectory_batch]
            construction_start = time.perf_counter()
            weights = rzf_beamforming(
                stored, masks, pmax_w, device, noise_power
            )
            construction_s += time.perf_counter() - construction_start
            checks = beamformer_checks(weights, masks, num_users, pmax_w)
            finite &= checks[0]
            max_unassociated = max(max_unassociated, checks[1])
            max_power = max(max_power, checks[2])
            batch_rates = calculate_rates(
                weights, current, NUM_AP, device, noise_power
            ).cpu().numpy()
            finite &= bool(np.isfinite(batch_rates).all())
            rates[start:stop] = batch_rates
            weight_digest.update(weights.cpu().numpy().tobytes())
    rate_frames = rates.reshape(trajectories, len(time_indices), num_users)
    user_means = rate_frames.mean(axis=1)
    trajectory_sum_rates = user_means.sum(axis=1)
    return {
        "rate_frames": rate_frames,
        "user_time_average_rates": user_means,
        "trajectory_sum_rates": trajectory_sum_rates,
        "trajectory_p05_user_rate": np.percentile(user_means, 5, axis=1),
        "finite": finite,
        "max_unassociated_abs": max_unassociated,
        "max_ap_power_w": max_power,
        "constraints_passed": bool(
            finite and max_unassociated == 0 and max_power <= pmax_w + 1e-6
        ),
        "rzf_construction_s": construction_s,
        "total_evaluation_s": time.perf_counter() - total_start,
        "beamformer_sha256": weight_digest.hexdigest(),
    }


def simulate_cell(loader, cell, args, device, reference_time_indices=None):
    true_channels = loader.true_channels
    active = loader.association_mask.transpose(0, 2, 1)
    state = FeedbackState(
        true_channels[:, 0],
        loader.association_mask,
        budget=cell["budget"],
        scheduler=cell["scheduler"],
        scheduler_seed=args.scheduler_seed,
        t0_link_power=loader.path_loss_factors[:, 0] ** 2,
        rhos=loader.rhos[:, 0],
    )
    updates = np.zeros(active.shape[:1] + (loader.episode_steps,) + active.shape[1:], bool)
    updates[:, 0] = active
    ages = np.empty(updates.shape, dtype=np.int32)
    ages[:, 0] = state.ages
    priorities = np.zeros(updates.shape, dtype=np.float32)
    stored = np.empty_like(true_channels)
    stored[:, 0] = state.stored_channels
    nmse = np.zeros((loader.batch_size, loader.episode_steps), dtype=np.float64)

    for period in range(1, loader.episode_steps):
        update_mask = state.select_updates()
        priorities[:, period] = state.last_priority
        state.apply_updates(true_channels[:, period], update_mask)
        updates[:, period] = update_mask
        ages[:, period] = state.ages
        stored[:, period] = state.stored_channels
        error = np.abs(true_channels[:, period] - state.stored_channels) ** 2
        signal = np.abs(true_channels[:, period]) ** 2
        nmse[:, period] = (
            (error * active[..., None]).sum(axis=(1, 2, 3))
            / (signal * active[..., None]).sum(axis=(1, 2, 3))
        )

    time_indices = np.arange(0, loader.episode_steps, args.eval_time_stride)
    rate = evaluate_rates(
        stored,
        true_channels,
        loader.association_mask,
        time_indices,
        num_users=args.K,
        pmax_w=10 ** ((args.pmax_dbm - 30) / 10),
        noise_power=args.noise_power,
        device=device,
        frame_batch_size=args.batch_size,
    )
    reference_rate = None
    if reference_time_indices is not None:
        reference_rate = evaluate_rates(
            stored,
            true_channels,
            loader.association_mask,
            reference_time_indices,
            num_users=args.K,
            pmax_w=10 ** ((args.pmax_dbm - 30) / 10),
            noise_power=args.noise_power,
            device=device,
            frame_batch_size=args.batch_size,
        )

    steady_updates = updates[:, 1:]
    per_trajectory_updates = steady_updates.sum(axis=(1, 2, 3))
    active_opportunities = active.sum(axis=(1, 2)) * max(loader.episode_steps - 1, 0)
    actual_fraction = np.divide(
        per_trajectory_updates,
        active_opportunities,
        out=np.zeros(loader.batch_size),
        where=active_opportunities > 0,
    )
    age_metrics = {name: np.empty(loader.batch_size) for name in ("mean", "median", "p95", "max")}
    for trajectory in range(loader.batch_size):
        values = ages[trajectory][:, active[trajectory]]
        age_metrics["mean"][trajectory] = values.mean()
        age_metrics["median"][trajectory] = np.median(values)
        age_metrics["p95"][trajectory] = np.percentile(values, 95)
        age_metrics["max"][trajectory] = values.max()
    all_active_ages = ages[np.broadcast_to(active[:, None], ages.shape)]

    counts = steady_updates.sum(axis=-1)
    loads = active.sum(axis=-1)[:, None]
    budget_violations = int(np.count_nonzero(counts > cell["budget"]))
    fill_violations = int(
        np.count_nonzero(counts != np.minimum(cell["budget"], loads))
    )
    mask_violations = int(np.count_nonzero(steady_updates & ~active[:, None]))
    active_expanded = np.broadcast_to(active[:, None, :, :, None], stored.shape)
    state_boundary_passed = bool(
        np.all(ages[~np.broadcast_to(active[:, None], ages.shape)] == -1)
        and (
            cell["kind"] != "full_current"
            or all(
                np.all(ages[index][:, active[index]] == 0)
                for index in range(loader.batch_size)
            )
        )
        and (cell["kind"] != "full_current" or np.array_equal(
            stored[active_expanded], true_channels[active_expanded]
        ))
    )
    summary = {
        "kind": cell["kind"],
        "scheduler": cell["scheduler"],
        "budget": cell["budget"],
        "trajectory_average_sum_rate": float(rate["trajectory_sum_rates"].mean()),
        "trajectory_average_p05_user_rate": float(rate["trajectory_p05_user_rate"].mean()),
        "actual_update_fraction": float(
            per_trajectory_updates.sum() / active_opportunities.sum()
        ),
        "updates_per_ap_frame": float(
            per_trajectory_updates.mean()
            / max((loader.episode_steps - 1) * NUM_AP, 1)
        ),
        "mean_age_frames": float(all_active_ages.mean()),
        "median_age_frames": float(np.median(all_active_ages)),
        "p95_age_frames": float(np.percentile(all_active_ages, 95)),
        "max_age_frames": float(all_active_ages.max()),
        "mean_csi_nmse": float(nmse.mean()),
        "p95_csi_nmse": float(np.percentile(nmse, 95)),
        "budget_violations": budget_violations,
        "fill_violations": fill_violations,
        "association_violations": mask_violations,
        "finite": bool(rate["finite"] and np.isfinite(nmse).all()),
        "max_unassociated_abs": rate["max_unassociated_abs"],
        "max_ap_power_w": rate["max_ap_power_w"],
        "constraints_passed": bool(
            rate["constraints_passed"]
            and budget_violations == fill_violations == mask_violations == 0
            and state_boundary_passed
        ),
        "rzf_construction_s": rate["rzf_construction_s"],
        "total_evaluation_s": rate["total_evaluation_s"],
    }
    return {
        "summary": summary,
        "updates": updates,
        "ages": ages,
        "priorities": priorities,
        "nmse": nmse,
        "actual_fraction_per_trajectory": actual_fraction,
        "age_metrics": age_metrics,
        "rate": rate,
        "reference_rate": reference_rate,
        "stored_sha256": hashlib.sha256(stored.tobytes()).hexdigest(),
    }


def run(args, out_dir, logger, provenance):
    seed_everything(args.seed)
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
        logger.info("CUDA unavailable; using CPU")
    stage2_config, reference_compatible = validate_stage2_handoff(args, provenance)
    ap_coordinates = np.loadtxt(args.ap_coordinates)
    loader = MobilityEnvironment(
        args.M,
        args.trajectories,
        episode_steps=args.episode_steps,
        speed_kmh=args.speed_kmh,
        decision_period_s=args.decision_period_s,
        carrier_frequency_hz=args.carrier_frequency_hz,
        seed=args.seed,
        mobility_model=MOBILITY_STRAIGHT,
        bs_locations=ap_coordinates,
    ).generate_trajectories(args.K, 0.1)
    save_environment(loader, out_dir)
    environment_reproduced = (
        environment_matches_reference(loader, args.stage2_reference)
        if reference_compatible else None
    )
    if environment_reproduced is False:
        raise RuntimeError("Stage 2 environment reproduction failed")

    cells = build_cells(args.budgets, args.schedulers, args.K)
    config = {
        "execution_mode": "fixed_association_causal_feedback_evaluation",
        "cli": vars(args),
        "effective_device": str(requested_device),
        "num_ap": NUM_AP,
        "topology_type": TOPOLOGY_TYPE,
        "square_side_m": SQUARE_SIDE,
        "wrap_around": WRAP_AROUND,
        "height_difference_m": HEIGHT_DIFFERENCE,
        "association_policy": "stage2_fixed_instantaneous_rssi_threshold_t0",
        "association_threshold": 0.1,
        "beamformer": "rzf_stored_csi_rate_true_csi",
        "cells": cells,
        "scheduler_rng": {
            "root_seed": args.scheduler_seed,
            "hierarchy": "one identical Generator stream per budget cell; random rankings nested across B",
        },
        "stage2_reference_compatible": reference_compatible,
        "stage2_reference_eval_time_stride": stage2_config["cli"]["eval_time_stride"],
    }
    write_json(out_dir / "config.json", config)
    write_json(
        out_dir / "split_manifest.json",
        {
            "environment_seed": args.seed,
            "scheduler_seed": args.scheduler_seed,
            "trajectory_indices": list(range(args.trajectories)),
            "paired_across_all_cells": True,
        },
    )

    reference_indices = None
    if reference_compatible:
        stride = stage2_config["cli"]["eval_time_stride"]
        reference_indices = np.arange(0, args.episode_steps, stride)
    update_artifacts = {
        "cell_labels": np.asarray([cell["label"] for cell in cells]),
        "initial_acquisition_mask": loader.association_mask.transpose(0, 2, 1),
    }
    csi_artifacts = {}
    raw_artifacts = {
        "evaluation_time_indices": np.arange(0, args.episode_steps, args.eval_time_stride)
    }
    results = {}
    boundary = {
        "stage2_reference_compatible": reference_compatible,
        "environment_reproduced": environment_reproduced,
        "full_current_rate_reproduced": None,
        "zero_speed_hold_equals_full": None,
    }
    full_result = None
    hold_result = None

    for cell in cells:
        logger.info("Evaluating %s", cell["label"])
        result = simulate_cell(
            loader,
            cell,
            args,
            requested_device,
            reference_indices if cell["kind"] == "full_current" else None,
        )
        label = cell["label"]
        results[label] = result
        update_artifacts[f"{label}__update_mask"] = result["updates"]
        update_artifacts[f"{label}__active_link_age"] = result["ages"]
        if cell["kind"] not in ("hold_t0", "full_current"):
            update_artifacts[f"{label}__scheduler_priority"] = result["priorities"]
        csi_artifacts[f"{label}__csi_nmse_by_trajectory_frame"] = result["nmse"]
        csi_artifacts[f"{label}__actual_update_fraction_per_trajectory"] = result[
            "actual_fraction_per_trajectory"
        ]
        for name, values in result["age_metrics"].items():
            csi_artifacts[f"{label}__{name}_age_per_trajectory"] = values
        rate = result["rate"]
        raw_artifacts[f"{label}__per_trajectory_sum_rate"] = rate[
            "trajectory_sum_rates"
        ]
        raw_artifacts[f"{label}__per_trajectory_user_time_average_rates"] = rate[
            "user_time_average_rates"
        ]
        raw_artifacts[f"{label}__per_trajectory_p05_user_rate"] = rate[
            "trajectory_p05_user_rate"
        ]
        raw_artifacts[f"{label}__constraints_passed"] = np.asarray(
            result["summary"]["constraints_passed"]
        )
        raw_artifacts[f"{label}__stored_csi_sha256"] = np.asarray(
            result["stored_sha256"]
        )
        raw_artifacts[f"{label}__rzf_sha256"] = np.asarray(
            rate["beamformer_sha256"]
        )

        if cell["kind"] == "full_current":
            full_result = result
            if reference_compatible:
                with np.load(Path(args.stage2_reference) / "raw_metrics.npz") as frozen:
                    expected = frozen["rzf_per_trajectory_sum_rate"]
                actual = result["reference_rate"]["trajectory_sum_rates"]
                raw_artifacts["stage2_reference_rzf_per_trajectory_sum_rate"] = expected
                raw_artifacts["stage4_full_current_stride10_per_trajectory_sum_rate"] = actual
                boundary["full_current_rate_max_abs_error"] = float(
                    np.max(np.abs(actual - expected))
                )
                boundary["full_current_rate_reproduced"] = bool(
                    np.allclose(actual, expected, rtol=1e-6, atol=1e-8)
                )
                if not boundary["full_current_rate_reproduced"]:
                    raise RuntimeError("Stage 2 full-current RZF reproduction failed")
        elif cell["kind"] == "hold_t0":
            hold_result = result

        if (
            args.speed_kmh == 0
            and full_result is not None
            and hold_result is not None
            and boundary["zero_speed_hold_equals_full"] is None
        ):
            boundary["zero_speed_hold_equals_full"] = bool(
                full_result["stored_sha256"] == hold_result["stored_sha256"]
                and full_result["rate"]["beamformer_sha256"]
                == hold_result["rate"]["beamformer_sha256"]
                and np.allclose(
                    full_result["rate"]["trajectory_sum_rates"],
                    hold_result["rate"]["trajectory_sum_rates"],
                    rtol=1e-6,
                    atol=1e-8,
                )
            )
            if not boundary["zero_speed_hold_equals_full"]:
                raise RuntimeError("Zero-speed hold/full boundary failed")

    if full_result is not None:
        denominator = full_result["rate"]["trajectory_sum_rates"]
        for label, result in results.items():
            retention = result["rate"]["trajectory_sum_rates"] / denominator
            result["summary"]["mean_rate_retention"] = float(retention.mean())
            raw_artifacts[f"{label}__per_trajectory_rate_retention"] = retention

    constraints_passed = all(
        result["summary"]["constraints_passed"] for result in results.values()
    )
    usage_by_scheduler = {}
    for scheduler in SCHEDULERS:
        pairs = sorted(
            (
                result["summary"]["budget"],
                result["summary"]["actual_update_fraction"],
            )
            for result in results.values()
            if result["summary"]["scheduler"] == scheduler
            or result["summary"]["kind"] in ("hold_t0", "full_current")
        )
        usage_by_scheduler[scheduler] = bool(
            all(second[1] >= first[1] for first, second in zip(pairs, pairs[1:]))
        )
    usage_monotonic = all(usage_by_scheduler.values())
    completed = bool(constraints_passed and usage_monotonic)
    summary = {
        "speed_kmh": args.speed_kmh,
        "cells": {label: result["summary"] for label, result in results.items()},
        "boundary": boundary,
        "checks": {
            "constraints": constraints_passed,
            "actual_usage_nondecreasing": usage_monotonic,
            "actual_usage_nondecreasing_by_scheduler": usage_by_scheduler,
        },
        "completed": completed,
    }
    np.savez_compressed(out_dir / "update_traces.npz", **update_artifacts)
    np.savez_compressed(out_dir / "csi_state_metrics.npz", **csi_artifacts)
    np.savez_compressed(out_dir / "raw_metrics.npz", **raw_artifacts)
    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "completion.json",
        {"status": "complete" if completed else "failed_gate", "checks": summary["checks"]},
    )
    if not completed:
        raise RuntimeError("One or more Stage 4 gates failed")
    logger.info("Stage 4 setting completed")


def aggregate_results(root):
    import matplotlib.pyplot as plt

    root = Path(root).expanduser().resolve()
    settings = {}
    for path in sorted(root.glob("straight_*_kmh")):
        completion = json.loads((path / "completion.json").read_text())
        if completion.get("status") != "complete":
            raise RuntimeError(f"Incomplete Stage 4 setting: {path}")
        summary = json.loads((path / "summary.json").read_text())
        settings[str(summary["speed_kmh"])] = summary
    if not settings:
        raise RuntimeError("No Stage 4 settings found")
    write_json(root / "summary.json", {"settings": settings, "completed": True})

    _, axis = plt.subplots(figsize=(7, 5))
    for speed, summary in sorted(settings.items(), key=lambda item: float(item[0])):
        for scheduler in SCHEDULERS:
            values = [
                cell for cell in summary["cells"].values()
                if cell["kind"] in ("hold_t0", "full_current", scheduler)
            ]
            values.sort(key=lambda cell: cell["actual_update_fraction"])
            axis.plot(
                [cell["actual_update_fraction"] for cell in values],
                [cell["mean_rate_retention"] for cell in values],
                marker="o",
                label=f"{speed:g} km/h {scheduler}" if isinstance(speed, float) else f"{speed} km/h {scheduler}",
            )
    axis.set(xlabel="Actual active-link update fraction", ylabel="Sum-rate retention")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(root / "rate_retention_vs_update_fraction.png", dpi=200)
    plt.close()

    figure, axes = plt.subplots(2, len(settings), figsize=(5 * len(settings), 7), squeeze=False)
    for column, (speed, summary) in enumerate(sorted(settings.items(), key=lambda item: float(item[0]))):
        for scheduler in SCHEDULERS:
            values = [
                cell for cell in summary["cells"].values()
                if cell["kind"] in ("hold_t0", "full_current", scheduler)
            ]
            values.sort(key=lambda cell: cell["budget"])
            budgets = [cell["budget"] for cell in values]
            axes[0, column].plot(budgets, [cell["mean_age_frames"] for cell in values], marker="o", label=scheduler)
            axes[1, column].plot(budgets, [cell["mean_csi_nmse"] for cell in values], marker="o", label=scheduler)
        axes[0, column].set_title(f"{speed} km/h")
        axes[0, column].set_ylabel("Mean CSI age (frames)")
        axes[1, column].set_ylabel("Mean CSI NMSE")
        axes[1, column].set_xlabel("Budget B")
        axes[0, column].grid(True, alpha=0.3)
        axes[1, column].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "csi_age_nmse_vs_budget.png", dpi=200)
    plt.close(figure)


def parse_args():
    source = Path(__file__).resolve().parent
    stage1_run = (
        source.parent / "stage1/remote_backup_2026-08-24/results_stage1c_bpp_noise_1e-12"
        / "M2_K8_P15.0/run0"
    )
    parser = argparse.ArgumentParser(description="Stage 4 feedback/aged-CSI evaluator")
    parser.add_argument("--aggregate_root")
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--pmax_dbm", type=float, default=15.0)
    parser.add_argument("--noise_power", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scheduler_seed", type=int, default=4000)
    parser.add_argument("--speed_kmh", type=float, default=0.0)
    parser.add_argument("--decision_period_s", type=float, default=0.001)
    parser.add_argument("--carrier_frequency_hz", type=float, default=2.6e9)
    parser.add_argument("--episode_steps", type=int, default=2000)
    parser.add_argument("--trajectories", type=int, default=10)
    parser.add_argument("--eval_time_stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--budgets", type=int, nargs="+", default=(0, 1, 2, 3, 8))
    parser.add_argument("--schedulers", nargs="+", choices=SCHEDULERS, default=SCHEDULERS)
    parser.add_argument("--ap_coordinates", default=str(stage1_run / "arrays/BS_0.txt"))
    parser.add_argument("--stage1_config", default=str(stage1_run / "config.json"))
    parser.add_argument("--stage2_reference")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="results_stage4")
    args = parser.parse_args()
    if args.aggregate_root:
        return args
    if args.speed_kmh < 0:
        parser.error("--speed_kmh cannot be negative")
    for name in ("M", "K", "noise_power", "decision_period_s", "carrier_frequency_hz", "episode_steps", "trajectories", "eval_time_stride", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if any(budget < 0 or budget > args.K for budget in args.budgets):
        parser.error("--budgets values must be in [0,K]")
    if args.stage2_reference is None:
        speed_label = f"{args.speed_kmh:g}"
        args.stage2_reference = str(
            source.parent / f"stage2/results_stage2_bpp/straight_{speed_label}_kmh"
        )
    for name in ("ap_coordinates", "stage1_config"):
        path = Path(getattr(args, name)).expanduser().resolve()
        if not path.is_file():
            parser.error(f"{name} does not exist: {path}")
        setattr(args, name, str(path))
    reference = Path(args.stage2_reference).expanduser().resolve()
    for filename in ("completion.json", "config.json", "provenance.json", "environment.npz", "raw_metrics.npz"):
        if not (reference / filename).is_file():
            parser.error(f"Stage 2 reference artifact does not exist: {reference / filename}")
    args.stage2_reference = str(reference)
    return args


def main():
    args = parse_args()
    if args.aggregate_root:
        aggregate_results(args.aggregate_root)
        return
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    logger = configure_logging(out_dir)
    try:
        provenance = save_provenance(out_dir, args)
        run(args, out_dir, logger, provenance)
    except Exception:
        if not (out_dir / "completion.json").exists():
            write_json(out_dir / "completion.json", {"status": "error"})
        logger.exception("Stage 4 setting failed")
        raise


if __name__ == "__main__":
    main()
