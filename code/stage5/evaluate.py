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

from association import ASSOCIATION_PERIOD_FRAMES, TOP_L, switching_metrics
from controller import (
    ASSOCIATION_POLICIES,
    FIXED_TOP2,
    H3,
    STAGE4_BOUNDARY,
    TRACE_POLICIES,
    build_modular_trace,
    expand_trace,
)
from environment import MOBILITY_STRAIGHT, TOPOLOGY_TYPE, WRAP_AROUND, MobilityEnvironment
from feedback import SCHEDULERS, FeedbackState
from utils_return_indivial_rates import (
    HEIGHT_DIFFERENCE,
    SQUARE_SIDE,
    calculate_rates,
    rzf_beamforming,
)


NUM_AP = 5
LAMBDA_SWITCH = 0.5
RTOL = 1e-6
ATOL = 1e-8
SOURCE_FILES = (
    "environment.py",
    "utils_return_indivial_rates.py",
    "association.py",
    "feedback.py",
    "controller.py",
    "evaluate.py",
    "test_stage5.py",
    "run_exp-v5.sh",
)
EXPECTED_STAGE4_SOURCE_SHA256 = {
    "environment.py": "1416de41ccd87983c8d9f0a9d0d3c9b22c7b29c026a3830523701c6726c2ba6e",
    "utils_return_indivial_rates.py": "530549069179451187843158068cad4f1ea218c44d5475a44ffc8e51698346a2",
    "feedback.py": "cdafbe7f215b7c3dc1052f7d1b7b29daf93cad7785a0aa831770e909ed060d39",
    "evaluate.py": "986afad6862113563ce648f54f07ba823af47175425bcdba7917808f0b833462",
    "test_stage4.py": "dc5c8b973ba4134277f1c9a51007473839f035b6cfc897cd75ba422a1841be43",
    "run_exp-v4.sh": "81346b9308ecc83e9b3595645e15fb65975081e255fc46395d741e41aee2ecde",
}
EXPECTED_STAGE3_ASSOCIATION_SHA256 = (
    "252c08191396ef9c73f924b08ae652fef1a90bd83f2aaaec5aa4c421e6b19219"
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
    logger = logging.getLogger("stage5")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(out_dir / "run.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def evaluation_device(specification, default_device, stage4_boundary_device=None):
    if specification["role"] == "stage4_boundary" and stage4_boundary_device:
        return torch.device(stage4_boundary_device)
    return torch.device(default_device)


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

    stage4_reference = Path(args.stage4_reference)
    stage3_setting = Path(args.stage3_reference_root) / setting_name(args.speed_kmh)
    inputs = {
        "stage4_reference": str(stage4_reference),
        "stage4_reference_sha256": {
            name: sha256(stage4_reference / name)
            for name in (
                "completion.json",
                "config.json",
                "provenance.json",
                "environment.npz",
                "update_traces.npz",
                "csi_state_metrics.npz",
                "raw_metrics.npz",
            )
        },
        "stage3_reference_root": str(args.stage3_reference_root),
        "stage3_reference_sha256": {
            "completion.json": sha256(Path(args.stage3_reference_root) / "completion.json"),
            "provenance.json": sha256(Path(args.stage3_reference_root) / "provenance.json"),
            "association_traces.npz": sha256(stage3_setting / "association_traces.npz"),
            "raw_metrics.npz": sha256(stage3_setting / "raw_metrics.npz"),
        },
    }
    if args.boundary_reference:
        boundary = Path(args.boundary_reference)
        inputs["boundary_reference"] = str(boundary)
        inputs["boundary_reference_sha256"] = {
            name: sha256(boundary / name)
            for name in ("completion.json", "boundary_summary.json")
        }
    provenance = {
        "git_commit": commit,
        "git_dirty": bool(dirty),
        "git_status_short": dirty,
        "source_sha256": source_hashes,
        "input_artifacts": inputs,
        "ap_coordinates_sha256": sha256(args.ap_coordinates),
        "stage1_config_sha256": sha256(args.stage1_config),
    }
    write_json(out_dir / "provenance.json", provenance)
    return provenance


def setting_name(speed_kmh):
    return f"straight_{speed_kmh:g}_kmh"


def validate_handoffs(args, provenance):
    stage4_reference = Path(args.stage4_reference)
    stage4_completion = json.loads((stage4_reference / "completion.json").read_text())
    stage4_config = json.loads((stage4_reference / "config.json").read_text())
    stage4_provenance = json.loads((stage4_reference / "provenance.json").read_text())
    if stage4_completion.get("status") != "complete":
        raise RuntimeError("Stage 4 reference is not complete")
    for filename, expected in EXPECTED_STAGE4_SOURCE_SHA256.items():
        if stage4_provenance["source_sha256"].get(filename) != expected:
            raise RuntimeError(f"Stage 4 source hash mismatch: {filename}")
    for filename in ("environment.py", "utils_return_indivial_rates.py"):
        if provenance["source_sha256"].get(filename) != EXPECTED_STAGE4_SOURCE_SHA256[filename]:
            raise RuntimeError(f"Frozen Stage 5 source copy changed: {filename}")

    stage3_root = Path(args.stage3_reference_root)
    stage3_completion = json.loads((stage3_root / "completion.json").read_text())
    stage3_provenance = json.loads((stage3_root / "provenance.json").read_text())
    if stage3_completion.get("status") != "complete":
        raise RuntimeError("Stage 3 reference is not complete")
    if stage3_provenance["source_sha256"].get("association.py") != EXPECTED_STAGE3_ASSOCIATION_SHA256:
        raise RuntimeError("Stage 3 association source hash mismatch")
    expected_artifacts = {
        "ap_coordinates_sha256": EXPECTED_STAGE1C_AP_SHA256,
        "stage1_config_sha256": EXPECTED_STAGE1C_CONFIG_SHA256,
    }
    for name, expected in expected_artifacts.items():
        if stage4_provenance.get(name) != expected or provenance.get(name) != expected:
            raise RuntimeError(f"Frozen artifact hash mismatch: {name}")

    cli = stage4_config["cli"]
    fields = {
        "M": args.M,
        "K": args.K,
        "pmax_dbm": args.pmax_dbm,
        "noise_power": args.noise_power,
        "seed": args.seed,
        "speed_kmh": args.speed_kmh,
        "decision_period_s": args.decision_period_s,
        "carrier_frequency_hz": args.carrier_frequency_hz,
        "episode_steps": args.episode_steps,
        "trajectories": args.trajectories,
    }
    stage4_compatible = all(
        np.isclose(cli.get(name), expected, rtol=0, atol=0)
        if isinstance(expected, float)
        else cli.get(name) == expected
        for name, expected in fields.items()
    )
    stage3_setting = stage3_root / setting_name(args.speed_kmh)
    with np.load(stage3_setting / "association_traces.npz") as traces, np.load(
        stage3_setting / "raw_metrics.npz"
    ) as raw:
        stage3_compatible = bool(
            traces["hysteresis_lsf_top2_h3_db__mask"].shape
            == (
                args.trajectories,
                (args.episode_steps + ASSOCIATION_PERIOD_FRAMES - 1)
                // ASSOCIATION_PERIOD_FRAMES,
                args.K,
                NUM_AP,
            )
            and np.array_equal(
                raw["evaluation_time_indices"], np.arange(0, args.episode_steps, 10)
            )
        )
    compatible = bool(stage4_compatible and stage3_compatible)
    if args.require_parent_reproduction and not compatible:
        raise RuntimeError("Parent artifacts are incompatible with this run shape")
    return {
        "stage4": stage4_compatible,
        "stage3": stage3_compatible,
        "both": compatible,
    }


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
    values = {
        "ap_coordinates": loader.BS_Loc_array,
        "ue_positions": loader.ue_positions,
        "ue_speeds_mps": loader.ue_speeds_mps,
        "ue_directions_rad": loader.ue_directions_rad,
        "instantaneous_speeds_mps": loader.instantaneous_speeds_mps,
        "distances": loader.distances,
        "path_loss_factors": loader.path_loss_factors,
        "stage2_t0_association_mask": loader.association_mask,
        "rhos": loader.rhos,
    }
    with np.load(Path(reference) / "environment.npz") as frozen:
        return all(
            np.allclose(frozen[name], value, rtol=1e-14, atol=0)
            if np.issubdtype(np.asarray(value).dtype, np.floating)
            else np.array_equal(frozen[name], value)
            for name, value in values.items()
        )


def boundary_cells():
    return [
        cell(STAGE4_BOUNDARY, "round_robin", 2, "stage4_boundary"),
        cell(STAGE4_BOUNDARY, "mobility_age_priority", 2, "stage4_boundary"),
        cell(STAGE4_BOUNDARY, "round_robin", 8, "stage4_boundary"),
        cell(H3, "round_robin", 8, "stage3_boundary"),
        cell(H3, "round_robin", 0, "stationary_boundary"),
    ]


def main_cells(include_learned):
    cells = [
        cell(FIXED_TOP2, scheduler, 2, "primary")
        for scheduler in SCHEDULERS
    ] + [
        cell(H3, scheduler, 2, "primary") for scheduler in SCHEDULERS
    ] + [
        cell(H3, "round_robin", 0, "anchor"),
        cell(H3, "round_robin", 8, "anchor"),
    ]
    if include_learned:
        cells.extend(
            cell(policy, "mobility_age_priority", 2, "centralized_diagnostic")
            for policy in TRACE_POLICIES
        )
    return cells


def cell(association, scheduler, budget, role):
    return {
        "label": f"{association}__{scheduler}__B{budget}",
        "association": association,
        "scheduler": scheduler,
        "budget": budget,
        "role": role,
    }


def beamformer_checks(weights, association_mask, num_users, pmax_w):
    finite = bool(torch.isfinite(weights).all())
    flat = association_mask.transpose(0, 2, 1).reshape(
        len(association_mask), NUM_AP * num_users
    )
    mask = torch.as_tensor(flat, dtype=torch.bool, device=weights.device)
    mask = mask.unsqueeze(1).expand_as(weights)
    unassociated = weights[~mask]
    max_unassociated = float(unassociated.abs().max().item()) if unassociated.numel() else 0.0
    max_power = 0.0
    for ap in range(NUM_AP):
        block = weights[:, :, ap * num_users:(ap + 1) * num_users]
        max_power = max(max_power, float(block.square().sum(dim=(1, 2)).max().item()))
    return finite, max_unassociated, max_power


def evaluate_rates(
    stored_channels,
    true_channels,
    association_masks,
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
            masks = association_masks[trajectory_batch, time_batch]
            construction_start = time.perf_counter()
            weights = rzf_beamforming(stored, masks, pmax_w, device, noise_power)
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


def simulate_cell(loader, association_trace, specification, args, device, reference_indices=None):
    true_channels = loader.true_channels
    active_frames = expand_trace(association_trace, loader.episode_steps)
    active_ap_user = active_frames.transpose(0, 1, 3, 2)
    state = FeedbackState(
        true_channels[:, 0],
        active_frames[:, 0],
        budget=specification["budget"],
        scheduler=specification["scheduler"],
        t0_link_power=loader.path_loss_factors[:, 0] ** 2,
        rhos=loader.rhos[:, 0],
    )
    shape = active_ap_user.shape
    updates = np.zeros(shape, dtype=bool)
    ages = np.empty(shape, dtype=np.int32)
    ages[:, 0] = state.ages
    priorities = np.zeros(shape, dtype=np.float32)
    stored = np.empty_like(true_channels)
    stored[:, 0] = state.stored_channels
    nmse = np.zeros((loader.batch_size, loader.episode_steps), dtype=np.float64)

    for frame in range(1, loader.episode_steps):
        update_mask = state.step(true_channels[:, frame], active_frames[:, frame])
        updates[:, frame] = update_mask
        ages[:, frame] = state.ages
        priorities[:, frame] = state.last_priority
        stored[:, frame] = state.stored_channels
        error = np.abs(true_channels[:, frame] - state.stored_channels) ** 2
        signal = np.abs(true_channels[:, frame]) ** 2
        numerator = (error * active_ap_user[:, frame, ..., None]).sum(axis=(1, 2, 3))
        denominator = (signal * active_ap_user[:, frame, ..., None]).sum(axis=(1, 2, 3))
        nmse[:, frame] = np.divide(
            numerator,
            denominator,
            out=np.zeros(loader.batch_size),
            where=denominator > 0,
        )

    time_indices = np.arange(0, loader.episode_steps, args.eval_time_stride)
    pmax_w = 10 ** ((args.pmax_dbm - 30) / 10)
    rate = evaluate_rates(
        stored,
        true_channels,
        active_frames,
        time_indices,
        num_users=args.K,
        pmax_w=pmax_w,
        noise_power=args.noise_power,
        device=device,
        frame_batch_size=args.batch_size,
    )
    reference_rate = None
    if reference_indices is not None:
        reference_rate = evaluate_rates(
            stored,
            true_channels,
            active_frames,
            reference_indices,
            num_users=args.K,
            pmax_w=pmax_w,
            noise_power=args.noise_power,
            device=device,
            frame_batch_size=args.batch_size,
        )

    steady_updates = updates[:, 1:]
    steady_active = active_ap_user[:, 1:]
    per_trajectory_updates = steady_updates.sum(axis=(1, 2, 3))
    active_opportunities = steady_active.sum(axis=(1, 2, 3))
    actual_fraction = np.divide(
        per_trajectory_updates,
        active_opportunities,
        out=np.zeros(loader.batch_size),
        where=active_opportunities > 0,
    )
    age_metrics = {
        name: np.empty(loader.batch_size)
        for name in ("mean", "median", "p95", "max")
    }
    never_refreshed = np.empty(loader.batch_size)
    for trajectory in range(loader.batch_size):
        values = ages[trajectory][active_ap_user[trajectory]]
        age_metrics["mean"][trajectory] = values.mean()
        age_metrics["median"][trajectory] = np.median(values)
        age_metrics["p95"][trajectory] = np.percentile(values, 95)
        age_metrics["max"][trajectory] = values.max()
        ever_active = steady_active[trajectory].any(axis=0)
        ever_updated = steady_updates[trajectory].any(axis=0)
        never_refreshed[trajectory] = (
            np.count_nonzero(ever_active & ~ever_updated) / np.count_nonzero(ever_active)
            if ever_active.any()
            else 0.0
        )

    counts = steady_updates.sum(axis=-1)
    loads = steady_active.sum(axis=-1)
    budget_violations = int(np.count_nonzero(counts > specification["budget"]))
    fill_violations = int(
        np.count_nonzero(counts != np.minimum(specification["budget"], loads))
    )
    mask_violations = int(np.count_nonzero(steady_updates & ~steady_active))
    cardinality_violations = (
        0
        if specification["association"] == STAGE4_BOUNDARY
        else int(np.count_nonzero(active_frames.sum(axis=-1) != TOP_L))
    )
    all_active_ages = ages[active_ap_user]
    constraints_passed = bool(
        rate["constraints_passed"]
        and budget_violations == fill_violations == mask_violations == 0
        and cardinality_violations == 0
        and np.all(ages >= 0)
        and np.array_equal(stored[:, 0], true_channels[:, 0])
    )
    summary = {
        **specification,
        "trajectory_average_sum_rate": float(rate["trajectory_sum_rates"].mean()),
        "trajectory_average_p05_user_rate": float(rate["trajectory_p05_user_rate"].mean()),
        "actual_update_fraction": float(
            per_trajectory_updates.sum() / active_opportunities.sum()
        ),
        "updates_per_ap_frame": float(
            per_trajectory_updates.sum()
            / max(loader.batch_size * (loader.episode_steps - 1) * NUM_AP, 1)
        ),
        "mean_age_frames": float(all_active_ages.mean()),
        "median_age_frames": float(np.median(all_active_ages)),
        "p95_age_frames": float(np.percentile(all_active_ages, 95)),
        "max_age_frames": float(all_active_ages.max()),
        "never_refreshed_active_link_fraction": float(never_refreshed.mean()),
        "mean_csi_nmse": float(nmse.mean()),
        "p95_csi_nmse": float(np.percentile(nmse, 95)),
        "budget_violations": budget_violations,
        "fill_violations": fill_violations,
        "association_violations": mask_violations,
        "cardinality_violations": cardinality_violations,
        "finite": bool(rate["finite"] and np.isfinite(nmse).all()),
        "max_unassociated_abs": rate["max_unassociated_abs"],
        "max_ap_power_w": rate["max_ap_power_w"],
        "constraints_passed": constraints_passed,
        "rzf_construction_s": rate["rzf_construction_s"],
        "total_evaluation_s": rate["total_evaluation_s"],
    }
    return {
        "summary": summary,
        "active_frames": active_frames,
        "updates": updates,
        "ages": ages,
        "priorities": priorities,
        "stored": stored,
        "nmse": nmse,
        "actual_fraction_per_trajectory": actual_fraction,
        "never_refreshed_per_trajectory": never_refreshed,
        "age_metrics": age_metrics,
        "rate": rate,
        "reference_rate": reference_rate,
        "stored_sha256": hashlib.sha256(stored.tobytes()).hexdigest(),
    }


def compare_stage4_boundary(result, specification, reference):
    reference_labels = {
        ("round_robin", 2): "round_robin__B2",
        ("mobility_age_priority", 2): "mobility_age_priority__B2",
        ("round_robin", 8): "full_current__B8",
    }
    reference_label = reference_labels[
        (specification["scheduler"], specification["budget"])
    ]
    checks = {}
    max_abs_error = 0.0
    with np.load(Path(reference) / "raw_metrics.npz") as raw:
        pairs = {
            "sum_rate": "per_trajectory_sum_rate",
            "user_rates": "per_trajectory_user_time_average_rates",
            "p05_rate": "per_trajectory_p05_user_rate",
        }
        for name, suffix in pairs.items():
            actual = result["rate"][
                {
                    "sum_rate": "trajectory_sum_rates",
                    "user_rates": "user_time_average_rates",
                    "p05_rate": "trajectory_p05_user_rate",
                }[name]
            ]
            expected = raw[f"{reference_label}__{suffix}"]
            checks[name] = bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL))
            max_abs_error = max(max_abs_error, float(np.max(np.abs(actual - expected))))
    with np.load(Path(reference) / "update_traces.npz") as traces:
        expected_updates = traces[f"{reference_label}__update_mask"]
        expected_ages = traces[f"{reference_label}__active_link_age"]
        active = result["active_frames"].transpose(0, 1, 3, 2)
        checks["updates"] = bool(
            np.array_equal(result["updates"][:, 1:], expected_updates[:, 1:])
        )
        checks["active_ages"] = bool(
            np.array_equal(result["ages"][active], expected_ages[active])
        )
        priority_key = f"{reference_label}__scheduler_priority"
        if priority_key in traces:
            checks["priority"] = bool(
                np.array_equal(result["priorities"], traces[priority_key])
            )
    with np.load(Path(reference) / "csi_state_metrics.npz") as csi:
        pairs = {
            "nmse": (
                result["nmse"],
                csi[f"{reference_label}__csi_nmse_by_trajectory_frame"],
            ),
            "update_fraction": (
                result["actual_fraction_per_trajectory"],
                csi[f"{reference_label}__actual_update_fraction_per_trajectory"],
            ),
        }
        for name, values in result["age_metrics"].items():
            pairs[f"{name}_age"] = (
                values,
                csi[f"{reference_label}__{name}_age_per_trajectory"],
            )
        for name, (actual, expected) in pairs.items():
            checks[name] = bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL))
            max_abs_error = max(max_abs_error, float(np.max(np.abs(actual - expected))))
    return {
        "reference_label": reference_label,
        "checks": checks,
        "max_abs_error": max_abs_error,
        "passed": all(checks.values()),
    }


def compare_stage3_boundary(result, association_trace, reference_setting):
    policy = "hysteresis_lsf_top2_h3_db"
    checks = {}
    diagnostics = {}
    max_abs_error = 0.0
    with np.load(Path(reference_setting) / "association_traces.npz") as traces:
        checks["association_trace"] = bool(
            np.array_equal(association_trace, traces[f"{policy}__mask"])
        )
    with np.load(Path(reference_setting) / "raw_metrics.npz") as raw:
        keys = {
            "sum_rate": (
                result["reference_rate"]["trajectory_sum_rates"],
                raw[f"{policy}__rzf__per_trajectory_sum_rate"],
            ),
            "p05_rate": (
                result["reference_rate"]["trajectory_p05_user_rate"],
                raw[f"{policy}__rzf__per_trajectory_p05_user_rate"],
            ),
        }
        for name, (actual, expected) in keys.items():
            checks[name] = bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL))
            max_abs_error = max(max_abs_error, float(np.max(np.abs(actual - expected))))
        actual_users = result["reference_rate"]["user_time_average_rates"]
        expected_users = raw[f"{policy}__rzf__per_trajectory_user_time_average_rates"]
        diagnostics["user_rates_allclose"] = bool(
            np.allclose(actual_users, expected_users, rtol=RTOL, atol=ATOL)
        )
        diagnostics["user_rates_max_abs_error"] = float(
            np.max(np.abs(actual_users - expected_users))
        )
    return {
        "checks": checks,
        "diagnostics": diagnostics,
        "max_abs_error": max_abs_error,
        "passed": all(checks.values()),
    }


def validate_boundary_reference(path):
    path = Path(path)
    completion = json.loads((path / "completion.json").read_text())
    boundary = json.loads((path / "boundary_summary.json").read_text())
    if completion.get("status") != "complete" or not boundary.get("passed"):
        raise RuntimeError(f"Stage 5 boundary reference failed: {path}")
    return boundary


def store_result(result, label, update_artifacts, csi_artifacts, raw_artifacts):
    update_artifacts[f"{label}__update_mask"] = result["updates"]
    update_artifacts[f"{label}__candidate_link_age"] = result["ages"]
    update_artifacts[f"{label}__scheduler_priority"] = result["priorities"]
    csi_artifacts[f"{label}__stored_csi_rzf_input"] = result["stored"]
    csi_artifacts[f"{label}__csi_nmse_by_trajectory_frame"] = result["nmse"]
    csi_artifacts[f"{label}__actual_update_fraction_per_trajectory"] = result[
        "actual_fraction_per_trajectory"
    ]
    csi_artifacts[f"{label}__never_refreshed_fraction_per_trajectory"] = result[
        "never_refreshed_per_trajectory"
    ]
    for name, values in result["age_metrics"].items():
        csi_artifacts[f"{label}__{name}_age_per_trajectory"] = values
    rate = result["rate"]
    raw_artifacts[f"{label}__per_trajectory_sum_rate"] = rate["trajectory_sum_rates"]
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


def run(args, out_dir, logger, provenance):
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.info("CUDA unavailable; using CPU")
    stage4_boundary_device = (
        torch.device(args.stage4_boundary_device)
        if args.stage4_boundary_device
        else None
    )
    compatibility = validate_handoffs(args, provenance)
    logger.info("Gate 5.0 frozen parent hashes passed")
    boundary_reference = (
        validate_boundary_reference(args.boundary_reference)
        if args.boundary_reference
        else None
    )

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
        environment_matches_reference(loader, args.stage4_reference)
        if compatibility["stage4"]
        else None
    )
    if environment_reproduced is False:
        raise RuntimeError("Stage 4 environment reproduction failed")

    include_learned = bool(args.learned_diagnostics and compatibility["stage3"])
    if args.learned_diagnostics and not include_learned:
        logger.info("Skipping learned trace diagnostics for incompatible smoke shape")
    if args.phase == "boundary":
        specifications = boundary_cells()
    elif args.phase == "main":
        specifications = main_cells(include_learned)
    else:
        specifications = boundary_cells() + main_cells(include_learned)
        specifications = list({item["label"]: item for item in specifications}.values())

    config = {
        "stage": "5A",
        "execution_mode": "joint_dynamic_association_causal_feedback_evaluation",
        "phase": args.phase,
        "cli": vars(args),
        "effective_device": str(device),
        "stage4_boundary_device": (
            str(stage4_boundary_device) if stage4_boundary_device else str(device)
        ),
        "num_ap": NUM_AP,
        "top_l": TOP_L,
        "association_period_frames": ASSOCIATION_PERIOD_FRAMES,
        "feedback_period_frames": 1,
        "all_link_bootstrap_cost": args.trajectories * NUM_AP * args.K,
        "all_link_bootstrap_cost_per_trajectory": NUM_AP * args.K,
        "all_link_bootstrap_excluded_from_steady_budget": True,
        "current_lsf_assumption": "AP-local slow-timescale measurement outside the instantaneous small-scale CSI budget",
        "action_order": "association bids; UE top-2 arbitration; feedback selection; selected CSI reveal; stored-CSI RZF; true-CSI rate",
        "arbitration": "one AP message with K scalar bids per AP and association epoch; stable AP-index tie break",
        "beamformer": "rzf_stored_csi_rate_true_csi",
        "rate_retention_reference": f"{H3}__round_robin__B8",
        "lambda_switch": LAMBDA_SWITCH,
        "cells": specifications,
        "learned_diagnostics_included": include_learned,
        "learned_diagnostics_boundary": "frozen centralized Stage 3 traces; not a decentralized Stage 5 policy",
        "topology_type": TOPOLOGY_TYPE,
        "square_side_m": SQUARE_SIDE,
        "wrap_around": WRAP_AROUND,
        "height_difference_m": HEIGHT_DIFFERENCE,
        "parent_compatibility": compatibility,
        "environment_reproduction_tolerance": {"rtol": 1e-14, "atol": 0},
        "boundary_reference_summary": boundary_reference,
    }
    write_json(out_dir / "config.json", config)
    write_json(
        out_dir / "split_manifest.json",
        {
            "split": "seed0_development_test",
            "environment_seed": args.seed,
            "trajectory_indices": list(range(args.trajectories)),
            "paired_across_all_cells": True,
            "association_trace_source": "current local LSF, except frozen Stage 3 diagnostic traces",
        },
    )

    stage3_setting = Path(args.stage3_reference_root) / setting_name(args.speed_kmh)
    reference_traces = {}
    reference_actions = {}
    with np.load(stage3_setting / "association_traces.npz") as frozen:
        for policy in (H3,) + TRACE_POLICIES:
            key = f"{policy}__mask"
            if key in frozen:
                reference_traces[policy] = frozen[key].copy()
        for policy in TRACE_POLICIES:
            key = f"{policy}__actions"
            if key in frozen:
                reference_actions[policy] = frozen[key].copy()
    trace_cache = {}
    bids_cache = {}
    for policy in dict.fromkeys(item["association"] for item in specifications):
        external = reference_traces.get(policy) if policy in TRACE_POLICIES else None
        _, bids, trace = build_modular_trace(
            loader.path_loss_factors,
            policy,
            stage4_mask=loader.association_mask,
            external_trace=external,
        )
        trace_cache[policy] = trace
        bids_cache[policy] = bids

    decision_indices = np.arange(0, loader.episode_steps, ASSOCIATION_PERIOD_FRAMES)
    association_artifacts = {
        "decision_indices": decision_indices,
        "modular_ap_message_count_per_trajectory_epoch": np.asarray(NUM_AP),
        "modular_bid_scalar_count_per_trajectory_epoch": np.asarray(NUM_AP * args.K),
    }
    association_summary = {}
    for policy, trace in trace_cache.items():
        association_artifacts[f"{policy}__mask"] = trace
        if policy not in TRACE_POLICIES:
            association_artifacts[f"{policy}__raw_local_lsf_bids"] = bids_cache[policy]
        if policy in reference_actions:
            association_artifacts[f"{policy}__frozen_centralized_actions"] = reference_actions[policy]
        metrics = switching_metrics(
            trace,
            num_frames=args.episode_steps,
            frame_period_s=args.decision_period_s,
        )
        for name, values in metrics.items():
            association_artifacts[f"{policy}__{name}"] = values
        association_summary[policy] = {
            "mean_link_toggles_per_ue_s": float(metrics["link_toggles_per_ue_s"].mean()),
            "mean_serving_set_changes_per_ue_s": float(
                metrics["serving_set_changes_per_ue_s"].mean()
            ),
            "mean_ap_load": float(metrics["ap_load_mean_per_trajectory"].mean()),
            "mean_max_ap_load": float(metrics["ap_load_max_per_trajectory"].mean()),
            "zero_load_ap_fraction": float(
                metrics["zero_load_ap_fraction_per_trajectory"].mean()
            ),
        }

    update_artifacts = {
        "cell_labels": np.asarray([item["label"] for item in specifications]),
        "initial_all_link_acquisition_mask": np.ones(
            (args.trajectories, NUM_AP, args.K), dtype=bool
        ),
    }
    csi_artifacts = {}
    raw_artifacts = {
        "evaluation_time_indices": np.arange(
            0, args.episode_steps, args.eval_time_stride
        )
    }
    results = {}
    boundary = {
        "parent_compatible": compatibility["both"],
        "environment_reproduced": environment_reproduced,
        "stage4_cells": {},
        "stage3_h3_b8": None,
        "stationary_h3_zero_switch": None,
        "stationary_h3_b0_equals_b8": None,
    }

    boundary_labels = {item["label"] for item in boundary_cells()}
    for index, specification in enumerate(specifications):
        label = specification["label"]
        logger.info("Evaluating %s", label)
        reference_indices = None
        if (
            specification["association"] == H3
            and specification["budget"] == args.K
            and compatibility["stage3"]
        ):
            with np.load(stage3_setting / "raw_metrics.npz") as frozen:
                reference_indices = frozen["evaluation_time_indices"].copy()
        result = simulate_cell(
            loader,
            trace_cache[specification["association"]],
            specification,
            args,
            evaluation_device(specification, device, stage4_boundary_device),
            reference_indices,
        )
        results[label] = result
        store_result(result, label, update_artifacts, csi_artifacts, raw_artifacts)

        if specification["role"] == "stage4_boundary" and compatibility["stage4"]:
            check = compare_stage4_boundary(result, specification, args.stage4_reference)
            boundary["stage4_cells"][label] = check
        if specification["role"] in ("stage3_boundary", "anchor") and specification["budget"] == args.K and compatibility["stage3"]:
            boundary["stage3_h3_b8"] = compare_stage3_boundary(
                result, trace_cache[H3], stage3_setting
            )

        if args.phase == "all" and index == len(boundary_cells()) - 1:
            preliminary = boundary_passed(boundary, results, trace_cache, args)
            if args.require_parent_reproduction and not preliminary:
                logger.error("Parent boundary failed; main matrix will not run")
                specifications = specifications[:len(boundary_cells())]
                break

    boundary["stationary_h3_zero_switch"] = (
        bool(np.array_equal(trace_cache[H3][:, 1:], trace_cache[H3][:, :-1]))
        if args.speed_kmh == 0 and H3 in trace_cache
        else None
    )
    h3_b0 = f"{H3}__round_robin__B0"
    h3_b8 = f"{H3}__round_robin__B8"
    if args.speed_kmh == 0 and h3_b0 in results and h3_b8 in results:
        boundary["stationary_h3_b0_equals_b8"] = bool(
            results[h3_b0]["stored_sha256"] == results[h3_b8]["stored_sha256"]
            and np.allclose(
                results[h3_b0]["rate"]["trajectory_sum_rates"],
                results[h3_b8]["rate"]["trajectory_sum_rates"],
                rtol=RTOL,
                atol=ATOL,
            )
        )
    boundary["passed"] = (
        bool(boundary_reference["passed"])
        if args.phase == "main"
        else boundary_passed(boundary, results, trace_cache, args)
    )
    write_json(out_dir / "boundary_summary.json", boundary)

    if h3_b8 in results:
        denominator = results[h3_b8]["rate"]["trajectory_sum_rates"]
        for label, result in results.items():
            retention = result["rate"]["trajectory_sum_rates"] / denominator
            result["summary"]["mean_rate_retention_to_h3_b8"] = float(retention.mean())
            raw_artifacts[f"{label}__per_trajectory_rate_retention_to_h3_b8"] = retention

    for label, result in results.items():
        policy = result["summary"]["association"]
        switch = switching_metrics(
            trace_cache[policy],
            num_frames=args.episode_steps,
            frame_period_s=args.decision_period_s,
        )["link_toggles_per_trajectory"]
        epochs = trace_cache[policy].shape[1]
        utility = (
            result["rate"]["trajectory_sum_rates"] / args.K
            - LAMBDA_SWITCH * switch / (epochs * NUM_AP * args.K)
        )
        raw_artifacts[f"{label}__per_trajectory_utility"] = utility
        result["summary"]["trajectory_average_utility"] = float(utility.mean())

    primary_labels = {
        "fixed_rr": f"{FIXED_TOP2}__round_robin__B2",
        "fixed_priority": f"{FIXED_TOP2}__mobility_age_priority__B2",
        "h3_rr": f"{H3}__round_robin__B2",
        "h3_priority": f"{H3}__mobility_age_priority__B2",
    }
    did = None
    if all(label in results for label in primary_labels.values()):
        rates = {
            name: results[label]["rate"]["trajectory_sum_rates"]
            for name, label in primary_labels.items()
        }
        did = (rates["h3_priority"] - rates["h3_rr"]) - (
            rates["fixed_priority"] - rates["fixed_rr"]
        )
        raw_artifacts["association_scheduler_interaction_did_per_trajectory"] = did

    constraints_passed = all(
        result["summary"]["constraints_passed"] for result in results.values()
    )
    h3_nonzero = bool(
        H3 in association_summary
        and association_summary[H3]["mean_link_toggles_per_ue_s"] > 0
    )
    rr_label = primary_labels["h3_rr"]
    priority_label = primary_labels["h3_priority"]
    scheduler_updates_differ = (
        bool(not np.array_equal(results[rr_label]["updates"], results[priority_label]["updates"]))
        if rr_label in results and priority_label in results
        else None
    )
    checks = {
        "constraints": constraints_passed,
        "boundary": boundary["passed"] if args.phase in ("boundary", "all") else True,
        "boundary_reference": boundary_reference is not None if args.phase == "main" else True,
    }
    completed = all(checks.values())
    summary = {
        "speed_kmh": args.speed_kmh,
        "phase": args.phase,
        "association": association_summary,
        "cells": {label: result["summary"] for label, result in results.items()},
        "boundary": boundary,
        "interaction_difference_in_differences_mean": float(did.mean()) if did is not None else None,
        "signals": {
            "h3_nonzero_association_changes": h3_nonzero,
            "h3_round_robin_priority_updates_differ": scheduler_updates_differ,
        },
        "checks": checks,
        "completed": completed,
    }
    np.savez_compressed(out_dir / "association_traces.npz", **association_artifacts)
    np.savez_compressed(out_dir / "update_traces.npz", **update_artifacts)
    np.savez_compressed(out_dir / "csi_state_metrics.npz", **csi_artifacts)
    np.savez_compressed(out_dir / "raw_metrics.npz", **raw_artifacts)
    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "completion.json",
        {"status": "complete" if completed else "failed_gate", "checks": checks},
    )
    if not completed:
        raise RuntimeError("One or more Stage 5 gates failed")
    logger.info("Stage 5 %s setting completed", args.phase)


def boundary_passed(boundary, results, trace_cache, args):
    constraints = all(
        result["summary"]["constraints_passed"] for result in results.values()
    )
    if not args.require_parent_reproduction:
        parent = True
    else:
        parent = bool(
            boundary["environment_reproduced"]
            and len(boundary["stage4_cells"]) == 3
            and all(item["passed"] for item in boundary["stage4_cells"].values())
            and boundary["stage3_h3_b8"]
            and boundary["stage3_h3_b8"]["passed"]
        )
    stationary = True
    if args.speed_kmh == 0:
        h3_b0 = f"{H3}__round_robin__B0"
        h3_b8 = f"{H3}__round_robin__B8"
        stationary = bool(
            H3 in trace_cache
            and np.array_equal(trace_cache[H3][:, 1:], trace_cache[H3][:, :-1])
            and h3_b0 in results
            and h3_b8 in results
            and results[h3_b0]["stored_sha256"] == results[h3_b8]["stored_sha256"]
            and np.allclose(
                results[h3_b0]["rate"]["trajectory_sum_rates"],
                results[h3_b8]["rate"]["trajectory_sum_rates"],
                rtol=RTOL,
                atol=ATOL,
            )
        )
    return bool(constraints and parent and stationary)


def aggregate_results(root, phase, boundary_root=None):
    root = Path(root).expanduser().resolve()
    settings = {}
    for path in sorted(root.glob("straight_*_kmh")):
        completion = json.loads((path / "completion.json").read_text())
        if completion.get("status") != "complete":
            raise RuntimeError(f"Incomplete Stage 5 setting: {path}")
        summary = json.loads((path / "summary.json").read_text())
        settings[str(summary["speed_kmh"])] = summary
    if not settings:
        raise RuntimeError("No Stage 5 settings found")

    if phase == "boundary":
        passed = all(summary["boundary"]["passed"] for summary in settings.values())
        boundary_summary = {
            "settings": {
                speed: summary["boundary"] for speed, summary in settings.items()
            },
            "passed": passed,
        }
        write_json(root / "boundary_summary.json", boundary_summary)
        write_json(
            root / "summary.json",
            {"phase": phase, "settings": settings, "completed": passed},
        )
        write_json(
            root / "completion.json",
            {"status": "complete" if passed else "failed_gate"},
        )
        if not passed:
            raise RuntimeError("Stage 5 two-parent boundary failed")
        return

    if boundary_root is None:
        raise RuntimeError("Main aggregation requires --boundary_root")
    boundary_root = Path(boundary_root).expanduser().resolve()
    boundary_completion = json.loads((boundary_root / "completion.json").read_text())
    boundary_summary = json.loads((boundary_root / "boundary_summary.json").read_text())
    if boundary_completion.get("status") != "complete" or not boundary_summary.get("passed"):
        raise RuntimeError("Boundary root is not complete")
    write_json(root / "boundary_summary.json", boundary_summary)

    moving = [
        summary for speed, summary in settings.items() if float(speed) in (30.0, 80.0)
    ]
    checks = {
        "all_settings_complete": len(settings) == 3,
        "two_parent_boundary": True,
        "constraints": all(summary["checks"]["constraints"] for summary in settings.values()),
        "moving_h3_action_signal": any(
            summary["signals"]["h3_nonzero_association_changes"] for summary in moving
        ),
        "moving_scheduler_action_signal": any(
            summary["signals"]["h3_round_robin_priority_updates_differ"]
            for summary in moving
        ),
    }
    completed = all(checks.values())
    aggregate = {
        "phase": phase,
        "settings": settings,
        "checks": checks,
        "completed": completed,
        "stage5b_eligible": completed,
        "stage5b_status": (
            "eligible_to_implement_after_stage5a_audit"
            if completed
            else "skipped_no_joint_action_signal_or_failed_gate"
        ),
    }
    write_json(root / "summary.json", aggregate)
    make_plots(root, settings)
    write_json(
        root / "completion.json",
        {"status": "complete" if completed else "failed_gate", "checks": checks},
    )
    if not completed:
        raise RuntimeError("Stage 5A modular joint matrix failed Gate 5.4")


def make_plots(root, settings):
    import matplotlib.pyplot as plt

    ordered = sorted(settings.items(), key=lambda item: float(item[0]))
    primary = [
        f"{FIXED_TOP2}__round_robin__B2",
        f"{FIXED_TOP2}__mobility_age_priority__B2",
        f"{H3}__round_robin__B2",
        f"{H3}__mobility_age_priority__B2",
    ]
    labels = ["fixed + RR", "fixed + priority", "H3 + RR", "H3 + priority"]

    figure, axis = plt.subplots(figsize=(7, 5))
    for cell_label, display in zip(primary, labels):
        axis.plot(
            [float(speed) for speed, _ in ordered],
            [summary["cells"][cell_label]["mean_rate_retention_to_h3_b8"] for _, summary in ordered],
            marker="o",
            label=display,
        )
    axis.set(xlabel="Speed (km/h)", ylabel="Sum-rate retention relative to H3@B8")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "rate_feedback_retention.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    for speed, summary in ordered:
        for cell_label, display in zip(primary, labels):
            association = summary["cells"][cell_label]["association"]
            axis.scatter(
                summary["association"][association]["mean_link_toggles_per_ue_s"],
                summary["cells"][cell_label]["trajectory_average_sum_rate"],
                label=f"{speed:g} km/h {display}" if isinstance(speed, float) else f"{speed} km/h {display}",
            )
    axis.set(xlabel="Link toggles / UE / s", ylabel="Time-average sum rate")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(root / "rate_switching_pareto.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    for cell_label, display in zip(primary, labels):
        axis.plot(
            [summary["cells"][cell_label]["p95_age_frames"] for _, summary in ordered],
            [summary["cells"][cell_label]["trajectory_average_p05_user_rate"] for _, summary in ordered],
            marker="o",
            label=display,
        )
    axis.set(xlabel="Active-link p95 CSI age (frames)", ylabel="UE time-average rate p05")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "tail_rate_age_tradeoff.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.bar(
        [str(speed) for speed, _ in ordered],
        [summary["interaction_difference_in_differences_mean"] for _, summary in ordered],
    )
    axis.set(xlabel="Speed (km/h)", ylabel="Association × scheduler difference-in-differences")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(root / "association_scheduler_interaction.png", dpi=200)
    plt.close(figure)


def default_stage1_run(source):
    candidates = (
        source.parent / "results_stage1c_bpp_noise_1e-12/M2_K8_P15.0/run0",
        source.parent / "stage1/remote_backup_2026-08-24/results_stage1c_bpp_noise_1e-12/M2_K8_P15.0/run0",
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def parse_args():
    source = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Stage 5 joint association/feedback evaluator")
    parser.add_argument("--aggregate_root")
    parser.add_argument("--phase", choices=("boundary", "main", "all"), default="all")
    parser.add_argument("--boundary_root")
    parser.add_argument("--boundary_reference")
    parser.add_argument("--require_parent_reproduction", action="store_true")
    parser.add_argument("--learned_diagnostics", action="store_true")
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--pmax_dbm", type=float, default=15.0)
    parser.add_argument("--noise_power", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed_kmh", type=float, default=0.0)
    parser.add_argument("--decision_period_s", type=float, default=0.001)
    parser.add_argument("--carrier_frequency_hz", type=float, default=2.6e9)
    parser.add_argument("--episode_steps", type=int, default=2000)
    parser.add_argument("--trajectories", type=int, default=10)
    parser.add_argument("--eval_time_stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    stage1_run = default_stage1_run(source)
    parser.add_argument("--ap_coordinates", default=str(stage1_run / "arrays/BS_0.txt"))
    parser.add_argument("--stage1_config", default=str(stage1_run / "config.json"))
    parser.add_argument("--stage4_reference")
    parser.add_argument(
        "--stage3_reference_root",
        default=str(source.parent / "stage3/results_stage3b_seed0_dual_eval_straight"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage4_boundary_device")
    parser.add_argument("--out_dir", default="results_stage5")
    args = parser.parse_args()
    if args.aggregate_root:
        return args
    if args.M != 2 or args.K != 8:
        parser.error("Stage 5 contract fixes M=2 and K=8")
    if args.speed_kmh < 0:
        parser.error("--speed_kmh cannot be negative")
    for name in (
        "M",
        "K",
        "noise_power",
        "decision_period_s",
        "carrier_frequency_hz",
        "episode_steps",
        "trajectories",
        "eval_time_stride",
        "batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.episode_steps < ASSOCIATION_PERIOD_FRAMES + 1:
        parser.error("--episode_steps must cover at least one post-t0 association epoch")
    if args.phase == "main" and not args.boundary_reference:
        parser.error("--phase main requires --boundary_reference")
    if args.stage4_reference is None:
        args.stage4_reference = str(
            source.parent / "stage4/results_stage4_seed0" / setting_name(args.speed_kmh)
        )
    for name in ("ap_coordinates", "stage1_config"):
        path = Path(getattr(args, name)).expanduser().resolve()
        if not path.is_file():
            parser.error(f"{name} does not exist: {path}")
        setattr(args, name, str(path))
    stage4_reference = Path(args.stage4_reference).expanduser().resolve()
    for filename in (
        "completion.json",
        "config.json",
        "provenance.json",
        "environment.npz",
        "update_traces.npz",
        "csi_state_metrics.npz",
        "raw_metrics.npz",
    ):
        if not (stage4_reference / filename).is_file():
            parser.error(f"Stage 4 artifact does not exist: {stage4_reference / filename}")
    args.stage4_reference = str(stage4_reference)
    stage3_root = Path(args.stage3_reference_root).expanduser().resolve()
    setting = stage3_root / setting_name(args.speed_kmh)
    for path in (
        stage3_root / "completion.json",
        stage3_root / "provenance.json",
        setting / "association_traces.npz",
        setting / "raw_metrics.npz",
    ):
        if not path.is_file():
            parser.error(f"Stage 3 artifact does not exist: {path}")
    args.stage3_reference_root = str(stage3_root)
    if args.boundary_reference:
        boundary = Path(args.boundary_reference).expanduser().resolve()
        for filename in ("completion.json", "boundary_summary.json"):
            if not (boundary / filename).is_file():
                parser.error(f"Boundary artifact does not exist: {boundary / filename}")
        args.boundary_reference = str(boundary)
    args.out_dir = str(Path(args.out_dir).expanduser().resolve())
    return args


def main():
    args = parse_args()
    if args.aggregate_root:
        aggregate_results(args.aggregate_root, args.phase, args.boundary_root)
        return
    out_dir = Path(args.out_dir)
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
        logger.exception("Stage 5 setting failed")
        raise


if __name__ == "__main__":
    main()
