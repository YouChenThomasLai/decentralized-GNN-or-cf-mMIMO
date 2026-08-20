import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from environment import (
    DEFAULT_HOTSPOT_CENTERS,
    MOBILITY_HOTSPOT,
    MOBILITY_STRAIGHT,
    MobilityEnvironment,
    hotspot_process_diagnostics,
    symmetric_transition_matrix,
)
from evaluate import METHODS, TrajectoryEvaluator, load_frozen_model
from model_2 import node_update
from utils_return_indivial_rates import independent_channel_diagnostics


SOURCE_FILES = (
    "data.py",
    "environment.py",
    "evaluate.py",
    "model_2.py",
    "trainer_2.py",
    "utils_return_indivial_rates.py",
    "run_exp-v2.sh",
    "test_stage2.py",
)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
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


def save_provenance(out_dir, checkpoint_path=None):
    source_dir = Path(__file__).resolve().parent
    snapshot_dir = out_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for filename in SOURCE_FILES:
        source_path = source_dir / filename
        shutil.copy2(source_path, snapshot_dir / filename)
        source_hashes[filename] = sha256(source_path)
    repository = source_dir.parents[1]
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=repository, text=True
        ).strip()
        dirty_status = subprocess.check_output(
            ("git", "status", "--short"), cwd=repository, text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty_status = ["git status unavailable"]
    provenance = {
        "git_commit": commit,
        "git_dirty": bool(dirty_status),
        "git_status_short": dirty_status,
        "source_sha256": source_hashes,
        "checkpoint_sha256": (
            sha256(checkpoint_path) if checkpoint_path is not None else None
        ),
    }
    with open(out_dir / "provenance.json", "w") as output:
        json.dump(provenance, output, indent=2, sort_keys=True)
    return provenance


def configure_logging(out_dir):
    logger = logging.getLogger("stage2")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(out_dir / "run.log"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def flatten_diagnostics(prefix, diagnostics):
    return {f"{prefix}_{key}": value for key, value in diagnostics.items()}


def channel_gate_passed(diagnostics):
    return bool(
        diagnostics["finite"]
        and abs(float(diagnostics["real_mean"])) <= 0.02
        and abs(float(diagnostics["imag_mean"])) <= 0.02
        and abs(float(diagnostics["real_variance"]) - 0.5) <= 0.02
        and abs(float(diagnostics["imag_variance"]) - 0.5) <= 0.02
        and float(diagnostics["max_ar1_error"]) <= 0.02
    )


def save_environment(loader, out_dir):
    environment = {
        "mobility_model": np.asarray(loader.mobility_model),
        "ue_positions": loader.ue_positions,
        "ue_speeds_mps": loader.ue_speeds_mps,
        "ue_directions_rad": loader.ue_directions_rad,
        "instantaneous_speeds_mps": loader.instantaneous_speeds_mps,
        "distances": loader.distances,
        "path_loss_factors": loader.path_loss_factors,
        "association_mask": loader.association_mask,
        "rhos": loader.rhos,
    }
    if loader.mobility_model == MOBILITY_HOTSPOT:
        environment.update(
            {
                "hotspot_centers": loader.hotspot_centers,
                "hotspot_radius_m": np.asarray(loader.hotspot_radius_m),
                "transition_matrix": loader.transition_matrix,
                "hotspot_state": loader.hotspot_state,
                "mobility_phase": loader.mobility_phase,
                "parent_trace_ids": loader.parent_trace_ids,
                "clip_start_times_s": loader.clip_start_times_s,
                "hotspot_burn_in_s": np.asarray(loader.hotspot_burn_in_s),
            }
        )
    np.savez_compressed(out_dir / "environment.npz", **environment)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2 frozen-model mobility evaluator"
    )
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--pmax_dbm", "--Pmax", type=float, default=15.0)
    parser.add_argument("--noise_power", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed_kmh", type=float, default=0.0)
    parser.add_argument(
        "--mobility_model",
        choices=(MOBILITY_STRAIGHT, MOBILITY_HOTSPOT),
        default=MOBILITY_STRAIGHT,
    )
    parser.add_argument("--decision_period_s", type=float, default=0.001)
    parser.add_argument("--carrier_frequency_hz", type=float, default=2.6e9)
    parser.add_argument("--episode_steps", type=int, default=2000)
    parser.add_argument("--trajectories", type=int, default=10)
    parser.add_argument("--eval_time_stride", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hotspot_radius_m", type=float, default=10.0)
    parser.add_argument("--hotspot_stickiness", type=float, default=0.6)
    parser.add_argument("--hotspot_dwell_mean_s", type=float, default=5.0)
    parser.add_argument("--hotspot_dwell_shape", type=float, default=2.0)
    parser.add_argument("--hotspot_trace_duration_s", type=float, default=300.0)
    parser.add_argument("--diagnostic_events", type=int, default=10000)
    parser.add_argument("--diagnostic_min_outgoing", type=int, default=2000)
    parser.add_argument("--channel_diagnostic_samples", type=int, default=50000)
    parser.add_argument("--diagnostics_only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="results_stage2")
    args = parser.parse_args()
    positive = {
        "--M": args.M,
        "--K": args.K,
        "--noise_power": args.noise_power,
        "--decision_period_s": args.decision_period_s,
        "--carrier_frequency_hz": args.carrier_frequency_hz,
        "--episode_steps": args.episode_steps,
        "--trajectories": args.trajectories,
        "--eval_time_stride": args.eval_time_stride,
        "--batch_size": args.batch_size,
        "--diagnostic_events": args.diagnostic_events,
        "--diagnostic_min_outgoing": args.diagnostic_min_outgoing,
        "--channel_diagnostic_samples": args.channel_diagnostic_samples,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.speed_kmh < 0:
        parser.error("--speed_kmh cannot be negative")
    if not 0 <= args.hotspot_stickiness < 1:
        parser.error("--hotspot_stickiness must be in [0, 1)")
    if args.mobility_model == MOBILITY_HOTSPOT and args.speed_kmh <= 0:
        parser.error("Hotspot mobility requires --speed_kmh > 0")
    if not args.diagnostics_only and not args.checkpoint:
        parser.error("Frozen inference requires --checkpoint")
    if args.checkpoint:
        args.checkpoint = str(Path(args.checkpoint).expanduser().resolve())
        if not Path(args.checkpoint).is_file():
            parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    return args


def run(args, logger, out_dir, provenance):
    seed_everything(args.seed)
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
        logger.info("CUDA unavailable; using CPU")
    transition_matrix = symmetric_transition_matrix(
        len(DEFAULT_HOTSPOT_CENTERS), args.hotspot_stickiness
    )
    pmax_w = 10 ** ((args.pmax_dbm - 30) / 10)
    config = {
        "execution_mode": (
            "environment_diagnostics_only"
            if args.diagnostics_only
            else "frozen_stage1b_checkpoint_evaluation"
        ),
        "cli": vars(args),
        "effective_device": str(requested_device),
        "num_ap": 5,
        "pmax_w": pmax_w,
        "association_threshold": 0.1,
        "association_policy": "Stage 1 threshold at t=0, fixed for clip",
        "csi_policy": "full current true CSI on every evaluated frame",
        "observation_scope": {
            "centralized_gnn": "global current CSI",
            "decentralized_gnn": "predefined AP-local current CSI",
            "mrt": "current CSI on associated links",
            "rzf": "current CSI on associated links",
        },
        "channel_model": "Jakes-calibrated first-order Gauss-Markov AR(1)",
        "source_sha256": provenance["source_sha256"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "hotspot_transition_matrix": (
            transition_matrix.tolist()
            if args.mobility_model == MOBILITY_HOTSPOT
            else None
        ),
    }
    with open(out_dir / "config.json", "w") as output:
        json.dump(config, output, indent=2, sort_keys=True)

    loader = MobilityEnvironment(
        args.M,
        args.trajectories,
        episode_steps=args.episode_steps,
        speed_kmh=args.speed_kmh,
        decision_period_s=args.decision_period_s,
        carrier_frequency_hz=args.carrier_frequency_hz,
        seed=args.seed,
        mobility_model=args.mobility_model,
        hotspot_radius_m=args.hotspot_radius_m,
        transition_matrix=transition_matrix,
        dwell_mean_s=args.hotspot_dwell_mean_s,
        dwell_shape=args.hotspot_dwell_shape,
        hotspot_trace_duration_s=args.hotspot_trace_duration_s,
    ).generate_trajectories(args.K, 0.1)
    save_environment(loader, out_dir)
    mobility_diagnostics = loader.mobility_diagnostics()
    np.savez(out_dir / "mobility_diagnostics.npz", **mobility_diagnostics)
    trajectory_channel_diagnostics = loader.channel_diagnostics()
    np.savez(
        out_dir / "trajectory_channel_diagnostics.npz",
        **trajectory_channel_diagnostics,
    )

    diagnostic_speeds = (
        (0.0, args.speed_kmh / 3.6)
        if args.mobility_model == MOBILITY_HOTSPOT
        else (args.speed_kmh / 3.6,)
    )
    independent_diagnostics = {}
    independent_pass = True
    for index, speed_mps in enumerate(diagnostic_speeds):
        diagnostics = independent_channel_diagnostics(
            speed_mps,
            sample_count=args.channel_diagnostic_samples,
            seed=args.seed + 100 + index,
            carrier_frequency_hz=args.carrier_frequency_hz,
            decision_period_s=args.decision_period_s,
        )
        independent_diagnostics.update(
            flatten_diagnostics(f"speed_{speed_mps * 3.6:g}_kmh", diagnostics)
        )
        independent_pass &= channel_gate_passed(diagnostics)
    np.savez(
        out_dir / "independent_channel_diagnostics.npz",
        **independent_diagnostics,
    )

    mobility_pass = bool(
        float(mobility_diagnostics["max_position_radius_m"]) <= 100 + 1e-12
        and float(mobility_diagnostics["max_step_limit_violation_m"]) <= 1e-10
    )
    if args.mobility_model == MOBILITY_STRAIGHT:
        mobility_pass &= bool(
            float(mobility_diagnostics["max_step_error_m"]) < 1e-10
        )

    process_diagnostics = None
    process_pass = True
    if args.mobility_model == MOBILITY_HOTSPOT:
        process_diagnostics = hotspot_process_diagnostics(
            transition_matrix,
            args.hotspot_dwell_mean_s,
            args.hotspot_dwell_shape,
            args.speed_kmh / 3.6,
            radius_m=args.hotspot_radius_m,
            seed=args.seed + 200,
            minimum_events=args.diagnostic_events,
            minimum_outgoing_per_state=args.diagnostic_min_outgoing,
        )
        np.savez(
            out_dir / "hotspot_process_diagnostics.npz",
            **process_diagnostics,
        )
        process_pass = bool(
            np.nanmax(
                np.abs(
                    process_diagnostics["empirical_transition_matrix"]
                    - transition_matrix
                )
            )
            <= 0.05
            and np.max(
                np.abs(
                    process_diagnostics["event_chain_occupancy"] - 0.25
                )
            )
            <= 0.03
            and abs(
                float(process_diagnostics["event_dwell_mean_s"])
                / args.hotspot_dwell_mean_s
                - 1
            )
            <= 0.05
            and abs(
                float(process_diagnostics["merged_residence_mean_s"])
                / float(
                    process_diagnostics["merged_residence_mean_target_s"]
                )
                - 1
            )
            <= 0.05
            and int(process_diagnostics["event_count"])
            >= args.diagnostic_events
            and int(process_diagnostics["minimum_outgoing_event_count"])
            >= args.diagnostic_min_outgoing
            and float(
                process_diagnostics["self_transition_motion_distance_max_m"]
            )
            == 0
            and float(process_diagnostics["max_boundary_radius_m"])
            <= 100 + 1e-12
            and float(process_diagnostics["max_step_limit_violation_m"])
            <= 1e-10
            and float(process_diagnostics["transit_endpoint_error_max_m"])
            <= 1e-10
        )

    summary = {}
    raw_metrics = None
    inference_pass = True
    if not args.diagnostics_only:
        model = node_update(
            args.M, 6, pmax_w, 64, 5, requested_device
        ).to(requested_device)
        load_frozen_model(model, args.checkpoint, requested_device)
        logger.info("Loaded frozen checkpoint %s", args.checkpoint)
        evaluator = TrajectoryEvaluator(
            model,
            K=args.K,
            pmax_w=pmax_w,
            num_ap=5,
            device=requested_device,
            noise_power=args.noise_power,
            frame_batch_size=args.batch_size,
            time_stride=args.eval_time_stride,
        )
        summary, raw_metrics = evaluator.evaluate(loader)
        np.savez_compressed(out_dir / "raw_metrics.npz", **raw_metrics)
        inference_pass = all(
            bool(raw_metrics[f"{method}_constraints_passed"])
            for method in METHODS
        )
        for key, value in summary.items():
            logger.info("%s = %.8g", key, value)

    completed = bool(
        mobility_pass and independent_pass and process_pass and inference_pass
    )
    summary.update(
        {
            "mobility_gate_passed": mobility_pass,
            "channel_gate_passed": independent_pass,
            "hotspot_process_gate_passed": process_pass,
            "inference_gate_passed": inference_pass,
            "completed": completed,
        }
    )
    with open(out_dir / "summary.json", "w") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
    with open(out_dir / "summary.txt", "w") as output:
        for key, value in summary.items():
            output.write(f"{key}: {value}\n")
    with open(out_dir / "completion.json", "w") as output:
        json.dump(
            {
                "status": "complete" if completed else "failed_gate",
                "checks": {
                    "mobility": mobility_pass,
                    "channel": independent_pass,
                    "hotspot_process": process_pass,
                    "fair_inference": inference_pass,
                },
            },
            output,
            indent=2,
            sort_keys=True,
        )
    if not completed:
        raise RuntimeError("One or more Stage 2 gates failed")
    logger.info("Stage 2 setting completed")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(out_dir)
    try:
        provenance = save_provenance(out_dir, args.checkpoint)
        run(args, logger, out_dir, provenance)
    except Exception:
        completion_path = out_dir / "completion.json"
        if not completion_path.exists():
            with open(completion_path, "w") as output:
                json.dump({"status": "error"}, output, indent=2)
        logger.exception("Stage 2 setting failed")
        raise


if __name__ == "__main__":
    main()
