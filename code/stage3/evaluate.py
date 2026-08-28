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

from association import (
    ASSOCIATION_PERIOD_FRAMES,
    POLICIES,
    PRIMARY_TOP2_POLICIES,
    TOP_L,
    build_association_traces,
    switching_metrics,
)
from environment import (
    DEFAULT_HOTSPOT_CENTERS,
    MOBILITY_HOTSPOT,
    MOBILITY_STRAIGHT,
    TOPOLOGY_TYPE,
    WRAP_AROUND,
    MobilityEnvironment,
    symmetric_transition_matrix,
)
from model_2 import node_update
from utils_return_indivial_rates import (
    HEIGHT_DIFFERENCE,
    SQUARE_SIDE,
    calculate_rates,
    mrt_beamforming,
    rzf_beamforming,
)


BEAMFORMERS = ("rzf", "mrt", "centralized_gnn", "decentralized_gnn")
SOURCE_FILES = (
    "association.py",
    "data.py",
    "environment.py",
    "evaluate.py",
    "model_2.py",
    "run_exp-v3.sh",
    "test_stage3.py",
    "utils_return_indivial_rates.py",
)
EXPECTED_STAGE2_SOURCE_SHA256 = {
    "data.py": "2e34c6c5f343b645a839f704c351ca715114fa55263ca84840b5ca712aa37de4",
    "environment.py": "1416de41ccd87983c8d9f0a9d0d3c9b22c7b29c026a3830523701c6726c2ba6e",
    "model_2.py": "d916ebd5bcf4077b0ff6c9887d84aaf0d1eee30088d5720f8e770b9036225869",
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


def save_provenance(out_dir, checkpoint, ap_coordinates, stage1_config):
    source_dir = Path(__file__).resolve().parent
    snapshot_dir = out_dir / "source_snapshot"
    snapshot_dir.mkdir()
    source_hashes = {}
    for filename in SOURCE_FILES:
        path = source_dir / filename
        shutil.copy2(path, snapshot_dir / filename)
        source_hashes[filename] = sha256(path)
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
        "checkpoint_sha256": sha256(checkpoint),
        "ap_coordinates_sha256": sha256(ap_coordinates),
        "stage1_config_sha256": sha256(stage1_config),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return provenance


def configure_logging(out_dir):
    logger = logging.getLogger("stage3")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(out_dir / "run.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_frozen_model(checkpoint, *, antennas, pmax_w, device):
    model = node_update(antennas, 6, pmax_w, 64, 5, device).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    model.eval()
    return model


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
        if provenance["source_sha256"].get(filename) != expected:
            raise RuntimeError(f"Stage 3 frozen source copy changed: {filename}")
    expected_artifacts = {
        "checkpoint_sha256": EXPECTED_STAGE1C_CHECKPOINT_SHA256,
        "ap_coordinates_sha256": EXPECTED_STAGE1C_AP_SHA256,
        "stage1_config_sha256": EXPECTED_STAGE1C_CONFIG_SHA256,
    }
    for name, expected in expected_artifacts.items():
        if provenance[name] != expected or reference_provenance[name] != expected:
            raise RuntimeError(f"Frozen artifact hash mismatch: {name}")

    reference_cli = config["cli"]
    expected_cli = {
        "M": args.M,
        "K": args.K,
        "pmax_dbm": args.pmax_dbm,
        "noise_power": args.noise_power,
        "seed": args.seed,
        "speed_kmh": args.speed_kmh,
        "mobility_model": args.mobility_model,
        "decision_period_s": args.decision_period_s,
        "carrier_frequency_hz": args.carrier_frequency_hz,
        "episode_steps": args.episode_steps,
        "trajectories": args.trajectories,
        "eval_time_stride": args.eval_time_stride,
    }
    for name, expected in expected_cli.items():
        actual = reference_cli.get(name)
        equal = actual == expected
        if isinstance(expected, float):
            equal = np.isclose(actual, expected, rtol=0, atol=0)
        if not equal:
            raise RuntimeError(
                f"Stage 2 reference config mismatch for {name}: {actual} != {expected}"
            )
    return config, reference_provenance


def save_environment(loader, out_dir):
    values = {
        "mobility_model": np.asarray(loader.mobility_model),
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
    if loader.mobility_model == MOBILITY_HOTSPOT:
        values.update(
            {
                "hotspot_centers": loader.hotspot_centers,
                "hotspot_radius_m": np.asarray(loader.hotspot_radius_m),
                "transition_matrix": loader.transition_matrix,
                "hotspot_state": loader.hotspot_state,
                "mobility_phase": loader.mobility_phase,
                "mobility_sample_phase": loader.mobility_sample_phase,
                "parent_trace_ids": loader.parent_trace_ids,
                "clip_start_times_s": loader.clip_start_times_s,
                "hotspot_burn_in_s": np.asarray(loader.hotspot_burn_in_s),
            }
        )
    np.savez_compressed(out_dir / "environment.npz", **values)


class DynamicAssociationEvaluator:
    def __init__(
        self,
        model,
        *,
        num_users,
        pmax_w,
        num_ap,
        device,
        noise_power,
        frame_batch_size,
        time_stride,
        association_period_frames,
        beamformers=BEAMFORMERS,
    ):
        if frame_batch_size <= 0 or time_stride <= 0:
            raise ValueError("Batch size and time stride must be positive")
        unknown = set(beamformers) - set(BEAMFORMERS)
        if unknown:
            raise ValueError(f"Unknown beamformers: {sorted(unknown)}")
        if any("gnn" in method for method in beamformers) and model is None:
            raise ValueError("GNN evaluation requires a frozen model")
        self.model = model
        self.num_users = num_users
        self.pmax_w = pmax_w
        self.num_ap = num_ap
        self.device = device
        self.noise_power = noise_power
        self.frame_batch_size = frame_batch_size
        self.time_stride = time_stride
        self.association_period_frames = association_period_frames
        self.beamformers = tuple(beamformers)

    def _make_beamformers(self, loader, channels, masks):
        result = {}
        if "rzf" in self.beamformers:
            result["rzf"] = rzf_beamforming(
                channels, masks, self.pmax_w, self.device, self.noise_power
            )
        if "mrt" in self.beamformers:
            result["mrt"] = mrt_beamforming(
                channels, masks, self.pmax_w, self.device
            )
        if any("gnn" in method for method in self.beamformers):
            central, central_mask, local, local_masks = loader._format_frames(
                channels, masks
            )
            central = central.to(self.device)
            local = [feature.to(self.device) for feature in local]
            if "centralized_gnn" in self.beamformers:
                result["centralized_gnn"] = self.model(
                    central, central_mask, training=True, duplicate=False
                )
            if "decentralized_gnn" in self.beamformers:
                result["decentralized_gnn"] = self.model(
                    local, local_masks, training=False, duplicate=False
                )
        return result

    def _constraint_checks(self, weights, masks):
        finite = bool(torch.isfinite(weights).all())
        flat_mask = masks.transpose(0, 2, 1).reshape(
            len(masks), self.num_ap * self.num_users
        )
        mask = torch.as_tensor(
            flat_mask, dtype=torch.bool, device=weights.device
        ).unsqueeze(1).expand_as(weights)
        unassociated = weights[~mask]
        max_unassociated = (
            float(unassociated.abs().max().item()) if unassociated.numel() else 0.0
        )
        max_power = 0.0
        for ap in range(self.num_ap):
            start = ap * self.num_users
            stop = (ap + 1) * self.num_users
            power = weights[:, :, start:stop].square().sum(dim=(1, 2))
            max_power = max(max_power, float(power.max().item()))
        return finite, max_unassociated, max_power

    def evaluate(self, loader, traces, logger=None):
        if self.model is not None and any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise RuntimeError("Stage 3A requires a fully frozen model")
        time_indices = np.arange(0, loader.episode_steps, self.time_stride)
        trajectory_indices = np.repeat(
            np.arange(loader.batch_size), len(time_indices)
        )
        frame_indices = np.tile(time_indices, loader.batch_size)
        raw = {
            "evaluation_time_indices": time_indices,
            "evaluation_trajectory_indices": np.arange(loader.batch_size),
            "policy_names": np.asarray(tuple(traces)),
            "beamformer_names": np.asarray(self.beamformers),
            "shared_current_channel": np.asarray(True),
            "shared_noise_power": np.asarray(self.noise_power),
            "shared_pmax_w": np.asarray(self.pmax_w),
            "association_period_frames": np.asarray(
                self.association_period_frames
            ),
        }
        summary = {}

        with torch.inference_mode():
            for policy, trace in traces.items():
                started = time.monotonic()
                if logger:
                    logger.info("Evaluating association policy %s", policy)
                user_rate_sums = {
                    method: np.zeros((loader.batch_size, self.num_users))
                    for method in self.beamformers
                }
                frame_counts = np.zeros(loader.batch_size, dtype=np.int64)
                checks = {
                    method: {
                        "finite": True,
                        "max_unassociated_abs": 0.0,
                        "max_ap_power_w": 0.0,
                    }
                    for method in self.beamformers
                }
                for start in range(
                    0, len(trajectory_indices), self.frame_batch_size
                ):
                    stop = min(
                        start + self.frame_batch_size, len(trajectory_indices)
                    )
                    trajectory_batch = trajectory_indices[start:stop]
                    time_batch = frame_indices[start:stop]
                    channels = loader.true_channels[trajectory_batch, time_batch]
                    epochs = time_batch // self.association_period_frames
                    masks = trace[trajectory_batch, epochs]
                    weights_by_method = self._make_beamformers(
                        loader, channels, masks
                    )
                    np.add.at(frame_counts, trajectory_batch, 1)
                    for method, weights in weights_by_method.items():
                        finite, max_unassociated, max_power = self._constraint_checks(
                            weights, masks
                        )
                        checks[method]["finite"] &= finite
                        checks[method]["max_unassociated_abs"] = max(
                            checks[method]["max_unassociated_abs"],
                            max_unassociated,
                        )
                        checks[method]["max_ap_power_w"] = max(
                            checks[method]["max_ap_power_w"], max_power
                        )
                        rates = calculate_rates(
                            weights,
                            channels,
                            self.num_ap,
                            self.device,
                            self.noise_power,
                        ).cpu().numpy()
                        checks[method]["finite"] &= bool(
                            np.isfinite(rates).all()
                        )
                        np.add.at(
                            user_rate_sums[method], trajectory_batch, rates
                        )

                summary[policy] = {}
                for method in self.beamformers:
                    key = f"{policy}__{method}"
                    user_means = user_rate_sums[method] / frame_counts[:, None]
                    sum_rates = user_means.sum(axis=1)
                    p05_rates = np.percentile(user_means, 5, axis=1)
                    passed = bool(
                        checks[method]["finite"]
                        and checks[method]["max_unassociated_abs"] == 0
                        and checks[method]["max_ap_power_w"]
                        <= self.pmax_w + 1e-6
                    )
                    raw[f"{key}__per_trajectory_sum_rate"] = sum_rates
                    raw[
                        f"{key}__per_trajectory_user_time_average_rates"
                    ] = user_means
                    raw[f"{key}__per_trajectory_p05_user_rate"] = p05_rates
                    raw[f"{key}__finite"] = np.asarray(
                        checks[method]["finite"]
                    )
                    raw[f"{key}__max_unassociated_abs"] = np.asarray(
                        checks[method]["max_unassociated_abs"]
                    )
                    raw[f"{key}__max_ap_power_w"] = np.asarray(
                        checks[method]["max_ap_power_w"]
                    )
                    raw[f"{key}__constraints_passed"] = np.asarray(passed)
                    summary[policy][method] = {
                        "trajectory_average_sum_rate": float(sum_rates.mean()),
                        "trajectory_average_p05_user_rate": float(
                            p05_rates.mean()
                        ),
                        "constraints_passed": passed,
                    }
                if logger:
                    logger.info(
                        "Completed policy %s in %.1f s",
                        policy,
                        time.monotonic() - started,
                    )
        return summary, raw


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3A full-current-CSI dynamic-association evaluator"
    )
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--pmax_dbm", type=float, default=15.0)
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ap_coordinates", required=True)
    parser.add_argument("--stage1_config", required=True)
    parser.add_argument("--stage2_reference", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--beamformers",
        nargs="+",
        choices=BEAMFORMERS,
        default=list(BEAMFORMERS),
    )
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
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.M != 2 or args.K != 8:
        parser.error("Stage 3 primary contract fixes M=2 and K=8")
    if args.episode_steps % ASSOCIATION_PERIOD_FRAMES:
        parser.error("episode_steps must be divisible by 50")
    if args.speed_kmh < 0:
        parser.error("--speed_kmh cannot be negative")
    if args.mobility_model == MOBILITY_HOTSPOT and args.speed_kmh <= 0:
        parser.error("Hotspot mobility requires --speed_kmh > 0")
    for name in (
        "checkpoint",
        "ap_coordinates",
        "stage1_config",
        "stage2_reference",
    ):
        path = Path(getattr(args, name)).expanduser().resolve()
        if name == "stage2_reference":
            valid = path.is_dir()
        else:
            valid = path.is_file()
        if not valid:
            parser.error(f"{name} does not exist: {path}")
        setattr(args, name, str(path))
    args.out_dir = str(Path(args.out_dir).expanduser().resolve())
    return args


def run(args, out_dir, logger, provenance):
    seed_everything(args.seed)
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
        logger.info("CUDA unavailable; using CPU")
    stage2_config, stage2_provenance = validate_stage2_handoff(
        args, provenance
    )
    logger.info("Gate 3.0 Stage 2 handoff hashes and config passed")

    transition_matrix = symmetric_transition_matrix(
        len(DEFAULT_HOTSPOT_CENTERS), args.hotspot_stickiness
    )
    ap_coordinates = np.loadtxt(args.ap_coordinates)
    pmax_w = 10 ** ((args.pmax_dbm - 30) / 10)
    config = {
        "stage": "3A",
        "execution_mode": "full_current_csi_dynamic_association_heuristics",
        "cli": vars(args),
        "effective_device": str(requested_device),
        "num_ap": 5,
        "top_l": TOP_L,
        "association_period_frames": ASSOCIATION_PERIOD_FRAMES,
        "association_period_s": ASSOCIATION_PERIOD_FRAMES
        * args.decision_period_s,
        "policies": list(POLICIES),
        "beamformers": list(args.beamformers),
        "primary_beamformer": "rzf",
        "csi_policy": "full current true CSI on every evaluated frame",
        "policy_observation": "current large-scale link power at decision epoch",
        "topology_type": TOPOLOGY_TYPE,
        "square_side_m": SQUARE_SIDE,
        "wrap_around": WRAP_AROUND,
        "height_difference_m": HEIGHT_DIFFERENCE,
        "pmax_w": pmax_w,
        "ap_coordinates": ap_coordinates.tolist(),
        "source_sha256": provenance["source_sha256"],
        "stage2_source_sha256": EXPECTED_STAGE2_SOURCE_SHA256,
        "stage2_reference_provenance": stage2_provenance,
        "stage2_reference_config": stage2_config,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    seed_manifest = {
        "split": "development_test",
        "seed_sequence_root_entropy": args.seed,
        "environment_seed_hierarchy": {
            "geometry_spawn_key": [0],
            "channel_spawn_key": [1],
            "implementation": "environment.py MobilityEnvironment",
        },
        "random_top2_seed_sequence_entropy": [args.seed, 303],
        "trajectory_count": args.trajectories,
        "paired_across_policies": True,
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(seed_manifest, indent=2, sort_keys=True) + "\n"
    )

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
        bs_locations=ap_coordinates,
    ).generate_trajectories(args.K, 0.1)
    save_environment(loader, out_dir)
    decision_indices, traces = build_association_traces(
        loader.path_loss_factors,
        loader.true_channels,
        seed=args.seed,
        association_period_frames=ASSOCIATION_PERIOD_FRAMES,
    )

    association_raw = {"decision_indices": decision_indices}
    association_summary = {}
    cardinality_violations = 0
    for policy, trace in traces.items():
        association_raw[f"{policy}__mask"] = trace
        metrics = switching_metrics(
            trace,
            num_frames=args.episode_steps,
            frame_period_s=args.decision_period_s,
        )
        for name, values in metrics.items():
            association_raw[f"{policy}__{name}"] = values
        if policy in PRIMARY_TOP2_POLICIES:
            violations = int(np.count_nonzero(trace.sum(axis=-1) != TOP_L))
            cardinality_violations += violations
        association_summary[policy] = {
            "mean_link_toggles_per_ue_s": float(
                metrics["link_toggles_per_ue_s"].mean()
            ),
            "mean_serving_set_changes_per_ue_s": float(
                metrics["serving_set_changes_per_ue_s"].mean()
            ),
            "mean_ap_load": float(metrics["ap_load_mean_per_trajectory"].mean()),
            "mean_max_ap_load": float(
                metrics["ap_load_max_per_trajectory"].mean()
            ),
            "mean_ap_load_std": float(metrics["ap_load_std_per_trajectory"].mean()),
            "zero_load_ap_fraction": float(
                metrics["zero_load_ap_fraction_per_trajectory"].mean()
            ),
        }
    np.savez_compressed(out_dir / "association_traces.npz", **association_raw)

    zero_speed_pass = True
    if args.speed_kmh == 0:
        for policy in (
            "fixed_lsf_top2_t0",
            "current_lsf_top2",
            "hysteresis_lsf_top2_h0_db",
            "hysteresis_lsf_top2_h3_db",
            "hysteresis_lsf_top2_h6_db",
        ):
            zero_speed_pass &= not np.any(
                association_raw[f"{policy}__link_toggles_per_trajectory"]
            )
    hysteresis_totals = [
        association_raw[
            f"hysteresis_lsf_top2_h{margin:g}_db__link_toggles_per_trajectory"
        ].mean()
        for margin in (0.0, 3.0, 6.0)
    ]
    hysteresis_trend_pass = bool(
        hysteresis_totals[0] >= hysteresis_totals[1]
        and hysteresis_totals[1] >= hysteresis_totals[2]
    )
    moving_signal = bool(
        np.any(
            association_raw[
                "current_lsf_top2__serving_set_replacements_per_trajectory"
            ]
        )
    )
    logger.info(
        "Association traces ready; cardinality violations=%d, moving signal=%s",
        cardinality_violations,
        moving_signal,
    )

    model = None
    if any("gnn" in method for method in args.beamformers):
        model = load_frozen_model(
            args.checkpoint,
            antennas=args.M,
            pmax_w=pmax_w,
            device=requested_device,
        )
        logger.info("Loaded frozen Stage 1C checkpoint")
    evaluator = DynamicAssociationEvaluator(
        model,
        num_users=args.K,
        pmax_w=pmax_w,
        num_ap=5,
        device=requested_device,
        noise_power=args.noise_power,
        frame_batch_size=args.batch_size,
        time_stride=args.eval_time_stride,
        association_period_frames=ASSOCIATION_PERIOD_FRAMES,
        beamformers=args.beamformers,
    )
    rate_summary, raw_metrics = evaluator.evaluate(loader, traces, logger)
    np.savez_compressed(out_dir / "raw_metrics.npz", **raw_metrics)

    reference_raw = np.load(Path(args.stage2_reference) / "raw_metrics.npz")
    reproduction_errors = {}
    reproduction_pass = True
    baseline = "stage2_fixed_instantaneous_rssi_threshold_t0__rzf"
    for suffix in (
        "per_trajectory_sum_rate",
        "per_trajectory_user_time_average_rates",
        "per_trajectory_p05_user_rate",
    ):
        actual = raw_metrics[f"{baseline}__{suffix}"]
        expected = reference_raw[f"rzf_{suffix}"]
        reproduction_errors[suffix] = float(np.max(np.abs(actual - expected)))
        reproduction_pass &= bool(
            np.allclose(actual, expected, rtol=1e-6, atol=1e-8)
        )
    reference_raw.close()
    inference_pass = all(
        bool(raw_metrics[f"{policy}__{method}__constraints_passed"])
        for policy in POLICIES
        for method in args.beamformers
    )
    rate_finite = all(
        np.isfinite(
            rate_summary[policy][method]["trajectory_average_sum_rate"]
        )
        for policy in POLICIES
        for method in args.beamformers
    )
    completed = bool(
        cardinality_violations == 0
        and zero_speed_pass
        and hysteresis_trend_pass
        and reproduction_pass
        and inference_pass
        and rate_finite
    )
    summary = {
        "stage": "3A",
        "association": association_summary,
        "rates": rate_summary,
        "diagnostics": {
            "cardinality_violation_count": cardinality_violations,
            "zero_speed_switching_gate_passed": zero_speed_pass,
            "hysteresis_switching_totals": hysteresis_totals,
            "hysteresis_trend_gate_passed": hysteresis_trend_pass,
            "current_lsf_top2_has_moving_signal": moving_signal,
            "stage2_rzf_reproduction_max_abs_errors": reproduction_errors,
            "stage2_rzf_reproduction_gate_passed": reproduction_pass,
            "beamformer_constraints_gate_passed": inference_pass,
            "rate_finite_gate_passed": rate_finite,
        },
        "completed": completed,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    completion = {
        "status": "complete" if completed else "failed_gate",
        "checks": {
            "stage2_handoff": True,
            "stage2_rzf_reproduction": reproduction_pass,
            "association_cardinality": cardinality_violations == 0,
            "zero_speed_switching": zero_speed_pass,
            "hysteresis_trend": hysteresis_trend_pass,
            "beamformer_constraints": inference_pass,
            "finite_rates": rate_finite,
        },
        "diagnostic_moving_signal": moving_signal,
    }
    (out_dir / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    if not completed:
        raise RuntimeError("One or more Stage 3A gates failed")
    logger.info("Stage 3A setting completed")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_logging(out_dir)
    try:
        provenance = save_provenance(
            out_dir,
            args.checkpoint,
            args.ap_coordinates,
            args.stage1_config,
        )
        run(args, out_dir, logger, provenance)
    except Exception:
        completion_path = out_dir / "completion.json"
        if not completion_path.exists():
            completion_path.write_text('{"status": "error"}\n')
        logger.exception("Stage 3A setting failed")
        raise


if __name__ == "__main__":
    main()
