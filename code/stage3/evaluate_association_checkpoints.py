import argparse
import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from association import (
    ASSOCIATION_PERIOD_FRAMES,
    TOP_L,
    switching_metrics,
    top_l_mask,
)
from environment import MOBILITY_STRAIGHT, MobilityEnvironment
from evaluate import DynamicAssociationEvaluator
from train_association import (
    ACTION_DIM,
    HISTORY_DISCOUNT,
    NUM_AP,
    NUM_USERS,
    ObservationStatistics,
    load_actor,
    observation_dimension,
    seed_everything,
)


EVALUATION_SETTINGS = (
    ("straight_0_kmh", 0.0),
    ("straight_30_kmh", 30.0),
    ("straight_80_kmh", 80.0),
)
BASELINE_POLICIES = (
    "fixed_lsf_top2_t0",
    "current_lsf_top2",
    "hysteresis_lsf_top2_h3_db",
    "hysteresis_lsf_top2_h6_db",
)
LEARNED_POLICIES = ("sac_current", "sac_history")
SOURCE_FILES = (
    "association.py",
    "environment.py",
    "evaluate.py",
    "evaluate_association_checkpoints.py",
    "model_association.py",
    "train_association.py",
    "utils_return_indivial_rates.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def configure_logging(out_dir):
    logger = logging.getLogger("stage3b_checkpoint_evaluation")
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


def actor_association_trace(actor, lsf_power, statistics, *, variant, device):
    """Build deterministic actor masks without evaluating channel rewards."""
    lsf_power = np.asarray(lsf_power, dtype=np.float64)
    if lsf_power.ndim != 4 or lsf_power.shape[2:] != (NUM_USERS, NUM_AP):
        raise ValueError("LSF power must have shape [trajectory,epoch,UE,AP]")
    if lsf_power.shape[1] < 2:
        raise ValueError("Evaluation needs at least two association epochs")
    if variant not in LEARNED_POLICIES and variant not in ("current", "history"):
        raise ValueError("Unknown learned-policy variant")
    variant = variant.removeprefix("sac_")

    log_lsf = np.log(lsf_power + np.finfo(np.float64).tiny)
    previous_mask = top_l_mask(lsf_power[:, 0])
    history = (
        log_lsf[:, 0] >= statistics.good_lsf_threshold[None]
    ).astype(np.float32)
    trace = np.empty(lsf_power.shape, dtype=bool)
    trace[:, 0] = previous_mask
    actions = []

    for epoch in range(1, lsf_power.shape[1]):
        good = (
            log_lsf[:, epoch] >= statistics.good_lsf_threshold[None]
        ).astype(np.float32)
        history = HISTORY_DISCOUNT * history + (1 - HISTORY_DISCOUNT) * good
        standardized = (
            log_lsf[:, epoch] - statistics.log_lsf_mean[None]
        ) / statistics.log_lsf_std[None]
        load = previous_mask.sum(axis=1) / NUM_USERS
        parts = (
            standardized.reshape(len(lsf_power), -1),
            previous_mask.reshape(len(lsf_power), -1),
            load,
        )
        if variant == "history":
            parts += (history.reshape(len(lsf_power), -1),)
        observation = np.concatenate(parts, axis=1).astype(np.float32)
        if observation.shape[1] != observation_dimension(variant):
            raise RuntimeError("Stage 3B observation shape changed")
        tensor = torch.as_tensor(observation, device=device)
        with torch.inference_mode():
            action, _ = actor.sample(tensor, deterministic=True)
        action = action.cpu().numpy()
        previous_mask = top_l_mask(action.reshape(-1, NUM_USERS, NUM_AP) + 1.0)
        trace[:, epoch] = previous_mask
        actions.append(action)
    return trace, np.stack(actions, axis=1)


def load_checkpoint(checkpoint, expected_variant, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {
        "format_version": 1,
        "variant": expected_variant,
        "observation_dim": observation_dimension(expected_variant),
        "action_dim": ACTION_DIM,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(
                f"Checkpoint {checkpoint} has {name}={payload.get(name)!r}, "
                f"expected {value!r}"
            )
    statistics = ObservationStatistics(**payload["observation_statistics"])
    actor = load_actor(checkpoint, device)
    return actor, statistics, payload


def validate_input_source(path, source_dir):
    provenance_path = path.parent / "provenance.json"
    config_path = path.parent / "config.json"
    if not provenance_path.is_file() or not config_path.is_file():
        raise RuntimeError(f"Checkpoint metadata missing beside {path}")
    provenance = json.loads(provenance_path.read_text())
    for filename in (
        "association.py",
        "environment.py",
        "evaluate.py",
        "model_association.py",
        "train_association.py",
        "utils_return_indivial_rates.py",
    ):
        actual = sha256(source_dir / filename)
        expected = provenance["source_sha256"].get(filename)
        if actual != expected:
            raise RuntimeError(f"Pilot source mismatch for {filename}")
    return provenance, json.loads(config_path.read_text())


def save_provenance(args, out_dir):
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
    result = {
        "git_commit": commit,
        "git_dirty": bool(dirty),
        "git_status_short": dirty,
        "source_sha256": source_hashes,
        "checkpoint_sha256": {
            "sac_current": sha256(args.current_checkpoint),
            "sac_history": sha256(args.history_checkpoint),
        },
        "stage3a_evidence_sha256": {},
    }
    for setting, _ in EVALUATION_SETTINGS:
        evidence = Path(args.stage3a_root) / setting
        result["stage3a_evidence_sha256"][setting] = {
            filename: sha256(evidence / filename)
            for filename in (
                "association_traces.npz",
                "completion.json",
                "config.json",
                "environment.npz",
                "provenance.json",
                "raw_metrics.npz",
                "split_manifest.json",
                "summary.json",
            )
        }
    write_json(out_dir / "provenance.json", result)
    return result


def build_loader_and_traces(evidence_dir, expected_speed, source_dir):
    completion = json.loads((evidence_dir / "completion.json").read_text())
    config = json.loads((evidence_dir / "config.json").read_text())
    provenance = json.loads((evidence_dir / "provenance.json").read_text())
    cli = config["cli"]
    if completion.get("status") != "complete":
        raise RuntimeError(f"Incomplete Stage 3A evidence: {evidence_dir}")
    expected_values = {
        "mobility_model": MOBILITY_STRAIGHT,
        "speed_kmh": expected_speed,
        "seed": 0,
        "M": 2,
        "K": NUM_USERS,
        "episode_steps": 2000,
        "trajectories": 10,
        "eval_time_stride": 10,
    }
    for name, expected in expected_values.items():
        if cli.get(name) != expected:
            raise RuntimeError(
                f"{evidence_dir} has {name}={cli.get(name)!r}, expected {expected!r}"
            )
    for filename in (
        "association.py",
        "environment.py",
        "evaluate.py",
        "utils_return_indivial_rates.py",
    ):
        if sha256(source_dir / filename) != provenance["source_sha256"].get(
            filename
        ):
            raise RuntimeError(f"Stage 3A source mismatch for {filename}")

    ap_coordinates = np.loadtxt(cli["ap_coordinates"])
    seed_everything(cli["seed"])
    loader = MobilityEnvironment(
        cli["M"],
        cli["trajectories"],
        episode_steps=cli["episode_steps"],
        speed_kmh=cli["speed_kmh"],
        decision_period_s=cli["decision_period_s"],
        carrier_frequency_hz=cli["carrier_frequency_hz"],
        seed=cli["seed"],
        mobility_model=cli["mobility_model"],
        bs_locations=ap_coordinates,
    ).generate_trajectories(cli["K"], 0.1)
    with np.load(evidence_dir / "environment.npz") as stored:
        for name, actual in (
            ("ap_coordinates", loader.BS_Loc_array),
            ("ue_positions", loader.ue_positions),
            ("path_loss_factors", loader.path_loss_factors),
            ("rhos", loader.rhos),
        ):
            if not np.array_equal(actual, stored[name]):
                raise RuntimeError(
                    f"Stage 3A deterministic regeneration failed for {name}"
                )
    with np.load(evidence_dir / "association_traces.npz") as stored:
        traces = {
            policy: stored[f"{policy}__mask"].copy()
            for policy in BASELINE_POLICIES
        }
    return loader, traces, cli


def summarize_setting(rate_summary, raw_metrics, traces, cli):
    policies = {}
    duration_s = cli["episode_steps"] * cli["decision_period_s"]
    for policy, trace in traces.items():
        switching = switching_metrics(
            trace,
            num_frames=cli["episode_steps"],
            frame_period_s=cli["decision_period_s"],
        )
        rates = rate_summary[policy]["rzf"]
        policies[policy] = {
            "trajectory_average_sum_rate": rates[
                "trajectory_average_sum_rate"
            ],
            "trajectory_average_p05_user_rate": rates[
                "trajectory_average_p05_user_rate"
            ],
            "mean_link_toggles_per_ue_s": float(
                switching["link_toggles_per_ue_s"].mean()
            ),
            "mean_serving_set_changes_per_ue_s": float(
                switching["serving_set_changes_per_ue_s"].mean()
            ),
            "mean_ap_load": float(
                switching["ap_load_mean_per_trajectory"].mean()
            ),
            "mean_max_ap_load": float(
                switching["ap_load_max_per_trajectory"].mean()
            ),
            "zero_load_ap_fraction": float(
                switching["zero_load_ap_fraction_per_trajectory"].mean()
            ),
            "constraints_passed": rates["constraints_passed"],
            "trajectory_duration_s": duration_s,
        }
    paired = {}
    for learned in LEARNED_POLICIES:
        paired[learned] = {}
        for baseline in BASELINE_POLICIES:
            differences = {}
            for metric in ("sum_rate", "p05_user_rate"):
                learned_values = raw_metrics[
                    f"{learned}__rzf__per_trajectory_{metric}"
                ]
                baseline_values = raw_metrics[
                    f"{baseline}__rzf__per_trajectory_{metric}"
                ]
                delta = learned_values - baseline_values
                differences[metric] = {
                    "mean_paired_difference": float(delta.mean()),
                    "learned_win_fraction": float(np.mean(delta > 0)),
                }
            paired[learned][baseline] = differences
    return {"policies": policies, "paired_differences": paired}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired Stage 3B checkpoint evaluation on Stage 3A traces"
    )
    parser.add_argument("--current_checkpoint", required=True)
    parser.add_argument("--history_checkpoint", required=True)
    parser.add_argument("--stage3a_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lambda_switch", type=float, default=0.5)
    args = parser.parse_args()
    for name in ("current_checkpoint", "history_checkpoint"):
        path = Path(getattr(args, name)).expanduser().resolve()
        if not path.is_file():
            parser.error(f"{name} does not exist: {path}")
        setattr(args, name, str(path))
    root = Path(args.stage3a_root).expanduser().resolve()
    for setting, _ in EVALUATION_SETTINGS:
        if not (root / setting).is_dir():
            parser.error(f"Stage 3A setting does not exist: {root / setting}")
    if args.lambda_switch < 0:
        parser.error("--lambda_switch must be nonnegative")
    args.stage3a_root = str(root)
    args.out_dir = str(Path(args.out_dir).expanduser().resolve())
    return args


def run(args, out_dir, logger, provenance):
    source_dir = Path(__file__).resolve().parent
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.info("CUDA unavailable; using CPU")

    checkpoint_inputs = {
        "current": Path(args.current_checkpoint),
        "history": Path(args.history_checkpoint),
    }
    actors = {}
    statistics = {}
    checkpoint_metadata = {}
    for variant, checkpoint in checkpoint_inputs.items():
        _, pilot_config = validate_input_source(
            checkpoint, source_dir
        )
        actor, variant_statistics, payload = load_checkpoint(
            checkpoint, variant, device
        )
        cli = pilot_config["cli"]
        if cli["policy_seed"] != 0 or cli["lambda_switch"] != args.lambda_switch:
            raise RuntimeError("Checkpoint seed/lambda does not match evaluation")
        actors[variant] = actor
        statistics[variant] = variant_statistics
        checkpoint_metadata[f"sac_{variant}"] = {
            "step": int(payload["step"]),
            "validation_utility": float(payload["validation_utility"]),
            "checkpoint_sha256": provenance["checkpoint_sha256"][
                f"sac_{variant}"
            ],
        }

    config = {
        "stage": "3B",
        "execution_mode": "paired_seed0_checkpoint_development_evaluation",
        "cli": vars(args),
        "effective_device": str(device),
        "settings": [
            {"name": name, "mobility_model": MOBILITY_STRAIGHT, "speed_kmh": speed}
            for name, speed in EVALUATION_SETTINGS
        ],
        "policies": list(BASELINE_POLICIES + LEARNED_POLICIES),
        "beamformer": "rzf",
        "policy_seed": 0,
        "lambda_switch": args.lambda_switch,
        "top_l": TOP_L,
        "selection_rule": "retain both variants; no current/history winner selection",
        "checkpoint_metadata": checkpoint_metadata,
    }
    write_json(out_dir / "config.json", config)
    split_manifest = {
        "split": "development_test",
        "source": "frozen Stage 3A seed-0 straight settings",
        "settings": {
            name: {
                "seed_sequence_root_entropy": 0,
                "trajectory_count": 10,
                "speed_kmh": speed,
            }
            for name, speed in EVALUATION_SETTINGS
        },
        "paired_across_all_policies": True,
        "development_results_not_used_for_variant_selection": True,
        "normalization_statistics_source": "each checkpoint's training split only",
    }
    write_json(out_dir / "split_manifest.json", split_manifest)

    summaries = {}
    reproduction_errors = {}
    learned_determinism = {}
    cardinality_violations = 0
    constraints_pass = True
    finite_pass = True
    reproduction_pass = True

    for setting, speed in EVALUATION_SETTINGS:
        logger.info("Preparing paired development setting %s", setting)
        evidence_dir = Path(args.stage3a_root) / setting
        setting_dir = out_dir / setting
        setting_dir.mkdir()
        loader, traces, cli = build_loader_and_traces(
            evidence_dir, speed, source_dir
        )
        decision_indices = np.arange(
            0, cli["episode_steps"], ASSOCIATION_PERIOD_FRAMES
        )
        lsf_power = np.square(
            loader.path_loss_factors[:, decision_indices]
        ).transpose(0, 1, 3, 2)
        actions = {}
        for variant in ("current", "history"):
            name = f"sac_{variant}"
            trace, variant_actions = actor_association_trace(
                actors[variant],
                lsf_power,
                statistics[variant],
                variant=variant,
                device=device,
            )
            repeated_trace, repeated_actions = actor_association_trace(
                actors[variant],
                lsf_power,
                statistics[variant],
                variant=variant,
                device=device,
            )
            deterministic = bool(
                np.array_equal(trace, repeated_trace)
                and np.array_equal(variant_actions, repeated_actions)
            )
            learned_determinism.setdefault(name, {})[setting] = deterministic
            traces[name] = trace
            actions[name] = variant_actions
        for trace in traces.values():
            cardinality_violations += int(
                np.count_nonzero(trace.sum(axis=-1) != TOP_L)
            )

        evaluator = DynamicAssociationEvaluator(
            None,
            num_users=cli["K"],
            pmax_w=10 ** ((cli["pmax_dbm"] - 30) / 10),
            num_ap=NUM_AP,
            device=device,
            noise_power=cli["noise_power"],
            frame_batch_size=cli["batch_size"],
            time_stride=cli["eval_time_stride"],
            association_period_frames=ASSOCIATION_PERIOD_FRAMES,
            beamformers=("rzf",),
        )
        rate_summary, raw_metrics = evaluator.evaluate(loader, traces, logger)
        with np.load(evidence_dir / "raw_metrics.npz") as reference:
            errors = {}
            for policy in BASELINE_POLICIES:
                for suffix in (
                    "per_trajectory_sum_rate",
                    "per_trajectory_user_time_average_rates",
                    "per_trajectory_p05_user_rate",
                ):
                    name = f"{policy}__rzf__{suffix}"
                    error = float(np.max(np.abs(raw_metrics[name] - reference[name])))
                    errors[name] = error
                    reproduction_pass &= bool(
                        np.allclose(
                            raw_metrics[name], reference[name], rtol=1e-6, atol=1e-8
                        )
                    )
            reproduction_errors[setting] = errors
        constraints_pass &= all(
            values["rzf"]["constraints_passed"]
            for values in rate_summary.values()
        )
        finite_pass &= all(
            np.isfinite(values["rzf"]["trajectory_average_sum_rate"])
            and np.isfinite(values["rzf"]["trajectory_average_p05_user_rate"])
            for values in rate_summary.values()
        )
        np.savez_compressed(
            setting_dir / "association_traces.npz",
            decision_indices=decision_indices,
            **{f"{name}__mask": trace for name, trace in traces.items()},
            **{f"{name}__actions": value for name, value in actions.items()},
        )
        np.savez_compressed(setting_dir / "raw_metrics.npz", **raw_metrics)
        setting_summary = summarize_setting(
            rate_summary, raw_metrics, traces, cli
        )
        write_json(setting_dir / "summary.json", setting_summary)
        summaries[setting] = setting_summary

    aggregate = {}
    for policy in BASELINE_POLICIES + LEARNED_POLICIES:
        aggregate[policy] = {
            metric: float(
                np.mean(
                    [
                        summaries[setting]["policies"][policy][metric]
                        for setting, _ in EVALUATION_SETTINGS
                    ]
                )
            )
            for metric in (
                "trajectory_average_sum_rate",
                "trajectory_average_p05_user_rate",
                "mean_link_toggles_per_ue_s",
                "mean_serving_set_changes_per_ue_s",
            )
        }
    completed = bool(
        reproduction_pass
        and cardinality_violations == 0
        and constraints_pass
        and finite_pass
        and all(
            passed
            for by_setting in learned_determinism.values()
            for passed in by_setting.values()
        )
    )
    summary = {
        "stage": "3B",
        "settings": summaries,
        "equal_weight_setting_aggregate": aggregate,
        "diagnostics": {
            "stage3a_rzf_reproduction_max_abs_errors": reproduction_errors,
            "stage3a_rzf_reproduction_gate_passed": reproduction_pass,
            "learned_checkpoint_determinism": learned_determinism,
            "association_cardinality_violation_count": cardinality_violations,
            "beamformer_constraints_gate_passed": constraints_pass,
            "rate_finite_gate_passed": finite_pass,
        },
        "interpretation_rule": (
            "development evidence only; retain current and history for later "
            "scenario interaction tests"
        ),
        "completed": completed,
    }
    write_json(out_dir / "summary.json", summary)
    completion = {
        "status": "complete" if completed else "failed_gate",
        "checks": {
            "stage3a_deterministic_reproduction": reproduction_pass,
            "learned_checkpoint_determinism": all(
                passed
                for by_setting in learned_determinism.values()
                for passed in by_setting.values()
            ),
            "association_cardinality": cardinality_violations == 0,
            "beamformer_constraints": constraints_pass,
            "finite_rates": finite_pass,
        },
    }
    write_json(out_dir / "completion.json", completion)
    if not completed:
        raise RuntimeError("One or more Stage 3B development-evaluation gates failed")
    logger.info("Paired Stage 3B checkpoint development evaluation completed")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_logging(out_dir)
    try:
        provenance = save_provenance(args, out_dir)
        run(args, out_dir, logger, provenance)
    except Exception:
        completion_path = out_dir / "completion.json"
        if not completion_path.exists():
            write_json(completion_path, {"status": "error"})
        logger.exception("Stage 3B checkpoint development evaluation failed")
        raise


if __name__ == "__main__":
    main()
