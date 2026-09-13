"""Paired matching and crossed evaluation for Stage 5B policies."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from environment import MOBILITY_STRAIGHT, MobilityEnvironment
from evaluate import load_frozen_model
from rl_core import ANTENNAS, JointEpisode, NUM_USERS, evaluate_policy
from train_rl import (
    POLICY_NAMES,
    SOURCE_FILES,
    load_actor,
    sha256,
    validate_gate5_5,
    write_json,
)


PATHS = {
    "pi_rzf__rzf": ("rzf", "rzf"),
    "pi_c__centralized_gnn": ("centralized_gnn", "centralized_gnn"),
    "pi_d__decentralized_gnn": ("decentralized_gnn", "decentralized_gnn"),
    "pi_c__decentralized_gnn": ("centralized_gnn", "decentralized_gnn"),
    "pi_d__centralized_gnn": ("decentralized_gnn", "centralized_gnn"),
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_run(root, expected_beamformer, device):
    root = Path(root)
    completion = json.loads((root / "completion.json").read_text())
    config = json.loads((root / "config.json").read_text())
    if completion.get("status") != "complete":
        raise RuntimeError(f"Incomplete policy run: {root}")
    if config["reward_beamformer"] != expected_beamformer:
        raise RuntimeError(f"Policy/reward mismatch: {root}")
    actor, payload = load_actor(root / "model_best.pt", device)
    if payload["reward_beamformer"] != expected_beamformer:
        raise RuntimeError(f"Checkpoint/reward mismatch: {root}")
    return root, config, actor


def audit_protocol(runs):
    configs = [item[1] for item in runs.values()]
    manifests = [(item[0] / "split_manifest.json").read_bytes() for item in runs.values()]
    return {
        "same_actor_architecture": len(
            {json.dumps(config["architecture"], sort_keys=True) for config in configs}
        ) == 1,
        "same_transition_ceiling": len(
            {config["transition_ceiling"] for config in configs}
        ) == 1,
        "same_selection_rule": len(
            {config["selection_rule"] for config in configs}
        ) == 1,
        "same_split_manifest": len({sha256_bytes(value) for value in manifests}) == 1,
    }


def development_episodes(args):
    loader = MobilityEnvironment(
        ANTENNAS,
        args.trajectories,
        episode_steps=args.episode_steps,
        speed_kmh=args.speed_kmh,
        seed=args.environment_seed,
        mobility_model=MOBILITY_STRAIGHT,
        bs_locations=np.loadtxt(args.ap_coordinates),
    ).generate_trajectories(NUM_USERS, 0.1)
    return [
        JointEpisode(
            true_channels=loader.true_channels[index].copy(),
            lsf_power=np.square(loader.path_loss_factors[index]),
            rhos=loader.rhos[index].copy(),
            speed_kmh=args.speed_kmh,
            seed=args.environment_seed,
        )
        for index in range(args.trajectories)
    ]


def store_evaluation(path, evaluation):
    np.savez_compressed(
        path,
        **evaluation["per_trajectory"],
        raw_actions=evaluation["raw_actions"],
        associations=evaluation["associations"],
        updates=evaluation["updates"],
        constraints_passed=np.asarray(evaluation["constraints_passed"]),
        **{
            name: np.asarray(value)
            for name, value in evaluation["input_hashes"].items()
        },
    )


def mean_metrics(evaluation):
    return {
        name: float(values.mean())
        for name, values in evaluation["per_trajectory"].items()
    }


def save_provenance(args, out_dir, gate, runs):
    source = Path(__file__).resolve().parent
    write_json(
        out_dir / "provenance.json",
        {
            "source_sha256": {
                name: sha256(source / name) for name in SOURCE_FILES
            },
            "gate5_5": gate,
            "stage1_checkpoint_sha256": sha256(args.checkpoint),
            "ap_coordinates_sha256": sha256(args.ap_coordinates),
            "policy_checkpoint_sha256": {
                beamformer: sha256(item[0] / "model_best.pt")
                for beamformer, item in runs.items()
            },
        },
    )


def run_setting(args, out_dir):
    gate = validate_gate5_5(args.gate5_5_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    roots = {
        "rzf": args.pi_rzf_root,
        "centralized_gnn": args.pi_c_root,
        "decentralized_gnn": args.pi_d_root,
    }
    runs = {
        beamformer: load_run(root, beamformer, device)
        for beamformer, root in roots.items()
    }
    protocol = audit_protocol(runs)
    if not all(protocol.values()):
        raise RuntimeError("Policy architecture/split/ceiling/selection mismatch")
    save_provenance(args, out_dir, gate, runs)
    pmax_w = 10 ** ((args.pmax_dbm - 30) / 10)
    model = load_frozen_model(
        args.checkpoint,
        antennas=ANTENNAS,
        pmax_w=pmax_w,
        device=device,
    )
    episodes = development_episodes(args)
    baselines = {}
    for beamformer in POLICY_NAMES:
        baselines[beamformer] = evaluate_policy(
            None,
            episodes,
            beamformer,
            model=model if "gnn" in beamformer else None,
            device=device,
            pmax_w=pmax_w,
            noise_power=args.noise_power,
            baseline=True,
        )
        store_evaluation(
            out_dir / f"baseline_h3_priority_b2__{beamformer}.npz",
            baselines[beamformer],
        )

    evaluations = {}
    for label, (policy_beamformer, evaluation_beamformer) in PATHS.items():
        actor = runs[policy_beamformer][2]
        evaluations[label] = evaluate_policy(
            actor,
            episodes,
            evaluation_beamformer,
            model=model if "gnn" in evaluation_beamformer else None,
            device=device,
            pmax_w=pmax_w,
            noise_power=args.noise_power,
        )
        store_evaluation(out_dir / f"{label}.npz", evaluations[label])

    matching_labels = {
        "rzf": "pi_rzf__rzf",
        "centralized_gnn": "pi_c__centralized_gnn",
        "decentralized_gnn": "pi_d__decentralized_gnn",
    }
    matching = {}
    for beamformer, label in matching_labels.items():
        policy_metrics = mean_metrics(evaluations[label])
        baseline_metrics = mean_metrics(baselines[beamformer])
        matching[beamformer] = {
            "policy": policy_metrics,
            "h3_priority_b2": baseline_metrics,
            "sum_rate_delta": (
                policy_metrics["sum_rate"] - baseline_metrics["sum_rate"]
            ),
            "superiority": (
                policy_metrics["sum_rate"] > baseline_metrics["sum_rate"]
            ),
        }
    cross_policy_projection = {}
    for first, second in (
        ("rzf", "centralized_gnn"),
        ("rzf", "decentralized_gnn"),
        ("centralized_gnn", "decentralized_gnn"),
    ):
        first_values = evaluations[matching_labels[first]]
        second_values = evaluations[matching_labels[second]]
        label = f"{POLICY_NAMES[first]}__vs__{POLICY_NAMES[second]}"
        cross_policy_projection[label] = {
            "raw_actions_identical": np.array_equal(
                first_values["raw_actions"], second_values["raw_actions"]
            ),
            "raw_actions_max_abs_difference": float(np.max(np.abs(
                first_values["raw_actions"] - second_values["raw_actions"]
            ))),
            "associations_identical": np.array_equal(
                first_values["associations"], second_values["associations"]
            ),
            "updates_identical": np.array_equal(
                first_values["updates"], second_values["updates"]
            ),
        }
    crossed_action_checks = {
        "pi_c_actions_identical": np.array_equal(
            evaluations["pi_c__centralized_gnn"]["raw_actions"],
            evaluations["pi_c__decentralized_gnn"]["raw_actions"],
        ),
        "pi_c_associations_identical": np.array_equal(
            evaluations["pi_c__centralized_gnn"]["associations"],
            evaluations["pi_c__decentralized_gnn"]["associations"],
        ),
        "pi_c_stored_csi_identical": (
            evaluations["pi_c__centralized_gnn"]["input_hashes"][
                "stored_csi_sha256"
            ]
            == evaluations["pi_c__decentralized_gnn"]["input_hashes"][
                "stored_csi_sha256"
            ]
        ),
        "pi_d_actions_identical": np.array_equal(
            evaluations["pi_d__decentralized_gnn"]["raw_actions"],
            evaluations["pi_d__centralized_gnn"]["raw_actions"],
        ),
        "pi_d_associations_identical": np.array_equal(
            evaluations["pi_d__decentralized_gnn"]["associations"],
            evaluations["pi_d__centralized_gnn"]["associations"],
        ),
        "pi_d_stored_csi_identical": (
            evaluations["pi_d__decentralized_gnn"]["input_hashes"][
                "stored_csi_sha256"
            ]
            == evaluations["pi_d__centralized_gnn"]["input_hashes"][
                "stored_csi_sha256"
            ]
        ),
    }
    checks = {
        "gate5_5_complete": True,
        "protocol_identical": all(protocol.values()),
        "all_matching_and_crossed_constraints": all(
            item["constraints_passed"]
            for item in (*baselines.values(), *evaluations.values())
        ),
        "crossed_controller_actions_identical": all(crossed_action_checks.values()),
        "per_trajectory_metrics_saved": all(
            len(item["per_trajectory"]["sum_rate"]) == args.trajectories
            for item in (*baselines.values(), *evaluations.values())
        ),
    }
    completed = all(checks.values())
    config = {
        "stage": "5B-gate5.6-5.7",
        "cli": vars(args),
        "effective_device": str(device),
        "gate5_5": gate,
        "protocol_audit": protocol,
        "paths": PATHS,
        "matching_comparison_rule": "each policy only versus its own H3+priority@B2 baseline",
        "crossed_interpretation": "policy transfer diagnostic, not controller gain across beamformers",
    }
    summary = {
        "speed_kmh": args.speed_kmh,
        "environment_seed": args.environment_seed,
        "matching": matching,
        "crossed": {
            label: mean_metrics(evaluations[label])
            for label in (
                "pi_c__decentralized_gnn",
                "pi_d__centralized_gnn",
            )
        },
        "crossed_action_checks": crossed_action_checks,
        "cross_policy_projection": cross_policy_projection,
        "checks": checks,
        "completed": completed,
    }
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "completion.json",
        {"status": "complete" if completed else "failed_gate", "checks": checks},
    )
    if not completed:
        raise RuntimeError("Gate 5.6-5.7 setting failed")


def aggregate(root):
    root = Path(root).expanduser().resolve()
    settings = {}
    for path in sorted(root.glob("straight_*_kmh")):
        completion = json.loads((path / "completion.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        if completion.get("status") != "complete" or not summary.get("completed"):
            raise RuntimeError(f"Incomplete evaluation setting: {path}")
        settings[str(summary["speed_kmh"])] = summary
    checks = {
        "straight_0_30_80_complete": set(settings) == {"0.0", "30.0", "80.0"},
        "all_gate5_6_5_7_checks_passed": all(
            all(summary["checks"].values()) for summary in settings.values()
        ),
    }
    completed = all(checks.values())
    write_json(
        root / "summary.json",
        {"settings": settings, "checks": checks, "completed": completed},
    )
    write_json(
        root / "completion.json",
        {"status": "complete" if completed else "failed_gate", "checks": checks},
    )
    if not completed:
        raise RuntimeError("Gate 5.6-5.7 aggregation failed")


def exact_sign_pvalue(values):
    values = np.asarray(values)
    positive = int(np.count_nonzero(values > 0))
    negative = int(np.count_nonzero(values < 0))
    count = positive + negative
    if not count:
        return 1.0
    tail = min(positive, negative)
    probability = sum(math.comb(count, index) for index in range(tail + 1))
    return min(1.0, 2 * probability / (2 ** count))


def aggregate_formal(root):
    root = Path(root).expanduser().resolve()
    records = {}
    raw = {}
    completed_seeds = []
    protocol_records = []
    architectures = set()
    transition_ceilings = set()
    selection_rules = set()
    for seed_root in sorted(root.glob("seed_*")):
        policy_configs = []
        split_manifests = []
        for beamformer, policy in POLICY_NAMES.items():
            policy_root = seed_root / "policies" / policy
            policy_completion = json.loads(
                (policy_root / "completion.json").read_text()
            )
            config = json.loads((policy_root / "config.json").read_text())
            if policy_completion.get("status") != "complete":
                raise RuntimeError(f"Incomplete formal policy: {policy_root}")
            if config["reward_beamformer"] != beamformer:
                raise RuntimeError(f"Formal policy/reward mismatch: {policy_root}")
            policy_configs.append(config)
            split_manifests.append(
                (policy_root / "split_manifest.json").read_bytes()
            )
            architectures.add(json.dumps(config["architecture"], sort_keys=True))
            transition_ceilings.add(config["transition_ceiling"])
            selection_rules.add(config["selection_rule"])
        protocol_records.append({
            "same_actor_architecture": len({
                json.dumps(config["architecture"], sort_keys=True)
                for config in policy_configs
            }) == 1,
            "same_transition_ceiling": len({
                config["transition_ceiling"] for config in policy_configs
            }) == 1,
            "same_selection_rule": len({
                config["selection_rule"] for config in policy_configs
            }) == 1,
            "same_split_manifest": len({
                sha256_bytes(value) for value in split_manifests
            }) == 1,
            "execution_mode_formal": all(
                config["execution_mode"] == "formal"
                for config in policy_configs
            ),
        })
        evaluation = seed_root / "evaluation"
        completion = json.loads((evaluation / "completion.json").read_text())
        summary = json.loads((evaluation / "summary.json").read_text())
        if completion.get("status") != "complete" or not summary.get("completed"):
            raise RuntimeError(f"Incomplete formal seed: {seed_root}")
        seed = int(seed_root.name.split("_")[-1])
        completed_seeds.append(seed)
        for speed, setting in summary["settings"].items():
            for beamformer, comparison in setting["matching"].items():
                key = f"{beamformer}__{speed}_kmh"
                records.setdefault(key, []).append(comparison["sum_rate_delta"])
    checks = {
        "five_paired_seeds_complete": completed_seeds == [0, 1, 2, 3, 4],
        "all_values_finite": all(
            np.isfinite(values).all() for values in records.values()
        ),
        "all_matching_cells_present": len(records) == len(POLICY_NAMES) * 3,
        "within_seed_protocol_identical": all(
            all(values.values()) for values in protocol_records
        ),
        "same_architecture_across_seeds": len(architectures) == 1,
        "same_transition_ceiling_across_seeds": len(transition_ceilings) == 1,
        "same_selection_rule_across_seeds": len(selection_rules) == 1,
    }
    for key, values in records.items():
        values = np.asarray(values, dtype=np.float64)
        raw[f"{key}__sum_rate_delta"] = values
        records[key] = {
            "mean_sum_rate_delta": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "positive_seed_count": int(np.count_nonzero(values > 0)),
            "two_sided_exact_sign_pvalue": exact_sign_pvalue(values),
        }
    completed = all(checks.values())
    np.savez_compressed(root / "formal_paired_deltas.npz", **raw)
    write_json(
        root / "summary.json",
        {
            "seeds": completed_seeds,
            "protocol_audit": protocol_records,
            "matching_results": records,
            "checks": checks,
            "completed": completed,
            "claim_boundary": "five-seed formal paired evidence; crossed paths remain transfer diagnostics",
        },
    )
    write_json(
        root / "completion.json",
        {"status": "complete" if completed else "failed_gate", "checks": checks},
    )
    if not completed:
        raise RuntimeError("Formal multi-seed aggregation failed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate_root")
    parser.add_argument("--aggregate_formal_root")
    parser.add_argument("--speed_kmh", type=float, default=0.0)
    parser.add_argument("--environment_seed", type=int, default=0)
    parser.add_argument("--trajectories", type=int, default=10)
    parser.add_argument("--episode_steps", type=int, default=2000)
    parser.add_argument("--pmax_dbm", type=float, default=15.0)
    parser.add_argument("--noise_power", type=float, default=1e-12)
    parser.add_argument("--gate5_5_root")
    parser.add_argument("--pi_rzf_root")
    parser.add_argument("--pi_c_root")
    parser.add_argument("--pi_d_root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--ap_coordinates")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir")
    args = parser.parse_args()
    if args.aggregate_root or args.aggregate_formal_root:
        return args
    if args.speed_kmh not in (0, 30, 80):
        parser.error("Gate 5.7 fixes straight 0/30/80 km/h")
    if args.episode_steps != 2000 or args.trajectories != 10:
        parser.error("Gate 5.7 fixes 10 trajectories x 2000 frames")
    required = (
        "gate5_5_root", "pi_rzf_root", "pi_c_root", "pi_d_root",
        "checkpoint", "ap_coordinates", "out_dir",
    )
    for name in required:
        value = getattr(args, name)
        if not value:
            parser.error(f"--{name} is required")
        setattr(args, name, str(Path(value).expanduser().resolve()))
    return args


def main():
    args = parse_args()
    if args.aggregate_formal_root:
        aggregate_formal(args.aggregate_formal_root)
        return
    if args.aggregate_root:
        aggregate(args.aggregate_root)
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    try:
        run_setting(args, out_dir)
    except Exception:
        if not (out_dir / "completion.json").exists():
            write_json(out_dir / "completion.json", {"status": "error"})
        raise


if __name__ == "__main__":
    main()
