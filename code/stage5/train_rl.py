"""Train one beamformer-matched Stage 5B AP-local policy."""

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

from evaluate import GNN_BEAMFORMERS, load_frozen_model
from rl_core import (
    ANTENNAS,
    DISCOUNT,
    FEEDBACK_BUDGET,
    GRADIENT_CLIP_NORM,
    HIDDEN_DIM,
    LEARNING_RATE,
    LOCAL_ACTION_DIM,
    LOCAL_OBSERVATION_DIM,
    NUM_AP,
    ReplayBuffer,
    SACTrainer,
    TARGET_SMOOTHING,
    build_episode_pool,
    evaluate_policy,
)


POLICY_NAMES = {
    "rzf": "pi_rzf",
    "centralized_gnn": "pi_c",
    "decentralized_gnn": "pi_d",
}
SOURCE_FILES = (
    "association.py",
    "controller.py",
    "environment.py",
    "evaluate.py",
    "feedback.py",
    "model_2.py",
    "rl_core.py",
    "train_rl.py",
    "evaluate_rl.py",
    "test_stage5.py",
    "run_stage5c.sh",
    "utils_return_indivial_rates.py",
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
    logger = logging.getLogger("stage5b_rl")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(out_dir / "run.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def validate_gate5_5(path):
    root = Path(path)
    completion = json.loads((root / "completion.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if completion.get("status") != "complete" or not summary.get("completed"):
        raise RuntimeError("Gate 5.5 evidence is not complete")
    checks = summary.get("checks", {})
    if not checks or not all(checks.values()):
        raise RuntimeError("Gate 5.5 evidence checks did not all pass")
    return {
        "root": str(root.resolve()),
        "completion_sha256": sha256(root / "completion.json"),
        "summary_sha256": sha256(root / "summary.json"),
    }


def save_provenance(args, out_dir, gate):
    source = Path(__file__).resolve().parent
    snapshot = out_dir / "source_snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for filename in SOURCE_FILES:
        path = source / filename
        shutil.copy2(path, snapshot / filename)
        source_hashes[filename] = sha256(path)
    try:
        repository = source.parents[1]
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
        "gate5_5": gate,
        "checkpoint_sha256": sha256(args.checkpoint),
        "ap_coordinates_sha256": sha256(args.ap_coordinates),
        "stage1_config_sha256": sha256(args.stage1_config),
    }
    write_json(out_dir / "provenance.json", provenance)


def checkpoint_payload(trainer, args, step, validation_utility):
    return {
        "format_version": 1,
        "stage": "5B",
        "policy": POLICY_NAMES[args.beamformer],
        "reward_beamformer": args.beamformer,
        "local_observation_dim": LOCAL_OBSERVATION_DIM,
        "local_action_dim": LOCAL_ACTION_DIM,
        "num_ap": NUM_AP,
        "actor": trainer.actor.state_dict(),
        "critic": trainer.critic.state_dict(),
        "target_critic": trainer.target_critic.state_dict(),
        "log_alpha": trainer.log_alpha.detach().cpu(),
        "actor_optimizer": trainer.actor_optimizer.state_dict(),
        "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "alpha_optimizer": trainer.alpha_optimizer.state_dict(),
        "step": step,
        "validation_utility": validation_utility,
        "selection_rule": "highest matching validation mean utility; earliest tie",
    }


def load_actor(path, device):
    from rl_core import LocalActor

    payload = torch.load(path, map_location=device, weights_only=False)
    actor = LocalActor().to(device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    return actor, payload


def mean_metrics(evaluation):
    return {
        name: float(values.mean())
        for name, values in evaluation["per_trajectory"].items()
    }


def evaluations_equal(first, second):
    arrays = ("raw_actions", "associations", "updates")
    return bool(
        all(np.array_equal(first[name], second[name]) for name in arrays)
        and all(
            np.array_equal(first["per_trajectory"][name], values)
            for name, values in second["per_trajectory"].items()
        )
    )


def run(args, out_dir, logger):
    seed_everything(args.policy_seed)
    gate = validate_gate5_5(args.gate5_5_root)
    save_provenance(args, out_dir, gate)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.info("CUDA unavailable; using CPU")

    ap_coordinates = np.loadtxt(args.ap_coordinates)
    pmax_w = 10 ** ((args.pmax_dbm - 30) / 10)
    model = (
        load_frozen_model(
            args.checkpoint,
            antennas=ANTENNAS,
            pmax_w=pmax_w,
            device=device,
        )
        if args.beamformer in GNN_BEAMFORMERS
        else None
    )
    training, training_manifest = build_episode_pool(
        args.training_trajectories,
        args.policy_seed,
        "training",
        episode_steps=args.episode_steps,
        ap_coordinates=ap_coordinates,
    )
    validation, validation_manifest = build_episode_pool(
        args.validation_trajectories,
        args.policy_seed,
        "validation",
        episode_steps=args.episode_steps,
        ap_coordinates=ap_coordinates,
    )
    split_manifest = {
        "seed_sequence_roots": {
            "training": [args.policy_seed, 0x56, 1],
            "validation": [args.policy_seed, 0x56, 2],
            "formal_reserved": [args.policy_seed, 0x56, 3],
        },
        "training": training_manifest,
        "validation": validation_manifest,
        "paired_across_reward_beamformers": True,
        "development_test_excluded": True,
    }
    write_json(out_dir / "split_manifest.json", split_manifest)
    config = {
        "stage": "5B",
        "execution_mode": (
            "smoke" if args.smoke else "formal" if args.formal else "seed0_pilot"
        ),
        "policy": POLICY_NAMES[args.beamformer],
        "reward_beamformer": args.beamformer,
        "cli": vars(args),
        "effective_device": str(device),
        "architecture": {
            "actor": "parameter-shared AP-local two-layer MLP",
            "critic": "centralized twin two-layer MLP",
            "hidden_dim": HIDDEN_DIM,
            "rnn": False,
            "gnn_actor": False,
        },
        "contract": {
            "state": "local stored CSI, age/update history, previous updates/association, local LSF/load/budget",
            "local_action": "8 association bids + 8 feedback scores",
            "association_projection": "UE-side stable top-2",
            "association_bid_communication": "5 bids per UE per 50 ms association window",
            "feedback_projection": "AP-local stable top-B2 over active links",
            "reward": "matching-beamformer 50-frame sum-rate/K minus switching cost",
            "frozen_gnn_adapter": "batched inference; checked against the Gate 5.5 reference path",
            "current_lsf_assumption": "local slow-timescale measurement outside instantaneous CSI budget",
        },
        "transition_ceiling": args.max_steps,
        "selection_rule": "highest matching validation mean utility; earliest tie",
        "learning_rate": LEARNING_RATE,
        "discount": DISCOUNT,
        "target_smoothing": TARGET_SMOOTHING,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "feedback_budget": FEEDBACK_BUDGET,
    }
    write_json(out_dir / "config.json", config)

    baseline = evaluate_policy(
        None,
        validation,
        args.beamformer,
        model=model,
        device=device,
        pmax_w=pmax_w,
        noise_power=args.noise_power,
        baseline=True,
    )
    trainer = SACTrainer(device)
    replay = ReplayBuffer(args.replay_capacity, (args.policy_seed, 0x56, 4))
    rollout_rng = np.random.default_rng(
        np.random.SeedSequence((args.policy_seed, 0x56, 5))
    )
    history = {name: [] for name in (
        "step", "reward", "sum_rate", "link_toggles", "critic_loss",
        "actor_loss", "alpha_loss", "alpha", "entropy", "mean_q",
        "critic_gradient_norm", "actor_gradient_norm",
    )}
    validation_history = []
    best_utility = -np.inf
    best_step = 0
    environment = None
    observation = None
    done = True

    from rl_core import JointControlEnvironment

    for step in range(1, args.max_steps + 1):
        if done:
            episode = training[rollout_rng.integers(len(training))]
            environment = JointControlEnvironment(
                episode,
                args.beamformer,
                model=model,
                device=device,
                pmax_w=pmax_w,
                noise_power=args.noise_power,
            )
            observation = environment.reset()
        action = (
            rollout_rng.uniform(
                -1, 1, (NUM_AP, LOCAL_ACTION_DIM)
            ).astype(np.float32)
            if step <= args.warmup_steps
            else trainer.act(observation, deterministic=False)
        )
        next_observation, reward, done, info = environment.step(action)
        replay.add(observation, action, reward, next_observation, done)
        observation = next_observation
        history["step"].append(step)
        history["reward"].append(reward)
        history["sum_rate"].append(float(info["sum_rates"].mean()))
        history["link_toggles"].append(info["link_toggles"])
        if step > args.warmup_steps and len(replay) >= args.batch_size:
            for name, value in trainer.update(replay, args.batch_size).items():
                history[name].append(value)

        if step % args.validation_interval and step != args.max_steps:
            continue
        evaluation = evaluate_policy(
            trainer.actor,
            validation,
            args.beamformer,
            model=model,
            device=device,
            pmax_w=pmax_w,
            noise_power=args.noise_power,
        )
        utility = float(evaluation["per_trajectory"]["utility"].mean())
        record = {"step": step, **mean_metrics(evaluation)}
        validation_history.append(record)
        if utility > best_utility:
            best_utility = utility
            best_step = step
            torch.save(
                checkpoint_payload(trainer, args, step, best_utility),
                out_dir / "model_best.pt",
            )
        torch.save(
            checkpoint_payload(trainer, args, step, utility),
            out_dir / "model_latest.pt",
        )
        logger.info(
            "step=%d matching validation utility=%.6f sum-rate=%.6f",
            step,
            utility,
            record["sum_rate"],
        )

    np.savez_compressed(
        out_dir / "training_metrics.npz",
        **{name: np.asarray(values) for name, values in history.items()},
    )
    write_json(out_dir / "validation_history.json", validation_history)
    selected_actor, selected_payload = load_actor(
        out_dir / "model_best.pt", device
    )
    selected = evaluate_policy(
        selected_actor,
        validation,
        args.beamformer,
        model=model,
        device=device,
        pmax_w=pmax_w,
        noise_power=args.noise_power,
    )
    reloaded_actor, _ = load_actor(out_dir / "model_best.pt", device)
    reloaded = evaluate_policy(
        reloaded_actor,
        validation,
        args.beamformer,
        model=model,
        device=device,
        pmax_w=pmax_w,
        noise_power=args.noise_power,
    )
    finite = bool(
        all(
            np.isfinite(np.asarray(values)).all()
            for name, values in history.items()
            if name != "step"
        )
        and all(
            np.isfinite(values).all()
            for values in selected["per_trajectory"].values()
        )
    )
    reload_pass = evaluations_equal(selected, reloaded)
    architecture_checks = bool(
        selected["raw_actions"].shape[-2:] == (NUM_AP, LOCAL_ACTION_DIM)
        and np.all(selected["associations"].sum(axis=-1) == 2)
    )
    selection_check = bool(
        selected_payload["step"] == best_step
        and np.isclose(selected_payload["validation_utility"], best_utility)
    )
    checks = {
        "gate5_5_complete": True,
        "finite_loss_q_entropy_gradients_actions_metrics": finite,
        "checkpoint_reload_deterministic": reload_pass,
        "top2_b2_mask_power_constraints": selected["constraints_passed"],
        "parameter_shared_ap_local_shape": architecture_checks,
        "matching_reward_beamformer": True,
        "checkpoint_selection_consistent": selection_check,
    }
    completed = all(checks.values())
    baseline_metrics = mean_metrics(baseline)
    selected_metrics = mean_metrics(selected)
    summary = {
        "stage": "5B",
        "policy": POLICY_NAMES[args.beamformer],
        "reward_beamformer": args.beamformer,
        "smoke": args.smoke,
        "actual_steps": args.max_steps,
        "best_step": best_step,
        "selection_rule": config["selection_rule"],
        "matching_h3_priority_b2": baseline_metrics,
        "selected_validation": selected_metrics,
        "matching_sum_rate_delta": (
            selected_metrics["sum_rate"] - baseline_metrics["sum_rate"]
        ),
        "matching_baseline_superiority": (
            selected_metrics["sum_rate"] > baseline_metrics["sum_rate"]
        ),
        "checks": checks,
        "completed": completed,
    }
    np.savez_compressed(
        out_dir / "validation_metrics.npz",
        **{
            f"policy__{name}": values
            for name, values in selected["per_trajectory"].items()
        },
        **{
            f"baseline__{name}": values
            for name, values in baseline["per_trajectory"].items()
        },
        policy_raw_actions=selected["raw_actions"],
        policy_associations=selected["associations"],
        policy_updates=selected["updates"],
    )
    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "completion.json",
        {"status": "complete" if completed else "failed_gate", "checks": checks},
    )
    if not completed:
        raise RuntimeError("One or more Stage 5B RL gates failed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beamformer",
        choices=tuple(POLICY_NAMES),
        required=True,
    )
    parser.add_argument("--policy_seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--replay_capacity", type=int, default=4096)
    parser.add_argument("--validation_interval", type=int, default=64)
    parser.add_argument("--training_trajectories", type=int, default=8)
    parser.add_argument("--validation_trajectories", type=int, default=4)
    parser.add_argument("--episode_steps", type=int, default=500)
    parser.add_argument("--pmax_dbm", type=float, default=15.0)
    parser.add_argument("--noise_power", type=float, default=1e-12)
    parser.add_argument("--gate5_5_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ap_coordinates", required=True)
    parser.add_argument("--stage1_config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.formal:
        parser.error("--smoke and --formal are mutually exclusive")
    positive = (
        "max_steps", "batch_size", "replay_capacity", "validation_interval",
        "training_trajectories", "validation_trajectories",
        "episode_steps", "noise_power",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.policy_seed < 0 or args.warmup_steps < 0:
        parser.error("seed and warmup must be nonnegative")
    if args.episode_steps % 50 or args.episode_steps < 100:
        parser.error("episode_steps must be a multiple of 50 and at least 100")
    if args.replay_capacity < args.batch_size:
        parser.error("replay capacity must cover one batch")
    if not np.isclose(args.pmax_dbm, 15.0) or not np.isclose(
        args.noise_power, 1e-12
    ):
        parser.error("Stage 5 fixes 15 dBm and noise_power=1e-12")
    for name in (
        "checkpoint", "ap_coordinates", "stage1_config", "gate5_5_root"
    ):
        path = Path(getattr(args, name)).expanduser().resolve()
        if not path.exists():
            parser.error(f"{name} does not exist: {path}")
        setattr(args, name, str(path))
    args.out_dir = str(Path(args.out_dir).expanduser().resolve())
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_logging(out_dir)
    try:
        run(args, out_dir, logger)
    except Exception:
        if not (out_dir / "completion.json").exists():
            write_json(out_dir / "completion.json", {"status": "error"})
        logger.exception("Stage 5B RL training failed")
        raise


if __name__ == "__main__":
    main()
