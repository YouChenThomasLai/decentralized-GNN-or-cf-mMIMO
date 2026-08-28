import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from association import ASSOCIATION_PERIOD_FRAMES, TOP_L, top_l_mask
from environment import MOBILITY_STRAIGHT, MobilityEnvironment
from evaluate import (
    EXPECTED_STAGE1C_AP_SHA256,
    EXPECTED_STAGE1C_CHECKPOINT_SHA256,
    EXPECTED_STAGE1C_CONFIG_SHA256,
    EXPECTED_STAGE2_SOURCE_SHA256,
)
from model_association import (
    AssociationActor,
    TwinAssociationCritic,
    initialize_linear_layers,
)
from utils_return_indivial_rates import calculate_rates, rzf_beamforming


NUM_AP = 5
NUM_USERS = 8
ACTION_DIM = NUM_AP * NUM_USERS
HISTORY_DISCOUNT = 0.8
LEARNING_RATE = 1e-4
DISCOUNT = 0.99
TARGET_SMOOTHING = 0.005
GRADIENT_CLIP_NORM = 5.0
HIDDEN_DIM = 64
TRAINING_SETTINGS = (
    (MOBILITY_STRAIGHT, 30.0),
    (MOBILITY_STRAIGHT, 80.0),
)
SOURCE_FILES = (
    "association.py",
    "data.py",
    "environment.py",
    "evaluate.py",
    "model_2.py",
    "model_association.py",
    "run_exp-v3.sh",
    "run_stage3b.sh",
    "test_stage3.py",
    "test_stage3b.py",
    "train_association.py",
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
    logger = logging.getLogger("stage3b")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(out_dir / "run.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@dataclass(frozen=True)
class AssociationEpisode:
    lsf_power: np.ndarray
    reward_channels: np.ndarray


@dataclass(frozen=True)
class ObservationStatistics:
    log_lsf_mean: np.ndarray
    log_lsf_std: np.ndarray
    good_lsf_threshold: np.ndarray

    def as_dict(self):
        return {
            "log_lsf_mean": self.log_lsf_mean,
            "log_lsf_std": self.log_lsf_std,
            "good_lsf_threshold": self.good_lsf_threshold,
        }


def fit_observation_statistics(episodes):
    log_lsf = np.concatenate(
        [np.log(episode.lsf_power + np.finfo(np.float64).tiny) for episode in episodes]
    )
    standard_deviation = log_lsf.std(axis=0)
    return ObservationStatistics(
        log_lsf_mean=log_lsf.mean(axis=0),
        log_lsf_std=np.maximum(standard_deviation, 1e-6),
        good_lsf_threshold=np.median(log_lsf, axis=0),
    )


def observation_dimension(variant):
    base = ACTION_DIM + ACTION_DIM + NUM_AP
    return base + (ACTION_DIM if variant == "history" else 0)


class AssociationDecisionEnvironment:
    def __init__(
        self,
        episode,
        statistics,
        *,
        variant,
        lambda_switch,
        pmax_w,
        noise_power,
        device,
        decision_period_s=0.001,
    ):
        if variant not in ("current", "history"):
            raise ValueError("variant must be current or history")
        self.episode = episode
        self.statistics = statistics
        self.variant = variant
        self.lambda_switch = lambda_switch
        self.pmax_w = pmax_w
        self.noise_power = noise_power
        self.device = device
        self.decision_period_s = decision_period_s
        self.epoch = None
        self.previous_mask = None
        self.history = None

    def _log_lsf(self, epoch):
        return np.log(
            self.episode.lsf_power[epoch] + np.finfo(np.float64).tiny
        )

    def _advance_history(self):
        good = (
            self._log_lsf(self.epoch)
            >= self.statistics.good_lsf_threshold
        ).astype(np.float32)
        self.history = (
            HISTORY_DISCOUNT * self.history
            + (1 - HISTORY_DISCOUNT) * good
        )

    def _observation(self):
        standardized = (
            (self._log_lsf(self.epoch) - self.statistics.log_lsf_mean)
            / self.statistics.log_lsf_std
        )
        load = self.previous_mask.sum(axis=0) / NUM_USERS
        parts = (standardized.ravel(), self.previous_mask.ravel(), load)
        if self.variant == "history":
            parts += (self.history.ravel(),)
        observation = np.concatenate(parts).astype(np.float32)
        if observation.shape != (observation_dimension(self.variant),):
            raise RuntimeError("Stage 3B observation shape changed")
        return observation

    def reset(self):
        if len(self.episode.lsf_power) < 2:
            raise ValueError("An episode needs at least two association epochs")
        self.previous_mask = top_l_mask(self.episode.lsf_power[0])
        self.history = (
            self._log_lsf(0) >= self.statistics.good_lsf_threshold
        ).astype(np.float32)
        self.epoch = 1
        self._advance_history()
        return self._observation()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(NUM_USERS, NUM_AP)
        if not np.all(np.isfinite(action)):
            raise ValueError("Association action must be finite")
        mask = top_l_mask(action + 1.0)
        if np.any(mask.sum(axis=-1) != TOP_L):
            raise RuntimeError("Learned action violates top-2 contract")

        channels = self.episode.reward_channels[self.epoch - 1]
        masks = np.broadcast_to(
            mask, (len(channels), NUM_USERS, NUM_AP)
        ).copy()
        with torch.inference_mode():
            weights = rzf_beamforming(
                channels, masks, self.pmax_w, self.device, self.noise_power
            )
            rates = calculate_rates(
                weights, channels, NUM_AP, self.device, self.noise_power
            )
        mean_user_rate = float(rates.mean().item())
        sum_rate = float(rates.sum(dim=1).mean().item())
        link_toggles = int(np.count_nonzero(mask != self.previous_mask))
        switching_cost = self.lambda_switch * link_toggles / ACTION_DIM
        reward = mean_user_rate - switching_cost

        self.previous_mask = mask
        done = self.epoch == len(self.episode.lsf_power) - 1
        if done:
            next_observation = np.zeros(
                observation_dimension(self.variant), dtype=np.float32
            )
        else:
            self.epoch += 1
            self._advance_history()
            next_observation = self._observation()
        return next_observation, reward, done, {
            "mean_user_rate": mean_user_rate,
            "sum_rate": sum_rate,
            "link_toggles": link_toggles,
            "mask": mask.copy(),
        }


class ReplayBuffer:
    def __init__(self, capacity, observation_dim, action_dim, seed):
        self.capacity = capacity
        self.observations = np.empty((capacity, observation_dim), np.float32)
        self.actions = np.empty((capacity, action_dim), np.float32)
        self.rewards = np.empty((capacity, 1), np.float32)
        self.next_observations = np.empty(
            (capacity, observation_dim), np.float32
        )
        self.dones = np.empty((capacity, 1), np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, observation, action, reward, next_observation, done):
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.dones[index] = done
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        indices = self.rng.integers(self.size, size=batch_size)
        arrays = (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
        )
        return tuple(torch.as_tensor(value, device=device) for value in arrays)

    def __len__(self):
        return self.size


class SACTrainer:
    def __init__(self, observation_dim, action_dim, device):
        self.device = device
        self.actor = AssociationActor(
            observation_dim, action_dim, HIDDEN_DIM
        ).to(device)
        self.critic = TwinAssociationCritic(
            observation_dim, action_dim, HIDDEN_DIM
        ).to(device)
        self.target_critic = TwinAssociationCritic(
            observation_dim, action_dim, HIDDEN_DIM
        ).to(device)
        initialize_linear_layers(self.actor)
        initialize_linear_layers(self.critic)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=LEARNING_RATE
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=LEARNING_RATE
        )
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam(
            (self.log_alpha,), lr=LEARNING_RATE
        )
        self.target_entropy = -float(action_dim)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, observation, deterministic):
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.inference_mode():
            action, _ = self.actor.sample(tensor, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, replay, batch_size):
        observations, actions, rewards, next_observations, dones = replay.sample(
            batch_size, self.device
        )
        with torch.no_grad():
            next_actions, next_log_probability = self.actor.sample(
                next_observations
            )
            next_q1, next_q2 = self.target_critic(
                next_observations, next_actions
            )
            next_q = torch.minimum(next_q1, next_q2)
            next_q -= self.alpha.detach() * next_log_probability
            target = rewards + DISCOUNT * (1 - dones) * next_q

        q1, q2 = self.critic(observations, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), GRADIENT_CLIP_NORM
        )
        self.critic_optimizer.step()

        sampled_actions, log_probability = self.actor.sample(observations)
        actor_q1, actor_q2 = self.critic(observations, sampled_actions)
        actor_loss = (
            self.alpha.detach() * log_probability
            - torch.minimum(actor_q1, actor_q2)
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), GRADIENT_CLIP_NORM
        )
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_probability + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_parameter.lerp_(parameter, TARGET_SMOOTHING)

        values = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "critic_gradient_norm": float(critic_gradient),
            "actor_gradient_norm": float(actor_gradient),
            "mean_q": float(torch.minimum(q1, q2).mean().item()),
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError("Non-finite SAC update")
        if not all(
            torch.isfinite(parameter).all()
            for module in (self.actor, self.critic, self.target_critic)
            for parameter in module.parameters()
        ):
            raise FloatingPointError("Non-finite SAC parameter")
        return values


def _setting_counts(total):
    if total < len(TRAINING_SETTINGS):
        raise ValueError("Trajectory count must cover all training settings")
    counts = [total // len(TRAINING_SETTINGS)] * len(TRAINING_SETTINGS)
    for index in range(total % len(TRAINING_SETTINGS)):
        counts[index] += 1
    return counts


def build_episode_pool(
    *,
    count,
    root_seed,
    split,
    antennas,
    episode_steps,
    reward_time_stride,
    generation_batch_size,
    ap_coordinates,
    decision_period_s,
    carrier_frequency_hz,
):
    episodes = []
    manifest = []
    split_code = {"training": 1, "validation": 2}[split]
    root = np.random.SeedSequence((root_seed, 0x3B, split_code))
    children = iter(root.spawn(sum(_setting_counts(count))))
    decision_indices = np.arange(0, episode_steps, ASSOCIATION_PERIOD_FRAMES)
    reward_offsets = np.arange(0, ASSOCIATION_PERIOD_FRAMES, reward_time_stride)

    for (mobility_model, speed_kmh), setting_count in zip(
        TRAINING_SETTINGS, _setting_counts(count)
    ):
        remaining = setting_count
        while remaining:
            batch_size = min(remaining, generation_batch_size)
            seed = int(next(children).generate_state(1)[0])
            loader = MobilityEnvironment(
                antennas,
                batch_size,
                episode_steps=episode_steps,
                speed_kmh=speed_kmh,
                decision_period_s=decision_period_s,
                carrier_frequency_hz=carrier_frequency_hz,
                seed=seed,
                mobility_model=mobility_model,
                bs_locations=ap_coordinates,
            ).generate_trajectories(NUM_USERS, 0.1)
            lsf = np.square(
                loader.path_loss_factors[:, decision_indices]
            ).transpose(0, 1, 3, 2)
            reward_indices = (
                decision_indices[1:, None] + reward_offsets[None]
            )
            channels = loader.true_channels[:, reward_indices]
            for trajectory in range(batch_size):
                episodes.append(
                    AssociationEpisode(
                        lsf_power=lsf[trajectory].copy(),
                        reward_channels=channels[trajectory].copy(),
                    )
                )
            manifest.append(
                {
                    "mobility_model": mobility_model,
                    "speed_kmh": speed_kmh,
                    "seed": seed,
                    "trajectories": batch_size,
                }
            )
            remaining -= batch_size
    return episodes, manifest


def action_rank_diagnostics(actions, previous_actions=None):
    actions = np.asarray(actions, dtype=np.float64).reshape(-1, NUM_USERS, NUM_AP)
    ordered = np.sort(actions, axis=-1)
    diagnostics = {
        "mean_top2_margin": float(
            np.mean(ordered[..., -TOP_L] - ordered[..., -TOP_L - 1])
        ),
        "mean_abs_action": float(np.mean(np.abs(actions))),
        "mean_abs_action_change_from_previous": None,
        "serving_set_change_fraction_from_previous": None,
    }
    if previous_actions is None:
        return diagnostics
    previous = np.asarray(previous_actions, dtype=np.float64).reshape(actions.shape)
    diagnostics["mean_abs_action_change_from_previous"] = float(
        np.mean(np.abs(actions - previous))
    )
    masks = top_l_mask(actions + 1.0)
    previous_masks = top_l_mask(previous + 1.0)
    diagnostics["serving_set_change_fraction_from_previous"] = float(
        np.mean(np.any(masks != previous_masks, axis=-1))
    )
    return diagnostics


def top2_plateau_at_gate(validation_history, step, gate_step):
    if step != gate_step or len(validation_history) < 2:
        return False
    baseline_utility = validation_history[0]["mean_utility"]
    return all(
        record["mean_abs_action_change_from_previous"] > 0
        and record["serving_set_change_fraction_from_previous"] == 0
        and record["mean_utility"] == baseline_utility
        for record in validation_history[1:]
    )


def evaluate_actor(
    actor,
    episodes,
    statistics,
    *,
    variant,
    lambda_switch,
    pmax_w,
    noise_power,
    device,
    random_seed=None,
):
    rng = np.random.default_rng(random_seed)
    trajectory_utilities = []
    trajectory_sum_rates = []
    trajectory_toggles = []
    actions = []
    for episode in episodes:
        environment = AssociationDecisionEnvironment(
            episode,
            statistics,
            variant=variant,
            lambda_switch=lambda_switch,
            pmax_w=pmax_w,
            noise_power=noise_power,
            device=device,
        )
        observation = environment.reset()
        rewards = []
        sum_rates = []
        toggles = 0
        done = False
        while not done:
            if random_seed is None:
                tensor = torch.as_tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.inference_mode():
                    action, _ = actor.sample(tensor, deterministic=True)
                action = action.squeeze(0).cpu().numpy()
            else:
                action = rng.uniform(-1, 1, ACTION_DIM).astype(np.float32)
            observation, reward, done, info = environment.step(action)
            actions.append(action)
            rewards.append(reward)
            sum_rates.append(info["sum_rate"])
            toggles += info["link_toggles"]
        trajectory_utilities.append(np.mean(rewards))
        trajectory_sum_rates.append(np.mean(sum_rates))
        trajectory_toggles.append(toggles)
    duration_s = len(episodes[0].lsf_power) * ASSOCIATION_PERIOD_FRAMES * 0.001
    return {
        "mean_utility": float(np.mean(trajectory_utilities)),
        "mean_sum_rate": float(np.mean(trajectory_sum_rates)),
        "mean_link_toggles_per_ue_s": float(
            np.mean(trajectory_toggles) / (NUM_USERS * duration_s)
        ),
        "trajectory_utilities": np.asarray(trajectory_utilities),
        "trajectory_sum_rates": np.asarray(trajectory_sum_rates),
        "trajectory_link_toggles": np.asarray(trajectory_toggles),
        "actions": np.asarray(actions),
    }


def checkpoint_payload(trainer, statistics, args, step, validation_utility):
    return {
        "format_version": 1,
        "variant": args.variant,
        "observation_dim": observation_dimension(args.variant),
        "action_dim": ACTION_DIM,
        "actor": trainer.actor.state_dict(),
        "critic": trainer.critic.state_dict(),
        "target_critic": trainer.target_critic.state_dict(),
        "log_alpha": trainer.log_alpha.detach().cpu(),
        "actor_optimizer": trainer.actor_optimizer.state_dict(),
        "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "alpha_optimizer": trainer.alpha_optimizer.state_dict(),
        "observation_statistics": statistics.as_dict(),
        "step": step,
        "validation_utility": validation_utility,
    }


def load_actor(checkpoint, device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    actor = AssociationActor(
        payload["observation_dim"], payload["action_dim"], HIDDEN_DIM
    ).to(device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    return actor


def save_provenance(args, out_dir):
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
        "checkpoint_sha256": sha256(args.checkpoint),
        "ap_coordinates_sha256": sha256(args.ap_coordinates),
        "stage1_config_sha256": sha256(args.stage1_config),
        "stage3a_summary_sha256": sha256(
            Path(args.stage3a_evidence) / "summary.json"
        ),
        "stage3a_completion_sha256": sha256(
            Path(args.stage3a_evidence) / "completion.json"
        ),
    }
    write_json(out_dir / "provenance.json", provenance)
    return provenance


def validate_handoff(args, provenance):
    source_hashes = provenance["source_sha256"]
    for filename, expected in EXPECTED_STAGE2_SOURCE_SHA256.items():
        if source_hashes.get(filename) != expected:
            raise RuntimeError(f"Frozen Stage 2 source changed: {filename}")
    expected_artifacts = {
        "checkpoint_sha256": EXPECTED_STAGE1C_CHECKPOINT_SHA256,
        "ap_coordinates_sha256": EXPECTED_STAGE1C_AP_SHA256,
        "stage1_config_sha256": EXPECTED_STAGE1C_CONFIG_SHA256,
    }
    for name, expected in expected_artifacts.items():
        if provenance[name] != expected:
            raise RuntimeError(f"Frozen artifact hash mismatch: {name}")
    evidence_dir = Path(args.stage3a_evidence)
    completion = json.loads((evidence_dir / "completion.json").read_text())
    summary = json.loads((evidence_dir / "summary.json").read_text())
    diagnostics = summary.get("diagnostics", {})
    required = (
        completion.get("status") == "complete",
        diagnostics.get("current_lsf_top2_has_moving_signal") is True,
        diagnostics.get("rate_finite_gate_passed") is True,
        diagnostics.get("beamformer_constraints_gate_passed") is True,
        diagnostics.get("stage2_rzf_reproduction_gate_passed") is True,
    )
    if not all(required):
        raise RuntimeError("Stage 3A evidence does not pass provisional Gate 3.3")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3B centralized top-2 SAC association training"
    )
    parser.add_argument("--variant", choices=("current", "history"), required=True)
    parser.add_argument("--policy_seed", type=int, default=0)
    parser.add_argument("--lambda_switch", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=500_000)
    parser.add_argument("--warmup_steps", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replay_capacity", type=int, default=1_000_000)
    parser.add_argument("--validation_interval", type=int, default=10_000)
    parser.add_argument("--early_stop_after", type=int, default=100_000)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--training_trajectories", type=int, default=256)
    parser.add_argument("--validation_trajectories", type=int, default=16)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--episode_steps", type=int, default=2000)
    parser.add_argument("--reward_time_stride", type=int, default=10)
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--pmax_dbm", type=float, default=15.0)
    parser.add_argument("--noise_power", type=float, default=1e-12)
    parser.add_argument("--decision_period_s", type=float, default=0.001)
    parser.add_argument("--carrier_frequency_hz", type=float, default=2.6e9)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ap_coordinates", required=True)
    parser.add_argument("--stage1_config", required=True)
    parser.add_argument("--stage3a_evidence", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    positive = (
        "max_steps",
        "batch_size",
        "replay_capacity",
        "validation_interval",
        "early_stop_after",
        "early_stop_patience",
        "training_trajectories",
        "validation_trajectories",
        "generation_batch_size",
        "episode_steps",
        "reward_time_stride",
        "M",
        "K",
        "noise_power",
        "decision_period_s",
        "carrier_frequency_hz",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.warmup_steps < 0 or args.policy_seed < 0 or args.lambda_switch < 0:
        parser.error("seed, warmup, and lambda_switch must be nonnegative")
    if args.M != 2 or args.K != NUM_USERS:
        parser.error("Stage 3 fixes M=2 and K=8")
    if not np.isclose(args.decision_period_s, 0.001, rtol=0, atol=0):
        parser.error("Stage 3 fixes decision_period_s=0.001")
    if not np.isclose(args.carrier_frequency_hz, 2.6e9, rtol=0, atol=0):
        parser.error("Stage 3 fixes carrier_frequency_hz=2.6e9")
    if not np.isclose(args.pmax_dbm, 15.0, rtol=0, atol=0):
        parser.error("Stage 3 fixes pmax_dbm=15")
    if not np.isclose(args.noise_power, 1e-12, rtol=0, atol=0):
        parser.error("Stage 3 fixes noise_power=1e-12")
    if args.episode_steps < 2 * ASSOCIATION_PERIOD_FRAMES:
        parser.error("episode_steps must contain at least two association epochs")
    if args.episode_steps % ASSOCIATION_PERIOD_FRAMES:
        parser.error("episode_steps must be divisible by 50")
    if ASSOCIATION_PERIOD_FRAMES % args.reward_time_stride:
        parser.error("reward_time_stride must divide the 50-frame window")
    if args.replay_capacity < args.batch_size:
        parser.error("replay_capacity must be at least batch_size")
    for name in ("checkpoint", "ap_coordinates", "stage1_config"):
        path = Path(getattr(args, name)).expanduser().resolve()
        if not path.is_file():
            parser.error(f"{name} does not exist: {path}")
        setattr(args, name, str(path))
    evidence = Path(args.stage3a_evidence).expanduser().resolve()
    if not evidence.is_dir():
        parser.error(f"stage3a_evidence does not exist: {evidence}")
    args.stage3a_evidence = str(evidence)
    args.out_dir = str(Path(args.out_dir).expanduser().resolve())
    return args


def run(args, out_dir, logger, provenance):
    seed_everything(args.policy_seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.info("CUDA unavailable; using CPU")
    evidence = validate_handoff(args, provenance)
    logger.info("Provisional Gate 3.3 evidence passed: %s", args.stage3a_evidence)

    ap_coordinates = np.loadtxt(args.ap_coordinates)
    pmax_w = 10 ** ((args.pmax_dbm - 30) / 10)
    train_episodes, train_manifest = build_episode_pool(
        count=args.training_trajectories,
        root_seed=args.policy_seed,
        split="training",
        antennas=args.M,
        episode_steps=args.episode_steps,
        reward_time_stride=args.reward_time_stride,
        generation_batch_size=args.generation_batch_size,
        ap_coordinates=ap_coordinates,
        decision_period_s=args.decision_period_s,
        carrier_frequency_hz=args.carrier_frequency_hz,
    )
    validation_episodes, validation_manifest = build_episode_pool(
        count=args.validation_trajectories,
        root_seed=args.policy_seed,
        split="validation",
        antennas=args.M,
        episode_steps=args.episode_steps,
        reward_time_stride=args.reward_time_stride,
        generation_batch_size=args.generation_batch_size,
        ap_coordinates=ap_coordinates,
        decision_period_s=args.decision_period_s,
        carrier_frequency_hz=args.carrier_frequency_hz,
    )
    statistics = fit_observation_statistics(train_episodes)
    np.savez_compressed(out_dir / "observation_statistics.npz", **statistics.as_dict())
    split_manifest = {
        "scope": "straight_only_30_80_kmh",
        "policy_seed": args.policy_seed,
        "seed_sequence_roots": {
            "training": [args.policy_seed, 0x3B, 1],
            "validation": [args.policy_seed, 0x3B, 2],
            "development_test_reserved": [args.policy_seed, 0x3B, 3],
        },
        "training": train_manifest,
        "validation": validation_manifest,
        "paired_validation_across_variants": True,
        "stage2_evaluation_trajectories_excluded": True,
    }
    write_json(out_dir / "split_manifest.json", split_manifest)

    config = {
        "stage": "3B",
        "training_scope": "straight_only_30_80_kmh",
        "training_settings": [
            {"mobility": mobility, "speed_kmh": speed_kmh}
            for mobility, speed_kmh in TRAINING_SETTINGS
        ],
        "execution_mode": "smoke" if args.smoke else "training",
        "cli": vars(args),
        "effective_device": str(device),
        "observation_dim": observation_dimension(args.variant),
        "action_dim": ACTION_DIM,
        "top_l": TOP_L,
        "hidden_layers": [HIDDEN_DIM, HIDDEN_DIM],
        "learning_rate": LEARNING_RATE,
        "discount": DISCOUNT,
        "target_smoothing": TARGET_SMOOTHING,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "history_discount": HISTORY_DISCOUNT,
        "automatic_entropy_tuning": True,
        "primary_beamformer": "rzf",
        "gate3_3_evidence": evidence["diagnostics"],
        "gate3_3_status": "provisional_single_completed_moving_setting",
    }
    write_json(out_dir / "config.json", config)
    logger.info(
        "Built %d training and %d fixed validation trajectories",
        len(train_episodes),
        len(validation_episodes),
    )

    observation_dim = observation_dimension(args.variant)
    trainer = SACTrainer(observation_dim, ACTION_DIM, device)
    replay = ReplayBuffer(
        args.replay_capacity,
        observation_dim,
        ACTION_DIM,
        np.random.SeedSequence((args.policy_seed, 0x3B, 4)),
    )
    rollout_rng = np.random.default_rng(
        np.random.SeedSequence((args.policy_seed, 0x3B, 5))
    )
    random_validation = evaluate_actor(
        None,
        validation_episodes,
        statistics,
        variant=args.variant,
        lambda_switch=args.lambda_switch,
        pmax_w=pmax_w,
        noise_power=args.noise_power,
        device=device,
        random_seed=int(
            np.random.SeedSequence((args.policy_seed, 0x3B, 6)).generate_state(1)[0]
        ),
    )
    logger.info(
        "Random-top2 validation utility %.6f",
        random_validation["mean_utility"],
    )

    histories = {
        "step": [],
        "reward": [],
        "mean_user_rate": [],
        "sum_rate": [],
        "link_toggles": [],
        "critic_loss": [],
        "actor_loss": [],
        "alpha_loss": [],
        "alpha": [],
        "mean_q": [],
        "critic_gradient_norm": [],
        "actor_gradient_norm": [],
    }
    validation_history = []
    previous_validation_actions = None
    best_utility = -np.inf
    best_step = 0
    without_improvement = 0
    environment = None
    observation = None
    done = True
    actual_steps = 0
    early_stop_reason = None

    for step in range(1, args.max_steps + 1):
        if done:
            episode = train_episodes[rollout_rng.integers(len(train_episodes))]
            environment = AssociationDecisionEnvironment(
                episode,
                statistics,
                variant=args.variant,
                lambda_switch=args.lambda_switch,
                pmax_w=pmax_w,
                noise_power=args.noise_power,
                device=device,
                decision_period_s=args.decision_period_s,
            )
            observation = environment.reset()
        if step <= args.warmup_steps:
            action = rollout_rng.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        else:
            action = trainer.act(observation, deterministic=False)
        next_observation, reward, done, info = environment.step(action)
        replay.add(observation, action, reward, next_observation, done)
        observation = next_observation
        actual_steps = step
        histories["step"].append(step)
        histories["reward"].append(reward)
        histories["mean_user_rate"].append(info["mean_user_rate"])
        histories["sum_rate"].append(info["sum_rate"])
        histories["link_toggles"].append(info["link_toggles"])

        if step > args.warmup_steps and len(replay) >= args.batch_size:
            update_metrics = trainer.update(replay, args.batch_size)
            for name, value in update_metrics.items():
                histories[name].append(value)

        validate_now = step % args.validation_interval == 0 or step == args.max_steps
        if not validate_now:
            continue
        validation = evaluate_actor(
            trainer.actor,
            validation_episodes,
            statistics,
            variant=args.variant,
            lambda_switch=args.lambda_switch,
            pmax_w=pmax_w,
            noise_power=args.noise_power,
            device=device,
        )
        record = {
            "step": step,
            "mean_utility": validation["mean_utility"],
            "mean_sum_rate": validation["mean_sum_rate"],
            "mean_link_toggles_per_ue_s": validation[
                "mean_link_toggles_per_ue_s"
            ],
        }
        record.update(
            action_rank_diagnostics(
                validation["actions"], previous_validation_actions
            )
        )
        previous_validation_actions = validation["actions"].copy()
        validation_history.append(record)
        if validation["mean_utility"] > best_utility:
            best_utility = validation["mean_utility"]
            best_step = step
            without_improvement = 0
            torch.save(
                checkpoint_payload(
                    trainer, statistics, args, step, best_utility
                ),
                out_dir / "model_best.pt",
            )
        else:
            without_improvement += 1
        torch.save(
            checkpoint_payload(trainer, statistics, args, step, best_utility),
            out_dir / "model_latest.pt",
        )
        np.savez_compressed(
            out_dir / "training_metrics.npz",
            **{name: np.asarray(values) for name, values in histories.items()},
        )
        write_json(out_dir / "validation_history.json", validation_history)
        logger.info(
            "step=%d validation utility=%.6f sum-rate=%.6f toggles/UE/s=%.4f "
            "top2-margin=%.6f action-change=%s mask-change=%s",
            step,
            record["mean_utility"],
            record["mean_sum_rate"],
            record["mean_link_toggles_per_ue_s"],
            record["mean_top2_margin"],
            record["mean_abs_action_change_from_previous"],
            record["serving_set_change_fraction_from_previous"],
        )
        if top2_plateau_at_gate(
            validation_history, step, args.early_stop_after
        ):
            early_stop_reason = "top2_plateau_at_gate"
            logger.info(
                "Top-2 plateau persisted through %d steps; stopping before "
                "%d-step ceiling",
                step,
                args.max_steps,
            )
            break
        if (
            step >= args.early_stop_after
            and without_improvement >= args.early_stop_patience
        ):
            early_stop_reason = "validation_patience"
            logger.info(
                "Early stopping after %d validation checks",
                without_improvement,
            )
            break

    torch.save(
        checkpoint_payload(trainer, statistics, args, actual_steps, best_utility),
        out_dir / "model_latest.pt",
    )
    np.savez_compressed(
        out_dir / "training_metrics.npz",
        **{name: np.asarray(values) for name, values in histories.items()},
    )
    write_json(out_dir / "validation_history.json", validation_history)

    selected_actor = load_actor(out_dir / "model_best.pt", device)
    selected_evaluation = evaluate_actor(
        selected_actor,
        validation_episodes,
        statistics,
        variant=args.variant,
        lambda_switch=args.lambda_switch,
        pmax_w=pmax_w,
        noise_power=args.noise_power,
        device=device,
    )
    reloaded_actor = load_actor(out_dir / "model_best.pt", device)
    reloaded_evaluation = evaluate_actor(
        reloaded_actor,
        validation_episodes,
        statistics,
        variant=args.variant,
        lambda_switch=args.lambda_switch,
        pmax_w=pmax_w,
        noise_power=args.noise_power,
        device=device,
    )
    reload_pass = bool(
        np.array_equal(
            selected_evaluation["actions"], reloaded_evaluation["actions"]
        )
        and np.array_equal(
            selected_evaluation["trajectory_utilities"],
            reloaded_evaluation["trajectory_utilities"],
        )
    )
    finite_pass = bool(
        all(
            np.isfinite(np.asarray(values)).all()
            for name, values in histories.items()
            if name != "step"
        )
        and np.isfinite(selected_evaluation["mean_utility"])
    )
    beats_random = bool(
        selected_evaluation["mean_utility"] > random_validation["mean_utility"]
    )
    completed = finite_pass and reload_pass and (args.smoke or beats_random)
    summary = {
        "stage": "3B",
        "variant": args.variant,
        "policy_seed": args.policy_seed,
        "lambda_switch": args.lambda_switch,
        "smoke": args.smoke,
        "actual_steps": actual_steps,
        "best_step": best_step,
        "early_stop_reason": early_stop_reason,
        "selected_validation": {
            name: selected_evaluation[name]
            for name in (
                "mean_utility",
                "mean_sum_rate",
                "mean_link_toggles_per_ue_s",
            )
        },
        "selected_action_diagnostics": action_rank_diagnostics(
            selected_evaluation["actions"]
        ),
        "random_top2_validation": {
            name: random_validation[name]
            for name in (
                "mean_utility",
                "mean_sum_rate",
                "mean_link_toggles_per_ue_s",
            )
        },
        "checks": {
            "finite_training_and_validation": finite_pass,
            "checkpoint_reload_deterministic": reload_pass,
            "beats_random_top2": beats_random,
            "performance_gate_required": not args.smoke,
        },
        "completed": completed,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "completion.json",
        {
            "status": "complete" if completed else "failed_gate",
            "smoke": args.smoke,
            "checks": summary["checks"],
        },
    )
    if not completed:
        raise RuntimeError("One or more Stage 3B gates failed")
    logger.info("Stage 3B %s completed", "smoke" if args.smoke else "training")


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
            write_json(completion_path, {"status": "error", "smoke": args.smoke})
        logger.exception("Stage 3B run failed")
        raise


if __name__ == "__main__":
    main()
