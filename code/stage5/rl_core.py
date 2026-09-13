"""Minimal AP-local SAC core for Stage 5B."""

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from association import ASSOCIATION_PERIOD_FRAMES, TOP_L, top_l_mask
from controller import H3, build_modular_trace
from environment import MOBILITY_STRAIGHT, MobilityEnvironment
from evaluate import beamformer_checks, make_beamformer
from utils_return_indivial_rates import calculate_rates


NUM_AP = 5
NUM_USERS = 8
ANTENNAS = 2
FEEDBACK_BUDGET = 2
LOCAL_ACTION_DIM = 2 * NUM_USERS
LOCAL_OBSERVATION_DIM = 5 * NUM_USERS + 2 * ANTENNAS * NUM_USERS + 2
HIDDEN_DIM = 64
LEARNING_RATE = 1e-4
DISCOUNT = 0.99
TARGET_SMOOTHING = 0.005
GRADIENT_CLIP_NORM = 5.0
HISTORY_DISCOUNT = 0.9
LAMBDA_SWITCH = 0.5


def mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, output_dim),
    )


def initialize_linear_layers(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)


class LocalActor(nn.Module):
    """One parameter-shared feed-forward policy evaluated independently per AP."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(LOCAL_OBSERVATION_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.mean = nn.Linear(HIDDEN_DIM, LOCAL_ACTION_DIM)
        self.log_std = nn.Linear(HIDDEN_DIM, LOCAL_ACTION_DIM)

    def forward(self, local_observations):
        hidden = self.backbone(local_observations)
        return self.mean(hidden), self.log_std(hidden).clamp(-20, 2)

    def sample(self, local_observations, deterministic=False):
        mean, log_std = self(local_observations)
        if deterministic:
            return torch.tanh(mean), None
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw_actions = distribution.rsample()
        actions = torch.tanh(raw_actions)
        log_probability = distribution.log_prob(raw_actions)
        log_probability -= torch.log(1 - actions.square() + 1e-6)
        return actions, log_probability.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)


class CentralTwinCritic(nn.Module):
    """Twin Q networks over all AP observations and actions during training only."""

    def __init__(self):
        super().__init__()
        input_dim = NUM_AP * (LOCAL_OBSERVATION_DIM + LOCAL_ACTION_DIM)
        self.q1 = mlp(input_dim, 1)
        self.q2 = mlp(input_dim, 1)

    def forward(self, observations, actions):
        inputs = torch.cat(
            (observations.flatten(1), actions.flatten(1)), dim=-1
        )
        return self.q1(inputs), self.q2(inputs)


class ReplayBuffer:
    def __init__(self, capacity, seed):
        self.capacity = int(capacity)
        self.observations = np.empty(
            (capacity, NUM_AP, LOCAL_OBSERVATION_DIM), dtype=np.float32
        )
        self.actions = np.empty(
            (capacity, NUM_AP, LOCAL_ACTION_DIM), dtype=np.float32
        )
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
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
    def __init__(self, device):
        self.device = device
        self.actor = LocalActor().to(device)
        self.critic = CentralTwinCritic().to(device)
        self.target_critic = CentralTwinCritic().to(device)
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
        self.target_entropy = -float(NUM_AP * LOCAL_ACTION_DIM)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, observations, deterministic):
        tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.inference_mode():
            actions, _ = self.actor.sample(tensor, deterministic=deterministic)
        return actions.squeeze(0).cpu().numpy()

    def update(self, replay, batch_size):
        observations, actions, rewards, next_observations, dones = replay.sample(
            batch_size, self.device
        )
        with torch.no_grad():
            next_actions, next_log_probability = self.actor.sample(next_observations)
            next_q1, next_q2 = self.target_critic(
                next_observations, next_actions
            )
            target = rewards + DISCOUNT * (1 - dones) * (
                torch.minimum(next_q1, next_q2)
                - self.alpha.detach() * next_log_probability
            )

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
            self.log_alpha
            * (log_probability + self.target_entropy).detach()
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
            "entropy": float(-log_probability.mean().item()),
            "mean_q": float(torch.minimum(q1, q2).mean().item()),
            "critic_gradient_norm": float(critic_gradient),
            "actor_gradient_norm": float(actor_gradient),
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError("Non-finite Stage 5 SAC update")
        modules = (self.actor, self.critic, self.target_critic)
        if not all(
            torch.isfinite(parameter).all()
            for module in modules
            for parameter in module.parameters()
        ):
            raise FloatingPointError("Non-finite Stage 5 SAC parameter")
        return values


def project_association(local_actions):
    actions = np.asarray(local_actions, dtype=np.float32)
    if actions.shape != (NUM_AP, LOCAL_ACTION_DIM):
        raise ValueError("Actions must have shape [5,16]")
    if not np.isfinite(actions).all():
        raise ValueError("Actions must be finite")
    return top_l_mask(actions[:, :NUM_USERS].T + 1.0)


def project_feedback(scores, active, ages, previous_updates):
    """Stable AP-local top-B projection with causal age and one-frame cooldown."""
    scores = np.asarray(scores, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    effective = scores + np.minimum(ages / ASSOCIATION_PERIOD_FRAMES, 1.0)
    effective -= 2.0 * previous_updates
    effective[~active] = -np.inf
    updates = np.zeros_like(active)
    for ap in range(NUM_AP):
        count = min(FEEDBACK_BUDGET, int(active[ap].sum()))
        if count:
            order = np.argsort(-effective[ap], kind="stable")
            updates[ap, order[:count]] = True
    return updates, effective


@dataclass(frozen=True)
class JointEpisode:
    true_channels: np.ndarray
    lsf_power: np.ndarray
    rhos: np.ndarray
    speed_kmh: float
    seed: int


def build_episode_pool(
    count,
    root_seed,
    split,
    *,
    episode_steps,
    ap_coordinates,
):
    """Build the beamformer-independent 30/80 km/h split."""
    split_code = {"training": 1, "validation": 2, "formal": 3}[split]
    speeds = (30.0, 80.0)
    children = iter(
        np.random.SeedSequence((root_seed, 0x56, split_code)).spawn(count)
    )
    episodes = []
    manifest = []
    for index in range(count):
        speed = speeds[index % len(speeds)]
        seed = int(next(children).generate_state(1)[0])
        loader = MobilityEnvironment(
            ANTENNAS,
            1,
            episode_steps=episode_steps,
            speed_kmh=speed,
            seed=seed,
            mobility_model=MOBILITY_STRAIGHT,
            bs_locations=ap_coordinates,
        ).generate_trajectories(NUM_USERS, 0.1)
        episodes.append(
            JointEpisode(
                true_channels=loader.true_channels[0].copy(),
                lsf_power=np.square(loader.path_loss_factors[0]),
                rhos=loader.rhos[0].copy(),
                speed_kmh=speed,
                seed=seed,
            )
        )
        manifest.append({"speed_kmh": speed, "seed": seed})
    return episodes, manifest


class JointControlEnvironment:
    """One 50 ms transition using only legal AP-local controller inputs."""

    def __init__(
        self,
        episode,
        beamformer,
        *,
        model,
        device,
        pmax_w,
        noise_power,
        baseline=False,
    ):
        if len(episode.true_channels) % ASSOCIATION_PERIOD_FRAMES:
            raise ValueError("Episode length must be divisible by 50")
        self.episode = episode
        self.beamformer = beamformer
        self.model = model
        self.device = device
        self.pmax_w = pmax_w
        self.noise_power = noise_power
        self.baseline = baseline
        path_loss = np.sqrt(episode.lsf_power)[None]
        _, _, trace = build_modular_trace(path_loss, H3)
        self.h3_trace = trace[0]
        self.num_epochs = len(trace[0])
        self.reset()

    def reset(self):
        self.epoch = 0
        self.stored = self.episode.true_channels[0].copy()
        self.ages = np.zeros((NUM_AP, NUM_USERS), dtype=np.int32)
        self.previous_updates = np.zeros((NUM_AP, NUM_USERS), dtype=bool)
        self.update_history = np.zeros((NUM_AP, NUM_USERS), dtype=np.float32)
        self.association = self.h3_trace[0].copy()
        return self._observation()

    def _observation(self):
        frame = self.epoch * ASSOCIATION_PERIOD_FRAMES
        lsf = self.episode.lsf_power[frame]
        scale = np.sqrt(np.maximum(lsf, np.finfo(np.float64).tiny))[..., None]
        normalized = self.stored / scale
        local_lsf = np.clip(np.log10(lsf + np.finfo(np.float64).tiny), -16, -4) / 16
        active = self.association.T
        parts = (
            (np.clip(normalized.real, -10, 10) / 10).reshape(NUM_AP, -1),
            (np.clip(normalized.imag, -10, 10) / 10).reshape(NUM_AP, -1),
            np.minimum(self.ages / 2000.0, 1.0),
            self.update_history,
            self.previous_updates.astype(np.float32),
            active.astype(np.float32),
            local_lsf,
            active.sum(axis=-1, keepdims=True) / NUM_USERS,
            np.full((NUM_AP, 1), FEEDBACK_BUDGET / NUM_USERS),
        )
        observation = np.concatenate(parts, axis=-1).astype(np.float32)
        if observation.shape != (NUM_AP, LOCAL_OBSERVATION_DIM):
            raise RuntimeError("Stage 5 local observation contract changed")
        if not np.isfinite(observation).all():
            raise FloatingPointError("Non-finite Stage 5 observation")
        return observation

    def _baseline_feedback(self, active):
        priority = self.episode.lsf_power[0] * (
            1 - np.abs(self.episode.rhos[0][None]) ** (2 * (self.ages + 1))
        )
        priority[~active] = -np.inf
        updates = np.zeros_like(active)
        for ap in range(NUM_AP):
            count = min(FEEDBACK_BUDGET, int(active[ap].sum()))
            if count:
                order = np.argsort(-priority[ap], kind="stable")
                updates[ap, order[:count]] = True
        return updates, priority

    def step(self, local_actions=None):
        if self.baseline:
            association = self.h3_trace[self.epoch]
            feedback_scores = None
        else:
            actions = np.asarray(local_actions, dtype=np.float32)
            if actions.shape != (NUM_AP, LOCAL_ACTION_DIM):
                raise ValueError("Actions must have shape [5,16]")
            if not np.isfinite(actions).all():
                raise ValueError("Actions must be finite")
            association = (
                self.h3_trace[0]
                if self.epoch == 0
                else project_association(actions)
            )
            feedback_scores = actions[:, NUM_USERS:]

        previous_association = self.association
        self.association = association.copy()
        active = association.T
        link_toggles = (
            0
            if self.epoch == 0
            else int(np.count_nonzero(association != previous_association))
        )
        serving_set_changes = (
            0
            if self.epoch == 0
            else int(np.count_nonzero(np.any(
                association != previous_association, axis=-1
            )))
        )
        start = self.epoch * ASSOCIATION_PERIOD_FRAMES
        stop = start + ASSOCIATION_PERIOD_FRAMES
        stored_frames = []
        update_frames = []
        age_frames = []
        nmse_frames = []
        effective_frames = []

        for frame in range(start, stop):
            if frame == 0:
                updates = np.zeros_like(active)
                effective = np.zeros_like(self.ages, dtype=np.float64)
            else:
                if self.baseline:
                    updates, effective = self._baseline_feedback(active)
                else:
                    updates, effective = project_feedback(
                        feedback_scores,
                        active,
                        self.ages,
                        self.previous_updates,
                    )
                self.ages += 1
                self.stored[updates] = self.episode.true_channels[frame][updates]
                self.ages[updates] = 0
                self.previous_updates = updates.copy()
                self.update_history = (
                    HISTORY_DISCOUNT * self.update_history
                    + (1 - HISTORY_DISCOUNT) * updates
                )

            stored_frames.append(self.stored.copy())
            update_frames.append(updates.copy())
            age_frames.append(self.ages.copy())
            effective_frames.append(effective.copy())
            error = np.abs(self.episode.true_channels[frame] - self.stored) ** 2
            signal = np.abs(self.episode.true_channels[frame]) ** 2
            numerator = (error * active[..., None]).sum()
            denominator = (signal * active[..., None]).sum()
            nmse_frames.append(numerator / denominator if denominator else 0.0)

        stored_frames = np.asarray(stored_frames)
        masks = np.broadcast_to(
            association, (ASSOCIATION_PERIOD_FRAMES, NUM_USERS, NUM_AP)
        ).copy()
        with torch.inference_mode():
            weights = make_beamformer(
                stored_frames,
                masks,
                self.beamformer,
                pmax_w=self.pmax_w,
                noise_power=self.noise_power,
                device=self.device,
                model=self.model,
                batched_gnn=True,
            )
            rates = calculate_rates(
                weights,
                self.episode.true_channels[start:stop],
                NUM_AP,
                self.device,
                self.noise_power,
            ).cpu().numpy()
        finite, max_unassociated, max_power = beamformer_checks(
            weights, masks, NUM_USERS, self.pmax_w
        )
        mean_sum_rate = float(rates.sum(axis=-1).mean())
        switching_cost = LAMBDA_SWITCH * link_toggles / (NUM_AP * NUM_USERS)
        reward = mean_sum_rate / NUM_USERS - switching_cost
        update_frames = np.asarray(update_frames)
        age_frames = np.asarray(age_frames)
        steady_updates = update_frames[1:] if start == 0 else update_frames
        counts = steady_updates.sum(axis=-1)
        loads = np.broadcast_to(active.sum(axis=-1), counts.shape)
        constraints = bool(
            finite
            and np.isfinite(rates).all()
            and np.isfinite(reward)
            and max_unassociated == 0
            and max_power <= self.pmax_w + 1e-6
            and np.all(association.sum(axis=-1) == TOP_L)
            and not np.any(update_frames & ~active[None])
            and np.all(counts == np.minimum(FEEDBACK_BUDGET, loads))
        )

        done = self.epoch == self.num_epochs - 1
        self.epoch += 1
        next_observation = (
            np.zeros((NUM_AP, LOCAL_OBSERVATION_DIM), dtype=np.float32)
            if done
            else self._observation()
        )
        return next_observation, reward, done, {
            "sum_rates": rates.sum(axis=-1),
            "user_rates": rates,
            "link_toggles": link_toggles,
            "serving_set_changes": serving_set_changes,
            "association": association.copy(),
            "updates": update_frames,
            "ages": age_frames,
            "nmse": np.asarray(nmse_frames),
            "effective_feedback_scores": np.asarray(effective_frames),
            "stored_channels": stored_frames,
            "constraints_passed": constraints,
            "max_unassociated_abs": max_unassociated,
            "max_ap_power_w": max_power,
        }


def evaluate_policy(
    actor,
    episodes,
    beamformer,
    *,
    model,
    device,
    pmax_w,
    noise_power,
    baseline=False,
):
    trajectory = {
        name: []
        for name in (
            "utility",
            "sum_rate",
            "p05_user_rate",
            "link_toggles",
            "link_toggles_per_ue_s",
            "serving_set_changes",
            "serving_set_changes_per_ue_s",
            "switching_cost",
            "mean_age",
            "p95_age",
            "max_age",
            "never_refreshed_fraction",
            "nmse",
            "updates_per_ap_frame",
            "association_bid_messages",
            "association_bid_messages_per_ue_s",
            "max_unassociated_abs",
            "max_ap_power_w",
        )
    }
    raw_actions = []
    associations = []
    updates = []
    constraints = []
    stored_digest = hashlib.sha256()
    true_digest = hashlib.sha256()
    for episode in episodes:
        true_digest.update(episode.true_channels.tobytes())
        environment = JointControlEnvironment(
            episode,
            beamformer,
            model=model,
            device=device,
            pmax_w=pmax_w,
            noise_power=noise_power,
            baseline=baseline,
        )
        observation = environment.reset()
        rewards = []
        rates = []
        age_values = []
        nmse_values = []
        update_values = []
        toggle_count = 0
        serving_set_change_count = 0
        maximum_unassociated = 0.0
        maximum_power = 0.0
        done = False
        while not done:
            if baseline:
                action = np.zeros((NUM_AP, LOCAL_ACTION_DIM), dtype=np.float32)
            else:
                tensor = torch.as_tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.inference_mode():
                    sampled, _ = actor.sample(tensor, deterministic=True)
                action = sampled.squeeze(0).cpu().numpy()
            observation, reward, done, info = environment.step(action)
            raw_actions.append(action)
            associations.append(info["association"])
            updates.append(info["updates"])
            constraints.append(info["constraints_passed"])
            rewards.append(reward)
            rates.append(info["user_rates"])
            age_values.append(info["ages"])
            nmse_values.append(info["nmse"])
            update_values.append(info["updates"])
            toggle_count += info["link_toggles"]
            serving_set_change_count += info["serving_set_changes"]
            maximum_unassociated = max(
                maximum_unassociated, info["max_unassociated_abs"]
            )
            maximum_power = max(maximum_power, info["max_ap_power_w"])
            stored_digest.update(info["stored_channels"].tobytes())
        rates = np.concatenate(rates)
        ages = np.concatenate(age_values)
        active = np.repeat(
            np.asarray(associations[-environment.num_epochs:]).transpose(0, 2, 1)[:, None],
            ASSOCIATION_PERIOD_FRAMES,
            axis=1,
        ).reshape(-1, NUM_AP, NUM_USERS)
        active_ages = ages.reshape(-1, NUM_AP, NUM_USERS)[active]
        update_array = np.concatenate(update_values)
        ever_active = active.any(axis=0)
        ever_updated = update_array.astype(bool).any(axis=0)
        duration_s = len(rates) * 0.001
        trajectory["utility"].append(np.mean(rewards))
        trajectory["sum_rate"].append(rates.sum(axis=-1).mean())
        trajectory["p05_user_rate"].append(np.percentile(rates.mean(axis=0), 5))
        trajectory["link_toggles"].append(toggle_count)
        trajectory["link_toggles_per_ue_s"].append(
            toggle_count / (NUM_USERS * duration_s)
        )
        trajectory["serving_set_changes"].append(serving_set_change_count)
        trajectory["serving_set_changes_per_ue_s"].append(
            serving_set_change_count / (NUM_USERS * duration_s)
        )
        trajectory["switching_cost"].append(
            LAMBDA_SWITCH * toggle_count
            / (environment.num_epochs * NUM_AP * NUM_USERS)
        )
        trajectory["mean_age"].append(active_ages.mean())
        trajectory["p95_age"].append(np.percentile(active_ages, 95))
        trajectory["max_age"].append(active_ages.max())
        trajectory["never_refreshed_fraction"].append(
            np.count_nonzero(ever_active & ~ever_updated)
            / np.count_nonzero(ever_active)
        )
        trajectory["nmse"].append(np.concatenate(nmse_values).mean())
        trajectory["updates_per_ap_frame"].append(
            update_array[1:].sum()
            / ((len(update_array) - 1) * NUM_AP)
        )
        bid_messages = environment.num_epochs * NUM_AP * NUM_USERS
        trajectory["association_bid_messages"].append(bid_messages)
        trajectory["association_bid_messages_per_ue_s"].append(
            bid_messages / (NUM_USERS * duration_s)
        )
        trajectory["max_unassociated_abs"].append(maximum_unassociated)
        trajectory["max_ap_power_w"].append(maximum_power)
    return {
        "per_trajectory": {
            name: np.asarray(values, dtype=np.float64)
            for name, values in trajectory.items()
        },
        "raw_actions": np.asarray(raw_actions),
        "associations": np.asarray(associations),
        "updates": np.asarray(updates),
        "constraints_passed": bool(all(constraints)),
        "input_hashes": {
            "true_csi_sha256": true_digest.hexdigest(),
            "stored_csi_sha256": stored_digest.hexdigest(),
            "raw_actions_sha256": hashlib.sha256(
                np.asarray(raw_actions).tobytes()
            ).hexdigest(),
            "association_sha256": hashlib.sha256(
                np.asarray(associations).tobytes()
            ).hexdigest(),
            "updates_sha256": hashlib.sha256(
                np.asarray(updates).tobytes()
            ).hexdigest(),
        },
    }
