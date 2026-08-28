import tempfile
from pathlib import Path

import numpy as np
import torch

from association import TOP_L
from environment import MOBILITY_STRAIGHT
from model_association import AssociationActor
from evaluate_association_checkpoints import actor_association_trace
from train_association import (
    ACTION_DIM,
    NUM_AP,
    NUM_USERS,
    TRAINING_SETTINGS,
    AssociationDecisionEnvironment,
    AssociationEpisode,
    ReplayBuffer,
    SACTrainer,
    _setting_counts,
    action_rank_diagnostics,
    fit_observation_statistics,
    observation_dimension,
    top2_plateau_at_gate,
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PMAX_W = 10 ** ((15 - 30) / 10)


def assert_straight_only_training_contract():
    assert TRAINING_SETTINGS == (
        (MOBILITY_STRAIGHT, 30.0),
        (MOBILITY_STRAIGHT, 80.0),
    )
    assert _setting_counts(256) == [128, 128]
    assert _setting_counts(16) == [8, 8]


def assert_action_rank_diagnostics():
    first = np.tile([0.9, 0.8, 0.2, 0.1, 0.0], (NUM_USERS, 1)).ravel()
    same_mask = np.tile([0.8, 0.7, 0.3, 0.2, 0.1], (NUM_USERS, 1)).ravel()
    changed_mask = np.tile([0.8, 0.2, 0.7, 0.1, 0.0], (NUM_USERS, 1)).ravel()
    initial = action_rank_diagnostics(first[None])
    assert np.isclose(initial["mean_top2_margin"], 0.6)
    assert initial["serving_set_change_fraction_from_previous"] is None
    unchanged = action_rank_diagnostics(same_mask[None], first[None])
    assert unchanged["mean_abs_action_change_from_previous"] > 0
    assert unchanged["serving_set_change_fraction_from_previous"] == 0
    changed = action_rank_diagnostics(changed_mask[None], first[None])
    assert changed["serving_set_change_fraction_from_previous"] == 1


def assert_top2_plateau_gate():
    history = [
        {
            "mean_utility": 1.0,
            "mean_abs_action_change_from_previous": None,
            "serving_set_change_fraction_from_previous": None,
        },
        {
            "mean_utility": 1.0,
            "mean_abs_action_change_from_previous": 0.01,
            "serving_set_change_fraction_from_previous": 0.0,
        },
    ]
    assert not top2_plateau_at_gate(history, 90_000, 100_000)
    assert top2_plateau_at_gate(history, 100_000, 100_000)
    history[-1]["serving_set_change_fraction_from_previous"] = 0.125
    assert not top2_plateau_at_gate(history, 100_000, 100_000)


def synthetic_episode(seed=0):
    rng = np.random.default_rng(seed)
    lsf_power = rng.lognormal(
        mean=-2, sigma=0.5, size=(3, NUM_USERS, NUM_AP)
    )
    channels = (
        rng.standard_normal((2, 2, NUM_AP, NUM_USERS, 2))
        + 1j * rng.standard_normal((2, 2, NUM_AP, NUM_USERS, 2))
    ) / np.sqrt(2)
    return AssociationEpisode(lsf_power, channels)


def assert_observation_and_action_contract():
    episodes = [synthetic_episode(0), synthetic_episode(1)]
    statistics = fit_observation_statistics(episodes)
    assert observation_dimension("current") == 85
    assert observation_dimension("history") == 125
    for variant in ("current", "history"):
        environment = AssociationDecisionEnvironment(
            episodes[0],
            statistics,
            variant=variant,
            lambda_switch=0.5,
            pmax_w=PMAX_W,
            noise_power=1e-12,
            device=DEVICE,
        )
        observation = environment.reset()
        assert observation.shape == (observation_dimension(variant),)
        next_observation, reward, done, info = environment.step(
            np.zeros(ACTION_DIM, dtype=np.float32)
        )
        assert np.all(info["mask"].sum(axis=-1) == TOP_L)
        assert np.isfinite(reward)
        assert not done
        assert np.isfinite(next_observation).all()


def assert_observation_is_channel_causal():
    first = synthetic_episode(2)
    perturbed = AssociationEpisode(
        first.lsf_power.copy(), first.reward_channels * (3 + 4j)
    )
    statistics = fit_observation_statistics((first,))
    observations = []
    for episode in (first, perturbed):
        environment = AssociationDecisionEnvironment(
            episode,
            statistics,
            variant="history",
            lambda_switch=0.5,
            pmax_w=PMAX_W,
            noise_power=1e-12,
            device=DEVICE,
        )
        observations.append(environment.reset())
    assert np.array_equal(*observations)


def assert_sac_update_and_reload():
    torch.manual_seed(3)
    observation_dim = observation_dimension("current")
    trainer = SACTrainer(observation_dim, ACTION_DIM, DEVICE)
    replay = ReplayBuffer(16, observation_dim, ACTION_DIM, seed=3)
    rng = np.random.default_rng(3)
    for index in range(4):
        observation = rng.standard_normal(observation_dim).astype(np.float32)
        action = rng.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        next_observation = rng.standard_normal(observation_dim).astype(np.float32)
        replay.add(observation, action, float(index), next_observation, False)
    metrics = trainer.update(replay, 4)
    assert all(np.isfinite(value) for value in metrics.values())

    observation = rng.standard_normal(observation_dim).astype(np.float32)
    expected = trainer.act(observation, deterministic=True)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "actor.pt"
        torch.save(trainer.actor.state_dict(), checkpoint)
        reloaded = AssociationActor(observation_dim, ACTION_DIM).to(DEVICE)
        reloaded.load_state_dict(
            torch.load(checkpoint, map_location=DEVICE, weights_only=True)
        )
        tensor = torch.as_tensor(observation, device=DEVICE).unsqueeze(0)
        with torch.inference_mode():
            actual, _ = reloaded.sample(tensor, deterministic=True)
        assert np.array_equal(expected, actual.squeeze(0).cpu().numpy())


def assert_batched_checkpoint_trace_matches_environment():
    episodes = [synthetic_episode(4), synthetic_episode(5)]
    statistics = fit_observation_statistics(episodes)
    for variant in ("current", "history"):
        torch.manual_seed(6)
        actor = AssociationActor(
            observation_dimension(variant), ACTION_DIM
        ).to(DEVICE)
        lsf_power = np.stack([episode.lsf_power for episode in episodes])
        trace, actions = actor_association_trace(
            actor,
            lsf_power,
            statistics,
            variant=variant,
            device=DEVICE,
        )
        assert trace.shape == (2, 3, NUM_USERS, NUM_AP)
        assert actions.shape == (2, 2, ACTION_DIM)
        assert np.all(trace.sum(axis=-1) == TOP_L)
        for trajectory, episode in enumerate(episodes):
            environment = AssociationDecisionEnvironment(
                episode,
                statistics,
                variant=variant,
                lambda_switch=0.5,
                pmax_w=PMAX_W,
                noise_power=1e-12,
                device=DEVICE,
            )
            environment.reset()
            assert np.array_equal(trace[trajectory, 0], environment.previous_mask)
            for epoch, action in enumerate(actions[trajectory], start=1):
                _, _, _, info = environment.step(action)
                assert np.array_equal(trace[trajectory, epoch], info["mask"])


def main():
    assert_straight_only_training_contract()
    assert_action_rank_diagnostics()
    assert_top2_plateau_gate()
    assert_observation_and_action_contract()
    assert_observation_is_channel_causal()
    assert_sac_update_and_reload()
    assert_batched_checkpoint_trace_matches_environment()
    print("Stage 3B observation, SAC-update, and checkpoint checks passed.")


if __name__ == "__main__":
    main()
