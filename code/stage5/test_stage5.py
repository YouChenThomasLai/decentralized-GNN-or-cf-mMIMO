import hashlib
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from association import hysteresis_top_l_mask, top_l_mask
from controller import H3, build_modular_trace, expand_trace
from environment import MobilityEnvironment
from evaluate import (
    BEAMFORMERS,
    EXPECTED_STAGE1C_CHECKPOINT_SHA256,
    EXPECTED_STAGE1C_MODEL_SHA256,
    beamformer_checks,
    evaluation_device,
    format_gnn_inputs,
    load_frozen_model,
    make_beamformer,
    simulate_cell,
)
from feedback import FeedbackState
from rl_core import (
    LOCAL_ACTION_DIM,
    LOCAL_OBSERVATION_DIM,
    NUM_AP,
    JointControlEnvironment,
    JointEpisode,
    LocalActor,
    ReplayBuffer,
    SACTrainer,
    evaluate_policy,
)


NUM_USERS = 8
SOURCE = Path(__file__).resolve().parent
DEFAULT_STAGE1C_RUN = next(
    (
        path
        for path in (
            SOURCE.parent / "results_stage1c_bpp_noise_1e-12/M2_K8_P15.0/run0",
            SOURCE.parent
            / "stage1/remote_backup_2026-08-24/results_stage1c_bpp_noise_1e-12/M2_K8_P15.0/run0",
        )
        if path.is_dir()
    ),
    SOURCE.parent / "results_stage1c_bpp_noise_1e-12/M2_K8_P15.0/run0",
)
STAGE1C_RUN = Path(os.environ.get("STAGE1C_RUN_DIR", DEFAULT_STAGE1C_RUN))
CHECKPOINT = STAGE1C_RUN / "models/model_final_run0.pt"
AP_COORDINATES = np.loadtxt(STAGE1C_RUN / "arrays/BS_0.txt")
PMAX_W = 10 ** ((15 - 30) / 10)


def assert_frozen_stage4_sources():
    stage4 = SOURCE.parent / "stage4"
    if not (stage4 / "environment.py").is_file():
        stage4 = (
            stage4
            / "results_stage4_seed0/straight_0_kmh/source_snapshot"
        )
    for filename in ("environment.py", "utils_return_indivial_rates.py"):
        assert (SOURCE / filename).read_bytes() == (stage4 / filename).read_bytes()


def assert_top2_h3_and_ties():
    scores = np.asarray([[[4.0, 4.0, 1.0, 0.5, 0.25]]])
    initial = top_l_mask(scores)
    assert np.array_equal(np.flatnonzero(initial[0, 0]), (0, 1))
    unchanged = hysteresis_top_l_mask(
        np.asarray([[[4.0, 4.0, 7.9, 0.5, 0.25]]]), initial, 3.0
    )
    assert np.array_equal(unchanged, initial)
    changed = hysteresis_top_l_mask(
        np.asarray([[[4.0, 4.0, 8.1, 0.5, 0.25]]]), initial, 3.0
    )
    assert np.array_equal(np.flatnonzero(changed[0, 0]), (1, 2))


def assert_all_link_bootstrap_drop_rejoin_and_budget():
    initial = np.arange(6, dtype=np.float64).reshape(1, 2, 3, 1).astype(complex)
    association0 = np.asarray(
        [[[True, False], [True, False], [False, True]]], dtype=bool
    )
    state = FeedbackState(initial, association0, budget=1, scheduler="round_robin")
    assert np.array_equal(state.stored_channels, initial)
    assert np.all(state.ages == 0)

    association1 = np.asarray(
        [[[True, False], [False, True], [True, False]]], dtype=bool
    )
    current1 = initial + 100 + 10j
    updates1 = state.step(current1, association1)
    assert np.all(updates1.sum(axis=-1) <= 1)
    assert not updates1[0, 0, 2]
    assert state.stored_channels[0, 0, 2, 0] == initial[0, 0, 2, 0]
    assert state.ages[0, 0, 2] == 1
    assert state.ages[0, 0, 1] == 1

    association2 = np.asarray(
        [[[False, True], [True, False], [True, False]]], dtype=bool
    )
    previous = state.stored_channels.copy()
    updates2 = state.step(initial + 200 + 20j, association2)
    assert not updates2[0, 0, 1]
    assert state.stored_channels[0, 0, 1, 0] == previous[0, 0, 1, 0]
    assert state.ages[0, 0, 1] == 2
    assert np.all(state.ages >= 0)


def assert_priority_and_no_true_csi_leakage():
    initial = np.ones((1, 1, 3, 1), dtype=complex)
    association = np.ones((1, 3, 1), dtype=bool)
    power = np.asarray([[[1.0, 4.0, 2.0]]])
    rhos = np.asarray([[0.5, 0.99, 0.2]])
    states = [
        FeedbackState(
            initial,
            association,
            budget=1,
            scheduler="mobility_age_priority",
            t0_link_power=power,
            rhos=rhos,
        )
        for _ in range(2)
    ]
    first = [state.select_updates() for state in states]
    assert np.array_equal(first[0], first[1])
    current = initial + 10 + 3j
    perturbed = current.copy()
    perturbed[~first[0]] = 1e12 + 1e12j
    states[0].apply_updates(current, first[0])
    states[1].apply_updates(perturbed, first[1])
    assert np.array_equal(states[0].select_updates(), states[1].select_updates())


def assert_ap_local_bids_and_ue_arbitration_boundary():
    path_loss = np.ones((1, 2, 5, 1), dtype=np.float64)
    path_loss[0, :, :, 0] = np.sqrt([10.0, 9.0, 1.0, 0.5, 0.25])
    changed = path_loss.copy()
    changed[0, 1, 2, 0] = np.sqrt(30.0)
    _, bids_a, masks_a = build_modular_trace(
        path_loss, H3, association_period_frames=1
    )
    _, bids_b, masks_b = build_modular_trace(
        changed, H3, association_period_frames=1
    )
    assert np.array_equal(bids_a[..., 0], bids_b[..., 0])
    assert not np.array_equal(masks_a[:, 1], masks_b[:, 1])
    assert np.all(masks_a.sum(axis=-1) == 2)
    assert np.all(masks_b.sum(axis=-1) == 2)
    expanded = expand_trace(masks_b, 2, association_period_frames=1)
    assert np.array_equal(expanded, masks_b)


def assert_zero_speed_joint_evaluator():
    loader = MobilityEnvironment(
        2,
        1,
        episode_steps=51,
        speed_kmh=0,
        seed=17,
        bs_locations=AP_COORDINATES,
    ).generate_trajectories(NUM_USERS, 0.1)
    _, _, trace = build_modular_trace(loader.path_loss_factors, H3)
    args = SimpleNamespace(
        K=NUM_USERS,
        episode_steps=51,
        eval_time_stride=1,
        pmax_dbm=15.0,
        noise_power=1e-12,
        batch_size=64,
    )
    base = {
        "association": H3,
        "scheduler": "round_robin",
        "role": "test",
    }
    hold = simulate_cell(
        loader, trace, {**base, "label": "hold", "budget": 0}, args, torch.device("cpu")
    )
    full = simulate_cell(
        loader, trace, {**base, "label": "full", "budget": 8}, args, torch.device("cpu")
    )
    assert hold["summary"]["constraints_passed"]
    assert full["summary"]["constraints_passed"]
    assert hold["stored_sha256"] == full["stored_sha256"]
    assert np.allclose(
        hold["rate"]["trajectory_sum_rates"],
        full["rate"]["trajectory_sum_rates"],
        rtol=1e-6,
        atol=1e-8,
    )


def assert_frozen_gnn_adapter():
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == (
        EXPECTED_STAGE1C_CHECKPOINT_SHA256
    )
    assert hashlib.sha256((SOURCE / "model_2.py").read_bytes()).hexdigest() == (
        EXPECTED_STAGE1C_MODEL_SHA256
    )
    model = load_frozen_model(
        CHECKPOINT,
        antennas=2,
        pmax_w=PMAX_W,
        device=torch.device("cpu"),
    )
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())

    loader = MobilityEnvironment(
        2,
        2,
        episode_steps=2,
        speed_kmh=30,
        seed=23,
        bs_locations=AP_COORDINATES,
    ).generate_trajectories(NUM_USERS, 0.1)
    scores = np.square(loader.path_loss_factors[:, 0]).transpose(0, 2, 1)
    mask = top_l_mask(scores)
    stored = loader.true_channels[:, 0].copy()
    actual = format_gnn_inputs(stored, mask, torch.device("cpu"))
    expected = loader._format_frames(stored, mask)
    assert torch.equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert all(torch.equal(a, b) for a, b in zip(actual[2], expected[2]))
    assert all(np.array_equal(a, b) for a, b in zip(actual[3], expected[3]))

    perturbed = stored.copy()
    perturbed[~mask.transpose(0, 2, 1)] = 1e12 + 1e12j
    with torch.inference_mode():
        for beamformer in BEAMFORMERS[1:]:
            weights = make_beamformer(
                stored,
                mask,
                beamformer,
                pmax_w=PMAX_W,
                noise_power=1e-12,
                device=torch.device("cpu"),
                model=model,
            )
            batched = make_beamformer(
                stored,
                mask,
                beamformer,
                pmax_w=PMAX_W,
                noise_power=1e-12,
                device=torch.device("cpu"),
                model=model,
                batched_gnn=True,
            )
            reference = model(
                actual[0] if beamformer == "centralized_gnn" else actual[2],
                actual[1] if beamformer == "centralized_gnn" else actual[3],
                training=beamformer == "centralized_gnn",
                duplicate=False,
            )
            one_at_a_time = torch.cat(
                [
                    make_beamformer(
                        stored[index:index + 1],
                        mask[index:index + 1],
                        beamformer,
                        pmax_w=PMAX_W,
                        noise_power=1e-12,
                        device=torch.device("cpu"),
                        model=model,
                    )
                    for index in range(len(stored))
                ]
            )
            hidden_perturbed = make_beamformer(
                perturbed,
                mask,
                beamformer,
                pmax_w=PMAX_W,
                noise_power=1e-12,
                device=torch.device("cpu"),
                model=model,
            )
            assert torch.equal(weights, reference)
            assert torch.equal(weights, one_at_a_time)
            assert torch.allclose(weights, batched, rtol=1e-6, atol=5e-7)
            assert torch.equal(weights, hidden_perturbed)
            finite, max_unassociated, max_power = beamformer_checks(
                weights, mask, NUM_USERS, PMAX_W
            )
            assert finite
            assert max_unassociated == 0
            assert max_power <= PMAX_W + 1e-6


def assert_frozen_gnn_stored_csi_smoke():
    loader = MobilityEnvironment(
        2,
        2,
        episode_steps=100,
        speed_kmh=30,
        seed=29,
        bs_locations=AP_COORDINATES,
    ).generate_trajectories(NUM_USERS, 0.1)
    _, _, trace = build_modular_trace(loader.path_loss_factors, H3)
    args = SimpleNamespace(
        K=NUM_USERS,
        episode_steps=100,
        eval_time_stride=25,
        pmax_dbm=15.0,
        noise_power=1e-12,
        batch_size=8,
    )
    model = load_frozen_model(
        CHECKPOINT,
        antennas=2,
        pmax_w=PMAX_W,
        device=torch.device("cpu"),
    )
    result = simulate_cell(
        loader,
        trace,
        {
            "association": H3,
            "scheduler": "mobility_age_priority",
            "budget": 2,
            "role": "gate5_5_smoke",
            "label": "gate5_5_smoke",
        },
        args,
        torch.device("cpu"),
        model=model,
        beamformers=BEAMFORMERS,
    )
    assert result["summary"]["constraints_passed"]
    assert set(result["rates"]) == set(BEAMFORMERS)
    assert np.all(result["updates"][:, 1:].sum(axis=-1) <= 2)
    for rate in result["rates"].values():
        assert rate["finite"]
        assert np.isfinite(rate["trajectory_sum_rates"]).all()


def assert_gate5_5_scope():
    combined = "\n".join(
        (SOURCE / filename).read_text()
        for filename in (
            "association.py",
            "feedback.py",
            "controller.py",
            "model_2.py",
            "evaluate.py",
        )
    )
    assert "torch.optim" not in combined
    assert (SOURCE / "model_2.py").is_file()
    assert (SOURCE / "run_stage5b.sh").is_file()
    assert not any(
        (SOURCE / filename).exists()
        for filename in ("model_joint.py", "train_joint.py")
    )


def rl_episode(seed=31):
    loader = MobilityEnvironment(
        2,
        1,
        episode_steps=100,
        speed_kmh=30,
        seed=seed,
        bs_locations=AP_COORDINATES,
    ).generate_trajectories(NUM_USERS, 0.1)
    return JointEpisode(
        true_channels=loader.true_channels[0].copy(),
        lsf_power=np.square(loader.path_loss_factors[0]),
        rhos=loader.rhos[0].copy(),
        speed_kmh=30,
        seed=seed,
    )


def assert_ap_local_actor_and_fixed_action_contract():
    torch.manual_seed(37)
    actor = LocalActor().eval()
    observations = torch.zeros(1, NUM_AP, LOCAL_OBSERVATION_DIM)
    changed = observations.clone()
    changed[:, 2] = 1
    with torch.inference_mode():
        actions, _ = actor.sample(observations, deterministic=True)
        changed_actions, _ = actor.sample(changed, deterministic=True)
    assert actions.shape == (1, NUM_AP, LOCAL_ACTION_DIM)
    assert torch.equal(actions[:, :2], changed_actions[:, :2])
    assert not torch.equal(actions[:, 2], changed_actions[:, 2])
    assert torch.equal(actions[:, 3:], changed_actions[:, 3:])
    assert not hasattr(actor, "actors")


def assert_rl_no_leakage_projection_and_constraints():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    episode = rl_episode()
    first = JointControlEnvironment(
        episode,
        "rzf",
        model=None,
        device=device,
        pmax_w=PMAX_W,
        noise_power=1e-12,
    )
    observation = first.reset()
    actor = LocalActor().to(device).eval()
    with torch.inference_mode():
        actions, _ = actor.sample(
            torch.from_numpy(observation).to(device).unsqueeze(0),
            deterministic=True,
        )
    actions = actions[0].cpu().numpy()
    next_observation, _, _, info = first.step(actions)
    assert info["constraints_passed"]
    assert np.all(info["association"].sum(axis=-1) == 2)
    assert np.all(info["updates"][1:].sum(axis=-1) <= 2)

    hidden = deepcopy(episode)
    hidden.true_channels[1:50][~info["updates"][1:]] = 1e12 + 1e12j
    second = JointControlEnvironment(
        hidden,
        "rzf",
        model=None,
        device=device,
        pmax_w=PMAX_W,
        noise_power=1e-12,
    )
    second_observation = second.reset()
    second_next, _, _, second_info = second.step(actions)
    assert np.array_equal(observation, second_observation)
    assert np.array_equal(info["association"], second_info["association"])
    assert np.array_equal(info["updates"], second_info["updates"])
    assert np.array_equal(next_observation, second_next)

    with torch.inference_mode():
        next_actions, _ = actor.sample(
            torch.from_numpy(next_observation).to(device).unsqueeze(0),
            deterministic=True,
        )
    _, _, done, next_info = first.step(next_actions[0].cpu().numpy())
    assert done
    assert next_info["constraints_passed"]
    assert np.all(next_info["association"].sum(axis=-1) == 2)
    evaluation = evaluate_policy(
        actor,
        [episode],
        "rzf",
        model=None,
        device=device,
        pmax_w=PMAX_W,
        noise_power=1e-12,
    )
    metrics = evaluation["per_trajectory"]
    assert metrics["association_bid_messages"][0] == 2 * NUM_AP * NUM_USERS
    assert metrics["association_bid_messages_per_ue_s"][0] == 100


def assert_replay_twin_critic_and_finite_update():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(41)
    replay = ReplayBuffer(16, 41)
    for _ in range(8):
        observation = rng.normal(
            size=(NUM_AP, LOCAL_OBSERVATION_DIM)
        ).astype(np.float32)
        action = rng.uniform(
            -1, 1, size=(NUM_AP, LOCAL_ACTION_DIM)
        ).astype(np.float32)
        replay.add(observation, action, 1.0, observation * 0.9, False)
    trainer = SACTrainer(device)
    metrics = trainer.update(replay, 8)
    assert all(np.isfinite(value) for value in metrics.values())
    assert trainer.critic.q1 is not trainer.critic.q2


def assert_minimal_rl_scope():
    combined = "\n".join(
        (SOURCE / filename).read_text()
        for filename in ("rl_core.py", "train_rl.py", "evaluate_rl.py")
    ).lower()
    assert "lstm" not in combined
    assert "gru" not in combined
    assert "graphconv" not in combined
    assert "torch_geometric" not in combined


def assert_parent_native_boundary_devices():
    stage4 = {"role": "stage4_boundary"}
    stage3 = {"role": "stage3_boundary"}
    assert evaluation_device(stage4, "cuda:0", "cpu") == torch.device("cpu")
    assert evaluation_device(stage3, "cuda:0", "cpu") == torch.device("cuda:0")


def main():
    assert_frozen_stage4_sources()
    assert_top2_h3_and_ties()
    assert_all_link_bootstrap_drop_rejoin_and_budget()
    assert_priority_and_no_true_csi_leakage()
    assert_ap_local_bids_and_ue_arbitration_boundary()
    assert_zero_speed_joint_evaluator()
    assert_frozen_gnn_adapter()
    assert_frozen_gnn_stored_csi_smoke()
    assert_gate5_5_scope()
    assert_ap_local_actor_and_fixed_action_contract()
    assert_rl_no_leakage_projection_and_constraints()
    assert_replay_twin_critic_and_finite_update()
    assert_minimal_rl_scope()
    assert_parent_native_boundary_devices()
    print("Stage 5 dynamic CSI, causality, locality, boundary, and constraint checks passed.")


if __name__ == "__main__":
    main()
