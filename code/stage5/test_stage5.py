import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from association import hysteresis_top_l_mask, top_l_mask
from controller import H3, build_modular_trace, expand_trace
from environment import MobilityEnvironment
from evaluate import evaluation_device, simulate_cell
from feedback import FeedbackState


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
AP_COORDINATES = np.loadtxt(STAGE1C_RUN / "arrays/BS_0.txt")


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


def assert_stage5a_only_scope():
    combined = "\n".join(
        (SOURCE / filename).read_text()
        for filename in ("association.py", "feedback.py", "controller.py", "evaluate.py")
    )
    assert "torch.optim" not in combined
    assert not any(
        (SOURCE / filename).exists()
        for filename in ("model_joint.py", "train_joint.py", "run_stage5b.sh")
    )


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
    assert_stage5a_only_scope()
    assert_parent_native_boundary_devices()
    print("Stage 5 dynamic CSI, causality, locality, boundary, and constraint checks passed.")


if __name__ == "__main__":
    main()
