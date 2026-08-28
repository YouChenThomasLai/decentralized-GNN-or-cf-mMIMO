import os
from pathlib import Path

import numpy as np
import torch

from association import (
    POLICIES,
    PRIMARY_TOP2_POLICIES,
    build_association_traces,
    hysteresis_top_l_mask,
    switching_metrics,
    threshold_mask,
    top_l_mask,
)
from environment import MobilityEnvironment
from evaluate import BEAMFORMERS, DynamicAssociationEvaluator, load_frozen_model


NUM_AP = 5
NUM_USERS = 8
PMAX = 10 ** ((15 - 30) / 10)
DEVICE = torch.device("cpu")
DEFAULT_STAGE1C_RUN = (
    Path(__file__).resolve().parents[1]
    / "stage1/remote_backup_2026-08-24/results_stage1c_bpp_noise_1e-12"
    / "M2_K8_P15.0/run0"
)
STAGE1C_RUN = Path(os.environ.get("STAGE1C_RUN_DIR", DEFAULT_STAGE1C_RUN))
CHECKPOINT = STAGE1C_RUN / "models/model_final_run0.pt"
AP_COORDINATES = np.loadtxt(STAGE1C_RUN / "arrays/BS_0.txt")


def assert_frozen_stage2_sources():
    source = Path(__file__).resolve().parent
    stage2 = source.parent / "stage2"
    for filename in (
        "data.py",
        "environment.py",
        "model_2.py",
        "utils_return_indivial_rates.py",
    ):
        assert (source / filename).read_bytes() == (stage2 / filename).read_bytes()


def assert_selection_rules():
    scores = np.asarray([[[5.0, 5.0, 5.0, 4.0, 3.0]]])
    selected = top_l_mask(scores)
    assert np.array_equal(np.flatnonzero(selected[0, 0]), (0, 1))
    assert np.array_equal(
        threshold_mask(np.asarray([[[10.0, 1.0, 0.99, 0.0, 0.0]]])),
        np.asarray([[[True, True, False, False, False]]]),
    )

    previous = np.asarray([[[True, True, False, False, False]]])
    replacement_scores = np.asarray([[[10.0, 9.0, 20.0, 19.0, 1.0]]])
    h3 = hysteresis_top_l_mask(replacement_scores, previous, 3)
    h6 = hysteresis_top_l_mask(replacement_scores, previous, 6)
    assert np.array_equal(np.flatnonzero(h3[0, 0]), (0, 2))
    assert np.array_equal(h6, previous)
    assert np.array_equal(
        hysteresis_top_l_mask(replacement_scores, previous, 0),
        top_l_mask(replacement_scores),
    )


def assert_switching_semantics():
    trace = np.asarray(
        [[[[True, True, False, False, False]],
          [[False, True, True, False, False]]]]
    )
    metrics = switching_metrics(trace, num_frames=100, frame_period_s=0.001)
    assert np.array_equal(metrics["link_toggles_by_epoch"], [[2]])
    assert np.array_equal(metrics["serving_set_replacements_by_epoch"], [[1]])
    assert np.array_equal(metrics["users_changed_by_epoch"], [[1]])
    assert np.array_equal(metrics["link_toggles_per_ue_s"], [20])
    assert np.array_equal(metrics["serving_set_changes_per_ue_s"], [10])


def synthetic_inputs(time_steps=100):
    path_loss = np.ones((1, time_steps, NUM_AP, NUM_USERS))
    path_loss[:, :, 2:] = 0.1
    channels = np.ones(
        (1, time_steps, NUM_AP, NUM_USERS, 2), dtype=np.complex128
    )
    return path_loss, channels


def assert_constraints_determinism_and_causality():
    path_loss, channels = synthetic_inputs()
    first_indices, first = build_association_traces(
        path_loss, channels, seed=7
    )
    second_indices, second = build_association_traces(
        path_loss, channels, seed=7
    )
    assert np.array_equal(first_indices, (0, 50))
    assert np.array_equal(first_indices, second_indices)
    assert tuple(first) == POLICIES
    for policy in POLICIES:
        assert np.array_equal(first[policy], second[policy])
    for policy in PRIMARY_TOP2_POLICIES:
        assert np.all(first[policy].sum(axis=-1) == 2)

    future_path_loss = path_loss.copy()
    future_channels = channels.copy()
    future_path_loss[:, 51:] *= 100
    future_channels[:, 51:] *= 3 + 4j
    _, perturbed = build_association_traces(
        future_path_loss, future_channels, seed=7
    )
    for policy in POLICIES:
        assert np.array_equal(first[policy], perturbed[policy])

    moving_path_loss = path_loss.copy()
    moving_path_loss[:, 50:, :2] = 0.1
    moving_path_loss[:, 50:, 2:4] = 2.0
    _, moving = build_association_traces(moving_path_loss, channels, seed=7)
    assert np.any(moving["current_lsf_top2"][:, 1] != moving["current_lsf_top2"][:, 0])


def assert_zero_speed_gate():
    loader = MobilityEnvironment(
        2,
        2,
        episode_steps=100,
        speed_kmh=0,
        seed=11,
        bs_locations=AP_COORDINATES,
    ).generate_trajectories(NUM_USERS, 0.1)
    _, traces = build_association_traces(
        loader.path_loss_factors, loader.true_channels, seed=11
    )
    for policy in (
        "fixed_lsf_top2_t0",
        "current_lsf_top2",
        "hysteresis_lsf_top2_h0_db",
        "hysteresis_lsf_top2_h3_db",
        "hysteresis_lsf_top2_h6_db",
    ):
        assert not np.any(traces[policy][:, 1:] != traces[policy][:, :-1])


def assert_paired_evaluator_contract():
    assert CHECKPOINT.is_file()
    loader = MobilityEnvironment(
        2,
        1,
        episode_steps=4,
        speed_kmh=30,
        seed=17,
        bs_locations=AP_COORDINATES,
    ).generate_trajectories(NUM_USERS, 0.1)
    _, all_traces = build_association_traces(
        loader.path_loss_factors,
        loader.true_channels,
        seed=17,
        association_period_frames=2,
    )
    traces = {
        "fixed_lsf_top2_t0": all_traces["fixed_lsf_top2_t0"],
        "current_lsf_top2": all_traces["current_lsf_top2"],
    }
    model = load_frozen_model(
        CHECKPOINT, antennas=2, pmax_w=PMAX, device=DEVICE
    )
    evaluator = DynamicAssociationEvaluator(
        model,
        num_users=NUM_USERS,
        pmax_w=PMAX,
        num_ap=NUM_AP,
        device=DEVICE,
        noise_power=1e-12,
        frame_batch_size=2,
        time_stride=1,
        association_period_frames=2,
    )
    summary, raw = evaluator.evaluate(loader, traces)
    assert raw["shared_current_channel"]
    for policy in traces:
        for method in BEAMFORMERS:
            key = f"{policy}__{method}"
            assert raw[f"{key}__constraints_passed"]
            assert np.isfinite(
                summary[policy][method]["trajectory_average_sum_rate"]
            )


def assert_stage3a_source_is_evaluation_only():
    source = Path(__file__).resolve().parent
    combined = "\n".join(
        (source / filename).read_text()
        for filename in ("association.py", "evaluate.py")
    )
    assert "torch.optim" not in combined


def main():
    assert_frozen_stage2_sources()
    assert_selection_rules()
    assert_switching_semantics()
    assert_constraints_determinism_and_causality()
    assert_zero_speed_gate()
    assert_paired_evaluator_contract()
    assert_stage3a_source_is_evaluation_only()
    print("Stage 3A association, causality, and paired-evaluation checks passed.")


if __name__ == "__main__":
    main()
