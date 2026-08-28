import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from environment import MobilityEnvironment
from evaluate import build_cells, simulate_cell
from feedback import SCHEDULERS, FeedbackState


NUM_USERS = 8
DEFAULT_STAGE1C_RUN = (
    Path(__file__).resolve().parents[1]
    / "stage1/remote_backup_2026-08-24/results_stage1c_bpp_noise_1e-12"
    / "M2_K8_P15.0/run0"
)
STAGE1C_RUN = Path(os.environ.get("STAGE1C_RUN_DIR", DEFAULT_STAGE1C_RUN))
AP_COORDINATES = np.loadtxt(STAGE1C_RUN / "arrays/BS_0.txt")


def toy_inputs():
    initial = np.arange(8, dtype=np.float64).reshape(1, 2, 4, 1).astype(complex)
    association = np.asarray(
        [[[True, False], [False, False], [True, False], [True, False]]]
    )
    return initial, association


def assert_frozen_stage2_sources():
    source = Path(__file__).resolve().parent
    stage2 = source.parent / "stage2"
    for filename in ("environment.py", "utils_return_indivial_rates.py"):
        assert (source / filename).read_bytes() == (stage2 / filename).read_bytes()


def assert_round_robin_and_zero_load():
    initial, association = toy_inputs()
    state = FeedbackState(
        initial, association, budget=1, scheduler="round_robin"
    )
    expected = (0, 2, 3, 0)
    for period, user in enumerate(expected, start=1):
        updates = state.select_updates()
        assert np.array_equal(np.flatnonzero(updates[0, 0]), (user,))
        assert not updates[0, 1].any()
        current = initial + period * (10 + 1j)
        state.apply_updates(current, updates)
        assert np.array_equal(state.stored_channels[updates], current[updates])
        assert np.all(state.ages[~state.active] == -1)


def assert_hold_and_full_boundaries():
    initial, association = toy_inputs()
    hold = FeedbackState(initial, association, budget=0, scheduler="round_robin")
    for period in range(1, 4):
        updates = hold.step(initial + period)
        assert not updates.any()
        assert np.all(hold.ages[hold.active] == period)
        assert np.all(hold.ages[~hold.active] == -1)

    full = FeedbackState(initial, association, budget=4, scheduler="round_robin")
    for period in range(1, 4):
        current = initial + period * (2 + 3j)
        updates = full.step(current)
        assert np.array_equal(updates, full.active)
        assert np.all(full.ages[full.active] == 0)
        assert np.array_equal(
            full.stored_channels[full.active], current[full.active]
        )


def assert_causality_determinism_and_nested_random():
    initial, association = toy_inputs()
    states = [
        FeedbackState(
            initial,
            association,
            budget=budget,
            scheduler="random",
            scheduler_seed=17,
        )
        for budget in (1, 2, 3)
    ]
    perturbed = FeedbackState(
        initial,
        association,
        budget=1,
        scheduler="random",
        scheduler_seed=17,
    )
    first = states[0].select_updates()
    assert np.array_equal(first, perturbed.select_updates())
    current = initial.copy()
    current[states[0].active] = 1e9 + 1e9j
    states[0].apply_updates(current, first)
    perturbed.apply_updates(-current, first)
    assert np.array_equal(
        states[0].select_updates(), perturbed.select_updates()
    )

    states = [
        FeedbackState(
            initial,
            association,
            budget=budget,
            scheduler="random",
            scheduler_seed=23,
        )
        for budget in (1, 2, 3)
    ]
    for period in range(4):
        masks = [state.select_updates() for state in states]
        assert np.all(masks[0] <= masks[1])
        assert np.all(masks[1] <= masks[2])
        for state, mask in zip(states, masks):
            state.apply_updates(initial + period, mask)


def assert_mobility_age_priority():
    initial = np.ones((1, 1, 3, 1), dtype=complex)
    association = np.ones((1, 3, 1), dtype=bool)
    power = np.asarray([[[1.0, 4.0, 2.0]]])
    rhos = np.asarray([[0.5, 0.99, 0.2]])
    state = FeedbackState(
        initial,
        association,
        budget=1,
        scheduler="mobility_age_priority",
        t0_link_power=power,
        rhos=rhos,
    )
    expected_scores = power * (1 - rhos[:, None] ** 2)
    updates = state.select_updates()
    assert np.array_equal(state.last_priority, expected_scores)
    assert np.array_equal(np.flatnonzero(updates[0, 0]), (2,))

    tied = FeedbackState(
        initial,
        association,
        budget=1,
        scheduler="mobility_age_priority",
        t0_link_power=power,
        rhos=np.ones((1, 3)),
    )
    assert np.array_equal(np.flatnonzero(tied.select_updates()[0, 0]), (0,))


def assert_cell_matrix_and_zero_speed_evaluator():
    cells = build_cells((0, 1, 2, 3, 8), SCHEDULERS, NUM_USERS)
    assert len(cells) == 11
    assert cells[0]["kind"] == "full_current"
    assert cells[1]["kind"] == "hold_t0"

    loader = MobilityEnvironment(
        2,
        1,
        episode_steps=4,
        speed_kmh=0,
        seed=11,
        bs_locations=AP_COORDINATES,
    ).generate_trajectories(NUM_USERS, 0.1)
    args = SimpleNamespace(
        K=NUM_USERS,
        scheduler_seed=29,
        episode_steps=4,
        eval_time_stride=1,
        pmax_dbm=15.0,
        noise_power=1e-12,
        batch_size=4,
    )
    full = simulate_cell(loader, cells[0], args, torch.device("cpu"))
    hold = simulate_cell(loader, cells[1], args, torch.device("cpu"))
    assert full["summary"]["constraints_passed"]
    assert hold["summary"]["constraints_passed"]
    assert full["stored_sha256"] == hold["stored_sha256"]
    assert full["rate"]["beamformer_sha256"] == hold["rate"]["beamformer_sha256"]
    assert np.allclose(
        full["rate"]["trajectory_sum_rates"],
        hold["rate"]["trajectory_sum_rates"],
        rtol=1e-6,
        atol=1e-8,
    )


def assert_evaluation_only_scope():
    source = Path(__file__).resolve().parent
    combined = "\n".join(
        (source / filename).read_text()
        for filename in ("feedback.py", "evaluate.py")
    )
    assert "torch.optim" not in combined
    assert not any((source / filename).exists() for filename in ("model_2.py", "trainer_2.py"))


def main():
    assert_frozen_stage2_sources()
    assert_round_robin_and_zero_load()
    assert_hold_and_full_boundaries()
    assert_causality_determinism_and_nested_random()
    assert_mobility_age_priority()
    assert_cell_matrix_and_zero_speed_evaluator()
    assert_evaluation_only_scope()
    print("Stage 4 feedback, causality, boundary, and determinism checks passed.")


if __name__ == "__main__":
    main()
