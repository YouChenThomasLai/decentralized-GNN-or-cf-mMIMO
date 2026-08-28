import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import MyDataLoader
from environment import (
    MOBILITY_HOTSPOT,
    MOBILITY_PHASE_DWELL,
    MOBILITY_PHASE_TRANSIT,
    MobilityEnvironment,
    SnapshotEnvironment,
    hotspot_process_diagnostics,
    symmetric_transition_matrix,
)
from evaluate import METHODS, TrajectoryEvaluator, load_frozen_model
from model_2 import node_update
from trainer_2 import seed_everything
from utils_return_indivial_rates import (
    DIRECT_CHANNEL_FADING,
    DIRECT_CHANNEL_SCALE,
    DIRECT_PATH_LOSS_EXPONENT,
    HEIGHT_DIFFERENCE,
    SQUARE_SIDE,
    cal_loss,
    generate_channel,
    independent_channel_diagnostics,
    jakes_correlation,
    mrt_beamforming,
    rzf_beamforming,
    wrapped_3d_distance,
    wrapped_displacement,
    wrapped_horizontal_distance,
)


NUM_AP = 5
NUM_ANTENNAS = 2
NUM_USERS = 3
BATCH_SIZE = 2
PMAX = 10 ** ((15 - 30) / 10)
DEVICE = torch.device("cpu")
DEFAULT_STAGE1C_RUN = (
    Path(__file__).resolve().parents[1]
    / "stage1/remote_backup_2026-08-24/results_stage1c_bpp_noise_1e-12/"
    / "M2_K8_P15.0/run0"
)
STAGE1C_RUN = Path(os.environ.get("STAGE1C_RUN_DIR", DEFAULT_STAGE1C_RUN))
CHECKPOINT = STAGE1C_RUN / "models/model_final_run0.pt"
AP_COORDINATES_PATH = STAGE1C_RUN / "arrays/BS_0.txt"
AP_COORDINATES = np.loadtxt(AP_COORDINATES_PATH)


def assert_mask_and_power(weights, association_mask, num_users, pmax=PMAX):
    flat_mask = association_mask.transpose(0, 2, 1).reshape(
        len(association_mask), NUM_AP * num_users
    )
    mask = torch.from_numpy(flat_mask).unsqueeze(1).expand_as(weights)
    assert torch.count_nonzero(weights[~mask]) == 0
    for ap in range(NUM_AP):
        start = ap * num_users
        stop = (ap + 1) * num_users
        power = weights[:, :, start:stop].square().sum(dim=(1, 2))
        assert torch.all(power <= pmax + 1e-6)


def assert_copied_stage1_snapshot_path():
    seed_everything(0)
    loader = SnapshotEnvironment(
        NUM_ANTENNAS, BATCH_SIZE, AP_COORDINATES
    )
    central, central_mask = loader.gen_training_data(NUM_USERS, 0.1)
    local, local_masks = loader.gen_testing_data(
        NUM_USERS, 0.1, regenerate_channels=False
    )
    channels = loader.get_stacked_channels()
    association_mask = loader.get_association_mask()
    model = node_update(NUM_ANTENNAS, 6, PMAX, 64, NUM_AP, DEVICE)
    model.eval()
    with torch.inference_mode():
        beamformers = (
            model(central, central_mask, training=True),
            model(local, local_masks, training=False),
            mrt_beamforming(channels, association_mask, PMAX, DEVICE),
            rzf_beamforming(
                channels, association_mask, PMAX, DEVICE, noise_power=1e-12
            ),
        )
    for weights in beamformers:
        assert_mask_and_power(weights, association_mask, NUM_USERS)
        assert all(
            torch.isfinite(value).all()
            for value in loader.compute_loss(weights, DEVICE, 1e-12)
        )


def assert_stage1_t0_compatibility():
    np.random.seed(11)
    initial_positions = np.asarray(
        ((10.0, 5.0), (-25.0, 8.0), (3.0, -40.0))
    )
    reference = SnapshotEnvironment(
        NUM_ANTENNAS, BATCH_SIZE, AP_COORDINATES
    )
    stage1_channels = np.stack(
        [
            generate_channel(
                NUM_ANTENNAS,
                NUM_USERS,
                BATCH_SIZE,
                location,
                initial_positions,
            )
            for location in reference.BS_Loc_array
        ],
        axis=1,
    )
    distances = wrapped_3d_distance(
        reference.BS_Loc_array[None, :, None, :],
        initial_positions[None, None, :, :],
    )
    path_loss = (
        DIRECT_CHANNEL_FADING
        * distances ** (-DIRECT_PATH_LOSS_EXPONENT)
        / 10 ** DIRECT_CHANNEL_SCALE
    )
    initial_normalized = stage1_channels / path_loss[..., None]
    loader = MobilityEnvironment(
        NUM_ANTENNAS,
        BATCH_SIZE,
        episode_steps=8,
        speed_kmh=0,
        seed=12,
        bs_locations=AP_COORDINATES,
    ).generate_trajectories(
        NUM_USERS,
        0.1,
        initial_positions=initial_positions,
        initial_normalized_channels=initial_normalized,
    )
    assert np.allclose(
        loader.true_channels[:, 0], stage1_channels, atol=1e-12, rtol=1e-12
    )
    assert np.array_equal(
        loader.true_channels,
        np.broadcast_to(loader.true_channels[:, :1], loader.true_channels.shape),
    )
    rssi = np.sum(np.abs(stage1_channels) ** 2, axis=-1)
    stage1_mask = (
        rssi >= np.max(rssi, axis=1, keepdims=True) * 0.1
    ).transpose(0, 2, 1)
    assert np.array_equal(loader.association_mask, stage1_mask)

    ap_features = []
    for ap in range(NUM_AP):
        feature = np.concatenate(
            (stage1_channels[:, ap].real, stage1_channels[:, ap].imag), axis=-1
        )
        feature = (feature * stage1_mask[:, :, ap, None]).astype(np.float32)
        ap_features.append(
            F.normalize(torch.from_numpy(feature).unsqueeze(1), dim=2)
        )
    stage1_feature = torch.cat(ap_features, dim=2)
    stage2_feature = loader.get_frames(np.arange(BATCH_SIZE), 0)[0]
    assert torch.allclose(stage1_feature, stage2_feature, atol=1e-6, rtol=1e-6)

    weights = mrt_beamforming(stage1_channels, stage1_mask, PMAX, DEVICE)
    stage1_rate = cal_loss(weights, stage1_channels, NUM_AP, DEVICE, 1e-12)
    stage2_rate = loader.compute_loss(
        weights,
        DEVICE,
        1e-12,
        np.arange(BATCH_SIZE),
        np.zeros(BATCH_SIZE, dtype=int),
    )
    for expected, actual in zip(stage1_rate, stage2_rate):
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)


def build_straight(seed, speed_kmh=80):
    return MobilityEnvironment(
        1,
        3,
        episode_steps=80,
        speed_kmh=speed_kmh,
        seed=seed,
        bs_locations=AP_COORDINATES,
    ).generate_trajectories(2, 0.1)


def assert_straight_kinematics_and_reproducibility():
    first = build_straight(7)
    second = build_straight(7)
    different = build_straight(8)
    for name in (
        "ue_positions",
        "ue_directions_rad",
        "normalized_channels",
        "true_channels",
        "association_mask",
    ):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert not np.array_equal(first.ue_positions, different.ue_positions)
    assert not np.array_equal(first.true_channels, different.true_channels)

    steps = wrapped_horizontal_distance(
        first.ue_positions[:, :-1], first.ue_positions[:, 1:]
    )
    expected = first.ue_speeds_mps[:, None] * first.decision_period_s
    assert np.max(np.abs(steps - expected)) < 1e-10
    assert np.max(np.abs(first.ue_positions)) < SQUARE_SIDE / 2
    assert first.distances.min() >= HEIGHT_DIFFERENCE
    expected_path_loss = (
        DIRECT_CHANNEL_FADING
        * first.distances ** (-DIRECT_PATH_LOSS_EXPONENT)
        / 10 ** DIRECT_CHANNEL_SCALE
    )
    assert np.array_equal(first.path_loss_factors, expected_path_loss)
    assert np.allclose(
        first.true_channels,
        first.normalized_channels * first.path_loss_factors[..., None],
    )
    assert not hasattr(first, "stored_channels")

    assert np.array_equal(
        wrapped_displacement((99.0, 0.0), (-99.0, 0.0)),
        np.asarray((2.0, 0.0)),
    )
    assert np.isclose(
        wrapped_3d_distance((99.0, 0.0), (-99.0, 0.0)),
        np.sqrt(2.0**2 + HEIGHT_DIFFERENCE**2),
    )


def assert_channel_contract():
    expected = np.asarray((1.0, 0.999485, 0.949178, 0.666090))
    actual = jakes_correlation(np.asarray((0, 3, 30, 80)) / 3.6)
    assert np.allclose(actual, expected, atol=1e-6, rtol=0)
    for seed, speed_kmh in enumerate((0, 3, 30, 80)):
        diagnostics = independent_channel_diagnostics(
            speed_kmh / 3.6, sample_count=20000, seed=100 + seed
        )
        assert diagnostics["finite"]
        assert abs(float(diagnostics["real_mean"])) <= 0.02
        assert abs(float(diagnostics["imag_mean"])) <= 0.02
        assert abs(float(diagnostics["real_variance"]) - 0.5) <= 0.02
        assert abs(float(diagnostics["imag_variance"]) - 0.5) <= 0.02
        assert float(diagnostics["max_ar1_error"]) <= 0.02
        assert np.array_equal(
            diagnostics["ar1_correlation"],
            diagnostics["rho"] ** diagnostics["lags"],
        )


def assert_hotspot_contract():
    loader = MobilityEnvironment(
        1,
        4,
        episode_steps=2000,
        speed_kmh=30,
        seed=0,
        mobility_model=MOBILITY_HOTSPOT,
        bs_locations=AP_COORDINATES,
    ).generate_trajectories(2, 0.1)
    assert loader.hotspot_state.shape == (4, 2000, 2)
    assert loader.mobility_phase.shape == (4, 2000, 2)
    assert loader.mobility_sample_phase.shape == (4, 2000, 2)
    assert set(np.unique(loader.mobility_phase)) == {
        MOBILITY_PHASE_DWELL,
        MOBILITY_PHASE_TRANSIT,
    }
    assert np.array_equal(
        loader.mobility_phase == MOBILITY_PHASE_TRANSIT,
        loader.instantaneous_speeds_mps > 0,
    )
    assert loader.hotspot_burn_in_s >= 120
    clip_duration = (loader.episode_steps - 1) * loader.decision_period_s
    assert np.all(loader.clip_start_times_s >= loader.hotspot_burn_in_s)
    assert np.all(
        loader.clip_start_times_s + clip_duration
        <= loader.hotspot_burn_in_s + loader.hotspot_trace_duration_s + 1e-12
    )
    diagnostics = loader.mobility_diagnostics()
    assert diagnostics["max_abs_coordinate_m"] < SQUARE_SIDE / 2
    assert diagnostics["min_ap_ue_distance_m"] >= HEIGHT_DIFFERENCE
    assert diagnostics["max_step_limit_violation_m"] <= 1e-10
    assert diagnostics["dwell_channel_max_abs_change"] == 0
    assert np.allclose(
        diagnostics["empirical_lag1_by_phase"],
        diagnostics["ar1_lag1_by_phase"],
        atol=0.02,
    )

    variants = ((0.2, 2.0, 2.5), (0.6, 5.0, 12.5), (0.8, 10.0, 50.0))
    for stickiness, dwell_mean, residence_target in variants:
        matrix = symmetric_transition_matrix(4, stickiness)
        process = hotspot_process_diagnostics(
            matrix,
            dwell_mean,
            2.0,
            30 / 3.6,
            seed=200,
            minimum_events=10000,
            minimum_outgoing_per_state=2000,
        )
        assert process["event_count"] >= 10000
        assert process["minimum_outgoing_event_count"] >= 2000
        assert np.max(
            np.abs(process["empirical_transition_matrix"] - matrix)
        ) <= 0.05
        assert np.max(np.abs(process["event_chain_occupancy"] - 0.25)) <= 0.03
        assert abs(float(process["event_dwell_mean_s"]) / dwell_mean - 1) <= 0.05
        assert np.isclose(
            process["merged_residence_mean_target_s"], residence_target
        )
        assert abs(
            float(process["merged_residence_mean_s"]) / residence_target - 1
        ) <= 0.05
        assert process["self_transition_motion_distance_max_m"] == 0
        assert process["max_abs_coordinate_m"] <= 45 + 1e-12
        assert process["max_step_limit_violation_m"] <= 1e-10
        assert process["transit_endpoint_error_max_m"] <= 1e-10
        assert np.isclose(
            process["natural_time_dwell_fraction"]
            + process["natural_time_transit_fraction"],
            1,
        )


def assert_frozen_evaluator_contract():
    assert CHECKPOINT.is_file()
    seed_everything(17)
    loader = MobilityEnvironment(
        2,
        2,
        episode_steps=8,
        speed_kmh=30,
        seed=17,
        bs_locations=AP_COORDINATES,
    ).generate_trajectories(8, 0.1)
    model = node_update(2, 6, PMAX, 64, NUM_AP, DEVICE)
    load_frozen_model(model, CHECKPOINT, DEVICE)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    evaluator = TrajectoryEvaluator(
        model,
        K=8,
        pmax_w=PMAX,
        num_ap=NUM_AP,
        device=DEVICE,
        noise_power=1e-12,
        frame_batch_size=4,
        time_stride=2,
    )
    summary, raw = evaluator.evaluate(loader)
    assert np.array_equal(raw["evaluation_time_indices"], (0, 2, 4, 6))
    for method in METHODS:
        assert np.isfinite(summary[method])
        assert np.isfinite(summary[f"{method}_p05_user_rate"])
        assert raw[f"{method}_constraints_passed"]
        assert raw[f"{method}_per_trajectory_user_time_average_rates"].shape == (
            2,
            8,
        )
    assert raw["shared_current_channel"]
    assert raw["shared_fixed_association_mask"]


def assert_evaluation_only_source():
    source_dir = Path(__file__).resolve().parent
    assert (source_dir / "model_2.py").read_bytes() == (
        source_dir.parent / "stage1/model_2.py"
    ).read_bytes()
    source = "\n".join(
        (source_dir / filename).read_text()
        for filename in ("trainer_2.py", "evaluate.py")
    )
    assert "torch.optim" not in source
    assert "SummaryWriter" not in source
    assert not (source_dir / "train.py").exists()


def main():
    assert MyDataLoader is MobilityEnvironment
    assert_copied_stage1_snapshot_path()
    assert_stage1_t0_compatibility()
    assert_straight_kinematics_and_reproducibility()
    assert_channel_contract()
    assert_hotspot_contract()
    assert_frozen_evaluator_contract()
    assert_evaluation_only_source()
    print("Stage 2 mobility and frozen-evaluation checks passed.")


if __name__ == "__main__":
    main()
