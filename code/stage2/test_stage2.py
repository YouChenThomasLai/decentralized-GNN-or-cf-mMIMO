import numpy as np
import torch
import torch.nn.functional as F

from data import MyDataLoader
from model_2 import node_update
from utils_return_indivial_rates import (
    DIRECT_CHANNEL_FADING,
    DIRECT_CHANNEL_SCALE,
    DIRECT_PATH_LOSS_EXPONENT,
    cal_loss,
    generate_channel,
    jakes_correlation,
    mrt_beamforming,
    rzf_beamforming,
)


BATCH_SIZE = 2
NUM_AP = 5
NUM_ANTENNAS = 2
NUM_USERS = 3
EPISODE_STEPS = 12
PMAX = 10 ** ((15 - 30) / 10)
DEVICE = torch.device("cpu")


def build_loader(seed, speed_kmh=30, batch_size=BATCH_SIZE, steps=EPISODE_STEPS):
    return MyDataLoader(
        NUM_ANTENNAS,
        batch_size,
        episode_steps=steps,
        speed_kmh=speed_kmh,
        seed=seed,
    ).generate_trajectories(NUM_USERS, 0.1)


def assert_mask_and_power(beamformer, association_mask):
    batch_size = association_mask.shape[0]
    flat_mask = association_mask.transpose(0, 2, 1).reshape(
        batch_size, NUM_AP * NUM_USERS
    )
    expanded_mask = torch.from_numpy(flat_mask).unsqueeze(1).expand_as(
        beamformer
    )
    assert torch.count_nonzero(beamformer[~expanded_mask]) == 0

    for ap in range(NUM_AP):
        start = ap * NUM_USERS
        stop = (ap + 1) * NUM_USERS
        power = beamformer[:, :, start:stop].square().sum(dim=(1, 2))
        assert torch.all(power <= PMAX + 1e-6)


def assert_data_contract_and_dynamics():
    loader = build_loader(0)
    assert loader.ue_positions.shape == (
        BATCH_SIZE,
        EPISODE_STEPS,
        NUM_USERS,
        2,
    )
    assert loader.ue_speeds_mps.shape == (BATCH_SIZE, NUM_USERS)
    assert loader.ue_directions_rad.shape == (BATCH_SIZE, NUM_USERS)
    assert loader.true_channels.shape == (
        BATCH_SIZE,
        EPISODE_STEPS,
        NUM_AP,
        NUM_USERS,
        NUM_ANTENNAS,
    )
    assert loader.association_mask.shape == (BATCH_SIZE, NUM_USERS, NUM_AP)
    assert loader.association_mask.dtype == np.bool_
    assert np.array_equal(loader.true_channels, loader.stored_channels)
    assert np.array_equal(
        loader.get_association_mask(expand_time=True),
        np.broadcast_to(
            loader.association_mask[:, None],
            (BATCH_SIZE, EPISODE_STEPS, NUM_USERS, NUM_AP),
        ),
    )

    steps = np.linalg.norm(np.diff(loader.ue_positions, axis=1), axis=-1)
    expected_steps = loader.ue_speeds_mps * loader.decision_period_s
    assert np.max(np.abs(steps - expected_steps[:, None])) < 1e-10
    assert np.max(np.linalg.norm(loader.ue_positions, axis=-1)) <= 100 + 1e-12

    expected_path_loss = (
        DIRECT_CHANNEL_FADING
        * loader.distances ** (-DIRECT_PATH_LOSS_EXPONENT)
        / 10 ** DIRECT_CHANNEL_SCALE
    )
    assert np.allclose(loader.path_loss_factors, expected_path_loss)
    assert np.allclose(
        loader.true_channels,
        loader.normalized_channels * loader.path_loss_factors[..., None],
    )

    centralized = loader.get_centralized_features()
    decentralized = loader.get_decentralized_features()
    assert centralized.shape == (
        BATCH_SIZE,
        EPISODE_STEPS,
        1,
        NUM_AP * NUM_USERS,
        2 * NUM_ANTENNAS,
    )
    assert len(decentralized) == NUM_AP
    assert all(
        feature.shape
        == (
            BATCH_SIZE,
            EPISODE_STEPS,
            1,
            NUM_USERS,
            2 * NUM_ANTENNAS,
        )
        for feature in decentralized
    )


def assert_reproducibility_and_stationarity():
    first = build_loader(7)
    second = build_loader(7)
    different = build_loader(8)
    for name in (
        "ue_positions",
        "ue_directions_rad",
        "true_channels",
        "association_mask",
    ):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert not np.array_equal(first.ue_positions, different.ue_positions)
    assert not np.array_equal(first.true_channels, different.true_channels)

    stationary = build_loader(9, speed_kmh=0)
    assert np.array_equal(
        stationary.ue_positions,
        np.broadcast_to(
            stationary.ue_positions[:, :1], stationary.ue_positions.shape
        ),
    )
    assert np.array_equal(
        stationary.true_channels,
        np.broadcast_to(
            stationary.true_channels[:, :1], stationary.true_channels.shape
        ),
    )


def assert_stage1_t0_compatibility():
    np.random.seed(11)
    initial_positions = np.array(
        [[10.0, 5.0], [-25.0, 8.0], [3.0, -40.0]]
    )
    reference = MyDataLoader(NUM_ANTENNAS, BATCH_SIZE, episode_steps=1)
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
    distances = np.linalg.norm(
        reference.BS_Loc_array[None, :, None, :]
        - initial_positions[None, None, :, :],
        axis=-1,
    )
    path_loss = (
        DIRECT_CHANNEL_FADING
        * distances ** (-DIRECT_PATH_LOSS_EXPONENT)
        / 10 ** DIRECT_CHANNEL_SCALE
    )
    initial_normalized = stage1_channels / path_loss[..., None]

    loader = MyDataLoader(
        NUM_ANTENNAS,
        BATCH_SIZE,
        episode_steps=EPISODE_STEPS,
        speed_kmh=0,
        seed=12,
    ).generate_trajectories(
        NUM_USERS,
        0.1,
        initial_positions=initial_positions,
        initial_normalized_channels=initial_normalized,
    )
    assert np.allclose(
        loader.true_channels[:, 0], stage1_channels, atol=1e-12, rtol=1e-12
    )

    rssi = np.sum(np.abs(stage1_channels) ** 2, axis=-1)
    stage1_mask = (
        rssi >= np.max(rssi, axis=1, keepdims=True) * 0.1
    ).transpose(0, 2, 1)
    assert np.array_equal(loader.association_mask, stage1_mask)

    stage1_ap_features = []
    for ap in range(NUM_AP):
        feature = np.concatenate(
            (stage1_channels[:, ap].real, stage1_channels[:, ap].imag),
            axis=-1,
        )
        feature = (feature * stage1_mask[:, :, ap, None]).astype(np.float32)
        stage1_ap_features.append(
            F.normalize(torch.from_numpy(feature).unsqueeze(1), dim=2)
        )
    stage1_feature = torch.cat(stage1_ap_features, dim=2)
    stage2_feature = loader.get_frames(np.arange(BATCH_SIZE), 0)[0]
    assert torch.allclose(stage1_feature, stage2_feature, atol=1e-6, rtol=1e-6)

    mrt = mrt_beamforming(stage1_channels, stage1_mask, PMAX, DEVICE)
    stage1_rate = cal_loss(mrt, stage1_channels, NUM_AP, DEVICE)
    stage2_rate = loader.compute_loss(
        mrt,
        DEVICE,
        trajectory_indices=np.arange(BATCH_SIZE),
        time_indices=np.zeros(BATCH_SIZE, dtype=int),
    )
    for expected, actual in zip(stage1_rate, stage2_rate):
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)


def assert_beamforming_contract():
    loader = build_loader(13)
    trajectory_indices = np.arange(BATCH_SIZE)
    time_indices = np.array([1, 4])
    central, central_mask, local, local_masks = loader.get_frames(
        trajectory_indices, time_indices
    )
    channels = loader.get_stacked_channels(trajectory_indices, time_indices)
    association_mask = loader.association_mask[trajectory_indices]
    model = node_update(
        NUM_ANTENNAS, 6, PMAX, 64, NUM_AP, DEVICE
    ).to(DEVICE)
    model.eval()
    with torch.no_grad():
        beamformers = (
            model(central, central_mask, training=True),
            model(local, local_masks, training=False),
            mrt_beamforming(channels, association_mask, PMAX, DEVICE),
            rzf_beamforming(
                channels, association_mask, PMAX, DEVICE, noise_power=1e-12
            ),
        )
    for beamformer in beamformers:
        assert beamformer.shape == (
            BATCH_SIZE,
            2 * NUM_ANTENNAS,
            NUM_AP * NUM_USERS,
        )
        assert_mask_and_power(beamformer, association_mask)
        outputs = loader.compute_loss(
            beamformer,
            DEVICE,
            1e-12,
            trajectory_indices,
            time_indices,
        )
        assert all(torch.isfinite(output).all() for output in outputs)


def assert_channel_statistics():
    expected = np.array([1.0, 0.999485, 0.949178, 0.666090])
    actual = jakes_correlation(np.array([0, 3, 30, 80]) / 3.6)
    assert np.allclose(actual, expected, atol=1e-6, rtol=0)

    loader = MyDataLoader(
        1,
        32,
        episode_steps=500,
        speed_kmh=30,
        seed=21,
    ).generate_trajectories(2, 0.1)
    diagnostics = loader.diagnostics(
        lags=(1, 2, 5, 10), bootstrap_samples=200
    )
    error = np.abs(
        diagnostics["empirical_correlation"]
        - diagnostics["theoretical_correlation"]
    )
    assert np.all(error <= 0.02)
    assert np.all(np.isfinite(diagnostics["bootstrap_95_ci"]))
    assert np.all(np.isfinite(loader.true_channels))


def main():
    assert_data_contract_and_dynamics()
    assert_reproducibility_and_stationarity()
    assert_stage1_t0_compatibility()
    assert_beamforming_contract()
    assert_channel_statistics()
    print("Stage 2 mobility checks passed.")


if __name__ == "__main__":
    main()
