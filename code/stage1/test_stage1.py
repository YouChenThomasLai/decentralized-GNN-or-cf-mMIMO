import numpy as np
import torch

from data import MyDataLoader
from environment import SnapshotEnvironment, TOPOLOGY_TYPE, WRAP_AROUND
from model_2 import node_update
from trainer_2 import Trainer, seed_everything
from utils_return_indivial_rates import (
    DIRECT_CHANNEL_FADING,
    DIRECT_CHANNEL_SCALE,
    DIRECT_PATH_LOSS_EXPONENT,
    HEIGHT_DIFFERENCE,
    SQUARE_SIDE,
    mrt_beamforming,
    rzf_beamforming,
    sample_square_bpp,
    wrapped_3d_distance,
    wrapped_displacement,
    wrapped_horizontal_distance,
)


BATCH_SIZE = 2
NUM_AP = 5
NUM_ANTENNAS = 2
NUM_USERS = 8
PMAX = 10 ** ((15 - 30) / 10)
DEVICE = torch.device("cpu")
STAGE1C_NOISE_POWER = 1e-12
EXPECTED_EVAL_METHODS = {
    "centralized_gnn",
    "decentralized_gnn",
    "mrt",
    "rzf",
}


def build_snapshot(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader = SnapshotEnvironment(NUM_ANTENNAS, BATCH_SIZE)
    centralized_feature, centralized_index = loader.gen_training_data(
        NUM_USERS, 0.1
    )
    channels_before = loader.get_stacked_channels().copy()
    mask_before = loader.get_association_mask().copy()

    decentralized_feature, decentralized_index = loader.gen_testing_data(
        NUM_USERS, 0.1, regenerate_channels=False
    )
    model = node_update(
        NUM_ANTENNAS, 6, PMAX, 64, NUM_AP, DEVICE
    ).to(DEVICE)
    model.eval()
    with torch.no_grad():
        centralized_w = model(
            centralized_feature, centralized_index, training=True
        )
        decentralized_w = model(
            decentralized_feature, decentralized_index, training=False
        )
        mrt_w = mrt_beamforming(
            channels_before, mask_before, PMAX, DEVICE
        )
        rzf_w = rzf_beamforming(
            channels_before,
            mask_before,
            PMAX,
            DEVICE,
            STAGE1C_NOISE_POWER,
        )

    assert np.array_equal(channels_before, loader.get_stacked_channels())
    assert np.array_equal(mask_before, loader.get_association_mask())
    return {
        "loader": loader,
        "channels": channels_before,
        "mask": mask_before,
        "centralized_feature": centralized_feature,
        "decentralized_feature": decentralized_feature,
        "beamformers": {
            "centralized_gnn": centralized_w,
            "decentralized_gnn": decentralized_w,
            "mrt": mrt_w,
            "rzf": rzf_w,
        },
    }


def assert_mask_and_power(beamformer, association_mask):
    flat_mask = association_mask.transpose(0, 2, 1).reshape(
        BATCH_SIZE, NUM_AP * NUM_USERS
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


def assert_evaluator_contract():
    seed_everything(17)
    trainer = Trainer(
        M=NUM_ANTENNAS,
        K=NUM_USERS,
        batch_size=BATCH_SIZE,
        n_iter=1,
        pmax_dbm=15,
        noise_power=1e-12,
        device="cpu",
    )
    metrics = trainer.eval(4)
    assert metrics.keys() == EXPECTED_EVAL_METHODS
    assert all(np.isfinite(value) and value > 0 for value in metrics.values())
    return metrics


def assert_topology_contract():
    left = np.array([99.0, 0.0])
    right = np.array([-99.0, 0.0])
    assert np.array_equal(
        wrapped_displacement(left, right), np.array([2.0, 0.0])
    )
    assert wrapped_horizontal_distance(left, right) == 2.0
    assert np.isclose(
        wrapped_3d_distance(left, right),
        np.sqrt(2.0**2 + HEIGHT_DIFFERENCE**2),
    )
    assert wrapped_3d_distance(left, left) == HEIGHT_DIFFERENCE

    first = sample_square_bpp(5, rng=np.random.default_rng(11))
    second = sample_square_bpp(5, rng=np.random.default_rng(11))
    assert np.array_equal(first, second)
    assert np.all((-SQUARE_SIDE / 2 <= first) & (first < SQUARE_SIDE / 2))


def assert_straight_trajectory_association_signal():
    topology_rng = np.random.RandomState(0)
    ap_locations = sample_square_bpp(5, rng=topology_rng)
    trajectory_rng = np.random.RandomState(10_000)
    initial = sample_square_bpp(
        10 * NUM_USERS, rng=trajectory_rng
    ).reshape(10, NUM_USERS, 2)
    angles = trajectory_rng.uniform(
        0, 2 * np.pi, size=(10, NUM_USERS)
    )
    velocity = (30 / 3.6) * np.stack(
        (np.cos(angles), np.sin(angles)), axis=-1
    )
    times = np.arange(2000) * 0.001
    positions = initial[:, None] + times[None, :, None, None] * velocity[:, None]
    positions = wrapped_displacement(np.zeros(2), positions)
    distance_3d = wrapped_3d_distance(
        positions[:, :, :, None, :], ap_locations
    )
    amplitude = (
        DIRECT_CHANNEL_FADING
        * distance_3d ** (-DIRECT_PATH_LOSS_EXPONENT)
        / 10**DIRECT_CHANNEL_SCALE
    )
    received_power = amplitude**2
    current_lsf_threshold = received_power >= received_power.max(
        axis=3, keepdims=True
    ) * 0.1
    fixed_threshold_t0 = current_lsf_threshold[:, :1]
    mismatch_fraction = np.mean(
        current_lsf_threshold != fixed_threshold_t0
    )
    switching_epochs = np.count_nonzero(
        np.any(
            current_lsf_threshold[:, 1:] != current_lsf_threshold[:, :-1],
            axis=3,
        )
    )
    assert np.isfinite(received_power).all()
    assert current_lsf_threshold.any(axis=3).all()
    assert mismatch_fraction > 0
    assert switching_epochs > 0
    return mismatch_fraction, switching_epochs


def main():
    assert MyDataLoader is SnapshotEnvironment
    assert_topology_contract()
    first = build_snapshot(0)
    assert first["loader"].topology_type == TOPOLOGY_TYPE
    assert first["loader"].wrap_around is WRAP_AROUND
    assert first["loader"].square_side == SQUARE_SIDE
    assert first["loader"].height_difference == HEIGHT_DIFFERENCE
    assert np.all(
        (-SQUARE_SIDE / 2 <= first["loader"].BS_Loc_array)
        & (first["loader"].BS_Loc_array < SQUARE_SIDE / 2)
    )
    assert np.all(
        (-SQUARE_SIDE / 2 <= first["loader"].user_loc)
        & (first["loader"].user_loc < SQUARE_SIDE / 2)
    )
    distances = wrapped_3d_distance(
        first["loader"].user_loc[:, None, :],
        first["loader"].BS_Loc_array,
    )
    assert np.all(distances >= HEIGHT_DIFFERENCE)
    expected_w_shape = (
        BATCH_SIZE,
        2 * NUM_ANTENNAS,
        NUM_AP * NUM_USERS,
    )
    assert first["channels"].shape == (
        BATCH_SIZE,
        NUM_AP,
        NUM_USERS,
        NUM_ANTENNAS,
    )
    assert np.iscomplexobj(first["channels"])
    assert first["mask"].shape == (BATCH_SIZE, NUM_USERS, NUM_AP)
    assert first["mask"].dtype == np.bool_
    assert first["centralized_feature"].shape == (
        BATCH_SIZE,
        1,
        NUM_AP * NUM_USERS,
        2 * NUM_ANTENNAS,
    )
    for feature in first["decentralized_feature"]:
        assert feature.shape == (
            BATCH_SIZE,
            1,
            NUM_USERS,
            2 * NUM_ANTENNAS,
        )

    for beamformer in first["beamformers"].values():
        assert isinstance(beamformer, torch.Tensor)
        assert beamformer.shape == expected_w_shape
        assert_mask_and_power(beamformer, first["mask"])
        outputs = first["loader"].compute_loss(
            beamformer, DEVICE, STAGE1C_NOISE_POWER
        )
        assert all(torch.isfinite(output).all() for output in outputs)
        assert outputs[1] > 0

    second = build_snapshot(0)
    assert np.array_equal(
        first["loader"].BS_Loc_array, second["loader"].BS_Loc_array
    )
    assert np.array_equal(first["channels"], second["channels"])
    assert np.array_equal(first["mask"], second["mask"])
    assert torch.allclose(
        first["centralized_feature"],
        second["centralized_feature"],
        atol=1e-6,
        rtol=0,
    )
    for method in first["beamformers"]:
        assert torch.allclose(
            first["beamformers"][method],
            second["beamformers"][method],
            atol=1e-6,
            rtol=0,
        )

    different = build_snapshot(1)
    assert not np.array_equal(
        first["loader"].BS_Loc_array, different["loader"].BS_Loc_array
    )

    metrics = assert_evaluator_contract()
    mismatch_fraction, switching_epochs = (
        assert_straight_trajectory_association_signal()
    )
    print(
        "Stage 1C snapshot/evaluator checks passed; "
        f"MRT={metrics['mrt']:.6g}, RZF={metrics['rzf']:.6g}, "
        f"straight-trajectory association mismatch={mismatch_fraction:.6g}, "
        f"switching epochs={switching_epochs}."
    )


if __name__ == "__main__":
    main()
