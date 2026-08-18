import numpy as np
import torch

from data import MyDataLoader
from model_2 import node_update
from utils_return_indivial_rates import mrt_beamforming, rzf_beamforming


BATCH_SIZE = 2
NUM_AP = 5
NUM_ANTENNAS = 2
NUM_USERS = 3
PMAX = 10 ** ((15 - 30) / 10)
DEVICE = torch.device("cpu")


def build_snapshot(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader = MyDataLoader(NUM_ANTENNAS, BATCH_SIZE)
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
            channels_before, mask_before, PMAX, DEVICE
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


def main():
    first = build_snapshot(0)
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
        outputs = first["loader"].compute_loss(beamformer, DEVICE)
        assert all(torch.isfinite(output).all() for output in outputs)
        assert outputs[1] > 0

    second = build_snapshot(0)
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


if __name__ == "__main__":
    main()
