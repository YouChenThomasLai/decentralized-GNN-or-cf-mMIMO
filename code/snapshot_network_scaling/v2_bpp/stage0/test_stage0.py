from pathlib import Path
import tempfile

import numpy as np
import torch

from data import MyDataLoader
from evaluate import evaluate_snapshot
from geometry import wrapped_displacement, wrapped_distance
from model_2 import node_update
from trainer_2 import seed_everything


BATCH_SIZE = 1
NUM_AP = 3
NUM_ANTENNAS = 1
NUM_RIS_ELEMENTS = 3
NUM_RIS = 2
NUM_USERS = 2
PMAX = 10 ** ((15 - 30) / 10)
SQUARE_SIDE = 20.0
DEVICE = torch.device("cpu")


def assert_mask_and_power(beamformer, association_mask):
    flat_mask = association_mask.transpose(0, 2, 1).reshape(
        BATCH_SIZE, NUM_AP * NUM_USERS
    )
    expanded_mask = torch.from_numpy(flat_mask).unsqueeze(1).expand_as(
        beamformer
    )
    assert torch.count_nonzero(beamformer[~expanded_mask]) == 0
    for ap in range(NUM_AP):
        block = beamformer[
            :, :, ap * NUM_USERS : (ap + 1) * NUM_USERS
        ]
        assert torch.all(block.square().sum(dim=(1, 2)) <= PMAX + 1e-6)


def build(seed):
    seed_everything(seed)
    loader = MyDataLoader(
        NUM_ANTENNAS,
        NUM_RIS_ELEMENTS,
        NUM_RIS,
        BATCH_SIZE,
        num_ap=NUM_AP,
        square_side=SQUARE_SIDE,
        topology_seed=seed,
        channel_seed=seed,
    )
    loader.BS_RIS_association()
    feature, e, index, e_dir, _ = loader.gen_training_data(
        NUM_USERS, 0.1, 0.1
    )
    snapshot_id = loader.snapshot_id
    channels = loader.get_channel_batch()
    model = node_update(
        NUM_ANTENNAS,
        NUM_RIS_ELEMENTS,
        NUM_RIS,
        1,
        PMAX,
        2,
        8,
        NUM_AP,
        DEVICE,
        NUM_USERS,
    ).to(DEVICE)
    model.eval()
    with torch.no_grad():
        centralized_w, centralized_theta = model(
            feature, e, index, e_dir, training=True
        )
        local_feature, local_e, local_index, local_e_dir = (
            loader.gen_testing_data(
                NUM_USERS, 0.1, 0.1, regenerate_channels=False
            )
        )
        decentralized_w, decentralized_theta = model(
            local_feature,
            local_e,
            local_index,
            local_e_dir,
            training=False,
        )
    assert loader.snapshot_id == snapshot_id
    assert all(
        np.array_equal(before, after)
        for before, after in zip(channels, loader.get_channel_batch())
    )
    return loader, model, feature, (
        centralized_w,
        centralized_theta,
        decentralized_w,
        decentralized_theta,
    )


def main():
    displacement = wrapped_displacement(
        np.array([1.0, 1.0]), np.array([19.0, 1.0]), SQUARE_SIDE
    )
    assert np.array_equal(displacement, np.array([-2.0, 0.0]))
    assert wrapped_distance(
        np.array([1.0, 1.0]), np.array([19.0, 1.0]), SQUARE_SIDE
    ) == 2.0

    loader, model, feature, outputs = build(0)
    centralized_w, centralized_theta, decentralized_w, decentralized_theta = (
        outputs
    )
    assert loader.BS_Loc_array.shape == (NUM_AP, 2)
    assert loader.RIS_Loc_array.shape == (NUM_RIS, 2)
    assert loader.user_loc.shape == (BATCH_SIZE, NUM_USERS, 2)
    assert loader.validate_topology()["passed"]
    assert feature.shape == (
        BATCH_SIZE,
        NUM_RIS,
        NUM_AP * NUM_USERS,
        2 * NUM_ANTENNAS * (NUM_RIS_ELEMENTS + 1),
    )
    for beamformer, theta in (
        (centralized_w, centralized_theta),
        (decentralized_w, decentralized_theta),
    ):
        assert beamformer.shape == (
            BATCH_SIZE,
            2 * NUM_ANTENNAS,
            NUM_AP * NUM_USERS,
        )
        assert theta.shape == (
            BATCH_SIZE,
            NUM_RIS,
            NUM_RIS_ELEMENTS,
            2,
        )
        assert_mask_and_power(beamformer, loader.get_association_mask())
        rates = loader.compute_rates(beamformer, theta, PMAX, DEVICE)
        assert rates.shape == (BATCH_SIZE, NUM_USERS)
        assert torch.isfinite(rates).all()

    metrics, details = evaluate_snapshot(
        model,
        loader,
        1,
        K=NUM_USERS,
        batch_size=BATCH_SIZE,
        associate_threshold=0.1,
        pmax_w=PMAX,
        device=DEVICE,
        return_details=True,
    )
    assert {
        "centralized",
        "decentralized",
        "absolute_gap",
        "relative_gap_percent",
        "visible_links_per_ap_mean",
        "paired_snapshot_count",
    } <= metrics.keys()
    assert metrics["paired_snapshot_count"] == BATCH_SIZE
    assert details["user_locations"].shape == (
        BATCH_SIZE,
        NUM_USERS,
        2,
    )
    assert details["association_mask"].shape == (
        BATCH_SIZE,
        NUM_USERS,
        NUM_AP,
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        history_path = Path(temporary_directory) / "arrays" / "history.npz"
        loader.save_history(history_path)
        assert history_path.is_file()

    second_loader, _, second_feature, second_outputs = build(0)
    assert np.array_equal(loader.BS_Loc_array, second_loader.BS_Loc_array)
    assert np.array_equal(loader.RIS_Loc_array, second_loader.RIS_Loc_array)
    assert torch.equal(feature, second_feature)
    for first, second in zip(outputs, second_outputs):
        assert torch.equal(first, second)
    different_loader = MyDataLoader(
        NUM_ANTENNAS,
        NUM_RIS_ELEMENTS,
        NUM_RIS,
        BATCH_SIZE,
        num_ap=NUM_AP,
        square_side=SQUARE_SIDE,
        topology_seed=1,
        channel_seed=0,
    )
    assert not np.array_equal(
        loader.BS_Loc_array, different_loader.BS_Loc_array
    )
    print("Stage 0 v2 BPP implementation contracts passed.")


if __name__ == "__main__":
    main()
