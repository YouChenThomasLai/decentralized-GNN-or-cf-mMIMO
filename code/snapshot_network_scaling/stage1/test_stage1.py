import numpy as np
import torch

from data import MyDataLoader
from environment import SnapshotEnvironment
from model_2 import node_update
from train import Trainer, seed_everything
from utils_return_indivial_rates import (
    mrt_beamforming,
    rzf_beamforming,
    wrap_around_distances,
)


BATCH_SIZE = 2
NUM_AP = 3
NUM_ANTENNAS = 2
NUM_USERS = 3
PMAX = 10 ** ((15 - 30) / 10)
SQUARE_SIDE = 60.0
DEVICE = torch.device("cpu")


def build_snapshot(seed):
    seed_everything(seed)
    loader = SnapshotEnvironment(
        NUM_ANTENNAS,
        BATCH_SIZE,
        num_ap=NUM_AP,
        square_side=SQUARE_SIDE,
        topology_seed=seed,
    )
    centralized_feature, centralized_index = loader.gen_training_data(
        NUM_USERS, 0.1
    )
    channels_before = loader.get_stacked_channels().copy()
    mask_before = loader.get_association_mask().copy()
    decentralized_feature, decentralized_index = loader.gen_testing_data(
        NUM_USERS, 0.1, regenerate_channels=False
    )
    model = node_update(
        NUM_ANTENNAS, 2, PMAX, 16, NUM_AP, DEVICE
    ).to(DEVICE)
    model.eval()
    with torch.no_grad():
        beamformers = {
            "centralized_gnn": model(
                centralized_feature, centralized_index, training=True
            ),
            "decentralized_gnn": model(
                decentralized_feature, decentralized_index, training=False
            ),
            "mrt": mrt_beamforming(
                channels_before, mask_before, PMAX, DEVICE
            ),
            "rzf": rzf_beamforming(
                channels_before, mask_before, PMAX, DEVICE, 1e-12
            ),
        }
    assert np.array_equal(channels_before, loader.get_stacked_channels())
    assert np.array_equal(mask_before, loader.get_association_mask())
    return loader, centralized_feature, beamformers


def assert_mask_and_power(beamformer, association_mask):
    num_users = association_mask.shape[1]
    flat_mask = association_mask.transpose(0, 2, 1).reshape(
        BATCH_SIZE, NUM_AP * num_users
    )
    expanded_mask = torch.from_numpy(flat_mask).unsqueeze(1).expand_as(
        beamformer
    )
    assert torch.count_nonzero(beamformer[~expanded_mask]) == 0
    for ap in range(NUM_AP):
        block = beamformer[:, :, ap * num_users : (ap + 1) * num_users]
        power = block.square().sum(dim=(1, 2))
        assert torch.all(power <= PMAX + 1e-6)


def assert_evaluator_contract():
    def evaluate(seed):
        seed_everything(seed)
        trainer = Trainer(
            M=NUM_ANTENNAS,
            K=NUM_USERS,
            batch_size=BATCH_SIZE,
            n_iter=1,
            pmax_dbm=15,
            noise_power=1e-12,
            device="cpu",
            num_ap=NUM_AP,
            square_side=SQUARE_SIDE,
            topology_seed=seed,
        )
        return trainer.eval(4, return_details=True)

    first_metrics, first_details = evaluate(17)
    second_metrics, second_details = evaluate(17)
    required = {
        "centralized_gnn",
        "decentralized_gnn",
        "absolute_gap",
        "relative_gap_percent",
        "serving_ap_count_mean",
        "local_to_global_visibility_ratio_mean",
    }
    assert required <= first_metrics.keys()
    for key in first_metrics:
        if key.endswith("_seconds_per_sample"):
            continue
        assert np.isclose(first_metrics[key], second_metrics[key], atol=1e-6)
    for key in first_details:
        assert np.array_equal(first_details[key], second_details[key])


def main():
    assert MyDataLoader is SnapshotEnvironment
    assert np.isclose(
        wrap_around_distances(
            np.array([1.0, 1.0]), np.array([59.0, 1.0]), SQUARE_SIDE
        ).item(),
        2.0,
    )
    loader, centralized_feature, beamformers = build_snapshot(0)
    assert loader.BS_Loc_array.shape == (NUM_AP, 2)
    assert np.all((loader.BS_Loc_array >= 0) & (loader.BS_Loc_array < SQUARE_SIDE))
    assert loader.get_stacked_channels().shape == (
        BATCH_SIZE,
        NUM_AP,
        NUM_USERS,
        NUM_ANTENNAS,
    )
    assert centralized_feature.shape == (
        BATCH_SIZE,
        1,
        NUM_AP * NUM_USERS,
        2 * NUM_ANTENNAS,
    )
    for beamformer in beamformers.values():
        assert beamformer.shape == (
            BATCH_SIZE,
            2 * NUM_ANTENNAS,
            NUM_AP * NUM_USERS,
        )
        assert_mask_and_power(beamformer, loader.get_association_mask())
        outputs = loader.compute_loss(beamformer, DEVICE, 1e-12)
        assert all(torch.isfinite(output).all() for output in outputs)

    second_loader, second_feature, second_beamformers = build_snapshot(0)
    assert np.array_equal(
        loader.get_stacked_channels(), second_loader.get_stacked_channels()
    )
    assert torch.equal(centralized_feature, second_feature)
    for method in beamformers:
        assert torch.equal(beamformers[method], second_beamformers[method])
    assert_evaluator_contract()
    print("Stage 1 scaling checks passed.")


if __name__ == "__main__":
    main()
