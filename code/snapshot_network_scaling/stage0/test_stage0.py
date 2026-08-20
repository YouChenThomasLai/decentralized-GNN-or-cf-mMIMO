import numpy as np
import torch

from data import MyDataLoader
from evaluate import evaluate_snapshot
from model_2 import node_update
from trainer_2 import seed_everything


BATCH_SIZE = 1
NUM_AP = 3
NUM_ANTENNAS = 1
NUM_RIS_ELEMENTS = 3
NUM_RIS = 2
NUM_USERS = 2
PMAX = 10 ** ((15 - 30) / 10)
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
        spatial_scale=20,
    )
    loader.BS_RIS_association()
    assert all(len(ap.RIS_array) == NUM_RIS for ap in loader.BS_array)
    feature, e, index, e_dir, _ = loader.gen_training_data(
        NUM_USERS, 0.1, 0.1
    )
    snapshot_id = loader.snapshot_id
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
    return loader, model, feature, (
        centralized_w,
        centralized_theta,
        decentralized_w,
        decentralized_theta,
    )


def main():
    loader, model, feature, outputs = build(0)
    centralized_w, centralized_theta, decentralized_w, decentralized_theta = (
        outputs
    )
    assert loader.BS_Loc_array.shape == (NUM_AP, 2)
    assert loader.RIS_Loc_array.shape == (NUM_RIS, 2)
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
        "ap_ris_degree_mean",
    } <= metrics.keys()
    assert metrics["ap_ris_degree_mean"] == NUM_RIS
    assert details["association_mask"].shape == (
        BATCH_SIZE,
        NUM_USERS,
        NUM_AP,
    )

    second_loader, _, second_feature, second_outputs = build(0)
    assert np.array_equal(
        loader.BS_Loc_array, second_loader.BS_Loc_array
    )
    assert torch.equal(feature, second_feature)
    for first, second in zip(outputs, second_outputs):
        assert torch.equal(first, second)
    print("Stage 0 scaling checks passed.")


if __name__ == "__main__":
    main()
