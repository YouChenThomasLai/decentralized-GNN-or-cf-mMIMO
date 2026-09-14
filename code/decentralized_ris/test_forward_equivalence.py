import os
import tempfile

import numpy as np
import torch

from data import MyDataLoader
from model import BaselineNet, load_checkpoint
from ris_action_variants import VariantNet


def test_forward_equivalence():
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cpu")
    pmax = 10 ** ((15 - 30) / 10)
    dataloader = MyDataLoader(2, 30, 4, 2)
    dataloader.BS_RIS_association()

    baseline = BaselineNet(2, 30, 4, 6, pmax, 64, 5, device).to(device)
    variant = VariantNet(2, 30, 4, 6, pmax, 64, 5, device, arch="r0").to(device)
    variant.load_state_dict(baseline.state_dict(), strict=True)

    features, edges, masks, direct, _ = dataloader.gen_training_data(8, 0.1, 0.1)
    with torch.no_grad():
        expected = baseline.centralized(features, edges, masks, direct)
        actual = variant.centralized(features, edges, masks, direct)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))

    features, edges, masks, direct = dataloader.gen_testing_data(
        8, 0.1, 0.1, regenerate_channels=False
    )
    with torch.no_grad():
        expected = baseline.decentralized(features, edges, masks, direct)
        actual = variant.decentralized(features, edges, masks, direct)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))

    legacy_state = dict(baseline.state_dict())
    legacy_state["update_list.0.fc.0.weight"] = torch.zeros(1)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        checkpoint = handle.name
    try:
        torch.save(legacy_state, checkpoint)
        load_checkpoint(baseline, checkpoint, device)
    finally:
        os.unlink(checkpoint)


if __name__ == "__main__":
    test_forward_equivalence()
