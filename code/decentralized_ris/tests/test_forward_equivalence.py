import os
import tempfile

import numpy as np
import torch

from simulation import ChannelSimulator
from model import BaselineNet, load_checkpoint
from variants import VariantNet


def test_forward_equivalence():
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cpu")
    pmax = 10 ** ((15 - 30) / 10)
    dataloader = ChannelSimulator(2, 30, 4, 2)

    baseline = BaselineNet(2, 30, 4, 6, pmax, 64, 5, device).to(device)
    variant = VariantNet(2, 30, 4, 6, pmax, 64, 5, device, arch="r0").to(device)
    variant.load_state_dict(baseline.state_dict(), strict=True)

    features, edges, masks, direct, _ = dataloader.training_batch(8, 0.1, 0.1)
    with torch.no_grad():
        expected = baseline.centralized(features, edges, masks, direct)
        actual = variant.centralized(features, edges, masks, direct)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))

    features, edges, masks, direct = dataloader.decentralized_batch(
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
