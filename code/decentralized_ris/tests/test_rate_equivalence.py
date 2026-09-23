import numpy as np
import torch

from model import BaselineNet
from rates import RatePrecompute
from simulation import ChannelSimulator


def test_rate_equivalence():
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cpu")
    pmax = 10 ** ((15 - 30) / 10)
    simulator = ChannelSimulator(2, 30, 4, 2)
    features, edges, masks, direct, _ = simulator.training_batch(8, 0.1, 0.1)
    model = BaselineNet(2, 30, 4, 6, pmax, 64, 5, device)

    with torch.no_grad():
        beamformer, phase = model.centralized(features, edges, masks, direct)
        _, reference, _ = simulator.loss(beamformer, phase, device)
        vectorized = RatePrecompute(simulator, device).sum_rate(beamformer, phase).mean()

    assert torch.allclose(reference, vectorized, atol=2e-5), (
        float(reference), float(vectorized)
    )


if __name__ == "__main__":
    test_rate_equivalence()
