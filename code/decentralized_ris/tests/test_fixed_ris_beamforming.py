"""Numerical checks for the E13 fixed-G2-RIS active-beamforming controls."""

import math

import numpy as np
import torch

from experiments.fixed_ris_beamforming import (
    complex_channels,
    embed_beamformer,
    local_precoder,
    locality_control,
    per_ap_power,
    rescale_to_full_power,
)
from rates import RatePrecompute
from simulation import ChannelSimulator


PMAX = 10 ** ((15 - 30) / 10)


def random_case(seed=0, batch=3, n_ap=4, k_user=5, n_antennas=2):
    generator = torch.Generator().manual_seed(seed)
    effective = torch.randn(
        (batch, n_ap * k_user, 2 * n_antennas), generator=generator
    )
    served = torch.rand((batch, n_ap, k_user), generator=generator) > 0.4
    served[:, 0, :] = False                       # an AP that serves nobody
    served[:, 1, 0] = True                        # an AP that serves someone
    return effective, served


def test_mrt_is_channel_matched_and_power_feasible():
    effective, served = random_case()
    weights, _ = local_precoder(effective, served, PMAX, "mrt")
    gain = complex_channels(effective, served.shape[1], effective.shape[2] // 2).conj()

    # Cauchy-Schwarz holds with equality exactly when w is aligned with g.
    inner = (gain.conj() * weights).sum(dim=3).abs()
    bound = gain.abs().pow(2).sum(dim=3).sqrt() * weights.abs().pow(2).sum(dim=3).sqrt()
    mask = served.to(inner.dtype)
    torch.testing.assert_close(inner * mask, bound * mask, atol=1e-6, rtol=1e-5)

    power = weights.reshape(weights.shape[0], weights.shape[1], -1).abs().pow(2).sum(dim=2)
    active = served.any(dim=2)
    torch.testing.assert_close(
        power[active], torch.full_like(power[active], PMAX), atol=1e-9, rtol=1e-6
    )
    assert float(power[~active].abs().max()) == 0.0


def test_precoders_zero_every_unserved_column():
    effective, served = random_case(seed=1)
    for kind in ("mrt", "rzf"):
        weights, _ = local_precoder(effective, served, PMAX, kind)
        assert float(weights[~served].abs().max()) == 0.0


def test_rzf_nulls_intra_ap_interference_when_it_can():
    """With |K_l| <= M and a vanishing regularizer, RZF becomes zero-forcing."""
    effective, served = random_case(seed=2, k_user=2)
    served[:] = True
    weights, residual = local_precoder(effective, served, PMAX, "rzf", sigma=1e-12)
    gain = complex_channels(effective, served.shape[1], effective.shape[2] // 2).conj()

    cross = torch.einsum("bakm,bajm->bakj", gain.conj(), weights).abs()
    diagonal = torch.diagonal(cross, dim1=2, dim2=3)
    off = cross - torch.diag_embed(diagonal)
    assert float(off.max()) < 1e-4 * float(diagonal.min())
    assert residual < 1e-4


def test_rzf_approaches_mrt_as_the_regularizer_dominates():
    effective, served = random_case(seed=3)
    heavy, _ = local_precoder(effective, served, PMAX, "rzf", sigma=1e9)
    reference, _ = local_precoder(effective, served, PMAX, "mrt")
    torch.testing.assert_close(heavy, reference, atol=1e-5, rtol=1e-4)


def test_precoders_use_only_their_own_ap_block():
    effective, served = random_case(seed=4)
    assert locality_control(effective, served, PMAX) == 0.0


def test_full_power_rescaling_keeps_direction():
    effective, served = random_case(seed=5)
    weights, _ = local_precoder(effective, served, PMAX, "rzf", alpha=torch.full(
        (effective.shape[0], served.shape[1]), 0.25))
    embedded = embed_beamformer(weights)
    rescaled = rescale_to_full_power(embedded, served, PMAX, served.shape[1])

    power = per_ap_power(rescaled, served.shape[1])
    active = served.any(dim=2)
    torch.testing.assert_close(
        power[active], torch.full_like(power[active], PMAX), atol=1e-9, rtol=1e-6
    )
    assert float(power[~active].abs().max()) == 0.0
    # Direction is untouched: the rescaled block is a positive multiple.
    scaled = embedded * 2.0
    torch.testing.assert_close(
        rescale_to_full_power(scaled, served, PMAX, served.shape[1]), rescaled,
        atol=1e-6, rtol=1e-5,
    )


def test_embedding_and_conjugation_reproduce_the_rate_model():
    """One user per AP removes interference, so the MRT rate is available in closed form.

    This pins the layout of `embed_beamformer` and the g = conj(h) convention
    against `rates`, where a transposed block or a missing conjugate would still
    produce plausible-looking numbers.
    """
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cpu")
    n_ap, n_antennas, n_ris, n_elements = 3, 2, 2, 4
    simulator = ChannelSimulator(n_antennas, n_elements, n_ris, 2, n_ap=n_ap)
    simulator.training_batch(1, 0.1, 0.1)
    precompute = RatePrecompute(simulator, device)

    angles = torch.rand((2, n_ris, n_elements)) * 2 * math.pi
    theta = torch.stack((angles.cos(), angles.sin()), dim=-1)
    effective = precompute.effective_channel(theta)
    served = torch.ones((2, n_ap, 1), dtype=torch.bool)

    weights, _ = local_precoder(effective, served, PMAX, "mrt")
    rate = precompute.sum_rate_from_effective(embed_beamformer(weights), effective)

    norms = complex_channels(effective, n_ap, n_antennas).abs().pow(2).sum(dim=3).sqrt()
    signal = PMAX * norms.sum(dim=1).squeeze(1) ** 2
    expected = torch.log2(1.0 + signal / RatePrecompute.SIGMA)
    torch.testing.assert_close(rate, expected, atol=1e-4, rtol=1e-5)


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            case()
            print(f"[ok] {name}")
