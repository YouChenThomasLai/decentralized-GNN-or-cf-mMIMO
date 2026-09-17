import numpy as np
import torch

from simulation import ChannelSimulator
from variants import (
    ARCHS,
    VariantNet,
    canonical_arch,
    circular_consensus,
    decode_phase,
    default_consensus,
    encode_phase,
    local_energy,
)


def model_kwargs(arch):
    return dict(
        M=2,
        N=3,
        L=2,
        D=1,
        Pmax=0.03,
        ch=4,
        AP=2,
        device=torch.device("cpu"),
        users_per_ap=2,
        arch=arch,
        consensus="energy",
    )


def test_canonical_graph_method_names_and_legacy_configs():
    assert "r3a" not in ARCHS and "r3b" not in ARCHS
    assert canonical_arch("r3a") == "g1"
    assert canonical_arch("r3b") == "g2"
    assert canonical_arch("r1_shared", "energy") == "g0"
    assert all(
        default_consensus(arch) == "energy" for arch in ("g0", "g1", "g2")
    )

    for legacy, current in (("r3a", "g1"), ("r3b", "g2")):
        net = VariantNet(**model_kwargs(legacy))
        assert net.describe()["arch"] == current


def test_local_energy_uses_only_own_served_channels():
    edges = torch.arange(1, 13, dtype=torch.float32).reshape(1, 2, 6)
    edges.requires_grad_()
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0, 1.0]])

    expected = edges[:, :, 2:4] * mask[:, None, 2:4]
    actual = local_energy(edges, mask, ap_index=1, users_per_ap=2)
    torch.testing.assert_close(actual, expected.sum(dim=2))
    assert not actual.requires_grad

    changed = edges.detach().clone()
    changed[:, :, :2] = -1e9                    # another AP's CSI
    changed[:, :, 3] = 1e9                     # own, but not served
    torch.testing.assert_close(
        local_energy(changed, mask, 1, 2), actual
    )


def test_all_graph_arms_share_energy_consensus_and_gradient_boundary():
    np.random.seed(0)
    torch.manual_seed(0)
    simulator = ChannelSimulator(2, 3, 2, 2, n_ap=2)
    features, edges, masks, direct, _ = simulator.training_batch(2, 0.1, 0.1)

    for arch in ("g0", "g1", "g2"):
        torch.manual_seed(0)
        net = VariantNet(**model_kwargs(arch))
        trace = {}
        beamformer, theta = net.centralized(
            features, edges, masks, direct, trace=trace
        )

        expected_weights = trace["energy"] / trace["energy"].sum(
            dim=1, keepdim=True
        ).clamp(min=1e-12)
        torch.testing.assert_close(trace["weights"], expected_weights)
        assert not trace["energy"].requires_grad
        assert trace["proposals"].requires_grad

        resultant = (
            trace["proposals"] * trace["weights"][..., None, None]
        ).sum(dim=1)
        expected_theta = resultant / resultant.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-12)
        torch.testing.assert_close(theta, expected_theta)
        torch.testing.assert_close(
            theta.norm(dim=-1), torch.ones_like(theta[..., 0]), atol=1e-6, rtol=0
        )

        (-beamformer.square().mean() - theta[..., 0].mean()).backward()
        assert any(parameter.grad is not None for parameter in net.parameters())


def test_paper_and_own_only_use_identical_local_energy():
    np.random.seed(1)
    torch.manual_seed(1)
    simulator = ChannelSimulator(2, 3, 2, 2, n_ap=2)
    simulator.training_batch(2, 0.1, 0.1)
    features, edges, masks, direct = simulator.decentralized_batch(
        2, 0.1, 0.1, regenerate_channels=False
    )

    for arch in ("g0", "g1", "g2"):
        torch.manual_seed(1)
        net = VariantNet(**model_kwargs(arch))
        paper_trace, own_trace = {}, {}
        with torch.no_grad():
            paper = net.decentralized(
                features, edges, masks, direct, trace=paper_trace
            )
            net.decentralized(
                features,
                edges,
                masks,
                direct,
                trace=own_trace,
                include_cross_ap_csi=False,
            )
            replay = net.decentralized(
                features, edges, masks, direct
            )
        torch.testing.assert_close(paper_trace["energy"], own_trace["energy"])
        for first, second in zip(paper, replay):
            torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_phase_angle_codec_preserves_consensus_and_rate():
    angles = torch.tensor(
        [-torch.pi, -torch.pi + 1e-6, -0.0, 0.0, torch.pi - 1e-6, torch.pi]
    )
    proposals = torch.stack((angles.cos(), angles.sin()), dim=-1)
    torch.testing.assert_close(
        decode_phase(encode_phase(proposals)), proposals, rtol=0, atol=2e-7
    )

    torch.manual_seed(2)
    proposals = torch.nn.functional.normalize(torch.randn(2, 3, 2, 4, 2), dim=-1)
    active = torch.ones(2, 3)
    weights = torch.rand(2, 3, 2)
    expected, _ = circular_consensus(proposals, active, weights=weights)
    actual, _ = circular_consensus(
        decode_phase(encode_phase(proposals)), active, weights=weights
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-6)

    np.random.seed(2)
    simulator = ChannelSimulator(2, 3, 2, 2, n_ap=2)
    simulator.training_batch(2, 0.1, 0.1)
    features, edges, masks, direct = simulator.decentralized_batch(
        2, 0.1, 0.1, regenerate_channels=False
    )
    for arch in ("g0", "g1", "g2"):
        torch.manual_seed(2)
        net = VariantNet(**model_kwargs(arch))
        with torch.no_grad():
            raw_w, raw_theta = net.decentralized(
                features, edges, masks, direct, phase_codec=False
            )
            trace = {}
            codec_w, codec_theta = net.decentralized(
                features, edges, masks, direct, trace=trace
            )
            _, raw_rate, _ = simulator.loss(raw_w, raw_theta, torch.device("cpu"))
            _, codec_rate, _ = simulator.loss(
                codec_w, codec_theta, torch.device("cpu")
            )
        torch.testing.assert_close(codec_w, raw_w, rtol=0, atol=0)
        torch.testing.assert_close(codec_theta, raw_theta, rtol=0, atol=3e-6)
        torch.testing.assert_close(codec_rate, raw_rate, rtol=0, atol=1e-6)
        assert trace["phase_angles"].shape == (2, 2, 2, 3)
        assert net.describe()["ap_to_cpu_reals_per_ap_ris"] == 4


if __name__ == "__main__":
    test_canonical_graph_method_names_and_legacy_configs()
    test_local_energy_uses_only_own_served_channels()
    test_all_graph_arms_share_energy_consensus_and_gradient_boundary()
    test_paper_and_own_only_use_identical_local_energy()
    test_phase_angle_codec_preserves_consensus_and_rate()
