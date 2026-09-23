import numpy as np
import torch

from dnn import DNN_ARCHS, DnnNet, default_consensus, flat_input_dim, is_dnn
from simulation import ChannelSimulator
from variants import VariantNet, local_energy, unit_modulus_error


SETTING = dict(M=2, N=3, L=2, Pmax=0.03, AP=2, users_per_ap=2)


def dnn_kwargs(arch, width=6, depth=2):
    return dict(
        M=SETTING["M"],
        N=SETTING["N"],
        L=SETTING["L"],
        D=1,
        Pmax=SETTING["Pmax"],
        ch=4,
        AP=SETTING["AP"],
        device=torch.device("cpu"),
        users_per_ap=SETTING["users_per_ap"],
        arch=arch,
        width=width,
        depth=depth,
    )


def batch():
    np.random.seed(0)
    torch.manual_seed(0)
    simulator = ChannelSimulator(
        SETTING["M"], SETTING["N"], SETTING["L"], 2, n_ap=SETTING["AP"]
    )
    central = simulator.training_batch(SETTING["users_per_ap"], 0.1, 0.1)
    decentral = simulator.decentralized_batch(
        SETTING["users_per_ap"], 0.1, 0.1, regenerate_channels=False
    )
    return simulator, central, decentral


def test_arch_registry_and_deployment_modes():
    assert DNN_ARCHS == ("d0", "d1")
    assert is_dnn("d0") and is_dnn("d1") and not is_dnn("g2")
    assert default_consensus("d0") == "direct"
    assert default_consensus("d1") == "energy"

    d0 = DnnNet(**dnn_kwargs("d0"))
    d1 = DnnNet(**dnn_kwargs("d1"))
    assert not d0.supports_decentralized and d1.supports_decentralized
    assert d0.describe()["ap_to_cpu_reals_per_ap_ris"] is None
    assert d1.describe()["ap_to_cpu_reals_per_ap_ris"] == SETTING["N"] + 1
    assert all(net.describe()["cpu_trainable_parameters"] == 0 for net in (d0, d1))

    # The extra input block is the AP's served-user mask, which is what makes
    # one shared trunk produce a different proposal at every AP.
    k_total = SETTING["AP"] * SETTING["users_per_ap"]
    assert d1.in_dim - d0.in_dim == k_total
    assert d0.in_dim == flat_input_dim(
        SETTING["M"], SETTING["N"], SETTING["L"], k_total, "d0"
    )


def test_outputs_satisfy_the_shared_action_constraints():
    simulator, central, decentral = batch()
    features, edges, masks, direct, _ = central
    d_features, d_edges, d_masks, d_direct = decentral
    k_total = SETTING["AP"] * SETTING["users_per_ap"]

    for arch in DNN_ARCHS:
        net = DnnNet(**dnn_kwargs(arch))
        beamformer, phase = net.centralized(features, edges, masks, direct)
        assert beamformer.shape == (2, 2 * SETTING["M"], k_total)
        assert phase.shape == (2, SETTING["L"], SETTING["N"], 2)
        assert torch.isfinite(beamformer).all() and torch.isfinite(phase).all()
        assert unit_modulus_error(phase) < 1e-6

        power = beamformer.reshape(
            2, 2 * SETTING["M"], SETTING["AP"], SETTING["users_per_ap"]
        ).permute(0, 2, 1, 3).reshape(2, SETTING["AP"], -1).pow(2).sum(dim=2).detach()
        assert float(power.max()) <= SETTING["Pmax"] + 1e-6

        loss, rate, _ = simulator.loss(beamformer, phase, torch.device("cpu"))
        loss.backward()
        assert torch.isfinite(rate)
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in net.parameters()
        )

        if arch == "d0":
            try:
                net.decentralized(d_features, d_edges, d_masks, d_direct)
            except NotImplementedError:
                pass
            else:
                raise AssertionError("d0 must not expose a decentralized path")
            continue

        beamformer, phase = net.decentralized(d_features, d_edges, d_masks, d_direct)
        assert beamformer.shape == (2, 2 * SETTING["M"], k_total)
        assert unit_modulus_error(phase) < 1e-6
        assert torch.isfinite(beamformer).all() and torch.isfinite(phase).all()


def test_d1_consensus_weight_uses_only_own_served_channels():
    """The E05/E06 locality property must hold for the DNN interface too."""
    _, _, decentral = batch()
    d_features, d_edges, d_masks, d_direct = decentral
    net = DnnNet(**dnn_kwargs("d1"))

    paper, own = {}, {}
    net.decentralized(d_features, d_edges, d_masks, d_direct, trace=paper)
    net.decentralized(
        d_features, d_edges, d_masks, d_direct, trace=own, include_cross_ap_csi=False
    )
    torch.testing.assert_close(paper["energy"], own["energy"], rtol=0, atol=0)
    assert not paper["energy"].requires_grad

    edges = torch.cat(list(d_edges), dim=2)
    served = torch.zeros(edges.shape[0], edges.shape[2])
    served[:, : SETTING["users_per_ap"]] = torch.as_tensor(
        np.asarray(d_masks[0]), dtype=torch.float32
    )
    expected = local_energy(
        edges * served[:, None, :], served, 0, SETTING["users_per_ap"]
    )
    torch.testing.assert_close(paper["energy"][:, 0], expected)


def test_registered_width_matches_g2_capacity_within_ten_percent():
    device = torch.device("cpu")
    pmax = 10 ** ((15.0 - 30) / 10)
    g2 = VariantNet(2, 30, 4, 6, pmax, 64, 5, device, arch="g2", users_per_ap=8)
    target = g2.describe()["effective_parameters"]
    for arch in DNN_ARCHS:
        net = DnnNet(
            2, 30, 4, 6, pmax, 64, 5, device,
            arch=arch, users_per_ap=8, width=40, depth=3,
        )
        ratio = net.describe()["effective_parameters"] / target
        assert 0.9 <= ratio <= 1.1, (arch, ratio)


if __name__ == "__main__":
    test_arch_registry_and_deployment_modes()
    test_outputs_satisfy_the_shared_action_constraints()
    test_d1_consensus_weight_uses_only_own_served_channels()
    test_registered_width_matches_g2_capacity_within_ten_percent()
    print("ok")
