import numpy as np
import torch
import torch.nn.functional as F

from model import RisMerge
from simulation import ChannelSimulator
from variants import VariantNet, _unit_from_pairs, circular_consensus


def test_commuted_reduction_and_projection_order():
    torch.manual_seed(0)
    merge = RisMerge(3)
    latents = [torch.randn(2, 4, 12) for _ in range(5)]

    original = merge(latents)
    commuted = merge.forward_commuted(latents)
    torch.testing.assert_close(commuted, original, rtol=1e-6, atol=1e-6)

    # A concrete counterexample: Pi(z1 + z2) != Pi(Pi(z1) + Pi(z2)).
    merge = RisMerge(1)
    with torch.no_grad():
        merge.f_merge.weight.zero_()
        merge.f_merge.bias.zero_()
        merge.f_merge.weight[0, 0] = 1.0
        merge.f_merge.weight[1, 1] = 1.0
    latents = [
        torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        torch.tensor([[[0.0, 2.0, 0.0, 0.0]]]),
    ]
    project_after_sum = merge.forward_commuted(latents)
    logits = torch.stack(merge.local_logits(latents), dim=1)
    proposals = _unit_from_pairs(logits, n_ris=1, n_elem=1)
    project_before_sum, _ = circular_consensus(
        proposals, torch.ones(1, 2), logits=None
    )
    assert not torch.allclose(project_before_sum, project_after_sum)


def test_shared_variants_are_checkpoint_compatible():
    kwargs = dict(
        M=2,
        N=3,
        L=2,
        D=1,
        Pmax=0.03,
        ch=4,
        AP=2,
        device=torch.device("cpu"),
        users_per_ap=2,
    )
    r0 = VariantNet(**kwargs, arch="r0")
    r0c = VariantNet(**kwargs, arch="r0c")
    r1_shared = VariantNet(**kwargs, arch="r1_shared")

    r0c.load_state_dict(r0.state_dict(), strict=True)
    r1_shared.load_state_dict(r0.state_dict(), strict=True)
    assert r0.describe()["effective_parameters"] == r1_shared.describe()[
        "effective_parameters"
    ]

    np.random.seed(0)
    simulator = ChannelSimulator(2, 3, 2, 2, n_ap=2)
    features, edges, masks, direct, _ = simulator.training_batch(2, 0.1, 0.1)
    with torch.no_grad():
        r0_out = r0.centralized(features, edges, masks, direct)
        r0c_out = r0c.centralized(features, edges, masks, direct)
        _, r1_phase = r1_shared.centralized(features, edges, masks, direct)
    for original, commuted in zip(r0_out, r0c_out):
        torch.testing.assert_close(commuted, original, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        r1_phase.norm(dim=-1), torch.ones_like(r1_phase[..., 0]), atol=1e-6, rtol=0
    )

    features, edges, masks, direct = simulator.decentralized_batch(
        2, 0.1, 0.1, regenerate_channels=False
    )
    with torch.no_grad():
        r0_out = r0.decentralized(features, edges, masks, direct)
        r0c_out = r0c.decentralized(features, edges, masks, direct)
    for original, commuted in zip(r0_out, r0c_out):
        torch.testing.assert_close(commuted, original, rtol=1e-6, atol=1e-6)


def test_ap_ris_mag_changes_only_the_consensus_weight():
    kwargs = dict(
        M=2, N=3, L=2, D=1, Pmax=0.03, ch=4, AP=2,
        device=torch.device("cpu"), users_per_ap=2,
    )
    r0 = VariantNet(**kwargs, arch="r0")
    r1_shared = VariantNet(**kwargs, arch="r1_shared")
    ap_ris_mag = VariantNet(**kwargs, arch="r1_ap_ris_mag")
    r1_shared.load_state_dict(r0.state_dict(), strict=True)
    ap_ris_mag.load_state_dict(r0.state_dict(), strict=True)

    described = ap_ris_mag.describe()
    assert described["effective_parameters"] == r0.describe()["effective_parameters"]
    assert described["cpu_trainable_parameters"] == 0
    assert described["ap_to_cpu_reals_per_ap_ris"] == kwargs["N"] + 1

    # Uniform weights must reproduce the unweighted rule exactly, so r1_ap_ris_mag
    # differs from r1_shared only through the magnitudes it feeds in.
    proposals = F.normalize(torch.randn(2, 3, 2, 4, 2), dim=-1)
    equal, _ = circular_consensus(proposals, torch.ones(2, 3))
    weighted, _ = circular_consensus(
        proposals, torch.ones(2, 3), weights=torch.ones(2, 3, 2)
    )
    torch.testing.assert_close(weighted, equal, rtol=1e-6, atol=1e-6)

    np.random.seed(0)
    simulator = ChannelSimulator(2, 3, 2, 2, n_ap=2)
    features, edges, masks, direct, _ = simulator.training_batch(2, 0.1, 0.1)
    trace = {}
    with torch.no_grad():
        _, shared_phase = r1_shared.centralized(features, edges, masks, direct)
        _, ap_ris_mag_phase = ap_ris_mag.centralized(
            features, edges, masks, direct, trace
        )

    torch.testing.assert_close(
        ap_ris_mag_phase.norm(dim=-1),
        torch.ones_like(ap_ris_mag_phase[..., 0]),
        atol=1e-6, rtol=0,
    )
    assert not torch.allclose(ap_ris_mag_phase, shared_phase)

    # The weight is exactly s_{l,r} = mean_n ||z_{l,r,n}||, normalised over APs.
    expected = trace["z_pairs"].norm(dim=-1).mean(dim=3)
    expected = expected / expected.sum(dim=1, keepdim=True)
    torch.testing.assert_close(trace["weights"], expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_commuted_reduction_and_projection_order()
    test_shared_variants_are_checkpoint_compatible()
    test_ap_ris_mag_changes_only_the_consensus_weight()
