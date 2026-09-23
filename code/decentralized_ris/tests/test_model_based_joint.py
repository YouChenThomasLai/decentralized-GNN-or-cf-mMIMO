"""Numerical checks for the E12 model-based joint-optimization pair."""

import math

import numpy as np
import torch

import model_based as mb
from rates import RatePrecompute
from simulation import ChannelSimulator


PMAX = 10 ** ((15 - 30) / 10)
DTYPE = torch.complex128


def small_case(seed=0, batch=2, n_ap=3, k_user=3, n_antennas=2, n_ris=2, n_elements=6):
    """A miniature system drawn from the real simulator, not from noise."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    simulator = ChannelSimulator(n_antennas, n_elements, n_ris, batch, n_ap=n_ap)
    simulator.training_batch(k_user, 0.1, 0.1)
    ris, direct, served = mb.channel_tensors(simulator, torch.device("cpu"), DTYPE)
    return simulator, ris, direct, served


def test_fp_auxiliaries_match_the_wmmse_parameterization():
    """The quadratic transform and WMMSE must be the same surrogate.

    Huang et al. write the auxiliaries as (gamma, xi) and the WMMSE literature as
    (alpha, nu).  Both readings appear in the method report, so the identity is
    pinned here rather than asserted in prose.
    """
    _, ris, direct, served = small_case()
    v = mb.initial_phase(ris, direct, served, "ones")
    gain = mb.effective_channels(ris, direct, v)
    weights = mb.mrt_initial(gain, served, PMAX)
    a = mb.aggregate(gain, weights)

    gamma, xi = mb.fp_auxiliaries(a)
    coefficient, scale = mb.surrogate_terms(gamma, xi)
    total = a.abs().pow(2).sum(dim=2) + mb.SIGMA
    nu = torch.diagonal(a, dim1=1, dim2=2) / total
    alpha = 1.0 + gamma

    torch.testing.assert_close(coefficient.real, alpha * nu.abs().pow(2))
    torch.testing.assert_close(scale, alpha.to(nu.dtype) * nu)
    # gamma is the SINR, so the rate follows from it without touching `rates`.
    torch.testing.assert_close(torch.log2(1.0 + gamma).sum(dim=1), mb.sum_rate(gain, weights))
    # At the optimal auxiliaries the surrogate is tight against the sum rate.
    torch.testing.assert_close(
        mb.surrogate_value(a, gamma, xi), mb.sum_rate(gain, weights)
    )


def test_complex_rate_matches_the_maintained_path():
    """The solvers' rate must equal the one every learned arm is scored with.

    A transposed block, a dropped conjugate, or theta instead of conj(theta)
    would still produce plausible numbers, so this pins the whole convention
    chain against `rates` and against the scalar `simulator.loss` path.
    """
    simulator, ris, direct, served = small_case(seed=1)
    device = torch.device("cpu")
    precompute = RatePrecompute(simulator, device)

    generator = torch.Generator().manual_seed(0)
    angles = 2 * math.pi * torch.rand(
        (ris.shape[0], ris.shape[3], ris.shape[5]), generator=generator
    )
    theta = torch.stack((angles.cos(), angles.sin()), dim=-1)
    v = mb.from_theta(theta).to(DTYPE)

    gain = mb.effective_channels(ris, direct, v)
    weights = mb.mrt_initial(gain, served, PMAX)
    embedded = mb.embed_beamformer(weights)

    mine = mb.sum_rate(gain, weights)
    reference = precompute.sum_rate_from_effective(
        embedded, precompute.effective_channel(theta)
    )
    torch.testing.assert_close(
        mine.to(torch.float32), reference, atol=2e-5, rtol=1e-5
    )
    _, scalar, _ = simulator.loss(embedded, theta, device)
    torch.testing.assert_close(
        mine.mean().to(torch.float32), scalar, atol=2e-5, rtol=1e-5
    )


def test_phase_conventions_round_trip():
    _, ris, direct, served = small_case(seed=2)
    generator = torch.Generator().manual_seed(1)
    angles = 2 * math.pi * torch.rand(
        (ris.shape[0], ris.shape[3], ris.shape[5]), generator=generator
    )
    theta = torch.stack((angles.cos(), angles.sin()), dim=-1)
    torch.testing.assert_close(mb.to_theta(mb.from_theta(theta)), theta)
    # Conjugation is what makes the effective channel affine in v; a missing one
    # would leave the round trip intact but break this identity.
    v = mb.from_theta(theta).to(DTYPE)
    assert float((v.conj() - torch.complex(theta[..., 0], theta[..., 1]).to(DTYPE)).abs().max()) == 0.0


def test_power_solve_respects_the_budget_and_the_kkt_cases():
    _, ris, direct, served = small_case(seed=3)
    v = mb.initial_phase(ris, direct, served, "ones")
    gain = mb.effective_channels(ris, direct, v)
    weights = mb.mrt_initial(gain, served, PMAX)
    gamma, xi = mb.fp_auxiliaries(mb.aggregate(gain, weights))
    coefficient, scale = mb.surrogate_terms(gamma, xi)

    local = gain[:, 0]
    system = torch.einsum("bk,bkm,bkn->bmn", coefficient, local, local.conj())
    rhs = scale.unsqueeze(-1) * local
    block = mb._solve_power_constrained(system, rhs, served[:, 0], PMAX, 80)

    power = block.abs().pow(2).sum(dim=(1, 2))
    assert float((power - PMAX).max()) <= 1e-9
    assert float(block[~served[:, 0]].abs().max()) == 0.0

    # A budget far above the unconstrained optimum must leave mu at zero, so the
    # solve reduces to the plain stationarity condition.
    generous = mb._solve_power_constrained(system, rhs, served[:, 0], 1e6, 80)
    masked = rhs * served[:, 0].unsqueeze(-1).to(rhs.dtype)
    expected = torch.linalg.solve(system, masked.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(
        generous, expected * served[:, 0].unsqueeze(-1).to(rhs.dtype)
    )

    # A short bisection must still return the feasible side of its bracket.
    short = mb._solve_power_constrained(system, rhs, served[:, 0], PMAX, 4)
    assert float((short.abs().pow(2).sum(dim=(1, 2)) - PMAX).max()) <= 1e-12


def test_beamformer_step_is_feasible_and_improves_the_surrogate():
    _, ris, direct, served = small_case(seed=4)
    v = mb.initial_phase(ris, direct, served, "ones")
    gain = mb.effective_channels(ris, direct, v)
    weights = mb.mrt_initial(gain, served, PMAX)
    gamma, xi = mb.fp_auxiliaries(mb.aggregate(gain, weights))

    before = mb.surrogate_value(mb.aggregate(gain, weights), gamma, xi)
    updated = mb.beamformer_step(gain, weights, gamma, xi, served, PMAX, 2, 80)
    after = mb.surrogate_value(mb.aggregate(gain, updated), gamma, xi)

    assert float((before - after).max()) <= 1e-9
    assert float((mb.per_ap_power(updated) - PMAX).max()) <= 1e-9
    assert float(updated[~served].abs().max()) == 0.0


def test_phase_solvers_decrease_the_same_quadratic():
    """MM is the published solver; coordinate descent is the control on it."""
    _, ris, direct, served = small_case(seed=5)
    batch = ris.shape[0]
    v = mb.initial_phase(ris, direct, served, "random").reshape(batch, -1)
    gain = mb.effective_channels(ris, direct, v.reshape(batch, ris.shape[3], ris.shape[5]))
    weights = mb.mrt_initial(gain, served, PMAX)
    gamma, xi = mb.fp_auxiliaries(mb.aggregate(gain, weights))
    system, cross = mb.phase_quadratic(ris, direct, weights, gamma, xi)

    before = mb.quadratic_value(system, cross, v)
    for solver in (mb.mm_phase_step, mb.cd_phase_step):
        updated = solver(system, cross, v, 12)
        assert float((updated.abs() - 1.0).abs().max()) < 1e-12
        assert float((mb.quadratic_value(system, cross, updated) - before).max()) <= 1e-9


def test_phase_quadratic_reproduces_the_surrogate_it_stands_for():
    """Z and q must describe the actual objective, not merely be quadratic.

    The surrogate is evaluated directly at two different phases and compared with
    the quadratic form's difference, which catches a wrong conjugate or a missing
    direct-link term in q.
    """
    _, ris, direct, served = small_case(seed=6)
    batch, n_ris, n_elements = ris.shape[0], ris.shape[3], ris.shape[5]
    v = mb.initial_phase(ris, direct, served, "ones")
    gain = mb.effective_channels(ris, direct, v)
    weights = mb.mrt_initial(gain, served, PMAX)
    gamma, xi = mb.fp_auxiliaries(mb.aggregate(gain, weights))
    system, cross = mb.phase_quadratic(ris, direct, weights, gamma, xi)

    other = mb.initial_phase(ris, direct, served, "random")
    flat = [v.reshape(batch, -1), other.reshape(batch, -1)]
    quadratic = [mb.quadratic_value(system, cross, value) for value in flat]
    surrogate = [
        mb.surrogate_value(
            mb.aggregate(
                mb.effective_channels(
                    ris, direct, value.reshape(batch, n_ris, n_elements)
                ),
                weights,
            ),
            gamma,
            xi,
        )
        for value in flat
    ]
    # The surrogate is maximized while the quadratic form is minimized, and they
    # differ by a constant and the 1/ln 2 scaling of the rate units.
    torch.testing.assert_close(
        (quadratic[1] - quadratic[0]) / math.log(2.0),
        surrogate[0] - surrogate[1],
        atol=1e-9,
        rtol=1e-7,
    )


def test_ring_incidence_encodes_connected_consensus():
    for n_ap in (3, 5):
        edges, incidence = mb.ring_incidence(n_ap)
        assert len(edges) == n_ap
        # A_l^H A_l = 2 I: every AP sits on exactly two ring edges, one with each
        # sign, which is what makes the penalty term rho * I.
        for entries in incidence:
            assert len(entries) == 2
            assert sorted(sign for _, sign in entries) == [-1.0, 1.0]
        copies = torch.arange(n_ap, dtype=torch.float32)
        residual = torch.stack([copies[head] - copies[tail] for head, tail in edges])
        assert float(residual.abs().max()) > 0.0
        equal = torch.ones(n_ap)
        residual = torch.stack([equal[head] - equal[tail] for head, tail in edges])
        assert float(residual.abs().max()) == 0.0


def test_centralized_solver_is_monotone_and_feasible():
    _, ris, direct, served = small_case(seed=7)
    record = mb.centralized_fp(
        ris, direct, served, PMAX, max_iterations=15, trace=True
    )
    trace = record["trace"]
    assert record["monotonicity_violation"] <= 1e-9
    assert record["phase_block_violation"] <= 1e-9
    assert float((trace[:, 0] - trace[:, -1]).max()) < 0.0
    assert float((mb.per_ap_power(record["weights"]) - PMAX).max()) <= 1e-9
    assert float((record["v"].abs() - 1.0).abs().max()) < 1e-12


def test_centralized_batch_freezes_each_converged_sample():
    """Batching must not keep optimizing samples after their stopping point."""
    _, ris, direct, served = small_case(seed=10)
    probe = mb.centralized_fp(
        ris, direct, served, PMAX, max_iterations=2, tolerance=0.0, trace=True
    )
    first_improvement = (probe["trace"][:, 1] - probe["trace"][:, 0]).abs()
    tolerance = float(first_improvement.mean())
    assert bool((first_improvement <= tolerance).any())
    assert bool((first_improvement > tolerance).any())

    batched = mb.centralized_fp(
        ris, direct, served, PMAX, max_iterations=8, tolerance=tolerance
    )
    individual = [
        mb.centralized_fp(
            ris[index : index + 1],
            direct[index : index + 1],
            served[index : index + 1],
            PMAX,
            max_iterations=8,
            tolerance=tolerance,
        )
        for index in range(ris.shape[0])
    ]
    torch.testing.assert_close(
        batched["rate"], torch.cat([record["rate"] for record in individual])
    )
    torch.testing.assert_close(
        batched["iterations"],
        torch.cat([record["iterations"] for record in individual]),
    )


def test_admm_reaches_consensus_when_the_objective_is_suppressed():
    """Isolates the consensus machinery from the rate objective.

    With a vanishing power budget the local quadratics vanish, so only the
    augmented-Lagrangian terms drive the copies.  They must agree exactly; if
    they do not, a sign or an incidence entry is wrong.  The schedule here is the
    pre-registered one, a single multiplier update per sweep; under the literal
    per-activation update of eq. (8e) the multiplier picks up increments before
    the first pass completes and then holds the copies apart, which is the
    instability E12 reports rather than a property this test should pin.
    """
    _, ris, direct, served = small_case(seed=8)
    record = mb.incremental_admm(
        ris, direct, served, 1e-30, max_sweeps=20, rho=1.0, init="random",
        dual_period="sweep",
    )
    assert float(record["primal_residual"].max()) < 1e-10
    assert float(record["consensus_residual"].max()) < 1e-10


def test_admm_keeps_every_arm_feasible():
    _, ris, direct, served = small_case(seed=9)
    record = mb.incremental_admm(
        ris, direct, served, PMAX, max_sweeps=10, rho_scale=0.4, dual_period="sweep"
    )
    assert float((mb.per_ap_power(record["weights"]) - PMAX).max()) <= 1e-9
    assert float(record["weights"][~served].abs().max()) == 0.0
    assert float((record["v"].abs() - 1.0).abs().max()) < 1e-12
    # The deployed phase is one physical configuration, not five.
    assert record["v"].shape == (ris.shape[0], ris.shape[3], ris.shape[5])


def test_penalty_scale_supports_single_precision():
    _, ris, direct, served = small_case(seed=11)
    scale = mb.penalty_scale(
        ris.to(torch.complex64), direct.to(torch.complex64), served, PMAX,
        init="matched",
    )
    assert scale.dtype == torch.float32
    assert bool(torch.isfinite(scale).all())
    assert bool((scale > 0).all())


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            case()
            print(f"[ok] {name}")
