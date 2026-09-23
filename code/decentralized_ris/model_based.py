"""Model-based joint active/passive beamforming references for the E12 arm.

Benchmark item 5 of the plan in `doc/research_positioning.md`: the mandatory
model-based family.  Nothing here is learned.  Both solvers optimize the same
unweighted sum rate, on the same channel samples, under the same constraints as
every learned arm: a per-AP power budget, unit-modulus RIS elements, and the
fixed association mask that zeroes the columns an AP does not serve.

Two complete methods share this module:

    centralized_fp      full-CSI fractional-programming BCD at the CPU
    incremental_admm    AP-local CSI, per-AP RIS copies, incremental consensus ADMM

Both follow Huang et al. [7] (`doc/research_positioning.md` reference 7): the
Lagrangian-dual plus quadratic transform of Shen and Yu, its closed-form
auxiliary updates (9) and (10), the per-BS precoder (11) with a bisection on the
power multiplier, and the MM reflection update (15)/(17).  `incremental_admm` is
their Algorithm 1; `centralized_fp` is the centralized counterpart of the same
formulation, so the two arms differ in information and coordination structure
and not in their surrogate, solver, or constraint handling.  The choice of
source, and why the pre-registered first choice was not usable, is recorded in
the E12 report.

Notation.  `rates` pairs the stored channel with the beamformer as h^T w, so the
conjugated channel g = conj(h) is the one satisfying the usual g^H w convention.
Writing the RIS variable in the conjugate domain as well, v_r = conj(theta_r),
makes the effective channel *affine* in the optimization variable:

    g_{l,k}(v) = d_{l,k} + sum_r A_{l,r,k} v_r,      d_{l,k} = conj(h^d_{l,k}),

with A_{l,r,k} in C^{M x N} the stored AP-RIS-user cascade.  `to_theta` maps v
back to the (cos, sin) layout `rates` and `quantize_phase` expect.  Every
formula below is written in terms of g, v, and the aggregate

    a_{k,j} = sum_l g_{l,k}^H w_{l,j}  =  phi_{k,j} + psi_{k,j},

which is the only quantity that couples APs.  That is what makes the ADMM arm
implementable with AP-local CSI: an AP needs its own channels plus these K^2
scalars, never another AP's channel matrices.
"""

import numpy as np
import torch

from rates import RatePrecompute


SIGMA = RatePrecompute.SIGMA
EPS = 1e-12


def channel_tensors(simulator, device, dtype=torch.complex64):
    """Stored simulator channels -> the complex form both solvers optimize.

    Returns `(ris, direct, served)` with shapes (B, L, K, R, M, N), (B, L, K, M)
    and (B, L, K).  The association mask is read from the simulator's own
    stations, so it is bit-identical to the mask every learned arm receives.
    """
    channels, direct = simulator.all_channels()
    batch, n_antennas, n_elements, n_ris, k_total = channels.shape
    n_ap = simulator.n_ap
    k_user = k_total // n_ap

    ris = torch.as_tensor(channels, dtype=dtype, device=device)
    ris = ris.reshape(batch, n_antennas, n_elements, n_ris, n_ap, k_user)
    ris = ris.permute(0, 4, 5, 3, 1, 2).contiguous()
    gain = torch.as_tensor(direct, dtype=dtype, device=device).conj()
    gain = gain.reshape(batch, n_ap, k_user, n_antennas).contiguous()
    served = np.stack(
        [station.user_mask for station in simulator.base_stations], axis=1
    )
    served = torch.as_tensor(served, dtype=torch.bool, device=device)
    return ris, gain, served


def to_theta(v):
    """(B, R, N) conjugate-domain variable -> the stored (cos, sin) pair.

    theta = conj(v), so the stored real part is Re(v) and the stored imaginary
    part is -Im(v).  `rates.quantize_phase` then rounds this arm on exactly the
    grid the learned arms are rounded on.
    """
    return torch.stack((v.real, -v.imag), dim=-1).to(torch.float32)


def from_theta(theta):
    """Stored (cos, sin) pair -> the conjugate-domain variable v = conj(theta)."""
    return torch.complex(theta[..., 0], -theta[..., 1])


def embed_beamformer(weights):
    """(B, L, K, M) complex precoders -> the (B, 2M, L*K) layout `rates` wants."""
    batch, n_ap, k_user, n_antennas = weights.shape
    real = weights.real.reshape(batch, n_ap * k_user, n_antennas)
    imag = weights.imag.reshape(batch, n_ap * k_user, n_antennas)
    return torch.cat((real, imag), dim=2).transpose(2, 1).contiguous().to(torch.float32)


def effective_channels(ris, direct, v):
    """g_{l,k}(v) = d_{l,k} + sum_r A_{l,r,k} v_r."""
    return direct + torch.einsum("blkrmn,brn->blkm", ris, v)


def contributions(gain, weights):
    """Per-AP aggregate blocks c_{l,k,j} = g_{l,k}^H w_{l,j}."""
    return torch.einsum("blkm,bljm->blkj", gain.conj(), weights)


def aggregate(gain, weights):
    """a_{k,j} = sum_l g_{l,k}^H w_{l,j}; the only inter-AP coupling term."""
    return torch.einsum("blkm,bljm->bkj", gain.conj(), weights)


def sum_rate_from_aggregate(a, sigma=SIGMA):
    """Unweighted sum rate in bps/Hz, from the K x K aggregate matrix."""
    power = a.abs().pow(2)
    signal = torch.diagonal(power, dim1=1, dim2=2)
    interference = power.sum(dim=2) - signal
    return torch.log2(1.0 + signal / (interference + sigma)).sum(dim=1)


def sum_rate(gain, weights, sigma=SIGMA):
    return sum_rate_from_aggregate(aggregate(gain, weights), sigma)


def fp_auxiliaries(a, sigma=SIGMA):
    """Closed-form auxiliary updates of the quadratic transform, omega_k = 1.

    gamma_k is the current SINR (Huang et al. eq. 9) and

        xi_k = sqrt(1 + gamma_k) a_{k,k} / (sum_j |a_{k,j}|^2 + sigma^2)

    is eq. (10).  These are the same quantities a WMMSE pass would call
    alpha_k = 1 + gamma_k and nu_k = a_{k,k} / (sum_j |a_{k,j}|^2 + sigma^2),
    through |xi_k|^2 = alpha_k |nu_k|^2 and sqrt(1 + gamma_k) xi_k = alpha_k nu_k;
    `tests/test_model_based_joint.py` pins that identity so the two readings of
    the surrogate cannot drift apart.
    """
    total = a.abs().pow(2).sum(dim=2) + sigma
    diagonal = torch.diagonal(a, dim1=1, dim2=2)
    signal = diagonal.abs().pow(2)
    gamma = signal / (total - signal).clamp(min=EPS)
    xi = (1.0 + gamma).sqrt().to(a.dtype) * diagonal / total.to(a.dtype)
    return gamma, xi


def surrogate_terms(gamma, xi):
    """(|xi_k|^2, sqrt(1 + gamma_k) xi_k): the only way gamma and xi enter."""
    return xi.abs().pow(2).to(xi.dtype), (1.0 + gamma).sqrt().to(xi.dtype) * xi


def surrogate_value(a, gamma, xi, sigma=SIGMA):
    """The quadratic-transform objective actually being maximized, omega_k = 1.

    f = sum_k [ log2(1+gamma_k) - gamma_k / ln 2
                + 2 sqrt(1+gamma_k) Re{xi_k^* a_{k,k}} / ln 2
                - |xi_k|^2 (sum_j |a_{k,j}|^2 + sigma^2) / ln 2 ]

    Reported in bits so that it is directly comparable with the sum rate it
    lower-bounds; at the optimal gamma and xi the two coincide.
    """
    coefficient, scale = surrogate_terms(gamma, xi)
    diagonal = torch.diagonal(a, dim1=1, dim2=2)
    quadratic = coefficient.real * (a.abs().pow(2).sum(dim=2) + sigma)
    linear = 2.0 * (scale.conj() * diagonal).real
    return (
        torch.log2(1.0 + gamma)
        + (linear - quadratic - gamma) / np.log(2.0)
    ).sum(dim=1)


def per_ap_power(weights):
    """||W_l||_F^2 for (B, L, K, M) complex precoders."""
    return weights.abs().pow(2).sum(dim=(2, 3))


def mrt_initial(gain, served, pmax):
    """Deterministic masked full-power MRT start, shared by both solvers."""
    weights = gain * served.unsqueeze(-1).to(gain.dtype)
    norm = weights.abs().pow(2).sum(dim=(2, 3), keepdim=True).sqrt()
    scale = (norm > 0).to(norm.dtype) * np.sqrt(pmax) / norm.clamp(min=EPS)
    return weights * scale.to(gain.dtype)


def _solve_power_constrained(system, rhs, served, pmax, bisect_iters):
    """Huang et al. eq. (11): (Phi_l + mu_l I)^-1 (rhs), with mu_l by bisection.

    `system` is the Hermitian M x M matrix Phi_l = sum_k |xi_k|^2 g_{l,k} g_{l,k}^H
    and `rhs` the (B, K, M) right-hand side.  Because M is tiny, one
    eigendecomposition turns the multiplier search into scalar arithmetic: with
    Phi_l = U diag(lam) U^H and p_j = U^H rhs_j the block power is
    sum_n E_n / (lam_n + mu)^2, decreasing in mu, so the bisection needs no
    further matrix solves.  The bracket mu_hi = sqrt(sum_n E_n / P) is valid
    because the power is at most (sum_n E_n) / mu^2.  mu_l = 0 is kept whenever
    the unconstrained block already fits the budget, which is the KKT
    complementary-slackness case the paper's "= P_b" phrasing skips over.
    """
    mask = served.to(rhs.dtype)
    rhs = rhs * mask.unsqueeze(-1)
    lam, basis = torch.linalg.eigh(system)
    lam = lam.clamp(min=0.0)
    projected = torch.einsum("bmn,bjm->bjn", basis.conj(), rhs)
    energy = projected.abs().pow(2).sum(dim=1)

    def block_power(mu):
        return (energy / (lam + mu.unsqueeze(-1)).clamp(min=EPS).pow(2)).sum(dim=1)

    zero = torch.zeros(lam.shape[0], device=lam.device, dtype=lam.dtype)
    unconstrained = block_power(zero)
    high = (energy.sum(dim=1) / pmax).clamp(min=EPS).sqrt()
    low = zero
    for _ in range(bisect_iters):
        middle = 0.5 * (low + high)
        over = block_power(middle) > pmax
        low = torch.where(over, middle, low)
        high = torch.where(over, high, middle)
    # `high` is the feasible endpoint; the midpoint can remain marginally over
    # budget when the power curve is steep and the iteration count is finite.
    mu = torch.where(unconstrained > pmax, high, zero)

    shifted = (lam + mu.unsqueeze(-1)).clamp(min=EPS).unsqueeze(1)
    weights = torch.einsum("bmn,bjn->bjm", basis, projected / shifted.to(rhs.dtype))
    return weights * mask.unsqueeze(-1)


def beamformer_step(
    gain, weights, gamma, xi, served, pmax, cycles, bisect_iters, rest=None
):
    """Huang et al. eq. (11) applied to each AP block in a fixed order.

    The stationarity condition of the surrogate for AP l is

        (Phi_l + mu_l I) w_{l,j} = sqrt(1+gamma_j) xi_j g_{l,j} - Omega_{l,j},
        Omega_{l,j} = sum_k |xi_k|^2 g_{l,k} (a_{k,j} - g_{l,k}^H w_{l,j}),

    so each block is minimized exactly with the others held fixed.  The only
    quantity taken from APs outside `gain`/`weights` is `rest`, the aggregate
    they contribute: zero for the centralized arm, the peers' published
    phi + psi for an ADMM node.  The masked columns are zeroed before the
    multiplier bisection, so the per-AP budget is spent only on served users.
    """
    coefficient, scale = surrogate_terms(gamma, xi)
    a = aggregate(gain, weights)
    if rest is not None:
        a = a + rest
    weights = weights.clone()
    for _ in range(cycles):
        for index in range(gain.shape[1]):
            local = gain[:, index]
            own = torch.einsum("bkm,bjm->bkj", local.conj(), weights[:, index])
            others = a - own
            system = torch.einsum("bk,bkm,bkn->bmn", coefficient, local, local.conj())
            rhs = scale.unsqueeze(-1) * local
            rhs = rhs - torch.einsum("bk,bkm,bkj->bjm", coefficient, local, others)
            block = _solve_power_constrained(
                system, rhs, served[:, index], pmax, bisect_iters
            )
            weights[:, index] = block
            a = others + torch.einsum("bkm,bjm->bkj", local.conj(), block)
    return weights


def phase_quadratic(ris, direct, weights, gamma, xi, rest=None):
    """Huang et al. eq. (12)-(13): the surrogate as v^H Z v + 2 Re(v^H q).

    With a_{k,j} = base_{k,j} + v^H t_{k,j}, where t_{k,j} collects the cascade
    terms of the APs represented here and `base` the direct-link terms, the
    surrogate is quadratic in v:

        Z = sum_{k,j} |xi_k|^2 t_{k,j} t_{k,j}^H,
        q = sum_{k,j} |xi_k|^2 t_{k,j} base_{k,j}^* - sum_k sqrt(1+gamma_k) xi_k^* t_{k,k}.

    `rest` carries the aggregate of the APs that are *not* represented: zero for
    the centralized arm, the peers' published aggregate for an ADMM node.  The
    printed q in Huang et al. drops a conjugation and the AP's own direct-link
    term; the form above is the corrected one and matches Xu et al. eq. (31)
    after the sign convention is aligned.
    """
    coefficient, scale = surrogate_terms(gamma, xi)
    linear = torch.einsum("blkrmn,bljm->bkjrn", ris.conj(), weights)
    batch, k_user = linear.shape[0], linear.shape[1]
    linear = linear.reshape(batch, k_user, k_user, -1)
    base = torch.einsum("blkm,bljm->bkj", direct.conj(), weights)
    if rest is not None:
        base = base + rest

    weighted = linear * coefficient.sqrt()[:, :, None, None]
    weighted = weighted.reshape(batch, k_user * k_user, -1)
    system = torch.einsum("bap,baq->bpq", weighted, weighted.conj())

    cross = torch.einsum("bk,bkjp,bkj->bp", coefficient, linear, base.conj())
    index = torch.arange(k_user, device=linear.device)
    served_term = torch.einsum("bk,bkp->bp", scale.conj(), linear[:, index, index])
    return system, cross - served_term


def curvature_bound(system, exact=False):
    """An upper bound on lambda_max(Z), which is all the MM majorizer needs.

    For a Hermitian matrix the induced infinity norm bounds the spectral radius,
    so max_p sum_q |Z_pq| >= lambda_max.  On this problem it runs about 1.4x
    loose, which costs a proportional number of MM iterations at 0.09 ms each,
    against 400 ms for a batched eigendecomposition of the R*N x R*N matrix.
    `exact=True` restores lambda_max itself and exists so the two can be compared
    on a subset.
    """
    if exact:
        return torch.linalg.eigvalsh(system)[:, -1]
    return system.abs().sum(dim=2).amax(dim=1)


def mm_phase_step(system, cross, v, iterations, curvature=None, exact=False):
    """Huang et al. eq. (15)/(17): majorize-minimize on the unit-modulus set.

    Majorizing v^H Z v at v^t with any zeta >= lambda_max(Z) leaves a purely
    linear surrogate, whose unit-modulus minimizer is closed form:

        v <- -exp(j arg((Z - zeta I) v^t + q)).

    Each step is non-increasing in the true quadratic, so the reflection block
    never undoes the beamformer block.  The shifted matrix is never materialized:
    (Z - zeta I) v is one matrix-vector product and one scaling.
    """
    if curvature is None:
        curvature = curvature_bound(system, exact)
    shift = curvature.to(v.dtype).unsqueeze(1)
    for _ in range(iterations):
        target = torch.einsum("bpq,bq->bp", system, v) - shift * v + cross
        magnitude = target.abs()
        v = torch.where(
            magnitude > EPS,
            -target / magnitude.clamp(min=EPS).to(target.dtype),
            torch.ones_like(target),
        )
    return v


def cd_phase_step(system, cross, v, sweeps):
    """Cyclic unit-modulus coordinate descent; a control on the MM solver.

    Holding every other element fixed, element n only sees 2 Re(v_n^* (c_n + q_n))
    with c_n = sum_{m != n} Z_{nm} v_m, so its exact minimizer is
    v_n = -(c_n + q_n) / |c_n + q_n|.  This is not the published solver; it
    exists so that a second monotone method can be run on the same subproblem
    and confirm that the reported phase is not an artefact of MM's majorizer.
    """
    v = v.clone()
    product = torch.einsum("bpq,bq->bp", system, v)
    diagonal = torch.diagonal(system, dim1=1, dim2=2)
    for _ in range(sweeps):
        for index in range(v.shape[1]):
            partial = product[:, index] - diagonal[:, index] * v[:, index]
            target = partial + cross[:, index]
            magnitude = target.abs()
            updated = torch.where(
                magnitude > EPS,
                -target / magnitude.clamp(min=EPS).to(target.dtype),
                torch.ones_like(target),
            )
            product = product + system[:, :, index] * (updated - v[:, index]).unsqueeze(1)
            v[:, index] = updated
    return v


def quadratic_value(system, cross, v):
    """v^H Z v + 2 Re(v^H q); the objective both phase solvers decrease."""
    quadratic = torch.einsum("bp,bpq,bq->b", v.conj(), system, v).real
    return quadratic + 2.0 * torch.einsum("bp,bp->b", v.conj(), cross).real


def initial_phase(ris, direct, served, mode):
    """Deterministic RIS starts.  `ones` is the pre-registered primary choice."""
    batch, n_ris, n_elements = ris.shape[0], ris.shape[3], ris.shape[5]
    if mode == "ones":
        return torch.ones(
            (batch, n_ris, n_elements), dtype=ris.dtype, device=ris.device
        )
    if mode == "random":
        generator = torch.Generator(device="cpu").manual_seed(0)
        angles = 2 * np.pi * torch.rand(
            (batch, n_ris, n_elements), generator=generator
        )
        angles = angles.to(ris.device)
        return torch.complex(angles.cos(), angles.sin()).to(ris.dtype)
    if mode == "matched":
        # Align every element with the coherent sum of the served cascade-direct
        # products, i.e. the phase that maximizes the RIS path's alignment with
        # the direct link.  Deterministic and channel-dependent, unlike `ones`.
        masked = ris * served[:, :, :, None, None, None].to(ris.dtype)
        response = torch.einsum("blkrmn,blkm->brn", masked, direct.conj())
        magnitude = response.abs()
        return torch.where(
            magnitude > EPS,
            response.conj() / magnitude.clamp(min=EPS).to(response.dtype),
            torch.ones_like(response),
        )
    raise ValueError(f"unknown initialization {mode}")


def ring_incidence(n_ap):
    """Signed edge incidence of the ring used for the consensus constraint.

    Huang et al. write the consensus as t = sum_l A_l v_l = 0 over an undirected
    graph they leave unspecified.  The ring e_l = (l, l+1 mod L) is
    pre-registered here: it is connected, it matches the fixed activation order
    of their Algorithm 1, and it is the cheapest encoding, |E| = L, so the
    signaling ledger is not inflated by the choice.  Returns, per AP, the list of
    (edge, sign) pairs, with A_l^H A_l = 2 I for every AP.
    """
    edges = [(index, (index + 1) % n_ap) for index in range(n_ap)]
    incidence = [[] for _ in range(n_ap)]
    for edge, (head, tail) in enumerate(edges):
        incidence[head].append((edge, 1.0))
        incidence[tail].append((edge, -1.0))
    return edges, incidence


def centralized_fp(
    ris,
    direct,
    served,
    pmax,
    max_iterations=60,
    tolerance=1e-4,
    inner_cycles=2,
    mm_iterations=30,
    bisect_iters=30,
    init="ones",
    sigma=SIGMA,
    phase_solver="mm",
    exact_curvature=False,
    trace=False,
):
    """Centralized counterpart of Huang et al.'s formulation, with full CSI.

    One iteration refreshes (gamma, xi), minimizes the surrogate over the AP
    beamformers under the per-AP power budget, and runs the MM reflection update
    on the single global phase vector.  Both blocks are non-increasing on the
    same surrogate, so the achieved sum rate is monotone;
    `monotonicity_violation` reports the largest observed decrease rather than
    assuming there is none.  This is a local-optimization reference, not a
    global upper bound.
    """
    batch, n_ris, n_elements = ris.shape[0], ris.shape[3], ris.shape[5]
    v = initial_phase(ris, direct, served, init)
    gain = effective_channels(ris, direct, v)
    weights = mrt_initial(gain, served, pmax)

    history = [sum_rate(gain, weights, sigma)]
    violation = 0.0
    quadratic_violation = 0.0
    iterations = torch.zeros(batch, dtype=torch.long, device=ris.device)
    active = torch.ones(batch, dtype=torch.bool, device=ris.device)
    for _ in range(max_iterations):
        previous_weights = weights
        previous_v = v
        gamma, xi = fp_auxiliaries(aggregate(gain, weights), sigma)
        candidate_weights = beamformer_step(
            gain, weights, gamma, xi, served, pmax, inner_cycles, bisect_iters
        )

        system, cross = phase_quadratic(ris, direct, candidate_weights, gamma, xi)
        flat = v.reshape(batch, -1)
        before = quadratic_value(system, cross, flat)
        if phase_solver == "mm":
            flat = mm_phase_step(
                system, cross, flat, mm_iterations, exact=exact_curvature
            )
        else:
            flat = cd_phase_step(system, cross, flat, mm_iterations)
        phase_change = quadratic_value(system, cross, flat) - before
        quadratic_violation = max(
            quadratic_violation, float(phase_change[active].max())
        )
        candidate_v = flat.reshape(batch, n_ris, n_elements)
        weights = torch.where(
            active[:, None, None, None], candidate_weights, previous_weights
        )
        v = torch.where(active[:, None, None], candidate_v, previous_v)
        gain = effective_channels(ris, direct, v)

        rate = sum_rate(gain, weights, sigma)
        violation = max(violation, float((history[-1] - rate).max()))
        improvement = rate - history[-1]
        history.append(rate)
        iterations = iterations + active.to(iterations.dtype)
        active = active & (improvement.abs() > tolerance)
        if not bool(active.any()):
            break

    result = {
        "v": v,
        "weights": weights,
        "iterations": iterations,
        "converged": ~active,
        "monotonicity_violation": violation,
        "phase_block_violation": quadratic_violation,
        "rate": history[-1],
    }
    if trace:
        result["trace"] = torch.stack(history, dim=1)
    return result


def resolve_beamformer(
    ris, direct, served, v, pmax, rounds=8, inner_cycles=2, bisect_iters=30, sigma=SIGMA
):
    """Re-run only the beamformer block at a fixed RIS phase.

    Used for the rounded-phase diagnostic.  The maintained 2-bit rule keeps the
    beamformer and only quantizes the phase, so this is reported separately and
    never as the primary 2-bit number.
    """
    gain = effective_channels(ris, direct, v)
    weights = mrt_initial(gain, served, pmax)
    for _ in range(rounds):
        gamma, xi = fp_auxiliaries(aggregate(gain, weights), sigma)
        weights = beamformer_step(
            gain, weights, gamma, xi, served, pmax, inner_cycles, bisect_iters
        )
    return weights


def penalty_scale(ris, direct, served, pmax, init="ones", sigma=SIGMA):
    """Curvature of the AP-local RIS subproblem at the initial point.

    Huang et al. leave the penalty rho unspecified, and a raw constant is not
    transferable: the surrogate's curvature carries the channel scaling of this
    simulator.  The median over APs of lambda_max(Z_l) at the shared
    initialization gives a per-sample, deterministic, scale-free reference that
    the pre-registered rho is expressed as a multiple of.
    """
    v = initial_phase(ris, direct, served, init)
    gain = effective_channels(ris, direct, v)
    weights = mrt_initial(gain, served, pmax)
    gamma, xi = fp_auxiliaries(aggregate(gain, weights), sigma)
    curvature = []
    for index in range(served.shape[1]):
        system, _ = phase_quadratic(
            ris[:, index : index + 1],
            direct[:, index : index + 1],
            weights[:, index : index + 1],
            gamma,
            xi,
        )
        # This scale is not part of the optimized state.  Solve it in double
        # precision so the float32 control does not fail on nearly repeated
        # eigenvalues, then return it in the caller's real dtype.
        curvature.append(
            torch.linalg.eigvalsh(system.to(torch.complex128))[:, -1].to(ris.real.dtype)
        )
    return torch.stack(curvature, dim=1).median(dim=1).values


def incremental_admm(
    ris,
    direct,
    served,
    pmax,
    max_sweeps=60,
    tolerance=1e-4,
    rho=None,
    rho_scale=0.05,
    rho_growth=1.0,
    rho_cap=1e3,
    inner_cycles=1,
    mm_iterations=30,
    bisect_iters=30,
    init="ones",
    sigma=SIGMA,
    phase_solver="mm",
    dual_period="sweep",
    exact_curvature=False,
    trace=False,
):
    """Huang et al. Algorithm 1: incremental consensus ADMM over AP-local copies.

    AP l holds its own precoder W_l and its own copy v_l of the stacked
    R*N reflection vector; the copies are tied by the edge constraint
    t = sum_l A_l v_l = 0 on a ring.  Exactly one AP is active per activation:
    it removes its stale contribution from the running aggregates (phi, psi, t),
    refreshes (gamma, xi), updates W_l by eq. (11), updates v_l by the MM step
    on its local quadratic plus the augmented-Lagrangian terms

        Z <- Z + rho I,       q <- q + (rho/2) A_l^H (t_l + lambda / rho),

    updates the multiplier by lambda <- lambda + rho t, re-adds its fresh
    contribution, and passes the token on.  Under unit modulus the rho ||v_l||^2
    term is constant, so the penalty enters the local subproblem only through its
    linear part.  One sweep is L activations, which is the unit reported here so
    that the round count is comparable with a parallel scheme.

    The deployed phase is the unit-modulus projection of the mean copy; the RIS
    has one physical configuration, so the reported rate is the one a receiver
    would see, not any AP's local prediction.  `consensus_residual` reports how
    far the copies actually agree, which is what decides whether that projection
    is meaningful.
    """
    batch, n_ap, k_user = served.shape
    n_ris, n_elements = ris.shape[3], ris.shape[5]
    stacked = n_ris * n_elements
    edges, incidence = ring_incidence(n_ap)

    reference = penalty_scale(ris, direct, served, pmax, init, sigma)
    if rho is None:
        penalty = rho_scale * reference
    else:
        penalty = torch.full_like(reference, float(rho))
    ceiling = rho_cap * reference

    copies = initial_phase(ris, direct, served, init).reshape(batch, 1, stacked)
    copies = copies.repeat(1, n_ap, 1).contiguous()
    multipliers = torch.zeros(
        (batch, len(edges), stacked), dtype=ris.dtype, device=ris.device
    )
    identity = torch.eye(stacked, dtype=ris.dtype, device=ris.device)

    gain = effective_channels(
        ris, direct, copies[:, 0].reshape(batch, n_ris, n_elements)
    )
    weights = mrt_initial(gain, served, pmax)
    blocks = contributions(gain, weights)

    def residual():
        """t = sum_l A_l v_l, the per-edge consensus residual."""
        return torch.stack(
            [copies[:, head] - copies[:, tail] for head, tail in edges], dim=1
        )

    def deployed():
        mean = copies.mean(dim=1)
        magnitude = mean.abs()
        projected = torch.where(
            magnitude > EPS,
            mean / magnitude.clamp(min=EPS).to(mean.dtype),
            torch.ones_like(mean),
        )
        return projected.reshape(batch, n_ris, n_elements)

    history = []
    residuals = []
    primal = torch.zeros(batch, device=ris.device, dtype=ris.real.dtype)
    dual = torch.zeros_like(primal)
    sweeps = torch.zeros(batch, dtype=torch.long, device=ris.device)
    active = torch.ones(batch, dtype=torch.bool, device=ris.device)
    activations = 0
    for _ in range(max_sweeps):
        previous = copies.mean(dim=1)
        step = penalty[:, None, None].to(multipliers.dtype)
        for index in range(n_ap):
            # AP `index` sees its own channels, its own copy, and the K^2
            # aggregate its peers published; never another AP's channel matrix.
            rest = blocks.sum(dim=1) - blocks[:, index]
            own_ris = ris[:, index : index + 1]
            own_direct = direct[:, index : index + 1]
            own_served = served[:, index : index + 1]
            own_v = copies[:, index].reshape(batch, n_ris, n_elements)
            own_gain = effective_channels(own_ris, own_direct, own_v)
            own_weights = weights[:, index : index + 1]

            gamma, xi = fp_auxiliaries(rest + aggregate(own_gain, own_weights), sigma)
            own_weights = beamformer_step(
                own_gain, own_weights, gamma, xi, own_served, pmax,
                inner_cycles, bisect_iters, rest=rest,
            )

            system, cross = phase_quadratic(
                own_ris, own_direct, own_weights, gamma, xi, rest=rest
            )
            current = residual()
            column = penalty[:, None].to(system.dtype)
            for edge, sign in incidence[index]:
                # t_l on this edge excludes AP `index`'s own contribution, which
                # is what expanding ||A_l v_l + t_l + lambda / rho||^2 leaves.
                peer = current[:, edge] - sign * copies[:, index]
                cross = cross + 0.5 * sign * (
                    column * peer + multipliers[:, edge]
                )
            system = system + column[:, :, None] * identity
            if phase_solver == "mm":
                updated = mm_phase_step(
                    system, cross, copies[:, index], mm_iterations,
                    exact=exact_curvature,
                )
            else:
                updated = cd_phase_step(system, cross, copies[:, index], mm_iterations)

            updated = torch.where(active[:, None], updated, copies[:, index])
            own_weights = torch.where(
                active[:, None, None, None], own_weights, weights[:, index : index + 1]
            )
            copies[:, index] = updated
            weights[:, index] = own_weights[:, 0]
            if dual_period == "activation":
                multipliers = multipliers + active[:, None, None] * step * residual()
            own_gain = effective_channels(
                own_ris, own_direct, updated.reshape(batch, n_ris, n_elements)
            )
            blocks[:, index] = contributions(own_gain, own_weights)[:, 0]
            activations += 1

        current = residual()
        if dual_period == "sweep":
            multipliers = multipliers + active[:, None, None] * step * current
        primal = current.abs().pow(2).sum(dim=(1, 2)).sqrt()
        dual = penalty * np.sqrt(n_ap) * (
            copies.mean(dim=1) - previous
        ).abs().pow(2).sum(dim=1).sqrt()
        penalty = torch.where(
            active, torch.minimum(penalty * rho_growth, ceiling), penalty
        )

        phase = deployed()
        history.append(sum_rate(effective_channels(ris, direct, phase), weights, sigma))
        residuals.append(primal)
        sweeps = sweeps + active.to(sweeps.dtype)
        if len(history) > 1:
            active = active & (
                ((history[-1] - history[-2]).abs() > tolerance) | (primal > tolerance)
            )
        if not bool(active.any()):
            break

    phase = deployed()
    gain = effective_channels(ris, direct, phase)
    mean = copies.mean(dim=1, keepdim=True)
    return {
        "v": phase,
        "weights": weights,
        "copies": copies,
        "sweeps": sweeps,
        "activations": activations,
        "converged": ~active,
        "primal_residual": primal,
        "dual_residual": dual,
        "consensus_residual": (copies - mean).abs().amax(dim=(1, 2)),
        "penalty": penalty,
        "penalty_reference": reference,
        "rate": sum_rate(gain, weights, sigma),
        **(
            {
                "trace": torch.stack(history, dim=1),
                "primal_trace": torch.stack(residuals, dim=1),
            }
            if trace
            else {}
        ),
    }
