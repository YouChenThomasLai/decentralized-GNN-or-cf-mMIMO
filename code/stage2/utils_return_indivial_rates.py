import numpy as np
import torch
import torch.nn.functional as F


DIRECT_CHANNEL_SCALE = -7
DIRECT_CHANNEL_FADING = 10 ** (-4.5)
DIRECT_PATH_LOSS_EXPONENT = 3.5
NOISE_POWER = 1e-12
SPEED_OF_LIGHT_MPS = 3e8


def gen_location(K, l):
    center = np.array([0, 0])
    locations = np.zeros((K, 2))
    for k in range(K):
        theta = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(0, l)
        x = length * np.cos(theta)
        y = length * np.sin(theta)
        locations[k, :] = center + np.array([x, y])

    return locations


def gen_fixed_location(K, l):
    center = np.array([0, 0])
    locations = np.zeros((K, 2))
    theta = 2 * np.pi / K
    for k in range(K):
        x = l * np.cos(k * theta)
        y = l * np.sin(k * theta)
        locations[k, :] = center + np.array([x, y])

    return locations


def gen_BS_location(K, l):
    center = np.array([0, 0])
    locations = np.zeros((K, 2))
    for k in range(K):
        theta = np.random.uniform(0, 2 * np.pi)
        x = l * np.cos(theta)
        y = l * np.sin(theta)
        locations[k, :] = center + np.array([x, y])
    return locations


def element_wise_mean(x):
    return torch.mean(x, dim=1)


def element_wise_max(x):
    return torch.amax(x, dim=1)


def im2re(M):
    M1 = np.concatenate((M.real, -M.imag), axis=2)
    M2 = np.concatenate((M.imag, M.real), axis=2)
    M_mat = np.concatenate((M1, M2), axis=1)
    return torch.Tensor(M_mat)


def gen_LOS(num_rev, num_trans, Rician_factor, L):
    LOS_array = []
    for _ in range(L):
        AoA = np.ones((num_rev, 1), dtype=np.complex128)
        AoD = np.ones((num_trans, 1), dtype=np.complex128)
        angle_AoA = 2 * np.pi * np.random.uniform(0, 1)
        for n in range(1, num_rev):
            AoA[n, :] = np.exp(1j * n * np.pi * np.sin(angle_AoA))
        angle_AoD = 2 * np.pi * np.random.uniform(0, 1)
        for n in range(1, num_trans):
            AoD[n, :] = np.exp(1j * n * np.pi * np.sin(angle_AoD))
        mat_LOS = np.dot(AoA, np.conj(AoD.T))
        LOS = np.sqrt(Rician_factor / (Rician_factor + 1)) * mat_LOS
        if L == 1:
            return LOS
        LOS_array.append(LOS)

    return LOS_array


class Channel:
    def __init__(self, num_trans, num_rev, factor):
        self.num_trans = num_trans
        self.num_rev = num_rev
        self.factor = factor
        self.AoA = np.ones((num_rev, 1), dtype=np.complex128)
        self.AoD = np.ones((num_trans, 1), dtype=np.complex128)
        self.mat = np.zeros((num_rev, num_trans), dtype=np.complex128)
        self.NLOS = np.zeros((num_rev, num_trans), dtype=np.complex128)
        self.LOS = np.zeros((num_rev, num_trans), dtype=np.complex128)

    def generate_value(self, mean_NLOS, cov_NLOS, LOS):
        self.LOS = LOS
        R_NLOS = np.linalg.cholesky(cov_NLOS)
        mat_real_NLOS = (
            np.ones((self.num_rev, self.num_trans)) * mean_NLOS
            + np.matmul(np.random.randn(self.num_rev, self.num_trans), R_NLOS)
        )
        mat_imag_NLOS = (
            np.ones((self.num_rev, self.num_trans)) * mean_NLOS
            + np.matmul(np.random.randn(self.num_rev, self.num_trans), R_NLOS)
        )
        self.NLOS = (
            np.sqrt(1 / (self.factor + 1))
            * (mat_real_NLOS + 1j * mat_imag_NLOS)
            / np.sqrt(2)
        )
        self.mat = self.NLOS + self.LOS

    def large_scale_loss(
        self, fading_NLOS, exp_NLOS, fading_LOS, exp_LOS, dist
    ):
        self.large_scale_fading_NLOS = fading_NLOS * dist ** (-exp_NLOS)
        self.large_scale_fading_LOS = fading_LOS * dist ** (-exp_LOS)
        self.ori_mat = (
            self.large_scale_fading_NLOS * self.NLOS
            + self.large_scale_fading_LOS * self.LOS
        )
        self.mat = self.ori_mat
        return self.mat


def generate_channel(M, K, batch_size, loc_BS, loc_user):
    channel_bs_user = []

    for _ in range(batch_size):
        sample_channels = []
        for k in range(K):
            h_LOS = gen_LOS(1, M, 0, 1)
            h_bs = Channel(M, 1, 0)
            h_bs.generate_value(0, np.eye(M), h_LOS)
            h_bs = h_bs.large_scale_loss(
                DIRECT_CHANNEL_FADING,
                DIRECT_PATH_LOSS_EXPONENT,
                0,
                0,
                np.linalg.norm(loc_BS - loc_user[k, :]),
            )
            sample_channels.append(h_bs[0])
        channel_bs_user.append(sample_channels)

    return np.asarray(channel_bs_user) / 10 ** DIRECT_CHANNEL_SCALE


def jakes_correlation(
    speed_mps,
    carrier_frequency_hz=2.6e9,
    decision_period_s=0.001,
):
    """Return the one-period Jakes coefficient for one or more UE speeds."""
    speed_mps = np.asarray(speed_mps, dtype=np.float64)
    doppler_hz = carrier_frequency_hz * speed_mps / SPEED_OF_LIGHT_MPS
    arguments = 2 * np.pi * doppler_hz * decision_period_s
    values = torch.special.bessel_j0(
        torch.as_tensor(arguments, dtype=torch.float64)
    ).cpu().numpy()
    return float(values) if values.ndim == 0 else values


def complex_normal(shape, rng=np.random):
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) / np.sqrt(2)


def distance_and_path_loss(ue_positions, bs_locations):
    """Return distances and Stage 1 amplitude factors over a trajectory."""
    ue_positions = np.asarray(ue_positions, dtype=np.float64)
    bs_locations = np.asarray(bs_locations, dtype=np.float64)
    offsets = (
        bs_locations[None, None, :, None, :]
        - ue_positions[:, :, None, :, :]
    )
    distances = np.linalg.norm(offsets, axis=-1)
    if np.any(distances == 0):
        raise ValueError("An AP and UE cannot occupy the same position")
    path_loss = (
        DIRECT_CHANNEL_FADING
        * distances ** (-DIRECT_PATH_LOSS_EXPONENT)
        / 10 ** DIRECT_CHANNEL_SCALE
    )
    return distances, path_loss


def generate_temporal_channels(
    initial_normalized_channels,
    ue_positions,
    bs_locations,
    ue_speeds_mps,
    carrier_frequency_hz=2.6e9,
    decision_period_s=0.001,
    rng=np.random,
):
    """Generate stationary first-order Gauss-Markov small-scale fading."""
    initial = np.asarray(initial_normalized_channels, dtype=np.complex128)
    positions = np.asarray(ue_positions, dtype=np.float64)
    speeds = np.asarray(ue_speeds_mps, dtype=np.float64)
    batch_size, episode_steps, num_users = positions.shape[:3]
    if initial.shape[0] != batch_size or initial.shape[2] != num_users:
        raise ValueError("Initial channels do not match trajectory dimensions")
    if speeds.shape != (batch_size, num_users):
        raise ValueError("ue_speeds_mps must have shape [B, K]")

    rhos = jakes_correlation(
        speeds, carrier_frequency_hz, decision_period_s
    )
    if np.any(np.abs(rhos) > 1):
        raise ValueError("Jakes coefficients must lie in [-1, 1]")

    normalized = np.empty(
        (batch_size, episode_steps, *initial.shape[1:]),
        dtype=np.complex128,
    )
    normalized[:, 0] = initial
    rho = rhos[:, None, :, None]
    innovation_scale = np.sqrt(np.maximum(0.0, 1 - rho**2))
    for period in range(1, episode_steps):
        innovation = complex_normal(initial.shape, rng)
        normalized[:, period] = (
            rho * normalized[:, period - 1]
            + innovation_scale * innovation
        )

    distances, path_loss = distance_and_path_loss(positions, bs_locations)
    channels = normalized * path_loss[..., None]
    return normalized, channels, distances, path_loss, rhos


def temporal_channel_diagnostics(
    normalized_channels,
    rhos,
    lags=(1, 2, 5, 10, 20),
    bootstrap_samples=500,
    rng=None,
):
    """Estimate normalized-channel autocorrelation and bootstrap intervals."""
    normalized = np.asarray(normalized_channels)
    rhos = np.asarray(rhos, dtype=np.float64)
    if normalized.ndim != 5:
        raise ValueError("normalized_channels must have shape [B,T,A,K,M]")
    if rhos.shape != (normalized.shape[0], normalized.shape[3]):
        raise ValueError("rhos must have shape [B, K]")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(0) if rng is None else rng

    sequences = normalized.transpose(0, 2, 3, 4, 1).reshape(
        -1, normalized.shape[1]
    )
    sequence_rhos = np.broadcast_to(
        rhos[:, None, :, None],
        (
            normalized.shape[0],
            normalized.shape[2],
            normalized.shape[3],
            normalized.shape[4],
        ),
    ).reshape(-1)
    valid_lags = np.asarray(
        sorted({int(lag) for lag in lags if 0 < lag < normalized.shape[1]})
    )
    if valid_lags.size == 0:
        raise ValueError("At least one lag must be between 1 and T - 1")

    empirical = []
    theory = []
    confidence_intervals = []
    for lag in valid_lags:
        numerators = np.mean(
            (sequences[:, lag:] * sequences[:, :-lag].conj()).real,
            axis=1,
        )
        denominators = np.mean(np.abs(sequences[:, :-lag]) ** 2, axis=1)
        empirical.append(float(numerators.sum() / denominators.sum()))
        theory.append(float(np.mean(sequence_rhos**lag)))
        sample_indices = rng.integers(
            0,
            sequences.shape[0],
            size=(bootstrap_samples, sequences.shape[0]),
        )
        bootstrap_values = (
            numerators[sample_indices].sum(axis=1)
            / denominators[sample_indices].sum(axis=1)
        )
        confidence_intervals.append(
            np.quantile(bootstrap_values, (0.025, 0.975))
        )

    reduce_axes = (0, 2, 3, 4)
    return {
        "lags": valid_lags,
        "empirical_correlation": np.asarray(empirical),
        "theoretical_correlation": np.asarray(theory),
        "bootstrap_95_ci": np.asarray(confidence_intervals),
        "real_mean_by_period": normalized.real.mean(axis=reduce_axes),
        "imag_mean_by_period": normalized.imag.mean(axis=reduce_axes),
        "real_variance_by_period": normalized.real.var(axis=reduce_axes),
        "imag_variance_by_period": normalized.imag.var(axis=reduce_axes),
    }


def _complex_channels(channels, device, dtype=None):
    channels = torch.as_tensor(channels, device=device)
    if not torch.is_complex(channels):
        raise ValueError("Direct channels must be complex-valued")
    return channels.to(dtype=dtype) if dtype is not None else channels


def _complex_to_real_beamformers(beamformers):
    batch_size, num_ap, num_users, num_antennas = beamformers.shape
    beamformers = beamformers.reshape(
        batch_size, num_ap * num_users, num_antennas
    ).transpose(1, 2)
    return torch.cat((beamformers.real, beamformers.imag), dim=1).float()


def _normalize_ap_beamformers(beamformers, pmax):
    flat = beamformers.flatten(start_dim=1)
    if torch.count_nonzero(flat) == 0:
        return beamformers
    flat = F.normalize(flat, dim=1, eps=1e-12) * np.sqrt(pmax)
    return flat.reshape_as(beamformers)


def mrt_beamforming(channels, association_mask, pmax, device):
    channels = _complex_channels(channels, device)
    association_mask = torch.as_tensor(
        association_mask, dtype=torch.bool, device=device
    )
    beamformers = torch.zeros_like(channels)

    for ap in range(channels.shape[1]):
        ap_beamformers = channels[:, ap].conj()
        ap_beamformers = ap_beamformers * association_mask[:, :, ap, None]
        beamformers[:, ap] = _normalize_ap_beamformers(ap_beamformers, pmax)

    return _complex_to_real_beamformers(beamformers)


def rzf_beamforming(
    channels, association_mask, pmax, device, noise_power=NOISE_POWER
):
    channels = _complex_channels(channels, device)
    association_mask = torch.as_tensor(
        association_mask, dtype=torch.bool, device=device
    )
    beamformers = torch.zeros_like(channels)

    for sample in range(channels.shape[0]):
        for ap in range(channels.shape[1]):
            served = association_mask[sample, :, ap]
            num_served = int(served.sum())
            if num_served == 0:
                continue
            H = channels[sample, ap, served]
            alpha = num_served * noise_power / pmax
            regularized = H @ H.conj().transpose(0, 1)
            regularized = regularized + alpha * torch.eye(
                num_served, dtype=H.dtype, device=device
            )
            weights = H.conj().transpose(0, 1) @ torch.linalg.solve(
                regularized,
                torch.eye(num_served, dtype=H.dtype, device=device),
            )
            beamformers[sample, ap, served] = weights.transpose(0, 1)
        for ap in range(channels.shape[1]):
            beamformers[sample, ap] = _normalize_ap_beamformers(
                beamformers[sample, ap].unsqueeze(0), pmax
            ).squeeze(0)

    return _complex_to_real_beamformers(beamformers)


def calculate_rates(
    W, channel_bs_user, num_BS, device, noise_power=NOISE_POWER
):
    batch_size, two_m, virtual_users = W.shape
    num_users = virtual_users // num_BS
    M = two_m // 2
    complex_w = W[:, :M] + 1j * W[:, M:]
    complex_w = complex_w.transpose(1, 2).reshape(
        batch_size, num_BS, num_users, M
    )
    channels = _complex_channels(
        channel_bs_user, device, dtype=complex_w.dtype
    )

    effective_channels = torch.einsum(
        "bakm,bajm->bkj", channels, complex_w
    )
    received_power = effective_channels.abs().square()
    signal = received_power.diagonal(dim1=1, dim2=2)
    interference = received_power.sum(dim=2) - signal
    sinr = signal / (interference + noise_power)
    return torch.log1p(sinr) / np.log(2)


def cal_loss(
    W, channel_bs_user, num_BS, device, noise_power=NOISE_POWER
):
    rate = calculate_rates(
        W, channel_bs_user, num_BS, device, noise_power
    )
    sum_rate = rate.sum(dim=1)

    return -sum_rate.mean(), sum_rate.mean(), rate.mean(dim=0)


def user_pruning(K, ratio, duplicate=False):
    if duplicate:
        A = np.random.uniform(0, 1, size=(K, K))
        A = (A + A.T) / 2
        A[A > ratio] = 1
        A[A <= ratio] = 0
        A[np.eye(K, dtype=bool)] = 0
    else:
        A = 1 - np.eye(K)

    return A
