import numpy as np
import torch
import torch.nn.functional as F


DIRECT_CHANNEL_SCALE = -7
DIRECT_CHANNEL_FADING = 10 ** (-4.5)
DIRECT_PATH_LOSS_EXPONENT = 3.5
NOISE_POWER = 1e-12
SPEED_OF_LIGHT_MPS = 3e8
SQUARE_SIDE = 200.0
HEIGHT_DIFFERENCE = 10.0


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


def sample_square_bpp(count, side_length=SQUARE_SIDE, rng=None):
    if count <= 0 or side_length <= 0:
        raise ValueError("count and side_length must be positive")
    generator = np.random if rng is None else rng
    half_side = side_length / 2
    return generator.uniform(-half_side, half_side, size=(count, 2))


def wrapped_displacement(source, target, side_length=SQUARE_SIDE):
    if side_length <= 0:
        raise ValueError("side_length must be positive")
    delta = np.asarray(target) - np.asarray(source)
    return (delta + side_length / 2) % side_length - side_length / 2


def wrapped_horizontal_distance(source, target, side_length=SQUARE_SIDE):
    return np.linalg.norm(
        wrapped_displacement(source, target, side_length), axis=-1
    )


def wrapped_3d_distance(
    source,
    target,
    side_length=SQUARE_SIDE,
    height_difference=HEIGHT_DIFFERENCE,
):
    if height_difference <= 0:
        raise ValueError("height_difference must be positive")
    horizontal = wrapped_horizontal_distance(source, target, side_length)
    return np.sqrt(horizontal**2 + height_difference**2)


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
                wrapped_3d_distance(loc_BS, loc_user[k, :]),
            )
            sample_channels.append(h_bs[0])
        channel_bs_user.append(sample_channels)

    return np.asarray(channel_bs_user) / 10 ** DIRECT_CHANNEL_SCALE


def jakes_correlation(
    speed_mps,
    carrier_frequency_hz=2.6e9,
    decision_period_s=0.001,
):
    """Return the Jakes-calibrated one-step AR(1) coefficient."""
    speed_mps = np.asarray(speed_mps, dtype=np.float64)
    doppler_hz = carrier_frequency_hz * speed_mps / SPEED_OF_LIGHT_MPS
    argument = 2 * np.pi * doppler_hz * decision_period_s
    values = torch.special.bessel_j0(
        torch.as_tensor(argument, dtype=torch.float64)
    ).cpu().numpy()
    return float(values) if values.ndim == 0 else values


def complex_normal(shape, rng):
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) / np.sqrt(2)


def distance_and_path_loss(ue_positions, bs_locations):
    distances = wrapped_3d_distance(
        np.asarray(ue_positions)[:, :, None, :, :],
        np.asarray(bs_locations)[None, None, :, None, :],
    )
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
    rng=None,
):
    """Generate the plan's stationary first-order Gauss-Markov channel."""
    rng = np.random.default_rng() if rng is None else rng
    initial = np.asarray(initial_normalized_channels, dtype=np.complex128)
    positions = np.asarray(ue_positions, dtype=np.float64)
    speeds = np.asarray(ue_speeds_mps, dtype=np.float64)
    batch_size, episode_steps, num_users = positions.shape[:3]
    if initial.shape[0] != batch_size or initial.shape[2] != num_users:
        raise ValueError("Initial channels do not match trajectory dimensions")
    if speeds.shape == (batch_size, num_users):
        speeds = np.broadcast_to(
            speeds[:, None], (batch_size, episode_steps, num_users)
        )
    if speeds.shape != (batch_size, episode_steps, num_users):
        raise ValueError("ue_speeds_mps must have shape [B,K] or [B,T,K]")

    rhos = jakes_correlation(
        speeds, carrier_frequency_hz, decision_period_s
    )
    normalized = np.empty(
        (batch_size, episode_steps, *initial.shape[1:]),
        dtype=np.complex128,
    )
    normalized[:, 0] = initial
    for period in range(1, episode_steps):
        rho = rhos[:, period - 1, None, :, None]
        innovation = complex_normal(initial.shape, rng)
        normalized[:, period] = (
            rho * normalized[:, period - 1]
            + np.sqrt(np.maximum(0.0, 1 - rho**2)) * innovation
        )

    distances, path_loss = distance_and_path_loss(positions, bs_locations)
    return (
        normalized,
        normalized * path_loss[..., None],
        distances,
        path_loss,
        rhos,
    )


def temporal_channel_diagnostics(normalized_channels, rhos, lags=(1, 2, 5, 10)):
    normalized = np.asarray(normalized_channels)
    rhos = np.asarray(rhos, dtype=np.float64)
    valid_lags = np.asarray(
        sorted({int(lag) for lag in lags if 0 < lag < normalized.shape[1]})
    )
    if normalized.ndim != 5 or not valid_lags.size:
        raise ValueError("Expected channels [B,T,A,K,M] and valid positive lags")
    sequences = normalized.transpose(0, 2, 3, 4, 1).reshape(
        -1, normalized.shape[1]
    )
    empirical = []
    theoretical = []
    for lag in valid_lags:
        previous = sequences[:, :-lag]
        following = sequences[:, lag:]
        empirical.append(
            float(
                np.sum((following * previous.conj()).real)
                / np.sum(np.abs(previous) ** 2)
            )
        )
        products = np.ones(
            (rhos.shape[0], rhos.shape[1] - lag, rhos.shape[2])
        )
        for offset in range(lag):
            products *= rhos[:, offset : offset + products.shape[1]]
        theoretical.append(float(products.mean()))
    return {
        "lags": valid_lags,
        "empirical_correlation": np.asarray(empirical),
        "ar1_correlation": np.asarray(theoretical),
        "max_ar1_error": np.asarray(
            np.max(np.abs(np.asarray(empirical) - theoretical))
        ),
        "real_mean": np.asarray(normalized.real.mean()),
        "imag_mean": np.asarray(normalized.imag.mean()),
        "real_variance": np.asarray(normalized.real.var()),
        "imag_variance": np.asarray(normalized.imag.var()),
        "finite": np.asarray(np.isfinite(normalized).all()),
    }


def independent_channel_diagnostics(
    speed_mps,
    lags=(1, 2, 5, 10, 20),
    sample_count=50000,
    seed=0,
    carrier_frequency_hz=2.6e9,
    decision_period_s=0.001,
):
    """Statistical gate using independent normalized channel sequences."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    lags = np.asarray(sorted({int(lag) for lag in lags if lag > 0}))
    rng = np.random.default_rng(seed)
    rho = jakes_correlation(
        speed_mps, carrier_frequency_hz, decision_period_s
    )
    samples = np.empty((sample_count, int(lags.max()) + 1), np.complex128)
    samples[:, 0] = complex_normal(sample_count, rng)
    scale = np.sqrt(max(0.0, 1 - rho**2))
    for period in range(1, samples.shape[1]):
        samples[:, period] = (
            rho * samples[:, period - 1]
            + scale * complex_normal(sample_count, rng)
        )
    empirical = np.asarray(
        [
            np.sum((samples[:, lag] * samples[:, 0].conj()).real)
            / np.sum(np.abs(samples[:, 0]) ** 2)
            for lag in lags
        ]
    )
    ar1 = rho**lags
    doppler_hz = carrier_frequency_hz * speed_mps / SPEED_OF_LIGHT_MPS
    exact_jakes = torch.special.bessel_j0(
        torch.as_tensor(
            2 * np.pi * doppler_hz * decision_period_s * lags,
            dtype=torch.float64,
        )
    ).cpu().numpy()
    return {
        "speed_mps": np.asarray(speed_mps),
        "doppler_hz": np.asarray(doppler_hz),
        "rho": np.asarray(rho),
        "lags": lags,
        "empirical_correlation": empirical,
        "ar1_correlation": ar1,
        "exact_jakes_reference": exact_jakes,
        "max_ar1_error": np.asarray(np.max(np.abs(empirical - ar1))),
        "real_mean": np.asarray(samples.real.mean()),
        "imag_mean": np.asarray(samples.imag.mean()),
        "real_variance": np.asarray(samples.real.var()),
        "imag_variance": np.asarray(samples.imag.var()),
        "finite": np.asarray(np.isfinite(samples).all()),
        "sample_count": np.asarray(sample_count),
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
    rate = torch.log1p(sinr) / np.log(2)
    return rate


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
