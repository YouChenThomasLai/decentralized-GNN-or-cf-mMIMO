import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils_return_indivial_rates import (
    cal_loss,
    complex_normal,
    gen_fixed_location,
    generate_temporal_channels,
    temporal_channel_diagnostics,
)


class Base_station(Dataset):
    def __init__(self, M, loc):
        super().__init__()
        self.M = M
        self.loc = np.asarray(loc)
        self.user_array = None
        self.channel_bs_user = None

    def get_loc(self):
        return self.loc

    def set_user(self, users):
        self.user_array = users

    def get_user(self):
        return self.user_array

    def set_channel(self, channel_bs_user):
        self.channel_bs_user = channel_bs_user

    def get_channel(self):
        return self.channel_bs_user


class MyDataLoader(Dataset):
    """Generate reproducible batches of fixed-association UE trajectories."""

    def __init__(
        self,
        M,
        batch_size,
        episode_steps=2000,
        speed_kmh=0.0,
        decision_period_s=0.001,
        carrier_frequency_hz=2.6e9,
        trajectory_radius_m=100.0,
        seed=None,
    ):
        super().__init__()
        if M <= 0 or batch_size <= 0 or episode_steps <= 0:
            raise ValueError("M, batch_size, and episode_steps must be positive")
        if decision_period_s <= 0 or carrier_frequency_hz <= 0:
            raise ValueError("Channel timing and carrier frequency must be positive")
        if trajectory_radius_m <= 0:
            raise ValueError("trajectory_radius_m must be positive")

        self.M = M
        self.batch_size = batch_size
        self.episode_steps = episode_steps
        self.speed_kmh = speed_kmh
        self.decision_period_s = decision_period_s
        self.carrier_frequency_hz = carrier_frequency_hz
        self.trajectory_radius_m = trajectory_radius_m
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        self.length = 100
        self.BS_Loc_array = gen_fixed_location(5, self.length * 2)
        self.BS_array = [
            Base_station(M, location) for location in self.BS_Loc_array
        ]
        self.K = None
        self.ue_positions = None
        self.ue_speeds_mps = None
        self.ue_directions_rad = None
        self.normalized_channels = None
        self.true_channels = None
        self.stored_channels = None
        self.distances = None
        self.path_loss_factors = None
        self.rhos = None
        self.association_mask = None

    def _make_initial_positions(self, K, initial_positions):
        if initial_positions is not None:
            positions = np.asarray(initial_positions, dtype=np.float64)
            if positions.shape == (K, 2):
                positions = np.broadcast_to(
                    positions, (self.batch_size, K, 2)
                ).copy()
            if positions.shape != (self.batch_size, K, 2):
                raise ValueError("initial_positions must have shape [K,2] or [B,K,2]")
        else:
            positions = np.empty((self.batch_size, K, 2), dtype=np.float64)
            for trajectory in range(self.batch_size):
                for user in range(K):
                    angle = self.rng.uniform(0, 2 * np.pi)
                    radius = self.rng.uniform(0, self.trajectory_radius_m)
                    positions[trajectory, user] = radius * np.array(
                        [np.cos(angle), np.sin(angle)]
                    )
        if np.any(np.linalg.norm(positions, axis=-1) > self.trajectory_radius_m):
            raise ValueError("Initial UE positions must be inside the trajectory disk")
        return positions

    def _make_speeds(self, K, speed_kmh):
        speeds = np.asarray(speed_kmh, dtype=np.float64)
        if speeds.ndim == 0:
            speeds = np.full((self.batch_size, K), speeds)
        elif speeds.shape == (K,):
            speeds = np.broadcast_to(speeds, (self.batch_size, K)).copy()
        if speeds.shape != (self.batch_size, K):
            raise ValueError("speed_kmh must be scalar, [K], or [B,K]")
        if np.any(speeds < 0):
            raise ValueError("UE speeds cannot be negative")
        return speeds / 3.6

    def _make_positions(self, initial_positions, speeds_mps):
        total_time = (self.episode_steps - 1) * self.decision_period_s
        travel_distances = speeds_mps * total_time
        if np.any(travel_distances > 2 * self.trajectory_radius_m):
            raise ValueError("A straight trajectory cannot fit inside the UE disk")

        directions = np.empty_like(speeds_mps)
        for trajectory in range(self.batch_size):
            for user in range(speeds_mps.shape[1]):
                for _ in range(100000):
                    direction = self.rng.uniform(0, 2 * np.pi)
                    endpoint = initial_positions[trajectory, user] + (
                        travel_distances[trajectory, user]
                        * np.array([np.cos(direction), np.sin(direction)])
                    )
                    if np.linalg.norm(endpoint) <= self.trajectory_radius_m:
                        directions[trajectory, user] = direction
                        break
                else:
                    raise ValueError("Could not sample an in-disk straight trajectory")

        unit_directions = np.stack(
            (np.cos(directions), np.sin(directions)), axis=-1
        )
        times = (
            np.arange(self.episode_steps, dtype=np.float64)
            * self.decision_period_s
        )
        return (
            initial_positions[:, None]
            + times[None, :, None, None]
            * speeds_mps[:, None, :, None]
            * unit_directions[:, None]
        ), directions

    def generate_trajectories(
        self,
        K,
        ratio=0.1,
        speed_kmh=None,
        initial_positions=None,
        initial_normalized_channels=None,
    ):
        if K <= 0:
            raise ValueError("K must be positive")
        if not 0 <= ratio <= 1:
            raise ValueError("Association ratio must lie in [0, 1]")
        self.K = K
        initial_positions = self._make_initial_positions(K, initial_positions)
        self.ue_speeds_mps = self._make_speeds(
            K, self.speed_kmh if speed_kmh is None else speed_kmh
        )
        self.ue_positions, self.ue_directions_rad = self._make_positions(
            initial_positions, self.ue_speeds_mps
        )

        expected_shape = (
            self.batch_size,
            len(self.BS_array),
            K,
            self.M,
        )
        if initial_normalized_channels is None:
            initial_normalized_channels = complex_normal(
                expected_shape, self.rng
            )
        else:
            initial_normalized_channels = np.asarray(
                initial_normalized_channels, dtype=np.complex128
            )
            if initial_normalized_channels.shape != expected_shape:
                raise ValueError(
                    "initial_normalized_channels must have shape [B,A,K,M]"
                )

        (
            self.normalized_channels,
            self.true_channels,
            self.distances,
            self.path_loss_factors,
            self.rhos,
        ) = generate_temporal_channels(
            initial_normalized_channels,
            self.ue_positions,
            self.BS_Loc_array,
            self.ue_speeds_mps,
            self.carrier_frequency_hz,
            self.decision_period_s,
            self.rng,
        )
        self.stored_channels = self.true_channels.copy()

        rssi = np.sum(np.abs(self.true_channels[:, 0]) ** 2, axis=-1)
        strongest = np.max(rssi, axis=1, keepdims=True)
        self.association_mask = (rssi >= strongest * ratio).transpose(0, 2, 1)
        for ap, base_station in enumerate(self.BS_array):
            base_station.set_channel(self.true_channels[:, 0, ap])
            base_station.set_user(self.association_mask[:, :, ap])
        return self

    def BS_user_association(self, K, ratio):
        self.generate_trajectories(K, ratio)

    def _require_data(self):
        if self.true_channels is None:
            raise RuntimeError("No trajectory batch has been generated")

    def _format_frames(self, channels, association_mask):
        features = np.concatenate((channels.real, channels.imag), axis=-1)
        ap_mask = association_mask.transpose(0, 2, 1)
        features = (features * ap_mask[..., None]).astype(np.float32)
        decentralized = []
        for ap in range(features.shape[1]):
            feature = torch.from_numpy(features[:, ap]).unsqueeze(1)
            decentralized.append(F.normalize(feature, dim=2))
        centralized = torch.cat(decentralized, dim=2)
        centralized_index = ap_mask.reshape(features.shape[0], -1)
        decentralized_index = [
            association_mask[:, :, ap] for ap in range(features.shape[1])
        ]
        return centralized, centralized_index, decentralized, decentralized_index

    def get_frames(self, trajectory_indices, time_indices):
        self._require_data()
        trajectory_indices = np.asarray(trajectory_indices, dtype=np.int64)
        time_indices = np.asarray(time_indices, dtype=np.int64)
        if trajectory_indices.ndim == 0:
            trajectory_indices = trajectory_indices.reshape(1)
        if time_indices.ndim == 0:
            time_indices = np.full(trajectory_indices.shape, time_indices)
        if trajectory_indices.shape != time_indices.shape:
            raise ValueError("trajectory_indices and time_indices must match")
        channels = self.stored_channels[trajectory_indices, time_indices]
        mask = self.association_mask[trajectory_indices]
        return self._format_frames(channels, mask)

    def get_centralized_features(self):
        self._require_data()
        batch_size, episode_steps = self.true_channels.shape[:2]
        trajectory_indices = np.repeat(np.arange(batch_size), episode_steps)
        time_indices = np.tile(np.arange(episode_steps), batch_size)
        features = self.get_frames(trajectory_indices, time_indices)[0]
        return features.reshape(
            batch_size,
            episode_steps,
            1,
            len(self.BS_array) * self.K,
            2 * self.M,
        )

    def get_decentralized_features(self):
        self._require_data()
        batch_size, episode_steps = self.true_channels.shape[:2]
        trajectory_indices = np.repeat(np.arange(batch_size), episode_steps)
        time_indices = np.tile(np.arange(episode_steps), batch_size)
        features = self.get_frames(trajectory_indices, time_indices)[2]
        return [
            feature.reshape(
                batch_size, episode_steps, 1, self.K, 2 * self.M
            )
            for feature in features
        ]

    def gen_training_data(self, K, ratio, duplicate=False):
        self.generate_trajectories(K, ratio)
        ap_mask = self.association_mask.transpose(0, 2, 1)
        return self.get_centralized_features(), ap_mask.reshape(self.batch_size, -1)

    def gen_testing_data(
        self, K, ratio, duplicate=False, regenerate_channels=True
    ):
        if regenerate_channels:
            self.generate_trajectories(K, ratio)
        else:
            self._require_data()
        return self.get_decentralized_features(), [
            self.association_mask[:, :, ap] for ap in range(len(self.BS_array))
        ]

    def get_stacked_channels(self, trajectory_indices=None, time_indices=None):
        self._require_data()
        if trajectory_indices is None and time_indices is None:
            return self.true_channels
        if trajectory_indices is None or time_indices is None:
            raise ValueError("Both trajectory_indices and time_indices are required")
        return self.true_channels[trajectory_indices, time_indices]

    def get_true_channels(self):
        self._require_data()
        return self.true_channels

    def get_stored_channels(self):
        self._require_data()
        return self.stored_channels

    def get_association_mask(self, expand_time=False):
        self._require_data()
        if expand_time:
            return np.broadcast_to(
                self.association_mask[:, None],
                (self.batch_size, self.episode_steps, self.K, len(self.BS_array)),
            )
        return self.association_mask

    def diagnostics(self, lags=(1, 2, 5, 10, 20), bootstrap_samples=500):
        self._require_data()
        return temporal_channel_diagnostics(
            self.normalized_channels,
            self.rhos,
            lags,
            bootstrap_samples,
            np.random.default_rng(0),
        )

    def compute_loss(
        self,
        W,
        device,
        noise_power=None,
        trajectory_indices=None,
        time_indices=None,
    ):
        if trajectory_indices is None or time_indices is None:
            raise ValueError("Stage 2 loss requires trajectory and time indices")
        kwargs = {} if noise_power is None else {"noise_power": noise_power}
        return cal_loss(
            W,
            self.get_stacked_channels(trajectory_indices, time_indices),
            len(self.BS_array),
            device,
            **kwargs,
        )
