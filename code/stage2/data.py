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


MOBILITY_STRAIGHT = "straight"
MOBILITY_HOTSPOT = "hotspot_semi_markov"
MOBILITY_PHASE_DWELL = 0
MOBILITY_PHASE_TRANSIT = 1
MOBILITY_PHASE_LABELS = np.asarray(("dwell", "transit"))
DEFAULT_HOTSPOT_CENTERS = np.asarray(
    ((35.0, 0.0), (0.0, 35.0), (-35.0, 0.0), (0.0, -35.0))
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
        mobility_model=MOBILITY_STRAIGHT,
        hotspot_centers=None,
        hotspot_radius_m=10.0,
        transition_matrix=None,
        dwell_mean_s=5.0,
        dwell_shape=2.0,
        hotspot_trace_duration_s=300.0,
    ):
        super().__init__()
        if M <= 0 or batch_size <= 0 or episode_steps <= 0:
            raise ValueError("M, batch_size, and episode_steps must be positive")
        if decision_period_s <= 0 or carrier_frequency_hz <= 0:
            raise ValueError("Channel timing and carrier frequency must be positive")
        if trajectory_radius_m <= 0:
            raise ValueError("trajectory_radius_m must be positive")
        if mobility_model not in (MOBILITY_STRAIGHT, MOBILITY_HOTSPOT):
            raise ValueError(
                "mobility_model must be straight or hotspot_semi_markov"
            )

        self.M = M
        self.batch_size = batch_size
        self.episode_steps = episode_steps
        self.speed_kmh = speed_kmh
        self.decision_period_s = decision_period_s
        self.carrier_frequency_hz = carrier_frequency_hz
        self.trajectory_radius_m = trajectory_radius_m
        self.mobility_model = mobility_model
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
        self.instantaneous_speeds_mps = None
        self.normalized_channels = None
        self.true_channels = None
        self.stored_channels = None
        self.distances = None
        self.path_loss_factors = None
        self.rhos = None
        self.association_mask = None
        self.hotspot_centers = None
        self.hotspot_radius_m = None
        self.transition_matrix = None
        self.dwell_mean_s = None
        self.dwell_shape = None
        self.hotspot_trace_duration_s = None
        self.hotspot_state = None
        self.mobility_phase = None
        self.parent_trace_ids = None
        self.clip_start_times_s = None
        self.dwell_durations_s = None
        self.dwell_trajectory_indices = None
        self.dwell_user_indices = None
        self.dwell_hotspot_states = None
        self.dwell_start_times_s = None
        self.transition_trajectory_indices = None
        self.transition_user_indices = None
        self.transition_from_states = None
        self.transition_to_states = None

        if mobility_model == MOBILITY_HOTSPOT:
            self._configure_hotspots(
                hotspot_centers,
                hotspot_radius_m,
                transition_matrix,
                dwell_mean_s,
                dwell_shape,
                hotspot_trace_duration_s,
            )

    def _configure_hotspots(
        self,
        centers,
        radius_m,
        transition_matrix,
        dwell_mean_s,
        dwell_shape,
        trace_duration_s,
    ):
        centers = np.asarray(
            DEFAULT_HOTSPOT_CENTERS if centers is None else centers,
            dtype=np.float64,
        )
        if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] == 0:
            raise ValueError("hotspot_centers must have shape [J,2]")
        if not np.all(np.isfinite(centers)):
            raise ValueError("hotspot_centers must be finite")
        if not np.isfinite(radius_m) or radius_m < 0:
            raise ValueError("hotspot_radius_m must be finite and nonnegative")
        if np.any(
            np.linalg.norm(centers, axis=1) + radius_m
            > self.trajectory_radius_m
        ):
            raise ValueError("Every hotspot region must lie inside the UE disk")
        if not np.all(
            np.isfinite((dwell_mean_s, dwell_shape, trace_duration_s))
        ) or any(
            value <= 0
            for value in (dwell_mean_s, dwell_shape, trace_duration_s)
        ):
            raise ValueError("Dwell and macro-trace parameters must be positive")

        hotspot_count = centers.shape[0]
        if transition_matrix is None:
            stickiness = 0.6
            if hotspot_count == 1:
                transition_matrix = np.ones((1, 1))
            else:
                transition_matrix = np.full(
                    (hotspot_count, hotspot_count),
                    (1 - stickiness) / (hotspot_count - 1),
                )
                np.fill_diagonal(transition_matrix, stickiness)
        transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
        if transition_matrix.shape != (hotspot_count, hotspot_count):
            raise ValueError("transition_matrix must have shape [J,J]")
        if not np.all(np.isfinite(transition_matrix)) or np.any(
            transition_matrix < 0
        ) or not np.allclose(
            transition_matrix.sum(axis=1), 1.0
        ):
            raise ValueError("transition_matrix rows must be nonnegative and sum to 1")

        episode_duration_s = (self.episode_steps - 1) * self.decision_period_s
        if trace_duration_s < episode_duration_s:
            raise ValueError("hotspot_trace_duration_s must contain one full clip")
        self.hotspot_centers = centers
        self.hotspot_radius_m = float(radius_m)
        self.transition_matrix = transition_matrix
        self.dwell_mean_s = float(dwell_mean_s)
        self.dwell_shape = float(dwell_shape)
        self.hotspot_trace_duration_s = float(trace_duration_s)

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
        if not np.all(np.isfinite(positions)):
            raise ValueError("Initial UE positions must be finite")
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
        if not np.all(np.isfinite(speeds)) or np.any(speeds < 0):
            raise ValueError("UE speeds must be finite and nonnegative")
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

    def _sample_hotspot_target(self, state):
        angle = self.rng.uniform(0, 2 * np.pi)
        radius = self.hotspot_radius_m * np.sqrt(self.rng.uniform())
        return self.hotspot_centers[state] + radius * np.array(
            (np.cos(angle), np.sin(angle))
        )

    def _make_hotspot_positions(self, initial_positions, speeds_mps):
        batch_size, num_users = speeds_mps.shape
        sample_times = np.arange(self.episode_steps) * self.decision_period_s
        clip_duration_s = sample_times[-1]
        positions = np.empty(
            (batch_size, self.episode_steps, num_users, 2), dtype=np.float64
        )
        states = np.empty(
            (batch_size, self.episode_steps, num_users), dtype=np.int64
        )
        phases = np.empty_like(states, dtype=np.uint8)
        clip_starts = self.rng.uniform(
            0,
            self.hotspot_trace_duration_s - clip_duration_s,
            size=batch_size,
        )
        dwell_events = []
        transition_events = []

        for trajectory in range(batch_size):
            query_times = clip_starts[trajectory] + sample_times
            for user in range(num_users):
                speed = speeds_mps[trajectory, user]
                time_s = 0.0
                position = initial_positions[trajectory, user].copy()
                state = int(self.rng.randint(len(self.hotspot_centers)))
                target = self._sample_hotspot_target(state)
                segments = []

                while time_s < self.hotspot_trace_duration_s:
                    offset = target - position
                    distance = np.linalg.norm(offset)
                    if distance > 1e-12:
                        if speed == 0:
                            segments.append(
                                (
                                    time_s,
                                    self.hotspot_trace_duration_s,
                                    MOBILITY_PHASE_TRANSIT,
                                    state,
                                    position.copy(),
                                    np.zeros(2),
                                )
                            )
                            break
                        transit_duration_s = distance / speed
                        end_s = min(
                            time_s + transit_duration_s,
                            self.hotspot_trace_duration_s,
                        )
                        segments.append(
                            (
                                time_s,
                                end_s,
                                MOBILITY_PHASE_TRANSIT,
                                state,
                                position.copy(),
                                offset / transit_duration_s,
                            )
                        )
                        if end_s == self.hotspot_trace_duration_s:
                            break
                        position = target
                        time_s = end_s

                    dwell_duration_s = max(
                        float(
                            self.rng.gamma(
                                self.dwell_shape,
                                self.dwell_mean_s / self.dwell_shape,
                            )
                        ),
                        np.finfo(np.float64).eps,
                    )
                    dwell_events.append(
                        (trajectory, user, state, time_s, dwell_duration_s)
                    )
                    end_s = min(
                        time_s + dwell_duration_s,
                        self.hotspot_trace_duration_s,
                    )
                    segments.append(
                        (
                            time_s,
                            end_s,
                            MOBILITY_PHASE_DWELL,
                            state,
                            position.copy(),
                            np.zeros(2),
                        )
                    )
                    if end_s == self.hotspot_trace_duration_s:
                        break
                    time_s = end_s
                    next_state = int(
                        self.rng.choice(
                            len(self.hotspot_centers),
                            p=self.transition_matrix[state],
                        )
                    )
                    transition_events.append(
                        (trajectory, user, state, next_state)
                    )
                    state = next_state
                    target = self._sample_hotspot_target(state)

                segment_ends = np.asarray([segment[1] for segment in segments])
                for period, query_time in enumerate(query_times):
                    segment_index = min(
                        np.searchsorted(segment_ends, query_time, side="right"),
                        len(segments) - 1,
                    )
                    start_s, end_s, phase, state, start, velocity = segments[
                        segment_index
                    ]
                    elapsed_s = min(query_time - start_s, end_s - start_s)
                    positions[trajectory, period, user] = (
                        start + elapsed_s * velocity
                    )
                    states[trajectory, period, user] = state
                    phases[trajectory, period, user] = phase

        instantaneous_speeds = np.empty(states.shape, dtype=np.float64)
        if self.episode_steps == 1:
            instantaneous_speeds[:, 0] = np.where(
                phases[:, 0] == MOBILITY_PHASE_TRANSIT, speeds_mps, 0.0
            )
        else:
            instantaneous_speeds[:, :-1] = (
                np.linalg.norm(np.diff(positions, axis=1), axis=-1)
                / self.decision_period_s
            )
            instantaneous_speeds[:, -1] = instantaneous_speeds[:, -2]

        initial_displacements = (
            positions[:, 1] - positions[:, 0]
            if self.episode_steps > 1
            else np.zeros((batch_size, num_users, 2))
        )
        directions = np.arctan2(
            initial_displacements[..., 1], initial_displacements[..., 0]
        )
        self.hotspot_state = states
        self.mobility_phase = phases
        self.instantaneous_speeds_mps = instantaneous_speeds
        self.parent_trace_ids = np.arange(batch_size)
        self.clip_start_times_s = clip_starts

        dwell_events = np.asarray(dwell_events, dtype=np.float64).reshape(-1, 5)
        self.dwell_trajectory_indices = dwell_events[:, 0].astype(np.int64)
        self.dwell_user_indices = dwell_events[:, 1].astype(np.int64)
        self.dwell_hotspot_states = dwell_events[:, 2].astype(np.int64)
        self.dwell_start_times_s = dwell_events[:, 3]
        self.dwell_durations_s = dwell_events[:, 4]
        transition_events = np.asarray(
            transition_events, dtype=np.int64
        ).reshape(-1, 4)
        self.transition_trajectory_indices = transition_events[:, 0]
        self.transition_user_indices = transition_events[:, 1]
        self.transition_from_states = transition_events[:, 2]
        self.transition_to_states = transition_events[:, 3]
        return positions, directions

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
        if self.mobility_model == MOBILITY_HOTSPOT:
            self.ue_positions, self.ue_directions_rad = (
                self._make_hotspot_positions(initial_positions, self.ue_speeds_mps)
            )
            channel_speeds = self.instantaneous_speeds_mps
        else:
            self.ue_positions, self.ue_directions_rad = self._make_positions(
                initial_positions, self.ue_speeds_mps
            )
            self.instantaneous_speeds_mps = np.broadcast_to(
                self.ue_speeds_mps[:, None],
                (self.batch_size, self.episode_steps, K),
            ).copy()
            channel_speeds = self.ue_speeds_mps

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
            channel_speeds,
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

    def get_frames(
        self, trajectory_indices, time_indices, association_mask=None
    ):
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
        mask = (
            self.association_mask[trajectory_indices]
            if association_mask is None
            else np.asarray(association_mask, dtype=bool)
        )
        if mask.shape != (trajectory_indices.size, self.K, len(self.BS_array)):
            raise ValueError("association_mask must have shape [batch,K,A]")
        return self._format_frames(channels, mask)

    def get_current_association_mask(
        self, trajectory_indices, time_indices, ratio=0.1
    ):
        self._require_data()
        channels = self.true_channels[trajectory_indices, time_indices]
        rssi = np.sum(np.abs(channels) ** 2, axis=-1)
        strongest = np.max(rssi, axis=1, keepdims=True)
        return (rssi >= strongest * ratio).transpose(0, 2, 1)

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

    def hotspot_diagnostics(self):
        self._require_data()
        if self.mobility_model != MOBILITY_HOTSPOT:
            raise RuntimeError("Hotspot diagnostics require hotspot mobility")
        hotspot_count = len(self.hotspot_centers)
        transition_counts = np.zeros(
            (hotspot_count, hotspot_count), dtype=np.int64
        )
        np.add.at(
            transition_counts,
            (self.transition_from_states, self.transition_to_states),
            1,
        )
        row_totals = transition_counts.sum(axis=1, keepdims=True)
        empirical_transition_matrix = np.divide(
            transition_counts,
            row_totals,
            out=np.full_like(transition_counts, np.nan, dtype=np.float64),
            where=row_totals != 0,
        )
        occupancy = np.bincount(
            self.hotspot_state.ravel(), minlength=hotspot_count
        ).astype(np.float64)
        occupancy /= occupancy.sum()
        stationary_system = np.vstack(
            (self.transition_matrix.T - np.eye(hotspot_count), np.ones(hotspot_count))
        )
        stationary_target = np.linalg.lstsq(
            stationary_system,
            np.append(np.zeros(hotspot_count), 1.0),
            rcond=None,
        )[0]
        rssi = np.sum(np.abs(self.true_channels) ** 2, axis=-1)
        strongest_ap = np.argmax(rssi, axis=2)
        strongest_ap_change_count = np.sum(
            strongest_ap[:, 1:] != strongest_ap[:, :-1], axis=1
        )
        expanded_fixed_mask = np.broadcast_to(
            self.association_mask[:, None],
            (*strongest_ap.shape, len(self.BS_array)),
        )
        strongest_ap_covered = np.take_along_axis(
            expanded_fixed_mask, strongest_ap[..., None], axis=-1
        )[..., 0]

        normalized = self.normalized_channels.transpose(0, 1, 3, 2, 4)
        distances = self.distances.transpose(0, 1, 3, 2)
        path_loss = self.path_loss_factors.transpose(0, 1, 3, 2)
        phase_distance = []
        phase_path_loss = []
        phase_empirical_lag1 = []
        phase_theoretical_lag1 = []
        for phase in (MOBILITY_PHASE_DWELL, MOBILITY_PHASE_TRANSIT):
            period_mask = self.mobility_phase == phase
            phase_distance.append(
                float(distances[period_mask].mean())
                if np.any(period_mask)
                else np.nan
            )
            phase_path_loss.append(
                float(path_loss[period_mask].mean())
                if np.any(period_mask)
                else np.nan
            )
            edge_mask = period_mask[:, :-1]
            previous = normalized[:, :-1][edge_mask]
            following = normalized[:, 1:][edge_mask]
            if previous.size:
                phase_empirical_lag1.append(
                    float(
                        np.sum((following * previous.conj()).real)
                        / np.sum(np.abs(previous) ** 2)
                    )
                )
                phase_theoretical_lag1.append(
                    float(self.rhos[:, :-1][edge_mask].mean())
                )
            else:
                phase_empirical_lag1.append(np.nan)
                phase_theoretical_lag1.append(np.nan)
        return {
            "configured_transition_matrix": self.transition_matrix,
            "transition_counts": transition_counts,
            "empirical_transition_matrix": empirical_transition_matrix,
            "clip_hotspot_occupancy": occupancy,
            "stationary_occupancy_target": stationary_target,
            "dwell_event_count": np.asarray(self.dwell_durations_s.size),
            "dwell_mean_s": np.asarray(
                self.dwell_durations_s.mean()
                if self.dwell_durations_s.size
                else np.nan
            ),
            "dwell_quantiles_s": (
                np.quantile(self.dwell_durations_s, (0.05, 0.5, 0.95))
                if self.dwell_durations_s.size
                else np.full(3, np.nan)
            ),
            "dwell_fraction": np.asarray(
                np.mean(self.mobility_phase == MOBILITY_PHASE_DWELL)
            ),
            "transit_fraction": np.asarray(
                np.mean(self.mobility_phase == MOBILITY_PHASE_TRANSIT)
            ),
            "strongest_ap_change_count": strongest_ap_change_count,
            "fixed_serving_set_coverage": strongest_ap_covered.mean(
                axis=1
            ),
            "phase_labels": MOBILITY_PHASE_LABELS,
            "mean_distance_by_phase": np.asarray(phase_distance),
            "mean_path_loss_by_phase": np.asarray(phase_path_loss),
            "empirical_lag1_by_phase": np.asarray(
                phase_empirical_lag1
            ),
            "theoretical_lag1_by_phase": np.asarray(
                phase_theoretical_lag1
            ),
        }

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
