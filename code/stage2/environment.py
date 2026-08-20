import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils_return_indivial_rates import (
    cal_loss,
    complex_normal,
    gen_fixed_location,
    gen_location,
    generate_channel,
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


def symmetric_transition_matrix(hotspot_count, stickiness):
    if hotspot_count <= 0 or not 0 <= stickiness <= 1:
        raise ValueError("Invalid hotspot count or stickiness")
    if hotspot_count == 1:
        return np.ones((1, 1))
    matrix = np.full(
        (hotspot_count, hotspot_count),
        (1 - stickiness) / (hotspot_count - 1),
    )
    np.fill_diagonal(matrix, stickiness)
    return matrix


def stationary_distribution(transition_matrix):
    matrix = np.asarray(transition_matrix, dtype=np.float64)
    system = np.vstack(
        (matrix.T - np.eye(len(matrix)), np.ones(len(matrix)))
    )
    return np.linalg.lstsq(
        system, np.append(np.zeros(len(matrix)), 1.0), rcond=None
    )[0]


def _sample_disk(rng, center, radius_m):
    angle = rng.uniform(0, 2 * np.pi)
    radius = radius_m * np.sqrt(rng.uniform())
    return np.asarray(center) + radius * np.asarray(
        (np.cos(angle), np.sin(angle))
    )


def hotspot_process_diagnostics(
    transition_matrix,
    dwell_mean_s,
    dwell_shape,
    transit_speed_mps,
    *,
    centers=DEFAULT_HOTSPOT_CENTERS,
    radius_m=10.0,
    seed=0,
    minimum_events=10000,
    minimum_outgoing_per_state=2000,
    burn_in_events=1000,
    decision_period_s=0.001,
):
    """Run the independent long-stream Gate 2.3 diagnostic."""
    matrix = np.asarray(transition_matrix, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if transit_speed_mps <= 0 or decision_period_s <= 0:
        raise ValueError(
            "Hotspot transit speed and decision period must be positive"
        )
    rng = np.random.default_rng(seed)
    state = int(rng.integers(len(centers)))
    position = _sample_disk(rng, centers[state], radius_m)

    def event_step(state, position):
        dwell = max(
            float(rng.gamma(dwell_shape, dwell_mean_s / dwell_shape)),
            np.finfo(np.float64).eps,
        )
        next_state = int(rng.choice(len(centers), p=matrix[state]))
        distance = 0.0
        self_motion = 0.0
        step_limit_violation = 0.0
        endpoint_error = 0.0
        max_radius = float(np.linalg.norm(position))
        if next_state != state:
            target = _sample_disk(rng, centers[next_state], radius_m)
            offset = target - position
            distance = float(np.linalg.norm(offset))
            transit_s = distance / transit_speed_mps
            velocity = offset / transit_s
            step_s = min(decision_period_s, transit_s)
            step_limit_violation = max(
                0.0,
                float(np.linalg.norm(step_s * velocity))
                - transit_speed_mps * decision_period_s,
            )
            position = position + transit_s * velocity
            endpoint_error = float(np.linalg.norm(position - target))
            max_radius = max(max_radius, float(np.linalg.norm(position)))
        return (
            next_state,
            position,
            dwell,
            distance,
            self_motion,
            max_radius,
            step_limit_violation,
            endpoint_error,
        )

    for _ in range(burn_in_events):
        state, position, *_ = event_step(state, position)

    transition_counts = np.zeros_like(matrix, dtype=np.int64)
    event_state_counts = np.zeros(len(matrix), dtype=np.int64)
    dwell_time_by_state = np.zeros(len(matrix))
    dwell_durations = []
    residence_durations = []
    residence_s = 0.0
    transit_time_s = 0.0
    max_boundary_radius_m = 0.0
    self_transition_motion_distance_max_m = 0.0
    max_step_limit_violation_m = 0.0
    transit_endpoint_error_max_m = 0.0
    event_count = 0
    while (
        event_count < minimum_events
        or np.min(transition_counts.sum(axis=1))
        < minimum_outgoing_per_state
    ):
        previous_state = state
        (
            state,
            position,
            dwell,
            transit_distance,
            self_motion,
            max_radius,
            step_limit_violation,
            endpoint_error,
        ) = event_step(state, position)
        dwell_durations.append(dwell)
        residence_s += dwell
        event_state_counts[previous_state] += 1
        dwell_time_by_state[previous_state] += dwell
        transition_counts[previous_state, state] += 1
        transit_time_s += transit_distance / transit_speed_mps
        max_boundary_radius_m = max(max_boundary_radius_m, max_radius)
        self_transition_motion_distance_max_m = max(
            self_transition_motion_distance_max_m, self_motion
        )
        max_step_limit_violation_m = max(
            max_step_limit_violation_m, step_limit_violation
        )
        transit_endpoint_error_max_m = max(
            transit_endpoint_error_max_m, endpoint_error
        )
        if state != previous_state:
            residence_durations.append(residence_s)
            residence_s = 0.0
        event_count += 1
        if event_count > 100 * minimum_events:
            raise RuntimeError("Could not satisfy hotspot event-count gate")
    if residence_s:
        residence_durations.append(residence_s)

    dwell_durations = np.asarray(dwell_durations)
    residence_durations = np.asarray(residence_durations)
    row_totals = transition_counts.sum(axis=1, keepdims=True)
    empirical_matrix = transition_counts / row_totals
    stationary = stationary_distribution(matrix)
    natural_total = dwell_durations.sum() + transit_time_s
    diagonal = np.diag(matrix)
    return {
        "configured_transition_matrix": matrix,
        "transition_counts": transition_counts,
        "empirical_transition_matrix": empirical_matrix,
        "event_state_counts": event_state_counts,
        "event_chain_occupancy": event_state_counts / event_state_counts.sum(),
        "stationary_occupancy_target": stationary,
        "event_count": np.asarray(event_count),
        "minimum_outgoing_event_count": np.asarray(row_totals.min()),
        "event_dwell_mean_s": np.asarray(dwell_durations.mean()),
        "event_dwell_variance_s2": np.asarray(dwell_durations.var()),
        "event_dwell_quantiles_s": np.quantile(
            dwell_durations, (0.05, 0.5, 0.95)
        ),
        "merged_residence_count": np.asarray(residence_durations.size),
        "merged_residence_mean_s": np.asarray(residence_durations.mean()),
        "merged_residence_variance_s2": np.asarray(
            residence_durations.var()
        ),
        "merged_residence_quantiles_s": np.quantile(
            residence_durations, (0.05, 0.5, 0.95)
        ),
        "merged_residence_mean_target_s": np.asarray(
            np.sum(stationary * dwell_mean_s / (1 - diagonal))
        ),
        "natural_time_hotspot_occupancy": dwell_time_by_state / natural_total,
        "natural_time_dwell_fraction": np.asarray(
            dwell_durations.sum() / natural_total
        ),
        "natural_time_transit_fraction": np.asarray(
            transit_time_s / natural_total
        ),
        "natural_time_total_s": np.asarray(natural_total),
        "max_boundary_radius_m": np.asarray(max_boundary_radius_m),
        "max_step_limit_violation_m": np.asarray(
            max_step_limit_violation_m
        ),
        "transit_endpoint_error_max_m": np.asarray(
            transit_endpoint_error_max_m
        ),
        "self_transition_motion_distance_max_m": np.asarray(
            self_transition_motion_distance_max_m
        ),
    }


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


class SnapshotEnvironment(Dataset):
    def __init__(self, M, batch_size):
        super().__init__()
        self.M = M
        self.batch_size = batch_size
        self.length = 100
        self.BS_Loc_array = gen_fixed_location(5, self.length * 2)
        self.BS_array = [
            Base_station(M, location) for location in self.BS_Loc_array
        ]
        self.user_loc = None
        self.association_mask = None
        self.K = None

    def BS_user_association(self, K, ratio):
        self.K = K
        self.user_loc = gen_location(K, self.length)
        RSSI = np.zeros((self.batch_size, K, len(self.BS_array)))

        for ap, base_station in enumerate(self.BS_array):
            channel_bs_user = generate_channel(
                self.M,
                K,
                self.batch_size,
                base_station.get_loc(),
                self.user_loc,
            )
            base_station.set_channel(channel_bs_user)
            RSSI[:, :, ap] = np.sum(
                np.abs(channel_bs_user) ** 2, axis=2
            ).real

        strongest = np.max(RSSI, axis=2, keepdims=True)
        self.association_mask = RSSI >= strongest * ratio
        for ap, base_station in enumerate(self.BS_array):
            base_station.set_user(self.association_mask[:, :, ap])

    def load_data(self, base_station):
        channel_bs_user = base_station.get_channel()
        user_index = base_station.get_user()
        channel_feature = np.concatenate(
            (channel_bs_user.real, channel_bs_user.imag), axis=2
        )
        user_feature = np.zeros_like(channel_feature, dtype=np.float32)
        user_feature[user_index] = channel_feature[user_index]
        user_feature = torch.from_numpy(user_feature).unsqueeze(1)
        user_feature = F.normalize(user_feature, dim=2)
        return user_feature, user_index

    def gen_training_data(self, K, ratio, duplicate=False):
        self.BS_user_association(K, ratio)
        ap_data = [self.load_data(base_station) for base_station in self.BS_array]
        user_feature = torch.cat([data[0] for data in ap_data], dim=2)
        user_index = np.concatenate([data[1] for data in ap_data], axis=1)
        return user_feature, user_index

    def gen_testing_data(
        self, K, ratio, duplicate=False, regenerate_channels=True
    ):
        if regenerate_channels:
            self.BS_user_association(K, ratio)
        elif self.association_mask is None:
            raise RuntimeError("No stored channel batch to reformat")
        ap_data = [self.load_data(base_station) for base_station in self.BS_array]
        user_feature = [data[0] for data in ap_data]
        user_index = [data[1] for data in ap_data]
        return user_feature, user_index

    def get_stacked_channels(self):
        if self.association_mask is None:
            raise RuntimeError("No stored channel batch")
        return np.stack(
            [base_station.get_channel() for base_station in self.BS_array],
            axis=1,
        )

    def get_association_mask(self):
        if self.association_mask is None:
            raise RuntimeError("No stored association mask")
        return self.association_mask

    def compute_loss(self, W, device, noise_power=None):
        kwargs = {} if noise_power is None else {"noise_power": noise_power}
        return cal_loss(
            W,
            self.get_stacked_channels(),
            len(self.BS_array),
            device,
            **kwargs,
        )


class MobilityEnvironment(SnapshotEnvironment):
    """Stage 1 snapshot environment extended with a trajectory time axis."""

    def __init__(
        self,
        M,
        batch_size,
        episode_steps=2000,
        speed_kmh=0.0,
        decision_period_s=0.001,
        carrier_frequency_hz=2.6e9,
        trajectory_radius_m=100.0,
        seed=0,
        mobility_model=MOBILITY_STRAIGHT,
        hotspot_centers=None,
        hotspot_radius_m=10.0,
        transition_matrix=None,
        dwell_mean_s=5.0,
        dwell_shape=2.0,
        hotspot_trace_duration_s=300.0,
    ):
        super().__init__(M, batch_size)
        if episode_steps <= 0 or decision_period_s <= 0:
            raise ValueError("episode_steps and decision_period_s must be positive")
        if carrier_frequency_hz <= 0 or trajectory_radius_m <= 0:
            raise ValueError("Carrier frequency and trajectory radius must be positive")
        if mobility_model not in (MOBILITY_STRAIGHT, MOBILITY_HOTSPOT):
            raise ValueError("Unknown mobility model")
        self.episode_steps = episode_steps
        self.speed_kmh = speed_kmh
        self.decision_period_s = decision_period_s
        self.carrier_frequency_hz = carrier_frequency_hz
        self.trajectory_radius_m = trajectory_radius_m
        self.seed = seed
        self.mobility_model = mobility_model
        self.hotspot_centers = np.asarray(
            DEFAULT_HOTSPOT_CENTERS
            if hotspot_centers is None
            else hotspot_centers,
            dtype=np.float64,
        )
        self.hotspot_radius_m = float(hotspot_radius_m)
        self.transition_matrix = np.asarray(
            symmetric_transition_matrix(len(self.hotspot_centers), 0.6)
            if transition_matrix is None
            else transition_matrix,
            dtype=np.float64,
        )
        self.dwell_mean_s = float(dwell_mean_s)
        self.dwell_shape = float(dwell_shape)
        self.hotspot_trace_duration_s = float(hotspot_trace_duration_s)
        self._validate_hotspot_config()
        self.ue_positions = None
        self.ue_speeds_mps = None
        self.ue_directions_rad = None
        self.instantaneous_speeds_mps = None
        self.normalized_channels = None
        self.true_channels = None
        self.distances = None
        self.path_loss_factors = None
        self.rhos = None
        self.hotspot_state = None
        self.mobility_phase = None
        self.mobility_sample_phase = None
        self.parent_trace_ids = None
        self.clip_start_times_s = None
        self.hotspot_burn_in_s = None

    def _validate_hotspot_config(self):
        centers = self.hotspot_centers
        matrix = self.transition_matrix
        if centers.ndim != 2 or centers.shape[1] != 2 or not len(centers):
            raise ValueError("hotspot_centers must have shape [J,2]")
        if np.any(
            np.linalg.norm(centers, axis=1) + self.hotspot_radius_m
            > self.trajectory_radius_m
        ):
            raise ValueError("Every hotspot must lie inside the UE disk")
        if matrix.shape != (len(centers), len(centers)):
            raise ValueError("transition_matrix must have shape [J,J]")
        if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1):
            raise ValueError("transition_matrix must be row stochastic")
        if any(
            value <= 0
            for value in (
                self.dwell_mean_s,
                self.dwell_shape,
                self.hotspot_trace_duration_s,
            )
        ):
            raise ValueError("Hotspot dwell and trace values must be positive")
        clip_duration = (self.episode_steps - 1) * self.decision_period_s
        if self.hotspot_trace_duration_s < clip_duration:
            raise ValueError("Hotspot retained trace must contain one full clip")

    def _make_speeds(self, K, speed_kmh):
        speeds = np.asarray(speed_kmh, dtype=np.float64)
        if speeds.ndim == 0:
            speeds = np.full((self.batch_size, K), speeds)
        elif speeds.shape == (K,):
            speeds = np.broadcast_to(speeds, (self.batch_size, K)).copy()
        if speeds.shape != (self.batch_size, K):
            raise ValueError("speed_kmh must be scalar, [K], or [B,K]")
        if not np.all(np.isfinite(speeds)) or np.any(speeds < 0):
            raise ValueError("Speeds must be finite and nonnegative")
        if self.mobility_model == MOBILITY_HOTSPOT and np.any(speeds == 0):
            raise ValueError("Hotspot mobility requires positive transit speed")
        return speeds / 3.6

    def _make_initial_positions(self, K, initial_positions, rng):
        if initial_positions is None:
            angles = rng.uniform(0, 2 * np.pi, (self.batch_size, K))
            radii = rng.uniform(0, self.trajectory_radius_m, (self.batch_size, K))
            return radii[..., None] * np.stack(
                (np.cos(angles), np.sin(angles)), axis=-1
            )
        positions = np.asarray(initial_positions, dtype=np.float64)
        if positions.shape == (K, 2):
            positions = np.broadcast_to(
                positions, (self.batch_size, K, 2)
            ).copy()
        if positions.shape != (self.batch_size, K, 2):
            raise ValueError("initial_positions must have shape [K,2] or [B,K,2]")
        if np.any(np.linalg.norm(positions, axis=-1) > self.trajectory_radius_m):
            raise ValueError("Initial positions must lie inside the UE disk")
        return positions

    def _make_straight_positions(self, initial_positions, speeds_mps, rng):
        total_time = (self.episode_steps - 1) * self.decision_period_s
        travel_distances = speeds_mps * total_time
        if np.any(travel_distances > 2 * self.trajectory_radius_m):
            raise ValueError("A straight trajectory cannot fit inside the UE disk")
        start_radii = np.linalg.norm(initial_positions, axis=-1)
        if np.any(travel_distances > self.trajectory_radius_m + start_radii):
            raise ValueError(
                "A straight trajectory cannot fit from its initial position"
            )
        directions = np.empty_like(speeds_mps)
        for trajectory in range(self.batch_size):
            for user in range(speeds_mps.shape[1]):
                start_radius = start_radii[trajectory, user]
                travel_distance = travel_distances[trajectory, user]
                if start_radius > 0 and np.isclose(
                    travel_distance,
                    self.trajectory_radius_m + start_radius,
                    rtol=0,
                    atol=np.finfo(np.float64).eps * self.trajectory_radius_m,
                ):
                    position = initial_positions[trajectory, user]
                    directions[trajectory, user] = (
                        np.arctan2(-position[1], -position[0]) % (2 * np.pi)
                    )
                    continue
                while True:
                    direction = rng.uniform(0, 2 * np.pi)
                    endpoint = initial_positions[trajectory, user] + (
                        travel_distance
                        * np.asarray((np.cos(direction), np.sin(direction)))
                    )
                    if np.linalg.norm(endpoint) <= self.trajectory_radius_m:
                        directions[trajectory, user] = direction
                        break
        unit_directions = np.stack(
            (np.cos(directions), np.sin(directions)), axis=-1
        )
        times = np.arange(self.episode_steps) * self.decision_period_s
        positions = (
            initial_positions[:, None]
            + times[None, :, None, None]
            * speeds_mps[:, None, :, None]
            * unit_directions[:, None]
        )
        return positions, directions

    def _configured_burn_in(self, speed_mps):
        stationary = stationary_distribution(self.transition_matrix)
        center_distances = np.linalg.norm(
            self.hotspot_centers[:, None] - self.hotspot_centers[None],
            axis=-1,
        )
        expected_transit_s = np.sum(
            stationary[:, None]
            * self.transition_matrix
            * center_distances
        ) / speed_mps
        return max(120.0, 10 * (self.dwell_mean_s + expected_transit_s))

    def _hotspot_segments(self, rng, speed_mps, duration_s):
        state = int(rng.integers(len(self.hotspot_centers)))
        position = _sample_disk(
            rng, self.hotspot_centers[state], self.hotspot_radius_m
        )
        time_s = 0.0
        segments = []
        while time_s < duration_s:
            dwell_s = max(
                float(
                    rng.gamma(
                        self.dwell_shape,
                        self.dwell_mean_s / self.dwell_shape,
                    )
                ),
                np.finfo(np.float64).eps,
            )
            end_s = min(time_s + dwell_s, duration_s)
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
            if end_s == duration_s:
                break
            time_s = end_s
            next_state = int(
                rng.choice(
                    len(self.hotspot_centers),
                    p=self.transition_matrix[state],
                )
            )
            if next_state == state:
                continue
            target = _sample_disk(
                rng,
                self.hotspot_centers[next_state],
                self.hotspot_radius_m,
            )
            offset = target - position
            transit_s = np.linalg.norm(offset) / speed_mps
            end_s = min(time_s + transit_s, duration_s)
            segments.append(
                (
                    time_s,
                    end_s,
                    MOBILITY_PHASE_TRANSIT,
                    next_state,
                    position.copy(),
                    offset / transit_s,
                )
            )
            elapsed_s = end_s - time_s
            position = position + elapsed_s * offset / transit_s
            if end_s == duration_s:
                break
            position = target
            state = next_state
            time_s = end_s
        return segments

    def _make_hotspot_positions(self, speeds_mps, geometry_seed):
        batch_size, num_users = speeds_mps.shape
        clip_duration = (self.episode_steps - 1) * self.decision_period_s
        speed_floor = float(speeds_mps.min())
        burn_in_s = self._configured_burn_in(speed_floor)
        total_duration = burn_in_s + self.hotspot_trace_duration_s
        sample_offsets = np.arange(self.episode_steps) * self.decision_period_s
        positions = np.empty(
            (batch_size, self.episode_steps, num_users, 2), dtype=np.float64
        )
        states = np.empty(
            (batch_size, self.episode_steps, num_users), dtype=np.int64
        )
        sample_phases = np.empty_like(states, dtype=np.uint8)
        clip_starts = np.empty(batch_size)

        for trajectory, parent_seed in enumerate(geometry_seed.spawn(batch_size)):
            children = parent_seed.spawn(num_users + 1)
            clip_rng = np.random.default_rng(children[0])
            clip_starts[trajectory] = burn_in_s + clip_rng.uniform(
                0,
                self.hotspot_trace_duration_s - clip_duration,
            )
            query_times = clip_starts[trajectory] + sample_offsets
            for user in range(num_users):
                segments = self._hotspot_segments(
                    np.random.default_rng(children[user + 1]),
                    speeds_mps[trajectory, user],
                    total_duration,
                )
                ends = np.asarray([segment[1] for segment in segments])
                for period, query_time in enumerate(query_times):
                    index = min(
                        np.searchsorted(ends, query_time, side="right"),
                        len(segments) - 1,
                    )
                    start, end, phase, state, origin, velocity = segments[index]
                    elapsed = min(query_time - start, end - start)
                    positions[trajectory, period, user] = origin + elapsed * velocity
                    states[trajectory, period, user] = state
                    sample_phases[trajectory, period, user] = phase

        instantaneous_speeds = np.empty(states.shape, dtype=np.float64)
        if self.episode_steps > 1:
            instantaneous_speeds[:, :-1] = (
                np.linalg.norm(np.diff(positions, axis=1), axis=-1)
                / self.decision_period_s
            )
            instantaneous_speeds[:, -1] = np.where(
                sample_phases[:, -1] == MOBILITY_PHASE_TRANSIT,
                speeds_mps,
                0.0,
            )
        else:
            instantaneous_speeds[:, 0] = np.where(
                sample_phases[:, 0] == MOBILITY_PHASE_TRANSIT,
                speeds_mps,
                0.0,
            )
        phases = np.where(
            instantaneous_speeds > 0,
            MOBILITY_PHASE_TRANSIT,
            MOBILITY_PHASE_DWELL,
        ).astype(np.uint8)
        displacement = (
            positions[:, 1] - positions[:, 0]
            if self.episode_steps > 1
            else np.zeros((batch_size, num_users, 2))
        )
        directions = np.arctan2(displacement[..., 1], displacement[..., 0])
        self.hotspot_state = states
        self.mobility_phase = phases
        self.mobility_sample_phase = sample_phases
        self.instantaneous_speeds_mps = instantaneous_speeds
        self.parent_trace_ids = np.arange(batch_size)
        self.clip_start_times_s = clip_starts
        self.hotspot_burn_in_s = burn_in_s
        return positions, directions

    def generate_trajectories(
        self,
        K,
        ratio=0.1,
        speed_kmh=None,
        initial_positions=None,
        initial_normalized_channels=None,
    ):
        if K <= 0 or not 0 <= ratio <= 1:
            raise ValueError("Invalid K or association threshold")
        self.K = K
        root_seed = np.random.SeedSequence(self.seed)
        geometry_seed, channel_seed = root_seed.spawn(2)
        geometry_rng = np.random.default_rng(geometry_seed)
        channel_rng = np.random.default_rng(channel_seed)
        self.ue_speeds_mps = self._make_speeds(
            K, self.speed_kmh if speed_kmh is None else speed_kmh
        )
        if self.mobility_model == MOBILITY_HOTSPOT:
            if initial_positions is not None:
                raise ValueError(
                    "Hotspot initial positions are generated in hotspot disks"
                )
            self.ue_positions, self.ue_directions_rad = (
                self._make_hotspot_positions(
                    self.ue_speeds_mps, geometry_seed
                )
            )
        else:
            initial_positions = self._make_initial_positions(
                K, initial_positions, geometry_rng
            )
            self.ue_positions, self.ue_directions_rad = (
                self._make_straight_positions(
                    initial_positions, self.ue_speeds_mps, geometry_rng
                )
            )
            self.instantaneous_speeds_mps = np.broadcast_to(
                self.ue_speeds_mps[:, None],
                (self.batch_size, self.episode_steps, K),
            ).copy()

        initial_shape = (
            self.batch_size,
            len(self.BS_array),
            K,
            self.M,
        )
        if initial_normalized_channels is None:
            initial_normalized_channels = complex_normal(
                initial_shape, channel_rng
            )
        else:
            initial_normalized_channels = np.asarray(
                initial_normalized_channels, dtype=np.complex128
            )
            if initial_normalized_channels.shape != initial_shape:
                raise ValueError("Initial channels must have shape [B,A,K,M]")
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
            self.instantaneous_speeds_mps,
            self.carrier_frequency_hz,
            self.decision_period_s,
            channel_rng,
        )
        rssi = np.sum(np.abs(self.true_channels[:, 0]) ** 2, axis=-1)
        strongest = np.max(rssi, axis=1, keepdims=True)
        self.association_mask = (rssi >= strongest * ratio).transpose(0, 2, 1)
        for ap, base_station in enumerate(self.BS_array):
            base_station.set_channel(self.true_channels[:, 0, ap])
            base_station.set_user(self.association_mask[:, :, ap])
        return self

    def BS_user_association(self, K, ratio):
        self.generate_trajectories(K, ratio)

    def _require_trajectories(self):
        if self.true_channels is None:
            raise RuntimeError("No trajectories have been generated")

    def _format_frames(self, channels, association_mask):
        features = np.concatenate((channels.real, channels.imag), axis=-1)
        ap_mask = association_mask.transpose(0, 2, 1)
        features = (features * ap_mask[..., None]).astype(np.float32)
        decentralized = [
            F.normalize(torch.from_numpy(features[:, ap]).unsqueeze(1), dim=2)
            for ap in range(features.shape[1])
        ]
        return (
            torch.cat(decentralized, dim=2),
            ap_mask.reshape(features.shape[0], -1),
            decentralized,
            [association_mask[:, :, ap] for ap in range(features.shape[1])],
        )

    def get_frames(self, trajectory_indices, time_indices):
        self._require_trajectories()
        trajectory_indices = np.asarray(trajectory_indices, dtype=np.int64)
        time_indices = np.asarray(time_indices, dtype=np.int64)
        if trajectory_indices.ndim == 0:
            trajectory_indices = trajectory_indices.reshape(1)
        if time_indices.ndim == 0:
            time_indices = np.full(trajectory_indices.shape, time_indices)
        if trajectory_indices.shape != time_indices.shape:
            raise ValueError("Trajectory and time indices must match")
        return self._format_frames(
            self.true_channels[trajectory_indices, time_indices],
            self.association_mask[trajectory_indices],
        )

    def get_stacked_channels(self, trajectory_indices=None, time_indices=None):
        self._require_trajectories()
        if trajectory_indices is None and time_indices is None:
            return self.true_channels
        if trajectory_indices is None or time_indices is None:
            raise ValueError("Both trajectory and time indices are required")
        return self.true_channels[trajectory_indices, time_indices]

    def get_association_mask(self, expand_time=False):
        self._require_trajectories()
        if expand_time:
            return np.broadcast_to(
                self.association_mask[:, None],
                (
                    self.batch_size,
                    self.episode_steps,
                    self.K,
                    len(self.BS_array),
                ),
            )
        return self.association_mask

    def channel_diagnostics(self, lags=(1, 2, 5, 10)):
        self._require_trajectories()
        return temporal_channel_diagnostics(
            self.normalized_channels, self.rhos, lags
        )

    def mobility_diagnostics(self):
        self._require_trajectories()
        step_distances = np.linalg.norm(
            np.diff(self.ue_positions, axis=1), axis=-1
        )
        step_limits = (
            self.ue_speeds_mps[:, None] * self.decision_period_s
        )
        diagnostics = {
            "max_position_radius_m": np.asarray(
                np.linalg.norm(self.ue_positions, axis=-1).max()
            ),
            "max_step_limit_violation_m": np.asarray(
                max(0.0, float(np.max(step_distances - step_limits)))
                if step_distances.size
                else 0.0
            ),
            "association_constant": np.asarray(True),
        }
        if self.mobility_model == MOBILITY_STRAIGHT:
            expected = self.ue_speeds_mps[:, None] * self.decision_period_s
            diagnostics["max_step_error_m"] = np.asarray(
                np.max(np.abs(step_distances - expected))
                if step_distances.size
                else 0.0
            )
            return diagnostics

        dwell_mask = (
            self.mobility_phase[:, :-1] == MOBILITY_PHASE_DWELL
        )
        transit_mask = ~dwell_mask
        normalized = self.normalized_channels.transpose(0, 1, 3, 2, 4)
        empirical = []
        theoretical = []
        for mask in (dwell_mask, transit_mask):
            previous = normalized[:, :-1][mask]
            following = normalized[:, 1:][mask]
            empirical.append(
                float(
                    np.sum((following * previous.conj()).real)
                    / np.sum(np.abs(previous) ** 2)
                )
                if previous.size
                else np.nan
            )
            theoretical.append(
                float(self.rhos[:, :-1][mask].mean())
                if np.any(mask)
                else np.nan
            )
        hotspot_count = len(self.hotspot_centers)
        dwell_state_counts = np.bincount(
            self.hotspot_state[:, :-1][dwell_mask],
            minlength=hotspot_count,
        )
        total_periods = dwell_mask.size
        diagnostics.update(
            {
                "hotspot_burn_in_s": np.asarray(self.hotspot_burn_in_s),
                "clip_start_times_s": self.clip_start_times_s,
                "clip_hotspot_occupancy": dwell_state_counts / total_periods,
                "clip_dwell_fraction": np.asarray(dwell_mask.mean()),
                "clip_transit_fraction": np.asarray(transit_mask.mean()),
                "phase_labels": MOBILITY_PHASE_LABELS,
                "empirical_lag1_by_phase": np.asarray(empirical),
                "ar1_lag1_by_phase": np.asarray(theoretical),
                "dwell_channel_max_abs_change": np.asarray(
                    np.max(
                        np.abs(
                            normalized[:, 1:][dwell_mask]
                            - normalized[:, :-1][dwell_mask]
                        )
                    )
                    if np.any(dwell_mask)
                    else np.nan
                ),
            }
        )
        return diagnostics

    def compute_loss(
        self,
        W,
        device,
        noise_power=None,
        trajectory_indices=None,
        time_indices=None,
    ):
        if trajectory_indices is None or time_indices is None:
            raise ValueError("Mobility loss requires trajectory and time indices")
        kwargs = {} if noise_power is None else {"noise_power": noise_power}
        return cal_loss(
            W,
            self.get_stacked_channels(trajectory_indices, time_indices),
            len(self.BS_array),
            device,
            **kwargs,
        )


MyDataLoader = MobilityEnvironment
