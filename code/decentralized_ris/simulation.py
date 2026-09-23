import numpy as np
import torch
import torch.nn.functional as F

from rates import (
    generate_channels,
    generate_los,
    random_locations,
    ring_locations,
    sum_rate_loss,
)


class BaseStation:
    def __init__(self, n_antennas, n_elements, n_ris, location, ris_locations):
        self.location = np.asarray(location)[None, :]
        self.ris_locations = ris_locations
        self.ris_ids = np.arange(n_ris)
        self.los_bs_ris = generate_los(n_elements, n_antennas, 10, n_ris)
        self.user_mask = None
        self.test_user_mask = None
        self.channels = None
        self.direct_channels = None


class ChannelSimulator:
    """Stateful topology and channel generator used by training and evaluation."""

    def __init__(self, n_antennas, n_elements, n_ris, batch_size, n_ap=5):
        self.n_antennas = n_antennas
        self.n_elements = n_elements
        self.n_ris = n_ris
        self.batch_size = batch_size
        self.n_ap = n_ap
        self.radius = 100
        self.ap_locations = ring_locations(n_ap, self.radius * 2)
        self.ris_locations = ring_locations(n_ris, self.radius)
        self.base_stations = [
            BaseStation(n_antennas, n_elements, n_ris, location, self.ris_locations)
            for location in self.ap_locations
        ]
        self.users_per_ap = None

    def associate_users(self, users_per_ap, ratio, testing_ratio):
        self.users_per_ap = users_per_ap
        user_locations = random_locations(users_per_ap, self.radius)
        received_power = np.zeros((self.batch_size, users_per_ap, self.n_ap))

        for ap_index, station in enumerate(self.base_stations):
            channels, direct, _, _ = generate_channels(
                self.n_antennas,
                self.n_elements,
                self.n_ris,
                users_per_ap,
                self.batch_size,
                station.los_bs_ris,
                0,
                station.ris_locations,
                station.location,
                user_locations,
            )
            station.channels = channels
            station.direct_channels = direct
            for user_index in range(users_per_ap):
                direct_user = direct[:, user_index, :].reshape(self.batch_size, 1, -1)
                power = np.matmul(direct_user, np.conj(direct_user.transpose(0, 2, 1)))
                received_power[:, user_index, ap_index] = np.real(power[:, 0, 0])

        strongest = np.max(received_power, axis=2, keepdims=True)
        masks = received_power >= strongest * ratio
        test_masks = received_power >= strongest * testing_ratio
        for ap_index, station in enumerate(self.base_stations):
            station.user_mask = masks[:, :, ap_index]
            station.test_user_mask = test_masks[:, :, ap_index]

    def _station_data(self, station):
        channels = station.channels
        direct = station.direct_channels
        mask = station.user_mask
        batch_size = channels.shape[0]
        features = np.zeros(
            (
                batch_size,
                self.users_per_ap,
                self.n_ris,
                2 * self.n_antennas,
                self.n_elements + 1,
            )
        )
        edges = np.zeros((batch_size, self.users_per_ap, self.n_ris))
        direct_edges = np.zeros((batch_size, self.users_per_ap))

        for ris_index in station.ris_ids:
            ris_channel = channels[:, :, :, ris_index, :].transpose((0, 3, 1, 2))
            ris_channel = ris_channel[mask, :, :]
            features[mask, ris_index, :self.n_antennas, :self.n_elements] = ris_channel.real
            features[mask, ris_index, self.n_antennas:, :self.n_elements] = ris_channel.imag
            covariance = np.matmul(ris_channel, np.conj(ris_channel.transpose(0, 2, 1)))
            edges[mask, ris_index] = np.real(np.trace(covariance, axis1=1, axis2=2))

            direct_user = direct[mask, :]
            features[mask, ris_index, :self.n_antennas, self.n_elements] = direct_user.real
            features[mask, ris_index, self.n_antennas:, self.n_elements] = direct_user.imag
            if direct_user.shape[0]:
                direct_user = direct_user.reshape((direct_user.shape[0], 1, -1))
                covariance = np.matmul(direct_user, np.conj(direct_user.transpose(0, 2, 1)))
                direct_edges[mask] = np.real(covariance[:, 0, 0])

        features = features.reshape((batch_size, self.users_per_ap, self.n_ris, -1))
        features = torch.tensor(features.transpose((0, 2, 1, 3)), dtype=torch.float32)
        features = F.normalize(features, dim=2)
        edges = torch.tensor(edges.transpose((0, 2, 1)), dtype=torch.float32)
        direct_edges = torch.tensor(direct_edges, dtype=torch.float32).unsqueeze(1)
        return features, edges, mask, direct_edges, station.test_user_mask

    def training_batch(self, users_per_ap, ratio, testing_ratio):
        self.associate_users(users_per_ap, ratio, testing_ratio)
        parts = [self._station_data(station) for station in self.base_stations]
        features, edges, masks, direct, test_masks = zip(*parts)
        return (
            torch.cat(features, dim=2),
            torch.cat(edges, dim=2),
            np.concatenate(masks, axis=1),
            torch.cat(direct, dim=2),
            list(test_masks),
        )

    def decentralized_batch(self, users_per_ap, ratio, testing_ratio, regenerate_channels=True):
        if regenerate_channels:
            self.associate_users(users_per_ap, ratio, testing_ratio)
        parts = [self._station_data(station) for station in self.base_stations]
        features, edges, masks, direct, _ = zip(*parts)
        return list(features), list(edges), list(masks), list(direct)

    def all_channels(self):
        channels = np.concatenate(
            [station.channels for station in self.base_stations], axis=4
        )
        direct = np.concatenate(
            [station.direct_channels for station in self.base_stations], axis=1
        )
        return channels, direct

    def loss(self, beamformer, phase, device):
        channels, direct = self.all_channels()
        return sum_rate_loss(
            beamformer, phase, channels, direct, self.n_ap, device
        )
