import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils_return_indivial_rates import (
    cal_rates,
    cal_loss,
    gen_square_location,
    generate_channel,
    wrap_around_distances,
)


DEFAULT_SQUARE_SIDE = 100 * np.sqrt(np.pi)


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
    def __init__(
        self,
        M,
        batch_size,
        num_ap=5,
        square_side=DEFAULT_SQUARE_SIDE,
        topology_seed=0,
    ):
        super().__init__()
        if num_ap <= 0 or square_side <= 0:
            raise ValueError("num_ap and square_side must be positive")
        self.M = M
        self.batch_size = batch_size
        self.num_ap = num_ap
        self.square_side = square_side
        self.topology_seed = topology_seed
        topology_rng = np.random.default_rng(topology_seed)
        self.BS_Loc_array = gen_square_location(
            num_ap, square_side, topology_rng
        )
        self.BS_array = [
            Base_station(M, location) for location in self.BS_Loc_array
        ]
        self.user_loc = None
        self.received_power = None
        self.association_mask = None
        self.K = None

    def BS_user_association(self, K, ratio):
        self.K = K
        self.user_loc = gen_square_location(K, self.square_side)
        RSSI = np.zeros((self.batch_size, K, len(self.BS_array)))

        for ap, base_station in enumerate(self.BS_array):
            channel_bs_user = generate_channel(
                self.M,
                K,
                self.batch_size,
                base_station.get_loc(),
                self.user_loc,
                self.square_side,
            )
            base_station.set_channel(channel_bs_user)
            RSSI[:, :, ap] = np.sum(
                np.abs(channel_bs_user) ** 2, axis=2
            ).real

        strongest = np.max(RSSI, axis=2, keepdims=True)
        self.received_power = RSSI
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

    def get_topology_batch(self):
        if self.user_loc is None or self.received_power is None:
            raise RuntimeError("No stored topology batch")
        distances = wrap_around_distances(
            self.user_loc, self.BS_Loc_array, self.square_side
        )
        return {
            "user_locations": self.user_loc.copy(),
            "nearest_ap_distance": distances.min(axis=1),
            "strongest_received_power": self.received_power.max(axis=2),
        }

    def compute_loss(self, W, device, noise_power=None):
        kwargs = {} if noise_power is None else {"noise_power": noise_power}
        return cal_loss(
            W,
            self.get_stacked_channels(),
            len(self.BS_array),
            device,
            **kwargs,
        )

    def compute_rates(self, W, device, noise_power=None):
        kwargs = {} if noise_power is None else {"noise_power": noise_power}
        return cal_rates(
            W,
            self.get_stacked_channels(),
            len(self.BS_array),
            device,
            **kwargs,
        )


MyDataLoader = SnapshotEnvironment
