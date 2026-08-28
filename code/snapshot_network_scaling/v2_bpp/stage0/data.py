from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from geometry import (
    association_diagnostics,
    coordinates_in_square,
    pairwise_wrapped_distances,
    sample_square_bpp,
)
from utils_return_indivial_rates import cal_loss, gen_LOS, generate_channel


class Base_station(Dataset):
    def __init__(self, M, N, L, loc, rng):
        super().__init__()
        self.M = M
        self.N = N
        self.L = L
        self.loc = np.asarray(loc)
        self.RIS_array = []
        self.user_array = None
        self.user_array_for_testing = None
        self.H = None
        self.channel_bs_user = None
        self.reset_los(rng)

    def reset_los(self, rng):
        los = gen_LOS(self.N, self.M, 10, self.L, rng)
        self.LOS_bs_ris = [los] if self.L == 1 else los

    def get_loc(self):
        return self.loc

    def set_RIS(self, ris_array):
        self.RIS_array = list(ris_array)

    def get_RIS_loc(self):
        return np.asarray([ris.get_loc() for ris in self.RIS_array])

    def get_RIS_id(self):
        return [ris.get_id() for ris in self.RIS_array]

    def set_user(self, users):
        self.user_array = users

    def set_user_for_testing(self, users):
        self.user_array_for_testing = users

    def get_user(self):
        return self.user_array

    def get_user_for_testing(self):
        return self.user_array_for_testing

    def get_LOS(self):
        return self.LOS_bs_ris

    def set_channel(self, H, channel_bs_user):
        self.H = H
        self.channel_bs_user = channel_bs_user

    def get_channel(self):
        return self.H, self.channel_bs_user


class RIS(Dataset):
    def __init__(self, ris_id, loc):
        super().__init__()
        self.id = ris_id
        self.loc = np.asarray(loc)

    def get_id(self):
        return self.id

    def get_loc(self):
        return self.loc


class MyDataLoader(Dataset):
    def __init__(
        self,
        M,
        N,
        L,
        batch_size,
        num_ap=5,
        square_side=100 * np.sqrt(np.pi),
        topology_seed=0,
        channel_seed=0,
        record_history=True,
    ):
        super().__init__()
        if min(M, N, L, batch_size, num_ap) <= 0 or square_side <= 0:
            raise ValueError("counts and square_side must be positive")
        self.M = M
        self.N = N
        self.L = L
        self.batch_size = batch_size
        self.num_ap = num_ap
        self.square_side = square_side
        self.length = square_side
        self.topology_seed = topology_seed
        self.channel_seed = channel_seed
        self.record_history = record_history

        topology_rng = np.random.default_rng(topology_seed)
        self.BS_Loc_array = sample_square_bpp(
            num_ap, square_side, topology_rng
        )
        self.RIS_Loc_array = sample_square_bpp(L, square_side, topology_rng)
        self.RIS_array = [
            RIS(ris_id, location)
            for ris_id, location in enumerate(self.RIS_Loc_array)
        ]
        self.channel_rng = np.random.default_rng(channel_seed)
        self.BS_array = [
            Base_station(M, N, L, location, self.channel_rng)
            for location in self.BS_Loc_array
        ]

        self.K = None
        self.user_loc = None
        self.received_power = None
        self.association_mask = None
        self.snapshot_id = 0
        self.ap_user_distances = None
        self.ap_ris_distances = None
        self.ris_user_distances = None
        self._history = []

    def reset_channel_seed(self, seed, clear_history=False):
        self.channel_seed = seed
        self.channel_rng = np.random.default_rng(seed)
        for base_station in self.BS_array:
            base_station.reset_los(self.channel_rng)
        if clear_history:
            self._history.clear()

    def BS_RIS_association(self):
        for base_station in self.BS_array:
            base_station.set_RIS(self.RIS_array)

    def BS_user_association(self, K, ratio, testing_ratio):
        self.K = K
        self.user_loc = sample_square_bpp(
            self.batch_size * K, self.square_side, self.channel_rng
        ).reshape(self.batch_size, K, 2)
        received_power = np.zeros((self.batch_size, K, self.num_ap))
        ap_user_distances = np.zeros_like(received_power)
        ap_ris_distances = np.zeros((self.num_ap, self.L))
        ris_user_distances = None

        for ap, base_station in enumerate(self.BS_array):
            outputs = generate_channel(
                self.M,
                self.N,
                self.L,
                K,
                self.batch_size,
                base_station.get_LOS(),
                0,
                self.RIS_Loc_array,
                base_station.get_loc(),
                self.user_loc,
                self.square_side,
                self.channel_rng,
                return_distances=True,
            )
            H, channel_bs_user, _, _, distances = outputs
            base_station.set_channel(H, channel_bs_user)
            received_power[:, :, ap] = np.sum(
                np.abs(channel_bs_user) ** 2, axis=2
            ).real
            ap_user_distances[:, :, ap] = distances["ap_user"]
            ap_ris_distances[ap] = distances["ap_ris"]
            if ris_user_distances is None:
                ris_user_distances = distances["ris_user"]
            elif not np.array_equal(ris_user_distances, distances["ris_user"]):
                raise RuntimeError("RIS-UE wrapped distances differ by AP")

        strongest = received_power.max(axis=2, keepdims=True)
        self.received_power = received_power
        self.association_mask = received_power >= strongest * ratio
        testing_mask = received_power >= strongest * testing_ratio
        for ap, base_station in enumerate(self.BS_array):
            base_station.set_user(self.association_mask[:, :, ap])
            base_station.set_user_for_testing(testing_mask[:, :, ap])

        self.ap_user_distances = ap_user_distances
        self.ap_ris_distances = ap_ris_distances
        self.ris_user_distances = ris_user_distances
        self.snapshot_id += 1
        if self.record_history:
            self._history.append(
                {
                    "snapshot_id": self.snapshot_id,
                    "user_locations": self.user_loc.copy(),
                    "association_mask": self.association_mask.copy(),
                    "nearest_ap_distance": ap_user_distances.min(axis=2),
                    "nearest_ris_distance": ris_user_distances.min(axis=2),
                    "strongest_received_power": received_power.max(axis=2),
                }
            )

    def load_data(self, base_station):
        H, channel_bs_user = base_station.get_channel()
        user_index = base_station.get_user()
        testing_index = base_station.get_user_for_testing()
        ris_ids = base_station.get_RIS_id()
        batch_size = H.shape[0]

        user_feature = np.zeros(
            (batch_size, self.K, self.L, 2 * self.M, self.N + 1)
        )
        e = np.zeros((batch_size, self.K, self.L))
        e_dir = np.zeros((batch_size, self.K))
        for ris_id in ris_ids:
            temp_H = H[:, :, :, ris_id, :].transpose((0, 3, 1, 2))
            visible_H = temp_H[user_index]
            user_feature[user_index, ris_id, : self.M, : self.N] = (
                visible_H.real
            )
            user_feature[user_index, ris_id, self.M :, : self.N] = (
                visible_H.imag
            )
            e[user_index, ris_id] = np.real(
                np.trace(
                    np.matmul(
                        visible_H, np.conj(visible_H.transpose(0, 2, 1))
                    ),
                    axis1=1,
                    axis2=2,
                )
            )

            direct = channel_bs_user[user_index]
            user_feature[user_index, ris_id, : self.M, self.N] = direct.real
            user_feature[user_index, ris_id, self.M :, self.N] = direct.imag
            if direct.size:
                e_dir[user_index] = np.sum(np.abs(direct) ** 2, axis=1).real

        user_feature = user_feature.reshape(
            batch_size, self.K, self.L, -1
        ).transpose((0, 2, 1, 3))
        user_feature = F.normalize(
            torch.tensor(user_feature, dtype=torch.float32), dim=2
        )
        e = torch.tensor(e.transpose((0, 2, 1)), dtype=torch.float32)
        e_dir = torch.tensor(e_dir, dtype=torch.float32).unsqueeze(1)
        return user_feature, e, user_index, e_dir, testing_index

    def gen_training_data(self, K, ratio, testing_ratio, duplicate=False):
        self.BS_user_association(K, ratio, testing_ratio)
        ap_data = [self.load_data(base_station) for base_station in self.BS_array]
        return (
            torch.cat([data[0] for data in ap_data], dim=2),
            torch.cat([data[1] for data in ap_data], dim=2),
            np.concatenate([data[2] for data in ap_data], axis=1),
            torch.cat([data[3] for data in ap_data], dim=2),
            [data[4] for data in ap_data],
        )

    def gen_testing_data(
        self,
        K,
        ratio,
        testing_ratio,
        duplicate=False,
        regenerate_channels=True,
    ):
        if regenerate_channels:
            self.BS_user_association(K, ratio, testing_ratio)
        elif self.association_mask is None:
            raise RuntimeError("No stored channel batch to reformat")
        ap_data = [self.load_data(base_station) for base_station in self.BS_array]
        return tuple([data[index] for data in ap_data] for index in range(4))

    def _stacked_channels(self):
        channels = [base_station.get_channel() for base_station in self.BS_array]
        return (
            np.concatenate([channel[0] for channel in channels], axis=4),
            np.concatenate([channel[1] for channel in channels], axis=1),
        )

    def get_channel_batch(self):
        return tuple(channel.copy() for channel in self._stacked_channels())

    def compute_loss(self, W, theta, Pmax, device):
        H, channel_bs_user = self._stacked_channels()
        return cal_loss(
            W, theta, H, channel_bs_user, Pmax, self.num_ap, device
        )

    def compute_rates(self, W, theta, Pmax, device):
        H, channel_bs_user = self._stacked_channels()
        return cal_loss(
            W,
            theta,
            H,
            channel_bs_user,
            Pmax,
            self.num_ap,
            device,
            return_raw=True,
        )[3]

    def get_association_mask(self):
        if self.association_mask is None:
            raise RuntimeError("No stored association mask")
        return self.association_mask

    def get_topology_batch(self):
        if self.user_loc is None or self.received_power is None:
            raise RuntimeError("No stored topology batch")
        diagnostics = association_diagnostics(self.association_mask)
        return {
            "user_locations": self.user_loc.copy(),
            "association_mask": self.association_mask.copy(),
            "ap_user_distances": self.ap_user_distances.copy(),
            "ap_ris_distances": self.ap_ris_distances.copy(),
            "ris_user_distances": self.ris_user_distances.copy(),
            "nearest_ap_distance": self.ap_user_distances.min(axis=2),
            "nearest_ris_distance": self.ris_user_distances.min(axis=2),
            "strongest_received_power": self.received_power.max(axis=2),
            **diagnostics,
            "visible_ap_ue_ris_feature_blocks": (
                diagnostics["visible_links_per_ap"] * self.L
            ),
            "global_ris_proposal_count_centralized": np.full(
                self.batch_size, self.num_ap * self.L
            ),
            "global_ris_proposal_count_decentralized": (
                diagnostics["active_ap_count"] * self.L
            ),
        }

    def validate_topology(self):
        expected_ap_user = pairwise_wrapped_distances(
            self.user_loc, self.BS_Loc_array, self.square_side
        )
        expected_ap_ris = pairwise_wrapped_distances(
            self.BS_Loc_array, self.RIS_Loc_array, self.square_side
        )
        expected_ris_user = pairwise_wrapped_distances(
            self.user_loc, self.RIS_Loc_array, self.square_side
        )
        H, direct = self._stacked_channels()
        checks = {
            "object_counts": (
                self.BS_Loc_array.shape == (self.num_ap, 2)
                and self.RIS_Loc_array.shape == (self.L, 2)
                and self.user_loc.shape == (self.batch_size, self.K, 2)
            ),
            "coordinates_in_square": all(
                coordinates_in_square(locations, self.square_side)
                for locations in (
                    self.BS_Loc_array,
                    self.RIS_Loc_array,
                    self.user_loc,
                )
            ),
            "wrap_ap_user": np.allclose(
                expected_ap_user, self.ap_user_distances
            ),
            "wrap_ap_ris": np.allclose(
                expected_ap_ris, self.ap_ris_distances
            ),
            "wrap_ris_user": np.allclose(
                expected_ris_user, self.ris_user_distances
            ),
            "finite_distances": all(
                np.isfinite(values).all()
                for values in (
                    self.ap_user_distances,
                    self.ap_ris_distances,
                    self.ris_user_distances,
                )
            ),
            "finite_channels": np.isfinite(H).all() and np.isfinite(direct).all(),
            "finite_rssi": np.isfinite(self.received_power).all(),
            "each_ue_has_serving_ap": np.all(self.association_mask.any(axis=2)),
            "all_ap_all_ris": all(
                len(base_station.RIS_array) == self.L
                for base_station in self.BS_array
            ),
        }
        checks["wrapped_geometry"] = all(
            checks[name]
            for name in ("wrap_ap_user", "wrap_ap_ris", "wrap_ris_user")
        )
        checks["passed"] = all(checks.values())
        return {name: bool(value) for name, value in checks.items()}

    def save_history(self, path):
        if not self._history:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        np.savez_compressed(
            path,
            snapshot_id=np.concatenate(
                [
                    np.full(self.batch_size, item["snapshot_id"], dtype=int)
                    for item in self._history
                ]
            ),
            user_locations=np.concatenate(
                [item["user_locations"] for item in self._history]
            ),
            association_mask=np.concatenate(
                [item["association_mask"] for item in self._history]
            ),
            nearest_ap_distance=np.concatenate(
                [item["nearest_ap_distance"] for item in self._history]
            ),
            nearest_ris_distance=np.concatenate(
                [item["nearest_ris_distance"] for item in self._history]
            ),
            strongest_received_power=np.concatenate(
                [item["strongest_received_power"] for item in self._history]
            ),
        )
