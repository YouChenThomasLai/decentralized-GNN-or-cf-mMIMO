import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class initial_layer(nn.Module):
    def __init__(self, M, ch):
        super().__init__()
        self.fu = nn.Sequential(
            nn.Linear(2 * M, ch * 2),
            nn.LeakyReLU(),
            nn.Linear(ch * 2, ch),
        )

    def forward(self, user_feature):
        return self.fu(user_feature).mean(dim=1)


class node_update_layer(nn.Module):
    def __init__(self, in_dim, ch):
        super().__init__()
        self.ch = ch
        self.fu = nn.Sequential(
            nn.Linear(in_dim * 2, ch * 2),
            nn.LeakyReLU(),
            nn.Linear(ch * 2, ch),
        )

    def forward(self, uk):
        updates = []
        for user in range(uk.shape[1]):
            if uk.shape[1] == 1:
                max_user = uk[:, user]
            else:
                other_users = torch.cat(
                    (uk[:, :user], uk[:, user + 1 :]), dim=1
                )
                max_user = other_users.amax(dim=1)
            update = self.fu(torch.cat((uk[:, user], max_user), dim=1))
            updates.append(update)
        return torch.cat((torch.stack(updates, dim=1), uk), dim=2)


class BS_readout(nn.Module):
    def __init__(self, M, in_dim):
        super().__init__()
        self.fu = nn.Linear(in_dim, M * 2)

    def forward(self, uk):
        return self.fu(uk).transpose(2, 1)


class coeff_DNN2(nn.Module):
    def __init__(self, in_dim, ch=64):
        super().__init__()
        self.fu = nn.Sequential(
            nn.Linear(in_dim, ch * 2),
            nn.LeakyReLU(),
            nn.Linear(ch * 2, ch),
        )
        self.final = nn.Linear(ch, 1)

    def forward(self, uk):
        coefficient = self.final(self.fu(uk).mean(dim=1))
        return torch.sigmoid(torch.clamp(coefficient, -20, 20))


class node_update(nn.Module):
    def __init__(self, M, D, Pmax, ch, AP, device):
        super().__init__()
        self.M = M
        self.D = D
        self.Pmax = Pmax
        self.ch = ch
        self.AP = AP
        self.device = device
        self.init_user = initial_layer(M, ch)
        self.update_list = nn.ModuleList(
            [node_update_layer(ch * (layer + 1), ch) for layer in range(D)]
        )
        self.AP_coeff_NN_list = nn.ModuleList(
            [coeff_DNN2(ch * (D + 1), ch) for _ in range(AP)]
        )
        self.BS_readout = BS_readout(M, ch * (D + 1))

    def _encode(self, user_feature):
        uk = self.init_user(user_feature)
        for update_layer in self.update_list:
            uk = update_layer(uk)
        return uk

    def _normalize_block(self, block, coefficient):
        flat = F.normalize(block.flatten(start_dim=1), dim=1, eps=1e-8)
        flat = flat * torch.sqrt(self.Pmax * coefficient)
        return flat.reshape_as(block)

    def _centralized_forward(self, user_feature, user_index):
        num_users = user_feature.shape[2] // self.AP
        samples = []
        for sample in range(user_feature.shape[0]):
            uk = self._encode(user_feature[sample].unsqueeze(0))
            W_out = self.BS_readout(uk)
            mask = torch.as_tensor(
                user_index[sample], dtype=torch.bool, device=W_out.device
            )
            blocks = []
            for ap in range(self.AP):
                start = ap * num_users
                stop = (ap + 1) * num_users
                block = W_out[:, :, start:stop]
                block = block * mask[start:stop].reshape(1, 1, -1)
                coefficient = self.AP_coeff_NN_list[ap](uk)
                blocks.append(self._normalize_block(block, coefficient))
            samples.append(torch.cat(blocks, dim=2))
        return torch.cat(samples, dim=0)

    def _decentralized_forward(self, user_feature, user_index):
        num_ap = len(user_feature)
        num_users = user_feature[0].shape[2]
        all_features = torch.cat(user_feature, dim=2)
        samples = []

        for sample in range(all_features.shape[0]):
            blocks = []
            for ap in range(num_ap):
                served = np.asarray(user_index[ap][sample], dtype=bool)
                if not served.any():
                    blocks.append(
                        torch.zeros(
                            1,
                            2 * self.M,
                            num_users,
                            device=all_features.device,
                            dtype=all_features.dtype,
                        )
                    )
                    continue

                visible = np.zeros(num_ap * num_users, dtype=bool)
                served_ids = np.flatnonzero(served)
                visible[ap * num_users + served_ids] = True
                for other_ap in range(num_ap):
                    if other_ap == ap:
                        continue
                    other_mask = np.asarray(
                        user_index[other_ap][sample], dtype=bool
                    )
                    shared_ids = served_ids[other_mask[served_ids]]
                    visible[other_ap * num_users + shared_ids] = True

                local_feature = all_features[sample].clone()
                visible = torch.as_tensor(
                    visible, dtype=torch.bool, device=all_features.device
                )
                local_feature[:, ~visible, :] = 0
                uk = self._encode(local_feature.unsqueeze(0))
                W_out = self.BS_readout(uk)
                start = ap * num_users
                stop = (ap + 1) * num_users
                mask = torch.as_tensor(
                    served, dtype=torch.bool, device=W_out.device
                )
                block = W_out[:, :, start:stop]
                block = block * mask.reshape(1, 1, -1)
                coefficient = self.AP_coeff_NN_list[ap](uk)
                blocks.append(self._normalize_block(block, coefficient))
            samples.append(torch.cat(blocks, dim=2))

        return torch.cat(samples, dim=0)

    def forward(
        self,
        user_feature,
        user_index,
        training=True,
        duplicate=False,
    ):
        if training:
            return self._centralized_forward(user_feature, user_index)
        return self._decentralized_forward(user_feature, user_index)
