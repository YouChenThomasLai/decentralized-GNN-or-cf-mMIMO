import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LEGACY_UNUSED_PARTS = (".fc.", ".edge_update.")


def load_checkpoint(model, path, map_location):
    """Load current bundles and old state dicts without accepting unknown keys."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [
        key for key in incompatible.unexpected_keys
        if not any(part in key for part in LEGACY_UNUSED_PARTS)
    ]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={unexpected}"
        )
    return payload


def _initial_layer(layer, user_feature, edges, direct_edges):
    users = layer.fu(user_feature).mean(dim=1)
    ris_edges = F.normalize(edges, p=1, dim=2)
    direct_edges = F.normalize(direct_edges, p=1, dim=2)
    ris_state = torch.matmul(ris_edges, users)
    direct_state = torch.matmul(direct_edges, users)
    ris_state = torch.cat(
        (ris_state, direct_state.expand(-1, ris_state.shape[1], -1)), dim=2
    )
    return users, layer.f(ris_state)


def _max_excluding_self(users):
    if users.shape[1] == 1:
        return users
    top_two, indices = users.topk(2, dim=1)
    node = torch.arange(users.shape[1], device=users.device).view(1, -1, 1)
    return torch.where(indices[:, 0:1, :] == node, top_two[:, 1:2, :], top_two[:, 0:1, :])


def _node_update_layer(layer, users, ris_state, edges, direct_edges):
    user_edges = F.normalize(edges.transpose(2, 1), p=1, dim=2)
    mean_ris = torch.matmul(user_edges, ris_state)
    updated = layer.fu(torch.cat((users, _max_excluding_self(users), mean_ris), dim=2))
    user_update = torch.cat((updated, users), dim=2)

    ris_edges = F.normalize(edges, p=1, dim=2)
    direct_edges = F.normalize(direct_edges, p=1, dim=2)
    ris_message = torch.matmul(ris_edges, users)
    direct_message = torch.matmul(direct_edges, users)
    mean_ris = torch.cat(
        (ris_message, direct_message.expand(-1, ris_message.shape[1], -1)), dim=2
    )
    ris_update = torch.cat((layer.f(torch.cat((ris_state, mean_ris), dim=2)), ris_state), dim=2)
    return user_update, ris_update

class InitialLayer(nn.Module):
    def __init__(self,M,N,L,ch, device):
        super().__init__()
        # self.device = 'cuda'
        self.device = device
        self.ch = ch                                                   #! self.ch = 64
        self.M = M                                                     #! self.M = 4
        self.N = N                                                     #! self.N = 30
        self.L = L                                                     #! self.L = 4
        self.in_dim = 2*M*(N+1)                                        #! self.in_dim = 248

        self.fu = nn.Sequential(
            nn.Linear(self.in_dim,self.ch*2),
            nn.LeakyReLU(),                                        #! Previous: nn.ReLU()
            nn.Linear(self.ch*2,self.ch)
        )

        self.f = nn.Sequential(
            nn.Linear(self.ch*2,self.ch*2),
            nn.LeakyReLU(),                                        #! Previous: nn.ReLU()
            nn.Linear(self.ch*2,self.ch)
        )

    def forward(self, user_feature, edges, direct_edges):
        return _initial_layer(self, user_feature, edges, direct_edges)

class NodeUpdateLayer(nn.Module):
    def __init__(self,in_dim,M,N,L,ch, device):
        super().__init__()
        # self.device = 'cuda'
        self.device = device
        self.M = M
        self.N = N
        self.L = L
        self.ch = ch
        layer = in_dim/ch
        self.in_dim_p = in_dim*3
        self.in_dim_c = in_dim*2
        self.in_dim_l = int(in_dim*2 + self.ch*layer)

        self.fu = nn.Sequential(
            nn.Linear(self.in_dim_p,self.ch*2),
            nn.LeakyReLU(),                                          #! Previous: nn.ReLU()
            nn.Linear(self.ch*2,self.ch)
        )

        self.f = nn.Sequential(
            nn.Linear(self.in_dim_l,self.ch*2),
            nn.LeakyReLU(),                                          #! Previous: nn.ReLU()
            nn.Linear(self.ch*2,self.ch)
        )

    def forward(self, users, ris_state, edges, direct_edges):
        return _node_update_layer(self, users, ris_state, edges, direct_edges)



class BeamformerReadout(nn.Module):
    def __init__(self,M,N,L,Pt,in_dim):
        super().__init__()
        self.M = M                                       # self.M = 4
        self.N = N                                       # self.N = 30
        self.L = L                                       # self.L = 4
        self.Pt = Pt                                     # self.Pt = 10.0
        self.in_dim = in_dim                             # self.in_dim = 256
        self.fu = nn.Linear(self.in_dim,self.M*2)

    def forward(self,uk):

        batch_size = uk.shape[0]
        self.K = uk.shape[1]
                                                        # uk.shape = torch.Size([1, 40, 256])
        W = self.fu(uk)                                 # W.shape = torch.Size([1, 40, 8])
        W = torch.transpose(W,2,1)                      # W.shape = torch.Size([1, 8, 40])

        #! NOTE: the following was previous normalization. Now normalization is done after coeff_DNN2 is called
        # K = W.shape[2]
        # W = F.normalize(W,dim=2)*np.sqrt(self.Pt)/self.K

        return W


class RisReadoutAp(nn.Module):
    def __init__(self,M,N,L,Pt,in_dim, in_dim2):
        super().__init__()
        self.M = M
        self.N = N
        self.L = L
        self.Pt = Pt
        self.in_dim = in_dim
        self.in_dim2 = in_dim2
        self.f = nn.Linear(self.in_dim,self.N*2)
        self.fe_AP = nn.Linear(32,self.N*2)                        #! 32 is hard coded, TODO: make this dynamic
        self.f_merge = nn.Linear(self.N*4,self.N*2)

    def forward(self, rl, e_AP):
        if e_AP.dim() == 1:                                        # handle (in_dim2,)
            e_AP = e_AP.unsqueeze(0)                               # (1, in_dim2)

        Rl = self.f(rl)                                            # (B, L, 2N)
        Rl_AP = self.fe_AP(e_AP)                                   # (B, 2N)
        Rl_AP = Rl_AP.unsqueeze(1).expand(-1, rl.size(1), -1)      # (B, L, 2N)
        Rl_custom = torch.cat((Rl, Rl_AP), dim=2)                  # (B, L, 4N)

        return Rl_custom

class RisMerge(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.f_merge = nn.Linear(self.N*4,self.N*2)

    def forward(self,Rl_list):
        """
        Rl_list: list of (B, L, 4N) tensors (from APs)
        return: (B, L, N, 2) final phase
        """
        # Merge across APs
        Rl = torch.stack(Rl_list, dim=0).sum(dim=0)                # (B, L, 4N)
        Rl = self.f_merge(Rl)                                      # (B, L, 2N)

        phase_re = Rl[:,:,:self.N].unsqueeze(3)
        phase_im = Rl[:,:,self.N:].unsqueeze(3)
        phase = torch.cat((phase_re,phase_im),dim=3)
        phase = F.normalize(phase,dim=3)

        return phase


class PowerControl(nn.Module):
    def __init__(self,M,N,L,Pt,in_dim,ch=64):
        super().__init__()
        self.M = M                                  # self.M = 4
        self.N = N                                  # self.N = 30
        self.L = L                                  # self.L = 4
        self.Pt = Pt                                # self.Pt = 10.0
        self.in_dim = in_dim                        # self.in_dim = 256 (TODO: double check)
        self.ch = ch

        self.fu = nn.Sequential(
            nn.Linear(self.in_dim,self.ch*2),
            nn.LeakyReLU(),                                 #! Previous: nn.ReLU()
            nn.Linear(self.ch*2,self.ch)
        )

        # final scalar output
        self.final = nn.Linear(self.ch, 1)
        self.out_act = nn.Sigmoid()                         # squash to (0,1)

    def forward(self,uk):
        batch_size = uk.shape[0]
        self.K = uk.shape[1]                                # TODO: check if this is redundant
        W = self.fu(uk)                                     # [B, ?, ch]
        W = W.mean(dim=1)                                   # aggregate → [B, ch]
        out = self.final(W)                                 # [B, 1]
        out = torch.clamp(out, -20, 20)                     # safe range for sigmoid                  #! or else NaNs could occur
        out = self.out_act(out)                             # squash scalar → [0,1]

        return out


class BaselineNet(nn.Module):
    def __init__(self, M, N, L, D, Pmax, ch, AP, device):
        super().__init__()
        self.device = device
        self.N = N
        self.D = D
        self.M = M
        self.L = L
        self.ch = ch
        self.AP = AP
        self.Pmax = Pmax
        self.init_user = InitialLayer(M, N, L, ch, device)
        self.update_list = nn.ModuleList(
            [NodeUpdateLayer(ch * (depth + 1), M, N, L, ch, device) for depth in range(D)]
        )
        node_dim = ch * (D + 1)
        self.AP_coeff_NN_list = nn.ModuleList(
            [PowerControl(M, N, L, Pmax, node_dim) for _ in range(AP)]
        )
        self.RIS_readout_AP_list = nn.ModuleList(
            [RisReadoutAp(M, N, L, Pmax, node_dim, in_dim2=None) for _ in range(AP)]
        )
        self.RIS_merge = RisMerge(N)
        self.BS_readout = BeamformerReadout(M, N, L, Pmax, node_dim)

    def forward(self, user_feature, edges, user_index, direct_edges, training=True, duplicate=False):
        if duplicate:
            raise NotImplementedError("duplicate pruning is not supported")
        if training:
            return self.centralized(user_feature, edges, user_index, direct_edges)
        return self.decentralized(user_feature, edges, user_index, direct_edges)

    def centralized(self, user_feature, edges, user_index, direct_edges):
        mask = torch.as_tensor(user_index, dtype=torch.bool, device=self.device)
        mask_float = mask.to(user_feature.dtype)
        batch, total_nodes = mask.shape
        users_per_ap = total_nodes // self.AP

        user_feature = user_feature * mask_float[:, None, :, None]
        edges = edges * mask_float[:, None, :]
        direct_edges = direct_edges * mask_float[:, None, :]
        users, ris_state = _initial_layer(self.init_user, user_feature, edges, direct_edges)
        for layer in self.update_list:
            users, ris_state = _node_update_layer(layer, users, ris_state, edges, direct_edges)

        alpha = torch.cat([head(users) for head in self.AP_coeff_NN_list], dim=1)
        beamformer = self.BS_readout(users) * mask_float[:, None, :]
        width = beamformer.shape[1]
        blocks = beamformer.reshape(batch, width, self.AP, users_per_ap)
        blocks = blocks.permute(0, 2, 1, 3).reshape(batch, self.AP, -1)
        blocks = F.normalize(blocks, dim=2, eps=1e-8)
        blocks = blocks * torch.sqrt(self.Pmax * alpha).unsqueeze(2)
        beamformer = blocks.reshape(batch, self.AP, width, users_per_ap)
        beamformer = beamformer.permute(0, 2, 1, 3).reshape(batch, width, total_nodes)

        ris_parts = []
        for ap_index, head in enumerate(self.RIS_readout_AP_list):
            start = ap_index * users_per_ap
            edge_block = edges[:, :, start:start + users_per_ap].reshape(batch, -1)
            ris_parts.append(head(ris_state, edge_block))
        return beamformer, self.RIS_merge(ris_parts)

    def decentralized(
        self,
        user_feature,
        edges,
        user_index,
        direct_edges,
        include_cross_ap_csi=True,
    ):
        n_ap = len(user_feature)
        batch, _, users_per_ap = user_feature[0].shape[:3]
        total_nodes = users_per_ap * n_ap
        user_masks = torch.stack(
            [torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.device)
             for mask in user_index],
            dim=1,
        )
        all_features = torch.cat(list(user_feature), dim=2)
        all_edges = torch.cat(list(edges), dim=2)
        all_direct_edges = torch.cat(list(direct_edges), dim=2)
        ap_active = user_masks.any(dim=2)
        beamformer = torch.zeros((batch, 2 * self.M, total_nodes), device=self.device)
        ris_parts = []

        for ap_index in range(n_ap):
            if include_cross_ap_csi:
                visible = user_masks[:, ap_index:ap_index + 1, :] & user_masks
            else:
                visible = torch.zeros_like(user_masks)
                visible[:, ap_index, :] = user_masks[:, ap_index, :]
            visible = visible.reshape(batch, total_nodes).to(all_features.dtype)
            masked_edges = all_edges * visible[:, None, :]
            masked_direct = all_direct_edges * visible[:, None, :]
            users, ris_state = _initial_layer(
                self.init_user,
                all_features * visible[:, None, :, None],
                masked_edges,
                masked_direct,
            )
            for layer in self.update_list:
                users, ris_state = _node_update_layer(
                    layer, users, ris_state, masked_edges, masked_direct
                )

            served = torch.zeros((batch, total_nodes), device=self.device, dtype=all_features.dtype)
            start = ap_index * users_per_ap
            served[:, start:start + users_per_ap] = user_masks[:, ap_index, :].to(all_features.dtype)
            alpha = self.AP_coeff_NN_list[ap_index](users)
            block = (self.BS_readout(users) * served[:, None, :])[:, :, start:start + users_per_ap]
            block = F.normalize(block.reshape(batch, -1), dim=1, eps=1e-8)
            block = block * torch.sqrt(self.Pmax * alpha)
            beamformer[:, :, start:start + users_per_ap] = block.reshape(
                batch, 2 * self.M, users_per_ap
            )

            edge_block = edges[ap_index].reshape(batch, -1)
            part = self.RIS_readout_AP_list[ap_index](ris_state, edge_block)
            ris_parts.append(part * ap_active[:, ap_index].to(part.dtype)[:, None, None])

        return beamformer, self.RIS_merge(ris_parts)
