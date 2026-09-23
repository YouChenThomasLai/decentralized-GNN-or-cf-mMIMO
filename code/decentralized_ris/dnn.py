"""Conventional non-graph DNN control family for the E10 benchmark.

`variants` keeps the GNN backbone and changes only the RIS action interface.
This module keeps the task, the input tensors, the output constraints and the
unsupervised sum-rate loss, and removes the graph instead: no RIS
message-passing node, no edge-weighted neighbour aggregation, no
max-excluding-self, and no weight sharing over the (RIS, AP-UE node) axes.  The
masked input tensors are flattened into one vector and consumed by plain MLP
blocks, which is the conventional DNN control this literature uses.

    method  view at inference     backbone            RIS action
    ---------------------------------------------------------------------------
    d0      centralized           flat MLP            all R phase vectors at once
    d1      paper-decentralized   AP-shared flat MLP  N angles + energy, consensus

`d0` has no AP-originated RIS message, so it has no decentralized deployment
mode; `supports_decentralized` is False and the paired benchmark reads its
centralized rate only.  `d1` reuses `variants.local_energy` and
`variants.circular_consensus` unchanged, so it deploys behind exactly the G2
AP-to-CPU interface: N phase angles plus one strictly local energy scalar per
AP-RIS pair, fused by a parameter-free circular consensus.

Shapes follow `model` and `variants`: B batch, R RIS, K total AP-UE nodes
(= APs x users per AP), A APs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from variants import (
    _unit_from_pairs,
    circular_consensus,
    decode_phase,
    encode_phase,
    local_energy,
)

DNN_ARCHS = ("d0", "d1")
DNN_CONSENSUS = ("direct",)

_POWER_CH = 64


def is_dnn(arch):
    return arch in DNN_ARCHS


def default_consensus(arch):
    """`d0` reads its phases straight out of the head; `d1` reuses G2's rule."""
    return "energy" if arch == "d1" else "direct"


def flat_input_dim(M, N, n_ris, k_total, arch):
    """Length of the flattened input vector, so the width can be pre-registered."""
    node_dim = 2 * M * (N + 1)
    size = n_ris * k_total * node_dim + 2 * n_ris * k_total + k_total
    return size + (k_total if arch == "d1" else 0)


def flatten_inputs(user_feature, edges, direct_edges):
    """Masked model inputs -> one vector per batch element.

    The three edge tensors enter in the same normalized form the GNN encoder
    uses (`e_h` over nodes, `e_l` over RIS, `e_dir_n` over nodes), so the
    control differs from G2 in representation rather than in input scaling.
    """
    batch = user_feature.shape[0]
    e_h = F.normalize(edges, p=1, dim=2)                                # (B, R, K)
    e_l = F.normalize(edges.transpose(2, 1), p=1, dim=2)                # (B, K, R)
    e_dir_n = F.normalize(direct_edges, p=1, dim=2)                     # (B, 1, K)
    return torch.cat(
        (
            user_feature.reshape(batch, -1),
            e_h.reshape(batch, -1),
            e_l.reshape(batch, -1),
            e_dir_n.reshape(batch, -1),
        ),
        dim=1,
    )


class MlpTrunk(nn.Module):
    """Input projection plus width-preserving blocks, LeakyReLU as in `model`."""

    def __init__(self, in_dim, width, depth):
        super().__init__()
        assert depth >= 1, depth
        layers = [nn.Linear(in_dim, width), nn.LeakyReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.LeakyReLU()]
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class FlatPowerControl(nn.Module):
    """`model.PowerControl` on a flat latent: same shape, clamp and squash."""

    def __init__(self, in_dim, ch=_POWER_CH):
        super().__init__()
        self.fu = nn.Sequential(
            nn.Linear(in_dim, ch * 2), nn.LeakyReLU(), nn.Linear(ch * 2, ch))
        self.final = nn.Linear(ch, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, z):
        out = torch.clamp(self.final(self.fu(z)), -20, 20)
        return self.out_act(out)


class DnnNet(nn.Module):
    """Parameter-matched conventional DNN in the `VariantNet` call signature."""

    def __init__(self, M, N, L, D, Pmax, ch, AP, device,
                 arch="d1", identity="none", consensus=None, tau=1.0, ris_loc=None,
                 users_per_ap=8, width=40, depth=3):
        super().__init__()
        assert arch in DNN_ARCHS, arch
        if consensus is None:
            consensus = default_consensus(arch)
        if consensus != default_consensus(arch):
            raise ValueError(f"{arch} requires consensus={default_consensus(arch)}")
        if identity != "none":
            raise ValueError("the DNN control has no per-RIS identity input")

        self.device = device
        self.M, self.N, self.L, self.AP = M, N, L, AP
        self.Pmax = Pmax
        self.arch, self.consensus, self.tau = arch, consensus, tau
        self.width, self.depth = width, depth
        self.users_per_ap = users_per_ap
        self.k_total = AP * users_per_ap
        self.in_dim = flat_input_dim(M, N, L, self.k_total, arch)
        self.supports_decentralized = arch == "d1"

        self.trunk = MlpTrunk(self.in_dim, width, depth)
        beam_out = 2 * M * (self.k_total if arch == "d0" else users_per_ap)
        self.beam_head = nn.Linear(width, beam_out)
        self.phase_head = nn.Linear(width, L * 2 * N)
        self.AP_coeff_NN_list = nn.ModuleList(
            [FlatPowerControl(width) for _ in range(AP)])

    # ------------------------------------------------------------------ utils
    def cpu_trainable_parameters(self):
        """Neither arm puts a trainable parameter on the central unit."""
        return 0

    def unused_parameters(self):
        return 0

    def describe(self):
        total = sum(p.numel() for p in self.parameters())
        return {
            "arch": self.arch,
            "consensus": self.consensus,
            "effective_parameters": total,
            "unused_parameters": 0,
            "identity": "none",
            "total_parameters": total,
            "cpu_trainable_parameters": 0,
            "ap_to_cpu_reals_per_ap_ris": None if self.arch == "d0" else self.N + 1,
            "deployment": "centralized" if self.arch == "d0" else "paper_decentralized",
            "flat_input_dim": self.in_dim,
            "width": self.width,
            "depth": self.depth,
        }

    # --------------------------------------------------------------- readouts
    def _power_scaled_block(self, raw, served_block, alpha):
        """Mask, L2-normalize and scale one AP's beamformer block, as in `variants`."""
        batch = raw.shape[0]
        block = raw.reshape(batch, 2 * self.M, -1) * served_block[:, None, :]
        block = F.normalize(block.reshape(batch, -1), dim=1, eps=1e-8)
        block = block * torch.sqrt(self.Pmax * alpha)
        return block.reshape(batch, 2 * self.M, -1)

    def _proposals(self, latent):
        batch = latent.shape[0]
        flat = self.phase_head(latent).reshape(batch, self.L, 2 * self.N)
        return _unit_from_pairs(flat, self.L, self.N)

    # --------------------------------------------------------------- forwards
    def forward(self, user_feature, e, user_index, e_dir, training=True):
        if training:
            return self.centralized(user_feature, e, user_index, e_dir)
        return self.decentralized(user_feature, e, user_index, e_dir)

    def centralized(self, user_feature, e, user_index, e_dir, trace=None):
        device = self.device
        mask = torch.as_tensor(user_index, dtype=torch.bool, device=device)
        m_f = mask.to(user_feature.dtype)
        batch, k_total = mask.shape
        n_ap = self.AP
        k_user = k_total // n_ap

        uf = user_feature * m_f[:, None, :, None]
        e_m = e * m_f[:, None, :]
        e_dir_m = e_dir * m_f[:, None, :]
        flat = flatten_inputs(uf, e_m, e_dir_m)

        if self.arch == "d0":
            latent = self.trunk(flat)
            alpha = torch.cat(
                [head(latent) for head in self.AP_coeff_NN_list], dim=1)   # (B, A)
            raw = self.beam_head(latent).reshape(batch, 2 * self.M, k_total)
            beamformer = torch.zeros_like(raw)
            for ap in range(n_ap):
                lo, hi = ap * k_user, (ap + 1) * k_user
                beamformer[:, :, lo:hi] = self._power_scaled_block(
                    raw[:, :, lo:hi], m_f[:, lo:hi], alpha[:, ap:ap + 1]
                )
            return beamformer, self._proposals(latent)

        beamformer = torch.zeros((batch, 2 * self.M, k_total), device=device)
        proposals, energies = [], []
        for ap in range(n_ap):
            ap_mask = torch.zeros_like(m_f)
            lo, hi = ap * k_user, (ap + 1) * k_user
            ap_mask[:, lo:hi] = m_f[:, lo:hi]
            latent = self.trunk(torch.cat((flat, ap_mask), dim=1))
            beamformer[:, :, lo:hi] = self._power_scaled_block(
                self.beam_head(latent), ap_mask[:, lo:hi],
                self.AP_coeff_NN_list[ap](latent),
            )
            proposals.append(self._proposals(latent))
            energies.append(local_energy(e_m, ap_mask, ap, k_user))

        # Every AP is present in the centralized view, matching `variants`.
        active = torch.ones((batch, n_ap), device=device, dtype=beamformer.dtype)
        return beamformer, self._merge(proposals, energies, active, trace)

    def decentralized(self, user_feature, e, user_index, e_dir, trace=None,
                      include_cross_ap_csi=True, phase_codec=True):
        """Eq. (10) local-CSI inference with G2's AP-to-CPU RIS interface."""
        if not self.supports_decentralized:
            raise NotImplementedError(
                "d0 emits every action from one centralized view and has no "
                "AP-originated RIS message; use d1 for decentralized deployment"
            )
        device = self.device
        n_ap = len(user_feature)
        batch, k_user = user_feature[0].shape[0], user_feature[0].shape[2]
        k_total = k_user * n_ap

        ui = torch.stack([torch.as_tensor(np.asarray(u), dtype=torch.bool, device=device)
                          for u in user_index], dim=1)                      # (B, A, K_user)
        uf_all = torch.cat(list(user_feature), dim=2)
        e_all = torch.cat(list(e), dim=2)
        e_dir_all = torch.cat(list(e_dir), dim=2)

        ap_active = ui.any(dim=2)                                           # (B, A)
        beamformer = torch.zeros((batch, 2 * self.M, k_total), device=device)
        proposals, energies = [], []

        for ap in range(n_ap):
            if include_cross_ap_csi:
                vis = (ui[:, ap:ap + 1, :] & ui).reshape(batch, k_total)
            else:
                vis = torch.zeros((batch, k_total), device=device, dtype=torch.bool)
                vis[:, ap * k_user:(ap + 1) * k_user] = ui[:, ap, :]
            vis = vis.to(uf_all.dtype)
            e_m = e_all * vis[:, None, :]
            flat = flatten_inputs(
                uf_all * vis[:, None, :, None], e_m, e_dir_all * vis[:, None, :]
            )

            served = torch.zeros((batch, k_total), device=device, dtype=uf_all.dtype)
            lo, hi = ap * k_user, (ap + 1) * k_user
            served[:, lo:hi] = ui[:, ap, :].to(uf_all.dtype)
            latent = self.trunk(torch.cat((flat, served), dim=1))
            beamformer[:, :, lo:hi] = self._power_scaled_block(
                self.beam_head(latent), served[:, lo:hi],
                self.AP_coeff_NN_list[ap](latent),
            )
            proposals.append(self._proposals(latent))
            energies.append(local_energy(e_m, served, ap, k_user))

        theta = self._merge(
            proposals, energies, ap_active.to(beamformer.dtype), trace,
            phase_codec=phase_codec,
        )
        return beamformer, theta

    def _merge(self, proposals, energies, active, trace=None, phase_codec=False):
        stacked = torch.stack(proposals, dim=1)                             # (B, A, R, N, 2)
        angles = encode_phase(stacked) if phase_codec else None
        if angles is not None:
            stacked = decode_phase(angles)
        weights = torch.stack(energies, dim=1)                              # (B, A, R)
        theta, w = circular_consensus(stacked, active, None, self.tau, weights)
        if trace is not None:
            trace.update(proposals=stacked, weights=w, active=active,
                         logits=None, z_pairs=None, latents=None)
            trace["energy"] = weights
            trace["phase_angles"] = angles
        return theta
