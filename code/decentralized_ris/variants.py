"""Per-RIS action representation variants for the decentralized RIS GNN.

The baseline model makes every AP emit a 4N latent phase feature and lets a
CPU-side trainable `RIS_merge.f_merge` decode the summed features into a phase.
This module keeps the paper's centralized-training / decentralized-inference
protocol and its eq. (10) local-CSI visibility, and replaces only the RIS
representation, the AP output interface and the aggregation:

    arch      RIS path                       AP output        aggregation
    ------------------------------------------------------------------------
    r0        RIS message-passing node       4N latent        CPU W_reduce
    r0c       RIS message-passing node       2N raw logits    sum, then project
    r1_shared RIS message-passing node       2N unit proposal circular mean
    r1_ap_ris_mag as r1_shared, plus one     2N+1 (proposal   magnitude-weighted
              importance weight per AP-RIS   + weight)        circular mean
    r1        RIS message-passing node       2N unit proposal circular mean
              with AP-local heads (capacity control)
    r3a       node-free per-RIS link tokens  2N proposal      parameter-free
    r3b       r3a + parameter-free per-RIS   2N proposal      parameter-free
              context in the AP-UE update

`r0` reuses `model`'s submodules under their original attribute names, so an
existing checkpoint remains compatible.

Shapes follow `model`: B batch, R RIS, K total AP-UE nodes (= AP x users
per AP), A APs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model
from model import _initial_layer, _max_excluding_self, _node_update_layer

ARCHS = ("r0", "r0c", "r1_shared", "r1_ap_ris_mag", "r1", "r3a", "r3b")
IDENTITIES = ("none", "scalar", "onehot", "learned", "physical")
CONSENSUS = ("wreduce", "equal", "confidence", "ap_ris_mag")

_RAW_LOGIT_ARCHS = ("r0", "r0c")
_SHARED_REDUCTION_ARCHS = ("r0", "r0c", "r1_shared", "r1_ap_ris_mag")
_LOCAL_PROJECTION_ARCHS = ("r1_shared", "r1_ap_ris_mag")
_RIS_NODE_ARCHS = _SHARED_REDUCTION_ARCHS + ("r1",)

_LEARNED_ID_DIM = 8
_PHYSICAL_ID_DIM = 4


def identity_dim(kind, n_ris):
    return {"none": 0, "scalar": 1, "onehot": n_ris,
            "learned": _LEARNED_ID_DIM, "physical": _PHYSICAL_ID_DIM}[kind]


def default_consensus(arch):
    if arch in _RAW_LOGIT_ARCHS:
        return "wreduce"
    return "ap_ris_mag" if arch == "r1_ap_ris_mag" else "equal"


def unit_modulus_error(theta):
    """max | |theta| - 1 | over the (re, im) last axis of a (..., N, 2) tensor."""
    return (theta.norm(dim=-1) - 1.0).abs().max().item()


def _unit_from_pairs(flat, n_ris, n_elem):
    """(B, R, 2N) real/imag readout -> (B, R, N, 2) with exactly unit modulus.

    `F.normalize` returns the zero vector when an element's (re, im) pair is
    degenerate, which would break the unit-modulus gate, so those elements fall
    back to phase 0 instead.
    """
    pair = torch.stack((flat[..., :n_elem], flat[..., n_elem:]), dim=-1)   # (B, R, N, 2)
    norm = pair.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(pair)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1e-12, pair / norm.clamp(min=1e-12), fallback)


def circular_consensus(proposals, active, logits=None, tau=1.0, weights=None):
    """Parameter-free consensus over AP proposals.

    proposals: (B, A, R, N, 2) unit-modulus per-AP proposals.
    active:    (B, A) 1.0 where the AP contributes a proposal.
    logits:    (B, A, R) AP confidences, or None for equal weighting.
    weights:   (B, A, R) non-negative importance weights used directly, in
               place of the softmax over `logits`.  The consensus direction is
               invariant to their overall scale.

    Returns the consensus phase (B, R, N, 2) and the weights (B, A, R).
    """
    b, n_ap, n_ris = proposals.shape[0], proposals.shape[1], proposals.shape[2]
    act = active.unsqueeze(2).expand(b, n_ap, n_ris)                       # (B, A, R)
    if weights is not None:
        w = weights * act
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    elif logits is None:
        w = act / act.sum(dim=1, keepdim=True).clamp(min=1e-12)
    else:
        masked = logits / tau
        masked = masked.masked_fill(act <= 0, torch.finfo(masked.dtype).min)
        w = torch.softmax(masked, dim=1) * act
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    resultant = (proposals * w[..., None, None]).sum(dim=1)                # (B, R, N, 2)
    norm = resultant.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(resultant)
    fallback[..., 0] = 1.0
    theta = torch.where(norm > 1e-12, resultant / norm.clamp(min=1e-12), fallback)
    return theta, w


class RISIdentity(nn.Module):
    """The per-RIS descriptor p_r appended to every link token.

    Every kind keeps r as a structural axis; they differ only in what extra
    coordinates the encoder sees. `physical` is the only one that varies with
    the channel realization, and it is computed from the masked edge weights of
    the current graph view, so an AP can evaluate it from its own local CSI.
    """

    def __init__(self, kind, n_ris, ris_loc=None):
        super().__init__()
        assert kind in IDENTITIES, kind
        self.kind = kind
        self.n_ris = n_ris
        self.dim = identity_dim(kind, n_ris)
        if kind == "scalar":
            idx = torch.arange(n_ris, dtype=torch.float32) / max(n_ris - 1, 1)
            self.register_buffer("code", idx.unsqueeze(1))
        elif kind == "onehot":
            self.register_buffer("code", torch.eye(n_ris))
        elif kind == "learned":
            self.embed = nn.Embedding(n_ris, _LEARNED_ID_DIM)
        elif kind == "physical":
            loc = torch.zeros(n_ris, 2) if ris_loc is None else torch.as_tensor(
                np.asarray(ris_loc), dtype=torch.float32)
            scale = loc.abs().max().clamp(min=1.0)
            self.register_buffer("loc", loc / scale)

    def forward(self, e_masked, mask):
        """e_masked: (B, R, K) masked edge weights, mask: (B, K). -> (B, R, dim)."""
        b = e_masked.shape[0]
        if self.kind == "none":
            return None
        if self.kind in ("scalar", "onehot"):
            return self.code.unsqueeze(0).expand(b, -1, -1)
        if self.kind == "learned":
            idx = torch.arange(self.n_ris, device=e_masked.device)
            return self.embed(idx).unsqueeze(0).expand(b, -1, -1)
        # physical: fixed geometry plus two scale-free channel-derived coordinates
        energy = e_masked.sum(dim=2)                                        # (B, R)
        share = energy / energy.sum(dim=1, keepdim=True).clamp(min=1e-12)
        seen = (e_masked.abs() > 0).to(e_masked.dtype).sum(dim=2)
        vis = seen / max(e_masked.shape[2], 1)
        loc = self.loc.unsqueeze(0).expand(b, -1, -1)
        return torch.cat((loc, share.unsqueeze(2), vis.unsqueeze(2)), dim=2)


class LinkEncoder(nn.Module):
    """phi(h_tilde, e, p_r) -> per-(RIS, AP-UE node) link token, shared over r and k."""

    def __init__(self, in_dim, ch, id_dim):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(in_dim + 3 + id_dim, ch * 2), nn.LeakyReLU(), nn.Linear(ch * 2, ch))

    def forward(self, uf, e_h, e_l, e_dir_n, pr):
        b, n_ris, k = e_h.shape[0], e_h.shape[1], e_h.shape[2]
        feats = [uf, e_h.unsqueeze(3), e_l.transpose(1, 2).unsqueeze(3),
                 e_dir_n.expand(-1, n_ris, -1).unsqueeze(3)]
        if pr is not None:
            feats.append(pr.unsqueeze(2).expand(b, n_ris, k, pr.shape[2]))
        return self.f(torch.cat(feats, dim=3))


class ApNodeUpdateLayer(nn.Module):
    """AP-UE node update without an RIS message-passing node.

    Mirrors `NodeUpdateLayer`'s AP-UE branch (self, element-wise max over the
    other nodes) and optionally a parameter-free per-RIS context that reads the
    link-token bank instead of an RIS state.
    """

    def __init__(self, in_dim, ch, ctx_dim):
        super().__init__()
        self.fu = nn.Sequential(
            nn.Linear(in_dim * 2 + ctx_dim, ch * 2), nn.LeakyReLU(), nn.Linear(ch * 2, ch))

    def forward(self, uk, ctx):
        parts = [uk, _max_excluding_self(uk)]
        if ctx is not None:
            parts.append(ctx)
        return torch.cat((self.fu(torch.cat(parts, dim=2)), uk), dim=2)


class NodeFreePhaseHead(nn.Module):
    """Shared masked pooling + phase head producing one proposal per (AP, RIS)."""

    def __init__(self, ch, node_dim, n_elem):
        super().__init__()
        self.proj_uk = nn.Linear(node_dim, ch)
        self.phi = nn.Sequential(
            nn.Linear(ch * 2, ch * 2), nn.LeakyReLU(), nn.Linear(ch * 2, ch))
        self.phase = nn.Linear(ch * 2, n_elem * 2)
        self.conf = nn.Linear(ch * 2, 1)

    def link_tokens(self, z0, uk):
        """z0: (B, R, K, ch), uk: (B, K, node_dim) -> (B, R, K, ch).

        The tokens depend only on the graph view, not on which AP pools them, so
        the centralized path builds them once and pools them A times.
        """
        n_ris = z0.shape[1]
        u = self.proj_uk(uk).unsqueeze(1).expand(-1, n_ris, -1, -1)
        return self.phi(torch.cat((z0, u), dim=3))

    def forward(self, link, mask):
        """link: (B, R, K, ch) from `link_tokens`, mask: (B, K) over this AP's nodes."""
        m = mask[:, None, :, None]
        cnt = m.sum(dim=2)                                                  # (B, R, 1)
        mean = (link * m).sum(dim=2) / cnt.clamp(min=1.0)
        neg = torch.finfo(link.dtype).min
        mx = torch.where(m > 0, link, torch.full_like(link, neg)).amax(dim=2)
        mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
        pooled = torch.cat((mean, mx), dim=2)                               # (B, R, 2ch)
        return self.phase(pooled), self.conf(pooled).squeeze(2)


class ApConfidenceHead(nn.Module):
    """Scalar per-(AP, RIS) confidence for the node-retained arches."""

    def __init__(self, n_elem):
        super().__init__()
        self.f = nn.Linear(n_elem * 4, 1)

    def forward(self, latent):
        return self.f(latent).squeeze(2)


class VariantNet(nn.Module):
    """GNN with a configurable RIS representation and action path."""

    def __init__(self, M, N, L, D, Pmax, ch, AP, device,
                 arch="r0", identity="none", consensus=None, tau=1.0, ris_loc=None,
                 users_per_ap=8):
        super().__init__()
        assert arch in ARCHS, arch
        if consensus is None:
            consensus = default_consensus(arch)
        assert consensus in CONSENSUS, consensus
        if arch in _RAW_LOGIT_ARCHS and consensus != "wreduce":
            raise ValueError("r0/r0c use the shared reduction head; set consensus=wreduce")
        if arch not in _RAW_LOGIT_ARCHS and consensus == "wreduce":
            raise ValueError(
                "the proposal arches replace the CPU decoder; use equal, "
                "confidence, or ap_ris_mag"
            )
        if arch == "r1_shared" and consensus != "equal":
            raise ValueError("r1_shared is the clean projection-order control; use consensus=equal")
        if arch == "r1_ap_ris_mag" and consensus != "ap_ris_mag":
            raise ValueError(
                "r1_ap_ris_mag is defined by its magnitude weighting; "
                "use consensus=ap_ris_mag"
            )
        if consensus == "ap_ris_mag" and arch != "r1_ap_ris_mag":
            raise ValueError(
                "consensus=ap_ris_mag needs the shared pre-projection logits "
                "of r1_ap_ris_mag"
            )
        if arch in _RIS_NODE_ARCHS and identity != "none":
            raise ValueError("explicit RIS identity is screened on the node-free arches only")

        self.device = device
        self.M, self.N, self.L, self.D, self.ch, self.AP = M, N, L, D, ch, AP
        self.Pmax = Pmax
        self.arch, self.consensus, self.tau = arch, consensus, tau
        self.ris_node = arch in _RIS_NODE_ARCHS
        self.ris_ctx = arch == "r3b"
        self.in_dim = 2 * M * (N + 1)
        node_dim = ch * (D + 1)

        # --- AP-UE (beamforming) path, shared by every arch -------------------
        if self.ris_node:
            self.init_user = model.InitialLayer(M, N, L, ch, device)
            self.update_list = nn.ModuleList(
                [model.NodeUpdateLayer(ch * (d + 1), M, N, L, ch, device) for d in range(D)])
        else:
            self.identity = RISIdentity(identity, L, ris_loc)
            self.link_encoder = LinkEncoder(self.in_dim, ch, self.identity.dim)
            self.node_pool = nn.Linear(ch * 2, ch)
            ctx_dim = ch if self.ris_ctx else 0
            self.update_list = nn.ModuleList(
                [ApNodeUpdateLayer(ch * (d + 1), ch, ctx_dim) for d in range(D)])

        self.AP_coeff_NN_list = nn.ModuleList(
            [model.PowerControl(M, N, L, Pmax, node_dim) for _ in range(AP)])
        self.BS_readout = model.BeamformerReadout(M, N, L, Pmax, node_dim)

        # --- RIS action path --------------------------------------------------
        if self.ris_node:
            self.RIS_readout_AP_list = nn.ModuleList(
                [model.RisReadoutAp(M, N, L, Pmax, node_dim, L * users_per_ap) for _ in range(AP)])
            if arch in _SHARED_REDUCTION_ARCHS:
                self.RIS_merge = model.RisMerge(N)
            elif consensus == "confidence":
                self.conf_head_list = nn.ModuleList([ApConfidenceHead(N) for _ in range(AP)])
        else:
            self.phase_head = NodeFreePhaseHead(ch, node_dim, N)

    # ------------------------------------------------------------------ utils
    def cpu_trainable_parameters(self):
        """Parameters that must run on the CPU (central unit) at inference time."""
        if self.arch == "r0":
            return sum(p.numel() for p in self.RIS_merge.parameters())
        return 0

    def unused_parameters(self):
        """Parameters no forward path reaches, so `total - unused` compares capacity fairly."""
        dead = []
        for name, module in self.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if (self.arch in _SHARED_REDUCTION_ARCHS
                    and leaf == "f_merge" and name.startswith("RIS_readout_AP_list")):
                dead.append(name)                      # these arches use one shared reduction head
            if not self.ris_node and name == "phase_head.conf" and self.consensus != "confidence":
                dead.append(name)
        return sum(p.numel() for n in dead for p in dict(self.named_modules())[n].parameters())

    def describe(self):
        total = sum(p.numel() for p in self.parameters())
        unused = self.unused_parameters()
        return {"arch": self.arch, "consensus": self.consensus,
                "effective_parameters": total - unused, "unused_parameters": unused,
                "identity": getattr(self, "identity", None).kind if not self.ris_node else "none",
                "total_parameters": total, "cpu_trainable_parameters": self.cpu_trainable_parameters(),
                "ap_to_cpu_reals_per_ap_ris": (4 * self.N if self.arch == "r0"
                                               else 2 * self.N
                                               + (1 if self.consensus in ("confidence", "ap_ris_mag") else 0))}

    # ------------------------------------------------------------- shared body
    def _backbone(self, uf, e_m, e_dir_m, mask):
        """Run the AP-UE message passing for one graph view.

        Returns the final node states, the RIS states (node-retained arches) and
        the link-token bank (node-free arches).
        """
        if self.ris_node:
            uk, rl = _initial_layer(self.init_user, uf, e_m, e_dir_m)
            for layer in self.update_list:
                uk, rl = _node_update_layer(layer, uk, rl, e_m, e_dir_m)
            return uk, rl, None

        e_h = F.normalize(e_m, p=1, dim=2)                                  # (B, R, K) over nodes
        e_l = F.normalize(e_m.transpose(2, 1), p=1, dim=2)                  # (B, K, R) over RIS
        e_dir_n = F.normalize(e_dir_m, p=1, dim=2)                          # (B, 1, K)
        pr = self.identity(e_m, mask)
        z0 = self.link_encoder(uf, e_h, e_l, e_dir_n, pr)                   # (B, R, K, ch)
        pooled = torch.cat((z0.mean(dim=1), z0.amax(dim=1)), dim=2)         # (B, K, 2ch)
        uk = self.node_pool(pooled)
        ctx = torch.einsum("bkr,brkc->bkc", e_l, z0) if self.ris_ctx else None
        for layer in self.update_list:
            uk = layer(uk, ctx)
        return uk, None, z0

    def _ap_proposal(self, uk, rl, link, e_m, ap_index, k_user, ap_mask):
        """One AP's RIS output: a shared-head latent, or a local-head proposal."""
        if self.ris_node:
            e_ap = e_m[:, :, ap_index * k_user:(ap_index + 1) * k_user].reshape(e_m.shape[0], -1)
            latent = self.RIS_readout_AP_list[ap_index](rl, e_ap)           # (B, R, 4N)
            if self.arch in _SHARED_REDUCTION_ARCHS:
                return latent, None, None
            flat = self.RIS_readout_AP_list[ap_index].f_merge(latent)       # (B, R, 2N)
            conf = (self.conf_head_list[ap_index](latent)
                    if self.consensus == "confidence" else None)
            return None, _unit_from_pairs(flat, self.L, self.N), conf
        flat, conf = self.phase_head(link, ap_mask)
        return None, _unit_from_pairs(flat, self.L, self.N), conf

    def _merge(self, latents, proposals, confs, active, trace=None):
        if self.arch == "r0":
            return self.RIS_merge(latents)
        if self.arch == "r0c":
            return self.RIS_merge.forward_commuted(latents)
        z_pairs, pair_weights = None, None
        if self.arch in _LOCAL_PROJECTION_ARCHS:
            # The same z_l = W_reduce q_l + b/A used by r0c, with the only
            # intervention being unit projection before rather than after sum.
            # These AP-side operations are batched here only for simulator
            # convenience; the modeled communication boundary is their output.
            flat = torch.stack(self.RIS_merge.local_logits(latents), dim=1)
            proposals = _unit_from_pairs(flat, self.L, self.N)
            # Keep all A terms in this controlled ablation.  An inactive AP has
            # a zeroed q_l and contributes only the evenly split learned bias.
            consensus_active = torch.ones_like(active)
            if trace is not None or self.arch == "r1_ap_ris_mag":
                # The pre-projection logit whose magnitude local projection
                # discards; the diagnostics test whether it carries reliability.
                z_pairs = torch.stack((flat[..., :self.N], flat[..., self.N:]), dim=-1)
            if self.arch == "r1_ap_ris_mag":
                # s_{l,r} = mean_n ||z_{l,r,n}||: one scalar per AP-RIS that the
                # AP computes from its own logits, so no parameter is added and
                # the transmitted phase stays unit modulus.  Call it an
                # importance weight, not a confidence: it tracks leave-one-AP-out
                # marginal contribution, not agreement with the other APs.
                pair_weights = z_pairs.norm(dim=-1).mean(dim=3)             # (B, A, R)
            if trace is None:
                z_pairs = None
        else:
            proposals = torch.stack(proposals, dim=1)                       # (B, A, R, N, 2)
            consensus_active = active
        logits = torch.stack(confs, dim=1) if self.consensus == "confidence" else None
        theta, w = circular_consensus(
            proposals, consensus_active, logits, self.tau, pair_weights
        )
        if trace is not None:
            trace.update(proposals=proposals, weights=w, active=consensus_active,
                         logits=logits, z_pairs=z_pairs, latents=latents)
            if self.arch in _LOCAL_PROJECTION_ARCHS:
                trace["source_active"] = active
            if self.consensus == "confidence":
                # Same proposals, equal weights: isolates the consensus rule from
                # the representation that produced the proposals.
                trace["theta_equal"], _ = circular_consensus(
                    proposals, consensus_active, None, self.tau
                )
        return theta

    # --------------------------------------------------------------- forwards
    def forward(self, user_feature, e, user_index, e_dir, training=True):
        if training:
            return self.centralized(user_feature, e, user_index, e_dir)
        return self.decentralized(user_feature, e, user_index, e_dir)

    def centralized(self, user_feature, e, user_index, e_dir, trace=None):
        device = self.device
        mask = torch.as_tensor(user_index, dtype=torch.bool, device=device)
        m_f = mask.to(user_feature.dtype)
        b, k_tot = mask.shape
        n_ap = self.AP
        k_user = k_tot // n_ap

        uf = user_feature * m_f[:, None, :, None]
        e_m = e * m_f[:, None, :]
        e_dir_m = e_dir * m_f[:, None, :]

        uk, rl, z0 = self._backbone(uf, e_m, e_dir_m, m_f)
        link = None if self.ris_node else self.phase_head.link_tokens(z0, uk)

        alphas, latents, proposals, confs = [], [], [], []
        for l in range(n_ap):
            ap_mask = torch.zeros_like(m_f)
            ap_mask[:, l * k_user:(l + 1) * k_user] = m_f[:, l * k_user:(l + 1) * k_user]
            alphas.append(self.AP_coeff_NN_list[l](uk))
            lat, prop, conf = self._ap_proposal(uk, rl, link, e_m, l, k_user, ap_mask)
            latents.append(lat)
            proposals.append(prop)
            confs.append(conf)

        alpha = torch.cat(alphas, dim=1)                                    # (B, A)
        W = self.BS_readout(uk) * m_f[:, None, :]
        two_m = W.shape[1]
        blk = W.reshape(b, two_m, n_ap, k_user).permute(0, 2, 1, 3).reshape(b, n_ap, -1)
        blk = F.normalize(blk, dim=2, eps=1e-8) * torch.sqrt(self.Pmax * alpha).unsqueeze(2)
        W = blk.reshape(b, n_ap, two_m, k_user).permute(0, 2, 1, 3).reshape(b, two_m, k_tot)

        # Every AP is present in the centralized view, matching the baseline.
        active = torch.ones((b, n_ap), device=device, dtype=W.dtype)
        theta = self._merge(latents, proposals, confs, active, trace)
        return W, theta

    def decentralized(self, user_feature, e, user_index, e_dir, trace=None):
        """Eq. (10) local-CSI inference: AP l sees link (l', k) iff both serve k.

        Pass a dict as `trace` to capture the per-AP proposals and the consensus
        weights for the Stage 2.5 confidence diagnostics.
        """
        device = self.device
        n_ap = len(user_feature)
        b, k_user = user_feature[0].shape[0], user_feature[0].shape[2]
        k_tot = k_user * n_ap

        ui = torch.stack([torch.as_tensor(np.asarray(u), dtype=torch.bool, device=device)
                          for u in user_index], dim=1)                      # (B, A, K_user)
        uf_all = torch.cat(list(user_feature), dim=2)
        e_all = torch.cat(list(e), dim=2)
        e_dir_all = torch.cat(list(e_dir), dim=2)

        ap_active = ui.any(dim=2)                                           # (B, A)
        W = torch.zeros((b, 2 * self.M, k_tot), device=device)
        latents, proposals, confs = [], [], []

        for l in range(n_ap):
            vis = (ui[:, l:l + 1, :] & ui).reshape(b, k_tot).to(uf_all.dtype)
            uf = uf_all * vis[:, None, :, None]
            e_m = e_all * vis[:, None, :]
            e_dir_m = e_dir_all * vis[:, None, :]
            uk, rl, z0 = self._backbone(uf, e_m, e_dir_m, vis)
            link = None if self.ris_node else self.phase_head.link_tokens(z0, uk)

            served = torch.zeros((b, k_tot), device=device, dtype=uf_all.dtype)
            served[:, l * k_user:(l + 1) * k_user] = ui[:, l, :].to(uf_all.dtype)
            alpha = self.AP_coeff_NN_list[l](uk)
            block = (self.BS_readout(uk) * served[:, None, :])[:, :, l * k_user:(l + 1) * k_user]
            block = block.reshape(b, -1)
            block = F.normalize(block, dim=1, eps=1e-8) * torch.sqrt(self.Pmax * alpha)
            W[:, :, l * k_user:(l + 1) * k_user] = block.reshape(b, 2 * self.M, k_user)

            lat, prop, conf = self._ap_proposal(uk, rl, link, e_m, l, k_user, served)
            act = ap_active[:, l].to(uf_all.dtype)
            # An AP that serves nobody is skipped upstream, so it must not vote.
            latents.append(None if lat is None else lat * act[:, None, None])
            proposals.append(prop)
            confs.append(conf)

        theta = self._merge(latents, proposals, confs, ap_active.to(W.dtype), trace)
        return W, theta
