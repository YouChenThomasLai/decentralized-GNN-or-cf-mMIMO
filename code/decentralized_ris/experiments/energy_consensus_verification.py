"""Independent verification of the energy-weighted RIS consensus comparison.

`experiments/mrc_proxy_diagnostic.py` reported that replacing the learned pair
scale s_{l,r} = mean_n ||z_{l,r,n}|| by the AP-local channel energy
E_{l,r} = sum_{k in K_l} ||H_{l,r,k}||_F^2 in the circular consensus *increases*
the decentralized sum rate.  A result in that direction is exactly the shape a
bug takes, so this script re-derives it from scratch and attacks the four ways
it could be an artifact:

  (1) data inconsistency  - the arms would have to see different channels,
      different beamformers or different proposals;
  (2) information leakage - E_{l,r} would have to depend on something AP l
      cannot see (another AP's CSI, global CSI, or the realized rate);
  (3) weight normalization - the advantage would have to come from the absolute
      numerical scale of the energy rather than the relative weights between
      APs, or from the arms using different normalization / projection / eps;
  (4) implementation error - the reimplemented consensus would have to differ
      from the deployed one, or the weights would have to be misaligned on the
      (batch, AP, RIS) axes.

Nothing here trains.  One frozen checkpoint, one training seed (0), fixed
evaluation seeds.  Every arm in a batch shares one channel realization, one
beamformer and one set of unit-modulus proposals; the *only* thing that changes
between arms is the consensus weight tensor.

Run from `code/decentralized_ris/`:

    python -m experiments.energy_consensus_verification --op main \
        --samples 3200 --eval_seed 20260922

    python -m experiments.energy_consensus_verification --op pmax5 \
        --samples 3200 --eval_seed 20260922
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

import variants
from evaluate import build_model, resolve_device, seed_everything, temporary_seed
from experiments.mrc_proxy_diagnostic import (
    ControlFailure,
    cluster_summary,
    pair_energy,
    paired_difference,
)
from experiments.mrc_proxy_diagnostic import weighted_consensus as reference_consensus
from experiments.proposal_diagnostics import _spearman_along
from model import load_checkpoint
from rates import RatePrecompute, quantize_phase
from simulation import ChannelSimulator


# The two epsilons that appear in the deployed consensus
# (`variants.circular_consensus`) and in the diagnostic's reimplementation.
# Every arm shares them; they are recorded in the payload so a reader can see
# that no arm got its own floor.
WEIGHT_SUM_EPS = 1e-12      # denominator floor when normalizing weights over APs
RESULTANT_EPS = 1e-12       # below this the circular mean falls back to phase 0

# The four arms the verification is *about*, in the order requested.
CORE_ARMS = ("r0c_per_element_mag", "learned_pair_s", "local_energy", "equal")

# Weight-construction probes that isolate normalization from alignment.
PROBE_ARMS = (
    "energy_x1e6",                    # absolute scale up
    "energy_x1e-6",                   # absolute scale down
    "energy_norm_over_ap",            # E / sum_l E   (the axis consensus normalizes)
    "energy_norm_over_ris",           # E / sum_r E   (changes relative AP weights)
    "energy_norm_by_max",             # E / max_{l,r} E  (per-sample rescale)
    "energy_rank_over_ap",            # ordering only, magnitudes destroyed
    "energy_perm_ap",                 # same values, AP assignment shuffled
    "energy_values_by_learned_order", # same values, assigned by learned-s ranking
    "learned_pair_s_active_masked",   # learned s with the proxies' active mask
)

ALL_ARMS = CORE_ARMS + PROBE_ARMS

CONTROL_TOL = {
    "reimplemented_vs_reference_consensus": 0.0,
    "learned_arm_vs_deployed_r1_ap_ris_mag": 1e-6,
    "equal_arm_vs_deployed_r1_shared": 1e-6,
    "per_element_arm_vs_deployed_r0c": 1e-4,
    "r0c_vs_r0_theta": 1e-4,
    "beamformer_identical_across_arms": 0.0,
    "fast_rate_vs_simulator_loss": 1e-3,
    "energy_vs_raw_channel_recompute_rel": 1e-5,
    "energy_unchanged_when_other_aps_corrupted": 0.0,
    # float32 theta tolerances use the repository's own unit-modulus budget
    # (1e-5, ~80 float32 ulp).  The mathematical identity is asserted separately
    # in float64 at 1e-12 and on the rate at 1e-3 bps/Hz, so a real difference
    # cannot hide inside the float32 budget.
    "energy_norm_over_ap_vs_local_energy": 1e-5,
    "energy_scale_invariance": 1e-5,
    "energy_scale_invariance_float64": 1e-12,
    "energy_scale_invariance_rate_bps": 1e-3,
    "determinism_replay": 0.0,
}


# --------------------------------------------------------------- operating points

# Only quantities that leave every learned tensor shape intact can move without
# retraining.  `fe_AP.in_features = L * K` and the phase head's `2N` output bake
# K and N into the checkpoint, so those two are *not* free: changing either one
# would require a new model.  Transmit power is a plain scalar multiplier on the
# beamformer block (`variants.py:505`) and never touches the phase path, and the
# association ratio changes |K_l| without changing any layer shape.
OPERATING_POINTS = {
    "main":    {"pmax_dbm": 15.0, "assoc_threshold": 0.1},
    "pmax5":   {"pmax_dbm": 5.0,  "assoc_threshold": 0.1},
    "pmax25":  {"pmax_dbm": 25.0, "assoc_threshold": 0.1},
    "assoc30": {"pmax_dbm": 15.0, "assoc_threshold": 0.3},
}


# ------------------------------------------------------------------- consensus

def instrumented_consensus(proposals, weights, health, arm):
    """`mrc_proxy_diagnostic.weighted_consensus` with numerical instrumentation.

    The arithmetic is character-for-character the deployed one; the extra code
    only *reads* intermediate tensors.  `reference_consensus` is run alongside
    in the controls and must agree exactly.
    """
    if weights.dim() == 3:
        weights = weights.unsqueeze(-1).expand(proposals.shape[:-1])
    weight_sum = weights.sum(dim=1, keepdim=True)
    w = weights / weight_sum.clamp(min=WEIGHT_SUM_EPS)
    resultant = (proposals * w[..., None]).sum(dim=1)
    norm = resultant.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(resultant)
    fallback[..., 0] = 1.0
    theta = torch.where(norm > RESULTANT_EPS,
                        resultant / norm.clamp(min=RESULTANT_EPS), fallback)

    row = health[arm]
    row["nan_in_weights"] += int((~torch.isfinite(weights)).sum())
    row["nan_in_theta"] += int((~torch.isfinite(theta)).sum())
    row["min_weight_sum"] = min(row["min_weight_sum"], float(weight_sum.min()))
    row["n_weight_sum_at_floor"] += int((weight_sum <= WEIGHT_SUM_EPS).sum())
    row["n_negative_weight"] += int((weights < 0).sum())
    # Near-cancellation: unit proposals under weights that sum to one give a
    # resultant of length <= 1; a length near 0 means the APs cancelled and the
    # emitted phase is numerically arbitrary.
    row["min_resultant_norm"] = min(row["min_resultant_norm"], float(norm.min()))
    row["n_resultant_below_1e-12"] += int((norm <= 1e-12).sum())
    row["n_resultant_below_1e-6"] += int((norm <= 1e-6).sum())
    row["n_resultant_below_1e-3"] += int((norm <= 1e-3).sum())
    row["n_projection_fallback"] += int((norm <= RESULTANT_EPS).sum())
    row["n_elements"] += int(norm.numel())
    row["max_unit_modulus_error"] = max(
        row["max_unit_modulus_error"], float(variants.unit_modulus_error(theta))
    )
    return theta


def consensus_float64(proposals, weights):
    """The same circular mean evaluated in double precision.

    Used only by the scale-invariance control: the identity
    consensus(c * w) == consensus(w) is exact in exact arithmetic, so the
    float32 residual must collapse by ~1e9 here.  If it does not, the arms are
    genuinely different and the float32 agreement was a coincidence.
    """
    p = proposals.double()
    w = weights.double()
    if w.dim() == 3:
        w = w.unsqueeze(-1).expand(p.shape[:-1])
    w = w / w.sum(dim=1, keepdim=True).clamp(min=WEIGHT_SUM_EPS)
    resultant = (p * w[..., None]).sum(dim=1)
    norm = resultant.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(resultant)
    fallback[..., 0] = 1.0
    return torch.where(norm > RESULTANT_EPS,
                       resultant / norm.clamp(min=RESULTANT_EPS), fallback)


def scale_invariance_float64(proposals, energy_masked):
    """max |theta(g(E)) - theta(E)| in float64 over the pure-rescale variants."""
    energy = energy_masked.double()
    base = consensus_float64(proposals, energy)
    rescaled = {
        "x1e6": energy * 1e6,
        "x1e-6": energy * 1e-6,
        "norm_over_ap": energy / energy.sum(dim=1, keepdim=True).clamp(min=1e-30),
        "norm_by_max": energy / energy.amax(dim=(1, 2), keepdim=True).clamp(min=1e-30),
    }
    return max(float((consensus_float64(proposals, w) - base).abs().max())
               for w in rescaled.values())


def new_health():
    return {
        arm: {
            "nan_in_weights": 0, "nan_in_theta": 0, "n_negative_weight": 0,
            "min_weight_sum": math.inf, "n_weight_sum_at_floor": 0,
            "min_resultant_norm": math.inf, "n_resultant_below_1e-12": 0,
            "n_resultant_below_1e-6": 0, "n_resultant_below_1e-3": 0,
            "n_projection_fallback": 0, "n_elements": 0,
            "max_unit_modulus_error": 0.0, "nan_in_rate": 0,
        }
        for arm in ALL_ARMS
    }


# ----------------------------------------------------------------- arm weights

def build_weights(magnitude, learned_s, energy, ap_active, perm_generator, device):
    """Every arm's consensus weight for one batch.

    `magnitude` is (B, A, R, N); everything else is (B, A, R).  `energy` is the
    raw AP-local channel energy, already zero off AP l's association mask.
    """
    active = ap_active.unsqueeze(2)                                     # (B, A, 1)
    energy_masked = energy.clamp(min=0.0) * active                      # the proxy
    n_ap = energy.shape[1]

    weights = {
        # --- the four arms under comparison, in the requested order --------
        "r0c_per_element_mag": magnitude,
        "learned_pair_s": learned_s,
        "local_energy": energy_masked,
        "equal": torch.ones_like(learned_s),
        # --- absolute-scale probes: the consensus must be blind to these ---
        "energy_x1e6": energy_masked * 1e6,
        "energy_x1e-6": energy_masked * 1e-6,
    }

    # E / sum_l E: normalizing along the very axis the consensus normalizes,
    # so this must reproduce `local_energy` exactly.
    over_ap = energy_masked.sum(dim=1, keepdim=True).clamp(min=1e-30)
    weights["energy_norm_over_ap"] = energy_masked / over_ap
    # E / sum_r E: normalizing along the RIS axis instead, which genuinely
    # changes the relative weight between APs at a given RIS.
    over_ris = energy_masked.sum(dim=2, keepdim=True).clamp(min=1e-30)
    weights["energy_norm_over_ris"] = energy_masked / over_ris
    # Per-sample rescale by the largest pair energy: another pure rescale.
    by_max = energy_masked.amax(dim=(1, 2), keepdim=True).clamp(min=1e-30)
    weights["energy_norm_by_max"] = energy_masked / by_max

    # Ordering only: keeps which AP is strongest, throws the magnitudes away.
    order = torch.argsort(energy_masked, dim=1)
    ranks = torch.zeros_like(energy_masked)
    positions = torch.arange(
        n_ap, device=device, dtype=energy_masked.dtype
    ).view(1, n_ap, 1).expand_as(energy_masked)
    ranks.scatter_(1, order, positions)
    weights["energy_rank_over_ap"] = (ranks + 1.0) * active

    # Same multiset of weights, shuffled across APs: destroys the alignment and
    # keeps the sharpness, so it separates "picks the right AP" from "is peaky".
    b, _, n_ris = energy_masked.shape
    perm = torch.stack(
        [torch.stack([torch.randperm(n_ap, generator=perm_generator)
                      for _ in range(n_ris)], dim=1) for _ in range(b)], dim=0
    ).to(device)                                                        # (B, A, R)
    weights["energy_perm_ap"] = torch.gather(energy_masked, 1, perm)

    # Same multiset again, but assigned by the learned scale's own ranking:
    # identical sharpness, learned alignment.
    sorted_values, _ = torch.sort(energy_masked, dim=1, descending=True)
    learned_order = torch.argsort(learned_s, dim=1, descending=True)
    by_learned = torch.zeros_like(energy_masked)
    by_learned.scatter_(1, learned_order, sorted_values)
    weights["energy_values_by_learned_order"] = by_learned

    # The learned scale under the proxies' own active mask, so the inactive-AP
    # convention cannot explain the gap.
    weights["learned_pair_s_active_masked"] = learned_s * active
    return weights


# ------------------------------------------------------------- leakage audit

def locality_audit(simulator, dedges, dmasks, energy, device, rng):
    """Prove E_{l,r} uses only AP l's own CSI and its own served-user set.

    Three checks, all empirical rather than by inspection:
      a. recompute E from the raw complex cascaded channels of AP l alone;
      b. corrupt every *other* AP's channels and show AP l's E does not move;
      c. show the association mask really restricts the sum (using a different
         AP's mask must change E), so K_l is not silently the full user set.
    """
    out = {"per_ap": []}
    worst_recompute, worst_locality = 0.0, 0.0
    mask_sensitive = []
    reference = energy.detach().cpu().numpy().astype(np.float64)        # (B, A, R)

    for index, station in enumerate(simulator.base_stations):
        own_mask = np.asarray(dmasks[index]).astype(bool)               # (B, K)
        # (a) E from |H|^2 summed over antennas and elements, own mask only.
        power = (np.abs(station.channels) ** 2).sum(axis=(1, 2))        # (B, R, K)
        recomputed = (power * own_mask[:, None, :]).sum(axis=2)         # (B, R)
        scale = max(float(np.abs(recomputed).max()), 1e-30)
        recompute_err = float(np.abs(recomputed - reference[:, index, :]).max() / scale)

        # (b) the edge tensor must be zero exactly off AP l's own mask.
        edges = dedges[index].detach().cpu().numpy()                    # (B, R, K)
        off_mask_mass = float(np.abs(edges[:, :, ~own_mask.any(axis=0)]).max()) \
            if (~own_mask.any(axis=0)).any() else 0.0
        off_mask_elementwise = float(
            np.abs(edges * (~own_mask)[:, None, :]).max()
        )

        # (c) a different AP's mask must give a different sum, or the mask is
        #     not doing anything and "local served-user set" would be vacuous.
        other = (index + 1) % len(simulator.base_stations)
        other_mask = np.asarray(dmasks[other]).astype(bool)
        with_other = (power * other_mask[:, None, :]).sum(axis=2)
        mask_sensitive.append(
            float(np.abs(with_other - recomputed).max() / scale)
        )

        worst_recompute = max(worst_recompute, recompute_err)
        out["per_ap"].append({
            "ap": index,
            "max_rel_error_vs_raw_channel_recompute": recompute_err,
            "max_abs_edge_mass_off_own_mask": off_mask_elementwise,
            "max_abs_edge_mass_on_never_served_users": off_mask_mass,
            "served_users_mean": float(own_mask.sum(axis=1).mean()),
            "rel_change_if_another_ap_mask_used": mask_sensitive[-1],
        })

    # (b) hard version: corrupt the other APs' channels, rebuild the per-AP
    # views, and require AP l's energy to be bit-identical.
    saved = [station.channels.copy() for station in simulator.base_stations]
    for index, station in enumerate(simulator.base_stations):
        for other, other_station in enumerate(simulator.base_stations):
            if other == index:
                continue
            noise = rng.normal(size=other_station.channels.shape) \
                + 1j * rng.normal(size=other_station.channels.shape)
            other_station.channels = saved[other] * (1.0 + 10.0 * noise)
        corrupted = simulator.decentralized_batch(
            simulator.users_per_ap, None, None, regenerate_channels=False
        )
        # Compare the float32 edge tensors themselves: identical inputs must
        # give bit-identical outputs, with no precision mismatch to explain away.
        corrupt_edges = corrupted[1][index].to(device)
        worst_locality = max(
            worst_locality,
            float((corrupt_edges - dedges[index]).abs().max()),
        )
        for other, other_station in enumerate(simulator.base_stations):
            other_station.channels = saved[other]

    out["max_rel_error_vs_raw_channel_recompute"] = worst_recompute
    out["max_abs_change_when_other_aps_corrupted"] = worst_locality
    out["min_rel_change_if_another_ap_mask_used"] = float(min(mask_sensitive))
    out["uses_ground_truth_rate"] = False
    out["note"] = (
        "E is a function of (station.channels, station.user_mask) only; it is "
        "computed before any beamformer or rate is evaluated in the loop."
    )
    return out


# ------------------------------------------------------------------ model set-up

def load_policy(run_dir, checkpoint_name, simulator, device, config, arch):
    cfg = dict(config)
    cfg["arch"] = arch
    cfg["consensus"] = variants.default_consensus(arch)
    path = None
    for folder in ("checkpoints", "models"):
        candidate = os.path.join(run_dir, folder, checkpoint_name)
        if os.path.exists(candidate):
            path = candidate
            break
    if path is None:
        raise SystemExit(f"no {checkpoint_name} under {run_dir}")
    net = build_model(cfg, simulator, device)
    load_checkpoint(net, path, device)
    net.eval()
    return net, path


# ------------------------------------------------------------------- main pass

def run_split(nets, simulator, config, device, args, threshold, control_batches):
    net_mag, net_r0, net_r0c, net_r1s = nets
    users = config["K"]
    batch_size = config["batch_size"]
    if args.samples <= 0 or args.samples % batch_size:
        raise SystemExit("samples must be a positive multiple of the batch size")
    n_batches = args.samples // batch_size

    cluster = {arm: [] for arm in ALL_ARMS}
    cluster_2bit = {arm: [] for arm in ALL_ARMS}
    samples = {arm: [] for arm in ALL_ARMS}
    samples_2bit = {arm: [] for arm in ALL_ARMS}
    health = new_health()
    corr_ap, corr_pair = [], []
    per_sample_all_active, per_sample_active_count = [], []
    cluster_index = []
    controls = {key: 0.0 for key in (
        "reimplemented_vs_reference_consensus",
        "learned_arm_vs_deployed_r1_ap_ris_mag",
        "equal_arm_vs_deployed_r1_shared",
        "per_element_arm_vs_deployed_r0c",
        "r0c_vs_r0_theta",
        "beamformer_identical_across_arms",
        "fast_rate_vs_simulator_loss",
        "energy_norm_over_ap_vs_local_energy",
        "energy_scale_invariance",
        "energy_scale_invariance_float64",
        "energy_scale_invariance_rate_bps",
    )}
    controls["control_batches"] = int(control_batches)
    audit = None

    # A private generator for the permutation probe, so the channel RNG stream
    # is untouched and the evaluation samples stay identical across arms.
    perm_generator = torch.Generator().manual_seed(args.perm_seed)
    audit_rng = np.random.default_rng(args.perm_seed + 1)

    with torch.no_grad(), temporary_seed(args.eval_seed):
        for index in range(n_batches):
            simulator.training_batch(users, threshold, threshold)
            dfeatures, dedges, dmasks, ddirect = simulator.decentralized_batch(
                users, threshold, threshold, regenerate_channels=False
            )
            dfeatures = [t.to(device) for t in dfeatures]
            dedges_dev = [t.to(device) for t in dedges]
            ddirect = [t.to(device) for t in ddirect]
            pre = RatePrecompute(simulator, device)

            trace = {}
            beamformer, theta_deployed = net_mag.decentralized(
                dfeatures, dedges_dev, dmasks, ddirect, trace
            )
            proposals = trace["proposals"]                               # (B,A,R,N,2)
            magnitude = trace["z_pairs"].norm(dim=-1)                    # (B,A,R,N)
            learned_s = magnitude.mean(dim=3)                            # (B,A,R)
            ap_active = trace["source_active"]                           # (B,A)

            energy, served, _ = pair_energy(dedges, dmasks, device)      # (B,A,R)
            weights = build_weights(
                magnitude, learned_s, energy, ap_active, perm_generator, device
            )

            thetas = {}
            for arm in ALL_ARMS:
                theta = instrumented_consensus(proposals, weights[arm], health, arm)
                thetas[arm] = theta
                rate = pre.sum_rate(beamformer, theta)
                rate_2bit = pre.sum_rate(beamformer, quantize_phase(theta, 2))
                health[arm]["nan_in_rate"] += int((~torch.isfinite(rate)).sum())
                samples[arm].append(rate.detach().cpu().numpy())
                samples_2bit[arm].append(rate_2bit.detach().cpu().numpy())
                cluster[arm].append(float(rate.mean()))
                cluster_2bit[arm].append(float(rate_2bit.mean()))

            # --- normalization controls, every batch, not just the first ----
            controls["energy_norm_over_ap_vs_local_energy"] = max(
                controls["energy_norm_over_ap_vs_local_energy"],
                float((thetas["energy_norm_over_ap"] - thetas["local_energy"]).abs().max()),
            )
            controls["energy_scale_invariance"] = max(
                controls["energy_scale_invariance"],
                float((thetas["energy_x1e6"] - thetas["local_energy"]).abs().max()),
                float((thetas["energy_x1e-6"] - thetas["local_energy"]).abs().max()),
                float((thetas["energy_norm_by_max"] - thetas["local_energy"]).abs().max()),
            )
            controls["energy_scale_invariance_float64"] = max(
                controls["energy_scale_invariance_float64"],
                scale_invariance_float64(proposals, weights["local_energy"]),
            )
            base_rate = pre.sum_rate(beamformer, thetas["local_energy"])
            controls["energy_scale_invariance_rate_bps"] = max(
                controls["energy_scale_invariance_rate_bps"],
                *[float((pre.sum_rate(beamformer, thetas[arm]) - base_rate).abs().max())
                  for arm in ("energy_x1e6", "energy_x1e-6",
                              "energy_norm_over_ap", "energy_norm_by_max")],
            )
            # The reimplemented consensus must reproduce the deployed weighting
            # bit-for-bit, which also pins the (batch, AP, RIS) alignment.
            controls["learned_arm_vs_deployed_r1_ap_ris_mag"] = max(
                controls["learned_arm_vs_deployed_r1_ap_ris_mag"],
                float((thetas["learned_pair_s"] - theta_deployed).abs().max()),
            )
            for arm in ALL_ARMS:
                controls["reimplemented_vs_reference_consensus"] = max(
                    controls["reimplemented_vs_reference_consensus"],
                    float((thetas[arm] - reference_consensus(
                        proposals, weights[arm])).abs().max()),
                )

            if index < control_batches:
                beamformer_r0, theta_r0 = net_r0.decentralized(
                    dfeatures, dedges_dev, dmasks, ddirect)
                _, theta_r0c = net_r0c.decentralized(
                    dfeatures, dedges_dev, dmasks, ddirect)
                _, theta_r1s = net_r1s.decentralized(
                    dfeatures, dedges_dev, dmasks, ddirect)
                controls["r0c_vs_r0_theta"] = max(
                    controls["r0c_vs_r0_theta"],
                    float((theta_r0c - theta_r0).abs().max()))
                controls["per_element_arm_vs_deployed_r0c"] = max(
                    controls["per_element_arm_vs_deployed_r0c"],
                    float((thetas["r0c_per_element_mag"] - theta_r0c).abs().max()))
                controls["equal_arm_vs_deployed_r1_shared"] = max(
                    controls["equal_arm_vs_deployed_r1_shared"],
                    float((thetas["equal"] - theta_r1s).abs().max()))
                controls["beamformer_identical_across_arms"] = max(
                    controls["beamformer_identical_across_arms"],
                    float((beamformer - beamformer_r0).abs().max()))
                _, reference_rate, _ = simulator.loss(
                    beamformer, thetas["local_energy"], device)
                fast = float(pre.sum_rate(beamformer, thetas["local_energy"]).mean())
                controls["fast_rate_vs_simulator_loss"] = max(
                    controls["fast_rate_vs_simulator_loss"],
                    abs(float(reference_rate) - fast))

            if index == 0 and args.audit:
                audit = locality_audit(
                    simulator, dedges_dev, dmasks, energy, device, audit_rng)

            # --- Spearman(learned s, E), per sample ------------------------
            s_np = learned_s.detach().cpu().numpy().astype(np.float64)
            e_np = weights["local_energy"].detach().cpu().numpy().astype(np.float64)
            across = _spearman_along(
                s_np.transpose(0, 2, 1), e_np.transpose(0, 2, 1), axis=2)  # (B,R)
            corr_ap.append(np.nanmean(across, axis=1))
            corr_pair.append(_spearman_along(
                s_np.reshape(s_np.shape[0], -1),
                e_np.reshape(e_np.shape[0], -1), axis=1))

            active_np = ap_active.detach().cpu().numpy().astype(bool)
            per_sample_all_active.append(active_np.all(axis=1))
            per_sample_active_count.append(active_np.sum(axis=1))
            cluster_index.append(np.full(active_np.shape[0], index, dtype=np.int64))

            if args.progress and (index + 1) % args.progress == 0:
                print(f"  [{index + 1:4d}/{n_batches}] "
                      f"learned={np.mean(cluster['learned_pair_s']):.5f} "
                      f"energy={np.mean(cluster['local_energy']):.5f} "
                      f"equal={np.mean(cluster['equal']):.5f}", flush=True)

    return {
        "cluster": {k: np.asarray(v) for k, v in cluster.items()},
        "cluster_2bit": {k: np.asarray(v) for k, v in cluster_2bit.items()},
        "samples": {k: np.concatenate(v) for k, v in samples.items()},
        "samples_2bit": {k: np.concatenate(v) for k, v in samples_2bit.items()},
        "corr_ap": np.concatenate(corr_ap),
        "corr_pair": np.concatenate(corr_pair),
        "all_active": np.concatenate(per_sample_all_active),
        "active_count": np.concatenate(per_sample_active_count),
        "cluster_index": np.concatenate(cluster_index),
        "controls": controls,
        "health": health,
        "audit": audit,
        "n_batches": n_batches,
    }


# ------------------------------------------------------------------ statistics

def win_loss_tie(a, b, tolerance):
    """Per-sample outcome counts for `a` against `b`."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    ties = int((np.abs(diff) <= tolerance).sum())
    return {
        "n": int(diff.size),
        "wins": int((diff > tolerance).sum()),
        "losses": int((diff < -tolerance).sum()),
        "ties": ties,
        "exact_ties": int((diff == 0.0).sum()),
        "win_fraction": float((diff > tolerance).mean()),
        "tolerance": tolerance,
        "median_diff": float(np.median(diff)),
        "min_diff": float(diff.min()),
        "max_diff": float(diff.max()),
    }


def contrast_table(result, tolerance):
    """Paired differences over the requested comparison order."""
    cluster, sample = result["cluster"], result["samples"]
    pairs = [
        ("local_energy", "learned_pair_s"),
        ("local_energy", "r0c_per_element_mag"),
        ("local_energy", "equal"),
        ("learned_pair_s", "r0c_per_element_mag"),
        ("learned_pair_s", "equal"),
        ("r0c_per_element_mag", "equal"),
        ("local_energy", "learned_pair_s_active_masked"),
        ("local_energy", "energy_norm_over_ris"),
        ("local_energy", "energy_rank_over_ap"),
        ("local_energy", "energy_perm_ap"),
        ("local_energy", "energy_values_by_learned_order"),
        ("energy_values_by_learned_order", "learned_pair_s"),
    ]
    table = {}
    for left, right in pairs:
        key = f"{left}_minus_{right}"
        entry = {
            "cluster": paired_difference(cluster[left], cluster[right]),
            "sample": paired_difference(sample[left], sample[right]),
            "win_loss_tie_per_sample": win_loss_tie(
                sample[left], sample[right], tolerance),
            "win_loss_tie_per_cluster": win_loss_tie(
                cluster[left], cluster[right], tolerance),
        }
        mask = result["all_active"]
        if mask.any() and not mask.all():
            entry["sample_all_active"] = paired_difference(
                sample[left][mask], sample[right][mask])
        table[key] = entry
    return table


def check_controls(controls, audit):
    failures = []
    for key, tol in CONTROL_TOL.items():
        if key not in controls:
            continue
        value = controls[key]
        if not math.isfinite(value) or value > tol:
            failures.append(f"{key} = {value:.3e} > tol {tol:.1e}")
    if audit is not None:
        value = audit["max_rel_error_vs_raw_channel_recompute"]
        if not math.isfinite(value) or value > CONTROL_TOL[
                "energy_vs_raw_channel_recompute_rel"]:
            failures.append(f"energy != recomputed tr(H H^H): {value:.3e}")
        value = audit["max_abs_change_when_other_aps_corrupted"]
        if value > CONTROL_TOL["energy_unchanged_when_other_aps_corrupted"]:
            failures.append(
                f"E changed when other APs' channels were corrupted: {value:.3e}")
        if audit["min_rel_change_if_another_ap_mask_used"] <= 0.0:
            failures.append("the served-user mask does not restrict E")
    return failures


# ------------------------------------------------------------------------ CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default="../../artifacts/decentralized_ris/results_extend_500k/"
                "M2_N30_L4_K8_P15.0_iter350000_seed0/run0")
    parser.add_argument("--checkpoint", default="resumable_final.pt")
    parser.add_argument("--op", default="main", choices=sorted(OPERATING_POINTS))
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--eval_seed", type=int, required=True)
    parser.add_argument("--perm_seed", type=int, default=917)
    parser.add_argument("--control_batches", type=int, default=25)
    parser.add_argument("--determinism_batches", type=int, default=3)
    parser.add_argument("--tie_tolerance", type=float, default=1e-6)
    parser.add_argument("--audit", action="store_true", default=True)
    parser.add_argument("--no-audit", dest="audit", action="store_false")
    parser.add_argument("--progress", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out_dir",
        default="../../artifacts/decentralized_ris/energy_consensus_verification")
    args = parser.parse_args()

    started = time.time()
    device = resolve_device(args.device)
    with open(os.path.join(args.run, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    config = summary["config"]
    if config["arch"] != "r0":
        raise SystemExit(f"expected an r0 checkpoint, got {config['arch']}")

    point = OPERATING_POINTS[args.op]
    eval_config = dict(config)
    eval_config["pmax_dbm"] = point["pmax_dbm"]
    threshold = point["assoc_threshold"]

    # Topology and LoS draws follow the *training* seed, exactly as evaluate.py
    # and mrc_proxy_diagnostic.py do, so the simulator is the frozen one.
    seed_everything(config["seed"])
    simulator = ChannelSimulator(
        config["M"], config["N"], config["L"], config["batch_size"], n_ap=config["AP"])
    nets = []
    for arch in ("r1_ap_ris_mag", "r0", "r0c", "r1_shared"):
        net, ckpt_path = load_policy(
            args.run, args.checkpoint, simulator, device, eval_config, arch)
        nets.append(net)

    print(f"[setup] checkpoint {ckpt_path}")
    print(f"[setup] tag={summary['tag']} training_seed={config['seed']} "
          f"best_val_iteration={summary.get('best_val', {}).get('iteration')}")
    print(f"[setup] operating point {args.op}: pmax_dbm={point['pmax_dbm']} "
          f"assoc_threshold={threshold} K={config['K']} N={config['N']}")
    print(f"[setup] samples={args.samples} eval_seed={args.eval_seed} device={device}")

    result = run_split(nets, simulator, eval_config, device, args, threshold,
                       args.control_batches)

    # --- determinism: rebuild everything and replay the first batches --------
    if args.determinism_batches > 0:
        replay_args = argparse.Namespace(**vars(args))
        replay_args.samples = args.determinism_batches * config["batch_size"]
        replay_args.progress = 0
        replay_args.audit = False
        seed_everything(config["seed"])
        replay_sim = ChannelSimulator(
            config["M"], config["N"], config["L"], config["batch_size"],
            n_ap=config["AP"])
        replay_nets = [
            load_policy(args.run, args.checkpoint, replay_sim, device,
                        eval_config, arch)[0]
            for arch in ("r1_ap_ris_mag", "r0", "r0c", "r1_shared")
        ]
        replay = run_split(replay_nets, replay_sim, eval_config, device,
                           replay_args, threshold, 0)
        worst = 0.0
        for arm, values in replay["cluster"].items():
            worst = max(worst, float(
                np.abs(values - result["cluster"][arm][:len(values)]).max()))
        result["controls"]["determinism_replay"] = worst
        print(f"[control] determinism replay of {args.determinism_batches} "
              f"batches: max|diff|={worst:.3e}")

    failures = check_controls(result["controls"], result["audit"])
    for key in sorted(k for k in result["controls"] if k in CONTROL_TOL):
        print(f"[control] {key:42s} {result['controls'][key]:.3e} "
              f"(tol {CONTROL_TOL[key]:.1e})")

    arms = {arm: cluster_summary(values)
            for arm, values in result["cluster"].items()}
    arms_sample = {arm: cluster_summary(values)
                   for arm, values in result["samples"].items()}
    arms_2bit = {arm: cluster_summary(values)
                 for arm, values in result["cluster_2bit"].items()}
    contrasts = contrast_table(result, args.tie_tolerance)

    correlations = {
        "spearman_learned_s_vs_energy_across_ap_within_ris":
            cluster_summary(result["corr_ap"]),
        "spearman_learned_s_vs_energy_across_ap_ris_pairs":
            cluster_summary(result["corr_pair"]),
    }

    health = {
        arm: {k: (None if v == math.inf else v) for k, v in row.items()}
        for arm, row in result["health"].items()
    }

    payload = {
        "command": " ".join([sys.executable.split(os.sep)[-1],
                             "-m", "experiments.energy_consensus_verification",
                             *sys.argv[1:]]),
        "argv": sys.argv,
        "config": {
            "run": args.run, "checkpoint": args.checkpoint,
            "checkpoint_path": ckpt_path, "tag": summary["tag"],
            "operating_point": args.op, "pmax_dbm": point["pmax_dbm"],
            "assoc_threshold": threshold, "samples": args.samples,
            "clusters": result["n_batches"], "batch_size": config["batch_size"],
            "eval_seed": args.eval_seed, "perm_seed": args.perm_seed,
            "training_seed": config["seed"], "model_config": eval_config,
            "weight_sum_eps": WEIGHT_SUM_EPS, "resultant_eps": RESULTANT_EPS,
            "tie_tolerance": args.tie_tolerance,
            "torch": torch.__version__, "numpy": np.__version__,
            "runtime_seconds": round(time.time() - started, 1),
        },
        "arms_cluster": arms,
        "arms_sample": arms_sample,
        "arms_2bit_cluster": arms_2bit,
        "contrasts": contrasts,
        "correlations": correlations,
        "controls": result["controls"],
        "control_failures": failures,
        "numerical_health": health,
        "locality_audit": result["audit"],
        "descriptives": {
            "samples_all_five_aps_active": int(result["all_active"].sum()),
            "samples_total": int(result["all_active"].size),
            "mean_active_aps": float(result["active_count"].mean()),
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{args.op}_seed{args.eval_seed}")
    with open(f"{stem}.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    np.savez(
        f"{stem}_per_sample.npz",
        cluster_index=result["cluster_index"],
        all_active=result["all_active"],
        active_count=result["active_count"],
        spearman_across_ap=result["corr_ap"],
        spearman_across_pairs=result["corr_pair"],
        **{f"sample__{k}": v for k, v in result["samples"].items()},
        **{f"sample_2bit__{k}": v for k, v in result["samples_2bit"].items()},
        **{f"cluster__{k}": v for k, v in result["cluster"].items()},
        **{f"cluster_2bit__{k}": v for k, v in result["cluster_2bit"].items()},
    )
    with open(f"{stem}_arms.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "mean_sum_rate", "cluster_sem", "cluster_ci_low",
                         "cluster_ci_high", "sample_sem", "mean_sum_rate_2bit",
                         "clusters", "samples"])
        for arm in ALL_ARMS:
            value, sample_value = arms[arm], arms_sample[arm]
            writer.writerow([
                arm, f"{value['mean']:.6f}", f"{value['sem']:.6f}",
                f"{value['ci_low']:.6f}", f"{value['ci_high']:.6f}",
                f"{sample_value['sem']:.6f}", f"{arms_2bit[arm]['mean']:.6f}",
                value["n"], sample_value["n"]])
    with open(f"{stem}_contrasts.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["contrast", "cluster_mean_diff", "cluster_sem",
                         "cluster_ci_low", "cluster_ci_high", "sample_mean_diff",
                         "sample_ci_low", "sample_ci_high", "wins", "losses",
                         "ties", "n_samples"])
        for name, entry in contrasts.items():
            c, s = entry["cluster"], entry["sample"]
            w = entry["win_loss_tie_per_sample"]
            writer.writerow([
                name, f"{c['mean']:.6f}", f"{c['sem']:.6f}", f"{c['ci_low']:.6f}",
                f"{c['ci_high']:.6f}", f"{s['mean']:.6f}", f"{s['ci_low']:.6f}",
                f"{s['ci_high']:.6f}", w["wins"], w["losses"], w["ties"], w["n"]])

    print("\n=== arms (decentralized continuous sum rate, bps/Hz) ===")
    for arm in ALL_ARMS:
        value = arms[arm]
        marker = "*" if arm in CORE_ARMS else " "
        print(f" {marker}{arm:32s} {value['mean']:9.5f} +/- {value['sem']:.5f} "
              f"  95% CI [{value['ci_low']:.5f}, {value['ci_high']:.5f}]"
              f"   2-bit {arms_2bit[arm]['mean']:9.5f}")
    print("\n=== paired differences (cluster-level CI) and per-sample W/L/T ===")
    for name, entry in contrasts.items():
        c, w = entry["cluster"], entry["win_loss_tie_per_sample"]
        print(f"  {name:56s} {c['mean']:+.5f} +/- {c['sem']:.5f} "
              f"  CI [{c['ci_low']:+.5f}, {c['ci_high']:+.5f}]"
              f"   W/L/T {w['wins']}/{w['losses']}/{w['ties']}")
    print("\n=== Spearman(learned s, E) ===")
    for name, value in correlations.items():
        print(f"  {name:52s} {value['mean']:+.4f} +/- {value['sem']:.4f} "
              f"  CI [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]")
    print("\n=== numerical health (worst over the split) ===")
    for arm in ALL_ARMS:
        row = health[arm]
        print(f"  {arm:32s} nan(w/theta/rate)="
              f"{row['nan_in_weights']}/{row['nan_in_theta']}/{row['nan_in_rate']}"
              f"  min|resultant|={row['min_resultant_norm']:.3e}"
              f"  fallbacks={row['n_projection_fallback']}"
              f"  |theta|-1={row['max_unit_modulus_error']:.2e}")
    if result["audit"]:
        print("\n=== locality audit ===")
        print(f"  E vs raw-channel recompute (rel)      "
              f"{result['audit']['max_rel_error_vs_raw_channel_recompute']:.3e}")
        print(f"  E change when other APs corrupted     "
              f"{result['audit']['max_abs_change_when_other_aps_corrupted']:.3e}")
        print(f"  E change under another AP's mask (rel)"
              f" {result['audit']['min_rel_change_if_another_ap_mask_used']:.3e}")

    if failures:
        print("\n[STOP] control failures:")
        for line in failures:
            print(f"  - {line}")
        with open(f"{stem}_CONTROL_FAILURE.txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(failures) + "\n")
        raise ControlFailure("; ".join(failures))
    print(f"\n[done] {stem}.json  ({payload['config']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
