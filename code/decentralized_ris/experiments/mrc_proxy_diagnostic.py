"""MRC-style alternative explanation for the r1_ap_ris_mag pair scale.

Section 6.4 of `doc/ris_contribution_analysis.md` raises the objection that the
learned pair scale

    s_{l,r} = (1/N) sum_n || z_{l,r,n} ||,   z_l = W_reduce q_l + b/A

may be nothing more than a learned stand-in for the AP-RIS large-scale gain,
because the RIS readout's second branch is fed the raw channel energies
e(l,r,k) = tr(H_{l,r,k} H_{l,r,k}^H).  Section 8.2 asks for the defensive
experiment: correlate s_{l,r} with that energy, then substitute a closed-form
parameter-free function of it for the learned scale and re-evaluate.

This script is pure evaluation of a frozen checkpoint.  It never trains, and it
holds the beamformer and the per-AP unit-modulus proposals fixed across arms, so
the only thing that changes between arms is the consensus weight:

    equal            w_{l,r}   = 1                      (= r1_shared)
    learned_pair_s   w_{l,r}   = s_{l,r}                (= r1_ap_ris_mag)
    proxy_<name>     w_{l,r}   = g(e(l,r,.))            (parameter-free)
    per_element_mag  w_{l,r,n} = ||z_{l,r,n}||          (= r0c = r0)

Every control that could invalidate the comparison is checked and aborts the
run on failure: R0c/R0 algebraic equivalence, the per-element ladder reproducing
R0c, unit modulus of every emitted phase, the fast rate path agreeing with
`ChannelSimulator.loss`, the beamformer being identical across arms, and an
exact re-run of the first batches.
"""

import argparse
import csv
import json
import math
import os

import numpy as np
import torch

import variants
from evaluate import build_model, resolve_device, seed_everything, temporary_seed
from experiments.proposal_diagnostics import _spearman_along
from model import load_checkpoint
from rates import RatePrecompute, quantize_phase
from simulation import ChannelSimulator


# The proxy family is fixed here, in source, before any holdout evaluation.
# Every member is parameter-free and computable by an AP from its own CSI plus
# the RIS geometry; none of them has a coefficient fitted to any split.
PROXY_NAMES = ("sum_energy", "mean_energy", "sqrt_energy", "log1p_energy", "geom_ap_ris")

CONTROL_TOL = {
    "r0c_vs_r0_theta": 1e-4,
    "per_element_vs_r0c_theta": 1e-4,
    "unit_modulus": 1e-5,
    "beamformer_across_arches": 1e-6,
    "fast_rate_vs_simulator_loss": 1e-3,
    "second_branch_input": 0.0,
    "second_branch_ris_constant": 0.0,
    "energy_identity_rel": 1e-5,
    "learned_arm_vs_model_theta": 1e-6,
    "equal_arm_vs_r1_shared_theta": 1e-6,
}


class ControlFailure(RuntimeError):
    """Raised when any equivalence, unit-modulus or determinism control fails."""


# ----------------------------------------------------------------- statistics

def _normal_quantile(p):
    """Phi^{-1}(p) by bisection on math.erf; exact to double precision."""
    low, high = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def student_t_quantile(p, df):
    """Cornish-Fisher expansion of the Student-t quantile (accurate for df >= 30)."""
    z = _normal_quantile(p)
    z2, z3 = z * z, z ** 3
    z5, z7, z9 = z ** 5, z ** 7, z ** 9
    g1 = (z3 + z) / 4.0
    g2 = (5.0 * z5 + 16.0 * z3 + 3.0 * z) / 96.0
    g3 = (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / 384.0
    g4 = (79.0 * z9 + 776.0 * z7 + 1482.0 * z5 - 1920.0 * z3 - 945.0 * z) / 92160.0
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


def cluster_summary(values, conf=0.95):
    """Mean, SEM and two-sided CI over independent batch clusters."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {"mean": None, "sem": None, "n": 0}
    mean = float(values.mean())
    if n == 1:
        return {"mean": mean, "sem": 0.0, "n": 1, "ci_low": mean, "ci_high": mean}
    sem = float(values.std(ddof=1) / math.sqrt(n))
    half = student_t_quantile(0.5 + conf / 2.0, n - 1) * sem
    return {
        "mean": mean, "sem": sem, "n": n,
        "ci_low": mean - half, "ci_high": mean + half,
    }


def paired_difference(a, b, margin=None, conf=0.95):
    """Summary of the paired difference a - b over clusters, with a one-sided bound.

    `margin` is a non-inferiority margin on the *loss* a - b: the upper bound of
    the one-sided confidence interval is compared against it.
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    out = cluster_summary(diff, conf)
    if out["n"] > 1:
        t_one = student_t_quantile(conf, out["n"] - 1)
        out["one_sided_upper"] = out["mean"] + t_one * out["sem"]
        out["t_one_sided"] = t_one
        if margin is not None:
            out["margin"] = margin
            out["non_inferior"] = bool(out["one_sided_upper"] < margin)
    return out


# ------------------------------------------------------------------ consensus

def weighted_consensus(proposals, weights):
    """Circular mean of unit proposals under non-negative weights.

    proposals: (B, A, R, N, 2); weights: (B, A, R) or (B, A, R, N).
    The direction is invariant to the overall scale of the weights, so the
    per-element magnitude arm reproduces Pi(sum_l z_l), i.e. r0c, exactly.
    """
    if weights.dim() == 3:
        weights = weights.unsqueeze(-1).expand(proposals.shape[:-1])
    w = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
    resultant = (proposals * w[..., None]).sum(dim=1)
    norm = resultant.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(resultant)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1e-12, resultant / norm.clamp(min=1e-12), fallback)


# ------------------------------------------------------- the proxy definitions

def pair_energy(edges, masks, device):
    """E_{l,r} = sum_{k served by l} e(l,r,k), and the served-user count.

    `edges[l]` is (B, R, K_ap) and is already zero outside AP l's association
    mask, so the sum runs over exactly the users AP l serves.  This is the same
    tensor the RIS readout's second branch receives.
    """
    stacked = torch.stack([tensor.to(device) for tensor in edges], dim=1)   # (B, A, R, K)
    energy = stacked.sum(dim=3)                                            # (B, A, R)
    served = torch.as_tensor(
        np.stack([np.asarray(mask) for mask in masks], axis=1).astype(np.float32),
        device=device,
    ).sum(dim=2)                                                           # (B, A)
    return energy, served, stacked


def proxy_weights(name, energy, served, ap_active, ap_ris_distance):
    """One parameter-free weight per (AP, RIS), masked to the active APs.

    An AP that serves nobody transmits nothing, so it cannot vote; every proxy
    is therefore multiplied by the AP-active indicator.
    """
    if name == "sum_energy":
        w = energy
    elif name == "mean_energy":
        w = energy / served.clamp(min=1.0).unsqueeze(2)
    elif name == "sqrt_energy":
        w = energy.sqrt()
    elif name == "log1p_energy":
        active = ap_active.unsqueeze(2)
        scale = (energy * active).sum(dim=1, keepdim=True) / active.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)
        w = torch.log1p(energy / scale.clamp(min=1e-30))
    elif name == "geom_ap_ris":
        w = ap_ris_distance.pow(-2.0).unsqueeze(0).expand_as(energy)
    else:
        raise ValueError(f"unknown proxy {name}")
    return w.clamp(min=0.0) * ap_active.unsqueeze(2)


# --------------------------------------------------------------- model set-up

def load_policy(run_dir, checkpoint_name, simulator, device, arch_override=None):
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    config = summary["config"].copy()
    if arch_override:
        config["arch"] = arch_override
        config["consensus"] = variants.default_consensus(arch_override)
    path = None
    for folder in ("checkpoints", "models"):
        candidate = os.path.join(run_dir, folder, checkpoint_name)
        if os.path.exists(candidate):
            path = candidate
            break
    if path is None:
        raise SystemExit(f"no {checkpoint_name} under {run_dir}")
    net = build_model(config, simulator, device)
    load_checkpoint(net, path, device)
    net.eval()
    return net, summary, path


# ------------------------------------------- task 1: document the second branch

def head_weight(net, ap_index):
    """Column norms of `fe_AP`, grouped by the RIS and user axes of its input.

    The input is the (R, K_ap) energy block flattened row-major, so reshaping
    the weight to (2N, R, K_ap) shows how much each RIS row and each served user
    contributes to the branch.  A flat profile means the branch aggregates
    uniformly; a peaked one means it is selective.
    """
    weight = net.RIS_readout_AP_list[ap_index].fe_AP.weight.detach()
    n_ris = net.L
    weight = weight.reshape(weight.shape[0], n_ris, -1)
    by_ris = weight.norm(dim=(0, 2))
    by_user = weight.norm(dim=(0, 1))
    return {
        "by_ris": (by_ris / by_ris.mean().clamp(min=1e-30)).cpu().numpy().round(4).tolist(),
        "by_user": (by_user / by_user.mean().clamp(min=1e-30)).cpu().numpy().round(4).tolist(),
    }


def document_second_branch(net, simulator, batch, device):
    """Prove which tensor the RIS readout's second branch consumes.

    Captures `fe_AP`'s input with a hook and checks it against (a) the masked
    per-AP edge tensor and (b) the Frobenius energy recomputed from the raw
    complex channels, then checks that the branch's contribution is constant
    across the RIS axis.
    """
    dfeatures, dedges, dmasks, ddirect = batch
    captured_input, captured_latent, handles = {}, {}, []
    for index, head in enumerate(net.RIS_readout_AP_list):
        handles.append(head.fe_AP.register_forward_pre_hook(
            lambda module, inputs, idx=index: captured_input.__setitem__(
                idx, inputs[0].detach().clone())))
        handles.append(head.register_forward_hook(
            lambda module, inputs, output, idx=index: captured_latent.__setitem__(
                idx, output.detach().clone())))
    with torch.no_grad():
        net.decentralized(dfeatures, dedges, dmasks, ddirect)
    for handle in handles:
        handle.remove()

    n_elem = net.N
    n_ris = net.L
    report = {"per_ap": []}
    worst_input, worst_const, worst_energy = 0.0, 0.0, 0.0
    for index in range(net.AP):
        expected = dedges[index].to(device)                                # (B, R, K)
        flat = expected.reshape(expected.shape[0], -1)
        got = captured_input[index]
        input_err = float((got - flat).abs().max())

        latent = captured_latent[index]                                    # (B, R, 4N)
        branch = latent[:, :, 2 * n_elem:]
        const_err = float((branch - branch[:, 0:1, :]).abs().max())

        # e(l,r,k) recomputed from the complex cascaded channel of AP l.
        channels = simulator.base_stations[index].channels                 # (B, M, N, R, K)
        energy = np.abs(channels) ** 2
        energy = energy.sum(axis=(1, 2)).astype(np.float64)                # (B, R, K)
        mask = np.asarray(dmasks[index]).astype(bool)                      # (B, K)
        energy = energy * mask[:, None, :]
        reference = expected.detach().cpu().numpy().astype(np.float64)
        scale = max(float(np.abs(energy).max()), 1e-30)
        energy_err = float(np.abs(energy - reference).max() / scale)

        weight = head_weight(net, index)
        worst_input = max(worst_input, input_err)
        worst_const = max(worst_const, const_err)
        worst_energy = max(worst_energy, energy_err)
        report["per_ap"].append({
            "ap": index,
            "fe_AP_input_shape": list(got.shape),
            "max_abs_input_minus_masked_edges": input_err,
            "max_abs_branch_variation_across_ris": const_err,
            "max_rel_energy_identity_error": energy_err,
            "fe_AP_weight_norm_by_ris": weight["by_ris"],
            "fe_AP_weight_norm_by_user": weight["by_user"],
        })

    # Does the branch's RIS profile follow AP-RIS geometry?  If it does, the
    # branch is reading large-scale gain, which is exactly the MRC objection.
    distance = np.linalg.norm(
        np.asarray(simulator.ap_locations)[:, None, :]
        - np.asarray(simulator.ris_locations)[None, :, :], axis=2,
    )
    profiles = np.asarray([row["fe_AP_weight_norm_by_ris"] for row in report["per_ap"]])
    order_match = [
        sorted(np.argsort(-profiles[i])[:2].tolist())
        == sorted(np.argsort(distance[i])[:2].tolist())
        for i in range(profiles.shape[0])
    ]
    report["ris_profile_vs_geometry"] = {
        "ap_ris_distance": distance.round(2).tolist(),
        "spearman_weight_vs_negative_distance": [
            float(_spearman_along(
                profiles[i:i + 1], -distance[i:i + 1], axis=1)[0])
            for i in range(profiles.shape[0])
        ],
        "two_largest_weights_are_two_nearest_ris": order_match,
        "user_profile_max_over_min": [
            float(max(row["fe_AP_weight_norm_by_user"])
                  / max(min(row["fe_AP_weight_norm_by_user"]), 1e-30))
            for row in report["per_ap"]
        ],
    }
    report.update({
        "fe_AP_in_features": int(net.RIS_readout_AP_list[0].fe_AP.in_features),
        "fe_AP_out_features": int(net.RIS_readout_AP_list[0].fe_AP.out_features),
        "n_ris": int(n_ris),
        "n_elements": int(n_elem),
        "users_per_ap": int(dedges[0].shape[2]),
        "max_abs_input_minus_masked_edges": worst_input,
        "max_abs_branch_variation_across_ris": worst_const,
        "max_rel_energy_identity_error": worst_energy,
    })
    failures = []
    if worst_input > CONTROL_TOL["second_branch_input"]:
        failures.append(f"fe_AP input != masked edges ({worst_input:.3e})")
    if worst_const > CONTROL_TOL["second_branch_ris_constant"]:
        failures.append(f"second branch varies across RIS ({worst_const:.3e})")
    if worst_energy > CONTROL_TOL["energy_identity_rel"]:
        failures.append(f"edge weight != tr(H H^H) ({worst_energy:.3e})")
    report["failures"] = failures
    return report


# ------------------------------------------------------------------ main pass

def arm_names(selected_proxy):
    names = ["equal", "learned_pair_s", "per_element_mag"]
    names += [f"proxy_{name}" for name in PROXY_NAMES]
    if selected_proxy:
        names.append(f"selected_proxy[{selected_proxy}]")
    return names


def run_split(net_mag, net_r0, net_r0c, net_r1s, simulator, config, device, args,
              control_batches):
    """Evaluate every arm on one split and return per-cluster records."""
    users = config["K"]
    threshold = config.get("assoc_threshold", 0.1)
    batch_size = config["batch_size"]
    if args.samples <= 0 or args.samples % batch_size:
        raise SystemExit("samples must be a positive multiple of the batch size")
    n_batches = args.samples // batch_size

    ap_ris_distance = torch.as_tensor(
        np.linalg.norm(
            np.asarray(simulator.ap_locations)[:, None, :]
            - np.asarray(simulator.ris_locations)[None, :, :],
            axis=2,
        ),
        dtype=torch.float32, device=device,
    )                                                                      # (A, R)

    arms = ["equal", "learned_pair_s", "per_element_mag"] + [
        f"proxy_{name}" for name in PROXY_NAMES
    ]
    cluster = {name: [] for name in arms}
    cluster_2bit = {name: [] for name in arms}
    samples = {name: [] for name in arms}
    corr_across_ap = {name: [] for name in PROXY_NAMES}
    corr_across_pair = {name: [] for name in PROXY_NAMES}
    extras = {
        "active_aps": [], "all_active": [], "energy_mean": [], "energy_max": [],
        "energy_min_positive": [], "learned_s_mean": [], "learned_s_cv_across_ap": [],
    }
    unit_error = {name: 0.0 for name in arms}
    controls = {
        "r0c_vs_r0_theta": 0.0, "per_element_vs_r0c_theta": 0.0,
        "beamformer_across_arches": 0.0, "fast_rate_vs_simulator_loss": 0.0,
        "learned_arm_vs_model_theta": 0.0, "equal_arm_vs_r1_shared_theta": 0.0,
        "control_batches": int(control_batches),
    }

    with torch.no_grad(), temporary_seed(args.eval_seed):
        for index in range(n_batches):
            simulator.training_batch(users, threshold, threshold)
            dfeatures, dedges, dmasks, ddirect = simulator.decentralized_batch(
                users, threshold, threshold, regenerate_channels=False
            )
            dfeatures = [tensor.to(device) for tensor in dfeatures]
            dedges_dev = [tensor.to(device) for tensor in dedges]
            ddirect = [tensor.to(device) for tensor in ddirect]
            pre = RatePrecompute(simulator, device)

            trace = {}
            beamformer, theta_model = net_mag.decentralized(
                dfeatures, dedges_dev, dmasks, ddirect, trace
            )
            proposals = trace["proposals"]                                 # (B, A, R, N, 2)
            z_pairs = trace["z_pairs"]                                     # (B, A, R, N, 2)
            magnitude = z_pairs.norm(dim=-1)                               # (B, A, R, N)
            learned_s = magnitude.mean(dim=3)                              # (B, A, R)
            source_active = trace["source_active"]                         # (B, A)

            energy, served, _ = pair_energy(dedges, dmasks, device)
            weights = {
                "equal": torch.ones_like(learned_s),
                "learned_pair_s": learned_s,
                "per_element_mag": magnitude,
            }
            for name in PROXY_NAMES:
                weights[f"proxy_{name}"] = proxy_weights(
                    name, energy, served, source_active, ap_ris_distance
                )

            thetas = {}
            for name, weight in weights.items():
                theta = weighted_consensus(proposals, weight)
                thetas[name] = theta
                unit_error[name] = max(
                    unit_error[name], float(variants.unit_modulus_error(theta))
                )
                rate = pre.sum_rate(beamformer, theta)
                samples[name].append(rate.detach().cpu().numpy())
                cluster[name].append(float(rate.mean()))
                rate_2bit = pre.sum_rate(beamformer, quantize_phase(theta, 2))
                cluster_2bit[name].append(float(rate_2bit.mean()))

            # The reimplemented consensus must reproduce the deployed one, which
            # also pins the (batch, AP, RIS) alignment of the proxy weights.
            controls["learned_arm_vs_model_theta"] = max(
                controls["learned_arm_vs_model_theta"],
                float((thetas["learned_pair_s"] - theta_model).abs().max()),
            )

            if index < control_batches:
                beamformer_r0, theta_r0 = net_r0.decentralized(
                    dfeatures, dedges_dev, dmasks, ddirect
                )
                _, theta_r0c = net_r0c.decentralized(
                    dfeatures, dedges_dev, dmasks, ddirect
                )
                controls["r0c_vs_r0_theta"] = max(
                    controls["r0c_vs_r0_theta"],
                    float((theta_r0c - theta_r0).abs().max()),
                )
                controls["per_element_vs_r0c_theta"] = max(
                    controls["per_element_vs_r0c_theta"],
                    float((thetas["per_element_mag"] - theta_r0c).abs().max()),
                )
                controls["beamformer_across_arches"] = max(
                    controls["beamformer_across_arches"],
                    float((beamformer - beamformer_r0).abs().max()),
                )
                _, theta_r1s = net_r1s.decentralized(
                    dfeatures, dedges_dev, dmasks, ddirect
                )
                controls["equal_arm_vs_r1_shared_theta"] = max(
                    controls["equal_arm_vs_r1_shared_theta"],
                    float((thetas["equal"] - theta_r1s).abs().max()),
                )
                _, reference_rate, _ = simulator.loss(
                    beamformer, thetas["learned_pair_s"], device
                )
                fast = float(pre.sum_rate(beamformer, thetas["learned_pair_s"]).mean())
                controls["fast_rate_vs_simulator_loss"] = max(
                    controls["fast_rate_vs_simulator_loss"],
                    abs(float(reference_rate) - fast),
                )

            # Correlations, computed on the deployed (local CSI) weights.
            s_np = learned_s.detach().cpu().numpy().astype(np.float64)     # (B, A, R)
            active_np = source_active.detach().cpu().numpy().astype(bool)
            for name in PROXY_NAMES:
                p_np = weights[f"proxy_{name}"].detach().cpu().numpy().astype(np.float64)
                # Across APs, within one RIS: the axis the consensus normalises.
                across = _spearman_along(
                    s_np.transpose(0, 2, 1), p_np.transpose(0, 2, 1), axis=2
                )                                                          # (B, R)
                corr_across_ap[name].append(float(np.nanmean(across)))
                # Across all AP-RIS pairs of one sample.
                flat_s = s_np.reshape(s_np.shape[0], -1)
                flat_p = p_np.reshape(p_np.shape[0], -1)
                corr_across_pair[name].append(
                    float(np.nanmean(_spearman_along(flat_s, flat_p, axis=1)))
                )

            energy_np = energy.detach().cpu().numpy().astype(np.float64)
            positive = energy_np[energy_np > 0]
            extras["active_aps"].append(float(active_np.sum(axis=1).mean()))
            extras["all_active"].append(float(active_np.all(axis=1).mean()))
            extras["energy_mean"].append(float(positive.mean()) if positive.size else np.nan)
            extras["energy_max"].append(float(energy_np.max()))
            extras["energy_min_positive"].append(
                float(positive.min()) if positive.size else np.nan
            )
            extras["learned_s_mean"].append(float(s_np.mean()))
            extras["learned_s_cv_across_ap"].append(
                float(np.mean(s_np.std(axis=1) / np.maximum(s_np.mean(axis=1), 1e-30)))
            )

            if args.progress and (index + 1) % args.progress == 0:
                print(f"  [{index + 1:4d}/{n_batches}] "
                      f"learned={np.mean(cluster['learned_pair_s']):.5f} "
                      f"equal={np.mean(cluster['equal']):.5f}", flush=True)

    controls["unit_modulus"] = max(unit_error.values())
    controls["unit_modulus_per_arm"] = unit_error
    return {
        "cluster": {name: np.asarray(values) for name, values in cluster.items()},
        "cluster_2bit": {name: np.asarray(v) for name, v in cluster_2bit.items()},
        "samples": {name: np.concatenate(values) for name, values in samples.items()},
        "corr_across_ap": {k: np.asarray(v) for k, v in corr_across_ap.items()},
        "corr_across_pair": {k: np.asarray(v) for k, v in corr_across_pair.items()},
        "extras": {k: np.asarray(v) for k, v in extras.items()},
        "controls": controls,
        "n_batches": n_batches,
    }


def check_controls(controls, second_branch):
    failures = list(second_branch.get("failures", []))
    for key in ("r0c_vs_r0_theta", "per_element_vs_r0c_theta", "unit_modulus",
                "beamformer_across_arches", "fast_rate_vs_simulator_loss",
                "learned_arm_vs_model_theta", "equal_arm_vs_r1_shared_theta"):
        value = controls.get(key)
        if value is None:
            continue
        if not math.isfinite(value) or value > CONTROL_TOL[key]:
            failures.append(f"{key} = {value:.3e} > tol {CONTROL_TOL[key]:.1e}")
    return failures


# ----------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default="../../artifacts/decentralized_ris/results_extend_500k/"
                "M2_N30_L4_K8_P15.0_iter350000_seed0/run0",
    )
    parser.add_argument("--checkpoint", default="resumable_final.pt")
    parser.add_argument("--split", required=True, choices=("calibration", "locked"))
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--eval_seed", type=int, required=True)
    parser.add_argument("--control_batches", type=int, default=25)
    parser.add_argument("--determinism_batches", type=int, default=3)
    parser.add_argument("--selected_proxy", default=None,
                        help="locked split only: the calibration-selected proxy")
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--progress", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out_dir",
        default="../../artifacts/decentralized_ris/mrc_proxy_diagnostic",
    )
    args = parser.parse_args()

    if args.split == "locked" and not args.selected_proxy:
        raise SystemExit("the locked split requires --selected_proxy from calibration")
    if args.selected_proxy and args.selected_proxy not in PROXY_NAMES:
        raise SystemExit(f"--selected_proxy must be one of {PROXY_NAMES}")

    device = resolve_device(args.device)
    with open(os.path.join(args.run, "summary.json"), encoding="utf-8") as handle:
        config = json.load(handle)["config"]

    # The topology and the LoS draws depend on the training seed, as in evaluate.py.
    seed_everything(config["seed"])
    simulator = ChannelSimulator(
        config["M"], config["N"], config["L"], config["batch_size"], n_ap=config["AP"]
    )
    net_mag, summary, ckpt_path = load_policy(
        args.run, args.checkpoint, simulator, device, arch_override="r1_ap_ris_mag"
    )
    net_r0, _, _ = load_policy(args.run, args.checkpoint, simulator, device)
    net_r0c, _, _ = load_policy(
        args.run, args.checkpoint, simulator, device, arch_override="r0c"
    )
    net_r1s, _, _ = load_policy(
        args.run, args.checkpoint, simulator, device, arch_override="r1_shared"
    )
    if summary["config"]["arch"] != "r0":
        raise SystemExit(f"expected an r0 checkpoint, got {summary['config']['arch']}")

    print(f"[setup] checkpoint {ckpt_path}")
    print(f"[setup] tag={summary['tag']} best_val_iteration="
          f"{summary.get('best_val', {}).get('iteration')}")
    print(f"[setup] split={args.split} samples={args.samples} seed={args.eval_seed}")

    # --- task 1: what the second branch actually reads -----------------------
    with temporary_seed(args.eval_seed), torch.no_grad():
        simulator.training_batch(config["K"], config.get("assoc_threshold", 0.1),
                                 config.get("assoc_threshold", 0.1))
        probe = simulator.decentralized_batch(
            config["K"], config.get("assoc_threshold", 0.1),
            config.get("assoc_threshold", 0.1), regenerate_channels=False
        )
        probe = (
            [t.to(device) for t in probe[0]], [t.to(device) for t in probe[1]],
            probe[2], [t.to(device) for t in probe[3]],
        )
        second_branch = document_second_branch(net_mag, simulator, probe, device)
    print("[task1] fe_AP input == masked per-AP edge tensor: "
          f"max|diff|={second_branch['max_abs_input_minus_masked_edges']:.3e}")
    print("[task1] edge weight == tr(H H^H) recomputed from channels: "
          f"max rel err={second_branch['max_rel_energy_identity_error']:.3e}")
    print("[task1] second branch is constant across the RIS axis: "
          f"max variation={second_branch['max_abs_branch_variation_across_ris']:.3e}")

    # --- main pass -----------------------------------------------------------
    result = run_split(net_mag, net_r0, net_r0c, net_r1s, simulator, config, device,
                       args, args.control_batches)

    # --- determinism: re-run the first batches from the same seeded state ----
    if args.determinism_batches > 0:
        replay_args = argparse.Namespace(**vars(args))
        replay_args.samples = args.determinism_batches * config["batch_size"]
        replay_args.progress = 0
        seed_everything(config["seed"])
        replay_sim = ChannelSimulator(
            config["M"], config["N"], config["L"], config["batch_size"], n_ap=config["AP"]
        )
        replay_mag, _, _ = load_policy(args.run, args.checkpoint, replay_sim, device,
                                       arch_override="r1_ap_ris_mag")
        replay_r0, _, _ = load_policy(args.run, args.checkpoint, replay_sim, device)
        replay_r0c, _, _ = load_policy(args.run, args.checkpoint, replay_sim, device,
                                        arch_override="r0c")
        replay_r1s, _, _ = load_policy(args.run, args.checkpoint, replay_sim, device,
                                        arch_override="r1_shared")
        replay = run_split(replay_mag, replay_r0, replay_r0c, replay_r1s, replay_sim,
                           config, device, replay_args, 0)
        worst = 0.0
        for name, values in replay["cluster"].items():
            reference = result["cluster"][name][:len(values)]
            worst = max(worst, float(np.abs(values - reference).max()))
        result["controls"]["determinism_max_abs_rate_diff"] = worst
        print(f"[control] determinism replay of {args.determinism_batches} batches: "
              f"max|diff|={worst:.3e}")
        if worst != 0.0:
            result["controls"].setdefault("extra_failures", []).append(
                f"determinism replay differs by {worst:.3e}"
            )

    failures = check_controls(result["controls"], second_branch)
    failures += result["controls"].get("extra_failures", [])
    for key in ("r0c_vs_r0_theta", "per_element_vs_r0c_theta", "unit_modulus",
                "beamformer_across_arches", "fast_rate_vs_simulator_loss",
                "learned_arm_vs_model_theta", "equal_arm_vs_r1_shared_theta"):
        print(f"[control] {key:32s} {result['controls'][key]:.3e} "
              f"(tol {CONTROL_TOL[key]:.1e})")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{args.split}_seed{args.eval_seed}")

    # --- summaries -----------------------------------------------------------
    cluster = result["cluster"]
    arms = {name: cluster_summary(values) for name, values in cluster.items()}
    arms_2bit = {name: cluster_summary(values)
                 for name, values in result["cluster_2bit"].items()}
    baseline = "learned_pair_s"
    differences = {}
    for name in cluster:
        if name == baseline:
            continue
        # Direction: a positive difference means the learned scale is ahead, so
        # the upper confidence bound is the loss incurred by dropping it.
        differences[f"{baseline}_minus_{name}"] = paired_difference(
            cluster[baseline], cluster[name], margin=args.margin
        )
    differences["learned_pair_s_minus_equal"] = paired_difference(
        cluster[baseline], cluster["equal"]
    )
    differences["per_element_mag_minus_learned_pair_s"] = paired_difference(
        cluster["per_element_mag"], cluster[baseline]
    )

    correlations = {
        name: {
            "across_ap_within_ris": cluster_summary(result["corr_across_ap"][name]),
            "across_ap_ris_pairs": cluster_summary(result["corr_across_pair"][name]),
        }
        for name in PROXY_NAMES
    }

    payload = {
        "config": {
            "run": args.run, "checkpoint": args.checkpoint, "checkpoint_path": ckpt_path,
            "tag": summary["tag"], "split": args.split, "samples": args.samples,
            "eval_seed": args.eval_seed, "batch_size": config["batch_size"],
            "clusters": result["n_batches"], "training_seed": config["seed"],
            "margin_bps_hz": args.margin, "proxy_family": list(PROXY_NAMES),
            "selected_proxy": args.selected_proxy,
            "model_config": config,
        },
        "second_branch": second_branch,
        "controls": result["controls"],
        "control_failures": failures,
        "arms": arms,
        "arms_2bit": arms_2bit,
        "paired_differences": differences,
        "correlations": correlations,
        "descriptives": {k: cluster_summary(v) for k, v in result["extras"].items()},
    }
    with open(f"{stem}.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    np.savez(
        f"{stem}_paired.npz",
        **{f"cluster__{k}": v for k, v in cluster.items()},
        **{f"cluster_2bit__{k}": v for k, v in result["cluster_2bit"].items()},
        **{f"sample__{k}": v for k, v in result["samples"].items()},
        **{f"corr_across_ap__{k}": v for k, v in result["corr_across_ap"].items()},
        **{f"corr_across_pair__{k}": v for k, v in result["corr_across_pair"].items()},
        **{f"extra__{k}": v for k, v in result["extras"].items()},
    )

    with open(f"{stem}_arms.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "mean_sum_rate", "sem", "ci_low", "ci_high",
                         "mean_sum_rate_2bit", "sem_2bit", "clusters"])
        for name, value in arms.items():
            discrete = arms_2bit[name]
            writer.writerow([name, f"{value['mean']:.6f}", f"{value['sem']:.6f}",
                             f"{value['ci_low']:.6f}", f"{value['ci_high']:.6f}",
                             f"{discrete['mean']:.6f}", f"{discrete['sem']:.6f}",
                             value["n"]])
    with open(f"{stem}_paired.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["contrast", "mean_diff", "sem", "ci_low", "ci_high",
                         "one_sided_upper", "margin", "non_inferior"])
        for name, value in differences.items():
            writer.writerow([
                name, f"{value['mean']:.6f}", f"{value['sem']:.6f}",
                f"{value['ci_low']:.6f}", f"{value['ci_high']:.6f}",
                f"{value.get('one_sided_upper', float('nan')):.6f}",
                value.get("margin", ""), value.get("non_inferior", ""),
            ])
    with open(f"{stem}_correlations.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["proxy", "grouping", "spearman", "sem", "ci_low", "ci_high"])
        for name, table in correlations.items():
            for grouping, value in table.items():
                writer.writerow([name, grouping, f"{value['mean']:.6f}",
                                 f"{value['sem']:.6f}", f"{value['ci_low']:.6f}",
                                 f"{value['ci_high']:.6f}"])

    print("\n=== arms (decentralized continuous sum rate, bps/Hz) ===")
    for name, value in arms.items():
        print(f"  {name:26s} {value['mean']:9.5f} +/- {value['sem']:.5f} "
              f"  95% CI [{value['ci_low']:.5f}, {value['ci_high']:.5f}]"
              f"   2-bit {arms_2bit[name]['mean']:9.5f}")
    print("\n=== Spearman(learned s, proxy) ===")
    for name, table in correlations.items():
        row = table["across_ap_within_ris"]
        pair = table["across_ap_ris_pairs"]
        print(f"  {name:14s} across-AP {row['mean']:+.4f} +/- {row['sem']:.4f}"
              f"   across-pair {pair['mean']:+.4f} +/- {pair['sem']:.4f}")
    print("\n=== paired differences vs learned_pair_s ===")
    for name, value in differences.items():
        upper = value.get("one_sided_upper")
        tail = "" if upper is None else f"  one-sided upper {upper:+.5f}"
        print(f"  {name:44s} {value['mean']:+.5f} +/- {value['sem']:.5f}{tail}")
    print("\n=== descriptives ===")
    for name, value in payload["descriptives"].items():
        print(f"  {name:26s} {value['mean']:.6g} +/- {value['sem']:.3g}")

    if failures:
        print("\n[STOP] control failures:")
        for line in failures:
            print(f"  - {line}")
        with open(f"{stem}_CONTROL_FAILURE.txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(failures) + "\n")
        raise ControlFailure("; ".join(failures))
    print(f"\n[done] {stem}.json")


if __name__ == "__main__":
    main()
