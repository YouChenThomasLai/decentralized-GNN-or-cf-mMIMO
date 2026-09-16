"""Counterfactual diagnostics for the R1-shared AP action interface.

These answer three questions without assuming that a learned scalar confidence
head is the repair:

1. Does local CSI actually disperse the per-AP proposals?  Compare the circular
   concentration rho_{r,n} = |mean_l exp(j phi_{l,r,n})| between the centralized
   and decentralized input modes.  A clear drop identifies local proposal
   disagreement as the mechanism.
2. Is the pre-projection magnitude ||z_{l,r,n}|| a reliability signal at all?
   Correlate it with angular agreement and with leave-one-AP-out marginal
   contribution.  Only a positive correlation licenses calling the magnitude an
   implicit confidence rather than an unconstrained scale.
3. How much of the R1-shared deficit is the phase representation itself and how
   much is its coupling to the beamformer?  Evaluate the 2x2 grid of R0/R1
   beamformers against R0/R1 phases on identical channels.

The magnitude questions are asked of both checkpoints on purpose.  Under r0 the
magnitude is functional -- it weights each AP's contribution to the sum that is
projected -- so that is where the implicit-confidence hypothesis must be tested.
Under r1_shared the magnitude is projected away and never reaches the loss, so
it serves as the control: there is no gradient pressure for it to mean anything.

Correlations are reported both per element and after aggregating to one number
per AP-RIS pair.  That contrast is the granularity test: a per-AP-RIS scalar
confidence can only recover what survives the aggregation.
"""

import argparse
import json
import os

import numpy as np
import torch

import variants
from evaluate import build_model, resolve_device, seed_everything, temporary_seed
from model import load_checkpoint
from simulation import ChannelSimulator
from variants import circular_consensus


# ------------------------------------------------------------------ statistics

def _rank_along(x, axis):
    """Ordinal ranks along `axis`; exact ties are measure-zero for these floats."""
    order = np.argsort(x, axis=axis)
    return np.argsort(order, axis=axis).astype(np.float64)


def _pearson_along(a, b, axis):
    a = a - a.mean(axis=axis, keepdims=True)
    b = b - b.mean(axis=axis, keepdims=True)
    num = (a * b).sum(axis=axis)
    den = np.sqrt((a * a).sum(axis=axis) * (b * b).sum(axis=axis))
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _spearman_along(a, b, axis):
    return _pearson_along(_rank_along(a, axis), _rank_along(b, axis), axis)


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "sem": None, "n": 0}
    return {
        "mean": float(values.mean()),
        "sem": float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0,
        "n": int(values.size),
    }


# ----------------------------------------------------------------- model setup

def load_variant(run_dir, checkpoint_name, simulator, device, arch_override=None):
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    config = summary["config"].copy()
    if arch_override:
        config["arch"] = arch_override
        config["consensus"] = variants.default_consensus(arch_override)
    path = os.path.join(run_dir, "checkpoints", checkpoint_name)
    if not os.path.exists(path):
        path = os.path.join(run_dir, "models", checkpoint_name)
    model = build_model(config, simulator, device)
    load_checkpoint(model, path, device)
    model.eval()
    return model, config, summary


# -------------------------------------------------------------- diagnostic (1)

def concentration(proposals, active=None):
    """rho_{r,n} = || mean_l p_{l,r,n} || for unit-modulus proposals p."""
    if active is None:
        resultant = proposals.mean(dim=1)
    else:
        w = active / active.sum(dim=1, keepdim=True).clamp(min=1e-12)
        resultant = (proposals * w[:, :, None, None, None]).sum(dim=1)
    return resultant.norm(dim=-1)                                   # (B, R, N)


# -------------------------------------------------------------- diagnostic (2)

def agreement_leave_one_out(proposals):
    """cos(phi_{l,r,n} - mean phase of the other APs), per (B, A, R, N)."""
    total = proposals.sum(dim=1, keepdim=True)                      # (B, 1, R, N, 2)
    others = total - proposals                                      # (B, A, R, N, 2)
    norm = others.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return (proposals * (others / norm)).sum(dim=-1)                # (B, A, R, N)


def leave_one_ap_out_rates(proposals, beamformer, simulator, device, full_rate):
    """Marginal sum-rate contribution of each (AP, RIS) pair under equal consensus.

    Dropping one AP's proposal for one RIS keeps every other vote intact, so the
    result is the contribution of that AP to that RIS rather than to the whole
    network.  This matches the granularity of the per-AP-RIS scalar confidence
    the plan would otherwise train.
    """
    b, n_ap, n_ris = proposals.shape[0], proposals.shape[1], proposals.shape[2]
    deltas = np.zeros((n_ap, n_ris), dtype=np.float64)
    ones = torch.ones((b, n_ap), device=proposals.device, dtype=proposals.dtype)
    full_theta, _ = circular_consensus(proposals, ones)
    for ap in range(n_ap):
        keep = ones.clone()
        keep[:, ap] = 0.0
        for ris in range(n_ris):
            dropped, _ = circular_consensus(proposals[:, :, ris:ris + 1], keep)
            theta = full_theta.clone()
            theta[:, ris:ris + 1] = dropped
            _, rate, _ = simulator.loss(beamformer, theta, device)
            deltas[ap, ris] = full_rate - float(rate)
    return deltas


# ------------------------------------------------- diagnostic (2c): granularity

def _weighted_consensus(proposals, weights):
    """Circular mean of unit proposals under arbitrary non-negative weights.

    `weights` broadcasts to (B, A, R, N).  The consensus direction is invariant
    to the overall scale of the weights, so weighting by ||z|| reproduces
    Pi(sum_l z_l) exactly -- that is, r0c.  Equal weights give r1_shared.  The
    ladder between them measures how much of r0c's advantage survives when the
    weight is coarsened to one scalar per AP-RIS pair, which is the granularity
    a scalar confidence head could represent.
    """
    w = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
    resultant = (proposals * w[..., None]).sum(dim=1)
    norm = resultant.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(resultant)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1e-12, resultant / norm.clamp(min=1e-12), fallback)


def weighting_ladder(proposals, z_pairs, beamformer, simulator, device):
    magnitude = z_pairs.norm(dim=-1)                                # (B, A, R, N)
    schemes = {
        "equal": torch.ones_like(magnitude),
        "pair_scalar": magnitude.mean(dim=3, keepdim=True).expand_as(magnitude),
        "pair_scalar_sqrt": magnitude.mean(dim=3, keepdim=True).sqrt().expand_as(magnitude),
        "per_element_sqrt": magnitude.sqrt(),
        "per_element": magnitude,
    }
    out = {}
    for name, weights in schemes.items():
        theta = _weighted_consensus(proposals, weights)
        _, rate, _ = simulator.loss(beamformer, theta, device)
        out[name] = float(rate)
    return out


# -------------------------------------------------------------- diagnostic (3)

def cross_rates(pairs, simulator, device):
    """Sum rate for every (beamformer source, phase source) combination."""
    out = {}
    for w_name, (beamformer, _) in pairs.items():
        for t_name, (_, phase) in pairs.items():
            _, rate, _ = simulator.loss(beamformer, phase, device)
            out[f"W_{w_name}__theta_{t_name}"] = float(rate)
    return out


# ----------------------------------------------------------------------- driver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0_run", required=True)
    parser.add_argument("--r1_run", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--loo_samples", type=int, default=160)
    parser.add_argument("--eval_seed", type=int, default=20260915)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out",
        default="../../artifacts/decentralized_ris/evaluation/proposal_diagnostics.json",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    with open(os.path.join(args.r0_run, "summary.json"), encoding="utf-8") as handle:
        base_config = json.load(handle)["config"]

    # Match evaluate.py: the topology and LoS draws depend on the training seed.
    seed_everything(base_config["seed"])
    simulator = ChannelSimulator(
        base_config["M"], base_config["N"], base_config["L"],
        base_config["batch_size"], n_ap=base_config["AP"],
    )
    r0, _, _ = load_variant(args.r0_run, args.checkpoint, simulator, device)
    r0_as_r1s, _, _ = load_variant(
        args.r0_run, args.checkpoint, simulator, device, arch_override="r1_shared"
    )
    r1s, _, _ = load_variant(args.r1_run, args.checkpoint, simulator, device)

    traced = {"r0_policy": r0_as_r1s, "r1_shared": r1s}
    batch_size = base_config["batch_size"]
    users = base_config["K"]
    threshold = base_config.get("assoc_threshold", 0.1)
    n_batches = args.samples // batch_size
    loo_batches = args.loo_samples // batch_size

    acc = {
        name: {
            "rho_centralized": [], "rho_decentralized": [], "rho_dec_active_only": [],
            "active_aps": [],
            "mag_vs_agreement_per_element": [], "mag_vs_agreement_per_ap_ris": [],
            "mag_vs_loo_per_ap_ris": [], "agreement_vs_loo_per_ap_ris": [],
            "agreement_within_share": [], "magnitude_within_share": [],
            "mag_cv_across_elements": [],
        }
        for name in traced
    }
    cross_dec, cross_cen = [], []

    with torch.no_grad(), temporary_seed(args.eval_seed):
        for index in range(n_batches):
            features, edges, masks, direct, _ = simulator.training_batch(
                users, threshold, threshold
            )
            features, edges, direct = (
                features.to(device), edges.to(device), direct.to(device)
            )
            cen_traces, cen_out = {}, {}
            for name, model in traced.items():
                trace = {}
                cen_out[name] = model.centralized(features, edges, masks, direct, trace)
                cen_traces[name] = trace
            w0c, t0c = r0.centralized(features, edges, masks, direct)
            cross_cen.append(cross_rates(
                {"r0": (w0c, t0c), "r1s": cen_out["r1_shared"]}, simulator, device
            ))

            dfeatures, dedges, dmasks, ddirect = simulator.decentralized_batch(
                users, threshold, threshold, regenerate_channels=False
            )
            dfeatures = [t.to(device) for t in dfeatures]
            dedges = [t.to(device) for t in dedges]
            ddirect = [t.to(device) for t in ddirect]
            dec_traces, dec_out = {}, {}
            for name, model in traced.items():
                trace = {}
                dec_out[name] = model.decentralized(dfeatures, dedges, dmasks, ddirect, trace)
                dec_traces[name] = trace
            w0d, t0d = r0.decentralized(dfeatures, dedges, dmasks, ddirect)
            cross_dec.append(cross_rates(
                {"r0": (w0d, t0d), "r1s": dec_out["r1_shared"]}, simulator, device
            ))

            for name in traced:
                store = acc[name]
                cen, dec = cen_traces[name], dec_traces[name]
                store["rho_centralized"].append(
                    float(concentration(cen["proposals"]).mean())
                )
                store["rho_decentralized"].append(
                    float(concentration(dec["proposals"]).mean())
                )
                source_active = dec["source_active"]
                store["rho_dec_active_only"].append(
                    float(concentration(dec["proposals"], source_active).mean())
                )
                store["active_aps"].append(float(source_active.sum(dim=1).mean()))

                # Magnitude against reliability, in the deployed (local CSI) mode.
                magnitude = dec["z_pairs"].norm(dim=-1).cpu().numpy().astype(np.float64)
                agree = agreement_leave_one_out(dec["proposals"]).cpu().numpy()
                agree = agree.astype(np.float64)
                store["mag_vs_agreement_per_element"].append(
                    float(np.nanmean(_spearman_along(magnitude, agree, axis=1)))
                )
                mag_lr = magnitude.mean(axis=3)                      # (B, A, R)
                agree_lr = agree.mean(axis=3)
                store["mag_vs_agreement_per_ap_ris"].append(
                    float(np.nanmean(_spearman_along(mag_lr, agree_lr, axis=1)))
                )

                # How much of the structure survives collapsing to one scalar
                # per AP-RIS pair?  A high within-pair share means a scalar
                # cannot represent it.
                for key, array in (("agreement", agree), ("magnitude", magnitude)):
                    # within: spread across elements of one AP-RIS pair.
                    # between: spread across AP-RIS pairs, inside each sample.
                    within = float(array.var(axis=3).mean())
                    pair_means = array.mean(axis=3).reshape(array.shape[0], -1)
                    between = float(pair_means.var(axis=1).mean())
                    store[f"{key}_within_share"].append(
                        float(within / (within + between)) if within + between > 0 else np.nan
                    )
                store["mag_cv_across_elements"].append(
                    float(np.mean(magnitude.std(axis=3) / np.maximum(magnitude.mean(axis=3), 1e-12)))
                )

                beamformer, phase = dec_out[name]
                for scheme, rate in weighting_ladder(
                    dec["proposals"], dec["z_pairs"], beamformer, simulator, device
                ).items():
                    store.setdefault(f"ladder_{scheme}", []).append(rate)

                if index < loo_batches:
                    _, full, _ = simulator.loss(beamformer, phase, device)
                    deltas = leave_one_ap_out_rates(
                        dec["proposals"], beamformer, simulator, device, float(full)
                    )
                    mag_ap_ris = mag_lr.mean(axis=0)                 # (A, R)
                    agree_ap_ris = agree_lr.mean(axis=0)
                    store["mag_vs_loo_per_ap_ris"].append(
                        float(np.nanmean(_spearman_along(
                            mag_ap_ris.T, deltas.T, axis=1)))
                    )
                    store["agreement_vs_loo_per_ap_ris"].append(
                        float(np.nanmean(_spearman_along(
                            agree_ap_ris.T, deltas.T, axis=1)))
                    )

    results = {
        "config": {
            "r0_run": args.r0_run, "r1_run": args.r1_run,
            "checkpoint": args.checkpoint, "samples": args.samples,
            "loo_samples": args.loo_samples, "eval_seed": args.eval_seed,
            "batches": n_batches,
        },
        "per_model": {
            name: {key: _summary(values) for key, values in store.items()}
            for name, store in acc.items()
        },
        "cross_substitution": {
            "decentralized": {
                key: _summary([row[key] for row in cross_dec]) for key in cross_dec[0]
            },
            "centralized": {
                key: _summary([row[key] for row in cross_cen]) for key in cross_cen[0]
            },
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    # Per-batch values so the cross-substitution contrasts can be paired: the
    # four cells share a channel realisation, so their differences are far
    # tighter than the individual cell standard errors suggest.
    paired = {
        f"cross_{mode}__{key}": np.asarray([row[key] for row in rows])
        for mode, rows in (("decentralized", cross_dec), ("centralized", cross_cen))
        for key in rows[0]
    }
    paired.update({
        f"{name}__{key}": np.asarray(values, dtype=np.float64)
        for name, store in acc.items() for key, values in store.items()
    })
    np.savez(os.path.splitext(args.out)[0] + "_paired.npz", **paired)

    for name, store in results["per_model"].items():
        print(f"\n=== {name} ===")
        for key, value in store.items():
            if value["mean"] is None:
                continue
            print(f"  {key:34s} {value['mean']:+.4f} +/- {value['sem']:.4f}  (n={value['n']})")
    for mode, table in results["cross_substitution"].items():
        print(f"\n=== cross substitution: {mode} ===")
        for key, value in table.items():
            print(f"  {key:26s} {value['mean']:8.5f} +/- {value['sem']:.5f}")
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
