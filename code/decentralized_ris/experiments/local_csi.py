"""Ablation on the paper's extra local-CSI feedback assumption (eq. 10, second set).

The paper lets a UE served by several APs report its CSI with *all* of its
serving APs to each of them. This script re-runs decentralized inference with
that second feedback set removed, so each AP sees only the CSI of its own
served links, and measures what the centralized-vs-decentralized gap becomes.

The canonical model exposes the visibility switch directly. `--verify` checks
that the paper setting matches the normal decentralized forward path.
"""

import argparse
import json
import os

import numpy as np
import torch

from evaluate import resolve_device, seed_everything
from experiments.discrete_cd import coordinate_descent
from model import BaselineNet, load_checkpoint
from rates import RatePrecompute, phase_levels, quantize_phase
from simulation import ChannelSimulator


def decentralized_forward(model, user_feature, e, user_index, e_dir, include_cross_ap_csi):
    W, theta = model.decentralized(
        user_feature, e, user_index, e_dir,
        include_cross_ap_csi=include_cross_ap_csi,
    )
    masks = np.stack(user_index, axis=1).astype(bool)
    if include_cross_ap_csi:
        visible = np.stack(
            [(masks[:, ap:ap + 1, :] & masks).sum(axis=(1, 2))
             for ap in range(masks.shape[1])],
            axis=1,
        )
    else:
        visible = masks.sum(axis=2)
    return W, theta, visible


def phase_stats(theta):
    """Circular spread of the RIS phases, to detect a collapsed readout."""
    ang = torch.atan2(theta[..., 1], theta[..., 0])                  # (B, R, N)
    # Circular variance within each RIS: 1 - |mean resultant vector|.
    mrl = torch.sqrt(torch.cos(ang).mean(dim=2) ** 2 + torch.sin(ang).mean(dim=2) ** 2)
    circ_var = (1 - mrl)
    quadrant = torch.floor(((ang % (2 * torch.pi)) / (torch.pi / 2))).long()
    frac = torch.stack([(quadrant == q).float().mean() for q in range(4)])
    return {
        "circular_variance_mean": float(circ_var.mean().item()),
        "quadrant_fractions": [round(float(x), 4) for x in frac],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--M", type=int, default=2)
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--L", type=int, default=4, help="number of RISs")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--num_ap", type=int, default=5)
    p.add_argument("--D", type=int, default=6)
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--pmax_dbm", type=float, default=15.0)
    p.add_argument("--assoc_threshold", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--samples", type=int, default=800)
    p.add_argument("--num_bits", type=int, default=2)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--greedy", action="store_true", help="also run the greedy phase search")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--out_dir", default="../../artifacts/decentralized_ris/local_csi_ablation"
    )
    args = p.parse_args()

    device = resolve_device(args.device)
    seed_everything(args.seed)

    dataloader = ChannelSimulator(
        args.M, args.N, args.L, args.batch_size, n_ap=args.num_ap
    )
    pmax_w = 10 ** ((args.pmax_dbm - 30) / 10)
    model = BaselineNet(args.M, args.N, args.L, args.D, pmax_w, args.ch,
                        args.num_ap, device, users_per_ap=args.K).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    levels = phase_levels(args.num_bits, device)
    keys = ["cen", "dec_paper", "dec_own_only"]
    rec = {f"{k}_{m}": [] for k in keys for m in ("cont", "round")}
    if args.greedy:
        rec.update({f"{k}_greedy": [] for k in keys})
    visible_counts = {"dec_paper": [], "dec_own_only": []}
    verified = None
    phase_diag = {}

    batches = max(1, args.samples // args.batch_size)
    for it in range(batches):
        uf, e, ui, ed, _ = dataloader.training_batch(
            args.K, args.assoc_threshold, args.assoc_threshold
        )
        uf, e, ed = uf.to(device), e.to(device), ed.to(device)
        pre = RatePrecompute(dataloader, device)

        with torch.no_grad():
            W_c, th_c = model(uf, e, ui, ed, training=True)

        outs = {"cen": (W_c, th_c)}
        for mode, flag in (("dec_paper", True), ("dec_own_only", False)):
            uf_d, e_d, ui_d, ed_d = dataloader.decentralized_batch(
                args.K, args.assoc_threshold, args.assoc_threshold,
                regenerate_channels=False)
            for i in range(len(uf_d)):
                uf_d[i], e_d[i], ed_d[i] = uf_d[i].to(device), e_d[i].to(device), ed_d[i].to(device)
            with torch.no_grad():
                W_d, th_d, vis = decentralized_forward(model, uf_d, e_d, ui_d, ed_d, flag)
            outs[mode] = (W_d, th_d)
            visible_counts[mode].append(vis)

            if mode == "dec_paper" and verified is None:
                uf_r, e_r, ui_r, ed_r = dataloader.decentralized_batch(
                    args.K, args.assoc_threshold, args.assoc_threshold,
                    regenerate_channels=False)
                for i in range(len(uf_r)):
                    uf_r[i], e_r[i], ed_r[i] = uf_r[i].to(device), e_r[i].to(device), ed_r[i].to(device)
                with torch.no_grad():
                    W_ref, th_ref = model(uf_r, e_r, ui_r, ed_r, training=False)
                verified = {
                    "max_abs_diff_W": float((W_d - W_ref).abs().max().item()),
                    "max_abs_diff_theta": float((th_d - th_ref).abs().max().item()),
                }
                print(f"[verify] reproduction vs model(training=False): "
                      f"max|dW|={verified['max_abs_diff_W']:.3e} "
                      f"max|dtheta|={verified['max_abs_diff_theta']:.3e}")

        for name, (Wx, thx) in outs.items():
            with torch.no_grad():
                rec[f"{name}_cont"].append(pre.sum_rate(Wx, thx).cpu().numpy())
                th_r = quantize_phase(thx, args.num_bits)
                rec[f"{name}_round"].append(pre.sum_rate(Wx, th_r).cpu().numpy())
                if args.greedy:
                    _, r_g, _ = coordinate_descent(pre, Wx, th_r, levels, args.rounds)
                    rec[f"{name}_greedy"].append(r_g.cpu().numpy())
            if not phase_diag.get(name):
                phase_diag[name] = phase_stats(thx)

        if (it + 1) % max(1, batches // 10) == 0:
            print(f"[{it+1}/{batches}] " + " ".join(
                f"{k}={np.concatenate(v).mean():.4f}" for k, v in rec.items()))

    stacked = {k: np.concatenate(v) for k, v in rec.items()}
    B = args.batch_size

    def clustered(v):
        g = v.reshape(-1, B).mean(axis=1)
        return float(v.mean()), float(g.std(ddof=1) / np.sqrt(len(g))), g

    summary = {"checkpoint": args.ckpt, "samples": int(len(stacked["cen_cont"])),
               "batch_size": B,
               "verification": verified, "phase_diagnostics": phase_diag,
               "means": {}, "gaps": {},
               "visible_nodes": {m: float(np.concatenate(v).mean()) for m, v in visible_counts.items()}}

    lines = []
    for k, v in stacked.items():
        m, se, _ = clustered(v)
        summary["means"][k] = {"mean": m, "clustered_se": se}
        lines.append(f"{k}: {m:.5f} (cSE {se:.4f})")

    for a, b in [("cen_cont", "dec_paper_cont"), ("cen_cont", "dec_own_only_cont"),
                 ("dec_paper_cont", "dec_own_only_cont")]:
        diff = stacked[a] - stacked[b]
        m, se, g = clustered(diff)
        summary["gaps"][f"{a}_minus_{b}"] = {
            "mean": m, "clustered_se": se, "t": m / se if se else float("nan"),
            "batch_win_fraction": float((g > 0).mean())}
        lines.append(f"{a} - {b}: {m:+.5f} cSE={se:.4f} t={m/se:7.2f} batch_win={100*(g>0).mean():.1f}%")

    lines.append(f"visible AP-UE nodes per AP: paper={summary['visible_nodes']['dec_paper']:.3f} "
                 f"own_only={summary['visible_nodes']['dec_own_only']:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, "paired_sum_rates.npz"), **stacked)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(phase_diag, indent=1))


if __name__ == "__main__":
    main()
