"""Continuous-phase achievable reference for a trained decentralized RIS checkpoint.

The greedy 2-bit search in `experiments/discrete_cd.py` lower-bounds the joint
optimum three times over: it is a coordinate-wise local optimum, it is confined
to the 2-bit grid, and it holds the beamformer fixed. This script removes the
last two restrictions by optimizing the true sum rate directly with Adam:

    phase-only : theta free (continuous), W frozen at the GNN output
    joint      : theta and W both free, started from the GNN solution
    joint-rand : same, started from random phases and beamformers

Unit modulus is exact by construction (theta = e^{j phi}), and the per-AP power
constraint reuses the same mask-then-normalize rule as `node_update`, so every
iterate stays inside the feasible set of Problem (8). Nothing in the baseline is
modified; the checkpoint is read only.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import resolve_device, seed_everything
from model import BaselineNet, load_checkpoint
from rates import RatePrecompute, quantize_phase
from simulation import ChannelSimulator


def apply_power_constraint(W_raw, mask, alpha_logit, pmax, num_ap):
    """Mask inactive AP-UE columns then renormalize each AP to sqrt(Pmax*alpha).

    Mirrors the normalization in `node_update.forward`, so the feasible set here
    is the same one the GNN optimizes over.
    """
    B, twoM, K_tot = W_raw.shape
    K_user = K_tot // num_ap
    W = W_raw * mask.unsqueeze(1)
    W = W.reshape(B, twoM, num_ap, K_user).permute(0, 2, 1, 3).reshape(B, num_ap, -1)
    alpha = torch.sigmoid(alpha_logit)                                   # (B, num_ap)
    W = F.normalize(W, dim=2, eps=1e-8) * torch.sqrt(pmax * alpha).unsqueeze(2)
    W = W.reshape(B, num_ap, twoM, K_user).permute(0, 2, 1, 3).reshape(B, twoM, K_tot)
    return W


def optimize(pre, W_gnn, theta_gnn, mask, pmax, num_ap, mode, steps, lr_phi, lr_w, seed):
    """Adam ascent on the true sum rate. Returns per-sample rates and the iterate."""
    device = W_gnn.device
    g = torch.Generator(device="cpu").manual_seed(seed)

    if mode == "joint_rand":
        phi0 = (2 * torch.pi * torch.rand(theta_gnn.shape[:-1], generator=g)).to(device)
        W0 = torch.randn(W_gnn.shape, generator=g).to(device)
        a0 = torch.zeros((W_gnn.shape[0], num_ap), device=device)
    else:
        phi0 = torch.atan2(theta_gnn[..., 1], theta_gnn[..., 0]).clone()
        W0 = W_gnn.clone()
        # alpha that reproduces the GNN's own per-AP transmit power exactly.
        K_user = W_gnn.shape[2] // num_ap
        p_l = (W_gnn.reshape(W_gnn.shape[0], W_gnn.shape[1], num_ap, K_user) ** 2).sum(dim=(1, 3))
        a0 = torch.logit((p_l / pmax).clamp(1e-4, 1 - 1e-4))

    phi = phi0.clone().requires_grad_(True)
    params = [{"params": [phi], "lr": lr_phi}]
    if mode == "phase_only":
        W_raw, alpha_logit = W0, a0
    else:
        W_raw = W0.clone().requires_grad_(True)
        alpha_logit = a0.clone().requires_grad_(True)
        params.append({"params": [W_raw, alpha_logit], "lr": lr_w})

    opt = torch.optim.Adam(params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    history = []
    for t in range(steps):
        theta = torch.stack((phi.cos(), phi.sin()), dim=-1)
        W = W_gnn if mode == "phase_only" else apply_power_constraint(
            W_raw, mask, alpha_logit, pmax, num_ap)
        rate = pre.sum_rate(W, theta)
        loss = -rate.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if t % max(1, steps // 8) == 0 or t == steps - 1:
            history.append(round(float(rate.mean().item()), 4))

    with torch.no_grad():
        theta = torch.stack((phi.cos(), phi.sin()), dim=-1)
        W = W_gnn if mode == "phase_only" else apply_power_constraint(
            W_raw, mask, alpha_logit, pmax, num_ap)
        rate = pre.sum_rate(W, theta)
        rate_q = pre.sum_rate(W, quantize_phase(theta, 2))
    return rate, rate_q, W, theta, history


def check_feasibility(W, theta, mask, pmax, num_ap):
    K_user = W.shape[2] // num_ap
    p_l = (W.reshape(W.shape[0], W.shape[1], num_ap, K_user) ** 2).sum(dim=(1, 3))
    leaked = (W * (1 - mask).unsqueeze(1)).abs().max()
    return {
        "max_power_over_budget": float((p_l - pmax).max().item()),
        "max_abs_modulus_error": float((theta.norm(dim=-1) - 1).abs().max().item()),
        "max_abs_masked_beamformer": float(leaked.item()),
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
    p.add_argument("--samples", type=int, default=320)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr_phi", type=float, default=0.05)
    p.add_argument("--lr_w", type=float, default=0.02)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", default="results_continuous_ceiling")
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

    keys = ["gnn", "gnn_round", "phase_only", "phase_only_q",
            "joint", "joint_q", "joint_rand", "joint_rand_q"]
    rec = {k: [] for k in keys}
    feas, hist = {}, {}

    batches = max(1, args.samples // args.batch_size)
    for it in range(batches):
        uf, e, ui, ed, _ = dataloader.training_batch(
            args.K, args.assoc_threshold, args.assoc_threshold
        )
        mask = torch.as_tensor(np.array(ui, dtype=bool), dtype=torch.float32, device=device)
        uf, e, ed = uf.to(device), e.to(device), ed.to(device)
        pre = RatePrecompute(dataloader, device)

        with torch.no_grad():
            W_gnn, theta_gnn = model(uf, e, ui, ed, training=True)
            rec["gnn"].append(pre.sum_rate(W_gnn, theta_gnn).cpu().numpy())
            rec["gnn_round"].append(
                pre.sum_rate(W_gnn, quantize_phase(theta_gnn, 2)).cpu().numpy())

        for mode in ("phase_only", "joint"):
            r, rq, W, th, h = optimize(pre, W_gnn, theta_gnn, mask, pmax_w, args.num_ap,
                                       mode, args.steps, args.lr_phi, args.lr_w, args.seed)
            rec[mode].append(r.detach().cpu().numpy())
            rec[f"{mode}_q"].append(rq.detach().cpu().numpy())
            hist.setdefault(mode, h)
            feas.setdefault(mode, check_feasibility(W.detach(), th.detach(), mask,
                                                    pmax_w, args.num_ap))

        best = best_q = None
        for r_idx in range(args.restarts):
            r, rq, W, th, h = optimize(pre, W_gnn, theta_gnn, mask, pmax_w, args.num_ap,
                                       "joint_rand", args.steps, args.lr_phi, args.lr_w,
                                       args.seed + 1000 * r_idx)
            r, rq = r.detach(), rq.detach()
            best = r if best is None else torch.maximum(best, r)
            best_q = rq if best_q is None else torch.maximum(best_q, rq)
            hist.setdefault("joint_rand", h)
            feas.setdefault("joint_rand", check_feasibility(W.detach(), th.detach(), mask,
                                                            pmax_w, args.num_ap))
        rec["joint_rand"].append(best.cpu().numpy())
        rec["joint_rand_q"].append(best_q.cpu().numpy())

        if (it + 1) % max(1, batches // 8) == 0:
            print(f"[{it+1}/{batches}] " + " ".join(
                f"{k}={np.concatenate(v).mean():.4f}" for k, v in rec.items()))

    stacked = {k: np.concatenate(v) for k, v in rec.items()}
    B = args.batch_size

    def clus(v):
        g = v.reshape(-1, B).mean(axis=1)
        return float(v.mean()), float(g.std(ddof=1) / np.sqrt(len(g))), g

    summary = {"checkpoint": args.ckpt, "samples": int(len(stacked["gnn"])),
               "steps": args.steps, "restarts": args.restarts,
               "feasibility": feas, "objective_history": hist,
               "means": {}, "gaps": {}}
    lines = []
    for k, v in stacked.items():
        m, se, _ = clus(v)
        summary["means"][k] = {"mean": m, "clustered_se": se}
        lines.append(f"{k}: {m:.5f} (cSE {se:.4f})")
    for a, b in [("phase_only", "gnn"), ("joint", "gnn"), ("joint_rand", "joint"),
                 ("joint", "phase_only"), ("joint_q", "gnn_round")]:
        d = stacked[a] - stacked[b]
        m, se, g = clus(d)
        summary["gaps"][f"{a}_minus_{b}"] = {
            "mean": m, "clustered_se": se, "t": m / se if se else float("nan"),
            "batch_win_fraction": float((g > 0).mean())}
        lines.append(f"{a} - {b}: {m:+.5f} cSE={se:.4f} t={m/se:7.2f} "
                     f"batch_win={100*(g>0).mean():.1f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, "paired_sum_rates.npz"), **stacked)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("feasibility:", json.dumps(feas, indent=1))
    print("objective history:", json.dumps(hist, indent=1))


if __name__ == "__main__":
    main()
