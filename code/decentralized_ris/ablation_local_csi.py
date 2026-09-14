"""Ablation on the paper's extra local-CSI feedback assumption (eq. 10, second set).

The paper lets a UE served by several APs report its CSI with *all* of its
serving APs to each of them. This script re-runs decentralized inference with
that second feedback set removed, so each AP sees only the CSI of its own
served links, and measures what the centralized-vs-decentralized gap becomes.

The decentralized forward pass is reproduced here rather than edited in
`model_2.py`, so the baseline stays byte-identical. `--verify` checks that the
reproduction matches `node_update.forward(training=False)` exactly when the
extra feedback is enabled.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from data import MyDataLoader
from model_2 import node_update
from utils_return_indivial_rates import discrete_mapping, user_pruning
from discrete_cd_baseline import RatePrecompute, coordinate_descent, phase_levels


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def decentralized_forward(model, user_feature, e, user_index, e_dir, include_cross_ap_csi):
    """Mirror of the `training=False` branch of `node_update.forward`.

    `include_cross_ap_csi=True` reproduces the paper. Setting it to False drops
    the second set of eq. (10): each AP then only sees links it serves itself.
    Also returns, per AP, how many AP-UE nodes were visible.
    """
    device = model.device
    n_ap = len(user_feature)
    batch_size = user_feature[0].shape[0]

    W = torch.zeros((batch_size, 2 * model.M, user_feature[0].shape[2] * n_ap)).to(device)
    theta = torch.zeros((batch_size, user_feature[0].shape[1], model.N, 2)).to(device)
    visible = np.zeros((batch_size, n_ap))

    for sample in range(batch_size):
        RIS_prior_merge = []
        for num_BS in range(n_ap):
            user_feature_cat = user_feature[0]
            e_cat = e[0]
            e_dir_cat = e_dir[0]
            for i in range(n_ap - 1):
                user_feature_cat = torch.cat((user_feature_cat, user_feature[i + 1]), dim=2)
                e_cat = torch.cat((e_cat, e[i + 1]), dim=2)
                e_dir_cat = torch.cat((e_dir_cat, e_dir[i + 1]), dim=2)

            k = len(user_index[num_BS][sample])
            new_user_index = np.array([False]).repeat(n_ap * k)
            new_user_index[num_BS * k:(num_BS + 1) * k] = user_index[num_BS][sample]
            actual_served_node = np.copy(new_user_index)

            if include_cross_ap_csi:
                served_user_id = np.where(user_index[num_BS][sample] == True)[0]
                if len(served_user_id) != 0:
                    for tmp_BS in range(n_ap):
                        if tmp_BS == num_BS:
                            continue
                        for user_id in served_user_id:
                            if user_index[tmp_BS][sample][user_id] == True:
                                new_user_index[tmp_BS * k:(tmp_BS + 1) * k][user_id] = True

            visible[sample, num_BS] = int(new_user_index.sum())

            new_user_index = new_user_index == False
            user_feature_cat[sample, :, new_user_index, :] = 0
            e_cat[sample, :, new_user_index] = 0
            e_dir_cat[sample, :, new_user_index] = 0
            new_user_index[new_user_index == False] = True
            user_feature_ext = user_feature_cat[sample, :, new_user_index, :].unsqueeze(0)

            A = user_pruning(user_feature_ext.shape[2], 0, duplicate=False)
            A = A == 1

            e_ext = e_cat[sample, :, new_user_index].unsqueeze(0)
            e_dir_ext = e_dir_cat[sample, :, new_user_index].unsqueeze(0)
            if np.sum(user_index[num_BS][sample]) == 0:
                continue

            uk, rl = model.init_user(user_feature_ext, e_ext, e_dir_ext)
            for i, _ in enumerate(model.update_list):
                uk, rl = model.update_list[i](uk, rl, e_ext, A, e_dir_ext)
                uk = uk.to(device)
                rl = rl.to(device)

            AP_coeff = model.AP_coeff_NN_list[num_BS](uk)
            W_out = model.BS_readout(uk)

            tmp_W = W_out[0, :, :]
            W[sample, :, actual_served_node] = tmp_W[:, actual_served_node]

            BS_W = W[sample, :, num_BS * k:(num_BS + 1) * k].reshape((1, -1))
            BS_W = F.normalize(BS_W, dim=1, eps=1e-8) * torch.sqrt(model.Pmax * AP_coeff)
            W[sample, :, num_BS * k:(num_BS + 1) * k] = BS_W.reshape((2 * model.M, -1))

            e_AP = e[num_BS][sample, :, :].reshape(1, -1)
            RIS_prior_merge.append(model.RIS_readout_AP_list[num_BS](rl, e_AP))

        theta[sample] = model.RIS_merge(RIS_prior_merge).squeeze()

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
    p.add_argument("--out_dir", default="results_local_csi_ablation")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    dataloader = MyDataLoader(args.M, args.N, args.L, args.batch_size)
    dataloader.BS_RIS_association()
    pmax_w = 10 ** ((args.pmax_dbm - 30) / 10)
    model = node_update(args.M, args.N, args.L, args.D, pmax_w, 2, args.ch,
                        args.num_ap, device).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
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
        uf, e, ui, ed, _ = dataloader.gen_training_data(
            args.K, args.assoc_threshold, args.assoc_threshold, duplicate=False)
        uf, e, ed = uf.to(device), e.to(device), ed.to(device)
        pre = RatePrecompute(dataloader, device)

        with torch.no_grad():
            W_c, th_c = model(uf, e, ui, ed, training=True, duplicate=False)

        outs = {"cen": (W_c, th_c)}
        for mode, flag in (("dec_paper", True), ("dec_own_only", False)):
            uf_d, e_d, ui_d, ed_d = dataloader.gen_testing_data(
                args.K, args.assoc_threshold, args.assoc_threshold,
                duplicate=False, regenerate_channels=False)
            for i in range(len(uf_d)):
                uf_d[i], e_d[i], ed_d[i] = uf_d[i].to(device), e_d[i].to(device), ed_d[i].to(device)
            with torch.no_grad():
                W_d, th_d, vis = decentralized_forward(model, uf_d, e_d, ui_d, ed_d, flag)
            outs[mode] = (W_d, th_d)
            visible_counts[mode].append(vis)

            if mode == "dec_paper" and verified is None:
                uf_r, e_r, ui_r, ed_r = dataloader.gen_testing_data(
                    args.K, args.assoc_threshold, args.assoc_threshold,
                    duplicate=False, regenerate_channels=False)
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
                th_r = discrete_mapping(thx, args.num_bits)
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
