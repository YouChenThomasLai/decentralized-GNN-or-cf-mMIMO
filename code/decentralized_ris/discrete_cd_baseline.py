"""Discrete RIS phase baseline for a trained decentralized RIS checkpoint.

The paper obtains its discrete ("D") curves by rounding the continuous GNN phase
output to the nearest point of the Q-bit grid. This script keeps the beamformer
fixed and instead searches the same grid with element-wise coordinate descent,
which measures how much of the rounding loss is recoverable without retraining.

Nothing here modifies the training pipeline: the checkpoint is loaded read-only
and evaluation reuses `MyDataLoader` exactly as `trainer_2.eval()` does.
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from data import MyDataLoader
from model import BaselineNet, load_checkpoint
from utils_return_indivial_rates import discrete_mapping


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_phase_like(theta, num_bits=None):
    """Mirror of `trainer_2._random_phase_like`."""
    if num_bits is None:
        phase = 2 * torch.pi * torch.rand_like(theta[..., 0])
    else:
        level = 2 ** num_bits
        phase = torch.randint(level, theta.shape[:-1], device=theta.device)
        phase = phase.to(theta.dtype) * (2 * torch.pi / level)
    return torch.stack((phase.cos(), phase.sin()), dim=-1)


class RatePrecompute:
    """Cached real-valued channel terms for one batch of channel realizations.

    Reproduces the arithmetic of `utils_return_indivial_rates.cal_loss`, but
    keeps the channel-dependent parts fixed so that a sum rate can be evaluated
    for many candidate phase vectors without rebuilding them.
    """

    SIGMA = (2e-2) ** 2

    def __init__(self, dataloader, device):
        H_list, direct_list = [], []
        for bs in dataloader.BS_array:
            H_bs, direct_bs = bs.get_channel()
            H_list.append(H_bs)
            direct_list.append(direct_bs)
        H = np.concatenate(H_list, axis=4)                       # (B, M, N, L, K_tot)
        direct = np.concatenate(direct_list, axis=1)             # (B, K_tot, M)

        self.device = device
        self.num_bs = len(dataloader.BS_array)
        self.batch, self.M, self.N, self.L, self.K_tot = (
            H.shape[0], H.shape[1], H.shape[2], H.shape[3], H.shape[4])
        self.K_user = self.K_tot // self.num_bs

        # A_re[b, l, k] is the (2M, 2N) real embedding used by cal_loss.
        A = np.transpose(H, (0, 3, 4, 1, 2))                     # (B, L, K_tot, M, N)
        top = np.concatenate([A.real, A.imag], axis=4)           # (B, L, K_tot, M, 2N)
        bot = np.concatenate([-A.imag, A.real], axis=4)
        A_re = np.concatenate([top, bot], axis=3)                # (B, L, K_tot, 2M, 2N)
        self.A_re = torch.as_tensor(A_re, dtype=torch.float32, device=device)

        # Per-element columns, so one RIS element can be updated in isolation.
        cols = self.A_re.reshape(self.batch, self.L, self.K_tot, 2 * self.M, 2, self.N)
        self.elem = cols.permute(0, 1, 5, 2, 3, 4).contiguous()  # (B, L, N, K_tot, 2M, 2)

        self.direct = torch.as_tensor(
            np.concatenate([direct.real, direct.imag], axis=2),
            dtype=torch.float32, device=device)                  # (B, K_tot, 2M)

    def effective_channel(self, theta):
        """concat_H in cal_loss: direct channel plus every RIS contribution."""
        phase = torch.cat((theta[..., 0], theta[..., 1]), dim=2)          # (B, L, 2N)
        contrib = torch.einsum('blkam,blm->bka', self.A_re, phase)
        return contrib + self.direct                                      # (B, K_tot, 2M)

    def element_contribution(self, l, n, value):
        """Contribution of RIS element (l, n) for a phase `value` of shape (B, 2)."""
        return torch.einsum('bkam,bm->bka', self.elem[:, l, n], value)

    def sum_rate_from_effective(self, W, eff):
        """Per-sample sum rate given the cached effective channels."""
        Wt = W.transpose(2, 1)                                            # (B, K_tot, 2M)
        W_re = Wt[:, :, :self.M].reshape(self.batch, self.num_bs, self.K_user, self.M)
        W_im = Wt[:, :, self.M:].reshape(self.batch, self.num_bs, self.K_user, self.M)
        A1 = eff[:, :, :self.M].reshape(self.batch, self.num_bs, self.K_user, self.M)
        A2 = eff[:, :, self.M:].reshape(self.batch, self.num_bs, self.K_user, self.M)

        z_re = torch.einsum('bsjm,bsim->bsij', W_re, A1) - torch.einsum('bsjm,bsim->bsij', W_im, A2)
        z_im = torch.einsum('bsjm,bsim->bsij', W_im, A1) + torch.einsum('bsjm,bsim->bsij', W_re, A2)

        # Contributions of all APs add coherently at the receiver.
        acc_re = z_re.sum(dim=1)                                          # (B, K_user, K_user)
        acc_im = z_im.sum(dim=1)

        idx = torch.arange(self.K_user, device=eff.device)
        signal = acc_re[:, idx, idx] ** 2 + acc_im[:, idx, idx] ** 2

        power = acc_re ** 2 + acc_im ** 2
        power[:, idx, idx] = 0.0
        interference = power.sum(dim=2)

        sinr = signal / (interference + self.SIGMA)
        return torch.log2(1.0 + sinr).sum(dim=1)                          # (B,)

    def sum_rate(self, W, theta):
        return self.sum_rate_from_effective(W, self.effective_channel(theta))


def coordinate_descent(pre, W, theta_init, levels, rounds, tol=1e-6):
    """Greedy per-element search over the discrete phase grid with W held fixed."""
    theta = theta_init.clone()
    eff = pre.effective_channel(theta)
    best_rate = pre.sum_rate_from_effective(W, eff)
    history = [best_rate.mean().item()]

    for _ in range(rounds):
        start = best_rate.clone()
        for l in range(pre.L):
            for n in range(pre.N):
                current = theta[:, l, n, :]
                rest = eff - pre.element_contribution(l, n, current)

                cand_eff = []
                cand_rate = []
                for level in levels:
                    value = level.expand(pre.batch, 2)
                    trial = rest + pre.element_contribution(l, n, value)
                    cand_eff.append(trial)
                    cand_rate.append(pre.sum_rate_from_effective(W, trial))

                rates = torch.stack(cand_rate, dim=1)                      # (B, levels)
                pick = rates.argmax(dim=1)
                best_rate = rates.gather(1, pick.unsqueeze(1)).squeeze(1)
                theta[:, l, n, :] = levels[pick]
                eff = torch.stack(cand_eff, dim=1)[torch.arange(pre.batch), pick]

        history.append(best_rate.mean().item())
        if (best_rate - start).abs().max().item() < tol:
            break

    return theta, best_rate, history


def phase_levels(num_bits, device):
    angles = torch.arange(2 ** num_bits, device=device, dtype=torch.float32)
    angles = angles * (2 * torch.pi / (2 ** num_bits))
    return torch.stack((angles.cos(), angles.sin()), dim=-1)               # (levels, 2)


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    dataloader = MyDataLoader(args.M, args.N, args.L, args.batch_size)
    dataloader.BS_RIS_association()

    pmax_w = 10 ** ((args.pmax_dbm - 30) / 10)
    model = BaselineNet(args.M, args.N, args.L, args.D, pmax_w, args.ch,
                        args.num_ap, device).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    levels = phase_levels(args.num_bits, device)
    keys = ["cont", "round", "cd", "cd_from_random", "random"]
    records = {f"{scheme}_{key}": [] for scheme in ("C", "D") for key in keys}
    cd_history = {"C": [], "D": []}
    verify_err = None
    verified_cd = {}

    batches = max(1, args.samples // args.batch_size)
    for it in range(batches):
        user_feature, e, user_index, e_dir, _ = dataloader.gen_training_data(
            args.K, args.assoc_threshold, args.assoc_threshold, duplicate=False)
        user_feature = user_feature.to(device)
        e = e.to(device)
        e_dir = e_dir.to(device)

        pre = RatePrecompute(dataloader, device)

        with torch.no_grad():
            W_c, theta_c = model(user_feature, e, user_index, e_dir, training=True, duplicate=False)

        if verify_err is None:
            _, ref_rate, _ = dataloader.compute_loss(W_c, theta_c, pmax_w, device)
            fast_rate = pre.sum_rate(W_c, theta_c).mean()
            verify_err = abs(ref_rate.item() - fast_rate.item())
            print(f"[check] cal_loss={ref_rate.item():.6f} fast={fast_rate.item():.6f} "
                  f"abs_err={verify_err:.3e}")

        # Decentralized inference reuses the same channel realizations.
        user_feature_d, e_d, user_index_d, e_dir_d = dataloader.gen_testing_data(
            args.K, args.assoc_threshold, args.assoc_threshold,
            duplicate=False, regenerate_channels=False)
        for idx in range(len(user_feature_d)):
            user_feature_d[idx] = user_feature_d[idx].to(device)
            e_d[idx] = e_d[idx].to(device)
            e_dir_d[idx] = e_dir_d[idx].to(device)
        with torch.no_grad():
            W_d, theta_d = model(user_feature_d, e_d, user_index_d, e_dir_d, training=False)

        for scheme, W, theta in (("C", W_c, theta_c), ("D", W_d, theta_d)):
            with torch.no_grad():
                rate_cont = pre.sum_rate(W, theta)
                theta_round = discrete_mapping(theta, args.num_bits)
                rate_round = pre.sum_rate(W, theta_round)
                theta_rand = random_phase_like(theta, args.num_bits)
                rate_rand = pre.sum_rate(W, theta_rand)

                theta_cd, rate_cd, hist = coordinate_descent(pre, W, theta_round, levels, args.rounds)
                _, rate_cd_rand, _ = coordinate_descent(pre, W, theta_rand, levels, args.rounds)

            if scheme not in verified_cd:
                # The headline number comes from the searched phases, so check them
                # against the original cal_loss path rather than only the cached one.
                _, ref_cd, _ = dataloader.compute_loss(W, theta_cd, pmax_w, device)
                modulus = theta_cd.norm(dim=-1)
                verified_cd[scheme] = {
                    "cal_loss": float(ref_cd.item()),
                    "fast": float(rate_cd.mean().item()),
                    "abs_err": abs(ref_cd.item() - rate_cd.mean().item()),
                    "max_modulus_err": float((modulus - 1).abs().max().item()),
                    "on_grid": bool(torch.isclose(
                        theta_cd.unsqueeze(-2), levels, atol=1e-5).all(dim=-1).any(dim=-1).all().item()),
                }
                print(f"[check-cd {scheme}] cal_loss={ref_cd.item():.6f} "
                      f"fast={rate_cd.mean().item():.6f} "
                      f"abs_err={verified_cd[scheme]['abs_err']:.3e} "
                      f"max|theta|-1={verified_cd[scheme]['max_modulus_err']:.2e} "
                      f"on_grid={verified_cd[scheme]['on_grid']}")

            records[f"{scheme}_cont"].append(rate_cont.cpu().numpy())
            records[f"{scheme}_round"].append(rate_round.cpu().numpy())
            records[f"{scheme}_cd"].append(rate_cd.cpu().numpy())
            records[f"{scheme}_cd_from_random"].append(rate_cd_rand.cpu().numpy())
            records[f"{scheme}_random"].append(rate_rand.cpu().numpy())
            cd_history[scheme].append(hist)

        if (it + 1) % max(1, batches // 10) == 0:
            print(f"[{it + 1}/{batches}] "
                  + " ".join(f"{k}={np.concatenate(v).mean():.4f}" for k, v in records.items()))

    stacked = {k: np.concatenate(v) for k, v in records.items()}
    return stacked, cd_history, verify_err, verified_cd


def summarize(stacked, args, verify_err):
    lines = []
    summary = {"checkpoint": args.ckpt, "samples": int(len(stacked["C_cont"])),
               "num_bits": args.num_bits, "rounds": args.rounds,
               "fast_rate_check_abs_err": verify_err, "means": {}, "deltas": {}}

    for key, value in stacked.items():
        summary["means"][key] = float(value.mean())
        lines.append(f"{key}: {value.mean():.5f}")

    for scheme in ("C", "D"):
        cont = stacked[f"{scheme}_cont"]
        rnd = stacked[f"{scheme}_round"]
        cd = stacked[f"{scheme}_cd"]
        cdr = stacked[f"{scheme}_cd_from_random"]
        gap = cont - rnd
        gain = cd - rnd
        summary["deltas"][f"{scheme}_quantization_gap"] = float(gap.mean())
        summary["deltas"][f"{scheme}_cd_gain_over_rounding"] = float(gain.mean())
        summary["deltas"][f"{scheme}_cd_gain_std"] = float(gain.std(ddof=1))
        summary["deltas"][f"{scheme}_cd_win_fraction"] = float((gain > 0).mean())
        summary["deltas"][f"{scheme}_recovered_fraction"] = float(gain.mean() / gap.mean())
        summary["deltas"][f"{scheme}_cd_vs_continuous"] = float((cd - cont).mean())
        summary["deltas"][f"{scheme}_cd_random_init_delta"] = float((cdr - cd).mean())
        lines.append(
            f"{scheme}: quantization_gap={gap.mean():.5f} "
            f"cd_gain={gain.mean():.5f} recovered={100 * gain.mean() / gap.mean():.2f}% "
            f"cd_minus_continuous={(cd - cont).mean():+.5f} "
            f"win_fraction={(gain > 0).mean():.4f}")

    return summary, lines


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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", default="results_discrete_cd")
    args = p.parse_args()

    stacked, cd_history, verify_err, verified_cd = evaluate(args)
    summary, lines = summarize(stacked, args, verify_err)
    summary["cd_verification"] = verified_cd

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, "paired_sum_rates.npz"), **stacked)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"[done] wrote {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
