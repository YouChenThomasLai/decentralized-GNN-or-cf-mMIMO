"""Train one per-RIS action representation variant under the Stage 1 protocol.

Everything except the RIS representation, the AP output interface and the
aggregation is held at the paper's configuration (A=5 APs, M=2, R=4 RIS, N=30,
K=8 UEs/AP, Pmax=15 dBm, D=6, ch=64, batch 8, Adam 1e-4, weight decay 1e-6).
The model is trained centrally on global CSI and validated on both the
centralized path and the eq. (10) local-CSI decentralized path; the
decentralized continuous-phase sum rate is the primary metric and the only one
used to pick a checkpoint.

Each run writes `summary.json` (config, gates, curves, best checkpoint) so the
screening leaderboard can be rebuilt without re-reading logs.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import ris_action_variants as rav
from model import load_checkpoint
from ris_action_variants import VariantNet, unit_modulus_error


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="r0", choices=rav.ARCHS)
    p.add_argument("--identity", default="none", choices=rav.IDENTITIES)
    p.add_argument("--consensus", default=None, choices=list(rav.CONSENSUS))
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--tag", default=None, help="run directory name; defaults to arch-identity-consensus")

    p.add_argument("--M", type=int, default=2)
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--L", type=int, default=4, help="number of RIS")
    p.add_argument("--K", type=int, default=8, help="UEs per AP")
    p.add_argument("--AP", type=int, default=5)
    p.add_argument("--D", type=int, default=6)
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--pmax_dbm", type=float, default=15.0)
    p.add_argument("--batch_size", type=int, default=8)

    p.add_argument("--n_iter", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--log_interval", type=int, default=250)
    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--test_sample_val", type=int, default=400)
    p.add_argument("--test_sample_final", type=int, default=0,
                   help="0 keeps the held-out test set untouched; use eval_variant.py instead")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", default="results_ris_action")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="500 iterations, gates only, no checkpointing")
    return p.parse_args()


def gate_report(model, trainer, K):
    """Functional gate: finite outputs, unit modulus, and a live backward pass."""
    uf, e, ui, ed, _ = trainer.dataloader.gen_training_data(
        K, trainer.training_associate_threshold, trainer.associate_threshold, duplicate=False)
    uf, e, ed = uf.to(model.device), e.to(model.device), ed.to(model.device)
    W, theta = model(uf, e, np.array(ui, dtype=bool), ed, training=True)
    loss, sum_rate, _ = trainer.dataloader.compute_loss(W, theta, trainer.pmax_w, trainer.device)
    loss.backward()
    gnorm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters()
                           if p.grad is not None)).item()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_params_t = sum(1 for _ in model.parameters())
    model.zero_grad(set_to_none=True)
    cen = {"W_finite": bool(torch.isfinite(W).all()), "theta_finite": bool(torch.isfinite(theta).all()),
           "unit_modulus_error": unit_modulus_error(theta), "sum_rate": float(sum_rate),
           "grad_norm": gnorm, "tensors_with_nonzero_grad": n_with_grad, "tensors_total": n_params_t}

    ufd, ed_, uid, edd = trainer.dataloader.gen_testing_data(
        K, trainer.associate_threshold, trainer.associate_threshold,
        duplicate=False, regenerate_channels=False)
    ufd = [t.to(model.device) for t in ufd]
    ed_ = [t.to(model.device) for t in ed_]
    edd = [t.to(model.device) for t in edd]
    with torch.no_grad():
        Wd, thd = model(ufd, ed_, uid, edd, training=False)
        _, sr_d, _ = trainer.dataloader.compute_loss(Wd, thd, trainer.pmax_w, trainer.device)
    dec = {"W_finite": bool(torch.isfinite(Wd).all()), "theta_finite": bool(torch.isfinite(thd).all()),
           "unit_modulus_error": unit_modulus_error(thd), "sum_rate": float(sr_d)}
    return {"centralized": cen, "decentralized": dec}


def main():
    args = parse_args()
    if args.smoke:
        args.n_iter = 500
        args.eval_interval = 500
        args.test_sample_val = 80
        args.save_every = 0

    # trainer_2 fixes the seed to 0 at import time; re-seed afterwards so the run
    # seed controls geometry, model init and channel draws together.
    import trainer_2
    from trainer_2 import Trainer

    import random
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    consensus = args.consensus or ("wreduce" if args.arch == "r0" else "equal")
    tag = args.tag or f"{args.arch}-{args.identity}-{consensus}"
    base = os.path.join(args.out_dir, f"{tag}_iter{args.n_iter}_seed{args.seed}")
    os.makedirs(os.path.join(base, "models"), exist_ok=True)
    os.makedirs(os.path.join(base, "arrays"), exist_ok=True)

    trainer = Trainer(args.M, args.N, args.L, args.K, args.batch_size,
                      args.pmax_dbm, device=args.device)
    device = trainer.device
    args.pmax_w = trainer.pmax_w
    model = VariantNet(args.M, args.N, args.L, args.D, trainer.pmax_w, args.ch, args.AP,
                       device, arch=args.arch, identity=args.identity,
                       consensus=consensus, tau=args.tau,
                       ris_loc=trainer.dataloader.RIS_Loc_array).to(device)
    trainer.model = model
    desc = model.describe()
    print(f"[cfg] {tag}: {json.dumps(desc)}")

    gates = gate_report(model, trainer, args.K)
    print(f"[gate] {json.dumps(gates)}")
    ok = (gates["centralized"]["W_finite"] and gates["centralized"]["theta_finite"]
          and gates["decentralized"]["W_finite"] and gates["decentralized"]["theta_finite"]
          and gates["centralized"]["unit_modulus_error"] < 1e-6
          and gates["decentralized"]["unit_modulus_error"] < 1e-6
          and gates["centralized"]["grad_norm"] > 0
          and np.isfinite(gates["centralized"]["grad_norm"]))
    if not ok:
        raise SystemExit(f"[gate] FAILED for {tag}")
    print("[gate] PASSED")

    start_iter, opt_state = 0, None
    if args.resume:
        bundle = load_checkpoint(model, args.resume, device)
        if isinstance(bundle, dict) and "model" in bundle:
            opt_state = bundle.get("optimizer")
            start_iter = int(bundle.get("iteration", 0))
        print(f"[resume] {args.resume} at iteration {start_iter}")

    trainer.opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if opt_state:
        trainer.opt.load_state_dict(opt_state)

    train_curve, val_curve = [], []
    best = {"decentralized": -float("inf"), "iteration": None}
    window, t0 = [], time.time()
    nan_iters = []

    for i in range(args.n_iter):
        loss, sum_rate, _ = trainer.train_batch()
        if not np.isfinite(loss):
            nan_iters.append(start_iter + i)
            if len(nan_iters) > 20:
                raise SystemExit(f"[abort] {tag}: {len(nan_iters)} non-finite losses")
        window.append(sum_rate)

        if (i + 1) % args.log_interval == 0:
            gnorm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                         for p in model.parameters() if p.grad is not None)))
            train_curve.append({"iteration": start_iter + i + 1,
                                "train_sum_rate": float(np.mean(window)), "grad_norm": gnorm})
            print(f"[train {tag} | {i+1}/{args.n_iter}] sum rate = {np.mean(window):.5f} "
                  f"grad_norm = {gnorm:.3e} ({(time.time()-t0)/(i+1)*1000:.0f} ms/it)", flush=True)
            window = []

        if (i + 1) % args.eval_interval == 0:
            (cen, cen_rand, dec, dec_rand, cen_disc, dec_disc,
             cen_rand_disc, dec_rand_disc) = trainer.eval(args.test_sample_val, 0, i)
            row = {"iteration": start_iter + i + 1, "centralized": float(cen),
                   "decentralized": float(dec), "centralized_discrete": float(cen_disc),
                   "decentralized_discrete": float(dec_disc),
                   "centralized_random_phase": float(cen_rand),
                   "decentralized_random_phase": float(dec_rand)}
            val_curve.append(row)
            print(f"[val {tag} | {i+1}] cen = {cen:.5f}  dec = {dec:.5f}  "
                  f"dec_2bit = {dec_disc:.5f}  dec_rand = {dec_rand:.5f}", flush=True)
            if dec > best["decentralized"]:
                best = dict(row)
                best["decentralized"] = float(dec)
                if not args.smoke:
                    torch.save(model.state_dict(), os.path.join(base, "models", "best_val_dec.pt"))

        if args.save_every and (i + 1) % args.save_every == 0:
            torch.save({"model": model.state_dict(), "optimizer": trainer.opt.state_dict(),
                        "iteration": start_iter + i + 1},
                       os.path.join(base, "models", "resumable_latest.pt"))

    final_gates = gate_report(model, trainer, args.K)
    summary = {"tag": tag, "config": {k: v for k, v in vars(args).items() if k != "pmax_w"},
               "model": desc, "initial_gates": gates, "final_gates": final_gates,
               "train_curve": train_curve, "val_curve": val_curve,
               "best_val": best, "nan_iterations": nan_iters,
               "wall_clock_s": time.time() - t0,
               "total_iterations": start_iter + args.n_iter}

    if args.test_sample_final:
        (cen, cen_rand, dec, dec_rand, cen_disc, dec_disc,
         cen_rand_disc, dec_rand_disc) = trainer.eval(args.test_sample_final, 0, args.n_iter)
        summary["final_eval"] = {"samples": args.test_sample_final, "centralized": float(cen),
                                 "decentralized": float(dec), "centralized_discrete": float(cen_disc),
                                 "decentralized_discrete": float(dec_disc),
                                 "centralized_random_phase": float(cen_rand),
                                 "decentralized_random_phase": float(dec_rand)}

    if not args.smoke:
        torch.save({"model": model.state_dict(), "optimizer": trainer.opt.state_dict(),
                    "iteration": start_iter + args.n_iter},
                   os.path.join(base, "models", "resumable_final.pt"))
        torch.save(model.state_dict(), os.path.join(base, "models", "model_final.pt"))
    with open(os.path.join(base, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[done] {tag}: best val decentralized = {best['decentralized']:.5f} "
          f"at iteration {best.get('iteration')}  -> {base}/summary.json", flush=True)


if __name__ == "__main__":
    main()
