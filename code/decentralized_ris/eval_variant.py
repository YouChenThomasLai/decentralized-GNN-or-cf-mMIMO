"""Paired evaluation of RIS action variants on one fixed channel set.

Stage 1.0 requires every candidate to be compared on identical data. Channel
generation is pure numpy and the call sequence is the same for every
architecture, so re-seeding immediately before the evaluation loop makes all
models see byte-identical channels. The AP/RIS geometry and the AP-RIS LOS
components are drawn when `ChannelSimulator` is constructed, so the dataloader is
rebuilt under the *training* seed (same deployment the models were trained for)
and only the test channel draws use `--eval_seed`.

Per-batch sum rates are stored so paired differences, paired t-tests and sign
tests can be recomputed without re-running the models.
"""

import argparse
import glob
import json
import os
import random

import numpy as np
import torch


def deterministic_state(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, K, threshold, pmax_w, device, samples, batch_size,
             eval_seed, num_bits=2):
    """Return per-batch sum rates for every phase mode on one fixed channel set."""
    from rates import quantize_phase, random_phase_like

    deterministic_state(eval_seed)
    model.eval()
    keys = ["centralized", "centralized_discrete", "centralized_random_phase",
            "decentralized", "decentralized_discrete", "decentralized_random_phase"]
    out = {k: [] for k in keys}
    um = {"centralized": 0.0, "decentralized": 0.0}
    per_user = []

    with torch.no_grad():
        for _ in range(samples // batch_size):
            uf, e, ui, ed, _ = dataloader.training_batch(K, threshold, threshold)
            uf, e, ed = uf.to(device), e.to(device), ed.to(device)
            W, th = model(uf, e, np.array(ui, dtype=bool), ed, training=True)
            um["centralized"] = max(um["centralized"], float((th.norm(dim=-1) - 1).abs().max()))
            _, sr, _ = dataloader.loss(W, th, device)
            out["centralized"].append(float(sr))
            _, sr, _ = dataloader.loss(W, quantize_phase(th, num_bits), device)
            out["centralized_discrete"].append(float(sr))
            _, sr, _ = dataloader.loss(W, random_phase_like(th), device)
            out["centralized_random_phase"].append(float(sr))

            ufd, ed_, uid, edd = dataloader.decentralized_batch(
                K, threshold, threshold, regenerate_channels=False)
            ufd = [t.to(device) for t in ufd]
            ed_ = [t.to(device) for t in ed_]
            edd = [t.to(device) for t in edd]
            Wd, thd = model(ufd, ed_, uid, edd, training=False)
            um["decentralized"] = max(um["decentralized"], float((thd.norm(dim=-1) - 1).abs().max()))
            _, sr, rate = dataloader.loss(Wd, thd, device)
            out["decentralized"].append(float(sr))
            per_user.append(rate.detach().cpu().numpy())
            _, sr, _ = dataloader.loss(Wd, quantize_phase(thd, num_bits), device)
            out["decentralized_discrete"].append(float(sr))
            _, sr, _ = dataloader.loss(Wd, random_phase_like(thd), device)
            out["decentralized_random_phase"].append(float(sr))

    return ({k: np.array(v) for k, v in out.items()}, um,
            np.mean(np.stack(per_user), axis=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="run directories written by train_variant.py (globs allowed)")
    p.add_argument("--checkpoint", default="best_val_dec.pt",
                   choices=["best_val_dec.pt", "model_final.pt"])
    p.add_argument("--samples", type=int, default=3200)
    p.add_argument("--eval_seed", type=int, default=20260914)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="results_ris_action/final_screening.json")
    args = p.parse_args()

    run_dirs = sorted({d for pat in args.runs for d in glob.glob(pat) if os.path.isdir(d)})
    if not run_dirs:
        raise SystemExit(f"no run directories matched {args.runs}")

    import trainer_2
    from simulation import ChannelSimulator
    from model import load_checkpoint
    from variants import VariantNet

    results = {}
    arrays = {}
    for run in run_dirs:
        with open(os.path.join(run, "summary.json")) as fh:
            summary = json.load(fh)
        cfg = summary["config"]
        ckpt = os.path.join(run, "models", args.checkpoint)
        if not os.path.exists(ckpt):
            print(f"[skip] {run}: no {args.checkpoint}")
            continue

        # Rebuild the seed-`cfg['seed']` deployment the model was trained on.
        deterministic_state(cfg["seed"])
        dataloader = ChannelSimulator(
            cfg["M"], cfg["N"], cfg["L"], cfg["batch_size"], n_ap=cfg["AP"]
        )
        pmax_w = 10 ** ((cfg["pmax_dbm"] - 30) / 10)
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        consensus = cfg["consensus"] or ("wreduce" if cfg["arch"] == "r0" else "equal")
        model = VariantNet(cfg["M"], cfg["N"], cfg["L"], cfg["D"], pmax_w, cfg["ch"],
                           cfg["AP"], device, arch=cfg["arch"], identity=cfg["identity"],
                           consensus=consensus, tau=cfg["tau"],
                           ris_loc=dataloader.ris_locations,
                           users_per_ap=cfg["K"]).to(device)
        load_checkpoint(model, ckpt, device)

        batches, um, per_user = evaluate(model, dataloader, cfg["K"], 0.1, pmax_w, device,
                                         args.samples, cfg["batch_size"], args.eval_seed)
        tag = summary["tag"]
        results[tag] = {"run_dir": run, "checkpoint": args.checkpoint,
                        "samples": args.samples, "eval_seed": args.eval_seed,
                        "unit_modulus_error": um, "model": summary["model"],
                        "best_val_iteration": summary["best_val"].get("iteration"),
                        "per_user_rate": per_user.tolist(),
                        **{k: float(v.mean()) for k, v in batches.items()},
                        **{f"{k}_sem": float(v.std(ddof=1) / np.sqrt(len(v)))
                           for k, v in batches.items()}}
        arrays[tag] = batches
        print(f"[eval] {tag:28s} cen = {batches['centralized'].mean():.5f}  "
              f"dec = {batches['decentralized'].mean():.5f}  "
              f"dec_2bit = {batches['decentralized_discrete'].mean():.5f}  "
              f"um = {max(um.values()):.2e}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    npz = os.path.splitext(args.out)[0] + "_paired.npz"
    np.savez(npz, **{f"{tag}__{k}": v for tag, b in arrays.items() for k, v in b.items()})
    print(f"[done] {args.out}  and  {npz}")


if __name__ == "__main__":
    main()
