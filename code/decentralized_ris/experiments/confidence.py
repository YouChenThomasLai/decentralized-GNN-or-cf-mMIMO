"""Does AP confidence carry information, or is the gain an artifact?

Stage 1.4 screening showed a large decentralized sum-rate gain from replacing
equal circular consensus with the AP-confidence softmax. Before that gain can be
attributed to "APs estimating their own proposal reliability", it has to survive
four checks, which is what this script measures on a trained checkpoint:

1. Counterfactual consensus: keep the model's own proposals, force equal
   weights. The rate lost is the part of the gain that is due to the consensus
   rule rather than to the representation that produced the proposals.
2. Weight concentration: entropy over the AP axis and the effective AP count
   exp(H). Collapse to a single AP is a valid mechanism but a different claim.
3. Proposal agreement: the circular concentration |sum_l a_l theta_hat_l|.
4. Leave-one-AP-out: the true marginal sum-rate contribution of each AP,
   rank-correlated against that AP's confidence. This is the only check that can
   support calling the scalar a reliability estimate.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--samples", type=int, default=320)
    p.add_argument("--eval_seed", type=int, default=20260914)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="results_ris_action/confidence_diagnostics.json")
    args = p.parse_args()

    from simulation import ChannelSimulator
    from evaluate import checkpoint_path, resolve_device, seed_everything
    from model import load_checkpoint
    from variants import VariantNet, circular_consensus

    report = {}
    for run in sorted({d for pat in args.runs for d in glob.glob(pat) if os.path.isdir(d)}):
        summary = json.load(open(os.path.join(run, "summary.json")))
        cfg = summary["config"]
        ckpt = checkpoint_path(run, args.checkpoint)
        if ckpt is None:
            print(f"[skip] {run}")
            continue

        seed_everything(cfg["seed"])
        dl = ChannelSimulator(
            cfg["M"], cfg["N"], cfg["L"], cfg["batch_size"], n_ap=cfg["AP"]
        )
        pmax_w = 10 ** ((cfg["pmax_dbm"] - 30) / 10)
        device = resolve_device(args.device)
        consensus = cfg["consensus"] or ("wreduce" if cfg["arch"] == "r0" else "equal")
        model = VariantNet(cfg["M"], cfg["N"], cfg["L"], cfg["D"], pmax_w, cfg["ch"], cfg["AP"],
                           device, arch=cfg["arch"], identity=cfg["identity"],
                           consensus=consensus, tau=cfg["tau"], ris_loc=dl.ris_locations,
                           users_per_ap=cfg["K"]).to(device)
        load_checkpoint(model, ckpt, device)
        model.eval()

        seed_everything(args.eval_seed)
        n_ap = cfg["AP"]
        acc = {"rate": [], "rate_equal": [], "entropy": [], "eff_ap": [],
               "rho": [], "max_weight": []}
        loo_gain, loo_conf = [], []

        with torch.no_grad():
            for _ in range(args.samples // cfg["batch_size"]):
                dl.training_batch(cfg["K"], 0.1, 0.1)
                ufd, e_, uid, edd = dl.decentralized_batch(
                    cfg["K"], 0.1, 0.1, regenerate_channels=False
                )
                ufd = [t.to(device) for t in ufd]
                e_ = [t.to(device) for t in e_]
                edd = [t.to(device) for t in edd]

                tr = {}
                W, th = model.decentralized(ufd, e_, uid, edd, trace=tr)
                _, sr, _ = dl.loss(W, th, device)
                acc["rate"].append(float(sr))

                w = tr["weights"]                                   # (B, A, R)
                act = tr["active"]
                ent = -(w.clamp(min=1e-12).log() * w).sum(dim=1)    # (B, R)
                acc["entropy"].append(float(ent.mean()))
                acc["eff_ap"].append(float(ent.exp().mean()))
                acc["max_weight"].append(float(w.max(dim=1).values.mean()))
                res = (tr["proposals"] * w[..., None, None]).sum(dim=1)
                acc["rho"].append(float(res.norm(dim=-1).mean()))

                if "theta_equal" in tr:
                    _, sr_e, _ = dl.loss(W, tr["theta_equal"], device)
                    acc["rate_equal"].append(float(sr_e))

                # Leave-one-AP-out: drop AP l's vote only (its beamformer stays),
                # so the change isolates that AP's contribution to the phase.
                if len(loo_gain) < 40:
                    for l in range(n_ap):
                        keep = act.clone()
                        keep[:, l] = 0.0
                        if float(keep.sum(dim=1).min()) == 0:
                            continue
                        th_l, _ = circular_consensus(tr["proposals"], keep,
                                                     tr["logits"], model.tau)
                        _, sr_l, _ = dl.loss(W, th_l, device)
                        loo_gain.append(float(sr) - float(sr_l))
                        loo_conf.append(float((w[:, l, :] * act[:, l:l + 1]).mean()))

        out = {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}
        out["n_ap"] = n_ap
        out["consensus"] = consensus
        if acc["rate_equal"]:
            out["consensus_rule_gain"] = out["rate"] - out["rate_equal"]
        if loo_gain:
            out["loo_confidence_rank_corr"] = spearman(np.array(loo_conf), np.array(loo_gain))
            out["loo_mean_gain"] = float(np.mean(loo_gain))
        report[summary["tag"]] = out
        print(f"[diag] {summary['tag']:28s} rate={out['rate']:.4f} "
              f"eff_AP={out['eff_ap']:.3f}/{n_ap} max_w={out['max_weight']:.3f} "
              f"rho={out['rho']:.4f}"
              + (f" forced-equal={out['rate_equal']:.4f} (rule gain {out['consensus_rule_gain']:+.4f})"
                 if acc["rate_equal"] else "")
              + (f" LOO_corr={out['loo_confidence_rank_corr']:+.3f}" if loo_gain else ""), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
