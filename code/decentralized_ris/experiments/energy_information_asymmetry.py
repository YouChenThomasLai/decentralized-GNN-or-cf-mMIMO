"""Which arm sees more information: the learned pair scale, or the energy proxy?

`variants.VariantNet.decentralized` gives AP l the graph view
`vis = ui[:, l] & ui`, i.e. AP l sees link (l', k) for every user k that it and
AP l' both serve.  So the learned scale s_{l,r} is a function of other APs'
CSI as well as its own, while E_{l,r} is a function of AP l's own cascaded
channels and its own served-user set alone.

If that is right, corrupting every *other* AP's channels must leave E bit-
identical while moving s.  That settles the leakage question in the direction
that matters: the proxy cannot be winning because it sees more, because it
demonstrably sees less.

    python -m experiments.energy_information_asymmetry --batches 5
"""

import argparse
import json
import os

import numpy as np
import torch

import variants
from evaluate import build_model, resolve_device, seed_everything, temporary_seed
from experiments.mrc_proxy_diagnostic import pair_energy
from model import load_checkpoint
from simulation import ChannelSimulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default="../../artifacts/decentralized_ris/e01_baseline_training/iter500000/"
                "M2_N30_L4_K8_P15.0_iter350000_seed0/run0")
    parser.add_argument("--checkpoint", default="resumable_final.pt")
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--eval_seed", type=int, default=20260922)
    parser.add_argument("--corrupt_seed", type=int, default=4242)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out",
        default="../../artifacts/decentralized_ris/e05_energy_consensus/verification/"
                "information_asymmetry.json")
    args = parser.parse_args()

    device = resolve_device(args.device)
    with open(os.path.join(args.run, "summary.json"), encoding="utf-8") as handle:
        config = json.load(handle)["config"]
    seed_everything(config["seed"])
    simulator = ChannelSimulator(
        config["M"], config["N"], config["L"], config["batch_size"],
        n_ap=config["AP"])
    cfg = dict(config)
    cfg["arch"], cfg["consensus"] = "r1_ap_ris_mag", "ap_ris_mag"
    net = build_model(cfg, simulator, device)
    load_checkpoint(net, os.path.join(args.run, "models", args.checkpoint), device)
    net.eval()

    rng = np.random.default_rng(args.corrupt_seed)
    threshold = config.get("assoc_threshold", 0.1)
    rows = []

    with torch.no_grad(), temporary_seed(args.eval_seed):
        for index in range(args.batches):
            simulator.training_batch(config["K"], threshold, threshold)
            clean = simulator.decentralized_batch(
                config["K"], threshold, threshold, regenerate_channels=False)
            df, de, dm, dd = clean
            trace = {}
            net.decentralized([t.to(device) for t in df], [t.to(device) for t in de],
                              dm, [t.to(device) for t in dd], trace)
            s_clean = trace["z_pairs"].norm(dim=-1).mean(dim=3)          # (B, A, R)
            e_clean, _, _ = pair_energy(de, dm, device)

            # Corrupt AP 0's channels; AP 0 is then the "other AP" for 1..A-1.
            saved = simulator.base_stations[0].channels.copy()
            noise = rng.normal(size=saved.shape) + 1j * rng.normal(size=saved.shape)
            simulator.base_stations[0].channels = saved * (1.0 + 5.0 * noise)
            df2, de2, dm2, dd2 = simulator.decentralized_batch(
                config["K"], threshold, threshold, regenerate_channels=False)
            trace2 = {}
            net.decentralized([t.to(device) for t in df2], [t.to(device) for t in de2],
                              dm2, [t.to(device) for t in dd2], trace2)
            s_dirty = trace2["z_pairs"].norm(dim=-1).mean(dim=3)
            e_dirty, _, _ = pair_energy(de2, dm2, device)
            simulator.base_stations[0].channels = saved

            # Look only at the APs whose own channels were NOT touched.
            def relative(a, b):
                scale = b[:, 1:, :].abs().max().clamp(min=1e-30)
                return float((a[:, 1:, :] - b[:, 1:, :]).abs().max() / scale)

            rows.append({
                "batch": index,
                "rel_change_in_learned_s_of_untouched_aps": relative(s_dirty, s_clean),
                "rel_change_in_energy_of_untouched_aps": relative(e_dirty, e_clean),
                "abs_change_in_energy_of_untouched_aps": float(
                    (e_dirty[:, 1:, :] - e_clean[:, 1:, :]).abs().max()),
            })
            print(f"  batch {index}: learned s moved "
                  f"{rows[-1]['rel_change_in_learned_s_of_untouched_aps']:.3e}, "
                  f"E moved {rows[-1]['rel_change_in_energy_of_untouched_aps']:.3e}")

    payload = {
        "description": (
            "AP 0's cascaded channels are corrupted; the quantities of APs 1..4 "
            "are then compared against their clean values.  E depends only on an "
            "AP's own channels and served-user set, so it must not move at all; "
            "s depends on the eq. (10) shared-user view, so it does move."
        ),
        "per_batch": rows,
        "max_rel_change_in_learned_s": max(
            r["rel_change_in_learned_s_of_untouched_aps"] for r in rows),
        "max_abs_change_in_energy": max(
            r["abs_change_in_energy_of_untouched_aps"] for r in rows),
        "conclusion": (
            "the learned pair scale reads other APs' CSI; the energy proxy does not"
        ),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nlearned s max relative change: {payload['max_rel_change_in_learned_s']:.3e}")
    print(f"energy   max absolute change: {payload['max_abs_change_in_energy']:.3e}")
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
