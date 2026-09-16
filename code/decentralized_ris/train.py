"""Single-seed training entry point for the baseline and RIS-action variants."""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import variants
from evaluate import build_model, evaluate_model, resolve_device, seed_everything, temporary_seed
from model import load_checkpoint
from simulation import ChannelSimulator
from variants import unit_modulus_error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="r0", choices=variants.ARCHS)
    parser.add_argument("--identity", default="none", choices=variants.IDENTITIES)
    parser.add_argument("--consensus", default=None, choices=list(variants.CONSENSUS))
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--tag", default=None)

    parser.add_argument("--M", type=int, default=2, help="antennas per AP")
    parser.add_argument("--N", type=int, default=30, help="elements per RIS")
    parser.add_argument("--L", type=int, default=4, help="number of RISs")
    parser.add_argument("--K", type=int, default=8, help="users per AP")
    parser.add_argument("--AP", type=int, default=5, help="number of APs")
    parser.add_argument("--D", type=int, default=6, help="message-passing depth")
    parser.add_argument("--ch", type=int, default=64, help="hidden width")
    parser.add_argument("--pmax_dbm", "--Pmax", dest="pmax_dbm", type=float, default=15.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--assoc_threshold", type=float, default=0.1)

    parser.add_argument("--n_iter", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--log_interval", type=int, default=250)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--test_sample_val", type=int, default=400)
    parser.add_argument("--test_sample_final", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_seed", type=int, default=20260913)
    parser.add_argument("--eval_seed", type=int, default=20260914)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="../../artifacts/decentralized_ris/runs")
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--resume")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def gate_report(model, simulator, users_per_ap, threshold, device):
    features, edges, masks, direct, _ = simulator.training_batch(
        users_per_ap, threshold, threshold
    )
    features, edges, direct = features.to(device), edges.to(device), direct.to(device)
    beamformer, phase = model.centralized(features, edges, masks, direct)
    loss, rate, _ = simulator.loss(beamformer, phase, device)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradient_norm = torch.sqrt(sum((gradient.detach() ** 2).sum() for gradient in gradients))
    central = {
        "beamformer_finite": bool(torch.isfinite(beamformer).all()),
        "phase_finite": bool(torch.isfinite(phase).all()),
        "unit_modulus_error": unit_modulus_error(phase),
        "sum_rate": float(rate.detach()),
        "gradient_norm": float(gradient_norm),
    }
    model.zero_grad(set_to_none=True)

    features, edges, masks, direct = simulator.decentralized_batch(
        users_per_ap, threshold, threshold, regenerate_channels=False
    )
    features = [tensor.to(device) for tensor in features]
    edges = [tensor.to(device) for tensor in edges]
    direct = [tensor.to(device) for tensor in direct]
    with torch.no_grad():
        beamformer, phase = model.decentralized(features, edges, masks, direct)
        _, rate, _ = simulator.loss(beamformer, phase, device)
    decentralized = {
        "beamformer_finite": bool(torch.isfinite(beamformer).all()),
        "phase_finite": bool(torch.isfinite(phase).all()),
        "unit_modulus_error": unit_modulus_error(phase),
        "sum_rate": float(rate.detach()),
    }
    return {"centralized": central, "decentralized": decentralized}


def gate_passed(report):
    central = report["centralized"]
    decentralized = report["decentralized"]
    return (
        central["beamformer_finite"]
        and central["phase_finite"]
        and decentralized["beamformer_finite"]
        and decentralized["phase_finite"]
        and central["unit_modulus_error"] < 1e-6
        and decentralized["unit_modulus_error"] < 1e-6
        and np.isfinite(central["gradient_norm"])
        and central["gradient_norm"] > 0
    )


def save_checkpoint(path, model, optimizer, iteration, config):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "config": config,
        },
        path,
    )


def save_metrics(path, train_curve, validation_curve):
    payload = {
        "train_iteration": np.asarray([row["iteration"] for row in train_curve]),
        "train_sum_rate": np.asarray([row["sum_rate"] for row in train_curve]),
        "train_gradient_norm": np.asarray([row["gradient_norm"] for row in train_curve]),
        "validation_iteration": np.asarray(
            [row["iteration"] for row in validation_curve]
        ),
    }
    for key in (
        "centralized",
        "centralized_discrete",
        "centralized_random_phase",
        "decentralized",
        "decentralized_discrete",
        "decentralized_random_phase",
    ):
        payload[f"validation_{key}"] = np.asarray(
            [row[key] for row in validation_curve]
        )
    np.savez(path, **payload)


def main():
    args = parse_args()
    if args.smoke:
        args.n_iter = 500
        args.eval_interval = 500
        args.test_sample_val = 80
        args.save_every = 0

    args.consensus = args.consensus or variants.default_consensus(args.arch)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    simulator = ChannelSimulator(
        args.M, args.N, args.L, args.batch_size, n_ap=args.AP
    )
    config = vars(args).copy()
    model = build_model(config, simulator, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_iteration = 0
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            if checkpoint.get("optimizer"):
                optimizer.load_state_dict(checkpoint["optimizer"])
            start_iteration = int(checkpoint.get("iteration", 0))

    tag = args.tag or (
        f"{args.arch}-{args.identity}-{args.consensus}_"
        f"M{args.M}_N{args.N}_L{args.L}_K{args.K}_P{args.pmax_dbm}"
    )
    run_dir = os.path.join(args.out_dir, f"{tag}_iter{args.n_iter}_seed{args.seed}")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(run_dir, "tensorboard")) if args.tensorboard else None

    gates = gate_report(
        model, simulator, args.K, args.assoc_threshold, device
    )
    print(f"[gate] {json.dumps(gates)}")
    if not gate_passed(gates):
        raise SystemExit("functional gate failed")

    train_curve = []
    validation_curve = []
    best = {"decentralized": -float("inf"), "iteration": None}
    window = []
    started = time.time()

    for step in range(1, args.n_iter + 1):
        model.train()
        features, edges, masks, direct, _ = simulator.training_batch(
            args.K, args.assoc_threshold, args.assoc_threshold
        )
        features, edges, direct = features.to(device), edges.to(device), direct.to(device)
        optimizer.zero_grad(set_to_none=True)
        beamformer, phase = model.centralized(features, edges, masks, direct)
        loss, rate, _ = simulator.loss(beamformer, phase, device)
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at iteration {start_iteration + step}")
        loss.backward()
        gradient_norm = torch.sqrt(
            sum(
                (parameter.grad.detach() ** 2).sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )
        optimizer.step()
        window.append(float(rate.detach()))
        iteration = start_iteration + step
        if writer:
            writer.add_scalar("train/loss", float(loss.detach()), iteration)
            writer.add_scalar("train/sum_rate", float(rate.detach()), iteration)

        if step % args.log_interval == 0 or step == args.n_iter:
            row = {
                "iteration": iteration,
                "sum_rate": float(np.mean(window)),
                "gradient_norm": float(gradient_norm),
            }
            train_curve.append(row)
            print(
                f"[train {tag} {step}/{args.n_iter}] rate={row['sum_rate']:.5f} "
                f"grad={row['gradient_norm']:.3e}"
            )
            window = []

        if step % args.eval_interval == 0 or step == args.n_iter:
            with temporary_seed(args.val_seed):
                batches, _, _ = evaluate_model(
                    model,
                    simulator,
                    args.K,
                    args.assoc_threshold,
                    device,
                    args.test_sample_val,
                    args.batch_size,
                )
            row = {"iteration": iteration}
            row.update({key: float(values.mean()) for key, values in batches.items()})
            validation_curve.append(row)
            print(
                f"[val {tag} {iteration}] cen={row['centralized']:.5f} "
                f"dec={row['decentralized']:.5f} "
                f"dec_2bit={row['decentralized_discrete']:.5f}"
            )
            if writer:
                for key, value in row.items():
                    if key != "iteration":
                        writer.add_scalar(f"validation/{key}", value, iteration)
            if row["decentralized"] > best["decentralized"]:
                best = dict(row)
                save_checkpoint(
                    os.path.join(checkpoint_dir, "best.pt"),
                    model,
                    optimizer,
                    iteration,
                    config,
                )

        if args.save_every and step % args.save_every == 0:
            save_checkpoint(
                os.path.join(checkpoint_dir, "last.pt"),
                model,
                optimizer,
                iteration,
                config,
            )

    final_iteration = start_iteration + args.n_iter
    save_checkpoint(
        os.path.join(checkpoint_dir, "last.pt"),
        model,
        optimizer,
        final_iteration,
        config,
    )
    save_metrics(os.path.join(run_dir, "metrics.npz"), train_curve, validation_curve)
    summary = {
        "tag": tag,
        "config": config,
        "model": model.describe(),
        "initial_gates": gates,
        "final_gates": gate_report(
            model, simulator, args.K, args.assoc_threshold, device
        ),
        "best_val": best,
        "total_iterations": final_iteration,
        "wall_clock_s": time.time() - started,
    }
    if args.test_sample_final:
        with temporary_seed(args.eval_seed):
            batches, unit_error, per_user = evaluate_model(
                model,
                simulator,
                args.K,
                args.assoc_threshold,
                device,
                args.test_sample_final,
                args.batch_size,
            )
        summary["final_eval"] = {
            "samples": args.test_sample_final,
            "unit_modulus_error": unit_error,
            "per_user_rate": per_user.tolist(),
            **{key: float(values.mean()) for key, values in batches.items()},
        }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if writer:
        writer.close()
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()
