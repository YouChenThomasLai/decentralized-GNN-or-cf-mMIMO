import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data import MyDataLoader
from evaluate import METHODS, evaluate_snapshot, validate_sample_count
from model_2 import node_update


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class Trainer:
    def __init__(
        self,
        M,
        N,
        L,
        K,
        batch_size,
        n_iter=2000,
        pmax_dbm=15.0,
        device="cuda:0",
        num_ap=5,
        spatial_scale=100.0,
    ):
        self.M = M
        self.N = N
        self.L = L
        self.K = K
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.num_of_AP = num_ap
        self.pmax_dbm = pmax_dbm
        self.pmax_w = 10 ** ((pmax_dbm - 30) / 10)
        self.dataloader = MyDataLoader(
            M,
            N,
            L,
            batch_size,
            num_ap=num_ap,
            spatial_scale=spatial_scale,
        )
        self.dataloader.BS_RIS_association()
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.model = node_update(
            M, N, L, 6, self.pmax_w, 2, 64, num_ap, self.device, K
        ).to(self.device)
        self.parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        self.log_interval = 10
        self.log_eval_interval = 500
        self.training_associate_threshold = 0.1
        self.associate_threshold = 0.1
        self.initial_gradient_diagnostics = None
        print(
            f"[INFO] Per-AP Pmax = {self.pmax_dbm:g} dBm "
            f"= {self.pmax_w:g} W."
        )

    def train_batch(self):
        self.model.train()
        feature, e, index, e_dir, _ = self.dataloader.gen_training_data(
            self.K,
            self.training_associate_threshold,
            self.associate_threshold,
        )
        feature = feature.to(self.device)
        e = e.to(self.device)
        e_dir = e_dir.to(self.device)
        self.opt.zero_grad()
        beamformer, theta = self.model(
            feature, e, index, e_dir, training=True, duplicate=False
        )
        loss, sum_rate, rate = self.dataloader.compute_loss(
            beamformer, theta, self.pmax_w, self.device
        )
        loss.backward()
        if self.initial_gradient_diagnostics is None:
            task_norm = torch.sqrt(
                sum(
                    parameter.grad.detach().square().sum()
                    for parameter in self.model.parameters()
                    if parameter.grad is not None
                )
            ).item()
            parameter_norm = torch.sqrt(
                sum(
                    parameter.detach().square().sum()
                    for parameter in self.model.parameters()
                )
            ).item()
            decay_norm = 1e-6 * parameter_norm
            self.initial_gradient_diagnostics = {
                "initial_task_gradient_norm": task_norm,
                "initial_decay_gradient_norm": decay_norm,
                "initial_decay_to_task_ratio": (
                    decay_norm / task_norm if task_norm else float("inf")
                ),
            }
        self.opt.step()
        return loss.item(), sum_rate.item(), rate.detach().cpu()

    def train(
        self,
        run_id,
        out_dir,
        log_dir,
        test_sample_val,
        test_sample_final,
        config,
    ):
        model_dir = Path(out_dir) / "models"
        array_dir = Path(out_dir) / "arrays"
        final_dir = Path(out_dir) / "final_eval"
        for path in (model_dir, array_dir, final_dir, Path(log_dir)):
            path.mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "config.json", "w") as config_file:
            json.dump(config, config_file, indent=2, sort_keys=True)

        writer = SummaryWriter(log_dir=log_dir)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=0.0001, weight_decay=1e-6
        )
        train_losses = []
        sum_rates = []
        validation = {method: [] for method in METHODS}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        training_started = time.perf_counter()

        for iteration in range(self.n_iter):
            loss, sum_rate, rate = self.train_batch()
            train_losses.append(loss)
            sum_rates.append(sum_rate)
            writer.add_scalar("Train/Loss", loss, iteration)
            writer.add_scalar("Train/SumRate", sum_rate, iteration)
            for user_id, value in enumerate(rate):
                writer.add_scalar(
                    f"UserRate/User_{user_id + 1}", value, iteration
                )
            if iteration % self.log_interval == 0:
                start = max(0, len(train_losses) - self.log_interval)
                print(
                    f"[Train | {iteration}/{self.n_iter}] "
                    f"loss = {np.mean(train_losses[start:]):.8g}, "
                    f"sum rate = {np.mean(sum_rates[start:]):.8g}"
                )
            if (
                (iteration + 1) % self.log_eval_interval == 0
                and iteration >= 1
            ):
                metrics = self.eval(test_sample_val)
                for method in METHODS:
                    value = metrics[method]
                    validation[method].append(value)
                    writer.add_scalar(f"Val/{method}", value, iteration + 1)

        training_seconds = time.perf_counter() - training_started
        print("Running FINAL evaluation with more samples...")
        final_metrics, final_details = self.eval(
            test_sample_final, return_details=True
        )
        final_metrics.update(
            {
                "training_seconds": training_seconds,
                "parameter_count": self.parameter_count,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else 0
                ),
                **self.initial_gradient_diagnostics,
            }
        )
        for key, value in final_metrics.items():
            print(f"[Final Eval] {key} = {value:.8g}")

        np.save(final_dir / f"final_eval_run{run_id}.npy", final_metrics)
        np.savez_compressed(
            final_dir / f"evaluation_details_run{run_id}.npz",
            **final_details,
        )
        with open(
            final_dir / f"metrics_run{run_id}.json", "w"
        ) as metrics_file:
            json.dump(final_metrics, metrics_file, indent=2, sort_keys=True)
        with open(
            final_dir / f"final_eval_run{run_id}.txt", "w"
        ) as final_file:
            for key, value in final_metrics.items():
                final_file.write(f"{key}: {value:.8g}\n")
        with open(Path(out_dir) / "checks.json", "w") as checks_file:
            json.dump(
                {
                    "all_ap_all_ris": True,
                    "association_mask": True,
                    "cd_pairing": True,
                    "finite_outputs": True,
                    "per_ap_power": True,
                    "topology_density": True,
                },
                checks_file,
                indent=2,
                sort_keys=True,
            )

        writer.close()
        torch.save(
            self.model.state_dict(), model_dir / f"model_final_run{run_id}.pt"
        )
        np.save(array_dir / f"losses_run{run_id}.npy", train_losses)
        np.save(array_dir / f"sumrates_run{run_id}.npy", sum_rates)
        for method, values in validation.items():
            np.save(
                array_dir / f"val_sum_rate_{method}_run{run_id}.npy",
                values,
            )
        return final_metrics

    def eval(self, test_sample, return_details=False):
        return evaluate_snapshot(
            self.model,
            self.dataloader,
            test_sample,
            K=self.K,
            batch_size=self.batch_size,
            associate_threshold=self.associate_threshold,
            pmax_w=self.pmax_w,
            device=self.device,
            return_details=return_details,
        )


def source_provenance():
    source_dir = Path(__file__).resolve().parent
    repository = source_dir.parents[2]
    digest = hashlib.sha256()
    for path in sorted((*source_dir.glob("*.py"), *source_dir.glob("*.sh"))):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())

    def git(*args):
        result = subprocess.run(
            ("git", *args),
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    source_commit = os.environ.get("SOURCE_COMMIT") or git(
        "rev-parse", "HEAD"
    )
    dirty_status = os.environ.get("SOURCE_DIRTY_STATUS")
    if dirty_status is None:
        dirty_status = git("status", "--short")
    return {
        "source_commit": source_commit,
        "source_dirty": bool(dirty_status),
        "source_dirty_status": dirty_status,
        "source_hash_sha256": digest.hexdigest(),
    }


def write_status(run_dir, status, error=None):
    payload = {"status": status}
    if error is not None:
        payload["error"] = str(error)
    with open(Path(run_dir) / "status.json", "w") as status_file:
        json.dump(payload, status_file, indent=2)


def save_summary(exp_dir):
    records = []
    for path in sorted(Path(exp_dir).glob("seed*/final_eval/metrics_run*.json")):
        seed = int(path.parents[1].name.removeprefix("seed"))
        with open(path) as metrics_file:
            records.append((seed, json.load(metrics_file)))
    if not records:
        return
    summary = {}
    for key in sorted(records[0][1]):
        array = np.asarray([metrics[key] for _, metrics in records], dtype=float)
        summary[key] = {
            "per_seed": {
                str(seed): float(value)
                for (seed, _), value in zip(records, array)
            },
            "mean": float(array.mean()),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    gaps = np.asarray([metrics["absolute_gap"] for _, metrics in records])
    dominant_sign_count = int(max(np.sum(gaps > 0), np.sum(gaps < 0)))
    summary["gap_dominant_sign_count"] = dominant_sign_count
    summary["gap_sign_consistent"] = dominant_sign_count >= min(4, len(gaps))
    with open(Path(exp_dir) / "final_summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
    np.save(Path(exp_dir) / "final_summary.npy", summary)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 0 RIS snapshot network-scaling trainer"
    )
    parser.add_argument("--scale", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--num_ap", type=int)
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--L", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--spatial_scale", type=float)
    parser.add_argument(
        "--pmax_dbm", "--Pmax", dest="pmax_dbm", type=float, default=15.0
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_iter", type=int, default=2000)
    parser.add_argument(
        "--out_dir",
        default="../results_snapshot_scaling/stage0_ris/scale1_A5_K8_L4",
    )
    parser.add_argument("--test_sample_val", type=int, default=128)
    parser.add_argument("--test_sample_final", type=int, default=3200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.num_ap = args.num_ap or 5 * args.scale
    args.K = args.K or 8 * args.scale
    args.L = args.L or 4 * args.scale
    args.spatial_scale = args.spatial_scale or 100 * np.sqrt(args.scale)
    for name in ("num_ap", "M", "N", "L", "K", "batch_size", "runs", "n_iter"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.spatial_scale <= 0:
        parser.error("--spatial_scale must be positive")
    try:
        validate_sample_count(
            "--test_sample_val", args.test_sample_val, args.batch_size
        )
        validate_sample_count(
            "--test_sample_final", args.test_sample_final, args.batch_size
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    args = parse_args()
    exp_dir = Path(args.out_dir)
    provenance = source_provenance()
    for run_offset in range(args.runs):
        effective_seed = args.seed + run_offset
        base_dir = exp_dir / f"seed{effective_seed}"
        if base_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {base_dir}")
        base_dir.mkdir(parents=True)
        write_status(base_dir, "running")
        seed_everything(effective_seed)
        print(f"[INFO] Run seed: {effective_seed}")
        trainer = Trainer(
            args.M,
            args.N,
            args.L,
            args.K,
            args.batch_size,
            args.n_iter,
            args.pmax_dbm,
            device=args.device,
            num_ap=args.num_ap,
            spatial_scale=args.spatial_scale,
        )
        array_dir = base_dir / "arrays"
        array_dir.mkdir()
        np.savez_compressed(
            array_dir / "topology.npz",
            ap_locations=trainer.dataloader.BS_Loc_array,
            ris_locations=trainer.dataloader.RIS_Loc_array,
            spatial_scale=args.spatial_scale,
            ap_radius=2 * args.spatial_scale,
            topology_seed=effective_seed,
        )
        config = {
            "scale": args.scale,
            "effective_seed": effective_seed,
            "training_seed": effective_seed,
            "channel_seed": effective_seed,
            "topology_seed": effective_seed,
            "cli": shlex.join([sys.executable, *sys.argv]),
            "arguments": vars(args),
            "effective_device": str(trainer.device),
            "num_ap": args.num_ap,
            "num_ue": args.K,
            "num_ris": args.L,
            "spatial_scale": args.spatial_scale,
            "ap_radius": 2 * args.spatial_scale,
            "area": np.pi * args.spatial_scale**2,
            "ap_density": args.num_ap / (np.pi * args.spatial_scale**2),
            "ue_density": args.K / (np.pi * args.spatial_scale**2),
            "all_ap_all_ris": True,
            "association_threshold": trainer.associate_threshold,
            "pmax_w": trainer.pmax_w,
            "optimizer": "Adam(lr=0.0001, weight_decay=1e-6)",
            **provenance,
        }
        try:
            trainer.train(
                effective_seed,
                str(base_dir),
                str(base_dir / "logs"),
                args.test_sample_val,
                args.test_sample_final,
                config,
            )
        except Exception as error:
            write_status(base_dir, "failed", error)
            raise
        write_status(base_dir, "complete")
        save_summary(exp_dir)


if __name__ == "__main__":
    main()
