import argparse
from datetime import datetime, timezone
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
from geometry import DEFAULT_SQUARE_SIDE, GATE_VERSION, pairwise_wrapped_distances
from model_2 import node_update
from utils_return_indivial_rates import CHANNEL_SCALE_EXPONENT, RATE_NOISE_POWER


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
        square_side=DEFAULT_SQUARE_SIDE,
        topology_seed=0,
        channel_seed=0,
        evaluation_seed=0,
    ):
        self.M = M
        self.N = N
        self.L = L
        self.K = K
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.num_of_AP = num_ap
        self.square_side = square_side
        self.topology_seed = topology_seed
        self.channel_seed = channel_seed
        self.evaluation_seed = evaluation_seed
        self.pmax_dbm = pmax_dbm
        self.pmax_w = 10 ** ((pmax_dbm - 30) / 10)
        loader_kwargs = {
            "num_ap": num_ap,
            "square_side": square_side,
            "topology_seed": topology_seed,
        }
        self.dataloader = MyDataLoader(
            M,
            N,
            L,
            batch_size,
            channel_seed=channel_seed,
            **loader_kwargs,
        )
        self.eval_dataloader = MyDataLoader(
            M,
            N,
            L,
            batch_size,
            channel_seed=evaluation_seed,
            **loader_kwargs,
        )
        self.dataloader.BS_RIS_association()
        self.eval_dataloader.BS_RIS_association()
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
        self.current_iteration = None
        self.curves = None
        self.validation = None
        self.training_started_at = None
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
        if not all(torch.isfinite(value).all() for value in (loss, sum_rate, rate)):
            raise FloatingPointError("Non-finite training outputs")
        loss.backward()
        gradients = [
            parameter.grad.detach()
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise FloatingPointError("Missing or non-finite gradients")
        gradient_norm = torch.sqrt(
            sum(value.square().sum() for value in gradients)
        ).item()
        parameter_norm = torch.sqrt(
            sum(
                parameter.detach().square().sum()
                for parameter in self.model.parameters()
            )
        ).item()
        decay_loss = 0.5e-6 * parameter_norm**2
        if self.initial_gradient_diagnostics is None:
            decay_norm = 1e-6 * parameter_norm
            self.initial_gradient_diagnostics = {
                "initial_task_gradient_norm": gradient_norm,
                "initial_decay_gradient_norm": decay_norm,
                "initial_decay_to_task_ratio": (
                    decay_norm / gradient_norm
                    if gradient_norm
                    else float("inf")
                ),
            }
        self.opt.step()
        if not all(
            torch.isfinite(parameter).all()
            for parameter in self.model.parameters()
        ):
            raise FloatingPointError("Non-finite model parameters")
        return (
            loss.item(),
            sum_rate.item(),
            rate.detach().cpu(),
            gradient_norm,
            parameter_norm,
            decay_loss,
        )

    def train(
        self,
        run_id,
        out_dir,
        log_dir,
        test_sample_val,
        test_sample_final,
        config,
    ):
        out_dir = Path(out_dir)
        model_dir = out_dir / "models"
        array_dir = out_dir / "arrays"
        final_dir = out_dir / "final_eval"
        for path in (model_dir, array_dir, final_dir, Path(log_dir)):
            path.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.json", "w") as config_file:
            json.dump(config, config_file, indent=2, sort_keys=True)

        writer = SummaryWriter(log_dir=log_dir)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=0.0001, weight_decay=1e-6
        )
        curves = {
            "losses": [],
            "sum_rates": [],
            "gradient_norms": [],
            "parameter_norms": [],
            "decay_losses": [],
        }
        validation = {method: [] for method in METHODS}
        self.curves = curves
        self.validation = validation
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        training_started = time.perf_counter()
        self.training_started_at = training_started

        try:
            for iteration in range(self.n_iter):
                self.current_iteration = iteration
                outputs = self.train_batch()
                (
                    loss,
                    sum_rate,
                    rate,
                    gradient_norm,
                    parameter_norm,
                    decay_loss,
                ) = outputs
                for name, value in zip(curves, outputs[:2] + outputs[3:]):
                    curves[name].append(value)
                writer.add_scalar("Train/Loss", loss, iteration)
                writer.add_scalar("Train/SumRate", sum_rate, iteration)
                writer.add_scalar("Train/GradientNorm", gradient_norm, iteration)
                writer.add_scalar("Train/ParameterNorm", parameter_norm, iteration)
                writer.add_scalar("Train/DecayLoss", decay_loss, iteration)
                for user_id, value in enumerate(rate):
                    writer.add_scalar(
                        f"UserRate/User_{user_id + 1}", value, iteration
                    )
                if iteration % self.log_interval == 0:
                    start = max(0, len(curves["losses"]) - self.log_interval)
                    print(
                        f"[Train | {iteration}/{self.n_iter}] "
                        f"loss = {np.mean(curves['losses'][start:]):.8g}, "
                        "sum rate = "
                        f"{np.mean(curves['sum_rates'][start:]):.8g}"
                    )
                if (iteration + 1) % self.log_eval_interval == 0:
                    metrics = self.eval(
                        test_sample_val,
                        evaluation_seed=self.evaluation_seed + iteration + 1,
                    )
                    for method in METHODS:
                        validation[method].append(metrics[method])
                        writer.add_scalar(
                            f"Val/{method}", metrics[method], iteration + 1
                        )

            training_seconds = time.perf_counter() - training_started
            print("Running FINAL paired evaluation...")
            final_metrics, final_details = self.eval(
                test_sample_final,
                return_details=True,
                evaluation_seed=self.evaluation_seed,
                clear_history=True,
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
                print(f"[Final Eval] {key} = {value}")

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
                    final_file.write(f"{key}: {value}\n")

            for name, values in curves.items():
                np.save(array_dir / f"{name}_run{run_id}.npy", values)
            for method, values in validation.items():
                np.save(
                    array_dir / f"val_sum_rate_{method}_run{run_id}.npy",
                    values,
                )
            torch.save(
                self.model.state_dict(), model_dir / f"model_final_run{run_id}.pt"
            )
            self.save_histories(array_dir)
            checks = self.computed_checks(
                final_metrics, final_details, curves, test_sample_final
            )
            with open(out_dir / "checks.json", "w") as checks_file:
                json.dump(checks, checks_file, indent=2, sort_keys=True)
            if not checks["passed"]:
                failed = [
                    name
                    for name, passed in checks.items()
                    if isinstance(passed, bool) and not passed
                ]
                raise RuntimeError(f"Final gates failed: {', '.join(failed)}")
            return final_metrics
        finally:
            writer.close()

    def computed_checks(self, metrics, details, curves, test_sample):
        topology_checks = self.eval_dataloader.validate_topology()
        area = self.square_side**2
        expected_area = DEFAULT_SQUARE_SIDE**2 * (self.num_of_AP / 5)
        density_values = {
            "ap": self.num_of_AP / area,
            "ue": self.K / area,
            "ris": self.L / area,
        }
        design_values = {
            "ap": 5 / DEFAULT_SQUARE_SIDE**2,
            "ue": 8 / DEFAULT_SQUARE_SIDE**2,
            "ris": 4 / DEFAULT_SQUARE_SIDE**2,
        }
        topology_density = np.isclose(area, expected_area) and all(
            np.isclose(density_values[name], design_values[name])
            for name in density_values
        )
        numerical_metrics = [
            value
            for value in metrics.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        finite_outputs = (
            np.isfinite(numerical_metrics).all()
            and all(np.isfinite(values).all() for values in curves.values())
            and all(
                np.isfinite(value).all()
                for value in details.values()
                if np.issubdtype(np.asarray(value).dtype, np.number)
            )
        )
        checks = {
            "all_ap_all_ris": topology_checks["all_ap_all_ris"],
            "association_mask": topology_checks["each_ue_has_serving_ap"],
            "cd_pairing": metrics["paired_snapshot_count"] == test_sample,
            "noise_and_power_pairing": bool(
                RATE_NOISE_POWER > 0 and self.pmax_w > 0
            ),
            "finite_outputs": bool(finite_outputs),
            "per_ap_power": bool(
                details["centralized_ap_power"].max() <= self.pmax_w + 1e-6
                and details["decentralized_ap_power"].max()
                <= self.pmax_w + 1e-6
            ),
            "topology_density": bool(topology_density),
            "wrapped_geometry": topology_checks["wrapped_geometry"],
            "topology_contract": topology_checks["passed"],
            "learned_vs_random": bool(
                metrics["learned_vs_random_centralized_positive"]
                and metrics["learned_vs_random_decentralized_positive"]
            ),
        }
        checks["passed"] = all(checks.values())
        checks["details"] = {
            "gate_version": GATE_VERSION,
            "computed_densities": density_values,
            "design_densities": design_values,
            "topology_checks": topology_checks,
            "max_centralized_ap_power": float(
                details["centralized_ap_power"].max()
            ),
            "max_decentralized_ap_power": float(
                details["decentralized_ap_power"].max()
            ),
        }
        return checks

    def save_histories(self, array_dir):
        array_dir = Path(array_dir)
        self.dataloader.save_history(array_dir / "training_topology_history.npz")
        self.eval_dataloader.save_history(
            array_dir / "evaluation_topology_history.npz"
        )

    def save_failure_artifacts(self, out_dir, error):
        out_dir = Path(out_dir)
        model_dir = out_dir / "models"
        array_dir = out_dir / "arrays"
        model_dir.mkdir(parents=True, exist_ok=True)
        array_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "iteration": self.current_iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.opt.state_dict() if hasattr(self, "opt") else None
            ),
        }
        torch.save(checkpoint, model_dir / "model_partial_failure.pt")
        if self.curves is not None:
            for name, values in self.curves.items():
                np.save(array_dir / f"{name}_partial.npy", values)
        with open(out_dir / "failure.json", "w") as failure_file:
            json.dump(
                {
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "iteration": self.current_iteration,
                    "elapsed_seconds": (
                        time.perf_counter() - self.training_started_at
                        if self.training_started_at is not None
                        else None
                    ),
                },
                failure_file,
                indent=2,
                sort_keys=True,
            )

    def eval(
        self,
        test_sample,
        return_details=False,
        evaluation_seed=None,
        clear_history=False,
    ):
        if evaluation_seed is not None:
            self.eval_dataloader.reset_channel_seed(
                evaluation_seed, clear_history=clear_history
            )
            torch.manual_seed(evaluation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(evaluation_seed)
        return evaluate_snapshot(
            self.model,
            self.eval_dataloader,
            test_sample,
            K=self.K,
            batch_size=self.batch_size,
            associate_threshold=self.associate_threshold,
            pmax_w=self.pmax_w,
            device=self.device,
            return_details=return_details,
        )


def repository_root(path):
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            return parent
    return path.parents[3]


def source_provenance():
    source_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    source_paths = sorted(
        (*source_dir.glob("*.py"), *source_dir.glob("*.sh"))
    )
    for path in source_paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())

    def git(*args):
        result = subprocess.run(
            ("git", *args),
            cwd=repository_root(source_dir),
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    dirty_status = os.environ.get("SOURCE_DIRTY_STATUS")
    if dirty_status is None:
        dirty_status = git("status", "--short")
    return {
        "source_commit": os.environ.get("SOURCE_COMMIT")
        or git("rev-parse", "HEAD"),
        "source_dirty": bool(dirty_status),
        "source_dirty_status": dirty_status,
        "source_hash_sha256": digest.hexdigest(),
    }


def write_status(run_dir, status, error=None, iteration=None):
    payload = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        payload.update(
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "finite_failure_onset_iteration": iteration,
            }
        )
    with open(Path(run_dir) / "status.json", "w") as status_file:
        json.dump(payload, status_file, indent=2, sort_keys=True)


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
        values = [metrics[key] for _, metrics in records]
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            continue
        array = np.asarray(values, dtype=float)
        summary[key] = {
            "per_seed": {
                str(seed): float(value)
                for (seed, _), value in zip(records, array)
            },
            "mean": float(array.mean()),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    with open(Path(exp_dir) / "final_summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 0 RIS v2 square-BPP network-scaling trainer"
    )
    parser.add_argument("--scale", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--num_ap", type=int)
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--L", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--square_side", type=float)
    parser.add_argument(
        "--pmax_dbm", "--Pmax", dest="pmax_dbm", type=float, default=15.0
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topology_seed", type=int)
    parser.add_argument("--channel_seed", type=int)
    parser.add_argument("--evaluation_seed", type=int)
    parser.add_argument("--seed_extension_reason")
    parser.add_argument("--n_iter", type=int, default=2000)
    parser.add_argument("--out_dir")
    parser.add_argument("--test_sample_val", type=int, default=128)
    parser.add_argument("--test_sample_final", type=int, default=3200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    design = {
        "num_ap": 5 * args.scale,
        "K": 8 * args.scale,
        "L": 4 * args.scale,
        "square_side": DEFAULT_SQUARE_SIDE * np.sqrt(args.scale),
    }
    for name, expected in design.items():
        supplied = getattr(args, name)
        if supplied is not None and not np.isclose(supplied, expected):
            parser.error(f"--{name} is frozen to {expected:g} for scale {args.scale}")
        setattr(args, name, expected if supplied is None else supplied)
    if args.M != 2 or args.N != 30 or not np.isclose(args.pmax_dbm, 15):
        parser.error("The v2 plan freezes M=2, N=30, and pmax_dbm=15")
    for name in ("batch_size", "runs", "n_iter"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    last_seed = args.seed + args.runs - 1
    if args.seed < 0 or last_seed > 4:
        parser.error("The frozen seed plan permits seeds 0 through 4 only")
    if last_seed >= 3 and not args.seed_extension_reason:
        parser.error("Seeds 3-4 require --seed_extension_reason")
    for name in ("topology_seed", "channel_seed", "evaluation_seed"):
        if getattr(args, name) is not None and getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative")
    try:
        validate_sample_count(
            "--test_sample_val", args.test_sample_val, args.batch_size
        )
        validate_sample_count(
            "--test_sample_final", args.test_sample_final, args.batch_size
        )
    except ValueError as error:
        parser.error(str(error))
    if args.out_dir is None:
        args.out_dir = (
            "../../results_snapshot_scaling_v2_bpp/stage0_ris/"
            f"scale{args.scale}_A{args.num_ap}_K{args.K}_L{args.L}"
        )
    return args


def main():
    args = parse_args()
    exp_dir = Path(args.out_dir)
    provenance = source_provenance()
    for run_offset in range(args.runs):
        seeds = {
            "training_seed": args.seed + run_offset,
            "topology_seed": (
                args.seed + run_offset
                if args.topology_seed is None
                else args.topology_seed + run_offset
            ),
            "channel_seed": (
                args.seed + run_offset
                if args.channel_seed is None
                else args.channel_seed + run_offset
            ),
            "evaluation_seed": (
                args.seed + run_offset
                if args.evaluation_seed is None
                else args.evaluation_seed + run_offset
            ),
        }
        run_seed = seeds["training_seed"]
        base_dir = exp_dir / f"seed{run_seed}"
        if base_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {base_dir}")
        base_dir.mkdir(parents=True)
        write_status(base_dir, "running")
        with open(base_dir / "requested_config.json", "w") as config_file:
            json.dump(
                {
                    "gate_version": GATE_VERSION,
                    "cli": shlex.join([sys.executable, *sys.argv]),
                    "arguments": vars(args),
                    **seeds,
                    **provenance,
                },
                config_file,
                indent=2,
                sort_keys=True,
            )
        seed_everything(run_seed)
        trainer = None
        try:
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
                square_side=args.square_side,
                topology_seed=seeds["topology_seed"],
                channel_seed=seeds["channel_seed"],
                evaluation_seed=seeds["evaluation_seed"],
            )
            ap_distances = pairwise_wrapped_distances(
                trainer.dataloader.BS_Loc_array,
                trainer.dataloader.BS_Loc_array,
                args.square_side,
            )
            np.fill_diagonal(ap_distances, np.inf)
            array_dir = base_dir / "arrays"
            array_dir.mkdir()
            np.savez_compressed(
                array_dir / "topology.npz",
                ap_locations=trainer.dataloader.BS_Loc_array,
                ris_locations=trainer.dataloader.RIS_Loc_array,
                ap_nearest_neighbor_distance=ap_distances.min(axis=1),
                ap_ris_wrapped_distances=pairwise_wrapped_distances(
                    trainer.dataloader.BS_Loc_array,
                    trainer.dataloader.RIS_Loc_array,
                    args.square_side,
                ),
                square_side=args.square_side,
                topology_seed=seeds["topology_seed"],
                wrap_around=True,
            )
            area = args.square_side**2
            config = {
                "gate_version": GATE_VERSION,
                "scale": args.scale,
                "effective_seed": run_seed,
                **seeds,
                "cli": shlex.join([sys.executable, *sys.argv]),
                "arguments": vars(args),
                "effective_device": str(trainer.device),
                "num_ap": args.num_ap,
                "num_ue": args.K,
                "num_ris": args.L,
                "square_side": args.square_side,
                "area": area,
                "ap_density": args.num_ap / area,
                "ue_density": args.K / area,
                "ris_density": args.L / area,
                "ue_ap_ratio": args.K / args.num_ap,
                "ris_ap_ratio": args.L / args.num_ap,
                "wrap_around": True,
                "los_angle_model": "existing stochastic angles (no geometric direction input)",
                "all_ap_all_ris": True,
                "association_threshold": trainer.associate_threshold,
                "pmax_w": trainer.pmax_w,
                "optimizer": "Adam(lr=0.0001, weight_decay=1e-6)",
                "channel_scale_exponent": CHANNEL_SCALE_EXPONENT,
                "rate_noise_power": RATE_NOISE_POWER,
                **provenance,
            }
            trainer.train(
                run_seed,
                str(base_dir),
                str(base_dir / "logs"),
                args.test_sample_val,
                args.test_sample_final,
                config,
            )
        except Exception as error:
            if trainer is not None:
                trainer.save_failure_artifacts(base_dir, error)
                trainer.save_histories(base_dir / "arrays")
            write_status(
                base_dir,
                "failed",
                error,
                None if trainer is None else trainer.current_iteration,
            )
            raise
        write_status(base_dir, "complete")
        save_summary(exp_dir)


if __name__ == "__main__":
    main()
