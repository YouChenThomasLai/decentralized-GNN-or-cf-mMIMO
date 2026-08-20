import argparse
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data import (
    DEFAULT_HOTSPOT_CENTERS,
    MOBILITY_HOTSPOT,
    MOBILITY_STRAIGHT,
    MyDataLoader,
)
from evaluate import METHODS, SUMMARY_METRICS, TrajectoryEvaluator
from model_2 import node_update
from utils_return_indivial_rates import (
    DIRECT_CHANNEL_FADING,
    DIRECT_CHANNEL_SCALE,
    DIRECT_PATH_LOSS_EXPONENT,
)


HOTSPOT_STICKINESS = {"low": 0.2, "medium": 0.6, "high": 0.9}


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


def validate_positive(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def source_checksums():
    source_dir = Path(__file__).resolve().parent
    filenames = (
        "data.py",
        "evaluate.py",
        "utils_return_indivial_rates.py",
        "model_2.py",
        "trainer_2.py",
        "run_exp-v2.sh",
        "test_stage2.py",
    )
    return {
        filename: hashlib.sha256((source_dir / filename).read_bytes()).hexdigest()
        for filename in filenames
    }


class Trainer:
    def __init__(
        self,
        M,
        K,
        batch_size,
        n_iter,
        pmax_dbm=10.0,
        noise_power=1e-12,
        speed_kmh=0.0,
        decision_period_s=0.001,
        carrier_frequency_hz=2.6e9,
        episode_steps=2000,
        train_trajectories=None,
        validation_trajectories=8,
        test_trajectories=40,
        eval_frame_batch_size=32,
        eval_time_stride=1,
        bootstrap_samples=500,
        seed=0,
        device="cuda:0",
        mobility_model=MOBILITY_STRAIGHT,
        hotspot_centers=None,
        hotspot_radius_m=10.0,
        transition_matrix=None,
        dwell_mean_s=5.0,
        dwell_shape=2.0,
        hotspot_trace_duration_s=300.0,
    ):
        self.M = M
        self.K = K
        self.pmax_dbm = pmax_dbm
        self.pmax_w = 10 ** ((pmax_dbm - 30) / 10)
        self.noise_power = noise_power
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.num_of_AP = 5
        self.episode_steps = episode_steps
        self.eval_frame_batch_size = eval_frame_batch_size
        self.eval_time_stride = eval_time_stride
        self.bootstrap_samples = bootstrap_samples
        self.seed = seed
        self.train_trajectory_count = (
            batch_size if train_trajectories is None else train_trajectories
        )
        self.training_rng = np.random.RandomState(seed + 3)

        loader_args = (
            M,
            episode_steps,
            speed_kmh,
            decision_period_s,
            carrier_frequency_hz,
            mobility_model,
            hotspot_centers,
            hotspot_radius_m,
            transition_matrix,
            dwell_mean_s,
            dwell_shape,
            hotspot_trace_duration_s,
        )
        self.train_data = self._make_loader(
            self.train_trajectory_count, seed, loader_args
        )
        self.validation_data = self._make_loader(
            validation_trajectories, seed + 1, loader_args
        )
        self.test_data = self._make_loader(
            test_trajectories, seed + 2, loader_args
        )
        self.dataloader = self.train_data

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.model = node_update(
            M,
            6,
            self.pmax_w,
            64,
            self.num_of_AP,
            self.device,
        ).to(self.device)
        self.log_interval = 10
        self.log_eval_interval = 500
        self.training_associate_threshold = 0.1
        self.associate_threshold = 0.1
        self.dup = False
        self.evaluator = TrajectoryEvaluator(
            self.model,
            self.K,
            self.pmax_w,
            self.num_of_AP,
            self.device,
            self.noise_power,
            self.episode_steps,
            self.eval_frame_batch_size,
            self.eval_time_stride,
            self.associate_threshold,
            self.bootstrap_samples,
        )
        for loader in (self.train_data, self.validation_data, self.test_data):
            loader.generate_trajectories(K, self.associate_threshold)
        print(
            f"[INFO] Per-AP Pmax = {self.pmax_dbm:g} dBm "
            f"= {self.pmax_w:g} W."
        )

    @staticmethod
    def _make_loader(trajectory_count, seed, loader_args):
        (
            M,
            episode_steps,
            speed_kmh,
            period_s,
            carrier_hz,
            mobility_model,
            hotspot_centers,
            hotspot_radius_m,
            transition_matrix,
            dwell_mean_s,
            dwell_shape,
            hotspot_trace_duration_s,
        ) = loader_args
        return MyDataLoader(
            M,
            trajectory_count,
            episode_steps,
            speed_kmh,
            period_s,
            carrier_hz,
            seed=seed,
            mobility_model=mobility_model,
            hotspot_centers=hotspot_centers,
            hotspot_radius_m=hotspot_radius_m,
            transition_matrix=transition_matrix,
            dwell_mean_s=dwell_mean_s,
            dwell_shape=dwell_shape,
            hotspot_trace_duration_s=hotspot_trace_duration_s,
        )

    def train_batch(self):
        self.model.train()
        trajectory_indices = self.training_rng.randint(
            0, self.train_trajectory_count, size=self.batch_size
        )
        time_indices = self.training_rng.randint(
            0, self.episode_steps, size=self.batch_size
        )
        user_feature, user_index, _, _ = self.train_data.get_frames(
            trajectory_indices, time_indices
        )
        user_feature = user_feature.to(self.device)

        self.opt.zero_grad()
        beamformers = self.model(
            user_feature,
            user_index,
            training=True,
            duplicate=self.dup,
        )
        loss, sum_rate, rate = self.train_data.compute_loss(
            beamformers,
            self.device,
            self.noise_power,
            trajectory_indices,
            time_indices,
        )
        loss.backward()
        self.opt.step()
        return loss.item(), sum_rate.item(), rate.detach().cpu()

    def train(self, run_id, out_dir, log_dir, config):
        model_dir = os.path.join(out_dir, "models")
        array_dir = os.path.join(out_dir, "arrays")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(array_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as config_file:
            json.dump(config, config_file, indent=2, sort_keys=True)

        writer = SummaryWriter(log_dir=log_dir)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=0.0001, weight_decay=1e-6
        )
        interval_loss = []
        interval_sum_rate = []
        train_total = []
        train_losses = []
        sum_rates = []
        validation = {metric: [] for metric in SUMMARY_METRICS}

        for iteration in range(self.n_iter):
            loss, sum_rate, rate = self.train_batch()
            train_losses.append(loss)
            sum_rates.append(sum_rate)
            interval_loss.append(loss)
            interval_sum_rate.append(sum_rate)

            writer.add_scalar("Train/Loss", loss, iteration)
            writer.add_scalar("Train/SumRate", sum_rate, iteration)
            for user_id, value in enumerate(rate):
                writer.add_scalar(
                    f"UserRate/User_{user_id + 1}", value, iteration
                )

            if iteration % self.log_interval == 0:
                mean_loss = np.mean(interval_loss)
                mean_sum_rate = np.mean(interval_sum_rate)
                print(
                    f"[Train | {iteration}/{self.n_iter}] "
                    f"loss = {mean_loss:.8g}, sum rate = {mean_sum_rate:.8g}"
                )
                train_total.append(mean_loss)
                interval_loss = []
                interval_sum_rate = []

            if (
                (iteration + 1) % self.log_eval_interval == 0
                and iteration >= 1
            ):
                metrics, _ = self.eval(self.validation_data)
                for metric, value in metrics.items():
                    validation.setdefault(metric, []).append(value)
                    writer.add_scalar(f"Val/{metric}", value, iteration + 1)
                    print(
                        f"[Val {metric} | {iteration + 1}/{self.n_iter}] "
                        f"value = {value:.8g}"
                    )

        checkpoint_path = os.path.join(
            model_dir, f"model_final_run{run_id}.pt"
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        print(
            "[INFO] Saved trained checkpoint before final eval: "
            f"{checkpoint_path}"
        )

        print("Running FINAL trajectory evaluation...")
        final_metrics, raw_metrics = self.eval(self.test_data)
        for metric, value in final_metrics.items():
            print(f"[Final Eval] {metric} = {value:.8g}")
        final_metrics = self._save_final_evaluation(
            run_id, out_dir, final_metrics, raw_metrics
        )

        writer.close()
        np.save(
            os.path.join(array_dir, f"losses_run{run_id}.npy"),
            np.asarray(train_losses),
        )
        np.save(
            os.path.join(array_dir, f"sumrates_run{run_id}.npy"),
            np.asarray(sum_rates),
        )
        np.save(
            os.path.join(array_dir, f"train_total_run{run_id}.npy"),
            np.asarray(train_total),
        )
        for metric, values in validation.items():
            np.save(
                os.path.join(array_dir, f"val_{metric}_run{run_id}.npy"),
                np.asarray(values),
            )
        return final_metrics

    def evaluate_checkpoint(self, run_id, out_dir, checkpoint_path, config):
        return self.evaluator.evaluate_checkpoint(
            self.test_data, run_id, out_dir, checkpoint_path, config
        )

    def _save_final_evaluation(
        self, run_id, out_dir, final_metrics, raw_metrics
    ):
        return self.evaluator.save_final_evaluation(
            self.test_data, run_id, out_dir, final_metrics, raw_metrics
        )

    def eval(self, loader):
        return self.evaluator.evaluate(loader)


def save_summary(exp_dir, seeds, run_results):
    keys = tuple(run_results[0])
    summary = {}
    for key in keys:
        values = np.asarray([result[key] for result in run_results])
        summary[key] = {
            "per_seed": dict(zip(seeds, values.tolist())),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    np.save(os.path.join(exp_dir, "final_summary.npy"), summary)
    with open(os.path.join(exp_dir, "final_summary.txt"), "w") as summary_file:
        for key in keys:
            metric = summary[key]
            summary_file.write(f"{key}\n")
            for seed, value in metric["per_seed"].items():
                summary_file.write(f"  seed {seed}: {value:.8g}\n")
            summary_file.write(
                f"  mean: {metric['mean']:.8g}\n"
                f"  std: {metric['std']:.8g}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 fixed-association mobility trainer"
    )
    parser.add_argument("--M", type=int, default=4, help="Antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of users")
    parser.add_argument(
        "--pmax_dbm",
        "--Pmax",
        dest="pmax_dbm",
        type=float,
        default=10.0,
        help="Per-AP maximum transmit power in dBm",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_iter", type=int, default=2000)
    parser.add_argument("--noise_power", type=float, default=1e-12)
    parser.add_argument("--speed_kmh", type=float, default=0.0)
    parser.add_argument(
        "--mobility_model",
        choices=(MOBILITY_STRAIGHT, MOBILITY_HOTSPOT),
        default=MOBILITY_STRAIGHT,
    )
    parser.add_argument("--decision_period_s", type=float, default=0.001)
    parser.add_argument("--carrier_frequency_hz", type=float, default=2.6e9)
    parser.add_argument("--episode_steps", type=int, default=2000)
    parser.add_argument("--train_trajectories", type=int)
    parser.add_argument("--test_sample_val", type=int, default=8)
    parser.add_argument("--test_sample_final", type=int, default=40)
    parser.add_argument("--eval_frame_batch_size", type=int, default=32)
    parser.add_argument("--eval_time_stride", type=int, default=1)
    parser.add_argument("--bootstrap_samples", type=int, default=500)
    parser.add_argument(
        "--hotspot_centers",
        default=json.dumps(DEFAULT_HOTSPOT_CENTERS.tolist()),
        help="JSON array with shape [J,2]",
    )
    parser.add_argument("--hotspot_radius_m", type=float, default=10.0)
    parser.add_argument(
        "--hotspot_stickiness",
        choices=tuple(HOTSPOT_STICKINESS),
        default="medium",
    )
    parser.add_argument(
        "--hotspot_transition_matrix",
        help="Optional JSON row-stochastic matrix; overrides stickiness",
    )
    parser.add_argument("--hotspot_dwell_mean_s", type=float, default=5.0)
    parser.add_argument("--hotspot_dwell_shape", type=float, default=2.0)
    parser.add_argument(
        "--hotspot_trace_duration_s", type=float, default=300.0
    )
    parser.add_argument("--bs_file", type=str, default="BS_{i}.txt")
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Frozen model state_dict to evaluate without training",
    )
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = (
            "results_stage2b_hotspot"
            if args.mobility_model == MOBILITY_HOTSPOT
            else "results_stage2"
        )

    checks = {
        "--M": args.M,
        "--K": args.K,
        "--batch_size": args.batch_size,
        "--runs": args.runs,
        "--n_iter": args.n_iter,
        "--episode_steps": args.episode_steps,
        "--test_sample_val": args.test_sample_val,
        "--test_sample_final": args.test_sample_final,
        "--eval_frame_batch_size": args.eval_frame_batch_size,
        "--eval_time_stride": args.eval_time_stride,
        "--bootstrap_samples": args.bootstrap_samples,
    }
    if args.train_trajectories is not None:
        checks["--train_trajectories"] = args.train_trajectories
    try:
        for name, value in checks.items():
            validate_positive(name, value)
    except ValueError as error:
        parser.error(str(error))
    if args.noise_power <= 0:
        parser.error("--noise_power must be positive")
    if args.speed_kmh < 0:
        parser.error("--speed_kmh cannot be negative")
    if args.decision_period_s <= 0 or args.carrier_frequency_hz <= 0:
        parser.error("Channel timing and carrier frequency must be positive")
    hotspot_centers = None
    transition_matrix = None
    if args.mobility_model == MOBILITY_HOTSPOT:
        try:
            hotspot_centers = np.asarray(
                json.loads(args.hotspot_centers), dtype=np.float64
            )
            if args.hotspot_transition_matrix:
                transition_matrix = np.asarray(
                    json.loads(args.hotspot_transition_matrix),
                    dtype=np.float64,
                )
            else:
                hotspot_count = len(hotspot_centers)
                stickiness = HOTSPOT_STICKINESS[args.hotspot_stickiness]
                if hotspot_count == 1:
                    transition_matrix = np.ones((1, 1))
                else:
                    transition_matrix = np.full(
                        (hotspot_count, hotspot_count),
                        (1 - stickiness) / (hotspot_count - 1),
                    )
                    np.fill_diagonal(transition_matrix, stickiness)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(f"Invalid hotspot JSON: {error}")
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            parser.error(f"--checkpoint does not exist: {checkpoint_path}")

    speed_label = f"{args.speed_kmh:g}".replace(".", "p")
    mobility_prefix = (
        "hotspot_" if args.mobility_model == MOBILITY_HOTSPOT else ""
    )
    exp_name = (
        f"{mobility_prefix}speed{speed_label}_M{args.M}_K{args.K}_"
        f"P{args.pmax_dbm:g}"
    )
    exp_dir = os.path.join(args.out_dir, exp_name)
    effective_seeds = []
    run_results = []

    for run_id in range(args.runs):
        effective_seed = args.seed + run_id
        seed_everything(effective_seed)
        print(f"[INFO] Run {run_id} effective seed: {effective_seed}")
        base_dir = os.path.join(exp_dir, f"run{run_id}")
        log_dir = os.path.join(base_dir, "logs")
        trainer = Trainer(
            args.M,
            args.K,
            args.batch_size,
            args.n_iter,
            args.pmax_dbm,
            args.noise_power,
            args.speed_kmh,
            args.decision_period_s,
            args.carrier_frequency_hz,
            args.episode_steps,
            args.train_trajectories,
            args.test_sample_val,
            args.test_sample_final,
            args.eval_frame_batch_size,
            args.eval_time_stride,
            args.bootstrap_samples,
            effective_seed,
            args.device,
            args.mobility_model,
            hotspot_centers,
            args.hotspot_radius_m,
            transition_matrix,
            args.hotspot_dwell_mean_s,
            args.hotspot_dwell_shape,
            args.hotspot_trace_duration_s,
        )
        array_dir = os.path.join(base_dir, "arrays")
        os.makedirs(array_dir, exist_ok=True)
        np.savetxt(
            os.path.join(array_dir, args.bs_file.format(i=run_id)),
            trainer.dataloader.BS_Loc_array,
            fmt="%f",
        )

        config = {
            "execution_mode": (
                "frozen_checkpoint_evaluation"
                if checkpoint_path
                else "matched_training"
            ),
            "effective_seed": effective_seed,
            "mobility_model": args.mobility_model,
            "hotspot_config": (
                {
                    "centers": trainer.test_data.hotspot_centers.tolist(),
                    "radius_m": trainer.test_data.hotspot_radius_m,
                    "transition_matrix": (
                        trainer.test_data.transition_matrix.tolist()
                    ),
                    "stickiness_preset": (
                        None
                        if args.hotspot_transition_matrix
                        else args.hotspot_stickiness
                    ),
                    "dwell_distribution": "Gamma(shape, mean / shape)",
                    "dwell_mean_s": trainer.test_data.dwell_mean_s,
                    "dwell_shape": trainer.test_data.dwell_shape,
                    "long_macro_trace_duration_s": (
                        trainer.test_data.hotspot_trace_duration_s
                    ),
                    "clip_sampling": "uniform start time",
                }
                if args.mobility_model == MOBILITY_HOTSPOT
                else None
            ),
            "cli": vars(args),
            "effective_device": str(trainer.device),
            "num_ap": trainer.num_of_AP,
            "association_threshold": trainer.associate_threshold,
            "association_policy": "instantaneous RSSI at t=0, then fixed",
            "fixed_association_regret": (
                "current-RSSI reassociation sum rate minus fixed-association "
                "sum rate, evaluated on identical frames"
            ),
            "csi_policy": "all stored CSI equals current true CSI",
            "trajectory_split_seeds": {
                "train": effective_seed,
                "validation": effective_seed + 1,
                "test": effective_seed + 2,
            },
            "train_trajectories": trainer.train_trajectory_count,
            "pmax_w": trainer.pmax_w,
            "direct_channel_fading": DIRECT_CHANNEL_FADING,
            "direct_path_loss_exponent": DIRECT_PATH_LOSS_EXPONENT,
            "direct_channel_scale_exponent": DIRECT_CHANNEL_SCALE,
            "noise_power": args.noise_power,
            "rzf_regularization": "alpha = K_a * noise_power / Pmax",
            "optimizer": (
                None
                if checkpoint_path
                else "Adam(lr=0.0001, weight_decay=1e-6)"
            ),
            "checkpoint": (
                {
                    "path": str(checkpoint_path),
                    "sha256": hashlib.sha256(
                        checkpoint_path.read_bytes()
                    ).hexdigest(),
                }
                if checkpoint_path
                else None
            ),
            "source_sha256": source_checksums(),
        }
        if checkpoint_path:
            result = trainer.evaluate_checkpoint(
                run_id, base_dir, checkpoint_path, config
            )
        else:
            result = trainer.train(run_id, base_dir, log_dir, config)
        run_results.append(result)
        effective_seeds.append(effective_seed)

    save_summary(exp_dir, effective_seeds, run_results)


if __name__ == "__main__":
    main()
