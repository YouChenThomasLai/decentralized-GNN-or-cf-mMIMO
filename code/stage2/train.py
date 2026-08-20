import json
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data import MOBILITY_STRAIGHT, MyDataLoader
from evaluate import SUMMARY_METRICS, TrajectoryEvaluator
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
