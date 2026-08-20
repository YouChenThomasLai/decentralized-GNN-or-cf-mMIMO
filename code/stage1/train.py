import json
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from environment import SnapshotEnvironment
from evaluate import METHODS, evaluate_snapshot
from model_2 import node_update
from utils_return_indivial_rates import NOISE_POWER


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
        noise_power=NOISE_POWER,
        device="cuda:0",
    ):
        self.M = M
        self.K = K
        self.pmax_dbm = pmax_dbm
        self.pmax_w = 10 ** ((pmax_dbm - 30) / 10)
        self.noise_power = noise_power
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.num_of_AP = 5
        self.dataloader = SnapshotEnvironment(M, batch_size)
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
        print(
            f"[INFO] Per-AP Pmax = {self.pmax_dbm:g} dBm "
            f"= {self.pmax_w:g} W."
        )

    def train_batch(self):
        self.model.train()
        user_feature, user_index = self.dataloader.gen_training_data(
            self.K,
            self.training_associate_threshold,
            duplicate=self.dup,
        )
        user_feature = user_feature.to(self.device)

        self.opt.zero_grad()
        W = self.model(
            user_feature,
            user_index,
            training=True,
            duplicate=self.dup,
        )
        loss, sum_rate, rate = self.dataloader.compute_loss(
            W, self.device, self.noise_power
        )
        loss.backward()
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
        validation = {method: [] for method in METHODS}

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
                metrics = self.eval(test_sample_val)
                for method, value in metrics.items():
                    validation[method].append(value)
                    writer.add_scalar(
                        f"Val/{method}", value, iteration + 1
                    )
                    print(
                        f"[Val {method} | {iteration + 1}/{self.n_iter}] "
                        f"sum rate = {value:.8g}"
                    )

        print("Running FINAL evaluation with more samples...")
        final_metrics = self.eval(test_sample_final)
        for method, value in final_metrics.items():
            print(f"[Final Eval] {method} = {value:.8g}")

        final_eval_results = {
            **final_metrics,
            "centralized_minus_decentralized": (
                final_metrics["centralized_gnn"]
                - final_metrics["decentralized_gnn"]
            ),
        }
        final_dir = os.path.join(out_dir, "final_eval")
        os.makedirs(final_dir, exist_ok=True)
        np.save(
            os.path.join(final_dir, f"final_eval_run{run_id}.npy"),
            final_eval_results,
        )
        with open(
            os.path.join(final_dir, f"final_eval_run{run_id}.txt"), "w"
        ) as final_file:
            for key, value in final_eval_results.items():
                final_file.write(f"{key}: {value:.8g}\n")

        writer.close()
        torch.save(
            self.model.state_dict(),
            os.path.join(model_dir, f"model_final_run{run_id}.pt"),
        )
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
        for method, values in validation.items():
            np.save(
                os.path.join(
                    array_dir, f"val_sum_rate_{method}_run{run_id}.npy"
                ),
                np.asarray(values),
            )
        return final_eval_results

    def eval(self, test_sample):
        return evaluate_snapshot(
            self.model,
            self.dataloader,
            test_sample,
            K=self.K,
            batch_size=self.batch_size,
            associate_threshold=self.associate_threshold,
            pmax_w=self.pmax_w,
            num_of_AP=self.num_of_AP,
            device=self.device,
            noise_power=self.noise_power,
        )
