import argparse
import json
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data import MyDataLoader
from model_2 import node_update
from utils_return_indivial_rates import (
    DIRECT_CHANNEL_FADING,
    DIRECT_CHANNEL_SCALE,
    DIRECT_PATH_LOSS_EXPONENT,
    NOISE_POWER,
    mrt_beamforming,
    rzf_beamforming,
)


METHODS = ("centralized_gnn", "decentralized_gnn", "mrt", "rzf")


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


def validate_sample_count(name, sample_count, batch_size):
    if sample_count <= 0 or sample_count % batch_size != 0:
        raise ValueError(
            f"{name} must be a positive multiple of batch_size={batch_size}"
        )


class Trainer:
    def __init__(
        self,
        M,
        K,
        batch_size,
        n_iter,
        pmax_dbm=10.0,
        device="cuda:0",
    ):
        self.M = M
        self.K = K
        self.pmax_dbm = pmax_dbm
        self.pmax_w = 10 ** ((pmax_dbm - 30) / 10)
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.num_of_AP = 5
        self.dataloader = MyDataLoader(M, batch_size)
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
        loss, sum_rate, rate = self.dataloader.compute_loss(W, self.device)
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
        validate_sample_count(
            "evaluation sample count", test_sample, self.batch_size
        )
        self.model.eval()
        sum_rates = {method: [] for method in METHODS}

        with torch.no_grad():
            for _ in range(test_sample // self.batch_size):
                centralized_feature, centralized_index = (
                    self.dataloader.gen_training_data(
                        self.K,
                        self.associate_threshold,
                        duplicate=False,
                    )
                )
                centralized_feature = centralized_feature.to(self.device)
                centralized_w = self.model(
                    centralized_feature,
                    centralized_index,
                    training=True,
                    duplicate=False,
                )
                _, centralized_rate, _ = self.dataloader.compute_loss(
                    centralized_w, self.device
                )
                sum_rates["centralized_gnn"].append(
                    centralized_rate.item()
                )

                decentralized_feature, decentralized_index = (
                    self.dataloader.gen_testing_data(
                        self.K,
                        self.associate_threshold,
                        duplicate=False,
                        regenerate_channels=False,
                    )
                )
                decentralized_feature = [
                    feature.to(self.device)
                    for feature in decentralized_feature
                ]
                decentralized_w = self.model(
                    decentralized_feature,
                    decentralized_index,
                    training=False,
                    duplicate=False,
                )
                _, decentralized_rate, _ = self.dataloader.compute_loss(
                    decentralized_w, self.device
                )
                sum_rates["decentralized_gnn"].append(
                    decentralized_rate.item()
                )

                channels = self.dataloader.get_stacked_channels()
                association_mask = self.dataloader.get_association_mask()
                mrt_w = mrt_beamforming(
                    channels,
                    association_mask,
                    self.pmax_w,
                    self.device,
                )
                _, mrt_rate, _ = self.dataloader.compute_loss(
                    mrt_w, self.device
                )
                sum_rates["mrt"].append(mrt_rate.item())

                rzf_w = rzf_beamforming(
                    channels,
                    association_mask,
                    self.pmax_w,
                    self.device,
                )
                _, rzf_rate, _ = self.dataloader.compute_loss(
                    rzf_w, self.device
                )
                sum_rates["rzf"].append(rzf_rate.item())

        return {
            method: float(np.mean(values))
            for method, values in sum_rates.items()
        }


def save_summary(exp_dir, seeds, run_results):
    keys = (*METHODS, "centralized_minus_decentralized")
    summary = {}
    for key in keys:
        values = np.asarray([result[key] for result in run_results])
        summary[key] = {
            "per_seed": dict(zip(seeds, values.tolist())),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    gaps = np.asarray(
        [result["centralized_minus_decentralized"] for result in run_results]
    )
    dominant_sign_count = int(max(np.sum(gaps > 0), np.sum(gaps < 0)))
    summary["gap_dominant_sign_count"] = dominant_sign_count
    summary["gap_sign_consistent"] = dominant_sign_count >= min(4, len(gaps))
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
        summary_file.write(
            f"gap_dominant_sign_count: {dominant_sign_count}/{len(gaps)}\n"
        )
        summary_file.write(
            f"gap_sign_consistent: {summary['gap_sign_consistent']}\n"
        )

    if len(gaps) >= 5 and dominant_sign_count < 4:
        print(
            "[WARN] The centralized-minus-decentralized gap does not have "
            "the same sign in at least four of five seeds."
        )


def main():
    parser = argparse.ArgumentParser(description="Stage 1 no-RIS trainer")
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_iter", type=int, default=2000)
    parser.add_argument("--bs_file", type=str, default="BS_{i}.txt")
    parser.add_argument("--out_dir", type=str, default="results_stage1")
    parser.add_argument("--test_sample_val", type=int, default=128)
    parser.add_argument("--test_sample_final", type=int, default=3200)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.n_iter <= 0:
        parser.error("--n_iter must be positive")
    try:
        validate_sample_count(
            "--test_sample_val", args.test_sample_val, args.batch_size
        )
        validate_sample_count(
            "--test_sample_final", args.test_sample_final, args.batch_size
        )
    except ValueError as error:
        parser.error(str(error))

    exp_name = f"M{args.M}_K{args.K}_P{args.pmax_dbm}"
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
            device=args.device,
        )
        array_dir = os.path.join(base_dir, "arrays")
        os.makedirs(array_dir, exist_ok=True)
        np.savetxt(
            os.path.join(array_dir, args.bs_file.format(i=run_id)),
            trainer.dataloader.BS_Loc_array,
            fmt="%f",
        )

        config = {
            "effective_seed": effective_seed,
            "cli": vars(args),
            "effective_device": str(trainer.device),
            "num_ap": trainer.num_of_AP,
            "association_threshold": trainer.associate_threshold,
            "pmax_w": trainer.pmax_w,
            "direct_channel_fading": DIRECT_CHANNEL_FADING,
            "direct_path_loss_exponent": DIRECT_PATH_LOSS_EXPONENT,
            "direct_channel_scale_exponent": DIRECT_CHANNEL_SCALE,
            "noise_power": NOISE_POWER,
            "rzf_regularization": "alpha = K_a * noise_power / Pmax",
            "optimizer": "Adam(lr=0.0001, weight_decay=1e-6)",
        }
        run_results.append(
            trainer.train(
                run_id,
                base_dir,
                log_dir,
                args.test_sample_val,
                args.test_sample_final,
                config,
            )
        )
        effective_seeds.append(effective_seed)

    save_summary(exp_dir, effective_seeds, run_results)


if __name__ == "__main__":
    main()
