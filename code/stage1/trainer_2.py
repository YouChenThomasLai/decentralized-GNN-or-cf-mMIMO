import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

from evaluate import METHODS, validate_sample_count
from train import Trainer, seed_everything
from utils_return_indivial_rates import (
    DIRECT_CHANNEL_FADING,
    DIRECT_CHANNEL_SCALE,
    DIRECT_PATH_LOSS_EXPONENT,
    NOISE_POWER,
)


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
    parser.add_argument("--noise_power", type=float, default=NOISE_POWER)
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
    if args.noise_power <= 0:
        parser.error("--noise_power must be positive")
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
            args.noise_power,
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
            "noise_power": args.noise_power,
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
