import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

from environment import DEFAULT_SQUARE_SIDE
from evaluate import validate_sample_count
from train import Trainer, seed_everything
from utils_return_indivial_rates import (
    DIRECT_CHANNEL_FADING,
    DIRECT_CHANNEL_SCALE,
    DIRECT_PATH_LOSS_EXPONENT,
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
        values = [metrics[key] for _, metrics in records]
        if not all(isinstance(value, (int, float)) for value in values):
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

    gaps = np.asarray([metrics["absolute_gap"] for _, metrics in records])
    dominant_sign_count = int(max(np.sum(gaps > 0), np.sum(gaps < 0)))
    summary["gap_dominant_sign_count"] = dominant_sign_count
    summary["gap_sign_consistent"] = dominant_sign_count >= min(4, len(gaps))
    with open(Path(exp_dir) / "final_summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
    np.save(Path(exp_dir) / "final_summary.npy", summary)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 no-RIS snapshot network-scaling trainer"
    )
    parser.add_argument("--scale", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--num_ap", type=int)
    parser.add_argument("--M", type=int, default=2, help="Antennas per AP")
    parser.add_argument("--K", type=int)
    parser.add_argument("--square_side", type=float)
    parser.add_argument(
        "--pmax_dbm", "--Pmax", dest="pmax_dbm", type=float, default=15.0
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topology_seed", type=int)
    parser.add_argument("--n_iter", type=int, default=2000)
    parser.add_argument("--noise_power", type=float, default=1e-12)
    parser.add_argument(
        "--out_dir",
        default="../results_snapshot_scaling/stage1_no_ris/scale1_A5_K8",
    )
    parser.add_argument("--test_sample_val", type=int, default=128)
    parser.add_argument("--test_sample_final", type=int, default=3200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.num_ap = args.num_ap or 5 * args.scale
    args.K = args.K or 8 * args.scale
    args.square_side = args.square_side or DEFAULT_SQUARE_SIDE * np.sqrt(
        args.scale
    )
    for name in ("num_ap", "M", "K", "batch_size", "runs", "n_iter"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.square_side <= 0 or args.noise_power <= 0:
        parser.error("--square_side and --noise_power must be positive")
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
        topology_seed = (
            effective_seed
            if args.topology_seed is None
            else args.topology_seed + run_offset
        )
        base_dir = exp_dir / f"seed{effective_seed}"
        if base_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {base_dir}")
        base_dir.mkdir(parents=True)
        write_status(base_dir, "running")
        seed_everything(effective_seed)
        print(f"[INFO] Run seed: {effective_seed}; topology seed: {topology_seed}")
        trainer = Trainer(
            args.M,
            args.K,
            args.batch_size,
            args.n_iter,
            args.pmax_dbm,
            args.noise_power,
            device=args.device,
            num_ap=args.num_ap,
            square_side=args.square_side,
            topology_seed=topology_seed,
        )
        array_dir = base_dir / "arrays"
        array_dir.mkdir()
        np.savez_compressed(
            array_dir / "topology.npz",
            ap_locations=trainer.dataloader.BS_Loc_array,
            square_side=args.square_side,
            topology_seed=topology_seed,
            wrap_around=True,
        )
        config = {
            "scale": args.scale,
            "effective_seed": effective_seed,
            "training_seed": effective_seed,
            "channel_seed": effective_seed,
            "topology_seed": topology_seed,
            "cli": shlex.join([sys.executable, *sys.argv]),
            "arguments": vars(args),
            "effective_device": str(trainer.device),
            "num_ap": args.num_ap,
            "num_ue": args.K,
            "square_side": args.square_side,
            "area": args.square_side**2,
            "ap_density": args.num_ap / args.square_side**2,
            "ue_density": args.K / args.square_side**2,
            "wrap_around": True,
            "association_threshold": trainer.associate_threshold,
            "pmax_w": trainer.pmax_w,
            "direct_channel_fading": DIRECT_CHANNEL_FADING,
            "direct_path_loss_exponent": DIRECT_PATH_LOSS_EXPONENT,
            "direct_channel_scale_exponent": DIRECT_CHANNEL_SCALE,
            "noise_power": args.noise_power,
            "rzf_regularization": "alpha = K_a * noise_power / Pmax",
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
