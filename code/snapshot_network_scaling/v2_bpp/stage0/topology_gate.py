import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from data import MyDataLoader
from geometry import DEFAULT_SQUARE_SIDE, GATE_VERSION, pairwise_wrapped_distances
from trainer_2 import source_provenance
from utils_return_indivial_rates import CHANNEL_SCALE_EXPONENT


SCALE_COUNTS = {
    1: (5, 8, 4),
    2: (10, 16, 8),
    4: (20, 32, 16),
}


def summarize(values):
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q05": float(np.percentile(values, 5)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "q95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def scale_directory(root, scale, num_ap, num_ue, num_ris):
    return Path(root) / f"scale{scale}_A{num_ap}_K{num_ue}_L{num_ris}"


def collect_scale(args, scale):
    num_ap, num_ue, num_ris = SCALE_COUNTS[scale]
    square_side = DEFAULT_SQUARE_SIDE * np.sqrt(scale)
    output_dir = scale_directory(
        args.output_root, scale, num_ap, num_ue, num_ris
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite topology gate: {output_dir}")
    output_dir.mkdir(parents=True)
    with open(output_dir / "status.json", "w") as status_file:
        json.dump({"status": "running"}, status_file, indent=2)

    names = (
        "user_locations",
        "association_mask",
        "nearest_ap_distance",
        "nearest_ris_distance",
        "strongest_received_power",
        "serving_ap_count",
        "associated_ue_count",
        "total_associated_links",
        "visible_links_per_ap",
        "local_to_global_visibility_ratio",
        "ap_pair_shared_ue",
        "connected_component_count",
        "largest_component_fraction",
        "bipartite_fully_connected",
        "visible_ap_ue_ris_feature_blocks",
        "global_ris_proposal_count_centralized",
        "global_ris_proposal_count_decentralized",
    )
    arrays = {name: [] for name in names}
    ap_locations = []
    ris_locations = []
    ap_nearest_neighbor = []
    ap_ris_distances = []
    topology_seed_per_snapshot = []
    channel_seed_per_snapshot = []
    contract_checks = []

    batches_per_seed = args.snapshots_per_seed // args.batch_size
    for topology_seed in args.topology_seeds:
        channel_seed = args.channel_seed_base + scale * 100_000 + topology_seed
        loader = MyDataLoader(
            args.M,
            args.N,
            num_ris,
            args.batch_size,
            num_ap=num_ap,
            square_side=square_side,
            topology_seed=topology_seed,
            channel_seed=channel_seed,
            record_history=False,
        )
        loader.BS_RIS_association()
        ap_locations.append(loader.BS_Loc_array)
        ris_locations.append(loader.RIS_Loc_array)
        ap_distances = pairwise_wrapped_distances(
            loader.BS_Loc_array, loader.BS_Loc_array, square_side
        )
        np.fill_diagonal(ap_distances, np.inf)
        ap_nearest_neighbor.append(ap_distances.min(axis=1))
        ap_ris_distances.append(
            pairwise_wrapped_distances(
                loader.BS_Loc_array, loader.RIS_Loc_array, square_side
            )
        )

        for _ in range(batches_per_seed):
            loader.BS_user_association(
                num_ue, args.association_threshold, args.association_threshold
            )
            topology = loader.get_topology_batch()
            for name in names:
                arrays[name].append(topology[name])
            topology_seed_per_snapshot.append(
                np.full(args.batch_size, topology_seed, dtype=int)
            )
            channel_seed_per_snapshot.append(
                np.full(args.batch_size, channel_seed, dtype=int)
            )
            contract_checks.append(loader.validate_topology())

    arrays = {
        name: np.concatenate(values, axis=0)
        for name, values in arrays.items()
    }
    arrays.update(
        {
            "ap_locations": np.asarray(ap_locations),
            "ris_locations": np.asarray(ris_locations),
            "ap_nearest_neighbor_distance": np.concatenate(
                ap_nearest_neighbor
            ),
            "ap_ris_wrapped_distances": np.asarray(ap_ris_distances),
            "topology_seed_per_snapshot": np.concatenate(
                topology_seed_per_snapshot
            ),
            "channel_seed_per_snapshot": np.concatenate(
                channel_seed_per_snapshot
            ),
        }
    )
    np.savez_compressed(output_dir / "topology_gate_samples.npz", **arrays)

    area = square_side**2
    densities = {
        "ap": num_ap / area,
        "ue": num_ue / area,
        "ris": num_ris / area,
    }
    design_densities = {
        "ap": 5 / DEFAULT_SQUARE_SIDE**2,
        "ue": 8 / DEFAULT_SQUARE_SIDE**2,
        "ris": 4 / DEFAULT_SQUARE_SIDE**2,
    }
    per_topology_seed = {}
    ap_nearest_by_seed = arrays["ap_nearest_neighbor_distance"].reshape(
        len(args.topology_seeds), num_ap
    )
    for seed_index, topology_seed in enumerate(args.topology_seeds):
        selected = arrays["topology_seed_per_snapshot"] == topology_seed
        per_topology_seed[str(topology_seed)] = {
            "ap_nearest_neighbor_median": float(
                np.median(ap_nearest_by_seed[seed_index])
            ),
            "ue_nearest_ap_median": float(
                np.median(arrays["nearest_ap_distance"][selected])
            ),
            "serving_ap_count_mean": float(
                arrays["serving_ap_count"][selected].mean()
            ),
            "associated_ue_count_mean": float(
                arrays["associated_ue_count"][selected].mean()
            ),
            "visible_links_per_ap_mean": float(
                arrays["visible_links_per_ap"][selected].mean()
            ),
            "ap_pair_shared_ue_fraction": float(
                arrays["ap_pair_shared_ue"][selected].mean()
            ),
            "strongest_received_power_median": float(
                np.median(arrays["strongest_received_power"][selected])
            ),
        }
    metrics = {
        "scale": scale,
        "num_ap": num_ap,
        "num_ue": num_ue,
        "num_ris": num_ris,
        "square_side": square_side,
        "area": area,
        "snapshot_count": int(arrays["association_mask"].shape[0]),
        "topology_seed_count": len(args.topology_seeds),
        "densities": densities,
        "design_densities": design_densities,
        "per_topology_seed": per_topology_seed,
        "ap_nearest_neighbor_distance": summarize(
            arrays["ap_nearest_neighbor_distance"]
        ),
        "ue_nearest_ap_distance": summarize(arrays["nearest_ap_distance"]),
        "ue_nearest_ris_distance": summarize(arrays["nearest_ris_distance"]),
        "ap_ris_wrapped_distance": summarize(
            arrays["ap_ris_wrapped_distances"]
        ),
        "strongest_received_power": summarize(
            arrays["strongest_received_power"]
        ),
        "serving_ap_count": summarize(arrays["serving_ap_count"]),
        "associated_ue_count": summarize(arrays["associated_ue_count"]),
        "total_associated_links": summarize(
            arrays["total_associated_links"]
        ),
        "visible_links_per_ap": summarize(arrays["visible_links_per_ap"]),
        "local_to_global_visibility_ratio": summarize(
            arrays["local_to_global_visibility_ratio"]
        ),
        "ap_pair_shared_ue_fraction": float(
            arrays["ap_pair_shared_ue"].mean()
        ),
        "connected_component_count": summarize(
            arrays["connected_component_count"]
        ),
        "bipartite_fully_connected_fraction": float(
            arrays["bipartite_fully_connected"].mean()
        ),
        "visible_ap_ue_ris_feature_blocks": summarize(
            arrays["visible_ap_ue_ris_feature_blocks"]
        ),
        "global_ris_proposal_count_centralized": summarize(
            arrays["global_ris_proposal_count_centralized"]
        ),
        "global_ris_proposal_count_decentralized": summarize(
            arrays["global_ris_proposal_count_decentralized"]
        ),
        "computed_checks": {
            "densities": all(
                np.isclose(densities[name], design_densities[name])
                for name in densities
            ),
            "numerical": all(
                np.isfinite(values).all()
                for values in arrays.values()
                if np.issubdtype(np.asarray(values).dtype, np.number)
            ),
            "each_ue_has_serving_ap": bool(
                arrays["association_mask"].any(axis=2).all()
            ),
            "implementation_contracts": all(
                check["passed"] for check in contract_checks
            ),
        },
    }
    with open(output_dir / "metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, sort_keys=True)
    config = {
        "gate_version": GATE_VERSION,
        "cli": shlex.join([sys.executable, *sys.argv]),
        "topology_seeds": args.topology_seeds,
        "channel_seed_base": args.channel_seed_base,
        "snapshots_per_seed": args.snapshots_per_seed,
        "association_threshold": args.association_threshold,
        "channel_scale_exponent": CHANNEL_SCALE_EXPONENT,
        "performance_metrics_computed": False,
        "wrap_around": True,
        "los_angle_model": "existing stochastic angles (no geometric direction input)",
        **source_provenance(),
    }
    with open(output_dir / "config.json", "w") as config_file:
        json.dump(config, config_file, indent=2, sort_keys=True)
    with open(output_dir / "status.json", "w") as status_file:
        json.dump({"status": "complete"}, status_file, indent=2)
    return metrics


def relative_difference(value, baseline):
    return abs(value / baseline - 1) if baseline else float("inf")


def evaluate_gate(scale_metrics):
    baseline = scale_metrics[1]
    criteria = {}

    criteria["computed_densities"] = all(
        metrics["computed_checks"]["densities"]
        for metrics in scale_metrics.values()
    )
    comparisons = {
        "ap_nearest_neighbor_stationary": (
            "ap_nearest_neighbor_distance",
            "median",
            0.10,
        ),
        "ue_nearest_ap_stationary": (
            "ue_nearest_ap_distance",
            "median",
            0.10,
        ),
        "serving_ap_count_stationary": ("serving_ap_count", "mean", 0.15),
        "associated_ue_count_stationary": (
            "associated_ue_count",
            "mean",
            0.15,
        ),
        "visible_links_per_ap_stationary": (
            "visible_links_per_ap",
            "mean",
            0.20,
        ),
    }
    comparison_details = {}
    for criterion, (metric, statistic, tolerance) in comparisons.items():
        baseline_value = baseline[metric][statistic]
        differences = {
            str(scale): relative_difference(
                values[metric][statistic], baseline_value
            )
            for scale, values in scale_metrics.items()
        }
        criteria[criterion] = all(
            difference <= tolerance for difference in differences.values()
        )
        comparison_details[criterion] = {
            "tolerance": tolerance,
            "relative_difference_from_scale1": differences,
        }

    overlaps = [
        scale_metrics[scale]["ap_pair_shared_ue_fraction"]
        for scale in sorted(scale_metrics)
    ]
    monotonic_increase = all(
        right >= left for left, right in zip(overlaps, overlaps[1:])
    ) and any(right > left for left, right in zip(overlaps, overlaps[1:]))
    criteria["ap_pair_overlap_not_monotonic_increase"] = not monotonic_increase
    criteria["ap_pair_overlap_not_near_100_percent"] = max(overlaps) < 0.95

    criteria["numerical"] = all(
        all(metrics["computed_checks"].values())
        for metrics in scale_metrics.values()
    )
    power_quantiles = ("q05", "q25", "median", "q75", "q95")
    power_differences = {
        str(scale): {
            quantile: relative_difference(
                metrics["strongest_received_power"][quantile],
                baseline["strongest_received_power"][quantile],
            )
            for quantile in power_quantiles
        }
        for scale, metrics in scale_metrics.items()
    }
    power_red_flag = any(
        difference > 0.15
        for differences in power_differences.values()
        for difference in differences.values()
    )
    mandatory_pass = all(criteria.values())
    status = (
        "FAIL"
        if not mandatory_pass
        else "RED_FLAG_REVIEW_REQUIRED"
        if power_red_flag
        else "PASS"
    )
    return {
        "gate_version": GATE_VERSION,
        "status": status,
        "mandatory_pass": mandatory_pass,
        "criteria": criteria,
        "comparison_details": comparison_details,
        "ap_pair_overlap_by_scale": {
            str(scale): value
            for scale, value in zip(sorted(scale_metrics), overlaps)
        },
        "strongest_power_red_flag": power_red_flag,
        "strongest_power_relative_differences": power_differences,
        "near_100_percent_threshold": 0.95,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mandatory topology-only gate for Stage 0 v2 BPP"
    )
    parser.add_argument(
        "--output-root",
        default="../../results_snapshot_scaling_v2_bpp/topology_gate",
    )
    parser.add_argument(
        "--topology-seeds", type=int, nargs="+", default=list(range(10))
    )
    parser.add_argument("--channel-seed-base", type=int, default=1_000_000)
    parser.add_argument("--snapshots-per-seed", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--association-threshold", type=float, default=0.1)
    parser.add_argument("--require-pass", type=Path)
    parser.add_argument("--approve-red-flag", type=Path)
    parser.add_argument("--review-reason")
    args = parser.parse_args()
    if args.require_pass is not None or args.approve_red_flag is not None:
        return args
    if len(args.topology_seeds) != len(set(args.topology_seeds)):
        parser.error("Topology seeds must be distinct")
    if len(args.topology_seeds) < 10:
        parser.error("Topology gate requires at least 10 distinct topology seeds")
    if (
        args.batch_size <= 0
        or args.snapshots_per_seed <= 0
        or args.snapshots_per_seed % args.batch_size
        or args.snapshots_per_seed * len(args.topology_seeds) < 9600
    ):
        parser.error(
            "Use a batch-aligned snapshots-per-seed totaling at least 9600 "
            "snapshots per scale"
        )
    if (
        args.M != 2
        or args.N != 30
        or not np.isclose(args.association_threshold, 0.1)
    ):
        parser.error("The plan freezes M=2, N=30, and threshold=0.1")
    return args


def main():
    args = parse_args()
    if args.approve_red_flag is not None:
        with open(args.approve_red_flag) as summary_file:
            summary = json.load(summary_file)
        if summary.get("status") != "RED_FLAG_REVIEW_REQUIRED":
            raise RuntimeError("Only a RED_FLAG_REVIEW_REQUIRED gate needs review")
        if not args.review_reason:
            raise ValueError("--review-reason is required")
        review_path = args.approve_red_flag.with_name("manual_review.json")
        if review_path.exists():
            raise FileExistsError(f"Refusing to overwrite review: {review_path}")
        with open(review_path, "w") as review_file:
            json.dump(
                {
                    "decision": "approved",
                    "automatic_status": summary["status"],
                    "gate_version": summary["gate_version"],
                    "reason": args.review_reason,
                    "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                review_file,
                indent=2,
                sort_keys=True,
            )
        print(f"Recorded topology red-flag approval: {review_path}")
        return
    if args.require_pass is not None:
        with open(args.require_pass) as summary_file:
            summary = json.load(summary_file)
        approved = summary.get("status") == "PASS"
        review_path = args.require_pass.with_name("manual_review.json")
        if summary.get("status") == "RED_FLAG_REVIEW_REQUIRED" and review_path.is_file():
            with open(review_path) as review_file:
                review = json.load(review_file)
            approved = (
                review.get("decision") == "approved"
                and review.get("gate_version") == summary.get("gate_version")
            )
        if not approved:
            raise RuntimeError(
                f"Topology gate is {summary.get('status')} without approval"
            )
        print(f"Topology gate accepted: {args.require_pass}")
        return

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    metrics = {}
    for scale, counts in SCALE_COUNTS.items():
        try:
            metrics[scale] = collect_scale(args, scale)
        except Exception as error:
            output_dir = scale_directory(
                args.output_root, scale, *counts
            )
            if output_dir.exists():
                with open(output_dir / "status.json", "w") as status_file:
                    json.dump(
                        {
                            "status": "failed",
                            "error": str(error),
                            "error_type": type(error).__name__,
                        },
                        status_file,
                        indent=2,
                        sort_keys=True,
                    )
            raise
    summary = evaluate_gate(metrics)
    summary["scales"] = {str(scale): value for scale, value in metrics.items()}
    summary_path = Path(args.output_root) / "summary.json"
    with open(summary_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
    print(f"Topology gate {summary['status']}: {summary_path}")


if __name__ == "__main__":
    main()
