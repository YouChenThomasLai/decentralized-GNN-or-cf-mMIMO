#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from environment import TOPOLOGY_TYPE, WRAP_AROUND
from utils_return_indivial_rates import (
    DIRECT_CHANNEL_FADING,
    DIRECT_CHANNEL_SCALE,
    DIRECT_PATH_LOSS_EXPONENT,
    HEIGHT_DIFFERENCE,
    SQUARE_SIDE,
    sample_square_bpp,
    wrapped_3d_distance,
    wrapped_displacement,
    wrapped_horizontal_distance,
)


NUM_AP = 5
NUM_USERS = 8
MIN_TOPOLOGY_SEEDS = 200
MIN_UE_DROPS = 100


def summarize(values):
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q05": float(np.percentile(values, 5)),
        "q95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def association_component_count(mask):
    num_users, num_ap = mask.shape
    parent = np.arange(num_ap + num_users)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for user, ap in np.argwhere(mask):
        left, right = find(ap), find(num_ap + user)
        if left != right:
            parent[right] = left
    return len({find(node) for node in range(len(parent))})


def topology_metrics(seed, ue_drops, threshold):
    # Match trainer_2.py's np.random.seed(effective_seed) convention exactly.
    rng = np.random.RandomState(seed)
    ap_locations = sample_square_bpp(NUM_AP, rng=rng)
    user_locations = sample_square_bpp(
        ue_drops * NUM_USERS, rng=rng
    ).reshape(ue_drops, NUM_USERS, 2)

    ap_distances = wrapped_horizontal_distance(
        ap_locations[:, None, :], ap_locations[None, :, :]
    )
    np.fill_diagonal(ap_distances, np.inf)
    horizontal = wrapped_horizontal_distance(
        user_locations[:, :, None, :], ap_locations
    )
    distance_3d = wrapped_3d_distance(
        user_locations[:, :, None, :], ap_locations
    )
    large_scale_amplitude = (
        DIRECT_CHANNEL_FADING
        * distance_3d ** (-DIRECT_PATH_LOSS_EXPONENT)
        / 10**DIRECT_CHANNEL_SCALE
    )
    received_power = large_scale_amplitude**2
    association = received_power >= received_power.max(
        axis=2, keepdims=True
    ) * threshold
    serving_ap_count = association.sum(axis=2)
    ap_load = association.sum(axis=1)
    pair_shared = np.stack(
        [
            np.any(
                association[:, :, first] & association[:, :, second],
                axis=1,
            )
            for first in range(NUM_AP)
            for second in range(first + 1, NUM_AP)
        ],
        axis=1,
    )
    component_count = np.asarray(
        [association_component_count(mask) for mask in association]
    )

    metrics = {
        "ap_nearest_neighbor_median_m": float(
            np.median(ap_distances.min(axis=1))
        ),
        "ue_nearest_ap_median_m": float(np.median(horizontal.min(axis=2))),
        "serving_ap_count_mean": float(serving_ap_count.mean()),
        "ap_load_mean": float(ap_load.mean()),
        "ap_pair_shared_ue_fraction": float(pair_shared.mean()),
        "association_graph_connected_fraction": float(
            np.mean(component_count == 1)
        ),
        "minimum_horizontal_distance_m": float(horizontal.min()),
        "minimum_3d_distance_m": float(distance_3d.min()),
    }
    arrays = {
        "ap_locations": ap_locations,
        "ap_nearest_neighbor_distance": ap_distances.min(axis=1),
        "ue_nearest_ap_distance": horizontal.min(axis=2),
        "serving_ap_count": serving_ap_count,
        "ap_load": ap_load,
        "ap_pair_shared_ue": pair_shared,
        "association_component_count": component_count,
        "minimum_horizontal_distance": horizontal.min(),
        "minimum_3d_distance": distance_3d.min(),
        "all_coordinates_in_bounds": bool(
            np.all((-SQUARE_SIDE / 2 <= ap_locations))
            and np.all(ap_locations < SQUARE_SIDE / 2)
            and np.all((-SQUARE_SIDE / 2 <= user_locations))
            and np.all(user_locations < SQUARE_SIDE / 2)
        ),
        "each_ue_has_serving_ap": bool(association.any(axis=2).all()),
    }
    return metrics, arrays


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1C training-free BPP topology qualification gate"
    )
    parser.add_argument(
        "--output_dir", default="results_stage1c_topology_gate"
    )
    parser.add_argument("--topology_seed_start", type=int, default=0)
    parser.add_argument(
        "--topology_seeds", type=int, default=MIN_TOPOLOGY_SEEDS
    )
    parser.add_argument(
        "--ue_drops_per_topology", type=int, default=MIN_UE_DROPS
    )
    parser.add_argument("--association_threshold", type=float, default=0.1)
    args = parser.parse_args()

    if args.topology_seeds < MIN_TOPOLOGY_SEEDS:
        parser.error(f"--topology_seeds must be at least {MIN_TOPOLOGY_SEEDS}")
    if args.ue_drops_per_topology < MIN_UE_DROPS:
        parser.error(
            f"--ue_drops_per_topology must be at least {MIN_UE_DROPS}"
        )
    if not 0 < args.association_threshold <= 1:
        parser.error("--association_threshold must be in (0, 1]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    seeds = range(
        args.topology_seed_start,
        args.topology_seed_start + args.topology_seeds,
    )
    per_topology = {}
    collected = {
        "ap_locations": [],
        "ap_nearest_neighbor_distance": [],
        "ue_nearest_ap_distance": [],
        "serving_ap_count": [],
        "ap_load": [],
        "ap_pair_shared_ue": [],
        "association_component_count": [],
        "minimum_horizontal_distance": [],
        "minimum_3d_distance": [],
    }
    coordinate_checks = []
    association_checks = []
    first_arrays = None
    for seed in seeds:
        metrics, arrays = topology_metrics(
            seed, args.ue_drops_per_topology, args.association_threshold
        )
        per_topology[str(seed)] = metrics
        for name in collected:
            collected[name].append(arrays[name])
        coordinate_checks.append(arrays["all_coordinates_in_bounds"])
        association_checks.append(arrays["each_ue_has_serving_ap"])
        if first_arrays is None:
            first_arrays = arrays

    collected = {
        name: np.asarray(values) for name, values in collected.items()
    }
    repeated = topology_metrics(
        args.topology_seed_start,
        args.ue_drops_per_topology,
        args.association_threshold,
    )[1]
    edge_displacement = wrapped_displacement(
        np.array([99.0, 0.0]), np.array([-99.0, 0.0])
    )
    checks = {
        "coordinate_bounds": all(coordinate_checks),
        "toroidal_displacement": bool(
            np.array_equal(edge_displacement, np.array([2.0, 0.0]))
        ),
        "fixed_counts": bool(
            collected["ap_locations"].shape
            == (args.topology_seeds, NUM_AP, 2)
            and collected["ue_nearest_ap_distance"].shape
            == (
                args.topology_seeds,
                args.ue_drops_per_topology,
                NUM_USERS,
            )
        ),
        "reproducibility": all(
            np.array_equal(first_arrays[name], repeated[name])
            for name in collected
        ),
        "finite_metrics": all(
            np.isfinite(values).all() for values in collected.values()
        ),
        "height_lower_bound": bool(
            collected["minimum_3d_distance"].min() >= HEIGHT_DIFFERENCE
        ),
        "each_ue_has_serving_ap": all(association_checks),
    }
    summary = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "topology_type": TOPOLOGY_TYPE,
        "square_side_m": SQUARE_SIDE,
        "coordinate_bounds_m": [-SQUARE_SIDE / 2, SQUARE_SIDE / 2],
        "wrap_around": WRAP_AROUND,
        "height_difference_m": HEIGHT_DIFFERENCE,
        "num_ap": NUM_AP,
        "num_users": NUM_USERS,
        "topology_seed_start": args.topology_seed_start,
        "topology_seed_count": args.topology_seeds,
        "ue_drops_per_topology": args.ue_drops_per_topology,
        "association_threshold": args.association_threshold,
        "lsf_received_power": (
            "(direct_channel_fading * distance_3d**"
            "(-direct_path_loss_exponent) / 10**direct_channel_scale)**2"
        ),
        "checks": checks,
        "ap_nearest_neighbor_distance_m": summarize(
            collected["ap_nearest_neighbor_distance"]
        ),
        "ue_nearest_ap_distance_m": summarize(
            collected["ue_nearest_ap_distance"]
        ),
        "serving_ap_count": summarize(collected["serving_ap_count"]),
        "ap_load": summarize(collected["ap_load"]),
        "ap_pair_shared_ue_fraction": float(
            collected["ap_pair_shared_ue"].mean()
        ),
        "association_graph_connected_fraction": float(
            np.mean(collected["association_component_count"] == 1)
        ),
        "minimum_horizontal_distance_m": float(
            collected["minimum_horizontal_distance"].min()
        ),
        "minimum_3d_distance_m": float(
            collected["minimum_3d_distance"].min()
        ),
        "per_topology_seed": per_topology,
    }
    np.savez_compressed(
        output_dir / "topology_gate_samples.npz", **collected
    )
    with open(output_dir / "summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
    public_summary = {
        key: value
        for key, value in summary.items()
        if key != "per_topology_seed"
    }
    print(json.dumps(public_summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
