import argparse
import json
from pathlib import Path


SCALE_NAMES = {
    1: "scale1_A5_K8_L4",
    2: "scale2_A10_K16_L8",
    4: "scale4_A20_K32_L16",
}


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required gate artifact is missing: {path}")
    with open(path) as source:
        return json.load(source)


def require_run(root, scale, seed):
    run_dir = Path(root) / SCALE_NAMES[scale] / f"seed{seed}"
    status = read_json(run_dir / "status.json")
    checks = read_json(run_dir / "checks.json")
    config = read_json(run_dir / "config.json")
    paired_seeds = (
        config.get("training_seed"),
        config.get("topology_seed"),
        config.get("channel_seed"),
        config.get("evaluation_seed"),
    )
    if (
        status.get("status") != "complete"
        or not checks.get("passed")
        or paired_seeds != (seed, seed, seed, seed)
    ):
        raise RuntimeError(f"Run gate failed: {run_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Enforce the planned v2 phase ordering"
    )
    parser.add_argument(
        "phase",
        choices=(
            "smoke",
            "trend",
            "scale4_seed0",
            "scale4_seeds12",
            "extend_scales12",
            "extend_scale4",
        ),
    )
    parser.add_argument(
        "--results-root", default="../../results_snapshot_scaling_v2_bpp"
    )
    args = parser.parse_args()
    root = Path(args.results_root)
    topology_root = root / "topology_gate"
    topology = read_json(topology_root / "summary.json")
    topology_accepted = topology.get("status") == "PASS"
    if topology.get("status") == "RED_FLAG_REVIEW_REQUIRED":
        review = read_json(topology_root / "manual_review.json")
        topology_accepted = (
            review.get("decision") == "approved"
            and review.get("gate_version") == topology.get("gate_version")
        )
    if not topology_accepted:
        raise RuntimeError(
            f"Topology gate is {topology.get('status')} without approval"
        )
    required_topology_seeds = (
        {0}
        if args.phase == "smoke"
        else {0, 1, 2, 3, 4}
        if args.phase.startswith("extend")
        else {0, 1, 2}
    )
    for scale in SCALE_NAMES:
        available = set(
            topology["scales"][str(scale)]["per_topology_seed"]
        )
        if not {str(seed) for seed in required_topology_seeds} <= available:
            raise RuntimeError(
                f"Scale {scale} topology gate lacks frozen seeds "
                f"{sorted(required_topology_seeds)}"
            )
    if args.phase == "smoke":
        return

    smoke_root = root / "stage0_ris" / "smoke"
    for scale in (1, 2):
        require_run(smoke_root, scale, 0)
    if args.phase == "trend":
        return

    training_root = root / "stage0_ris"
    for scale in (1, 2):
        for seed in (0, 1, 2):
            require_run(training_root, scale, seed)
    if args.phase in ("scale4_seed0", "extend_scales12"):
        return
    require_run(training_root, 4, 0)
    if args.phase == "extend_scale4":
        require_run(training_root, 4, 1)
        require_run(training_root, 4, 2)


if __name__ == "__main__":
    main()
