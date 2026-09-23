"""Pair a fresh E12 fixed-budget run with frozen G2 on the same channel seed."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.model_based_joint import clustered, paired
from variants import canonical_arch


REQUIRED_GATES = (
    "power_feasibility", "unit_modulus", "quantization_grid",
    "rate_path_equivalence", "locality", "no_unserved_column_leakage",
    "finite", "beats_random_phase",
)


def summarize(solver_dir, g2_json):
    with (solver_dir / "summary.json").open(encoding="utf-8") as handle:
        solver = json.load(handle)
    with g2_json.open(encoding="utf-8") as handle:
        g2 = json.load(handle)
    if len(g2) != 1:
        raise ValueError("expected exactly one G2 checkpoint")
    tag, g2_record = next(iter(g2.items()))
    if solver["eval_seed"] != g2_record["eval_seed"] or solver["samples"] != g2_record["samples"]:
        raise ValueError("G2 and E12 must have the same evaluation seed and sample count")
    if canonical_arch(g2_record["evaluated_as_arch"]) != "g2":
        raise ValueError("the learned arm must be the frozen G2 checkpoint")
    if (Path(solver["config_source"]).resolve() != Path(g2_record["run_dir"]).resolve()
            or g2_record["checkpoint"] != "iter150000.pt"):
        raise ValueError("solver config and G2-150k checkpoint do not match")
    if g2_record["unit_modulus_error"]["decentralized"] > 1e-5:
        raise ValueError("G2 phase is not unit modulus")
    batch_size = solver["config"]["batch_size"]
    clusters = solver["samples"] // batch_size
    with np.load(solver_dir / "paired_rates.npz") as rates, np.load(
        g2_json.with_name(g2_json.stem + "_paired.npz")
    ) as g2_rates:
        report = {"seed": solver["eval_seed"], "samples": solver["samples"],
                  "cluster_size": batch_size,
                  "g2_checkpoint": str(Path(g2_record["run_dir"]) / "checkpoints" /
                                       g2_record["checkpoint"]),
                  "phases": {}}
        for phase, g2_key in (("continuous", "decentralized"),
                              ("2bit", "decentralized_discrete")):
            g2_batch = g2_rates[f"{tag}__{g2_key}"]
            if g2_batch.size != clusters or not np.isfinite(g2_batch).all():
                raise ValueError("unexpected or nonfinite G2 batch rates")
            arms = {
                arm: rates[f"primary__{arm}_{phase}"].reshape(clusters, batch_size).mean(axis=1)
                for arm in ("centralized", "distributed")
            }
            report["phases"][phase] = {
                "g2": clustered(g2_batch, 1),
                **{arm: clustered(values, 1) for arm, values in arms.items()},
                "g2_minus_centralized": paired(g2_batch, arms["centralized"], 1),
                "g2_minus_distributed": paired(g2_batch, arms["distributed"], 1),
                "centralized_minus_distributed": paired(
                    arms["centralized"], arms["distributed"], 1
                ),
            }

    old_gates = solver["decision"]["numerical_gates"]
    report["fixed_budget_gates"] = {key: old_gates[key] for key in REQUIRED_GATES}
    report["fixed_budget_eligible"] = all(report["fixed_budget_gates"].values())
    report["original_convergence_diagnostics"] = {
        key: old_gates[key] for key in
        ("precision", "centralized_monotone", "consensus_feasible", "no_divergence")
    }
    ledger = solver["signaling"]
    config = solver["config"]
    g2_values = config["L"] * config["AP"] * (config["N"] + 1)
    central = ledger["centralized"]
    distributed = ledger["distributed"]
    coordination = {
        "g2": {"values": g2_values, "rounds": 1},
        "centralized": {
            "values": central["ap_to_cpu_values"] + central["cpu_to_ap_values"],
            "rounds": central["online_rounds"],
        },
        "distributed": {
            "values": distributed["ap_to_ap_values"] + distributed["ap_to_cpu_values"],
            "rounds": distributed["online_rounds"],
        },
    }
    for entry in coordination.values():
        entry["fp32_bits"] = 32 * entry["values"]
    report["coordination"] = coordination
    report["cpu_to_ris_actuation_values_common"] = config["L"] * config["N"]
    report["ue_to_ap_csi_acquisition"] = "excluded for all arms"
    report["interpretation"] = "fixed-budget feasible actions; no convergence or upper-bound claim"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("solver_dir", type=Path)
    parser.add_argument("g2_json", type=Path)
    args = parser.parse_args()
    report = summarize(args.solver_dir, args.g2_json)
    output = args.solver_dir / "comparison.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"eligible": report["fixed_budget_eligible"],
                      "continuous": report["phases"]["continuous"]}, indent=2))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
