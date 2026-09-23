"""Join E14 learned batches with the paired E12 fixed-budget solver samples."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.topology_screen import clustered


ELIGIBILITY = (
    "power_feasibility", "unit_modulus", "quantization_grid",
    "rate_path_equivalence", "locality", "no_unserved_column_leakage",
    "finite", "beats_random_phase",
)


def summarize(root, t0_solver):
    rows = {}
    for topology in ("t0", "t1", "t2"):
        learned_path = root / f"{topology}_learned.json"
        learned = json.loads(learned_path.read_text())
        learned_raw = np.load(learned_path.with_suffix(".npz"))
        solver_dir = t0_solver if topology == "t0" else root / f"{topology}_solver"
        solver = json.loads((solver_dir / "summary.json").read_text())
        solver_raw = np.load(solver_dir / "paired_rates.npz")
        batch_size = learned["config"]["batch_size"]
        if learned["samples"] != solver["samples"] or learned["eval_seed"] != solver["eval_seed"]:
            raise ValueError(f"{topology}: unpaired sample count or evaluation seed")
        if topology == "t0":
            e12_g2 = np.load(t0_solver / "g2_evaluation_paired.npz")
            reference = e12_g2[
                "graph-energy-g2-total150k-resume100k__decentralized"
            ]
            np.testing.assert_array_equal(learned_raw["g2_decentralized"], reference)
        gates = solver["decision"]["numerical_gates"]
        eligible = all(gates[key] for key in ELIGIBILITY)
        contrasts = {}
        for arm in ("centralized", "distributed"):
            for phase, key in (("continuous", "decentralized"),
                               ("2bit", "decentralized_discrete")):
                g2 = learned_raw[f"g2_{key}"]
                solved = solver_raw[f"primary__{arm}_{phase}"]
                if solved.size != g2.size * batch_size:
                    raise ValueError(f"{topology}: {arm} sample count differs")
                contrasts[f"g2_minus_{arm}_{phase}"] = clustered(
                    g2 - solved.reshape(-1, batch_size).mean(axis=1), 1
                )
        g2_bits = learned["csi_delivery"]["g2_total_bits_fp32"]["mean"]
        rows[topology.upper()] = {
            "learned": {name: learned["methods"][name]["rates"]
                        for name in ("g2", "r0")},
            "g2_minus_r0": learned["contrasts"],
            "solver": {arm: {phase: solver["metrics"][f"{arm}_{phase}"]
                             for phase in ("continuous", "2bit")}
                       for arm in ("centralized", "distributed")},
            "g2_minus_solver": contrasts,
            "solver_eligibility": {"pass": eligible,
                                   "gates": {key: gates[key] for key in ELIGIBILITY},
                                   "strict_gates": gates},
            "solver_diagnostics": {
                "max_precision_deviation": solver["controls"]["max_precision_deviation"],
                "max_consensus_residual": solver["solvers"]["distributed"]
                    ["consensus_residual"]["max"],
                "centralized_monotonicity_violation": solver["solvers"]
                    ["centralized"]["monotonicity_violation"],
                "distributed_capped_fraction": solver["solvers"]
                    ["distributed"]["realized_iterations"]["capped_fraction"],
            },
            "csi_delivery": learned["csi_delivery"],
            "coordination_bits": {
                "g2_with_raw_csi": g2_bits,
                "centralized": solver["signaling"]["centralized"]
                    ["total_online_bits_fp32"] - 32 * learned["config"]["L"]
                    * learned["config"]["N"],
                "distributed": solver["signaling"]["distributed"]
                    ["total_online_bits_fp32"] - 32 * learned["config"]["L"]
                    * learned["config"]["N"],
            },
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--t0_solver", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.root, args.t0_solver)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for topology, row in result.items():
        print(topology, {arm: row["solver"][arm]["continuous"]["mean"]
                         for arm in ("centralized", "distributed")},
              "eligible=", row["solver_eligibility"]["pass"])


if __name__ == "__main__":
    main()
