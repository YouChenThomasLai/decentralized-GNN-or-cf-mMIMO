"""Check that E12 versus G2 contrasts use the same ordered batch clusters."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.summarize_e12_fixed_budget import REQUIRED_GATES, summarize


def test_ordered_paired_clusters_and_boundary():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        gates = {key: True for key in REQUIRED_GATES}
        gates.update(precision=False, centralized_monotone=False,
                     consensus_feasible=False, no_divergence=False)
        solver = {
            "eval_seed": 7, "samples": 4, "config_source": "run",
            "config": {"batch_size": 2, "L": 1, "AP": 2, "N": 3},
            "decision": {"numerical_gates": gates},
            "signaling": {
                "centralized": {"ap_to_cpu_values": 10, "cpu_to_ap_values": 4,
                                "online_rounds": 2},
                "distributed": {"ap_to_ap_values": 30, "ap_to_cpu_values": 3,
                                "online_rounds": 5},
            },
        }
        (root / "summary.json").write_text(json.dumps(solver), encoding="utf-8")
        np.savez(root / "paired_rates.npz",
                 primary__centralized_continuous=[3, 3, 5, 5],
                 primary__distributed_continuous=[2, 2, 4, 4],
                 primary__centralized_2bit=[2, 2, 4, 4],
                 primary__distributed_2bit=[1, 1, 3, 3])
        g2_path = root / "g2.json"
        g2_path.write_text(json.dumps({"g2": {
            "eval_seed": 7, "samples": 4, "evaluated_as_arch": "g2",
            "checkpoint": "iter150000.pt", "run_dir": "run",
            "unit_modulus_error": {"decentralized": 0.0},
        }}), encoding="utf-8")
        np.savez(root / "g2_paired.npz", g2__decentralized=[4, 2],
                 g2__decentralized_discrete=[3, 1])

        result = summarize(root, g2_path)
        # The two paired differences are +1 and -3; unpaired means hide this.
        contrast = result["phases"]["continuous"]["g2_minus_centralized"]
        assert contrast["mean"] == -1
        assert contrast["cluster_win_fraction"] == 0.5
        assert result["fixed_budget_eligible"]
        assert result["coordination"]["g2"]["values"] == 8
        assert result["coordination"]["centralized"]["fp32_bits"] == 448


if __name__ == "__main__":
    test_ordered_paired_clusters_and_boundary()
    print("[ok] ordered paired clusters and boundary")
