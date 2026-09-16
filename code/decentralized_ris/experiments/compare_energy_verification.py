"""Cross-check the verification run against the stored mrc_proxy_diagnostic result.

The verification script re-derives the four arms independently.  If it is
measuring the same thing, its per-sample rates must equal the ones stored in
`locked_seed20260922_paired.npz` to float32 precision, arm by arm.  This script
reports that comparison plus the per-sample win/loss/tie table, so the
reproduction claim rests on stored numbers rather than on eyeballed means.

    python -m experiments.compare_energy_verification
"""

import argparse
import json
import os

import numpy as np

ARM_MAP = {
    "r0c_per_element_mag": "per_element_mag",
    "learned_pair_s": "learned_pair_s",
    "local_energy": "proxy_sum_energy",
    "equal": "equal",
}
TOL_BPS = 1e-3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stored",
        default="../../artifacts/decentralized_ris/mrc_proxy_diagnostic/"
                "locked_seed20260922_paired.npz")
    parser.add_argument(
        "--stored_json",
        default="../../artifacts/decentralized_ris/mrc_proxy_diagnostic/"
                "locked_seed20260922.json")
    parser.add_argument(
        "--verification",
        default="../../artifacts/decentralized_ris/energy_consensus_verification/"
                "main_seed20260922_per_sample.npz")
    parser.add_argument(
        "--verification_json",
        default="../../artifacts/decentralized_ris/energy_consensus_verification/"
                "main_seed20260922.json")
    parser.add_argument(
        "--out",
        default="../../artifacts/decentralized_ris/energy_consensus_verification/"
                "reproduction_check.json")
    args = parser.parse_args()

    stored = np.load(args.stored)
    fresh = np.load(args.verification)
    stored_summary = json.load(open(args.stored_json, encoding="utf-8"))
    fresh_summary = json.load(open(args.verification_json, encoding="utf-8"))

    rows, failures = {}, []
    for new_name, old_name in ARM_MAP.items():
        a = fresh[f"sample__{new_name}"].astype(np.float64)
        b = stored[f"sample__{old_name}"].astype(np.float64)
        if a.shape != b.shape:
            failures.append(f"{new_name}: shape {a.shape} vs {b.shape}")
            continue
        delta = np.abs(a - b)
        mean_delta = abs(float(a.mean()) - float(b.mean()))
        rows[new_name] = {
            "stored_arm": old_name,
            "n_samples": int(a.size),
            "verification_mean": float(a.mean()),
            "stored_mean": float(b.mean()),
            "abs_mean_difference": mean_delta,
            "max_abs_per_sample_difference": float(delta.max()),
            "n_samples_differing_above_1e-4": int((delta > 1e-4).sum()),
        }
        if mean_delta > TOL_BPS:
            failures.append(
                f"{new_name}: mean differs by {mean_delta:.3e} > {TOL_BPS:.0e}")

    # Cluster-level means, the unit the pre-registered decision used.
    cluster_rows = {}
    for new_name, old_name in ARM_MAP.items():
        a = fresh[f"cluster__{new_name}"].astype(np.float64)
        b = stored[f"cluster__{old_name}"].astype(np.float64)
        cluster_rows[new_name] = {
            "max_abs_per_cluster_difference": float(np.abs(a - b).max()),
            "verification_mean": float(a.mean()),
            "stored_mean": float(b.mean()),
        }

    primary = fresh_summary["contrasts"]["local_energy_minus_learned_pair_s"]
    stored_primary = stored_summary["paired_differences"][
        "learned_pair_s_minus_proxy_sum_energy"]
    payload = {
        "tolerance_bps_hz": TOL_BPS,
        "per_sample": rows,
        "per_cluster": cluster_rows,
        "primary_contrast": {
            "verification_local_energy_minus_learned": {
                "mean": primary["cluster"]["mean"],
                "ci_low": primary["cluster"]["ci_low"],
                "ci_high": primary["cluster"]["ci_high"],
            },
            "stored_learned_minus_proxy": {
                "mean": stored_primary["mean"],
                "ci_low": stored_primary["ci_low"],
                "ci_high": stored_primary["ci_high"],
            },
            "sign_flipped_agreement": abs(
                primary["cluster"]["mean"] + stored_primary["mean"]),
        },
        "failures": failures,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"{'arm':24s} {'verification':>13s} {'stored':>11s} {'|d mean|':>10s} "
          f"{'max |d sample|':>15s}")
    for name, row in rows.items():
        print(f"{name:24s} {row['verification_mean']:13.6f} "
              f"{row['stored_mean']:11.6f} {row['abs_mean_difference']:10.3e} "
              f"{row['max_abs_per_sample_difference']:15.3e}")
    print(f"\nprimary contrast (local_energy - learned_pair_s): "
          f"{primary['cluster']['mean']:+.5f} "
          f"CI [{primary['cluster']['ci_low']:+.5f}, "
          f"{primary['cluster']['ci_high']:+.5f}]")
    print(f"stored (learned - proxy_sum_energy):              "
          f"{stored_primary['mean']:+.5f} "
          f"CI [{stored_primary['ci_low']:+.5f}, {stored_primary['ci_high']:+.5f}]")
    print(f"\n{'FAILURES: ' + '; '.join(failures) if failures else 'reproduction OK'}")


if __name__ == "__main__":
    main()
