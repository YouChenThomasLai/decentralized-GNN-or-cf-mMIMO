"""Aggregate the energy-consensus verification across operating points.

Emits one markdown table per question the verification had to answer, so the
report quotes stored numbers rather than transcribed ones.

    python -m experiments.summarize_energy_verification
"""

import argparse
import json
import os

ORDER = ("main", "pmax5", "pmax25", "assoc30")
CORE = ("r0c_per_element_mag", "learned_pair_s", "local_energy", "equal")
SCALE_EQUIVALENT = ("energy_x1e6", "energy_x1e-6", "energy_norm_over_ap",
                    "energy_norm_by_max")
MARGIN = 0.2


def load(directory, seed):
    out = {}
    for name in ORDER:
        path = os.path.join(directory, f"{name}_seed{seed}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                out[name] = json.load(handle)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="../../artifacts/decentralized_ris/energy_consensus_verification")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    runs = load(args.dir, args.seed)
    lines = []

    def emit(text=""):
        lines.append(text)
        print(text)

    emit("### Arms by operating point (decentralized continuous sum rate, bps/Hz)")
    emit()
    emit("| operating point | " + " | ".join(CORE) + " |")
    emit("|---|" + "---:|" * len(CORE))
    for name, data in runs.items():
        cfg = data["config"]
        label = (f"`{name}` (P={cfg['pmax_dbm']:.0f} dBm, "
                 f"ratio={cfg['assoc_threshold']})")
        cells = [f"{data['arms_cluster'][arm]['mean']:.4f}" for arm in CORE]
        emit(f"| {label} | " + " | ".join(cells) + " |")

    emit()
    emit("### Primary contrast: `local_energy` − `learned_pair_s`")
    emit()
    emit("| operating point | mean ± SEM | 95% CI (cluster) | 95% CI (sample) | "
         "per-sample W/L/T | non-inferior (δ=0.2) | superior |")
    emit("|---|---:|---|---|---:|:--:|:--:|")
    for name, data in runs.items():
        entry = data["contrasts"]["local_energy_minus_learned_pair_s"]
        c, s = entry["cluster"], entry["sample"]
        w = entry["win_loss_tie_per_sample"]
        non_inferior = c["ci_low"] > -MARGIN
        superior = c["ci_low"] > 0.0
        emit(f"| `{name}` | {c['mean']:+.4f} ± {c['sem']:.4f} | "
             f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
             f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] | "
             f"{w['wins']}/{w['losses']}/{w['ties']} | "
             f"{'yes' if non_inferior else 'NO'} | {'yes' if superior else 'no'} |")

    emit()
    emit("### Normalization probes (is the advantage an absolute-scale artifact?)")
    emit()
    emit("| operating point | `local_energy` | ×10⁶ | ×10⁻⁶ | E/Σ_l E | E/max E | "
         "max \\|Δrate\\| | float64 \\|Δθ\\| |")
    emit("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, data in runs.items():
        arms = data["arms_cluster"]
        cells = [f"{arms[a]['mean']:.5f}" for a in SCALE_EQUIVALENT]
        emit(f"| `{name}` | {arms['local_energy']['mean']:.5f} | "
             + " | ".join(cells)
             + f" | {data['controls']['energy_scale_invariance_rate_bps']:.2e}"
             + f" | {data['controls']['energy_scale_invariance_float64']:.2e} |")

    emit()
    emit("### Alignment vs sharpness, and the relative-weight probe")
    emit()
    emit("| operating point | E/Σ_r E | rank only | AP-permuted | "
         "E values by learned order | `learned_pair_s` |")
    emit("|---|---:|---:|---:|---:|---:|")
    for name, data in runs.items():
        arms = data["arms_cluster"]
        emit(f"| `{name}` | {arms['energy_norm_over_ris']['mean']:.4f} | "
             f"{arms['energy_rank_over_ap']['mean']:.4f} | "
             f"{arms['energy_perm_ap']['mean']:.4f} | "
             f"{arms['energy_values_by_learned_order']['mean']:.4f} | "
             f"{arms['learned_pair_s']['mean']:.4f} |")

    emit()
    emit("### Spearman(s, E) and numerical health")
    emit()
    emit("| operating point | ρ across AP within RIS | ρ across 20 pairs | "
         "NaN (any arm) | min \\|resultant\\| | projection fallbacks | "
         "max \\|\\|θ\\|−1\\| |")
    emit("|---|---:|---:|---:|---:|---:|---:|")
    for name, data in runs.items():
        corr = data["correlations"]
        health = data["numerical_health"]
        nans = sum(row["nan_in_weights"] + row["nan_in_theta"] + row["nan_in_rate"]
                   for row in health.values())
        min_res = min(row["min_resultant_norm"] for row in health.values())
        fallbacks = sum(row["n_projection_fallback"] for row in health.values())
        unit = max(row["max_unit_modulus_error"] for row in health.values())
        a = corr["spearman_learned_s_vs_energy_across_ap_within_ris"]
        b = corr["spearman_learned_s_vs_energy_across_ap_ris_pairs"]
        emit(f"| `{name}` | {a['mean']:+.4f} ± {a['sem']:.4f} | "
             f"{b['mean']:+.4f} ± {b['sem']:.4f} | {nans} | {min_res:.2e} | "
             f"{fallbacks} | {unit:.1e} |")

    emit()
    emit("### Controls")
    emit()
    emit("| operating point | failures | reimpl vs reference | learned arm vs "
         "deployed | equal arm vs R1-shared | E vs raw recompute | "
         "E under other-AP corruption | determinism |")
    emit("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, data in runs.items():
        c, a = data["controls"], data["locality_audit"]
        emit(f"| `{name}` | {len(data['control_failures'])} | "
             f"{c['reimplemented_vs_reference_consensus']:.1e} | "
             f"{c['learned_arm_vs_deployed_r1_ap_ris_mag']:.1e} | "
             f"{c['equal_arm_vs_deployed_r1_shared']:.1e} | "
             f"{a['max_rel_error_vs_raw_channel_recompute']:.1e} | "
             f"{a['max_abs_change_when_other_aps_corrupted']:.1e} | "
             f"{c['determinism_replay']:.1e} |")

    emit()
    emit("### Commands")
    emit()
    emit("```bash")
    for name, data in runs.items():
        argv = data["argv"]
        emit("python -m experiments.energy_consensus_verification "
             + " ".join(argv[1:]))
    emit("```")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
