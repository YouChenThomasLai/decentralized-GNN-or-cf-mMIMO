"""Paired fixed-holdout report for graph representation + energy consensus."""

import argparse
import json
import math
import os

import numpy as np
import torch

from evaluate import build_model, checkpoint_path, resolve_device, seed_everything
from model import load_checkpoint
from rates import RatePrecompute, quantize_phase
from simulation import ChannelSimulator


EPS = 1e-12
MATCHED_FIELDS = (
    "M", "N", "L", "K", "AP", "D", "ch", "pmax_dbm", "batch_size",
    "assoc_threshold", "n_iter", "lr", "weight_decay", "seed",
)


def load_run(run_dir, device):
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    config = summary["config"]
    # ChannelSimulator samples fixed AP-RIS LoS components at construction.
    # Reset here so every arm gets the same topology, not merely the same UE
    # locations and small-scale draws during the holdout loop.
    seed_everything(config["seed"])
    simulator = ChannelSimulator(
        config["M"], config["N"], config["L"], config["batch_size"],
        n_ap=config["AP"],
    )
    model = build_model(config, simulator, device)
    checkpoint = checkpoint_path(run_dir, "best.pt")
    if checkpoint is None:
        raise SystemExit(f"missing best.pt under {run_dir}")
    load_checkpoint(model, checkpoint, device)
    model.eval()
    return summary, simulator, model, checkpoint


def clustered(values, batch_size):
    values = np.concatenate(values)
    clusters = values.reshape(-1, batch_size).mean(axis=1)
    sem = clusters.std(ddof=1) / math.sqrt(len(clusters)) if len(clusters) > 1 else 0.0
    return {
        "mean": float(values.mean()),
        "clustered_sem": float(sem),
        "samples": int(values.size),
    }


def add_trace_health(health, trace):
    proposals = trace["proposals"]
    weights = trace["weights"]
    energy = trace["energy"]
    weighted = (proposals * weights[..., None, None]).sum(dim=1)
    weighted_norm = weighted.norm(dim=-1)

    active = (energy > 0).to(proposals.dtype)
    equal = active / active.sum(dim=1, keepdim=True).clamp(min=EPS)
    equal_norm = (proposals * equal[..., None, None]).sum(dim=1).norm(dim=-1)

    health["weighted_concentration"].append(weighted_norm.detach().cpu().numpy())
    health["equal_concentration"].append(equal_norm.detach().cpu().numpy())
    health["projection_fallbacks"] += int((weighted_norm <= EPS).sum())
    health["proposal_nan"] += int((~torch.isfinite(proposals)).sum())
    health["weight_nan"] += int((~torch.isfinite(weights)).sum())
    health["energy_nan"] += int((~torch.isfinite(energy)).sum())


def evaluate_run(summary, simulator, model, samples, eval_seed, device):
    config = summary["config"]
    batch_size = config["batch_size"]
    if samples <= 0 or samples % batch_size:
        raise ValueError("samples must be a positive multiple of batch size")

    seed_everything(eval_seed)
    modes = ("centralized", "paper_decentralized", "own_only")
    records = {f"{mode}_{phase}": [] for mode in modes for phase in ("continuous", "2bit")}
    decompositions = {
        scope: {
            f"w_{w}__theta_{theta}_{phase}": []
            for w in ("centralized", scope)
            for theta in ("centralized", scope)
            for phase in ("continuous", "2bit")
        }
        for scope in ("paper_decentralized", "own_only")
    }
    unit_error = {mode: 0.0 for mode in modes}
    nan_count = {mode: {"beamformer": 0, "phase": 0, "rate": 0} for mode in modes}
    trace_health = {
        mode: {
            "weighted_concentration": [], "equal_concentration": [],
            "projection_fallbacks": 0, "proposal_nan": 0,
            "weight_nan": 0, "energy_nan": 0,
        }
        for mode in modes
    }
    energy_mode_max_error = 0.0

    for _ in range(samples // batch_size):
        features, edges, masks, direct, _ = simulator.training_batch(
            config["K"], config.get("assoc_threshold", 0.1),
            config.get("assoc_threshold", 0.1),
        )
        features, edges, direct = features.to(device), edges.to(device), direct.to(device)
        central_trace = {}
        with torch.no_grad():
            central = model.centralized(features, edges, masks, direct, central_trace)

        d_features, d_edges, d_masks, d_direct = simulator.decentralized_batch(
            config["K"], config.get("assoc_threshold", 0.1),
            config.get("assoc_threshold", 0.1), regenerate_channels=False,
        )
        d_features = [x.to(device) for x in d_features]
        d_edges = [x.to(device) for x in d_edges]
        d_direct = [x.to(device) for x in d_direct]
        paper_trace, own_trace = {}, {}
        with torch.no_grad():
            paper = model.decentralized(
                d_features, d_edges, d_masks, d_direct, trace=paper_trace
            )
            own = model.decentralized(
                d_features, d_edges, d_masks, d_direct, trace=own_trace,
                include_cross_ap_csi=False,
            )

        outputs = {
            "centralized": central,
            "paper_decentralized": paper,
            "own_only": own,
        }
        traces = {
            "centralized": central_trace,
            "paper_decentralized": paper_trace,
            "own_only": own_trace,
        }
        pre = RatePrecompute(simulator, device)

        for mode, (beamformer, theta) in outputs.items():
            for phase_name, phase in (
                ("continuous", theta), ("2bit", quantize_phase(theta, 2))
            ):
                rate = pre.sum_rate(beamformer, phase)
                records[f"{mode}_{phase_name}"].append(rate.cpu().numpy())
                nan_count[mode]["rate"] += int((~torch.isfinite(rate)).sum())
            nan_count[mode]["beamformer"] += int((~torch.isfinite(beamformer)).sum())
            nan_count[mode]["phase"] += int((~torch.isfinite(theta)).sum())
            unit_error[mode] = max(
                unit_error[mode], float((theta.norm(dim=-1) - 1).abs().max())
            )
            if traces[mode].get("proposals") is not None:
                add_trace_health(trace_health[mode], traces[mode])

        if paper_trace.get("energy") is not None:
            energy_mode_max_error = max(
                energy_mode_max_error,
                float((paper_trace["energy"] - own_trace["energy"]).abs().max()),
            )

        for scope in ("paper_decentralized", "own_only"):
            w = {"centralized": central[0], scope: outputs[scope][0]}
            theta = {"centralized": central[1], scope: outputs[scope][1]}
            for w_name, beamformer in w.items():
                for theta_name, phase in theta.items():
                    for phase_name, used_phase in (
                        ("continuous", phase), ("2bit", quantize_phase(phase, 2))
                    ):
                        key = f"w_{w_name}__theta_{theta_name}_{phase_name}"
                        decompositions[scope][key].append(
                            pre.sum_rate(beamformer, used_phase).cpu().numpy()
                        )

    metrics = {key: clustered(value, batch_size) for key, value in records.items()}
    gaps = {}
    for scope in ("paper_decentralized", "own_only"):
        for phase in ("continuous", "2bit"):
            a = records[f"centralized_{phase}"]
            b = records[f"{scope}_{phase}"]
            gaps[f"centralized_minus_{scope}_{phase}"] = clustered(
                [x - y for x, y in zip(a, b)], batch_size
            )

    proposal = {}
    fallback = {}
    for mode, health in trace_health.items():
        if health["weighted_concentration"]:
            weighted = np.concatenate([x.ravel() for x in health["weighted_concentration"]])
            equal = np.concatenate([x.ravel() for x in health["equal_concentration"]])
            proposal[mode] = {
                "energy_weighted_mean": float(weighted.mean()),
                "energy_weighted_p05": float(np.quantile(weighted, 0.05)),
                "equal_weighted_mean": float(equal.mean()),
                "equal_weighted_p05": float(np.quantile(equal, 0.05)),
            }
        fallback[mode] = {
            key: value for key, value in health.items()
            if key not in ("weighted_concentration", "equal_concentration")
        }

    return {
        "metrics": metrics,
        "centralized_to_decentralized_gap": gaps,
        "decomposition_2x2": {
            scope: {key: clustered(value, batch_size) for key, value in table.items()}
            for scope, table in decompositions.items()
        },
        "proposal_concentration": proposal,
        "numerical_health": {
            "nan_count": nan_count,
            "max_unit_modulus_error": unit_error,
            "projection": fallback,
            "paper_vs_own_energy_max_abs_error": energy_mode_max_error,
        },
        "_records": {key: np.concatenate(value) for key, value in records.items()},
    }


def render_report(payload):
    lines = [
        "# Graph representation + energy consensus screening",
        "",
        f"Fixed holdout: {payload['samples']} samples, seed {payload['eval_seed']}.",
        "",
        "| arm | cen cont. | cen 2-bit | paper cont. | paper 2-bit | own cont. | own 2-bit | cen−paper | cen−own | params | payload/AP-RIS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["arms"].items():
        m = row["metrics"]
        paper_gap = row["centralized_to_decentralized_gap"][
            "centralized_minus_paper_decentralized_continuous"
        ]
        own_gap = row["centralized_to_decentralized_gap"][
            "centralized_minus_own_only_continuous"
        ]
        model = row["model"]
        lines.append(
            f"| {name} | {m['centralized_continuous']['mean']:.4f} | "
            f"{m['centralized_2bit']['mean']:.4f} | "
            f"{m['paper_decentralized_continuous']['mean']:.4f} | "
            f"{m['paper_decentralized_2bit']['mean']:.4f} | "
            f"{m['own_only_continuous']['mean']:.4f} | "
            f"{m['own_only_2bit']['mean']:.4f} | {paper_gap['mean']:.4f} | "
            f"{own_gap['mean']:.4f} | "
            f"{model['effective_parameters']} | {model['ap_to_cpu_reals_per_ap_ris']} |"
        )

    lines += [
        "", "## Paired contrasts", "",
        "| contrast | metric | mean difference | clustered SE |",
        "|---|---|---:|---:|",
    ]
    for contrast, metrics in payload["paired_contrasts"].items():
        for metric, value in metrics.items():
            lines.append(
                f"| {contrast} | {metric} | {value['mean']:+.4f} | "
                f"{value['clustered_sem']:.4f} |"
            )

    lines += [
        "", "## Proposal concentration and numerical health", "",
        "| arm/mode | weighted mean | weighted p05 | equal mean | NaNs | max unit error | fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["arms"].items():
        for mode in ("centralized", "paper_decentralized", "own_only"):
            concentration = row["proposal_concentration"].get(mode)
            projection = row["numerical_health"]["projection"][mode]
            nan = row["numerical_health"]["nan_count"][mode]
            nan_total = sum(nan.values()) + sum(
                projection[key] for key in ("proposal_nan", "weight_nan", "energy_nan")
            )
            if concentration is None:
                c = ("n/a", "n/a", "n/a")
            else:
                c = (
                    f"{concentration['energy_weighted_mean']:.4f}",
                    f"{concentration['energy_weighted_p05']:.4f}",
                    f"{concentration['equal_weighted_mean']:.4f}",
                )
            lines.append(
                f"| {name}/{mode} | {c[0]} | {c[1]} | {c[2]} | {nan_total} | "
                f"{row['numerical_health']['max_unit_modulus_error'][mode]:.2e} | "
                f"{projection['projection_fallbacks']} |"
            )

    lines += ["", "## Phase/beamformer 2×2 decomposition", ""]
    for name, row in payload["arms"].items():
        lines += [
            f"### {name}", "",
            "| inference scope / phase | Wc, θc | Wc, θd | Wd, θc | Wd, θd |",
            "|---|---:|---:|---:|---:|",
        ]
        for scope in ("paper_decentralized", "own_only"):
            table = row["decomposition_2x2"][scope]
            for phase in ("continuous", "2bit"):
                values = [
                    table[f"w_{w}__theta_{theta}_{phase}"]["mean"]
                    for w, theta in (
                        ("centralized", "centralized"),
                        ("centralized", scope),
                        (scope, "centralized"),
                        (scope, scope),
                    )
                ]
                lines.append(
                    f"| {scope}/{phase} | " + " | ".join(f"{x:.4f}" for x in values) + " |"
                )

    lines += ["", "## Decision", "", payload["decision"]["text"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0", required=True)
    parser.add_argument("--g1", required=True)
    parser.add_argument("--g2", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--eval_seed", type=int, default=20260915)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out_dir", default="../../artifacts/decentralized_ris/e06_graph_energy_training/screening"
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    run_dirs = {"G0": args.g0, "G1": args.g1, "G2": args.g2, "R0": args.anchor}
    loaded = {name: load_run(path, device) for name, path in run_dirs.items()}
    anchor_config = loaded["R0"][0]["config"]
    mismatches = {}
    for name, (summary, _, _, _) in loaded.items():
        diff = {
            field: [summary["config"].get(field), anchor_config.get(field)]
            for field in MATCHED_FIELDS
            if summary["config"].get(field) != anchor_config.get(field)
        }
        if diff:
            mismatches[name] = diff
    if mismatches:
        raise SystemExit(f"runs are not matched: {mismatches}")

    arms = {}
    for name, (summary, simulator, model, checkpoint) in loaded.items():
        print(f"[evaluate] {name}: {checkpoint}")
        result = evaluate_run(
            summary, simulator, model, args.samples, args.eval_seed, device
        )
        result.update(
            run_dir=run_dirs[name], checkpoint=checkpoint,
            config=summary["config"], model=model.describe(),
        )
        arms[name] = result

    raw = {name: row.pop("_records") for name, row in arms.items()}
    contrasts = {}
    contrast_metrics = (
        "centralized_continuous", "paper_decentralized_continuous",
        "paper_decentralized_2bit", "own_only_continuous", "own_only_2bit",
    )
    for left, right in (
        ("G0", "R0"), ("G1", "G0"), ("G2", "G1"),
        ("G1", "R0"), ("G2", "R0"),
    ):
        contrasts[f"{left}_minus_{right}"] = {
            metric: clustered([raw[left][metric] - raw[right][metric]],
                              anchor_config["batch_size"])
            for metric in contrast_metrics
        }
    anchor = arms["R0"]["metrics"]["paper_decentralized_continuous"]["mean"]
    clear_behind, pass_10k = {}, []
    paired_vs_r0 = {}
    for name in ("G0", "G1", "G2"):
        rate = arms[name]["metrics"]["paper_decentralized_continuous"]["mean"]
        paired = np.load(os.path.join(run_dirs[name], "metrics.npz"))
        late = paired["validation_decentralized"][-3:]
        late_slope = float(np.polyfit(np.arange(late.size), late, 1)[0]) if late.size > 1 else 0.0
        deficit = anchor - rate
        differences = (raw[name]["paper_decentralized_continuous"]
                       - raw["R0"]["paper_decentralized_continuous"]).reshape(
            -1, anchor_config["batch_size"]
        ).mean(axis=1)
        paired_sem = differences.std(ddof=1) / math.sqrt(len(differences))
        paired_vs_r0[name] = {
            "mean": float(differences.mean()), "clustered_sem": float(paired_sem)
        }
        clear_behind[name] = bool(
            deficit > 0.05 * anchor and deficit > 2 * paired_sem
        )
        if not clear_behind[name]:
            pass_10k.append((rate, name, late_slope))

    if all(clear_behind.values()):
        decision = {
            "stop_long_training": True,
            "finalist": None,
            "clear_behind_r0": clear_behind,
            "text": (
                "G0/G1/G2 are all >5% and >2 conservative SE below R0. Stop; "
                "the evidence remains consistent with projection-before-consensus "
                "training parameterization as the limiting issue."
            ),
        }
    else:
        _, finalist, late_slope = max(pass_10k)
        decision = {
            "stop_long_training": False,
            "finalist": finalist,
            "finalist_last_three_validation_slope_per_1k": late_slope,
            "clear_behind_r0": clear_behind,
            "text": (
                f"{finalist} passes the matched 10k screen and is the finalist. "
                "Only this arm is eligible for a matched longer-training plan."
            ),
        }

    payload = {
        "samples": args.samples, "eval_seed": args.eval_seed,
        "matched_fields": list(MATCHED_FIELDS), "arms": arms,
        "paired_contrasts": contrasts,
        "paired_paper_decentralized_continuous_vs_r0": paired_vs_r0,
        "decision": decision,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "screening.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(render_report(payload))
    np.savez(
        os.path.join(args.out_dir, "paired_rates.npz"),
        **{f"{arm}__{metric}": values
           for arm, metrics in raw.items() for metric, values in metrics.items()},
    )
    print(render_report(payload))


if __name__ == "__main__":
    main()
