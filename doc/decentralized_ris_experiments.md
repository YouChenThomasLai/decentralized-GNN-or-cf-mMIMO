# Decentralized RIS experiment index

## Current conclusion

R0 is reproducible but needs roughly 350–500k iterations to reach its operational plateau. The
current finalist is G2-150k: node-free per-RIS link tokens, per-RIS context, and parameter-free local
energy consensus. On the shared 400-sample holdout it reaches 24.5358 paper-decentralized bps/Hz,
$2.2053\pm0.2841$ above R0-500k, while sending 31 rather than 120 fp32 scalars per AP–RIS pair.

All training results still use one training seed and one fixed topology. The next priority is the
minimal [external benchmark and deterministic topology screen](./research_positioning.md#8-benchmark-plan);
multi-seed training remains deferred until the baseline set and claim are frozen.

## Experiments

All reports use the same sections: status, conclusion, setup, results, interpretation, limitations,
and reproduction/artifacts.

| ID | Experiment | Status | Headline result |
|---|---|---|---|
| E01 | [Baseline reproduction and training budget](./experiments/e01_baseline_training.md) | Completed | R0 reaches an operational plateau around 350–400k; 500k paper-dec. rate is 21.399 |
| E02 | [Centralized, paper-decentralized, and own-only inputs](./experiments/e02_input_modes.md) | Completed diagnostic | Cen−dec shrinks 37.5% from 150k→500k; dec−own does not |
| E03 | [RIS phase headroom](./experiments/e03_phase_headroom.md) | Completed reference | 500k greedy phase search recovers 90.0% of the paper-dec. measurable gap |
| E04 | [RIS action-interface screening](./experiments/e04_action_interfaces.md) | Completed screening | Pair magnitude preserves a mature R0 policy but fails matched 10k retraining |
| E05 | [Parameter-free local-energy consensus](./experiments/e05_energy_consensus.md) | Verified fixed-checkpoint result | Local energy beats learned pair scale by $1.6194\pm0.0586$ |
| E06 | [Graph representation with energy consensus](./experiments/e06_graph_energy_training.md) | Completed; finalist selected | G2-150k beats R0-500k by $2.2053\pm0.2841$ in paper-dec. continuous rate |
| E07 | [AP-to-CPU message codec pilot](./experiments/e07_message_codec.md) | Provisional | Direction precision dominates, but one pre-declared control failed |
| E08 | [Historical baseline sweeps](./experiments/e08_baseline_sweeps.md) | Historical reproduction | Antenna/power trends reproduce the paper; not evidence for G2 |
| E09 | [Snapshot network-scaling pilot](./experiments/e09_network_scaling_archive.md) | Archived | No formal scalability claim; retain implementation lessons only |

## Open decisions

1. Freeze G2-150k and register the benchmark experiment before implementing the Stage-A baseline set;
   do not create an E-number from the plan alone.
2. Use training seed 0 for the benchmark screen and test the shortlisted frozen policies on the two
   deterministic topology controls; defer multi-seed training until the baseline set is frozen.
3. Decide the condition-aware identity control for E07 before rerunning its codec frontier.
4. Run a 500k continuous joint reference only if a joint-headroom claim is needed.

## Artifact policy

- Keep finalist and necessary milestone checkpoints, summaries, paired raw data, pre-registrations,
  and machine-readable controls.
- Delete smoke checkpoints, empty aborted runs, caches, duplicate copies rejected by a report, and
  unreferenced exports.
- Generated artifact reports may retain full audit tables; research conclusions belong in the
  corresponding tracked E01–E09 report.
