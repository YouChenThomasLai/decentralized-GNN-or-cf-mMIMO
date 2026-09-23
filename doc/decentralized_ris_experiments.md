# Decentralized RIS experiment index

## Current conclusion

R0 is reproducible but needs roughly 350–500k iterations to reach its operational plateau. The
current finalist is G2-150k: node-free per-RIS link tokens, per-RIS context, and parameter-free local
energy consensus. On the shared 400-sample holdout it reaches 24.5358 paper-decentralized bps/Hz,
$2.2053\pm0.2841$ above R0-500k, while sending 31 rather than 120 fp32 scalars per AP–RIS pair.
The owner-requested continuation reaches 24.6898 at 300k, only $0.1540\pm0.1030$ above 150k, so it
does not displace 150k as the compute-efficient finalist.

All method-level conclusions and follow-up comparisons therefore use G2. G1 is retained only as the
matched no-context ablation in E06; it supports component attribution but is not a second finalist,
external competitor, or continuation target.

E07 observes a positive 68-bit matched-budget contrast: G2 exceeds the best R0 wire format by
$1.5424\pm0.2363$ bps/Hz, repeated on a fresh evaluation seed. Its global phase-grid control fails,
and G2 pays the larger compression penalty. A fixed-backbone test detects no executable-phase
advantage over a same-bit Cartesian codec, so the codec is not an independent compression
contribution. The wider frontier remains descriptive.

All training results still use one training seed and one fixed training topology. E14 has now run the
[deterministic topology screen](./experiments/e14_topology_stress.md) on frozen checkpoints;
multi-seed training remains deferred until the baseline set and claim are frozen. E13 has settled the
component question inside that plan: under G2's own fused phase, its learned active beamformer beats
RIS-aware local MRT and RZF. The Stage-A complete-method comparison is also closed: E10's
parameter-matched conventional DNN reaches only
6.8192 paper-decentralized bps/Hz against G2-150k's 24.5358. E11 was stopped and excluded before a
valid holdout. E12 now has a fresh, paired fixed-budget **deployed-action** comparison against the
Huang-adapted model-based family: G2-150k reaches 22.9587 bps/Hz versus 28.9463 centralized and
28.0170 distributed on seed 20260920. All eight feasibility controls pass; G2 uses 31.5× and
2,348.6× fewer coordination bits, respectively, with one upstream round versus two and 1,001.
The original corrected E12 run remains diagnostic because four strict convergence/precision gates
failed; on the fresh seed, precision and worst-case consensus still fail. Thus Stage A has a
finite-budget external reference, but no converged or precision-stable optimizer benchmark.
The original E12 bit ratios exclude delivery of shared-UE cross-AP CSI needed by G2's
paper-decentralized observation. E14 counts that delivery under direct fp32 AP→AP unicast: G2's
total is 0.927–0.980 million bits/sample across T1/T2, above the centralized arm's 0.625 million
and about 48–50 times below distributed ADMM's 46.596 million. G2's paired advantage over R0
persists on both shifted layouts, while both fixed-budget model-based arms retain higher rates.
These figures are modeled payloads for a named route, not measured end-to-end traffic or latency.

E14 also bounds where that persistence matters. A mask-only calibration picked two AP ring radii
that change how much shared-UE CSI exists, and on a separate holdout seed the visible-link
fraction runs 0.2860, 0.3927 and 0.6170. G2 keeps the smaller relative decentralization gap
throughout, 4.25%, 2.24% and 2.10% against R0's 14.26%, 11.03% and 6.77%, but its paired advantage
narrows from $+3.6939$ to $+0.4846$ bps/Hz as visibility rises, while its modeled CSI delivery
grows from 0.99× to 2.35× the centralized arm's payload. Because visibility and operating SNR move
together along AP ring radius, a second pre-registered pair adds user spread as a degree of freedom
and matches centralized rate instead: on MLOW and MHIGH the holdout centralized rates agree within
0.8274 bps/Hz while the visible fraction differs by 0.289. There, lower visibility still costs
rate, $+1.44$ pp of relative gap for G2 $[+0.26,+2.61]$ and $+4.34$ pp for R0 $[+1.71,+6.96]$, so
visibility is not only a proxy for operating level. The declared test that G2's increase is the
smaller one passed marginally on that seed, $+2.90$ pp $[+0.01,+5.78]$, and a single pre-registered
confirmation on seed 20260929 with 3,200 samples per layout settles it at $+1.53$ pp
$[+0.60,+2.45]$. Quote the confirmation figure; the discovery one is the selected, inflated estimate.

E15 sweeps the remaining declared deployment condition, the common per-AP transmit power budget, on the same frozen checkpoints. G2 stays ahead of R0 in paper-decentralized rate at every point from 5 to 35 dBm in both phase settings, and the advantage is close to a constant 10% of R0's own rate across that 30 dB span rather than a growing one, so quote the absolute figure only with its budget. Centralized rate does not separate the two architectures anywhere on the grid, and both frozen policies lose proportionally more to decentralization as the budget rises, G2 from $-0.07\%$ to $3.63\%$ and R0 from $9.10\%$ to $12.62\%$. Because a frozen checkpoint scales one fixed spatial solution, its curve must saturate at that solution's interference ceiling; the sweep is a robustness result for policies trained at 15 dBm and is not a per-power retrained curve.

## Experiments

All reports use the same sections: status, conclusion, setup, results, interpretation, limitations,
and reproduction/artifacts.

| ID | Experiment | Status | Headline result |
|---|---|---|---|
| E01 | [Baseline reproduction and training budget](./experiments/e01_baseline_training.md) | Completed | R0 reaches an operational plateau around 350–400k; 500k paper-dec. rate is 21.399 |
| E02 | [Centralized, paper-decentralized, and own-only inputs](./experiments/e02_input_modes.md) | Completed diagnostic and G2/R0 endpoint | G2's own-only visibility penalty exceeds R0's by $4.7404\pm0.3469$ on a matched holdout |
| E03 | [RIS phase headroom](./experiments/e03_phase_headroom.md) | Completed reference | 500k greedy phase search recovers 90.0% of the paper-dec. measurable gap |
| E04 | [RIS action-interface screening](./experiments/e04_action_interfaces.md) | Completed screening | Pair magnitude preserves a mature R0 policy but fails matched 10k retraining |
| E05 | [Parameter-free local-energy consensus](./experiments/e05_energy_consensus.md) | Verified fixed-checkpoint result | Local energy beats learned pair scale by $1.6194\pm0.0586$ and recovers $0.9301\pm0.0047$ of the equal-to-best-found weight gap |
| E06 | [G2 graph representation and ablations](./experiments/e06_graph_energy_training.md) | Completed through 300k | G1 is retained as the no-context ablation; G2-300k gains only $0.1540\pm0.1030$ over 150k, so retain G2-150k as finalist |
| E07 | [Message codec and matched bit budget](./experiments/e07_message_codec.md) | Matched-budget comparison complete; 1 of 9 controls failed | Positive 68-bit contrast on two seeds; no isolated codec gain; wider frontier descriptive |
| E08 | [Historical baseline sweeps](./experiments/e08_baseline_sweeps.md) | Historical reproduction | Antenna/power trends reproduce the paper; not evidence for G2 |
| E09 | [Snapshot network-scaling pilot](./experiments/e09_network_scaling_archive.md) | Archived | No formal scalability claim; retain implementation lessons only |
| E10 | [Parameter-matched conventional DNN benchmark](./experiments/e10_dnn_benchmark.md) | Completed benchmark arm | Validation-selected DNN reaches 6.8192 paper-dec. bps/Hz against G2-150k's 24.5358 |
| E11 | [Same-interface MADDPG benchmark arm](./experiments/e11_maddpg_baseline.md) | Stopped and excluded | No valid holdout; infeasible exploration path found and active implementation removed |
| E12 | [Model-based joint-optimization pair](./experiments/e12_model_based_optimization.md) | Fresh fixed-budget comparison complete; strict convergence unverified | On paired seed G2 22.9587 vs 28.9463/28.0170 bps/Hz; modeled coordination payload excludes G2 input-CSI delivery |
| E13 | [Fixed-G2-RIS active-beamforming controls](./experiments/e13_fixed_ris_beamforming.md) | Completed component control | Under the same fused phase, G2 beats local MRT by $2.8848\pm0.1685$ |
| E14 | [Frozen-policy topology stress and CSI delivery](./experiments/e14_topology_stress.md) | Completed fixed-budget screen, visibility pair, and confirmed rate-matched pair | G2−R0 stays positive on T1/T2 and on both visibility pairs; at matched centralized rate lower visibility costs G2 $+2.06$ pp and R0 $+3.59$ pp of relative gap, a confirmed $+1.53$ pp $[+0.60,+2.45]$ difference; modeled G2 direct CSI+proposal traffic exceeds centralized traffic |
| E15 | [Frozen-policy transmit-power sweep](./experiments/e15_transmit_power_sweep.md) | Completed frozen-policy sweep; all six controls pass | G2−R0 paper-dec. stays positive from 5 to 35 dBm, near 10% of R0's rate at every point; decentralization cost rises with power for both |

E-numbers are assigned per research question. E12 contains the mandatory
centralized/distributed model-based pair, and E14 contains the two-layout Stage-B stress test.

## Open decisions

1. The fixed-budget E12 comparison closes the deployed-action screening question, while a
   **converged**, precision-stable model-based reference remains open if that stronger claim is
   required. Do not promote either failed-gate run or tune on their observed holdouts. A further
   solver or convergence study needs a new declaration and fresh data; Xu et al.'s learned
   D2-ADMM would be a learned, not conventional, comparison.
2. E14 completes the two deterministic topology controls and the visibility pair for the
   shortlisted frozen policies. Decide whether the now narrowed claim merits multi-seed training;
   no topology-randomized retraining was triggered, because the G2−R0 ordering did not fail on
   T1/T2 or on either visibility endpoint. The visibility-versus-operating-level question is now
   answered by the rate-matched pair: at matched centralized rate both models lose more at low
   visibility, and the comparative statement is now confirmed on a fresh seed: R0's extra loss over
   G2 is $+1.53$ pp $[+0.60,+2.45]$. Nothing in the visibility line is outstanding; the remaining
   topology decision is still whether the narrowed claim merits multi-seed training.
3. Decide whether E07's phase-grid legality control is restated in radians. The pilot's identity
   control is closed — exact message identity and the condition-aware bound $\max|\Delta\theta|\rho$
   both pass — but the new grid control was normalized by the grid step and fails for $b_p\ge6$ at a
   constant float32 angular deviation. Nothing was re-run. Until it is restated, the frontier above
   5-bit phase precision is uncertified; the 68-bit operating point and every decision reported are
   unaffected. Decide separately whether 2-bit RIS actuation should replace continuous actuation as
   E07's primary readout; it must be re-declared and confirmed before any claim uses it.
4. Run a 500k continuous joint reference only if a joint-headroom claim is needed.
5. Do not start G3 now. E07 does not establish an executable-phase codec gain, E06 does not detect a
   primary-metric gain from G2's added context at 40k, and E12 shows a rate deficit against the
   fixed-budget model-based pair. E14 has completed the topology screen and cross-AP CSI accounting;
   the direct-unicast total no longer supports a lower-bit claim against centralized optimization.
   Reconsider a confidence-weighted proposal variant only
   with a testable calibration target and a predeclared matched-budget comparison; do not assign a
   new E-number from this decision alone.
6. E15 closes the transmit-power condition for the frozen finalists and the owner declined a retrained arm. Decide whether a power-matched retrained curve is required before any per-power achievability claim is made; the frozen sweep supports a robustness claim only, and its high-power saturation is a property of a fixed spatial solution rather than a capability limit.

## Artifact policy

- Keep finalist and necessary milestone checkpoints, summaries, paired raw data, pre-registrations,
  and machine-readable controls.
- Delete smoke checkpoints, empty aborted runs, caches, duplicate copies rejected by a report, and
  unreferenced exports.
- Generated artifact reports may retain full audit tables; research conclusions belong in the
  corresponding tracked E-report.
