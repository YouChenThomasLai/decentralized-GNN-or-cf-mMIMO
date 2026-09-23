# E12 — Model-based joint-optimization pair

## 1. Status

- Question: what do centralized and distributed conventional joint active/passive optimizers achieve
  on the shared benchmark channels?
- Status: **Fresh fixed-budget feasible-action comparison completed; strict convergence claim still fails**
- Updated: 2026-09-20
- Scope: Huang et al. centralized counterpart and incremental ring ADMM; canonical T0 topology
- Primary artifacts: `artifacts/decentralized_ris/e12_model_based_optimization/fixed_budget_protocol/`
- Earlier strict-gate run: `artifacts/decentralized_ris/e12_model_based_optimization/corrected_protocol/`
- Original failed-gate run: `artifacts/decentralized_ris/e12_model_based_optimization/`
- Fresh fixed-budget deployment comparison: declared below before evaluation on seed 20260920;
  artifacts under `fixed_budget_protocol/`

## 2. Conclusion

The new, separately declared fixed-budget comparison closes the Stage-A **deployed-action** question
on 400 fresh paired samples (seed 20260920). All eight eligibility controls pass. With the frozen
G2-150k checkpoint on the same channels, continuous sum rates are 22.9587 G2, 28.9463 centralized,
and 28.0170 distributed bps/Hz. G2's paired deficits are $5.9876\pm0.2846$ and
$5.0583\pm0.2265$ bps/Hz (50 batch clusters). It retains 79.3% and 81.9% of those rates while
using 31.5× and 2,348.6× fewer counted **coordination bits** and one upstream round versus two
and 1,001. These bit ratios exclude delivery of shared-UE cross-AP CSI required by G2's input
view. Thus the evidence supports a conditional coordination trade-off, not a claim that G2
outperforms this model-based family in rate or total network traffic.

The earlier corrected run remains a separate diagnostic. It reached 31.8026/30.4652 bps/Hz on
seed 20260915 but failed four of thirteen strict gates: precision, centralized monotonicity,
worst-case consensus, and divergence. In the new seed, the strict precision and worst-case
consensus gates still fail (0.1050 bps/Hz and 0.3378 versus thresholds 0.01 and 0.05).
The fixed-budget result is therefore a feasible implementation after the stated online budget,
**not** a converged ADMM reference, precision-stable optimizer trajectory, global upper bound, or
paper-exact replication of Huang et al.

## 3. Setup

Both arms optimize the same unweighted sum rate with the simulator's fixed AP–UE association,
$P_{\max}=15$ dBm per AP, continuous unit-modulus RIS phases, direct links, and the same channel
samples. The centralized arm uses full CSI and one global RIS vector. The distributed arm uses only
AP-local CSI, one RIS copy per AP, a ring consensus graph, and one multiplier update per sweep.

The first-choice Xu et al. formulation was rejected because its RIS subproblem is solved by a learned
convolutional block and the paper provides no conventional replacement. E12 therefore uses Huang et
al.'s non-learned fractional-programming/MM formulation and its incremental ADMM coordination. The
method definition and documented deviations are maintained in
[the method report](../decentralized_ris_methods.md#model-based-pair).

The corrected protocol makes four changes to the original run:

1. Power bisection returns the feasible upper bracket endpoint instead of the midpoint.
2. A converged sample freezes its beamformer, phase, multiplier, and penalty while slower batch
   members continue.
3. The primary initialization is the previously declared `matched` construction rather than
   all-ones.
4. Penalty calibration uses 400 samples at seed 20260912. Only the maintained per-sweep schedule is
   recalibrated; the already rejected per-activation schedule is not rerun.

The penalty grid remains $\{0.05,0.1,0.2,0.4,0.8\}$. The rule selects the smallest value whose
calibration worst-case consensus residual is at most 0.05 with zero divergence. Only 0.8 qualifies:
its residual is 0.0362, compared with 0.0981 at 0.4. Primary and replication evaluation seeds are
20260915 and 20260918, each with 400 samples. Solvers use float64, a $10^{-4}$ stopping tolerance,
2,000 centralized iterations, 200 distributed sweeps, and 200 MM inner iterations.

The protocol declaration and thresholds are stored with the artifacts and in the `DECISION`
constant. The current Git history does not contain an immutable pre-run commit, so this report calls
them declared controls rather than claiming independently auditable preregistration.

### Fixed-budget deployment continuation (declared 2026-09-19, before fresh evaluation)

The failed-gate run answered whether the Huang adaptation meets a strict convergence and numerical
trajectory specification. A separate Stage-A estimand is the **rate of the physical action actually
deployed after a fixed online budget**. For this continuation, freeze the corrected implementation,
float64, `matched` initialization, calibrated `rho_scale=0.8`, per-sweep dual update, 2,000
centralized iterations, 200 distributed sweeps, 200 MM steps, and the same association and objective.
Evaluate one new 400-sample primary seed **20260920**, never used to select a solver parameter, and
evaluate the frozen G2-150k checkpoint on exactly that seed. No calibration, grid search, or
post-evaluation parameter change is permitted. The 20260915/20260918 runs remain diagnostic and
cannot be retrospectively promoted.

The eligibility gate for this *fixed-budget feasible-action* estimand is: physical phase unit
modulus, per-AP power feasibility, association mask, finite rates/actions, agreement with the
maintained rate evaluator, AP-local solver update, and a rate above the frozen random-phase smoke
control. The deployed distributed action is the already implemented unit-modulus projection of
the mean of the AP-local copies, and its rate is evaluated globally from that phase and the solved
beamformers. Report the final copy-consensus residual, primal divergence fraction, centralized
rate decrease, and float32-versus-float64 re-solve sensitivity **even if they fail the original
convergence gates**. Those diagnostics determine what cannot be claimed: no ADMM convergence,
precision-stable optimizer trajectory, or paper-exact Huang reproduction. A fixed-budget result is
admissible only as a reproducible, feasible *implementation under the stated compute budget*.

Use paired 8-sample batch clusters to report G2 versus each solver and centralized versus
distributed, in continuous and rounded 2-bit RIS actuation. Count AP→CPU, CPU→AP, and AP→AP
coordination under one common boundary; report CPU→RIS actuation separately because it is common
to all three methods. UE→AP CSI acquisition is excluded for all arms and must be stated. G2 spends
$RL(N+1)=620$ fp32 values upstream, one upstream round, and 120 values for CPU→RIS actuation;
the E12 ledger determines the solver counts. These are modeled payloads and sequential exchange
counts, not measured network latency. Do not label either model-based rate an upper bound.

## 4. Results

Fresh fixed-budget protocol, seed 20260920, 400 paired samples, 50 clusters of eight:

| Arm | Information / deployed action | Continuous | Clustered SE | Rounded 2-bit | Clustered SE | Coordination bits | Rounds |
|---|---|---:|---:|---:|---:|---:|---:|
| G2-150k | Paper-decentralized proposal + CPU fusion | 22.9587 | 0.5404 | 21.3847 | 0.4975 | 19,840 | 1 |
| Huang-adapted centralized | Full CSI, one global RIS vector | 28.9463 | 0.6902 | 24.5714 | 0.4821 | 624,640 | 2 |
| Huang-adapted distributed | AP-local CSI, projected mean RIS copy | 28.0170 | 0.6260 | 24.0111 | 0.4507 | 46,595,840 | 1,001 |

| Paired contrast | Continuous mean ± clustered SE | 95% CI | Rounded 2-bit mean ± clustered SE | 95% CI |
|---|---:|---:|---:|---:|
| G2 − centralized | $-5.9876\pm0.2846$ | $[-6.5454,-5.4299]$ | $-3.1867\pm0.1632$ | $[-3.5066,-2.8669]$ |
| G2 − distributed | $-5.0583\pm0.2265$ | $[-5.5023,-4.6144]$ | $-2.6264\pm0.1761$ | $[-2.9717,-2.2812]$ |
| Centralized − distributed | $+0.9293\pm0.1169$ | $[0.7002,1.1584]$ | $+0.5603\pm0.0853$ | $[0.3930,0.7276]$ |

All eight predeclared fixed-budget eligibility controls pass: power, unit modulus, 2-bit grid,
maintained-rate-path equivalence, AP locality, association mask, finite values, and separation from
random phase. The maximum power overshoot is zero, phase modulus error $3.8\times10^{-15}$,
rate-path deviation $1.8\times10^{-6}$ bps/Hz, and unserved-column leakage zero. G2's maximum
decentralized unit-modulus error is $1.2\times10^{-7}$. The old strict gates on this fresh seed are
11/13: centralized monotonicity and divergence now pass, while the float32 re-solve deviation is
0.1050 bps/Hz and the maximum distributed copy-consensus residual is 0.3378. Median residual is
0.000394; 99.5% of distributed samples hit 200 sweeps. The centralized median is 471 iterations,
with 10.75% hitting its 2,000-iteration cap.

The coordination boundary includes AP→CPU, CPU→AP, and AP→AP messages as modeled by each method.
G2's paper-decentralized observation includes shared-UE cross-AP CSI, while the distributed solver
starts from AP-local CSI. The cost and delivery rounds needed to make that CSI available to G2 were
not measured or included. Thus 31.5× and 2,348.6× compare the modeled coordination payloads
conditional on each method already having its required observations; they are not end-to-end
network-traffic ratios. The stored solver `summary.json` ledger calls excluded CSI acquisition
"identical across arms"; that phrase is inaccurate for G2's input view and is superseded by this
accounting note. The numerical counts are unchanged. CPU→RIS actuation is common
and reported separately as 120 fp32 values (3,840 bits) per sample, or 240 packed bits for the
rounded 2-bit actuator setting; UE→AP CSI acquisition is
excluded for all arms. In particular, the earlier E12 `total_online_bits_fp32` includes RIS
actuation and must not be directly divided by G2's AP→CPU payload. Round counts exclude common
actuation. The payload figures assume fp32 transmission for each counted real value although the
model-based solver computes in float64; no wire serialization or precision-performance study was
performed. Rounds count algorithmic sequential exchanges, not measured network latency.

The subsequent [E14 topology and CSI ledger](./e14_topology_stress.md) fills this gap for one
transport scenario. On the same T0 seed, direct fp32 AP→AP unicast of shared-UE raw CSI averages
953,674 bits/sample; adding G2's 19,840-bit proposal gives 973,514 bits and two sequential rounds.
That is 1.56× the centralized arm's 624,640 modeled coordination bits, whereas distributed ADMM
remains 47.9× above G2. The earlier 31.5× and 2,348.6× ratios remain valid only for the original
observation-conditional coordination boundary. UE feedback, multicast, compression, and physical
network latency are not measured by E14.

For provenance, the earlier corrected-protocol result follows; its rates are **diagnostic** and
must not be pooled with the new holdout or substituted for the table above.

Primary seed 20260915, 400 paired samples:

| Arm | Continuous | Clustered SE | 2-bit | Clustered SE | 2-bit re-solved | Random phase |
|---|---:|---:|---:|---:|---:|---:|
| `centralized` | 31.8026 | 0.9294 | 26.1789 | 0.5220 | 28.3765 | 6.1200 |
| `distributed` | 30.4652 | 0.7992 | 25.3379 | 0.5116 | 27.4891 | 6.2264 |

The paired centralized-minus-distributed contrast is $1.3373\pm0.1905$ bps/Hz, with a 95% interval
of $[0.9639,1.7108]$ and 50/50 cluster wins. At 2 bits it is $0.8410\pm0.1186$.

The replication seed preserves the ordering: 28.8458 centralized and 28.1205 distributed. Its
worst consensus residual is 0.0416 and divergence is zero. The main failure table is:

| Gate | Threshold | Primary | Replication |
|---|---:|---:|---:|
| Power overshoot | $\le10^{-6}$ | 0, pass | 0, pass |
| Precision deviation | $\le0.01$ bps/Hz | 0.2867, **fail** | 0.0753, **fail** |
| Centralized monotonicity | $\le10^{-6}$ bps/Hz | 0.00215, **fail** | 0, pass |
| Worst consensus residual | $\le0.05$ | 0.1517, **fail** | 0.0416, pass |
| Divergence rate | 0 | 0.0075, **fail** | 0, pass |

The other eight gates pass on the primary seed: unit modulus, quantization grid, maintained-rate-path
equivalence, AP locality, reflection-block monotonicity, no unserved-column leakage, finite values,
and separation from random phase. Thus 9 of 13 primary gates and 12 of 13 replication gates pass.

Solver and deployment measurements on the primary seed:

| Quantity | Centralized | Distributed |
|---|---:|---:|
| Realized iterations/sweeps, median | 508 | 200 |
| Fraction hitting cap | 19.0% | 99.75% |
| Counted online real values | 19,640 | 1,456,240 |
| Counted fp32 bits | 628,480 | 46,599,680 |
| Online rounds | 2 | 1,001 |
| Batch-1 end-to-end estimate, local RTX 5060 | 50.7 s | 48.2 s |

The declared 80-sample initialization diagnostic gives centralized/distributed rates of
25.0509/21.7771 for `ones`, 27.7978/24.5995 for `random`, and 28.9871/28.0820 for `matched`.

## 5. Interpretation

- On an untouched evaluation seed, the adapted model-based family achieves materially higher rate
  than G2, including after rounded 2-bit RIS actuation. Its online coordination is much heavier;
  G2's lower rate and lower bits/rounds make this a measured trade-off among these methods on T0.
- The 50-cluster paired intervals quantify channel-sample variation on this one topology. They do
  not cover training-seed, topology, solver-initialization, or hardware variability.
- The new fixed-budget gate applies to **deployed feasible actions**; the old 13-gate decision still
  rejects convergence-level claims. In particular, the maximum consensus residual exceeds its
  threshold even though the mean-copy projection is physically feasible and is the phase scored.
- The centralized arm has higher rate and fewer counted coordination bits than distributed ADMM
  here, but it requires full CSI at the CPU. AP-local CSI is the distributed arm's separate
  information-pattern benefit; the three reported axes alone do not describe that constraint.
- The original power and stopping-accounting defects are fixed; neither explains the remaining
  failed gates.
- Initialization is a first-order property of this non-convex benchmark. The all-ones run cannot be
  presented as the model-based family's representative performance.
- The 400-sample calibration still does not control a worst-case holdout tail. Selecting a larger
  penalty after seeing this holdout would violate the declared rule, so E12 is reported rather than
  retuned again.
- The float32 sensitivity indicates basin-level numerical instability, not ordinary rounding noise.
  A paper claim based on the exact 31.8026/30.4652 values would therefore be too strong.
- In the earlier strict run, iterative distributed coordination removed the CSI upload but used
  about 74 times the centralized arm's total counted online payload and 1,001 rounds. The fresh
  fixed-budget comparison establishes the same qualitative signaling direction on a paired seed;
  it does not establish a solver convergence guarantee.

## 6. Limitations

- The original corrected run failed four strict controls; it remains diagnostic. The fresh
  fixed-budget comparison passes its separately declared feasibility controls but still fails
  strict precision and worst-case copy-consensus controls. It is eligible only as a
  finite-budget implementation, not as a converged or precision-stable optimizer.
- Almost every distributed sample reaches the sweep cap; its converged optimum is not measured.
- The worst-case consensus rule does not generalize reliably from one 400-sample calibration set to
  another 400-sample holdout.
- Both methods are local optimizers and depend strongly on initialization.
- The G2 paper-decentralized proposal can use shared-UE cross-AP links, whereas the Huang-adapted
  distributed solver updates from AP-local CSI and exchanged aggregate state; the arms share the
  physical system, samples, objective and constraints, but their information patterns are not
  identical. The G2 cross-AP CSI delivery cost is outside the coordination ledger, so the quoted
  bit ratios cannot be generalized to total network traffic. Its 1,001 rounds also convey
  progressively updated aggregate information.
- Timing is implementation- and device-specific. The corrected run used a local RTX 5060 Laptop GPU;
  the original run used an RTX 5090, so their milliseconds should not be compared directly.
- Fresh fixed-budget timing used one batch-1 sample only as a resource check; its identical median
  and p95 are not an informative runtime distribution and are excluded from the main comparison.
- The association mask is a project-specific extension of the source formulations.
- The protocol declaration was written before the fresh evaluation, but was not frozen in an
  immutable pre-run commit. Its timing is therefore a reported procedure rather than an
  independently auditable preregistration.
- PyTorch warns that CUDA `median(dim=...)` lacks a deterministic implementation; the fixed-budget
  run may not be bit-for-bit repeatable even with deterministic-algorithm warnings enabled.

## 7. Reproduction and artifacts

```bash
cd code/decentralized_ris
RIS_PYTHON=<env python> DEVICE=cuda:0 bash scripts/e12_model_based_joint.sh
```

Fresh fixed-budget continuation (the commands below use the already calibrated $\rho=0.8$ and
write beside, rather than over, the corrected-protocol run):

```bash
cd code/decentralized_ris
python -m experiments.model_based_joint \
  --run ../../artifacts/decentralized_ris/e06_graph_energy_training/g2_long_training/g2/iter150000 \
  --samples 400 --eval_seed 20260920 --replication_seed 0 \
  --max_iterations 2000 --max_sweeps 200 --mm_iterations 200 \
  --rho_scale 0.8 --dual_period sweep --init matched --precision double \
  --init_samples 8 --precision_samples 40 --timing_samples 1 \
  --timing_iterations 3 --timing_sweeps 1 --device cuda:0 \
  --out_dir ../../artifacts/decentralized_ris/e12_model_based_optimization/fixed_budget_protocol
python evaluate.py \
  --runs ../../artifacts/decentralized_ris/e06_graph_energy_training/g2_long_training/g2/iter150000 \
  --checkpoint iter150000.pt --samples 400 --eval_seed 20260920 --device cuda:0 \
  --out ../../artifacts/decentralized_ris/e12_model_based_optimization/fixed_budget_protocol/g2_evaluation.json
python -m experiments.summarize_e12_fixed_budget \
  ../../artifacts/decentralized_ris/e12_model_based_optimization/fixed_budget_protocol \
  ../../artifacts/decentralized_ris/e12_model_based_optimization/fixed_budget_protocol/g2_evaluation.json
```

| Evidence | Location |
|---|---|
| Corrected calibration | `artifacts/decentralized_ris/e12_model_based_optimization/corrected_protocol/calibration.json` |
| Corrected summary and generated audit | `corrected_protocol/summary.json`, `corrected_protocol/report.md` |
| Corrected paired raw rates | `corrected_protocol/paired_rates.npz` |
| Corrected logs | `corrected_protocol/e12_calibration.log`, `corrected_protocol/e12_model_based_joint.log` |
| Fresh fixed-budget declaration and exact solver settings | `fixed_budget_protocol/protocol.json`, `fixed_budget_protocol/summary.json` |
| Fresh G2 evaluation and paired cluster inputs | `fixed_budget_protocol/g2_evaluation.json`, `fixed_budget_protocol/g2_evaluation_paired.npz` |
| Fresh model-based paired rates and cross-method summary | `fixed_budget_protocol/paired_rates.npz`, `fixed_budget_protocol/comparison.json` |
| Fresh generated audit and console log | `fixed_budget_protocol/report.md`, `fixed_budget_protocol/run.log` |
| Original failed-gate run | `artifacts/decentralized_ris/e12_model_based_optimization/summary.json` |
| Solver and protocol | `code/decentralized_ris/model_based.py`, `experiments/model_based_joint.py`, `experiments/summarize_e12_fixed_budget.py` |
| Regression checks | `code/decentralized_ris/tests/test_model_based_joint.py`, `tests/test_e12_fixed_budget_summary.py` |

The corrected run was executed locally on 2026-09-18 using the
`decentralized-inference` Conda environment. The original JSON and logs remain unchanged as
provenance; corrected artifacts are stored in a separate subdirectory.
The fresh fixed-budget continuation was executed locally on 2026-09-19 with training seed 0,
evaluation seed 20260920, 400 samples, G2 checkpoint `iter150000.pt`, and an RTX 5060 Laptop GPU.
Its own artifact subdirectory preserves the declaration, raw paired arrays, controls and generated
audit without altering either earlier run.
