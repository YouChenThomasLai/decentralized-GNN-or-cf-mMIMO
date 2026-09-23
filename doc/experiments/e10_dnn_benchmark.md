# E10 — Parameter-matched conventional DNN benchmark

## 1. Status

- Question: does G2's graph/link inductive bias carry the gain, or does a conventional non-graph DNN
  reach the same rate from the same inputs under the same constraints, loss, budget, capacity, and
  AP-to-CPU interface?
- Status: **Completed for the DNN family; both arms trained to the matched 150k budget**
- Updated: 2026-09-18
- Scope: benchmark item 3, the paired DNN family (`d0`, `d1`); training seed 0; T0 canonical topology
- Plan: [Benchmark plan, item 3](../research_positioning.md#82-comparison-roles-and-methods)
- Method under test: [G2](../decentralized_ris_methods.md#graph-variants) is the incumbent; `d0`/`d1`
  are defined in Section 3 of this report, not in the method report, because they are benchmark
  controls rather than maintained RIS interfaces
- Primary artifacts: `artifacts/decentralized_ris/e10_dnn_benchmark/`

## 2. Conclusion

A conventional flattened-input DNN does not reproduce G2's rate under matched inputs, constraints,
loss, capacity, and update budget. On the shared 400-sample holdout the validation-selected `d1`
checkpoint reaches 6.8192 paper-decentralized bps/Hz against G2-150k's 24.5358, a shortfall of
17.7166, and the selected `d0` checkpoint reaches 5.5171 against G2-150k's 25.2594 centralized. Both arms are well above
their own random-phase controls, so neither is broken; they simply learn a much weaker policy.

Two facts bound how strongly this may be read. Neither arm had converged at 150k: `d1` was still
gaining about 0.163 bps/Hz per 10k iterations over its final 50k, so the measured gap is a finite-budget
snapshot rather than a bound on its asymptotic shortfall. Against that, `d1` at 150k is still far
below what G2 and even R0 reach at 10k (18.6514 and 11.2905), which is a fifteenfold budget advantage
for the DNN, so the ordering is not plausibly a training-budget artifact.

The screen therefore passes: G2's advantage survives a parameter-matched conventional DNN. It does
not by itself establish a paper-level claim; E11 was subsequently skipped, and the corrected
model-based pair remains outstanding for Stage A.

## 3. Setup

### 3.1 The two arms

`variants` keeps the GNN backbone and changes the RIS action interface. The DNN control does the
opposite: it keeps the task and changes the representation. It removes the RIS message-passing node,
the edge-weighted neighbour aggregation, the max-excluding-self operator, and all weight sharing over
the (RIS, AP-UE node) axes, then consumes the flattened masked input tensors with plain MLP blocks.

| Arm | Inference view | Backbone | RIS action | AP→CPU RIS message |
|---|---|---|---|---|
| `d0` | centralized | flat MLP, one pass | all $R$ phase vectors from one head | none; no AP-originated message |
| `d1` | paper-decentralized | AP-shared flat MLP, one pass per AP | $N$ angles per RIS + local energy | $N+1$ reals per AP–RIS pair |
| G2 (incumbent) | paper-decentralized | node-free per-RIS link tokens | $N$ angles per RIS + local energy | $N+1$ reals per AP–RIS pair |

`d1` reuses `variants.local_energy` and `variants.circular_consensus` unchanged, so it deploys behind
exactly G2's interface: a strictly local energy scalar computed from the AP's own served channels, and
a parameter-free circular consensus with no trainable CPU-side parameter. The DNN replaces both the
phase-proposal and active-beamforming mappings while preserving that interface. `d0` has no AP-originated RIS message and therefore no
paper-decentralized deployment mode; its decentralized fields stay `NaN` rather than being filled by a
substitute rule, and its checkpoint is selected on its own centralized validation rate.

`d1` mirrors G2's own structure in which parts are AP-shared and which are per-AP: one shared trunk,
one shared beamformer head, one shared phase head, and $L=5$ per-AP power-control heads. The shared
trunk is told which AP it is running for by appending that AP's served-user mask to the input, which
is the same information G2's per-AP pooling mask carries.

### 3.2 Input vector

Both arms consume the same masked tensors the GNN sees, flattened. The three edge tensors enter in
the same normalized form the G2 link encoder uses, so the control differs in representation rather
than in input scaling:

| Block | Source | Length |
|---|---|---:|
| $\widetilde{\mathbf h}_{(l,r,k)}$ | masked `user_feature`, $R\,K_{\rm tot}\,2M(N+1)$ | 19,840 |
| $e_h$ | `edges` $\ell_1$-normalized over nodes | 160 |
| $e_l$ | `edges` $\ell_1$-normalized over RIS | 160 |
| $e_{\rm dir}$ | `direct_edges` $\ell_1$-normalized over nodes | 40 |
| served-user mask | `d1` only, AP identity | 40 |
| **Total** | `d0` / `d1` | **20,200 / 20,240** |

### 3.3 Capacity matching

Fairness rule from the plan: same optimizer, update budget, batch size and loss as G2; width adjusted
once to land within 10% of G2's effective parameter count. The declared choice is width 40 and
depth 3, meaning one input projection plus two width-preserving blocks with LeakyReLU, which is the
activation `model` already uses. Both arms land slightly above G2, so the control is not starved.

| Arm | Effective parameters | Ratio to G2 | Trunk share | CPU-side trainable |
|---|---:|---:|---:|---:|
| G2-150k | 868,421 | 1.0000 | — | 0 |
| `d0` | 895,565 | 1.0313 | 90.6% | 0 |
| `d1` | 891,917 | 1.0271 | 91.1% | 0 |

Roughly nine tenths of each arm's budget sits in the input projection. That is a property of the
control, not a tuning failure: without weight sharing over the RIS and node axes, a 20k-dimensional
input consumes the matched budget before any depth is bought. It is recorded here so the result is
read as a statement about the representation and not about a mis-sized network.

### 3.4 Training protocol

| Item | Value |
|---|---|
| System | $L=5$ APs, $M=2$ antennas/AP, $R=4$ RISs, $N=30$ elements/RIS, $K=8$ users/AP, $P_{\max}=15$ dBm |
| Budget | 150,000 iterations, matching the E06 G2 finalist |
| Optimizer | Adam, lr $10^{-4}$, weight decay $10^{-6}$, batch size 8 |
| Loss | unsupervised sum rate, unchanged |
| Training seed | 0 |
| Validation | every 1,000 iterations, 400 samples, seed 20260913; `best.pt` on the arm's deployment mode |
| Final evaluation | 400 samples, seed 20260915, the shared benchmark holdout |

The run artifact declares the width and depth above, the deployment mode of each arm, the
checkpoint-selection metric, the holdout seed, and the sample count. No hyperparameter was tuned on
the final holdout, and no arm was retrained after its numbers were seen. The current Git history does
not independently prove that the declaration predates launch, so this report does not call it an
immutable preregistration.

Deliberately out of scope here: the Stage-A cross-method comparison table, the signaling ledger, the
Stage-B topology stress test, and the model-based arms. This report covers the DNN family
only.

## 4. Results

### 4.1 Functional checks

| Check | `d0` | `d1` |
|---|---|---|
| Final gate, centralized | pass | pass |
| Final gate, paper-decentralized | not applicable | pass |
| Unit-modulus error at final evaluation | $1.79\times10^{-7}$ | $1.19\times10^{-7}$ |
| Effective parameters | 895,565 | 891,917 |
| Wall clock, RTX 5090 | 5.51 h | 5.64 h |

No NaN, projection fallback, or unit-modulus regression occurred in either run.

### 4.2 Final holdout, 400 samples, evaluation seed 20260915

| Method | Role | $R_{\rm cen}$ | $R_{\rm dec}$ | Cen−dec | 2-bit $R_{\rm dec}$ | Random phase | AP→CPU reals/AP–RIS |
|---|---|---:|---:|---:|---:|---:|---:|
| `d0`-150k | centralized DNN control | 5.5171 | n/a | n/a | 5.2225 (cen.) | 2.8062 (cen.) | none |
| `d1`-150k | decentralized DNN control | 6.5498 | 6.8192 | −0.2693 | 6.4681 | 3.6071 | 31 |
| G2-150k (E06) | proposed | 25.2594 | 24.5358 | 0.7236 | 22.8187 | — | 31 |
| R0-500k (E06) | paper GNN anchor | 24.9522 | 22.3305 | 2.6217 | 20.6797 | — | 120 |

Differences against the incumbent, in paper-decentralized continuous rate: `d1` is 17.7166 below
G2-150k and 15.5113 below R0-500k. In rounded 2-bit the shortfalls are 16.3506 and 14.2116.

### 4.3 Convergence state at the matched budget

Neither arm had flattened at 150k:

| Arm | 40k | 80k | 120k | 150k | Slope 50–100k | Slope 100–150k |
|---|---:|---:|---:|---:|---:|---:|
| `d0` (centralized) | 4.2932 | 4.7726 | 5.1501 | 5.3880 | +0.0998 | +0.0930 |
| `d1` (paper-dec.) | 4.4061 | 5.0917 | 5.6887 | 6.1878 | +0.1540 | +0.1630 |

Slopes are bps/Hz per 10k iterations from the 400-sample validation curve. `d0` decelerates slightly;
`d1` does not. Extrapolating `d1`'s terminal slope linearly, closing the 17.7166 gap would take about
1.09M further iterations, roughly seven times the matched budget, and linear extrapolation is
optimistic because the G2 and R0 curves both bend before their plateau.

The budget-free comparison is sharper. At 10k iterations E06 records G2 at 18.6514 and R0 at 11.2905
paper-decentralized. The validation-selected `d1` checkpoint reaches 6.8192, below both despite
fifteen times the updates.

### 4.4 Two secondary observations

`d1` beats `d0` in centralized rate, 6.5498 against 5.5171, although `d0` sees the same information
in one pass. The difference is the AP-shared structure: `d1` applies one trunk five times and averages
five proposals, which shares weight across APs and imposes a fixed aggregation.

`d1`'s paper-decentralized rate exceeds its own centralized rate, 6.8192 against 6.5498. The
inversion first appears at 15k iterations, is sustained from 19k, and holds in 88.7% of the 150 validation
points, and widens from +0.03 to +0.42 over training. G2 and R0 show the expected ordering.

## 5. Interpretation

- The screen's question is answered for this arm. Under matched inputs, constraints, loss, capacity,
  and updates, replacing the link-token graph with plain MLP blocks costs about 72% of the
  paper-decentralized rate. G2's advantage is not an artifact of a weak comparison class of one.
- The comparison that carries the weight is `d1` against G2, because they share the AP-to-CPU
  interface, the energy consensus, the deployment mode, and the holdout. The learned mappings differ
  for both phase proposals and active beamforming, so the result attributes the gap to the non-graph
  representation as a whole rather than to a single head.
- The `d0`-to-`d1` difference is **not** a centralized-to-decentralized information gap. It spans an
  architecture change as well, so it must never be reported as a retention figure comparable to G2's
  or R0's same-checkpoint gap. For the same reason `d1`'s own negative cen−dec gap is a property of
  that architecture, not evidence that decentralization helps in general. The most likely mechanism
  is that eq. (10) masking sparsifies a flat input that has no permutation structure to exploit,
  while in `d1`'s centralized mode all five APs receive an identical global vector and their
  proposals are correspondingly more correlated. That is a hypothesis; no intervention tested it.
- The non-convergence in Section 4.3 is the main thing that limits the claim. The defensible statement
  is that the DNN family is far behind at the matched budget and does not reach the GNNs' 10k rate at
  150k, not that its converged optimum has been measured.
- Beating this control is a screening result. E11 was skipped by owner decision; Section 8.6 of the
  benchmark plan still requires a corrected centralized/distributed model-based pair before Stage A
  is complete.

## 6. Limitations

- One training seed and one fixed topology, as in every training result in this repository so far.
  No paired confidence interval against G2 is quoted here, because the E06 reference numbers come
  from a different program; a single-program paired evaluation of all Stage-A arms is the step that
  produces those intervals.
- Neither arm converged within the matched budget, so the measured gap is a finite-budget result and
  not an estimate or bound of the asymptotic shortfall.
- The matched-capacity rule places about 91% of each arm's budget in the input projection. A reader
  could otherwise mistake the shortfall for under-sizing; the number is stated so that reading is
  closed off, but the coupling between the matching rule and the achievable width is real and cannot
  be removed without abandoning the rule.
- The flattened input is tied to the exact $(L, R, N, K, M)$ of T0. The arms cannot transfer to other
  network sizes, so no scaling statement can be made from them.
- This is a controlled DNN family in the role Chan et al. use a DNN control for. It is not a faithful
  reproduction of Hojatian et al., which has no RIS, and it must not be cited as one.
- `d0` has no decentralized deployment, so it contributes only a centralized reference point and
  cannot be shortlisted for the Stage-B topology stress test in that mode.
- Wall clock was measured while both arms shared one host. The workload is single-core per process on
  a 24-core machine, and a controlled restart produced identical validation metrics at one and eight
  BLAS threads, so this affects the timing figures only.

## 7. Reproduction and artifacts

Training was launched from the repository script, one arm per tmux window:

```bash
cd code/decentralized_ris
RIS_PYTHON=<env python> ARMS=d0 bash scripts/e10_dnn_baselines.sh
RIS_PYTHON=<env python> ARMS=d1 bash scripts/e10_dnn_baselines.sh
```

which expands to `train.py --arch {d0,d1} --n_iter 150000 --seed 0 --dnn_width 40 --dnn_depth 3
--eval_seed 20260915 --test_sample_final 400 --device cuda:0`.

| Evidence | Location |
|---|---|
| Model definition | `code/decentralized_ris/dnn.py` |
| Training entry point | `code/decentralized_ris/train.py` (`--arch d0`, `--arch d1`) |
| Launch script | `code/decentralized_ris/scripts/e10_dnn_baselines.sh` |
| Regression checks | `code/decentralized_ris/tests/test_dnn_baseline.py` |
| Declared protocol | `artifacts/decentralized_ris/e10_dnn_benchmark/preregistration.json` |
| Run bundles | `artifacts/decentralized_ris/e10_dnn_benchmark/dnn-{d0,d1}-150000_iter150000_seed0/` |
| Original last-checkpoint metrics and curves | each bundle's `summary.json` and `metrics.npz` |
| Validation-selected holdout | `best_holdout.json` and `best_holdout_paired.npz` |
| Training logs | `artifacts/decentralized_ris/e10_dnn_benchmark/dnn-{d0,d1}-150000.log` |

Both runs were executed on the remote GPU host `lab301-5090-tailscale` (RTX 5090,
`~/miniforge3/envs/decentralized-inference`) in tmux session `e10`, started 2026-09-18 00:51 and
finished 06:22 and 06:29 host time, and the complete bundles were copied back into the local E10
folder. The stored JSON retains the remote launch paths as provenance. The E06 reference rates quoted
in Section 4 are read from
[E06](./e06_graph_energy_training.md) and were produced by `experiments/graph_energy_screening.py` on
the same 400-sample holdout definition and evaluation seed 20260915.
