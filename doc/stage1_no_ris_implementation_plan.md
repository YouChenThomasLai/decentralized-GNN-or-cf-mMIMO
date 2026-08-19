# Stage 1 No-RIS Snapshot Baseline — Implementation Plan

## Status

- Revised: 2026-08-19 after the Stage 1A/1B result audit.
- Stage 1A implementation: complete. The direct-only code, compact test, and 5-seed antenna/power sweeps are complete; the unchanged Stage 0 noise places every method in the noise floor, so Stage 1A is retained as a negative control.
- Stage 1B calibration: complete. Configurable noise was added without changing the direct-channel model; the pre-registered `noise_power=1e-12` setting passed the seed-0 calibration and the 5-seed, 2000-iteration full-run numerical/learning gate.
- Result: Stage 1B is the snapshot source for Stage 2. The full evidence and limitations are recorded in `doc/decentralized_active_csi_experiment_report.md`.
- Provenance: the remainder of this file preserves the implementation plan that produced Stage 1A; the later Stage 1B calibration protocol is defined in `doc/decentralized_active_csi_experiment_plan.md`.

## Stage 1 Gate

Stage 1 answers one question: does the existing centralized-training/decentralized-inference GNN remain useful when only direct AP–UE channels exist?

The comparison must use the same direct-channel samples, association masks, noise model, and per-AP power constraint for:

1. centralized GNN inference;
2. decentralized GNN inference with the existing local-CSI visibility rule;
3. MRT;
4. local RZF.

Mobility, temporal state, stale CSI, feedback scheduling, dynamic association, and RL actions remain out of scope.

Gate decision (2026-08-19): Stage 1A completed the faithful-ablation gate but cannot answer the centralized/decentralized performance question because its rates and paired gaps are at the noise/float floor. Stage 1B passed the numerical/learning gate with complete finite artifacts and no checkpoint collapse. Its mean C−D gap is only 0.101% with an exact paired sign-flip $p=0.125$, so the result supports proceeding to Stage 2 but not a robust directional or equivalence claim.

## Findings from the Current Code

The previous plan was partly stale. Preserve these current fixes instead of reimplementing them:

- `trainer_2.py` already calls `gen_testing_data(..., regenerate_channels=False)`, so centralized and decentralized GNN evaluation reuse the same channel batch.
- `trainer_2.py` already converts the per-AP limit from dBm to watts before model normalization and loss evaluation.
- `discrete_mapping()` and random-phase evaluation are device-safe, but both disappear with the RIS output.
- The current phase merge is a PyTorch device operation; there is no separate CPU aggregation implementation to retain or replace.

Important behavior to preserve deliberately:

- five fixed AP positions on the radius-200 ring;
- random UEs within radius 100;
- direct-link RSSI association with threshold ratio `0.1`;
- the current local-CSI visibility expansion used by decentralized inference;
- six GNN update layers, hidden size 64, learned per-AP power coefficient, association masking, and per-AP normalization;
- current direct-channel scaling, noise power, optimizer, and learning rate for the first no-RIS comparison.

## Minimal File Scope

| Stage 1 file | Required change |
|---|---|
| `code/stage1/utils_return_indivial_rates.py` | Generate direct AP–UE channels only, evaluate direct-channel rates, and add MRT/RZF beamformers. |
| `code/stage1/data.py` | Remove RIS runtime objects and construct direct-channel GNN views without resampling between methods. |
| `code/stage1/model_2.py` | Remove RIS embeddings/messages/readout/merge and return only beamformers. |
| `code/stage1/trainer_2.py` | Remove phase evaluations, add seed/config controls, and compare all four methods on each stored batch. |
| `code/stage1/run_exp-v2.sh` | Point the antenna/power sweeps at the Stage 1 CLI and output names. |
| `code/stage1/test_stage1.py` | Add one compact assertion-based numerical and data-contract check during implementation. |

Do not copy or change `environment.yml`, `excel_helper.py`, or `plot-v2.py` yet. Reuse the existing environment. Adapt result summarization only after the Stage 1 metric schema is stable.

Do not add `__init__.py`, a configuration framework, new model classes, or new dependencies. The Stage 1 directory is a script directory, matching `code/`.

## Target Data Contract

Let `B` be batch size, `A=5` the number of APs, `K` the number of UEs, and `M` the antennas per AP.

| Value | Shape | Meaning |
|---|---|---|
| Direct channels | `[B, A, K, M]` complex | One direct channel for every AP–UE link. |
| Association mask | `[B, K, A]` boolean | Existing direct-RSSI threshold rule; fixed for the batch. |
| Centralized features | `[B, 1, A*K, 2*M]` real | AP-major virtual AP–UE nodes; the singleton axis preserves the current model indexing but is not a RIS node. |
| Decentralized features | list of `A` tensors `[B, 1, K, 2*M]` | One stored AP-local view per AP. |
| Centralized association mask | `[B, A*K]` boolean | AP-major flattening of the same association mask. |
| Beamformers | `[B, 2*M, A*K]` real | Real/imaginary coefficients in the current AP-major layout. |

Keeping the singleton feature-view axis avoids a broad rewrite of the model's sample and node indexing. It has no state, edges, phase, or learnable parameters. Remove it later only if a later architecture benefits from a three-dimensional input.

The loader generates one channel/association batch. Centralized features, decentralized features, MRT, and RZF are views or deterministic functions of that stored batch. No method may call channel generation during comparison.

## Implementation Sequence

### 1. Direct-channel utilities

In `utils_return_indivial_rates.py`:

- Keep `gen_location()`, `gen_fixed_location()`, `Channel`, and `user_pruning()` initially; avoid unrelated cleanup.
- Reduce `generate_channel()` to the existing direct AP–UE NLOS branch. Keep its path-loss constants and `scale = -7` behavior unchanged.
- Change `cal_loss()` to accept only `W`, stacked direct channels, AP count, and device. Remove `Theta`, cascaded `H`, phase conversion, and the RIS loop while retaining the current virtual-node signal/interference semantics and `noise_power = (2e-2)**2`.
- Implement MRT with `w_{a,k} = conj(h_{a,k})` for associated links under the current `h^T w` convention.
- Implement local RZF only over the UEs associated with each AP. Use `torch.linalg.solve`; do not form an explicit inverse. Use `alpha = K_a * noise_power / Pmax` as the initial regularization and record it in the run config.
- Zero unassociated columns, then normalize each AP block so its total power is at most `Pmax`.

Checkpoint: a hand-built direct channel and beamformer produce finite rates, and both baselines satisfy mask and power assertions.

### 2. Direct-only data loader

In `data.py`:

- Keep `Base_station`, but store only `channel_bs_user`; remove cascaded `H`, RIS arrays, RIS locations, RIS IDs, and `BS_RIS_association()`.
- Keep AP and UE placement and direct-RSSI association unchanged.
- Convert each complex direct channel to concatenated real/imaginary features only where the AP–UE association is true.
- Retain the current feature normalization axis so removing RIS is the only data-preprocessing change.
- Keep the current centralized AP-major concatenation and decentralized list construction.
- Expose the stored stacked channels and `[B,K,A]` association mask for rate evaluation and baselines.
- Preserve `regenerate_channels=False`; the trainer must request centralized data once and then reformat the stored batch for decentralized inference.

Checkpoint: direct channels and association masks are byte-identical before and after all four method inputs are built.

### 3. User-only GNN

In `model_2.py`, keep the existing class names and overall forward branches to limit trainer changes:

- Change the initial feature width from `2*M*(N+1)` to `2*M`.
- Let `initial_layer` return only the user embedding `uk`.
- In each `node_update_layer`, retain the user's prior state and max message from other visible user nodes. Remove the RIS-weighted mean message and change the update input width accordingly.
- Delete RIS embedding state `rl`, `RIS_readout_AP`, `RIS_merge`, `theta`, `e`, and `e_dir` from the runtime path.
- Keep `BS_readout`, the five per-AP coefficient networks, association masking, and AP-block normalization unchanged.
- Return only `W` from centralized and decentralized forward calls.
- Remove the unused `mean_ue` argument rather than carrying a dead Stage 0 parameter.

Checkpoint: centralized and decentralized forwards return `[B,2*M,A*K]`, all unassociated columns are zero, and every AP block satisfies the power limit.

### 4. Fair trainer and reproducible outputs

In `trainer_2.py`:

- Remove `N`, RIS count `L`, RIS filenames/locations, phase mapping, random-phase evaluation, and phase metric names.
- Add `--seed` and `--n_iter`. At the start of run `i`, seed Python, NumPy, and PyTorch with `seed + i` before constructing the loader and model.
- Save `config.json` per run with the effective seed, CLI values, AP count, association threshold, direct-channel scaling/noise constants, and RZF regularization rule.
- During evaluation, generate one batch, run centralized GNN, reformat without regeneration for decentralized GNN, then build MRT and RZF from the same stored channels and mask.
- Keep baselines evaluation-only; they do not enter the GNN loss or optimizer.
- Validate that evaluation sample counts are positive multiples of `batch_size` to avoid the current empty-mean case.
- Save final keys `centralized_gnn`, `decentralized_gnn`, `mrt`, `rzf`, and `centralized_minus_decentralized`.
- Retain the current checkpoint, loss-array, sum-rate-array, TensorBoard, and text/NumPy final-output layout where the field remains meaningful.

Checkpoint: a two-iteration CPU smoke run completes and writes finite metrics and its effective config.

### 5. Compact Stage 1 check

Add `test_stage1.py` with plain assertions and no test framework dependency. It must verify:

1. channel/features/model output follow the target shapes and no phase tensor is returned;
2. constructing all four method inputs does not alter or regenerate stored channels or masks;
3. unassociated beamformers are exactly zero;
4. every AP satisfies `sum_k ||w_{a,k}||^2 <= Pmax + 1e-6`;
5. loss, sum rate, and per-user rates are finite for all four methods;
6. two fresh CPU constructions with the same seed match within `1e-6`.

This one check is sufficient for Stage 1; do not build a larger test hierarchy yet.

### 6. Sweep script

Only after the smoke check passes:

- remove `--N` and `--L` from `run_exp-v2.sh`;
- keep the existing `M=1..5` and `Pmax=5..35 dBm` grids;
- use a Stage 1 result root and pass `--runs 5 --seed 0` for reportable sweeps;
- retain one-run commands separately for debugging rather than mixing them into the final sweep.

## Validation Commands

From `code/stage1/` after implementation:

```bash
python -m py_compile *.py
bash -n run_exp-v2.sh
python test_stage1.py
python trainer_2.py --M 2 --K 8 --Pmax 15 --batch_size 2 --runs 1 \
  --seed 0 --n_iter 2 --test_sample_val 2 --test_sample_final 2 \
  --device cpu --out_dir results_smoke
```

Full Stage 1 run:

```bash
python trainer_2.py --M 2 --K 8 --Pmax 15 --batch_size 8 --runs 5 \
  --seed 0 --n_iter 2000 --device cuda:0 --out_dir results_stage1
```

Do not commit `results_smoke/`, checkpoints, TensorBoard logs, NumPy outputs, or other generated artifacts.

## Acceptance Criteria

- All four methods use identical channel samples, association masks, noise, and per-AP power limits.
- Zero association-mask and power-limit violations occur in the sanity check and evaluation.
- At least five effective seeds are reported without selecting or dropping seeds.
- Report per-seed values plus mean and standard deviation for every method.
- The centralized-minus-decentralized gap has the same sign in at least four of five seeds; report failures instead of changing seeds.
- The antenna and transmit-power sweeps show reproducible trends; exact agreement with the RIS paper is not expected.
- Do not begin Stage 2 mobility work until the sanity check and five-seed comparison pass.

## Expected Outputs

| Output | Path |
|---|---|
| Effective run config | `results_stage1/.../run*/config.json` |
| Final metrics | `results_stage1/.../run*/final_eval/final_eval_run*.npy` |
| Readable final metrics | `results_stage1/.../run*/final_eval/final_eval_run*.txt` |
| Model checkpoint | `results_stage1/.../run*/models/model_final_run*.pt` |
| Training/validation arrays | `results_stage1/.../run*/arrays/` |
| TensorBoard logs | `results_stage1/.../run*/logs/` |

## Explicitly Deferred

- vectorizing the current sample/user loops;
- renaming legacy classes and files;
- changing channel, noise, association, optimizer, or GNN hyperparameters;
- adapting Excel/plot scripts;
- mobility, stale CSI, feedback budgets, dynamic association, RL, and architecture ablations.

These changes may be worthwhile later, but including them now would make the no-RIS comparison harder to attribute and review.
