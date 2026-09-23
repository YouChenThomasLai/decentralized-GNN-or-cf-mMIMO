# E01 — R0 baseline reproduction and training budget

## 1. Status

- Question: Can the paper baseline be reproduced, and how long must it train?
- Status: **Completed**
- Updated: 2026-09-20
- Scope: training seed 0, fixed topology, $M=2$, $N=30$, $R=4$, $K=8$, $P_{\max}=15$ dBm
- Method: [R0](../decentralized_ris_methods.md#r0)
- Primary artifacts: `artifacts/decentralized_ris/e01_baseline_training/`

## 2. Conclusion

The baseline is reproducible, but the original 2k budget is inadequate. Under the current optimizer,
seed, and topology, useful gains flatten around 350–400k; 500k is the operational plateau used for
later comparisons. This is a stopping rule for this trajectory, not a multi-seed convergence proof.

## 3. Setup

All checkpoints use seed 0, batch size 8, Adam with learning rate $10^{-4}$ and weight decay
$10^{-6}$. Final evaluation uses 3,200 samples. Because one UE geometry is drawn per batch, the
independent statistical unit is the batch cluster, not an individual sample.

The implementation matches the paper's geometry, channel, association, GNN, and training settings.
The paper does not specify noise power; the implementation scales channels by $10^7$ and uses
`sigma = (2e-2)**2` in the SINR calculation. Cross-implementation comparisons must preserve this
choice.

## 4. Results

| Metric | 2k | 40k | 100k | 150k | 500k |
|---|---:|---:|---:|---:|---:|
| Centralized, continuous phase | 7.581 | 16.101 | 20.415 | 22.066 | **23.590** |
| Centralized, rounded 2-bit | 6.996 | 14.954 | 18.961 | 20.587 | 22.035 |
| Paper-decentralized, continuous phase | 7.436 | 13.746 | 17.034 | 18.369 | **21.399** |
| Paper-decentralized, rounded 2-bit | 6.837 | 12.841 | 15.811 | 17.122 | 19.876 |

| Window | Centralized gain / 10k | Paper-decentralized gain / 10k |
|---|---:|---:|
| 2k→40k | +2.242 | +1.661 |
| 40k→100k | +0.719 | +0.548 |
| 100k→150k | +0.330 | +0.267 |
| 150k→500k | **+0.044** | **+0.087** |

On the last 400–500k validation window, the slopes are $-0.022\pm0.032$ and
$-0.005\pm0.032$ bps/Hz per 10k for centralized and paper-decentralized inference. The
centralized−decentralized gap grows from 0.146 at 2k to 3.697 at 150k, then contracts to 2.190 at
500k.

A separate, matched short benchmark on `lab301-5090` (RTX 5090) timed the frozen legacy R0 model
and current vectorized R0 model with batch size 8, identical starting weights, three warm-up steps,
and 30 measured training steps each. Both used the current data generator and rate objective. The
legacy path took 0.5955 s/step and the current path 0.1157 s/step, an observed **5.15×** speed ratio.
The fixed-batch outputs agreed within $1.5\times10^{-8}$ for beamforming and $6.1\times10^{-7}$ for
RIS phase. This is a short-step estimate excluding validation and checkpoint I/O, not a measured
1,000-step run or a comparison of the complete historical and current pipelines.

Paper-figure requalification and the current rerun differ by at most 0.16 bps/Hz on the eight
headline metrics. The fast evaluator agrees with the original rate path within
$9.5\times10^{-7}$.

## 5. Interpretation

- Training budget explains a substantial part of the apparent centralized–decentralized gap at
  150k; the gap cannot be read as a purely structural local-information limit.
- Post-hoc 2-bit loss grows as the policy improves and reaches roughly 1.5 bps/Hz at 500k.
- Extending R0 beyond 500k is not justified unless the optimizer, objective, or method changes.

## 6. Limitations

- One training seed and one fixed topology.
- The plateau is operational, not a formal convergence result.
- Historical paper reproduction is cross-GPU rather than bit-exact.
- The 500k checkpoint predates the model refactor and must be loaded with `strict=False`; all live
  parameter shapes match, and extra keys belong only to removed dead modules.
- Only the five milestone checkpoints exist, so a finer curve comes from the training-log validation
  series rather than from re-evaluated checkpoints.
- The 2k run is a separate from-scratch run of seed 0, while 40k to 500k form one resumed lineage;
  at step 2,000 the two report 7.445 and 6.694 bps/Hz of validation rate. Figure 1 draws them as one
  trajectory.

## 7. Reproduction and artifacts

| Evidence | Location |
|---|---|
| 2k | `artifacts/decentralized_ris/e01_baseline_training/iter002000/` |
| 40k | `artifacts/decentralized_ris/e01_baseline_training/iter040000/` |
| 100k / 150k | `artifacts/decentralized_ris/e01_baseline_training/iter100000/`, `iter150000/` |
| 500k and final evaluation | `artifacts/decentralized_ris/e01_baseline_training/iter500000/M2_N30_L4_K8_P15.0_iter350000_seed0/run0/` |
| Matched RTX 5090 speed benchmark | `artifacts/decentralized_ris/e01_baseline_training/r0_speed_rtx5090.json` |

Current training entry point: `code/decentralized_ris/train.py`. Historical checkpoints remain
loadable through `model.load_checkpoint`.

The speed benchmark uses training seed 1234, 30 timed steps, batch size 8, and no evaluation seed
or checkpoint. Reproduce with
`python code/decentralized_ris/experiments/benchmark_r0_speed.py --steps 30 --warmup 3 --out artifacts/decentralized_ris/e01_baseline_training/r0_speed_rtx5090.json`.
The benchmark JSON records the GPU, PyTorch version, per-step timing, output-equivalence check, and
source hashes.

[Figure 1](../figures/r0_training_milestones.svg) is generated by `python code/plot_progress_report.py`
from the five final-evaluation files and from each segment's `run0/arrays/val_sum_rate_*_run0.npy`
series, which hold 4, 40, 30, 25, and 175 points at intervals of 500, 1,000, 2,000, 2,000, and 2,000
steps; each segment's start iteration follows from the length of its `losses_run0.npy`. The legacy
trainer measured those points on freshly drawn channels without a fixed evaluation seed, so they are
independent noisy estimates, with a mean absolute successive difference of 0.81 bps/Hz over
150k–500k. Their sample count is not recorded; about 100 per point is inferred from the legacy
`--test_sample_val` default.
