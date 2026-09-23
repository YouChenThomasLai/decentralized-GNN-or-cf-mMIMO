# E02 — Centralized, paper-decentralized, and own-only input modes

## 1. Status

- Question: Which inference gap shrinks with training, and which remains tied to input visibility?
- Status: **Completed diagnostic**
- Updated: 2026-09-16
- Scope: 11 checkpoints, one shared 800-sample paired trajectory, seed 0 training
- Method: [CSI input modes](../decentralized_ris_methods.md#notation)
- Primary artifacts: `artifacts/decentralized_ris/e02_input_modes/`

## 2. Conclusion

From 150k to 500k, the centralized−paper-decentralized gap shrinks by 37.5%, while the
paper-decentralized−own-only gap does not shrink. Own-only is also unstable across nearby
checkpoints, so it must always be reported with a checkpoint identifier or a checkpoint average.

## 3. Setup

Each checkpoint and each input mode uses the same 800 channel samples. Within a batch, channel
generation occurs once and all modes reuse it. The three modes are:

- centralized: one model sees all relevant CSI;
- paper-decentralized: AP $l$ sees the paper's shared-UE neighborhood;
- own-only: AP $l$ sees only its own link block; this is an ablation, not the paper method.

## 4. Results

| Iterations | Centralized | Paper-dec. | Own-only | Cen−dec | Dec−own |
|---:|---:|---:|---:|---:|---:|
| 2k | 7.325 | 7.199 | 6.736 | 0.126 | 0.464 |
| 40k | 15.657 | 13.279 | 8.815 | 2.378 | 4.465 |
| 100k | 20.325 | 17.126 | 11.497 | 3.199 | 5.629 |
| 150k | 21.959 | 18.552 | 12.523 | **3.407** | 6.028 |
| 200k | 22.842 | 20.091 | 13.529 | 2.751 | 6.563 |
| 250k | 23.240 | 20.457 | 13.511 | 2.783 | 6.946 |
| 300k | 23.313 | 20.673 | 14.324 | 2.640 | 6.349 |
| 350k | 23.274 | 20.823 | 14.233 | 2.450 | 6.590 |
| 400k | 23.717 | 21.525 | 14.114 | 2.192 | 7.411 |
| 450k | 23.870 | 21.552 | 15.801 | 2.318 | 5.751 |
| 500k | 23.549 | 21.419 | 14.803 | **2.130** | 6.616 |

Key paired changes from 150k to 500k:

- Cen−dec: $-1.277\pm0.153$ bps/Hz, a 37.5% contraction.
- Dec−own: $+0.588\pm0.208$ bps/Hz; it becomes slightly larger.
- Own-only jumps $+1.687\pm0.155$ from 400k→450k and then falls
  $-0.998\pm0.150$ from 450k→500k.

Visibility and signaling under the fixed topology:

| Input mode | Visible AP–UE nodes per AP | UE→AP reals | Fronthaul reals |
|---|---:|---:|---:|
| Centralized | 40.000 | 11,879 | 12,215 |
| Paper-decentralized | 17.576 | 43,588 | 2,640 |
| Own-only | 4.790 | 11,879 | 2,640 |

## 5. Interpretation

- Longer centralized training helps the paper-decentralized path catch up, but does not remove the
  much larger own-only deficit.
- The own-only deficit still mixes missing information with centralized-training/local-inference
  mismatch; this experiment does not identify their individual shares.
- Paper-decentralized inference saves fronthaul relative to centralized inference, but its larger
  UE→AP feedback makes total signaling higher when air-interface feedback is counted.

## 6. Limitations

- Own-only was not retrained with its own input distribution.
- All checkpoints come from one training trajectory and one topology.
- The 800 samples give paired precision but do not add training-seed replication.

## 7. Reproduction and artifacts

`artifacts/decentralized_ris/e02_input_modes/` contains 11 `iter*/` cells, paired NPZ files,
`trajectory.json`, text summaries, logs, and log/linear-axis plots. The driver is
`code/decentralized_ris/scripts/e02_mode_trajectory.sh`.
