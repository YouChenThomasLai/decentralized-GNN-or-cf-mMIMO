# E01 — R0 baseline reproduction and training budget

## 1. Status

- Question: Can the paper baseline be reproduced, and how long must it train?
- Status: **Completed**
- Updated: 2026-09-16
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

## 7. Reproduction and artifacts

| Evidence | Location |
|---|---|
| 2k | `artifacts/decentralized_ris/e01_baseline_training/iter002000/` |
| 40k | `artifacts/decentralized_ris/e01_baseline_training/iter040000/` |
| 100k / 150k | `artifacts/decentralized_ris/e01_baseline_training/iter100000/`, `iter150000/` |
| 500k and final evaluation | `artifacts/decentralized_ris/e01_baseline_training/iter500000/M2_N30_L4_K8_P15.0_iter350000_seed0/run0/` |

Current training entry point: `code/decentralized_ris/train.py`. Historical checkpoints remain
loadable through `model.load_checkpoint`.
