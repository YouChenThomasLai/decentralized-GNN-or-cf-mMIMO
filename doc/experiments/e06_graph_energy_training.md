# E06 — Graph representation with energy consensus

## 1. Status

- Question: Does local-energy consensus remain effective when the model is trained end to end, and
  which graph representation should be used?
- Status: **Completed; G2-150k selected**
- Updated: 2026-09-16
- Scope: matched 10k screen, G1/G2 40k comparison, G2 continuation to 150k; seed 0
- Method: [G0/G1/G2](../decentralized_ris_methods.md#graph-variants)
- Primary artifacts: `artifacts/decentralized_ris/e06_graph_energy_training/`

## 2. Conclusion

Node-free link tokens (G1/G2) decisively outperform the RIS-node control G0. G1 and G2 are tied in
the primary paper-decentralized metric at 40k, but G2 protects the harsher own-only path. G2 continues
improving through 150k and exceeds the mature R0-500k reference by
$2.2053\pm0.2841$ bps/Hz in paper-decentralized continuous phase.

## 3. Setup

| Arm | Representation | Consensus | Effective parameters | Reals/AP–RIS |
|---|---|---|---:|---:|
| G0 | R0 RIS node and shared reduction | Local energy | 1,664,445 | 31 |
| G1 | Node-free per-RIS link tokens | Local energy | 819,269 | 31 |
| G2 | G1 + per-RIS context | Local energy | 868,421 | 31 |
| R0 | RIS node | Learned CPU reduction | 1,664,445 | 120 |

Training uses seed 0, the original optimizer state, and matched hyperparameters. Reported milestones
use the same 400-sample holdout and seed 20260915. G2-40k, 100k, and 150k are one continuation
trajectory, not separate experiments. The 40k raw summary retains an older in-memory $2N+1=61$
count; the maintained wire format sends $N$ angles plus one energy scalar, or 31 fp32 values. The
artifact migration note records this provenance distinction.

## 4. Results

10k screen:

| Arm | Centralized | Paper-dec. | Paper-dec. 2-bit | Own-only |
|---|---:|---:|---:|---:|
| G0 | 11.1712 | 10.7356 | 10.0320 | 9.2308 |
| G1 | 19.0608 | 18.4210 | 17.3728 | 8.6504 |
| G2 | 19.0753 | **18.6514** | **17.5178** | **14.3484** |
| R0-10k | 12.2638 | 11.2905 | 10.5469 | 8.9632 |

Long-training milestones:

| Checkpoint | Centralized | Paper-dec. | Paper-dec. 2-bit | Own-only |
|---|---:|---:|---:|---:|
| G1-40k | 23.5347 | 22.7843 | 21.1552 | 6.1246 |
| G2-40k | 23.2070 | 22.8673 | 21.3499 | 13.6408 |
| G2-100k | 24.7186 | 24.0195 | 22.2484 | 13.0252 |
| **G2-150k** | **25.2594** | **24.5358** | **22.8187** | 11.6458 |
| R0-500k | 24.9522 | 22.3305 | 20.6797 | n/a |

Key paired contrasts:

- G1-40k−G2-40k paper continuous: $-0.0830\pm0.1454$; no detectable context benefit in the
  primary setting.
- G2-150k−R0-500k paper continuous: $+2.2053\pm0.2841$.
- G2-150k−R0-500k paper 2-bit: $+2.1390\pm0.2555$.
- At G2-150k, replacing centralized phase with decentralized phase costs 0.2595, while replacing
  the beamformer costs 0.8222; the larger remaining error is in beamforming.

No NaN, projection fallback, or unit-modulus regression occurs. Paper-decentralized and own-only
energy weights are bit-identical, confirming that the weight uses only each AP's own served channels.

## 5. Interpretation

- Energy consensus is trainable end to end even though the weights themselves are fixed and detached.
- Removing the RIS node and using link tokens is the major representation gain.
- Per-RIS context does not improve the primary shared-UE setting at 40k, but prevents the severe
  own-only collapse of G1. No own-only-focused extension is justified under the deployment target.
- The G2-150k result closes this training screen; extending it to 500k is not the next useful step.

## 6. Limitations

- One training seed and topology.
- G0→G1 changes a representation package and capacity, not one isolated layer.
- G1/G2 context evidence at 40k is precise only for the current fixed holdout.
- Cross-topology and final multi-seed confirmation remain deferred until the method is frozen.

## 7. Reproduction and artifacts

| Evidence | Location |
|---|---|
| G0/G1/G2 matched 10k | `artifacts/decentralized_ris/e06_graph_energy_training/training_10k/` |
| G1 40k | `artifacts/decentralized_ris/e06_graph_energy_training/g1_iter040000/` |
| G2 40k/100k/150k | `artifacts/decentralized_ris/e06_graph_energy_training/g2_long_training/g2/iter*/` |
| Consolidation/provenance map | `artifacts/decentralized_ris/e06_graph_energy_training/g2_long_training/README.md` |
| Paired milestone JSON/NPZ | `artifacts/decentralized_ris/e06_graph_energy_training/g2_long_training/` |
| Generated full tables | `artifacts/decentralized_ris/e06_graph_energy_training/screening/report.md` |

The continuation script is `code/decentralized_ris/scripts/e06_g2_to150k.sh`; the paired evaluator is
`experiments/graph_energy_screening.py`. Stored configs retain the historical
`r1_shared + energy`／`r3a`／`r3b` identifiers as provenance; current code normalizes them to
G0／G1／G2 when loading.
