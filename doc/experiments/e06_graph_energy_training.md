# E06 — Graph representation with energy consensus

## 1. Status

- Question: Does local-energy consensus remain effective when G2 is trained end to end, and what
  effects can the G0 representation control and G1 no-context ablation attribute?
- Status: **Completed; G2-150k retained as the compute-efficient finalist**
- Updated: 2026-09-21
- Scope: matched 10k ablation screen, G1/G2 40k comparison, G2 continuation through 300k; seed 0
- Method: [G2 and its ablations](../decentralized_ris_methods.md#graph-variants)
- Primary artifacts: `artifacts/decentralized_ris/e06_graph_energy_training/`

## 2. Conclusion

G2 is the selected method and G1 is retained only as its no-context ablation. The node-free
representation package decisively outperforms the RIS-node control G0. G1 and G2 are tied in the
primary paper-decentralized metric at 40k, but the context retained by G2 protects the harsher
own-only path. G2-150k exceeds the mature R0-500k reference by $2.2053\pm0.2841$ bps/Hz in
paper-decentralized continuous
phase. Extending the same trajectory to 300k raises that metric by only $0.1540\pm0.1030$ and the
2-bit metric by $0.1372\pm0.1006$ relative to 150k. The point estimates improve, but neither change
reaches two clustered SE, so the extra 150k steps do not materially change the primary conclusion.

## 3. Setup

| Arm | Role | Representation | Consensus | Effective parameters | Reals/AP–RIS |
|---|---|---|---|---:|---:|
| G0 | Representation control | R0 RIS node and shared reduction | Local energy | 1,664,445 | 31 |
| G1 | G2 no-context ablation | Node-free per-RIS link tokens | Local energy | 819,269 | 31 |
| G2 | Proposed method | Node-free link tokens + per-RIS context | Local energy | 868,421 | 31 |
| R0 | Mature baseline | RIS node | Learned CPU reduction | 1,664,445 | 120 |

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
| G2-200k | 25.4714 | 24.4962 | 22.7975 | 11.6580 |
| G2-250k | 25.6292 | 24.5770 | 22.8737 | 12.7604 |
| G2-300k | 25.5478 | **24.6898** | **22.9559** | **13.4993** |
| R0-500k | 24.9522 | 22.3305 | 20.6797 | n/a |

Key paired contrasts:

- G2-40k−G1-40k paper continuous: $+0.0830\pm0.1454$; no detectable context benefit in the
  primary setting.
- G2-150k−R0-500k paper continuous: $+2.2053\pm0.2841$.
- G2-150k−R0-500k paper 2-bit: $+2.1390\pm0.2555$.
- G2-300k−G2-150k paper continuous: $+0.1540\pm0.1030$.
- G2-300k−G2-150k paper 2-bit: $+0.1372\pm0.1006$.
- G2-300k−G2-150k own-only continuous: $+1.8535\pm0.1740$.
- G2-300k−R0-500k paper continuous: $+2.3593\pm0.3056$.
- G2-300k−R0-500k paper 2-bit: $+2.2762\pm0.2617$.
- At G2-150k, replacing centralized phase with decentralized phase costs 0.2595, while replacing
  the beamformer costs 0.8222; the larger remaining error is in beamforming.

No NaN, projection fallback, or unit-modulus regression occurs. Paper-decentralized and own-only
energy weights are bit-identical, confirming that the weight uses only each AP's own served channels.

## 5. Interpretation

- Energy consensus is trainable end to end even though the weights themselves are fixed and detached.
- G2's full representation package, rather than an isolated context term, carries the major gain
  over the RIS-node control.
- The G1 ablation shows that per-RIS context does not improve the primary shared-UE setting at 40k,
  but prevents a severe own-only collapse. This result limits the context-specific claim; it does
  not promote G1 to a co-finalist or justify an own-only-focused extension.
- The 300k endpoint has the highest primary point estimates, but doubling the 150k training budget
  yields less than two clustered SE of improvement. G2-150k remains the compute-efficient finalist;
  G2-300k is retained as the highest observed endpoint rather than evidence of substantial remaining
  primary-setting headroom.
- Own-only inference recovers strongly after 200k, reaching 13.4993 at 300k. This does not change the
  selection target because own-only remains a secondary robustness diagnostic.

## 6. Limitations

- One training seed and topology.
- G2 versus G0 changes a representation package and capacity, not one isolated layer.
- The G1 no-context ablation at 40k is precise only for the current fixed holdout.
- The same fixed holdout was reused to inspect several milestones, so the 200k/250k/300k comparison
  is a trajectory diagnostic rather than independent confirmation.
- E14 tests frozen G2 on deterministic topology shifts, but E06 itself contains no retraining across
  topology or multi-seed confirmation.

## 7. Reproduction and artifacts

| Evidence | Location |
|---|---|
| G0/G1/G2 matched 10k | `artifacts/decentralized_ris/e06_graph_energy_training/training_10k/` |
| G1 40k | `artifacts/decentralized_ris/e06_graph_energy_training/g1_iter040000/` |
| G2 40k/100k/150k | `artifacts/decentralized_ris/e06_graph_energy_training/g2_long_training/g2/iter{040000,100000,150000}/` |
| G2 200k/250k checkpoints and 300k bundle | `artifacts/decentralized_ris/e06_graph_energy_training/g2_long_training/g2/iter300000/` |
| 150k/200k/250k/300k paired evaluation | `artifacts/decentralized_ris/e06_graph_energy_training/g2_long_training/evaluation/g2_extension_150k_300k_seed20260915.json` and `_paired.npz` |
| Consolidation/provenance map | `artifacts/decentralized_ris/e06_graph_energy_training/g2_long_training/README.md` |
| Paired milestone JSON/NPZ | `artifacts/decentralized_ris/e06_graph_energy_training/g2_long_training/` |
| Generated full tables | `artifacts/decentralized_ris/e06_graph_energy_training/screening/report.md` |

The continuation scripts are `code/decentralized_ris/scripts/e06_g2_to150k.sh` and
`code/decentralized_ris/scripts/e06_g2_to300k.sh`; the paired evaluator is
`experiments/graph_energy_screening.py`. Stored configs retain the historical
`r1_shared + energy`／`r3a`／`r3b` identifiers as provenance; current code normalizes them to
G0／G1／G2 when loading.

The 150k→300k run completed on `lab301-5090-tailscale` in 19,640.5 seconds. It was launched at
2026-09-18 10:10 +08 in tmux session `e06_g2_300k` and resumed the SHA-256
`f568d5b3…ed240a8c` endpoint checkpoint with its optimizer state, training seed 0, and the original
learning rate. The launch command was
`RIS_PYTHON=~/miniforge3/envs/decentralized-inference/bin/python ARTIFACT_ROOT=~/ThomasLai/artifacts/decentralized_ris DEVICE=cuda:0 G2_150K_CHECKPOINT=~/ThomasLai/artifacts/decentralized_ris/graph_energy_150k/graph-energy-g2-total150k-resume100k_iter50000_seed0/checkpoints/iter150000.pt bash scripts/e06_g2_to300k.sh`.
The `checkpoints` window in the same tmux session ran `scripts/watch_checkpoints.py` to retain and
verify the 200k and 250k milestones before the launch script retained the 300k endpoint. Evaluation
uses 400 paired samples with seed 20260915, matching all earlier E06 milestones.
