# E02 — Centralized, paper-decentralized, and own-only input modes

## 1. Status

- Question: Which inference gap shrinks with training, and which remains tied to input visibility?
- Status: **Completed diagnostic, with a frozen G2/R0 endpoint continuation**
- Updated: 2026-09-20
- Scope: 11-checkpoint R0 trajectory on 800 paired samples; G2-150k/R0-500k endpoint on 400
  newly paired samples; training seed 0
- Method: [CSI input modes](../decentralized_ris_methods.md#notation)
- Primary artifacts: `artifacts/decentralized_ris/e02_input_modes/`

## 2. Conclusion

From 150k to 500k, the centralized−paper-decentralized gap shrinks by 37.5%, while the
paper-decentralized−own-only gap does not shrink. Own-only is also unstable across nearby
checkpoints, so it must always be reported with a checkpoint identifier or a checkpoint average.
In the matched-seed frozen-checkpoint continuation, G2's own-only visibility penalty is *larger*
than R0's by $4.7404\pm0.3469$ bps/Hz (95% batch-cluster interval $[4.0605,5.4204]$).
Thus G2 preserves a smaller centralized−decentralized gap with the paper's shared-UE CSI, but
does not preserve it when that extra CSI is removed completely.

## 3. Setup

Each checkpoint and each input mode uses the same 800 channel samples. Within a batch, channel
generation occurs once and all modes reuse it. The three modes are:

- centralized: one model sees all relevant CSI;
- paper-decentralized: AP $l$ sees the paper's shared-UE neighborhood;
- own-only: AP $l$ sees only its own link block; this is an ablation, not the paper method.

The continuation freezes E06 G2-150k (`iter150000.pt`) and E01 R0-500k
(`model_final_run0.pt`), then evaluates centralized, paper-decentralized, and own-only modes on
the same 400 canonical-topology channel samples (evaluation seed 20260920, 50 batches of eight).
Only the inference visibility mask changes between paper and own-only. The predeclared contrast is
$(R_{\rm paper}-R_{\rm own})_{\rm G2}-(R_{\rm paper}-R_{\rm own})_{\rm R0}$; a negative interval
would support the proposed G2 robustness at this extreme endpoint.

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

Frozen-checkpoint continuation on the same 400 samples:

| Model | Centralized | Paper-dec. | Own-only | Cen−paper | Cen−own | Paper−own |
|---|---:|---:|---:|---:|---:|---:|
| G2-150k | 23.3572 | 22.9587 | 11.4554 | 0.3985 | 11.9018 | 11.5033 |
| R0-500k | 23.0898 | 20.7113 | 13.9484 | 2.3785 | 9.1414 | 6.7629 |

The paired difference of visibility penalties, G2−R0, is $+4.7404\pm0.3469$ bps/Hz with 95%
interval $[4.0605,5.4204]$. Mean visible AP–UE nodes per AP fall from 17.012 in paper mode to
4.697 in own-only mode for both methods. The centralized and paper batch arrays exactly equal
E14's T0 arrays for both models, confirming that the only new intervention is own-only visibility.

## 5. Interpretation

- Longer centralized training helps the paper-decentralized path catch up, but does not remove the
  much larger own-only deficit.
- The own-only deficit still mixes missing information with centralized-training/local-inference
  mismatch; this experiment does not identify their individual shares.
- Paper-decentralized inference saves fronthaul relative to centralized inference, but its larger
  UE→AP feedback makes total signaling higher when air-interface feedback is counted.
- The proposed ranking reverses at the own-only endpoint: G2 has a smaller C−D gap than R0 with
  full paper visibility, but a larger gap with no shared-UE cross-AP CSI. Its trained policy depends
  more strongly on that extra information at this checkpoint. This is an inference-time sensitivity,
  not evidence that G2 would remain worse after training for the restricted view.
- The own-only endpoint changes the information mode, so it cannot stand in for a topology that
  naturally supplies less shared-UE CSI. [E14's visibility pair](./e14_topology_stress.md) runs
  that test instead: holding the paper-decentralized mode fixed and moving the AP ring from 140 m
  to 350 m raises the visible-link fraction from 0.2860 to 0.6170, and G2 keeps the smaller
  relative gap at every point, 4.25%→2.10% against R0's 14.26%→6.77%. The reversal above is
  therefore a property of the ablation endpoint, not of naturally sparse visibility.

## 6. Limitations

- Own-only was not retrained with its own input distribution.
- All checkpoints come from one training trajectory and one topology.
- The 800 samples give paired precision but do not add training-seed replication.
- The G2/R0 continuation tests the two visibility endpoints only. It does not establish how gaps
  change under partial CSI removal or a model retrained for reduced visibility; E14's visibility
  pair covers the layout axis under an unchanged information mode.

## 7. Reproduction and artifacts

`artifacts/decentralized_ris/e02_input_modes/` contains 11 `iter*/` cells, paired NPZ files,
`trajectory.json`, text summaries, logs, and log/linear-axis plots. The driver is
`code/decentralized_ris/scripts/e02_mode_trajectory.sh`.

The continuation is in
`artifacts/decentralized_ris/e02_input_modes/g2_r0_visibility_seed20260920/`: pre-registration,
`results.json`, and the 50 paired batch rates per arm/mode in `results.npz`. From
`code/decentralized_ris/`, run:

```bash
python -m experiments.visibility_pair --samples 400 --eval_seed 20260920 --device cuda:0 --out ../../artifacts/decentralized_ris/e02_input_modes/g2_r0_visibility_seed20260920/results.json
```
