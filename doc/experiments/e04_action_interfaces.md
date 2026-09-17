# E04 — RIS action-interface screening

## 1. Status

- Question: What is lost when projection moves before AP aggregation, and can one pair-level scale
  recover it?
- Status: **Completed screening; superseded as the active direction by E05/E06**
- Updated: 2026-09-16
- Scope: matched 10k training plus post-training conversion of one mature 500k R0 checkpoint
- Method: [R0/R0c/R1 interfaces](../decentralized_ris_methods.md#action-interfaces)
- Primary artifacts: `artifacts/decentralized_ris/e04_action_interfaces/`

## 2. Conclusion

R0 and R0c are numerically equivalent. Equal circular consensus loses important magnitude
information, especially under decentralized inference. A single AP–RIS magnitude scalar can read a
mature R0 policy with no meaningful loss, but training that interface from scratch at 10k produces a
worse policy. The experiment established the importance of weighting, then E05 showed that the
learned scale can be replaced by simpler local energy.

## 3. Setup

All training runs use seed 0 and 10k iterations. Paired evaluations reuse identical channels within
each comparison. Zero-training conversions keep one checkpoint fixed and change only the
aggregation/projection rule. The pre-registered non-inferiority margin is 0.5 bps/Hz.

## 4. Results

Matched 10k R0 versus R1-Shared:

| Metric | R0 | R1-Shared | Difference ± clustered SE |
|---|---:|---:|---:|
| Centralized | 12.2638 | 11.9590 | $-0.3049\pm0.250$ |
| Paper-decentralized | 11.2905 | 9.6535 | $-1.6370\pm0.255$ |
| Paper-decentralized 2-bit | 10.5469 | 9.3004 | $-1.2466\pm0.236$ |

Counterfactual weighting ladder on the fixed R0 policy:

| Weighting | Paper-decentralized rate | Gain over equal |
|---|---:|---:|
| Equal | 10.0897 | — |
| Per-element magnitude (R0c) | 11.2905 | $+1.2008\pm0.1843$ |
| Pair-mean magnitude | **11.3698** | $+1.2802\pm0.1958$ |

The magnitude correlates with leave-one-AP-out contribution
($+0.4230\pm0.0343$) but not angular agreement ($-0.1382\pm0.0268$); it is therefore an
importance weight, not calibrated confidence.

Pre-registered zero-training conversion:

| Arm | Centralized | Paper-dec. | Paper-dec. 2-bit | R0−arm | Decision |
|---|---:|---:|---:|---:|---|
| R0 / R0c | 12.0682 | 11.2340 | 10.5363 | 0 | Equivalent |
| R1-Shared | 12.1815 | 10.0491 | 9.4929 | $+1.1849\pm0.1409$ | Failed |
| Pair magnitude | 12.0843 | **11.2592** | 10.5717 | $-0.0252\pm0.0404$ | Non-inferior |

Matched 10k retraining does not preserve this result: pair magnitude reaches 9.7943 versus R0's
11.1783 paper-decentralized rate, a loss of $1.3840\pm0.1362$.

On the mature 500k checkpoint and 3,200 samples:

| Arm | Centralized | Paper-dec. | Paper-dec. 2-bit | Difference from R0 |
|---|---:|---:|---:|---:|
| R0 / R0c | 23.5856 | 21.3812 | 19.8049 | 0 |
| R1-Shared | 23.5940 | 16.9721 | 15.7567 | $-4.4092\pm0.1147$ |
| Pair magnitude | 23.5826 | **21.4231** | **19.8388** | $+0.0419\pm0.0073$ |

## 5. Interpretation

- Projection before aggregation is not harmless; equal weighting discards policy-relevant radial
  information.
- One pair-level scalar is enough to preserve the mature R0 policy, but the same parameterization
  changes the optimization path when trained end to end.
- The small positive 500k difference is not claimed as superiority; E05 replaces its mechanism
  story with local-energy weighting.

## 6. Limitations

- Retraining evidence is a 10k, one-seed screen.
- Counterfactual diagnostics do not predict end-to-end retraining outcomes.
- R1-local and identity variants were not needed after E05/E06 established the stronger direction.

## 7. Reproduction and artifacts

| Evidence | Location |
|---|---|
| Matched training runs | `artifacts/decentralized_ris/e04_action_interfaces/training_10k/` |
| Fixed-checkpoint and paired evaluations | `artifacts/decentralized_ris/e04_action_interfaces/evaluation/` |
| Pre-registration | `artifacts/decentralized_ris/e04_action_interfaces/evaluation/prereg_r1_ap_ris_mag_zero_training.md` |

The removed 500-step smoke checkpoints and invalid old remote R0 copy are not required for the
conclusions above.
