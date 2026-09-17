# E05 — Parameter-free local-energy consensus

## 1. Status

- Question: Can the learned pair scale be replaced by a stricter AP-local, parameter-free weight?
- Status: **Verified fixed-checkpoint result**
- Updated: 2026-09-16
- Scope: mature 500k R0 checkpoint, 3,200 locked samples, training seed 0
- Method: [Local-energy consensus](../decentralized_ris_methods.md#energy-consensus)
- Primary artifacts: `artifacts/decentralized_ris/e05_energy_consensus/`

## 2. Conclusion

Yes. The local-energy weight beats the learned pair scale by $1.6194\pm0.0586$ bps/Hz on the
locked setting while using less information. Independent reimplementation is bit-exact, locality and
normalization controls pass, and no numerical fallback occurs. This verifies an inference-time
interface replacement, not a new trained method; E06 tests end-to-end training.

## 3. Setup

All arms share the same beamformer and unit-modulus phase proposals. Only the consensus weight
changes:

| Arm | Weight |
|---|---|
| Equal | $1$ |
| Learned pair scale | $s_{l,r}=N^{-1}\sum_n\lVert z_{l,r,n}\rVert$ |
| Local energy | $E_{l,r}=\sum_{k\in\mathcal K_l}\lVert H_{l,r,k}\rVert_F^2$ |
| R0c | Per-element $\lVert z_{l,r,n}\rVert$ |

Calibration uses 800 samples with seed 20260919. Locked evaluation uses 3,200 samples / 400 batch
clusters with seed 20260922. Criteria were written before locked evaluation.

## 4. Results

| Arm | Sum rate | Difference from learned pair scale |
|---|---:|---:|
| Equal | $17.1888\pm0.1572$ | $-4.4092\pm0.1173$ |
| R0c per-element magnitude | $21.5489\pm0.2218$ | $-0.0491\pm0.0064$ |
| Learned pair scale | $21.5980\pm0.2228$ | — |
| **Local energy** | **$23.2173\pm0.2191$** | **$+1.6194\pm0.0586$** |

The cluster CI for local energy minus learned scale is $[+1.5041,+1.7346]$; local energy wins on
393/400 clusters and 2,581/3,200 samples. It also exceeds R0c by $1.6684\pm0.0577$.

| Operating point | Energy−learned | 95% CI |
|---|---:|---|
| 15 dBm, association 0.1 | +1.6194 | $[+1.5041,+1.7346]$ |
| 5 dBm, association 0.1 | +1.0684 | $[+0.9896,+1.1472]$ |
| 25 dBm, association 0.1 | +2.1540 | $[+2.0054,+2.3025]$ |
| 15 dBm, association 0.3 | +2.9243 | $[+2.7725,+3.0762]$ |

The three power points reuse identical proposals and weights; only the association-0.3 case creates a
second weight structure. This is four performance settings but only two distinct weighting contexts.

Controls:

- independent implementation reproduces all 3,200 per-sample rates bit-exactly;
- perturbing every other AP's channel leaves $E_{l,r}$ bit-identical;
- recomputation from complex channels agrees within $8.7\times10^{-8}$ relative error;
- pure rescaling of $E$ changes mean rate by at most $1.5\times10^{-8}$ bps/Hz;
- no NaN, negative weight, denominator-floor activation, or projection fallback is observed.

## 5. Interpretation

- Learned $s$ and energy are only moderately aligned (across-AP Spearman about 0.366) and select the
  same strongest AP only 36% of the time.
- Energy weighting is much sharper: effective AP count 1.53 versus 3.30 for learned scale.
- Correct AP assignment and relative magnitudes are both necessary; permutation or rank-only controls
  lose substantial rate.
- The learned-importance mechanism claim is rejected. The supported claim is a parameter-free,
  AP-local energy weight with the same $N+1$-scalar payload.

## 6. Limitations

- Fixed checkpoint, training seed, topology, $N$, and AP/RIS counts.
- Only two distinct weight structures were evaluated.
- Association masks are produced by a network-level comparison, although all arms share the mask.
- Mechanism probes after the locked result are exploratory; they do not prove causality.

## 7. Reproduction and artifacts

| Evidence | Location |
|---|---|
| Pre-registration and discovery diagnostics | `artifacts/decentralized_ris/e05_energy_consensus/discovery/` |
| Independent verification plan, raw results, and controls | `artifacts/decentralized_ris/e05_energy_consensus/verification/` |
| Compact machine-readable tables | `e05_energy_consensus/verification/summary_tables.md` |

Relevant programs are `experiments/mrc_proxy_diagnostic.py` and the
`experiments/energy_consensus_verification*.py` utilities.
