# E05 — Parameter-free local-energy consensus

## 1. Status

- Question: Can the learned pair scale be replaced by a stricter AP-local, parameter-free weight?
- Status: **Verified fixed-checkpoint result**
- Updated: 2026-09-18
- Scope: mature 500k R0 checkpoint, 3,200 locked samples, training seed 0; extended in Section 4.1
  with a finite-budget oracle-style weight diagnostic on the frozen G2-150k checkpoint
- Method: [Local-energy consensus](../decentralized_ris_methods.md#energy-consensus)
- Primary artifacts: `artifacts/decentralized_ris/e05_energy_consensus/`

## 2. Conclusion

Yes. The local-energy weight beats the learned pair scale by $1.6194\pm0.0586$ bps/Hz on the
locked setting while using less information. Independent reimplementation is bit-exact, locality and
normalization controls pass, and no numerical fallback occurs. This verifies an inference-time
interface replacement, not a new trained method; E06 tests end-to-end training.

The later oracle-weight extension measures the remaining headroom found by a declared finite-budget
search over the weight variables. On the frozen G2-150k checkpoint the parameter-free energy weight
recovers $0.9301\pm0.0047$ of the rate gap between equal weighting and the best per-sample weight found
by that search, and its active-AP ranking is directionally aligned with the fitted weights. All six
declared controls pass on both seeds. The search improves energy by $1.2544\pm0.0960$ bps/Hz, but it
is non-convex and does not certify either global optimality or total remaining headroom.

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

### 3.1 Oracle-weight extension

The extension answers a different question about the same weight: not whether energy beats the
learned scale, but how close energy is to a strong fitted weight under a declared search. It runs on the frozen
[E06](./e06_graph_energy_training.md) G2-150k checkpoint under the paper-decentralized view, so the
arms share one beamformer, one set of transmitted phase proposals, and one channel realization; only
the consensus weight changes.

| Arm | Weight | Deployable |
|---|---|---|
| Equal | $1$ | yes |
| Energy | $E_{l,r}$ | yes, this is the deployed rule |
| Oracle-style reference | Best realized sum rate found by the declared per-sample search | no |

The reference searches the $L\times R=20$ non-negative weights of each channel sample directly against
the true continuous-phase sum rate, parameterized as $w=\operatorname{softplus}(u)$ with 300 Adam
steps at learning rate 0.05 from both an equal and an energy initialization, retaining the best
per-sample iterate. It therefore sees the realized rate and is reported separately from deployable
arms rather than as a competitor. Because this is a finite-budget local optimizer, it is not a
certified upper bound. Its 2-bit row evaluates the same continuous-optimized weights after actuation
quantization; it is not a separately optimized 2-bit oracle.

Motivation and boundary conditions for this comparison are in the method report's
[MLE argument](../decentralized_ris_methods.md#energy-consensus): the deployed circular consensus is
the maximum-likelihood fusion of the common phase exactly when the weights are proportional to each
AP's proposal concentration, so the open question is whether $E_{l,r}$ is a good concentration proxy.
Thresholds were declared in the `DECISION` constant of
`experiments/consensus_weight_oracle.py` before the run: gap recovery at least 0.50 with a 95%
clustered interval excluding zero, and a tie-corrected Spearman correlation of at least 0.30 over
active APs only.

## 4. Results

Unless an interval is written explicitly, every $\pm$ value below is one clustered standard error.

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

### 4.1 Oracle-weight results

Primary seed 20260915, 400 paired samples / 50 clusters, frozen G2-150k:

| Arm | Continuous | 2-bit |
|---|---:|---:|
| Equal | $8.2981\pm0.3661$ | $7.7650\pm0.3436$ |
| **Energy (deployed)** | $\mathbf{24.5358\pm0.6270}$ | $\mathbf{22.8187\pm0.5563}$ |
| Fitted reference (non-deployable) | $25.7902\pm0.6710$ | $23.6028\pm0.5788$ |

| Quantity | Continuous | 2-bit |
|---|---:|---:|
| Gap recovery $(E-\text{equal})/(\text{fit}-\text{equal})$ | $0.9301\pm0.0047$ | $0.9513\pm0.0039$ |
| Energy $-$ equal | $16.2377\pm0.3782$ | $15.0537\pm0.3382$ |
| Fitted reference $-$ energy | $1.2544\pm0.0960$ | $0.7841\pm0.0686$ |
| Spearman $(\bar E, w^{\rm fit})$ across active APs | $0.6030$, CI $[0.5799,0.6261]$ | — |
| Mean largest AP share | energy $0.7888$, fitted reference $0.6439$ | — |

The replication seed 20260918 preserves every conclusion: gap recovery $0.9340\pm0.0050$, Spearman
$0.6068$, fitted reference minus energy $1.1256\pm0.0961$.

Controls, all passing on both seeds:

- the diagnostic's energy arm reproduces the deployed fused phase exactly (max deviation $0$), and its
  continuous and 2-bit rates, 24.5358 and 22.8187, match the independently produced E06 G2-150k
  holdout values;
- the fast rate path agrees with the simulator loss within $3.8\times10^{-6}$;
- unit-modulus error is at most $1.8\times10^{-7}$;
- the fitted reference never falls below the energy arm on any sample, the smallest margin being
  $+0.0458$; this is guaranteed by retaining the energy initialization and checks the implementation;
- the rate gradient with respect to the weight parameters is non-zero, so the optimization path is
  real rather than silently detached;
- no non-finite value occurs.

## 5. Interpretation

- Learned $s$ and energy are only moderately aligned (across-AP Spearman about 0.366) and select the
  same strongest AP only 36% of the time.
- Energy weighting is much sharper: effective AP count 1.53 versus 3.30 for learned scale.
- Correct AP assignment and relative magnitudes are both necessary; permutation or rank-only controls
  lose substantial rate.
- The learned-importance mechanism claim is rejected. The supported claim is a parameter-free,
  AP-local energy weight with the same $N+1$-scalar payload.
- The oracle-weight extension supports the concentration-proxy reading of $E_{l,r}$ within the
  declared search: a parameter-free, strictly local scalar captures about 93% of the
  equal-to-best-found gap, and the fitted reference adds roughly 5% of the deployed rate. This shows
  limited headroom was found by this search, not that every trainable CPU-side weight has little room.
- The agreement is directional but not exact. A Spearman correlation of 0.60 means energy usually,
  not always, ranks the active APs as the fitted reference does, and energy concentrates more than the
  reference: its mean largest AP share is 0.79 against 0.64. A flatter transform of the same
  strictly local scalar is therefore the natural follow-up, but it must be declared and confirmed on
  a fresh evaluation seed rather than fitted on this holdout.
- After 2-bit actuation quantization, the advantage of the continuous-optimized fitted weights falls
  from 1.2544 to 0.7841 bps/Hz. This is descriptive; because the weights were not optimized for the
  quantized objective, it does not bound the best 2-bit weight choice.

## 6. Limitations

- Fixed checkpoint, training seed, topology, $N$, and AP/RIS counts.
- Only two distinct weight structures were evaluated.
- Association masks are produced by a network-level comparison, although all arms share the mask.
- Mechanism probes after the locked result are exploratory; they do not prove causality.
- The oracle-style arm is a per-sample, non-convex optimizer against the realized rate on 20 scalars,
  so it can fit the sample, but 300 Adam steps from two initializations do not certify the global
  optimum. The fitted-reference minus energy gap is therefore a lower bound on the headroom available
  to the unknown global optimum, while recovery relative to the fitted reference can overstate
  recovery relative to that optimum.
- The fitted search reuses the frozen proposals and only diagnoses the inference-time weight choice;
  a network retrained under a different weight would produce different proposals, so nothing here
  constrains the joint design.
- The extension uses one checkpoint, one topology, and two evaluation seeds under one training seed.
- The MLE argument in the method report assumes von Mises proposal errors that are independent across
  APs. Neither assumption is tested here; only its consequence for the rate is measured.

## 7. Reproduction and artifacts

| Evidence | Location |
|---|---|
| Pre-registration and discovery diagnostics | `artifacts/decentralized_ris/e05_energy_consensus/discovery/` |
| Independent verification plan, raw results, and controls | `artifacts/decentralized_ris/e05_energy_consensus/verification/` |
| Compact machine-readable tables | `e05_energy_consensus/verification/summary_tables.md` |
| Oracle-weight summary, declared criteria, and paired raw arrays | `artifacts/decentralized_ris/e05_energy_consensus/consensus_weight_oracle/` |

Relevant programs are `experiments/mrc_proxy_diagnostic.py` and the
`experiments/energy_consensus_verification*.py` utilities.

The oracle-weight extension is `experiments/consensus_weight_oracle.py`. From
`code/decentralized_ris/`:

```bash
python -m experiments.consensus_weight_oracle \
  --run ../../artifacts/decentralized_ris/e06_graph_energy_training/g2_long_training/g2/iter150000 \
  --checkpoint iter150000.pt --samples 400 \
  --eval_seed 20260915 --replication_seed 20260918 --device cuda:0
```

Training seed 0, evaluation seeds 20260915 and 20260918, 400 samples each, 300 oracle steps at
learning rate 0.05, about 135 s per seed on one RTX 5060 Laptop GPU.
