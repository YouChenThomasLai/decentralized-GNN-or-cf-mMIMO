# E07 — AP-to-CPU RIS message codec and matched bit budget

## 1. Status

- Question: At the same AP→CPU payload per AP–RIS pair and the same single fusion round, does G2's
  directly executable phase message reach a higher paper-decentralized sum rate than the R0 anchor's
  latent message?
- Status: **Matched-budget comparison complete and confirmed on a fresh seed; one newly declared
  numerical control failed, so the frontier above 5-bit phase precision is not certified**
- Updated: 2026-09-20
- Scope: frozen G2-150k and R0-500k checkpoints, no retraining; 800 calibration samples (seed
  20260916); 400 evaluation samples / 50 clusters on the shared holdout seed 20260915 and on the
  fresh confirmation seed 20260923
- Method: [R0/R0c/proposal interfaces](../decentralized_ris_methods.md#payload) and
  [G2](../decentralized_ris_methods.md#graph-variants)
- Primary artifacts: `artifacts/decentralized_ris/e07_message_codec/`

## 2. Conclusion

A positive matched-budget rate difference is observed at the pre-declared 68-bit point, although the
complete experiment did not pass its numerical controls. The difference has not been shown to arise
from the message being executable phase. At 68 bits per AP–RIS pair, G2 exceeds the best R0-family wire format
at the same or lower budget by $1.5424\pm0.2363$ bps/Hz, reproduced as $1.5003\pm0.2472$ on a
fresh evaluation seed. Three other pre-declared rules fail on both seeds. G2 does **not** beat the *uncompressed* R0
anchor at 68 bits ($+0.1698\pm0.2638$; interval includes zero). G2 also loses *more* from
compression than R0 does, by $0.6629\pm0.1191$, so the matched-budget margin is smaller than
the $2.2053$ bps/Hz uncompressed G2–R0 gap. Inside the R0 backbone, where
the architecture is held fixed, no difference is detected between executable phase and a same-bit
Cartesian vector quantizer ($+0.0823\pm0.0579$, and $+0.0011\pm0.0300$ on the fresh seed).

What the executable-phase interface does beat decisively is R0's *own* $4N$ latent message, which is
the worst tested codec family at a small budget: 17.2749 bps/Hz at 75 generous bits against 20.9579 for
the phase interface at 64. That advantage disappears once the budget reaches 250–300 bits, and it
does not show a confirmed advantage against a Cartesian codec on the commuted logits at the tested budgets.
It is a much narrower structural claim than the one the plan proposed.

The pilot's failed identity control is resolved: message identity is now exact, and the
condition-aware projection bound passes at $2.38\times10^{-7}$. A *new* control — that a decoded
angle lands on its declared grid — failed, because it was normalized by the grid step, which halves
with each added bit while float32 precision does not. The primary 68-bit operating point is unaffected
($1.19\times10^{-7}$ grid steps), but the gate as declared covers every precision and it failed, so by
the declared stopping rule nothing was re-run and the high-precision frontier remains uncertified.

## 3. Setup

The encoder/quantizer/decoder sits on the wire: after an AP forms its message and before the CPU
fuses. Nothing is retrained. Both frozen policies are scored on **identical channel draws**, so every
contrast is paired within a batch cluster.

| Backbone | Wire message | Fusion | fp32 bits/pair |
|---|---|---|---:|
| G2-150k | $N$ phase angles + 1 local-energy scalar | energy circular mean | 992 |
| R0-500k | $4N$ latent phase feature | learned $\mathbf W_{\rm reduce}$ | 3840 (2400 generous) |
| R0c | $2N$ pre-projection logits | sum, then project | 1920 |

R0 is granted every interface G2 uses — the same calibrated phase grid, a pair scalar, a 2-D
Cartesian VQ, polar per-element magnitude and progressive residuals — all applied to its own logits.
The baseline is therefore never handicapped by being denied the compressed interface under test. All
132 arms use codebooks fitted once on the calibration split and then frozen and pre-shared, so no
codebook is charged to the payload.

Two accountings are kept for R0. `bits_per_pair` follows the maintained $4N$ count. The **generous**
count charges only $2N(1+1/R)$ reals, because the last $2N$ latent coordinates are the
RIS-independent `fe_AP` block and an AP could send them once per AP rather than once per AP–RIS pair.
A declared gate verifies that redundancy exactly (max spread across RIS = 0). Every decision uses the
generous count, so the anchor is compared at its cheapest defensible payload: 2400 rather than 3840
bits.

Declared before the locked run, in
`artifacts/decentralized_ris/e07_message_codec/prereg_matched_budget.json`:

| Rule | Statement |
|---|---|
| Primary point | 68 bits/pair ($N b_p + b_s$ with $b_p=2$, $b_s=8$), continuous actuation |
| D1 | G2 at 68 bits minus the **uncompressed** R0 anchor is positive, 95% cluster interval excluding zero |
| D2 | G2 at 68 bits minus the best R0-family arm at **at most** 68 generous bits, same interval rule |
| D3 | G2's compression penalty is smaller in magnitude than the penalty the anchor pays to reach the budget (difference in differences) |
| D4 | Within the R0 backbone alone, the executable-phase interface beats the best latent or Cartesian interface at the same budget |
| Confirmation | arms chosen on the discovery seed are fixed and replayed on a fresh seed; a rule stands only if it passes on both |

The replacement for the pilot's failed control is exact message-level identity (threshold 0), a
relative bound on the R0c polar round trip ($10^{-6}$), the condition-aware projection bound
$\max|\Delta\theta|\rho\le10^{-5}$, a rate-identity bound of $10^{-4}$ bps/Hz, and reproduction of the
four maintained E06 holdout rates to $10^{-3}$ bps/Hz. A secondary readout at 2-bit RIS actuation was
declared alongside the continuous primary. Setting: $L=5$ APs, $M=2$ antennas/AP, $R=4$ RIS, $N=30$
elements/RIS, $K=8$ users, $P_{\max}=15$ dBm, association threshold 0.1, training seed 0.

**Naming.** "2-bit" in this report means the AP→CPU *message* phase precision. The RIS *actuation*
resolution is named explicitly wherever it is used. The two are independent and must not be conflated.

## 4. Results

### 4.1 Controls

| Control | Value | Threshold | Result |
|---|---:|---:|---|
| Message identity (G2 angle round trip) | $0$ | $0$ | Pass |
| R0c polar round trip, relative | $8.26\times10^{-8}$ | $10^{-6}$ | Pass |
| Condition-aware projection $\max\|\Delta\theta\|\rho$ | $2.38\times10^{-7}$ | $10^{-5}$ | Pass |
| Infinite-rate codec vs library interface, rate | $1.91\times10^{-6}$ | $10^{-4}$ | Pass |
| Unit modulus | $1.79\times10^{-7}$ | $10^{-5}$ | Pass |
| `fe_AP` latent block RIS-independent | $0$ | $0$ | Pass |
| Inactive-AP consensus weight | $0$ | $0$ | Pass |
| E06 reference reproduction (4 rates) | $\le3.4\times10^{-7}$ | $10^{-3}$ | Pass |
| **Phase-grid legality** | $7.63\times10^{-6}$ | $10^{-6}$ | **Fail** |

The grid failure is consistent with float32 rounding. The largest angular deviation from a grid
point is $1.873\times10^{-7}$ rad — about one float32 ulp of $2\pi$ — and is of the same order across precision:

| $b_p$ | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deviation (rad) | $0$ | $1.9{\rm e}{-7}$ | $9.4{\rm e}{-8}$ | $1.9{\rm e}{-7}$ | $1.9{\rm e}{-7}$ | $1.9{\rm e}{-7}$ | $1.9{\rm e}{-7}$ |
| Deviation (grid steps) | $0$ | $1.2{\rm e}{-7}$ | $1.2{\rm e}{-7}$ | $4.8{\rm e}{-7}$ | $9.5{\rm e}{-7}$ | $1.9{\rm e}{-6}$ | $7.6{\rm e}{-6}$ |

Expressed in grid steps the quantity grows as the step shrinks, so it crosses $10^{-6}$ at
$b_p\ge6$. The declared gate still failed.

### 4.2 Pre-declared decisions, continuous actuation

| Rule | Discovery, seed 20260915 | Confirmation, seed 20260923 | Result |
|---|---:|---:|---|
| D1 vs uncompressed R0 (2400 bits) | $+0.1698\ [-0.3603,+0.6999]$ | $+0.1516\ [-0.3542,+0.6574]$ | **Fail** |
| D2 vs best R0 $\le68$ bits | $+1.5424\ [+1.0676,+2.0172]$ | $+1.5003\ [+1.0035,+1.9971]$ | **Pass** |
| D3 compression penalty | $-0.6629\ [-0.9023,-0.4236]$ | $-0.5812\ [-0.7460,-0.4164]$ | **Fail** |
| D4 same-backbone interface | $+0.0823\ [-0.0342,+0.1987]$ | $+0.0011\ [-0.0592,+0.0613]$ | **Fail** |

D2's baseline is `r0_pairmag_bp2_bs4` at 64 bits, selected on the discovery seed and then fixed. The
68-bit frontier row instead compares arms at exactly 68 bits, explaining its different contrast. D4's
pair is the executable-phase `r0_pairmag_bp2_bs4` against the Cartesian `r0c_vqgain_b2_bs8`.

### 4.3 Matched-budget frontier, continuous actuation

| Bits/pair | Best G2 | Best R0 | G2 − R0 |
|---:|---:|---:|---:|
| 38 | 18.8344 | 17.2990 | $+1.5354\pm0.2128$ |
| 62 | 20.8276 | 20.7689 | $+0.0587\pm0.3196$ |
| 64 | 22.4715 | 20.9579 | $+1.5136\pm0.2476$ |
| **68** | **22.5003** | **20.9566** | $\mathbf{+1.5436\pm0.2336}$ |
| 98 | 23.9842 | 22.0227 | $+1.9615\pm0.2467$ |
| 128 | 24.4187 | 22.3188 | $+2.0999\pm0.2775$ |
| 188 | 24.5207 | 22.3835 | $+2.1372\pm0.2825$ |
| 968 | 24.5162 | 22.3889 | $+2.1273\pm0.2831$ |

The point-estimate margin generally grows with the budget toward the uncompressed G2–R0 gap of
$2.2053$, so compression narrows the observed difference instead of widening it. These frontier
points were selected on the discovery seed and the high-precision points fail the global grid gate;
they are descriptive, not individually confirmed claims. The one exception is 62 bits, where the
energy scalar is cut to 2 bits and the difference becomes indistinguishable from zero; spending those
two extra bits on the scalar rather than on phase recovers the whole margin at 64. Below 38 bits G2
has no arm at all in this sweep; deleting energy collapses the frozen G2 (§4.6). References: G2 fp32 message 24.5358
at 992 bits, R0 anchor 22.3305 at 2400 generous bits, R0c 22.3305 at 1920 bits.

### 4.4 R0's own latent message is the expensive part at a small budget

| Generous bits/pair | 75 | 150 | 225 | 300 | 450 | 600 | 2400 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r0_latent` rate | 17.2749 | 20.9051 | 22.1119 | 22.3629 | 22.4084 | 22.4168 | 22.3305 |

Against this family the executable-phase interface wins on the same backbone, as a descriptive
contrast that was **not** pre-declared: $+1.4137\ [+1.1104,+1.7170]$ at 128 vs 150 bits and
$+3.6831\ [+3.2918,+4.0744]$ at 64 vs 75 bits, both reproduced on the confirmation seed. The gap
closes by 250–300 bits: $+0.0264\ [-0.1587,+0.2115]$ at 248 vs 300 on the discovery seed and
$+0.0942\ [+0.0443,+0.1441]$ on the confirmation seed.

### 4.5 Secondary readout: 2-bit RIS actuation

| Contrast | Discovery | Confirmation |
|---|---:|---:|
| G2 at 68 bits − uncompressed R0 | $+1.6632\ [+1.1195,+2.2069]$ | $+1.6051\ [+1.1288,+2.0814]$ |
| G2 at 68 bits − best R0 $\le68$ bits | $+1.8219\ [+1.2934,+2.3505]$ | $+1.7713\ [+1.2141,+2.3286]$ |
| Compression penalty difference | $-0.3171\ [-0.5619,-0.0723]$ | $-0.2534\ [-0.4873,-0.0195]$ |

Compressing G2's message to 2-bit phase costs only $0.4758\pm0.0772$ when the RIS itself is actuated
at 2 bits, against $2.0355\pm0.1044$ under continuous actuation.

### 4.6 The energy scalar is load-bearing

Dropping the single energy scalar and fusing with equal weights collapses the frozen G2 policy to
6.72–8.30 bps/Hz at every phase precision, including fp32 angles. Simply deleting the scalar from
this trained policy is therefore not viable; a retrained 30-scalar policy was not evaluated.

## 5. Interpretation

- **The 68-bit matched-budget contrast is positive on both seeds, subject to the failed global
  grid control.** D2 passes on both seeds by about $1.5$ bps/Hz, while D3 shows G2 paying the larger
  compression penalty. This is consistent with G2's representation-package advantage over G0
  surviving compression; G1 remains only the context ablation, and this result does not isolate a
  causal backbone effect. At 62 bits the interval
  includes zero, and the high-precision frontier is uncertified.
- **The interface hypothesis is not supported in its strong form.** D4 holds the architecture fixed
  and detects no advantage for the executable-phase interface over a same-bit Cartesian
  VQ on the commuted logits. The plan's structural argument — that a latent message must be
  compressed in a space the reducer never saw quantized — is only supported against R0's *native*
  $4N$ latent at small budgets, not against a competent codec applied to the algebraically equivalent
  $2N$ logits, and not at budgets above about 250 bits where the latent family rejoins R0's own
  frontier.
- **"Fewer bits and still better than the uncompressed anchor" is not established at continuous
  actuation.** D1 compares 68 bits against the anchor's 2400 generous bits, a 35× reduction (56× on
  the maintained $4N$ count), and its interval includes zero on both seeds. The same
  contrast is clearly positive in the pre-declared 2-bit-actuation readout, but promoting that to the
  headline would require re-declaring the actuation setting as primary and confirming again; it must
  not be swapped in after seeing the result.
- **The single scalar is load-bearing for the frozen policy.** Section 4.6 shows why simply deleting it
  is not a viable compression of the trained G2; retraining without it remains untested.
- The failed grid control repeats the pilot's failure mode at a different place: a threshold applied
  to a quantity whose normalizer shrinks. Both should be expressed in the physical unit — radians for
  angles, and a condition-weighted deviation for projections — rather than in units that vanish.

## 6. Limitations

- One declared control failed. The frontier at $b_p\ge6$ is uncertified pending the owner's decision
  on restating the grid control in radians; nothing was re-run to obtain a pass.
- D2's baseline and D4's pair were selected on the discovery seed. The confirmation seed re-tests the
  same fixed arms, which controls selection but not the choice of the 68-bit primary point.
- Section 4.4's same-backbone contrasts are descriptive; they were not pre-declared and carry no
  pass/fail status.
- The equal-weight ablation is a zero-training interface change on a policy trained with energy
  weighting, so it measures sensitivity, not what an equal-weighted G2 could learn.
- Codebooks are reconstruction-aware, never rate-aware or task-aware, and quantization is never seen
  during training by either arm.
- One training seed, one fixed topology, two evaluation seeds, 400 samples in 50 clusters.
- The comparison covers AP→CPU payload only. Rounds are equal at one for both arms, so nothing here
  separates the methods on latency; that axis belongs to the E12 comparison.

## 7. Reproduction and artifacts

Declared rules: `artifacts/decentralized_ris/e07_message_codec/prereg_matched_budget.json`.
Run with `code/decentralized_ris/scripts/e07_matched_budget.sh`; the implementation is
`code/decentralized_ris/experiments/matched_budget_codec.py`, which reuses the pilot's quantizers
from `experiments/message_codec_sweep.py`; plotting is `code/plot_matched_budget.py`. Outputs are
`matched_budget_config.json`, `matched_budget_results.json`, `matched_budget_paired.npz`,
`matched_budget_frontier.csv`, `matched_budget_frontier.png/pdf`, `matched_budget_run.log` and
`matched_budget_run_meta.json` (414 s on one RTX 5060).

Checkpoints, both frozen: G2-150k at
`e06_graph_energy_training/g2_long_training/g2/iter150000/checkpoints/iter150000.pt` and R0-500k at
`e01_baseline_training/iter500000/M2_N30_L4_K8_P15.0_iter350000_seed0/run0/models/model_final_run0.pt`.
Training seed 0, calibration seed 20260916, discovery seed 20260915, confirmation seed 20260923.

The superseded pilot — 116 R0-only codec points on evaluation seed 20260920, stopped by the failed
identity control — is retained in `config.json`, `results.json`, `paired.npz`, `frontier.csv`,
`frontier.png/pdf`, `run.log`, `run_meta.json` and `command.sh`, with `report.md` as its memo. Its
phase-precision gaps at 4/5/6/8 bits are filled by the run above; its polar and progressive families
are re-measured here under passing identity controls.
