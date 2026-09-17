# E07 — AP-to-CPU RIS message codec pilot

## 1. Status

- Question: What bits–rate frontier is obtained when AP messages are quantized before consensus?
- Status: **Provisional; one pre-declared control failed**
- Updated: 2026-09-16
- Scope: fixed 500k R0 checkpoint, 800 calibration samples, 400 evaluation samples / 50 clusters
- Method: R0c polar messages and pair-scale compression in the [method report](../decentralized_ris_methods.md)
- Primary artifacts: `artifacts/decentralized_ris/e07_message_codec/`

## 2. Conclusion

The pilot identifies phase-direction resolution as the dominant bottleneck and finds little value in
per-element magnitude residuals once a pair scale is sent. However, the run stopped because an
infinity-bit identity control used an ill-conditioned post-projection maximum-error threshold. Until
the owner chooses a condition-aware replacement and the experiment is rerun, all frontier values are
provisional and cannot support a go/no-go claim.

## 3. Setup

The encoder/quantizer/decoder sits after each AP produces $z_{l,r,n}$ and before CPU consensus. The
pilot evaluates 116 codec points plus four fp32 references:

| Family | Bits per AP–RIS pair |
|---|---|
| Phase only | $Nb_p$ |
| Pair scale | $Nb_p+b_s$ |
| Polar per-element magnitude | $N(b_p+b_m)$ |
| Progressive residual | $Nb_p+b_s+Nb_\varepsilon$ |
| Cartesian VQ | $Nb_{vq}$ |
| Cartesian VQ + gain | $Nb_{vq}+b_s$ |

Codebooks, clipping ranges, and phase-grid offsets are estimated on calibration data and frozen.
Payload includes all transmitted indices; pre-shared codebooks are not counted per message.

## 4. Results

Failed control:

| Infinity-bit codec | Message identity | Max phase error | Threshold result |
|---|---:|---:|---|
| R0c polar | Exact | $4.77\times10^{-7}$ | Pass |
| Phase only | Exact | $4.86\times10^{-6}$ | Fail |
| Pair scale | Exact | $9.45\times10^{-6}$ | Fail |

Only 3/48,000 and 5/48,000 elements exceed $10^{-6}$. They have cancellation ratio below about
0.03. The message reconstruction is bit-exact; the discrepancy comes from float32 rounding amplified
by projecting an almost-zero resultant. The condition-aware quantity
$|\Delta\theta|\rho$ stays below $1.8\times10^{-7}$.

Selected provisional frontier points:

| Bits/pair | Codec | Paper-decentralized rate | Difference from R0c fp32 |
|---:|---|---:|---:|
| 68 | Pair scale, 2-bit phase + 8-bit scale | 19.4316 | $-1.2798\pm0.0755$ |
| 98 | Pair scale, 3-bit phase + 8-bit scale | 20.4550 | $-0.2563\pm0.0346$ |
| 188 | VQ + gain | 20.6260 | $-0.0853\pm0.0247$ |
| 968 | Pair scale, fp32 phase + 8-bit scale | 20.7461 | $+0.0348\pm0.0191$ |
| 1920 | R0c fp32 | 20.7113 | reference |

At 68 bits, pair scale and 2-D VQ+gain are tied ($-0.011\pm0.030$ relative to VQ). With phase fixed
at 2 bits, increasing per-element magnitude from 1 bit to fp32 saturates near 19.42; increasing phase
from 2 to 3 bits is worth about $+1.02\pm0.06$ bps/Hz. The 122–962 bit region is undersampled because
pair-scale phase precisions 4/5/6/8 bits were not run.

## 5. Interpretation

- Pair-level magnitude is a compact and interpretable choice, but it has not beaten a strong
  same-bit VQ baseline at the key 68-bit point.
- The nominal twofold compression is non-inferior only when phase remains fp32; deployable 2–3-bit
  phase incurs a measurable loss.
- The failed control is a threshold-design problem, not evidence of codec corruption, but the
  pre-declared stopping rule still makes the run provisional.

Before rerunning, replace the raw post-projection maximum-error control with one of:

1. exact message-level identity;
2. pre-normalization resultant relative error; or
3. a condition-aware bound such as $\max |\Delta\theta|\rho$.

## 6. Limitations

- Failed control and no independent identical-command replay.
- 400 samples / 50 clusters and one evaluation seed.
- Missing 4/5/6/8-bit pair-scale phase points.
- Reconstruction-aware rather than task-aware VQ.
- Fixed checkpoint, training seed, and topology.

## 7. Reproduction and artifacts

`artifacts/decentralized_ris/e07_message_codec/` contains `config.json`, `results.json`, paired NPZ,
frontier CSV/plots, logs, metadata, and the exact `command.sh`. The implementation is
`code/decentralized_ris/experiments/message_codec_sweep.py`; plotting uses
`code/plot_message_codec_frontier.py`.
