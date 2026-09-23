# E13 — Fixed-G2-RIS active-beamforming controls

## 1. Status

- Question: With the proposed G2 RIS phase fixed, does its learned active beamformer outperform
  RIS-aware local MRT and RIS-aware local RZF?
- Status: **Completed component control**
- Updated: 2026-09-18
- Scope: frozen G2-150k checkpoint, 400 paired holdout samples per evaluation seed, training seed 0,
  canonical T0 topology; no training
- Method: benchmark item 7 of the [benchmark plan](../research_positioning.md#8-benchmark-plan)
- Primary artifacts: `artifacts/decentralized_ris/e13_fixed_ris_beamforming/`

## 2. Conclusion

The learned active beamformer is the source of the advantage, not the per-AP power scale. Under the
identical fused RIS phase, G2 exceeds the best conventional local precoder by
$2.8848\pm0.1685$ bps/Hz in paper-decentralized continuous phase and by $3.0220\pm0.1564$ at rounded
2 bits, winning all 50 batch clusters. Replacing G2's learned per-AP power fractions with full power
costs only $0.0737\pm0.0224$, so the gap is a beamforming-direction effect. Under the declared
$\lambda_l$, local RZF is *worse* than local MRT by $1.8674\pm0.3729$; this is a property of the
regularizer in an $|\mathcal K_l|>M$ regime, not evidence about G2. By the declared stop rule
the conventional controls do not advance to Stage B. This is a component control: it does not compare
G2 against a complete model-based joint design, so Stage A remains incomplete.

## 3. Setup

Every arm receives the same frozen inputs: the G2-150k checkpoint, its paper-decentralized phase
proposals, the parameter-free local-energy consensus, the association mask, the channel tensors, the
per-AP constraint $\lVert\mathbf W_l\rVert_F^2\le P_{\max}$, and the quantization rule. Only the
AP-side active design changes.

| Arm | Active design at AP $l$ | Per-AP power | Interface |
|---|---|---|---|
| `g2` | native G2 readout | learned $\alpha_l$ | one-shot |
| `g2_full` | native G2 direction | full $P_{\max}$ | one-shot |
| `mrt_full` | RIS-aware local MRT | full $P_{\max}$ | two-stage |
| `rzf_full` | RIS-aware local RZF | full $P_{\max}$ | two-stage |
| `mrt_matched` | RIS-aware local MRT | G2's $\alpha_l$ | two-stage |
| `rzf_matched` | RIS-aware local RZF | G2's $\alpha_l$ | two-stage |

After the CPU fuses and broadcasts the phase, AP $l$ forms its effective channels
$\mathbf H_l^{\rm eff}$ to its associated users and computes
$\mathbf W_l\propto\mathbf H_l^{\rm eff}$ for MRT and
$\mathbf W_l\propto(\mathbf H_l^{\rm eff}\mathbf H_l^{{\rm eff},H}+\lambda_l\mathbf I)^{-1}
\mathbf H_l^{\rm eff}$ for RZF, followed by the common per-AP power normalization. The regularizer
$\lambda_l=|\mathcal K_l|\sigma^2/P_l$ with $\sigma^2=(2\times10^{-2})^2$ and $P_l=P_{\max}$ was
declared before evaluation, not tuned on the holdout, and is identical for the full-power and $\alpha$-matched
arms so that the pair differs only in power scale. MRT and RZF are recomputed from the phase actually
applied, so the 2-bit arms use the quantized effective channel.

The rate model pairs the stored channel with the beamformer as $\mathbf h^{T}\mathbf w$, so
$\mathbf H_l^{\rm eff}$ above denotes the matrix whose columns are the conjugated channels
$\mathbf g_{l,k}=\overline{\mathbf h_{l,k}}$, which satisfy the usual $\mathbf g^{H}\mathbf w$
convention.

Setting: $L=5$ APs, $M=2$ antennas/AP, $R=4$ RISs, $N=30$ elements/RIS, $K=8$ users,
$P_{\max}=15$ dBm, association threshold 0.1. Primary evaluation seed 20260918 with 400 paired
samples in 50 clusters of 8; seed 20260915 repeats the same protocol as a cross-experiment
consistency check against the E06 milestone table. Decision rules, the regularizer, and eight
numerical gates were fixed before the run and are recorded in
`e13_fixed_ris_beamforming/pre_registration.md` and in the `DECISION` constant of
`experiments/fixed_ris_beamforming.py`. The current Git history does not contain an immutable pre-run
commit, so these files document the declared protocol rather than proving preregistration timing.

## 4. Results

Paper-decentralized sum rate, primary seed 20260918:

| Arm | Continuous | Clustered SE | 2-bit | Clustered SE |
|---|---:|---:|---:|---:|
| **`g2`** | **23.0485** | 0.6763 | **21.4519** | 0.6309 |
| `g2_full` | 22.9748 | 0.6666 | 21.3872 | 0.6236 |
| `mrt_full` | 20.1637 | 0.6346 | 18.4299 | 0.5765 |
| `mrt_matched` | 20.1462 | 0.6811 | 18.4303 | 0.6306 |
| `rzf_full` | 18.2963 | 0.4190 | 16.4787 | 0.3694 |
| `rzf_matched` | 18.0915 | 0.4357 | 16.2566 | 0.3902 |

Paired contrasts on identical channel samples:

| Contrast | Continuous | 95% CI | Cluster wins | 2-bit |
|---|---:|---|---:|---:|
| `g2` − `mrt_full` | $+2.8848\pm0.1685$ | $[+2.5546,+3.2151]$ | 50/50 | $+3.0220\pm0.1564$ |
| `g2` − `mrt_matched` | $+2.9023\pm0.1501$ | $[+2.6081,+3.1966]$ | 50/50 | $+3.0216\pm0.1348$ |
| `g2` − `rzf_full` | $+4.7522\pm0.4235$ | $[+3.9221,+5.5824]$ | 50/50 | $+4.9732\pm0.4554$ |
| `g2` − `g2_full` | $+0.0737\pm0.0224$ | $[+0.0297,+0.1177]$ | 35/50 | $+0.0647\pm0.0207$ |
| `rzf_full` − `mrt_full` | $-1.8674\pm0.3729$ | $[-2.5983,-1.1364]$ | 3/50 | $-1.9512\pm0.4050$ |

The replication seed 20260915 reproduces the same ordering and reproduces E06's milestone exactly:
`g2` continuous is 24.5358 against E06's reported 24.5358, and `g2` − `mrt_full` is
$+3.0269\pm0.1554$. Every arm is worse under 2-bit rounding by roughly 1.6–1.8 bps/Hz, and the
ordering is unchanged.

All eight declared gates pass on both seeds: per-AP power overshoot $2.4\times10^{-7}$,
RZF solve relative residual $4.9\times10^{-6}$, conventional-precoder locality deviation exactly 0,
fused-phase unit-modulus error $1.2\times10^{-7}$, 2-bit grid distance exactly 0, scalar-versus-
vectorized rate deviation $1.9\times10^{-6}$ across every arm, no beamformer energy on unserved
columns, and no non-finite values.

Counted online signaling per sample, and batch-1 timing on one RTX 5090 after warm-up:

| Interface | AP→CPU | CPU→AP | CPU→RIS | Rounds | Median | p95 |
|---|---:|---:|---:|---:|---:|---:|
| `g2`, `g2_full` (one-shot) | 620 values | 0 | 120 values | 1 | 4.554 ms | 5.099 ms |
| MRT controls (two-stage) | 620 values | 600 values | 120 values | 2 | 4.650 ms | — |
| RZF controls (two-stage) | 620 values | 600 values | 120 values | 2 | 4.878 ms | — |

The CPU→AP broadcast is 19,200 bits at fp32 or 1,200 bits at 2 bits per element; UE→AP CSI
acquisition is excluded and is identical across arms. The two-stage medians add the local precoding
stage (0.096 ms for MRT, 0.324 ms for RZF) to the G2 stage and do not model the broadcast latency
itself.

Diagnostic excluded from the decision: scaling the declared $\lambda_l$ by 0.1 gives 11.6353
and by 10 gives 19.5535 on the first 80 samples, against 18.2963 at $\lambda_l$.

## 5. Interpretation

- The active-beamforming claim in benchmark question 2 is supported on this topology and checkpoint:
  under an identical RIS action, the learned readout beats both conventional local precoders by a
  margin far larger than the cluster uncertainty, on every cluster and on 93–97% of individual
  samples.
- The advantage is a direction effect, not a power-allocation effect. G2's learned $\alpha_l$ is worth
  only $0.0737$ bps/Hz over full power, and MRT is indifferent to which power rule it uses. A reader
  cannot attribute the gap to G2 spending its budget more cleverly.
- RZF losing to MRT is explained by the operating point, not by G2. 95.8% of AP blocks serve more
  users than the $M=2$ antennas available, so the declared $\lambda_l$ leaves a nearly
  zero-forcing precoder that cannot null what it is asked to null. The sensitivity diagnostic points
  the same way: rate increases monotonically as $\lambda_l$ grows toward the MRT limit. The correct
  reading is that this declared RZF is a weak control here, so `mrt_full` rather than RZF is the
  conventional arm the claim must clear.
- The conventional controls are strictly two-stage. They need the fused phase back at the APs before
  a beamformer exists, which adds one online round and a 600-value CPU→AP broadcast that the one-shot
  G2 path does not pay. Their computational stage is cheap (0.1–0.3 ms), so the cost is in rounds and
  latency structure, not arithmetic.
- By the declared stop rule the best conventional arm trails by 2.88 bps/Hz, far above the
  0.5 bps/Hz threshold, so neither MRT nor RZF advances to Stage B.

## 6. Limitations

- One checkpoint, one training seed, one topology. This is a fixed-checkpoint component control and
  carries no claim about G2 across topologies, $N$, or AP/RIS counts.
- The comparison is not a complete-method comparison. Both conventional arms reuse the proposed GNN's
  passive action, so this experiment cannot say whether G2 beats a model-based joint design. Stage A
  remains incomplete until a model-based centralized/distributed pair clears its gates.
- The conventional controls see less information than G2 (their own effective channels only, but
  exact ones), while G2 uses the broader paper-decentralized CSI view. The arms therefore differ in
  information as well as in architecture, and the gap should not be read as an architecture-only
  effect.
- Only one $\lambda_l$ enters the primary table. The sensitivity diagnostic shows RZF is far from its
  best achievable setting in this regime, so "RZF is weak here" is a statement about this
  declared rule, not about regularized precoding in general. Retuning $\lambda_l$ on this
  holdout is not permitted and was not done.
- Timing is secondary. It compares one implementation on one device, and the two-stage end-to-end
  figures omit the broadcast propagation delay, which is the part most likely to dominate a real
  deployment.

## 7. Reproduction and artifacts

| Evidence | Location |
|---|---|
| Declared protocol, thresholds, and gates | `artifacts/decentralized_ris/e13_fixed_ris_beamforming/pre_registration.md` |
| Machine-readable results, signaling, timing | `e13_fixed_ris_beamforming/summary.json` |
| Per-sample paired rates, both seeds | `e13_fixed_ris_beamforming/paired_rates.npz` |
| Generated audit tables | `e13_fixed_ris_beamforming/report.md` |
| Run log | `e13_fixed_ris_beamforming/e13_fixed_ris_beamforming.log` |
| Comparison plot | `e13_fixed_ris_beamforming/plots/fixed_ris_beamforming.{png,pdf}` |

The program is `experiments/fixed_ris_beamforming.py`, the maintained entry point is
`scripts/e13_fixed_ris_beamforming.sh`, the regression checks are
`tests/test_fixed_ris_beamforming.py`, and the figure is produced by
`code/plot_fixed_ris_beamforming.py`.

```bash
cd code/decentralized_ris
RIS_PYTHON=python DEVICE=cuda:0 bash scripts/e13_fixed_ris_beamforming.sh
```

The recorded run executed on `lab301-5090-tailscale` (RTX 5090) under
`~/ThomasLai/code/decentralized_ris` with `RIS_PYTHON=$HOME/rsma_env/bin/python`, reading the frozen
checkpoint from its original launch path
`~/ThomasLai/artifacts/decentralized_ris/graph_energy_150k/graph-energy-g2-total150k-resume100k_iter50000_seed0/checkpoints/iter150000.pt`.
That file is byte-identical to the maintained local copy under
`e06_graph_energy_training/g2_long_training/g2/iter150000/` (SHA-256 `f568d5b3…ed240a8c`), so the
remote path is provenance only. Results were copied back into the E13 artifact folder; nothing from
this experiment exists only on the remote host.

Command: `python -m experiments.fixed_ris_beamforming --run <G2-150k run dir> --checkpoint
iter150000.pt --samples 400 --eval_seed 20260918 --replication_seed 20260915 --device cuda:0`.
Training seed 0, evaluation seeds 20260918 and 20260915, 400 samples each, key metric
paper-decentralized continuous sum rate.
