# E15 — Frozen-policy sum rate versus the per-AP transmit power budget

## 1. Status

- Question: does the frozen G2-150k advantage over frozen R0-500k in paper-decentralized sum rate persist across the common per-AP transmit power budget $P_{\max,l}=P_{\max}$ for all $l$, and how does each method's decentralization cost move along that axis?
- Status: **Completed frozen-policy power sweep**; all six declared controls pass and the declared persistence rule is met at every swept power.
- Training seed 0; evaluation seed 20261003; 1,600 samples in 200 batches of eight per power point, with both models and all seven powers scored on the same channel draws.
- Scope: no retraining. Both checkpoints were trained at 15 dBm, so every other point is a frozen policy deployed off its training budget.
- Pre-registration: `artifacts/decentralized_ris/e15_transmit_power_sweep/preregistration.json`.
- Method: [G2 and the R0 baseline](../decentralized_ris_methods.md); the frozen checkpoints are the E06 and E01 finalists also used by [E14](./e14_topology_stress.md).
- Primary artifacts: `artifacts/decentralized_ris/e15_transmit_power_sweep/`.

## 2. Conclusion

G2's paper-decentralized advantage over R0 holds at every swept power from 5 to 35 dBm, in continuous and in rounded 2-bit actuation, with every 95% batch-clustered interval excluding zero. The absolute advantage grows with the budget, from $+1.3510$ $[+1.2063,+1.4956]$ bps/Hz at 5 dBm to $+2.9267$ $[+2.5713,+3.2821]$ at 35 dBm, but as a proportion of R0's own rate it is flat: 10.10%, 10.01%, 9.96%, 9.98%, 10.05%, 10.11% and 10.14% along the same grid. Over a 30 dB span the effect is therefore better described as an approximately constant multiplicative gain than as a growing one.

The two architectures are not separated in centralized inference. The centralized contrast never exceeds $+0.2488$ bps/Hz and its interval covers zero at 5, 25, 30 and 35 dBm; at 35 dBm R0 is nominally ahead by $0.0229$. The entire measured difference is produced by decentralized inference, which the relative decentralization gap $(C-D)/C$ shows directly: it rises from $-0.07\%$ to $3.63\%$ for G2 and from $9.10\%$ to $12.62\%$ for R0. Raising the budget therefore makes decentralization more expensive for both frozen policies, and G2 pays between one third and one twentieth of R0's proportional cost across the grid.

Both curves flatten towards the interference-limited limit of their own frozen solution, 32.2190 bps/Hz for G2 and 29.2446 for R0 in paper-decentralized mode; at 35 dBm each has reached 98.7% of it. That saturation is a property of a fixed spatial solution rather than a property of the method at high power, because a frozen checkpoint cannot back its transmit power off to suppress interference.

The 15 dBm point is an independent replication of the established headline contrast on a fresh seed and a larger sample: $+2.1416$ $[+1.9241,+2.3591]$ here, against $+2.2053\pm0.2841$ in [E06](./e06_graph_energy_training.md) and $+2.2474$ $[+1.8240,+2.6708]$ on T0 in [E14](./e14_topology_stress.md).

## 3. Setup

The layout is the canonical T0 setting: AP ring radius 200 m, RIS ring radius 100 m, user disc radius 100 m, $L=5$ APs, $M=2$ antennas per AP, $R=4$ RISs, $N=30$ elements per RIS, $K=8$ users per AP, association threshold 0.1, and the same channel law as E01. Only the common per-AP budget $P_{\max}$ changes, over $\{5,10,15,20,25,30,35\}$ dBm, the grid E08 already used for the historical R0 sweep. The frozen checkpoints are G2-150k (`e06_graph_energy_training/g2_long_training/g2/iter150000`) and R0-500k (`e01_baseline_training/iter500000/.../run0`), both trained at 15 dBm with training seed 0.

Sweeping a frozen checkpoint over $P_{\max}$ is exact rather than approximate in these architectures. Each AP block is L2-normalized before the power scale is applied,

$$\mathbf W_l=\sqrt{P_{\max}\,\alpha_l}\;\frac{\tilde{\mathbf W}_l}{\lVert\tilde{\mathbf W}_l\rVert_F},$$

and neither the power-split head that produces $\alpha_l$ nor the RIS phase readout takes $P_{\max}$ as an input; the model inputs are normalized channel features. A checkpoint evaluated at $P$ therefore emits exactly $\sqrt{P/P_{\text{ref}}}\,\mathbf W_l(P_{\text{ref}})$ with an unchanged phase. Two consequences follow and both are used here. First, one forward pass per batch serves all seven power points, so the sweep costs one evaluation rather than seven, and a declared control rebuilds each model at 5 and 35 dBm on the first two batches to confirm that the scaled path reproduces a genuine rebuild. Second, because every AP scales by the same factor, signal and interference scale together and each user's SINR $P S_k/(P I_k+\sigma^2)$ is strictly increasing in $P$ with the limit $\log_2(1+S_k/I_k)$; that limit is reported as the interference-limited ceiling, computed as the $\sigma^2\to0$ evaluation of the same frozen solution and reduced over users per sample, then averaged over the 200 batch clusters.

The declared primary readout is the paired difference of paper-decentralized continuous sum rate, G2 minus R0, at each power, summarized over 200 batch-cluster means with a 95% interval; the secondary readout is the same contrast under rounded 2-bit RIS actuation, which is an evaluation-time rounding and not a separately trained arm. Centralized rate, the relative decentralization gap $(C-D)/C$ where $C$ and $D$ are the centralized and paper-decentralized rates of the same model on the same draws, and the interference ceiling are declared context. The declared persistence rule is that the interval excludes zero in both phase settings at every swept power. Six controls were declared before the locked run: unit modulus of the deployed phase, agreement between the cached rate evaluator and the deployed `simulator.loss` path, the scaling identity against a genuine rebuild, per-AP power feasibility $\max_l\lVert\mathbf W_l\rVert_F^2/P_{\max}\le1+10^{-5}$, monotonicity in power, and finiteness. The declared stop rule was to report a failed control rather than adjust a threshold and re-run. A 16-sample smoke run on seed 20261002 was used only to exercise the controls, and the locked evaluation then used the unseen seed 20261003.

## 4. Results

| $P_{\max}$ dBm | G2 centralized | G2 paper-dec. | G2 2-bit dec. | G2 rel. gap | R0 centralized | R0 paper-dec. | R0 2-bit dec. | R0 rel. gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 14.7318 | 14.7238 | 13.6876 | $-0.07\%$ | 14.7011 | 13.3728 | 12.4348 | $9.10\%$ |
| 10 | 19.4476 | 19.3288 | 18.0453 | $0.46\%$ | 19.3119 | 17.5701 | 16.4030 | $8.97\%$ |
| 15 | 23.9619 | 23.6513 | 22.0315 | $1.13\%$ | 23.7348 | 21.5097 | 20.0306 | $9.27\%$ |
| 20 | 27.7644 | 27.1897 | 25.0927 | $1.89\%$ | 27.5157 | 24.7214 | 22.8003 | $10.03\%$ |
| 25 | 30.5326 | 29.6650 | 27.0156 | $2.64\%$ | 30.3488 | 26.9555 | 24.5258 | $11.03\%$ |
| 30 | 32.2243 | 31.1042 | 27.9914 | $3.25\%$ | 32.1513 | 28.2491 | 25.3963 | $11.97\%$ |
| 35 | 33.0768 | 31.7927 | 28.4010 | $3.63\%$ | 33.0998 | 28.8660 | 25.7612 | $12.62\%$ |

*Table 1.* Mean sum rate in bps/Hz of the frozen G2-150k and R0-500k checkpoints, training seed 0, on evaluation seed 20261003 with 1,600 samples in 200 batch clusters of eight per power point. Both models and all seven powers use the same channel draws and association masks on the canonical T0 layout; only $P_{\max}$ changes, and neither model was retrained away from its 15 dBm training budget. "Paper-dec." is the shared-UE decentralized inference mode of E02, "2-bit dec." rounds the same decentralized phase to a 2-bit grid at evaluation time, and "rel. gap" is the per-batch relative decentralization gap $(C-D)/C$ of that model. Its 95% intervals are $[-0.41,0.27]$, $[0.10,0.82]$, $[0.74,1.52]$, $[1.44,2.33]$, $[2.14,3.13]$, $[2.70,3.79]$ and $[3.06,4.20]$ percent for G2 and $[8.27,9.92]$, $[8.19,9.74]$, $[8.51,10.03]$, $[9.26,10.79]$, $[10.24,11.83]$, $[11.16,12.79]$ and $[11.79,13.45]$ for R0. Full per-sample arrays are in `sweep.npz` and all cluster summaries in `sweep.json` under E15.

| $P_{\max}$ dBm | G2−R0 paper-dec. [95% CI] | G2−R0 2-bit dec. [95% CI] | G2−R0 centralized [95% CI] | G2 over R0, paper-dec. |
|---:|---:|---:|---:|---:|
| 5 | $+1.3510$ $[+1.2063,+1.4956]$ | $+1.2528$ $[+1.1181,+1.3875]$ | $+0.0307$ $[-0.0525,+0.1139]$ | $10.10\%$ |
| 10 | $+1.7587$ $[+1.5801,+1.9374]$ | $+1.6423$ $[+1.4779,+1.8067]$ | $+0.1357$ $[+0.0248,+0.2466]$ | $10.01\%$ |
| 15 | $+2.1416$ $[+1.9241,+2.3591]$ | $+2.0009$ $[+1.8036,+2.1982]$ | $+0.2271$ $[+0.0822,+0.3720]$ | $9.96\%$ |
| 20 | $+2.4683$ $[+2.2084,+2.7281]$ | $+2.2925$ $[+2.0607,+2.5242]$ | $+0.2488$ $[+0.0659,+0.4316]$ | $9.98\%$ |
| 25 | $+2.7095$ $[+2.4080,+3.0110]$ | $+2.4898$ $[+2.2270,+2.7526]$ | $+0.1838$ $[-0.0377,+0.4053]$ | $10.05\%$ |
| 30 | $+2.8551$ $[+2.5200,+3.1901]$ | $+2.5951$ $[+2.3104,+2.8798]$ | $+0.0730$ $[-0.1819,+0.3279]$ | $10.11\%$ |
| 35 | $+2.9267$ $[+2.5713,+3.2821]$ | $+2.6397$ $[+2.3432,+2.9362]$ | $-0.0229$ $[-0.3003,+0.2544]$ | $10.14\%$ |

*Table 2.* Paired G2-minus-R0 differences in bps/Hz from the same run as Table 1: 200 batch-cluster means per power, 95% batch-clustered intervals, same frozen checkpoints and same channel draws for both models. Clustered standard errors on the paper-decentralized contrast are 0.0734, 0.0906, 0.1103, 0.1318, 0.1529, 0.1699 and 0.1802 bps/Hz along the grid. The last column is the same paper-decentralized contrast expressed as a proportion of R0's own mean rate at that power, which is the comparison that is not carried by the moving operating level. The declared persistence rule requires the two decentralized intervals to exclude zero at every power, which they do.

The interference-limited ceilings of the frozen solutions are 32.2190 $[31.5897,32.8483]$ bps/Hz for G2 and 29.2446 $[28.6169,29.8723]$ for R0 in paper-decentralized mode, against 33.6363 and 33.7474 in centralized mode. At 5 dBm each model has reached 45.7% of its own decentralized ceiling and at 35 dBm 98.7%. As a smoke control at the reference budget, replacing the deployed phase with a random phase collapses the decentralized rate to 6.4023 bps/Hz for G2 and 6.2372 for R0.

All six declared controls pass: maximum unit-modulus error $1.79\times10^{-7}$, maximum deviation between the cached and deployed rate paths $3.81\times10^{-6}$ bps/Hz, maximum deviation of the scaling identity from a genuine rebuild at 5 and 35 dBm $3.81\times10^{-6}$ bps/Hz, maximum per-AP power ratio $1.0000004$, smallest per-sample increment across adjacent powers $+1.76\times10^{-3}$ bps/Hz, and no non-finite value.

## 5. Interpretation

- The ordering established at 15 dBm is not an artifact of that operating point. It survives a 30 dB sweep in both phase settings, and the proportional size of the advantage is stable near 10% while its absolute size grows with the operating level, so the absolute figure should not be quoted without its budget.
- The two architectures reach the same centralized rate to within measurement error. What G2 adds is retention under decentralized inference, and the relative decentralization gap separates the models by roughly an order of magnitude at low power and by a factor of about 3.5 at high power.
- Both frozen policies lose more to decentralization as the budget rises. That is consistent with the interference-limited regime being the harder coordination problem: when noise dominates, each AP's locally visible CSI is nearly sufficient, and when interference dominates, the decisions that matter depend on links an AP cannot see. G2's per-RIS context and energy-weighted consensus narrow that loss but do not remove it.
- Saturation at the upper end is mechanical, not a statement about the method's high-power capability. A frozen checkpoint reproduces one spatial solution at all budgets, so its sum rate must converge to that solution's interference ceiling. The ceilings themselves are informative, because the frozen G2 decentralized solution has a 2.97 bps/Hz higher ceiling than R0's while their centralized ceilings agree within 0.11.

## 6. Limitations

- Nothing is retrained. Away from 15 dBm this is a deployment-robustness curve for policies trained at one budget, not the achievable sum rate of a power-matched trained policy, which is what the reference paper's figure reports. A policy trained at 30 dBm could reduce or reallocate transmit power to control interference; the frozen power split cannot, so the high-power points are a lower bound of unknown tightness, and the saturation level in particular must not be read as a capability limit of either method.
- One training seed, one training topology, one evaluation seed, and one layout. The sweep inherits every limitation of the frozen checkpoints it evaluates.
- The noise level $\sigma^2=(2\times10^{-2})^2$ is implicit in the implementation rather than specified with the channel model, so the dBm axis is meaningful only relative to that constant; this is the same caveat E08 records for the historical sweep.
- These numbers are not comparable with E08's power sweep, which trained a short-budget R0 at each point. Only the power grid is shared.
- No external reference was run at other powers. The E12 fixed-budget centralized and distributed model-based arms exist only at 15 dBm, so the sweep carries no upper reference away from the reference budget.
- The 2-bit column is evaluation-time rounding of a continuously trained phase, not a discretely trained arm, and the E07 decision about whether 2-bit actuation should become a primary readout is still open.

## 7. Reproduction and artifacts

From `code/decentralized_ris/`, with the `decentralized-inference` environment active:

```bash
python -m experiments.power_sweep --samples 1600 --eval_seed 20261003 --device cuda:0 \
  --out ../../artifacts/decentralized_ris/e15_transmit_power_sweep/sweep.json
python -m experiments.summarize_power_sweep \
  --summary ../../artifacts/decentralized_ris/e15_transmit_power_sweep/sweep.json \
  --figure ../../artifacts/decentralized_ris/e15_transmit_power_sweep/power_sweep.svg
```

The run took about three minutes on a local RTX 5060 Laptop GPU and loads no other host. `artifacts/decentralized_ris/e15_transmit_power_sweep/` holds `preregistration.json` with the declared question, readouts, controls and stop rule; `sweep.json` with every cluster summary, contrast, control value and gate; `sweep.npz` with the per-sample rate arrays for both models, all seven powers, both phase settings, the ceilings and the random-phase control; `sweep.md` with the generated results table; and `power_sweep.svg` with the two-panel figure of sum rate against $P_{\max}$ and of the relative decentralization gap. The checkpoint paths recorded inside `sweep.json` are the E06 and E01 finalists and are unchanged by this experiment.
