# E14 — Frozen-policy topology stress and shared-UE CSI delivery

## 1. Status

- Question: do frozen G2 and R0 retain their ordering on two fixed layouts, how do the E12
  fixed-budget arms compare there, and how much CSI must reach G2's AP copies?
- Status: **Completed fixed-budget topology screen**; all declared eligibility controls pass.
- Training seed 0; evaluation seed 20260920; 400 samples in 50 batches of eight per topology.
- Pre-registration: `artifacts/decentralized_ris/e14_topology_stress/preregistration.json`.
- Continuation question: when topology alone changes how much shared-UE CSI reaches each AP, does
  frozen G2 keep a larger decentralized-to-centralized retention ratio than frozen R0?
- Continuation status: **Completed visibility-pair continuation**; layouts were fixed from association masks
  only, and every rate was measured on a holdout seed that took no part in that selection.
- Mask-only calibration seed 20260921 with 80 samples; holdout evaluation seed 20260924 with 400
  samples in 50 batches of eight per layout; same frozen G2-150k and R0-500k checkpoints.
- Continuation pre-registration:
  `artifacts/decentralized_ris/e14_topology_stress/visibility_topology/preregistration.json`.
- Second continuation question: with centralized rate held approximately fixed, does lower
  shared-UE visibility still reduce each frozen model's retention, and is G2's reduction smaller?
- Second continuation status: **Completed rate-matched visibility pair and its fresh-seed
  confirmation**; the declared rate-match control passes on both seeds, both per-model intervals
  exclude zero, and the comparative difference-of-differences, which was marginal on the discovery
  seed, is confirmed on the confirmation seed.
- Rate-matched calibration seed 20260925 with 80 samples over a 44-point grid; holdout evaluation
  seed 20260927 with 400 samples per layout; same frozen checkpoints.
- Second continuation pre-registration:
  `artifacts/decentralized_ris/e14_topology_stress/rate_matched_visibility/preregistration.json`;
  confirmation seed 20260929 with 3,200 samples per layout, pre-registered separately in
  `rate_matched_visibility/confirmation/preregistration.json`.

## 2. Conclusion

G2's within-family paper-decentralized gain over R0 persists on both declared shifts, in continuous
and rounded 2-bit actuation. On T1/T2, the feasible fixed-budget E12 centralized and distributed
arms still have higher rate than G2; on T2, the continuous G2 deficits widen to 10.4922 and 8.6889
bps/Hz. The shared-UE CSI needed before proposal generation is substantial: under direct, lossless
fp32 AP→AP unicast, G2's T0 input delivery plus proposal payload averages 973,514 bits per sample,
versus the 19,840 proposal bits alone. This modeled total exceeds E12's T0 centralized coordination
count of 624,640 bits. It remains about 48–50 times below distributed ADMM's counted payload.
These statements are conditional on the named delivery route and the solvers' finite online budget;
they are not end-to-end network measurements or convergence claims.

The visibility pair asks how that ordering depends on visibility, using geometry instead of an
information-mode intervention. Three ring layouts whose AP radius sets shared-UE visibility (140, 200 and 350 m,
holdout visible fractions 0.2860, 0.3927 and 0.6170) give G2 retention $D/C$ of 0.9575, 0.9776 and
0.9790 against R0's 0.8574, 0.8897 and 0.9323, so the relative gap $(C-D)/C$ that decentralization
costs falls from 14.26% to 6.77% for R0 while G2 stays between 4.25% and 2.10%. G2's retention
advantage remains positive on every layout, but it narrows monotonically from $+0.1001$ to
$+0.0467$, and its absolute paper-decentralized advantage narrows from $+3.6939$ to $+0.4846$
bps/Hz. The high-visibility layout therefore keeps the established ordering with the weakest
margin. Moving the AP ring outwards also lowers received power, so centralized rate falls from
33.2281 to 10.0152 bps/Hz along the same axis; visibility and operating SNR are confounded here and
the association is descriptive. Modeled CSI delivery moves against the retention benefit: G2 needs
0.620, 0.888 and 1.466 million bits per sample, or 0.99×, 1.42× and 2.35× the centralized arm's
624,640 bits, so the layout where G2's decentralized advantage is smallest is also the one where
its input delivery is most expensive.

Because AP ring radius moves visibility and operating rate together, a second pre-registered pair
holds the centralized rate fixed instead. Searching a 44-point grid over AP ring radius and user
spread for layouts whose centralized rate matches within 1 bps/Hz for both models selected MLOW
(AP ring 280 m, user radius 140 m) against MHIGH (AP ring 170 m, user radius 40 m). On the fresh
holdout their centralized rates agree within 0.8274 bps/Hz, inside the declared 1.5 control, while
the visible fraction differs by 0.289. At that matched operating point lower visibility still
costs both models rate: G2's relative gap rises from 0.29% to 1.73% ($+1.44$ pp, 95% interval
$[+0.26,+2.61]$) and R0's from 6.85% to 11.19% ($+4.34$ pp, $[+1.71,+6.96]$). So visibility is not
merely a proxy for operating level, and the honest statement about G2 is that it degrades too,
about three times less than R0 in absolute percentage points. The declared secondary test of that
comparison, R0's increase minus G2's, was $+2.90$ pp with interval $[+0.01,+5.78]$ on the discovery
seed: it met the pre-registered rule at the boundary only. A single pre-registered confirmation on
seed 20260929 with 3,200 samples per layout settles it at $+1.53$ pp, interval $[+0.60,+2.45]$,
$z=3.23$, with the rate-match control again passing at 0.9185 bps/Hz. R0 therefore does lose more
than G2 when visibility falls at a matched operating point, by about 1.5 percentage points of
relative gap rather than the 2.9 the discovery seed suggested.

## 3. Setup

T0 is the canonical AP radius-200/RIS radius-100 layout. T1 fixes perturbed radii and angles; T2
fixes an asymmetric placement with two denser AP/RIS groups. Coordinates are stored in
`code/decentralized_ris/experiments/topology_t1.json` and `topology_t2.json`. User radius 100,
channel law, association threshold 0.1, $L=5$, $M=2$, $R=4$, $N=30$, $K=8$, and 15 dBm per AP are
unchanged. Each method uses the same layout, evaluation seed, and channel draw order. G2-150k and
R0-500k checkpoints are frozen. The E12 centralized/distributed arms use the already declared
matched initialization, 2,000 iteration/200 sweep caps, 200 MM iterations, and $\rho$ scale 0.8.
No training or holdout-based tuning occurs.
The E10 DNN and E13 fixed-G2-RIS beamforming controls did not advance from Stage A under the
documented shortlist rule, so they are not new topology competitors here.

The primary contrast is G2−R0 paper-decentralized rate, with a 95% interval based on 50 paired batch
means. Rounded 2-bit RIS actuation is evaluation only. A gain is retained on a shifted layout only
when the interval lower bound exceeds zero in both phase settings. The E12 arms may support only a
fixed-budget feasible-action contrast, contingent on their eligibility controls on that layout.

For each sample and user $k$, let $s_k$ be the number of serving APs. The number of ordered
source→target AP unicast deliveries is $N_{\text{copy}}=\sum_k s_k(s_k-1)$. One raw CSI delivery contains the $R M N$
complex cascaded coefficients and $M$ complex direct coefficients for a source AP and shared UE,
or $2M(RN+1)=484$ fp32 real values. The direct AP→AP case counts $32\cdot484 N_{\text{copy}}$ bits, followed by
G2's $RL(N+1)=620$ fp32 proposal values. A non-deduplicated CPU relay uses two hops per copy.
UE→AP acquisition, association/routing metadata, and common CPU→RIS actuation are excluded.
The same raw CSI delivery applies to R0; its AP→CPU proposal uses $RL(4N)=2,400$ fp32 values.
Counts are modeled payloads and sequential exchange rounds, not measured transport traffic or
latency. The exact normalized model-input tensor needs 501 real values per delivered AP–UE link
(features, RIS edges, direct edge); that alternative is reported separately as a descriptive
implementation-level sensitivity. The paper's shared-UE CSI may alternatively be fed back directly
by UEs; that would move this extra cost to the UE→AP link. The AP→AP unicast route is an explicit
accounting scenario, not a claim that the architecture mandates this transport.

The visibility pair uses the same channel law, user radius 100 m, RIS ring radius 100 m,
association threshold 0.1, frozen checkpoints, and per-AP power as T0, and changes only the AP ring
radius. For each sample, $s_k$ is again the number of APs serving user $k$; the visible fraction is
$V=\mathbb{E}\left[\sum_k s_k^2\right]/(A^2K)$, the share of all $A K$ AP–UE links that reach an AP
copy under the paper's shared-UE rule. Candidate radii 140, 170, 200, 280 and 350 m were scored from
association masks only on 80 calibration samples at seed 20260921, with no model, rate, or channel
realization used, giving $V=0.3183$, 0.3803, 0.4341 (the canonical T0 ring), 0.5632 and 0.6624. The
pre-registered rule selects the minimum and maximum non-T0 candidate, so 140 m became VLOW and
350 m VHIGH; the span 0.3441 clears the declared 0.10 minimum, so the evaluation proceeded. Their
coordinates were frozen into `experiments/topology_vlow.json` and `experiments/topology_vhigh.json`
before any rate was computed.

All three layouts were then evaluated on holdout seed 20260924, which took no part in the
selection, including a fresh T0 run so the three share evaluation conditions. The declared primary
readout is per-batch retention $D/C$ of paper-decentralized to centralized rate for each model and
layout, where $C$ and $D$ are the centralized and paper-decentralized rates as in E02, together
with the relative gap $(C-D)/C=1-D/C$ and 95% batch-clustered intervals; absolute rates,
$V$, and the per-AP visible-link counts are context. There is no retraining, no association
threshold change, and no own-only intervention: unlike E02's own-only endpoint, which removes
shared-UE CSI from the input while holding geometry fixed, here the information mode is unchanged
and geometry decides how much CSI exists to share. The E12 model-based arms were not re-run on
these layouts, so the continuation carries no external fixed-budget contrast.

The rate-matched pair keeps every fixed quantity of the first pair except the user disc radius,
which becomes a second layout degree of freedom so that visibility and centralized rate can be
moved independently. Along AP ring radius alone the two always fall together; adding user spread
separates them, because clustering users near the centre equalizes their distances to all APs,
which raises $V$, while spreading them out gives each user a dominant nearby AP, which lowers $V$
but raises rate. The declared grid is AP ring radius in {120, 140, 170, 200, 240, 280, 350} m
crossed with user radius in {20, 40, 60, 80, 100, 140, 180} m, keeping user radius at most 0.9 of
the AP ring radius, which leaves 44 points. Each point was scored on 80 calibration samples at
seed 20260925 for mask-only $V$ and for centralized-only rate of both frozen checkpoints; no
decentralized rate was computed at selection time. A pair is eligible when its centralized rates
match within 1.0 bps/Hz for **both** models and its visible fractions differ by at least 0.20, and
the rule takes the largest eligible span. Four pairs qualified; the selected one is MLOW at AP ring
280 m with user radius 140 m against MHIGH at AP ring 170 m with user radius 40 m, calibration span
0.2572 and worst calibration rate gap 0.3606 bps/Hz. The declared stop rule was to report failure
rather than widen the grid or loosen either threshold.

Both layouts were then evaluated on holdout seed 20260927, which took no part in the selection. The
pre-registered readouts are, per model, the increase in relative gap from MHIGH to MLOW as an
unpaired two-sample difference of 50 batch means, and the difference of those increases between R0
and G2, using the within-layout paired gap difference so the models share channels. A declared
control requires the two layouts' holdout centralized rates to stay within 1.5 bps/Hz for both
models; failing it would have invalidated the matched-rate claim with no reselection allowed.

Because the comparative contrast cleared its threshold only at the boundary, seed 20260927 is
treated as burned for that one statement and a separate pre-registration fixes a single
confirmation. It reuses the stored layouts and checkpoints with no reselection, changes only the
evaluation seed to 20260929, and raises the sample count to 3,200 per layout. The sample size was
set from power rather than convenience: the discovery standard error was 1.4715 pp at 400 samples,
and because a barely significant discovery estimate is likely inflated, 3,200 samples were chosen
to give a standard error near 0.52 pp, about 82% power for a true difference of 1.5 pp and 97% for
2.0 pp. The confirmation runs one test with the same estimator, adds unit-modulus and random-phase
controls, and declares in advance that a failure would be reported as unconfirmed with no third
seed and no change of estimator.

## 4. Results

| Layout | G2 centralized | G2 paper-dec. | G2 2-bit dec. | R0 centralized | R0 paper-dec. | R0 2-bit dec. |
|---|---:|---:|---:|---:|---:|---:|
| T0 | 23.3572 | 22.9587 | 21.3847 | 23.0898 | 20.7113 | 19.2009 |
| T1 | 24.5555 | 23.8777 | 22.1883 | 23.3402 | 20.6941 | 19.1653 |
| T2 | 18.4864 | 18.1892 | 16.9266 | 15.9290 | 14.1520 | 13.1049 |

| Layout | G2−R0 continuous, paired mean ± cluster SE [95% CI] | G2−R0 2-bit, same |
|---|---:|---:|
| T0 | $+2.2474\pm0.2160$ $[1.8240,2.6708]$ | $+2.1838\pm0.1982$ $[1.7953,2.5722]$ |
| T1 | $+3.1836\pm0.4188$ $[2.3626,4.0045]$ | $+3.0231\pm0.3907$ $[2.2572,3.7889]$ |
| T2 | $+4.0371\pm0.2997$ $[3.4497,4.6246]$ | $+3.8217\pm0.2767$ $[3.2792,4.3641]$ |

| Layout | Centralized solver continuous / 2-bit | Distributed solver continuous / 2-bit | G2−centralized continuous [95% CI] | G2−distributed continuous [95% CI] |
|---|---:|---:|---:|---:|
| T0 | 28.9463 / 24.5714 | 28.0170 / 24.0111 | $-5.9876$ $[-6.5454,-5.4299]$ | $-5.0583$ $[-5.5023,-4.6144]$ |
| T1 | 30.5706 / 25.1202 | 29.0571 / 24.3901 | $-6.6929$ $[-7.5849,-5.8010]$ | $-5.1794$ $[-5.7104,-4.6484]$ |
| T2 | 28.6813 / 23.0292 | 26.8781 / 22.0561 | $-10.4922$ $[-11.3891,-9.5952]$ | $-8.6889$ $[-9.4338,-7.9440]$ |

The rounded 2-bit G2 deficits versus the centralized/distributed solver are $-3.1867/-2.6264$
on T0, $-2.9318/-2.2018$ on T1, and $-6.1026/-5.1295$ on T2. All paired 95% intervals exclude
zero; full batch-clustered SEs and intervals are in `comparison.json`. Every topology passes the
eight E12 feasible-action gates: power, unit modulus, 2-bit grid, maintained-rate equivalence,
AP locality, association mask, finite values, and superiority to its random-phase smoke control.
The stronger original E12 convergence-level gate still fails. On T1 the float32 re-solve deviation
is 2.4401 bps/Hz and maximum copy-consensus residual 1.3624; on T2 they are 0.0603 and 0.7948.
T1 also fails centralized monotonicity and distributed primal-divergence checks. The distributed
sweep cap is reached by 99.75% of T1 and 100% of T2 samples.
As a hardware check, the T0 solver was re-run on the RTX 5090 with the same 400 samples and
online budgets, while shortening only post-hoc initialization/timing diagnostics. Its primary
per-sample rates differ from the original RTX 5060 E12 arrays by at most $6.0\times10^{-7}$
bps/Hz across the four continuous/2-bit solver outputs; the T0 means are unchanged at four decimals.

| Layout | Ordered CSI copies/sample | Raw CSI AP→AP bits | G2 raw CSI + proposal bits | R0 raw CSI + proposal bits | G2 exact-input sensitivity bits |
|---|---:|---:|---:|---:|---:|
| T0 | 61.575 | 953,674 | 973,514 | 1,030,474 | 1,007,010 |
| T1 | 58.555 | 906,900 | 926,740 | 983,700 | 958,594 |
| T2 | 62.005 | 960,333 | 980,173 | 1,037,133 | 1,013,904 |

All 400 samples on every layout require at least one cross-AP CSI copy. Direct delivery adds one
parallel AP→AP round before the AP→CPU proposal round; the non-deduplicated CPU relay adds two.
The maximum copy count is 126 on each layout. The shared-UE count varies across sampled UE
locations, so the bits shown are sample means; the artifact JSON has batch-clustered uncertainty.
For the direct case, G2's mean modeled bits are 1.56×, 1.48×, and 1.57× the centralized arm's
624,640 bits on T0/T1/T2. Distributed ADMM's 46,595,840 bits are 47.9×, 50.3×, and 47.5×
G2's direct-delivery totals. A non-deduplicated CPU relay would raise G2 to 1,927,187,
1,833,640, and 1,940,507 bits and three sequential rounds. Common CPU→RIS actuation is 3,840
fp32 bits or 240 packed 2-bit actuator bits per sample and is excluded from this table.

The visibility pair below is a separate holdout (seed 20260924) from the T0/T1/T2 screen above
(seed 20260920), so its T0 row is a fresh draw on the same layout, not a repetition of the table
above. Measured visibility on the holdout is lower than the 80-sample calibration values but keeps
the same ordering and a 0.331 span.

| Layout | AP ring | Holdout $V$ | Own-served links/AP | Shared-visible links/AP | Ordered CSI copies/sample | Max copies |
|---|---:|---:|---:|---:|---:|---:|
| VLOW | 140 m | 0.2860 | 3.6920 | 11.441 | 38.745 | 88 |
| T0 | 200 m | 0.3927 | 4.4995 | 15.707 | 56.040 | 120 |
| VHIGH | 350 m | 0.6170 | 5.9995 | 24.679 | 93.395 | 144 |

| Layout | Model | Centralized $C$ | Paper-dec. $D$ | $D/C$ ± cluster SE [95% CI] | Gap $(C-D)/C$ |
|---|---|---:|---:|---:|---:|
| VLOW | G2 | 33.2281 | 31.7566 | $0.9575\pm0.0076$ $[0.9425,0.9725]$ | 4.25% |
| VLOW | R0 | 32.7472 | 28.0627 | $0.8574\pm0.0090$ $[0.8398,0.8750]$ | 14.26% |
| T0 | G2 | 24.2173 | 23.6263 | $0.9776\pm0.0052$ $[0.9674,0.9879]$ | 2.24% |
| T0 | R0 | 24.2737 | 21.5721 | $0.8897\pm0.0097$ $[0.8707,0.9087]$ | 11.03% |
| VHIGH | G2 | 10.0152 | 9.7638 | $0.9790\pm0.0042$ $[0.9707,0.9873]$ | 2.10% |
| VHIGH | R0 | 9.9522 | 9.2792 | $0.9323\pm0.0069$ $[0.9188,0.9457]$ | 6.77% |

| Layout | $\Delta(D/C)$, G2−R0 [95% CI] | $\Delta D$ continuous [95% CI] | $\Delta D$ 2-bit [95% CI] |
|---|---:|---:|---:|
| VLOW | $+0.1001\pm0.0088$ $[0.0828,0.1173]$ | $+3.6939\pm0.3541$ $[3.0000,4.3879]$ | $+3.3001\pm0.3128$ $[2.6870,3.9131]$ |
| T0 | $+0.0879\pm0.0088$ $[0.0707,0.1051]$ | $+2.0543\pm0.2558$ $[1.5529,2.5556]$ | $+2.0522\pm0.2318$ $[1.5978,2.5066]$ |
| VHIGH | $+0.0467\pm0.0080$ $[0.0311,0.0623]$ | $+0.4846\pm0.1671$ $[0.1572,0.8121]$ | $+0.5562\pm0.1541$ $[0.2541,0.8583]$ |

R0's gap shrinks monotonically as visibility rises, and its VLOW and VHIGH intervals are disjoint.
G2's gap is small on every layout and its VLOW value is the largest; the three G2 intervals overlap,
but interval overlap is a conservative criterion. An unpaired two-sample comparison of the batch-mean
retentions, which the pre-registration did not declare and which is therefore exploratory, puts G2's
VLOW retention 0.0202 below T0 ($z=-2.18$) and 0.0215 below VHIGH ($z=-2.46$), with T0 and VHIGH
indistinguishable ($z=-0.19$); after a Bonferroni correction for the three pairs only VLOW–VHIGH
stays below 0.05. The same comparisons for R0 give $z=-2.45$, $-6.63$ and $-3.58$. The supported
reading is therefore that G2 degrades at low visibility too, roughly doubling its relative gap from
2.10% to 4.25%, and that this degradation is far smaller than R0's 6.77% to 14.26%; the data do not
support calling G2 insensitive to visibility. The paired G2−R0
retention advantage is positive and interval-separated from zero on all three layouts, and VHIGH's
interval is disjoint from both VLOW's and T0's. Under rounded 2-bit actuation the same ordering
holds: G2 retains 0.9508, 0.9764 and 0.9795 against R0's 0.8587, 0.8866 and 0.9304. The
cross-layout comparisons are unpaired, because each layout draws its own channels; only the
G2−R0 contrasts within a layout are paired. As a seed check on the fixed T0 layout, the
screen seed 20260920 gives per-batch $D/C$ of $0.9853\pm0.0040$ for G2 and $0.8965\pm0.0084$ for
R0, against $0.9776\pm0.0052$ and $0.8897\pm0.0097$ on this holdout, so the retention levels are
stable across the two draws.

| Layout | G2 CSI + proposal bits | R0 CSI + proposal bits | CPU-relay G2 bits | G2 ÷ centralized 624,640 |
|---|---:|---:|---:|---:|
| VLOW | 619,923 | 676,883 | 1,220,005 | 0.99× |
| T0 | 887,788 | 944,748 | 1,755,735 | 1.42× |
| VHIGH | 1,466,342 | 1,523,302 | 2,912,844 | 2.35× |

Every sample on every layout needs at least one cross-AP copy, so no layout here removes the
delivery requirement. Unit-modulus error stays at $1.19\times10^{-7}$ for both models on all three
layouts, and each model's random-phase control stays far below its scored rates (for example
1.6280 against 9.7638 bps/Hz for G2 on VHIGH).

The rate-matched pair below is a third holdout (seed 20260927) and shares no channel draw with
either earlier table. Its declared control passes: the two layouts' holdout centralized rates
differ by at most 0.8274 bps/Hz, against the 1.5 tolerance.

| Layout | AP ring | User radius | Holdout $V$ | Own-served links/AP | Shared-visible links/AP | Ordered CSI copies/sample |
|---|---:|---:|---:|---:|---:|---:|
| MLOW | 280 m | 140 m | 0.4211 | 4.6795 | 16.844 | 60.825 |
| MHIGH | 170 m | 40 m | 0.7105 | 6.5435 | 28.418 | 109.375 |

| Layout | Model | Centralized $C$ | Paper-dec. $D$ | $D/C$ ± cluster SE [95% CI] | Gap $(C-D)/C$ | 2-bit gap |
|---|---|---:|---:|---:|---:|---:|
| MLOW | G2 | 16.6859 | 16.3139 | $0.9827\pm0.0049$ $[0.9732,0.9923]$ | 1.73% | 2.02% |
| MLOW | R0 | 16.7030 | 14.7300 | $0.8881\pm0.0102$ $[0.8680,0.9082]$ | 11.19% | 11.64% |
| MHIGH | G2 | 17.5133 | 17.4599 | $0.9971\pm0.0035$ $[0.9902,1.0040]$ | 0.29% | 0.15% |
| MHIGH | R0 | 16.2573 | 15.1004 | $0.9315\pm0.0087$ $[0.9145,0.9484]$ | 6.85% | 6.44% |

| Pre-declared contrast | Estimate | 95% CI | $z$ | Declared rule |
|---|---:|---:|---:|---|
| G2 gap increase at low visibility | $+1.44$ pp | $[+0.26,+2.61]$ | $+2.40$ | met |
| R0 gap increase at low visibility | $+4.34$ pp | $[+1.71,+6.96]$ | $+3.23$ | met |
| R0 increase minus G2 increase | $+2.90$ pp | $[+0.01,+5.78]$ | $+1.97$ | met, marginal |

Within each layout the paired G2−R0 retention difference is $+0.0946\pm0.0109$
$[0.0733,0.1159]$ on MLOW and $+0.0657\pm0.0099$ $[0.0463,0.0851]$ on MHIGH, so the retention
ordering matches the first visibility pair. The paired rate difference moves the other way,
$+1.5839$ $[1.1442,2.0236]$ on MLOW against $+2.3595$ $[1.8994,2.8196]$ on MHIGH, because R0's own
centralized rate falls on MHIGH (16.2573 against G2's 17.5133) while G2's does not: the absolute
difference mixes a centralized-quality difference into the decentralization loss, which the
retention ratio separates. Unit-modulus error stays at $1.19\times10^{-7}$ and the random-phase
controls stay near 3.1–3.6 bps/Hz against scored rates above 14.7. Modeled CSI delivery again
tracks visibility rather than rate: 961,898 bits per sample on MLOW against 1,713,840 on MHIGH for
G2, with 60.825 and 109.375 ordered copies.

The confirmation on seed 20260929 uses 3,200 samples per layout, 400 batches, and repeats the
whole readout. Its rate-match control passes at 0.9185 bps/Hz, unit-modulus error stays at
$1.19\times10^{-7}$, and every scored rate exceeds its random-phase control.

| Layout | Holdout $V$ | Model | Centralized $C$ | Paper-dec. $D$ | $D/C$ ± cluster SE | Gap $(C-D)/C$ |
|---|---:|---|---:|---:|---:|---:|
| MLOW | 0.4323 | G2 | 16.5388 | 16.0727 | $0.9761\pm0.0017$ | 2.39% |
| MLOW | 0.4323 | R0 | 16.5302 | 14.7227 | $0.8919\pm0.0035$ | 10.81% |
| MHIGH | 0.7166 | G2 | 17.4573 | 17.4034 | $0.9967\pm0.0011$ | 0.33% |
| MHIGH | 0.7166 | R0 | 16.4693 | 15.2565 | $0.9278\pm0.0029$ | 7.22% |

| Contrast | Discovery seed 20260927 | Confirmation seed 20260929 | Declared rule |
|---|---:|---:|---|
| G2 gap increase at low visibility | $+1.44$ pp $[+0.26,+2.61]$ | $+2.06$ pp $[+1.66,+2.46]$, $z=10.02$ | met on both |
| R0 gap increase at low visibility | $+4.34$ pp $[+1.71,+6.96]$ | $+3.59$ pp $[+2.70,+4.48]$, $z=7.90$ | met on both |
| R0 increase minus G2 increase | $+2.90$ pp $[+0.01,+5.78]$ | $+1.53$ pp $[+0.60,+2.45]$, $z=3.23$ | marginal, then confirmed |

The confirmation's paired within-layout G2−R0 retention difference is $+0.0842$ $[0.0771,0.0913]$
on MLOW and $+0.0689$ $[0.0629,0.0749]$ on MHIGH, reproducing the discovery seed's ordering at
tighter precision. The comparative effect is about half the discovery estimate, which is the
expected direction for a contrast selected because it just cleared its threshold; the confirmation
was powered for that shrinkage in advance and would have failed at the original 400 samples.

## 5. Interpretation

The within-family G2 advantage survives both declared geometry shifts, with larger paired gaps on
T1 and T2 than T0. Because T1's absolute G2 rate rises and T2's falls, the shifts test different
operating conditions rather than a uniform degradation. The external fixed-budget comparison is
less favorable to G2 on T2: its paired deficit to both model-based arms increases, including with
rounded 2-bit actuation. CSI delivery dominates G2's proposal payload at all three layouts and
also applies to R0. The 56,960-bit proposal saving of G2 over R0 is only about 5–6% of their
modeled full direct-delivery traffic here. The original E12 statement that G2 uses 31.5× fewer
bits than centralized is valid only after both methods have their required inputs. Under this
explicit unicast transport model, centralized uses fewer modeled bits and the same two sequential
rounds; distributed uses far more bits and 1,001 rounds.

The visibility pair reaches E02's qualitative conclusion without E02's intervention: G2 loses
little by acting on per-AP views while R0 loses much, and the difference is largest where each AP
sees the smallest share of the shared-UE links. Read as a mechanism, R0's decentralized penalty
behaves like a copy-inconsistency cost that geometry can relieve — at $V=0.6170$ its APs hold
almost two thirds of all AP–UE links and its gap falls to 6.77% — while G2's local energy consensus
has already removed most of that cost at $V=0.2860$, leaving it a 2.10% gap and little room to
improve. That mechanism is a reading of the rate trend, not something this continuation measures;
no consensus residual or copy-disagreement diagnostic was run on these layouts. The practical consequence is a bounded claim: G2's decentralized advantage is a
sparse-visibility advantage. It is worth $+3.6939$ bps/Hz where each AP sees 29% of the links and
$+0.4846$ bps/Hz, about 5% of the achieved rate, where it sees 62%.

Two readings of the same trend cannot be separated here. AP ring radius sets both $V$ and the
AP–UE distance, so VHIGH is simultaneously the high-visibility and the low-rate layout, and the
shrinking gap may reflect reduced sensitivity to coordination at a weaker operating point rather
than better local information. Distinguishing them needs a layout family that moves $V$ while
holding the centralized rate roughly fixed, for example by compensating per-AP power or the user
radius, which is outside this pre-registration. The continuation does not weaken the T1/T2 result;
it bounds where the G2−R0 margin is large, and it pairs that bound with the opposite cost trend,
since the sparse-visibility layouts that favour G2 are also the cheap ones to supply with CSI.

Holding the centralized rate fixed changes what the visibility trend can be called. In the first
pair, the shrinking gap could be read as an operating-level effect; at a matched operating point
both models still lose more rate when each AP sees fewer of the shared-UE links, so visibility
itself carries part of the effect. The size is modest for G2 — 1.73% against 0.29% of its own
centralized rate — and roughly three times larger for R0, 11.19% against 6.85%. The claim this
supports is that G2 tolerates reduced visibility well, not that it is indifferent to it: its
relative gap at MLOW is about six times its gap at MHIGH, and the earlier VLOW point shows the
same direction. That G2's own advantage over R0 in retention is larger where visibility is scarce
is consistent across both pairs, and the direct statistical test of that comparison, marginal on
the discovery seed, is confirmed on a fresh seed at $+1.53$ pp $[+0.60,+2.45]$. The usable form of
the claim is therefore quantitative and modest: at a matched operating point, a 0.28 drop in
visible fraction costs G2 about 2 percentage points of relative rate and R0 about 3.6, so G2's
extra tolerance is worth roughly 1.5 points, not an order of magnitude.

The rate-matched pair also shows why proportional readouts matter here. On MHIGH the absolute
G2−R0 rate difference is the larger of the two layouts while the retention difference is the
smaller, because R0's centralized rate degrades on that layout as well. Reporting only absolute
bps/Hz would therefore have inverted the visibility conclusion.

## 6. Limitations

- These are two designed layouts and one frozen training seed, not a distributional topology or
  multi-seed generalization estimate. Channel draws differ across layouts; do not pair T0 with T1/T2.
- The 484-real raw-CSI case omits the preprocessing state needed to reconstruct the implementation's
  normalized features exactly. The 501-real processed-input case shows one exact-tensor alternative;
  neither includes serialization headers, mask/routing metadata, or network retransmissions.
- The direct and CPU-relay counts assume fp32 transmission for each counted real value. Compression,
  deduplication, direct UE feedback, multicast, and actual network latency were not tested.
- The E12 solvers are fixed-budget feasible implementations, not converged or precision-stable
  optimizer references.
- In the visibility family, $V$ and mean received power both track AP ring radius, so the retention
  trend is an association across layouts and not an isolated visibility effect. The
  pre-registration declared this limit before the rates were measured.
- The visibility pair is three layouts from one geometry family — a regular five-AP ring around a
  fixed RIS ring and user disc — with one training seed and frozen checkpoints. It is not a
  distributional topology result, and it does not license interpolation to untested radii.
- Cross-layout comparisons are unpaired because each layout draws its own channels; only the
  within-layout G2−R0 contrasts are paired. All intervals are normal approximations from 50
  batch means.
- Mask-only calibration used 80 samples, and holdout $V$ came out 0.03–0.05 below the calibration
  value on every layout. Read the calibration table as a ranking of candidates, not as an estimate
  of $V$.
- No E12 model-based arm, codec arm, or fixed-RIS control was run on VLOW or VHIGH, so the
  continuation carries no external comparison on those layouts.
- Matching the centralized rate removes the operating-level confound but not every difference.
  MLOW and MHIGH differ in AP ring radius and in user spread, so inter-user separation and channel
  correlation change with visibility; the design isolates visibility from rate level, not from
  geometry in general.
- Both matched layouts use a user radius other than the trained 100 m, so the frozen policies are
  evaluated out of distribution, as they already were for AP radius. R0's centralized rate falls
  by 0.45 bps/Hz between the two layouts while G2's rises by 0.83, which is part of why the
  declared control tolerance was set at 1.5 rather than at the 1.0 used for selection.
- The comparative difference-of-differences was $[+0.01,+5.78]$ pp on the discovery seed, at the
  boundary of its rule, and the single pre-registered confirmation returned $[+0.60,+2.45]$ pp.
  Quote the confirmation estimate, $+1.53$ pp, rather than the discovery one: the larger figure is
  the selected, inflated estimate. Both come from the same frozen layout pair and training seed.
- The confirmation used 3,200 samples per layout against 400 in the discovery run, so its intervals
  are not comparable in width to the other tables in this report.
- The rate-matched grid was scored on 80 calibration samples, and the holdout centralized rates
  moved by up to 1.5 bps/Hz from their calibration values. Selection precision, not just the
  declared tolerance, limits how tightly rate can be matched this way.
- G2 and R0 were evaluated locally on an RTX 5060 Laptop GPU; T1/T2 model-based solvers ran on an
  RTX 5090, while the reused T0 E12 solver result ran on the RTX 5060. Paired channel draws match
  by deterministic seed. The same-device T0 recheck found negligible rate differences between GPUs,
  but the failed float32-vs-float64 controls show that non-convex solver paths are numerically
  sensitive more broadly.

## 7. Reproduction and artifacts

From `code/decentralized_ris/` with the `decentralized-inference` environment:

```bash
python -m experiments.topology_screen --topology T0 --samples 400 --eval_seed 20260920 --device cuda:0 --out ../../artifacts/decentralized_ris/e14_topology_stress/t0_learned.json
python -m experiments.topology_screen --topology T1 --layout experiments/topology_t1.json --samples 400 --eval_seed 20260920 --device cuda:0 --out ../../artifacts/decentralized_ris/e14_topology_stress/t1_learned.json
python -m experiments.topology_screen --topology T2 --layout experiments/topology_t2.json --samples 400 --eval_seed 20260920 --device cuda:0 --out ../../artifacts/decentralized_ris/e14_topology_stress/t2_learned.json
bash scripts/e14_topology_model_based.sh
python -m experiments.summarize_topology_stress --root ../../artifacts/decentralized_ris/e14_topology_stress --t0_solver ../../artifacts/decentralized_ris/e12_model_based_optimization/fixed_budget_protocol --out ../../artifacts/decentralized_ris/e14_topology_stress/comparison.json
```

The learned checkpoints are G2 `iter150000.pt` in E06 and R0 `model_final_run0.pt` in E01.
The model-based script uses the E06 150k `summary.json` for configuration; on the remote host this
file was copied to `e14_topology_stress/config/summary.json` because the original E06 artifact was
not on that host. The remote launch used that copy; set `CONFIG_RUN` to its directory to reproduce
there with the maintained script. G2's first T0 batch matches E12's
stored paired evaluation exactly in centralized, paper-decentralized, and 2-bit rate. Outputs are
`artifacts/decentralized_ris/e14_topology_stress/t{0,1,2}_learned.json` and `.npz`, the two solver
subdirectories, and the pre-registration. Raw learned NPZ arrays hold 50 batch rates and 400 CSI
copy counts per topology. `experiments/summarize_topology_stress.py` joins those arrays with the
solver's per-sample NPZ and writes `comparison.json`, checking exact equality to E12's T0 G2
evaluation. T1/T2 solver artifacts were run on `lab301-5090-tailscale` and copied locally without
altering their original path strings. From the repository root,
`python code/plot_topology_stress.py --comparison artifacts/decentralized_ris/e14_topology_stress/comparison.json --out artifacts/decentralized_ris/e14_topology_stress/topology_rates.png`
generates `topology_rates.png` and `.pdf`; error bars are 1.96 times the batch-clustered standard
error. `t0_remote_check/` holds the RTX 5090 cross-device diagnostic; its online solver settings
match E12, with reduced diagnostic subset/timing settings that do not change the scored 400 samples.

The visibility pair reproduces from the same directory, writing into
`artifacts/decentralized_ris/e14_topology_stress/visibility_topology/`:

```bash
VIS=../../artifacts/decentralized_ris/e14_topology_stress/visibility_topology
python -m experiments.visibility_calibration --out $VIS/calibration.json --selection $VIS/selection.json \
  --low_layout experiments/topology_vlow.json --high_layout experiments/topology_vhigh.json
python -m experiments.topology_screen --topology T0 --samples 400 --eval_seed 20260924 --device cuda:0 --out $VIS/t0.json
python -m experiments.topology_screen --topology VLOW --layout experiments/topology_vlow.json --samples 400 --eval_seed 20260924 --device cuda:0 --out $VIS/vlow.json
python -m experiments.topology_screen --topology VHIGH --layout experiments/topology_vhigh.json --samples 400 --eval_seed 20260924 --device cuda:0 --out $VIS/vhigh.json
```

The calibration step is mask-only and needs no GPU; the three screen runs used the RTX 5060 Laptop
GPU with the same frozen checkpoints as the T0/T1/T2 screen. `visibility_calibration.py` was added
as the maintained program after the stored calibration was produced ad hoc; re-running it
reproduces every stored visible fraction to within $3.6\times10^{-15}$ and reselects the same two
radii, and it now also records its command line and the holdout seed. It leaves
`topology_vlow.json` and `topology_vhigh.json` untouched when they already hold the selected
coordinates, so re-running the calibration cannot silently rewrite a frozen layout. Outputs are `visibility_topology/{calibration,selection,
preregistration,t0,vlow,vhigh}.json` with matching `t0.npz`, `vlow.npz` and `vhigh.npz` holding the
50 per-batch rates per method and phase setting plus the 400 per-sample CSI copy counts. Retention
figures in the tables above are the `retention_per_batch` fields; the gap column is $1-D/C$.

The rate-matched pair reproduces the same way, writing into
`artifacts/decentralized_ris/e14_topology_stress/rate_matched_visibility/`:

```bash
MAT=../../artifacts/decentralized_ris/e14_topology_stress/rate_matched_visibility
python -m experiments.rate_matched_visibility --out $MAT/calibration.json --selection $MAT/selection.json \
  --low_layout experiments/topology_mlow.json --high_layout experiments/topology_mhigh.json --device cuda:0
python -m experiments.topology_screen --topology MLOW --layout experiments/topology_mlow.json --samples 400 --eval_seed 20260927 --device cuda:0 --out $MAT/mlow.json
python -m experiments.topology_screen --topology MHIGH --layout experiments/topology_mhigh.json --samples 400 --eval_seed 20260927 --device cuda:0 --out $MAT/mhigh.json
python -m experiments.summarize_rate_matched --root $MAT --out $MAT/summary.json
```

`rate_matched_visibility.py` runs the 44-point calibration grid, applies the eligibility rule, and
freezes `experiments/topology_mlow.json` and `experiments/topology_mhigh.json`; like the first
calibration it leaves an existing layout file untouched when the content already matches.
`summarize_rate_matched.py` recomputes the declared holdout control and the three pre-declared
contrasts from the per-batch arrays and writes `summary.json`, marking the run failed if the
control does not pass. Layout JSONs may now carry an optional `user_radius`, read by
`ChannelSimulator`; layouts without it keep the canonical 100 m user disc, so T0/T1/T2 and
VLOW/VHIGH are unaffected. All four runs used the RTX 5060 Laptop GPU and the same frozen
checkpoints. Outputs are `rate_matched_visibility/{preregistration,calibration,selection,mlow,
mhigh,summary}.json` with `mlow.npz` and `mhigh.npz`.

The confirmation reuses the same two frozen layouts and writes into
`rate_matched_visibility/confirmation/`:

```bash
CON=$MAT/confirmation
python -m experiments.topology_screen --topology MLOW --layout experiments/topology_mlow.json --samples 3200 --eval_seed 20260929 --device cuda:0 --out $CON/mlow.json
python -m experiments.topology_screen --topology MHIGH --layout experiments/topology_mhigh.json --samples 3200 --eval_seed 20260929 --device cuda:0 --out $CON/mhigh.json
python -m experiments.summarize_rate_matched --root $CON --out $CON/summary.json
```

No calibration or selection step runs again, by design. Each layout took about twelve minutes on
the RTX 5060 Laptop GPU. Outputs are `confirmation/{preregistration,mlow,mhigh,summary}.json` with
`mlow.npz` and `mhigh.npz` holding 400 batch means per method and phase setting.
