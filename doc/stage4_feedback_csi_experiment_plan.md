# Stage 4 — Feedback Budget 與 CSI Aging Qualification 結案摘要

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → closure
- Origin Date: 2026-08-25
- Last Updated: 2026-08-28
- Verification Status: ANALYZED
- Version Label: stage4_feedback_csi_closure_v4_compacted
- Parent Plan: `doc/decentralized_active_csi_experiment_plan.md`
- Living Report: `doc/decentralized_active_csi_experiment_report.md`
- Stage 2 Contract: `doc/stage2_mobility_implementation_plan.md`

## 1. 唯一任務與 disposition

Stage 4只驗證fixed association下的stored/stale-CSI state、per-AP hard feedback budget與no-leakage evaluator，並凍結一個rate-oriented scheduler及一個fair-refresh control。此任務已完成；Stage 4不再新增standalone scheduler、budget sweep或superiority study。

## 2. Frozen causal state

令true channel、stored CSI、age、fixed association與update action為$h_t$、$\hat h_t$、$a_t$、$A$、$U_t$。

- $t=0$：只初始化associated links，令$\hat h_0=h_0$、$a_0=0$。
- $t\ge1$：scheduler先讀pre-action stored state；只對$U_t=1$的links揭露$h_t$並reset age，其餘保留stored CSI且age加1。
- Beamformer只讀$\hat h_t$與$A$；rate只用同一期true $h_t$。
- Update必須滿足$U_t\le A$及每AP每frame$\sum_kU_{t,m,k}\le B$。
- Current/future未選link CSI不得影響action。

Stage 5改成dynamic association時另採all-link $t=0$ bootstrap；不得把兩種initialization overhead直接混比。

## 3. Methods與 boundaries

| Method | Role | 後續 disposition |
|---|---|---|
| B0 hold | no-refresh boundary | 只作diagnostic |
| B8 full current | information boundary | Stage 5/7 anchor |
| `round_robin@B2` | fair-refresh control | Mandatory carry-forward |
| `mobility_age_priority@B2` | rate-oriented modular handoff | Stage 5 baseline |
| B1/B3、random | historical trend cells | 不進正式matrix |

Priority使用predeclared mobility/age/slow-timescale information，不讀當期未揭露small-scale CSI。它不是fair scheduler。

## 4. Completed matrix與 gates

Straight 0/30/80 km/h、B0/B1/B2/B3/B8與三種causal scheduler共33 cells完成；每speed 10 paired trajectories、2000 frames、`eval_time_stride=1`。

| Gate | Verdict | Minimal evidence |
|---|---|---|
| Stage 2 handoff/provenance | PASS | Positions、channel、association與source hashes相容 |
| State transition/hard budget | PASS | Age/reset/hold/full contracts與33-cell budget constraints通過 |
| Causality/no leakage | PASS | Current/future hidden CSI perturbation不改action |
| Boundary reproduction | PASS | B8重現Stage 2；0 km/h hold/full一致 |
| Completion/artifacts | PASS | 33/33 cells complete、arrays finite、violations 0 |

## 5. Minimal B2 result

| Speed | RR retention | Priority retention | Priority vs RR mean-rate gain |
|---:|---:|---:|---:|
| 30 km/h | 83.271% | 95.549% | 15.336% |
| 80 km/h | 71.865% | 84.194% | 16.101% |

依事前speed-level aggregate-rate rule，priority在兩個moving speeds皆超過1%門檻，因此交接Stage 5。逐trajectory raw table已從文件移除；需要重算時使用raw artifacts。

## 6. 必須保留的trade-off

Priority將updates集中於較重要links，使energy-weighted CSI quality與mean rate較佳，但在B2時UE p05低於round-robin，且部分弱associated links整條trajectory未再更新，max age可達1999 frames。

因此可支持的結論是「priority提供aggregate-rate oriented handoff」，不能寫成「priority全面較佳」。Stage 5/7必須同時報：

- UE p05 rate；
- active-link mean/p95/max age；
- never-refreshed active-link fraction；
- energy-weighted CSI NMSE；
- actual feedback usage與violations。

## 7. Stage 5 handoff

Stage 5以Stage 4的state machine、full-stride evaluator、feedback schedulers、no-leakage tests與artifacts schema為primary base，再加入Stage 3 association。Stage 4 source保持唯讀，Stage 5使用stage-local copy與hash lineage。

Primary modular cells只需`round_robin@B2`與`mobility_age_priority@B2`；B0/B8保留boundaries。B1/B3/random sweep不因Stage 5結果重新開啟。

## 8. Evidence boundary與 artifact locator

本結果只有一個environment seed、一個AP layout與每speed 10條development trajectories；沒有formal p-value、equivalence或multiplicity-controlled inference。

- Source：`code/stage4/`
- Local results：`code/stage4/results_stage4_seed0/`
- Remote results：`ws2:~/ThomasLai/code/stage4/results_stage4_seed0/`

Formal scheduler/joint-policy claim留到Stage 7 paired multi-seed evaluation。
