# 從 Snapshot Beamforming 到 Decentralized Active-CSI Control：整體實驗計畫

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-18
- Last Updated: 2026-08-28
- Verification Status: ACTIVE（Stage 0–5A 已完成；Stage 5B frozen-GNN baseline 與 beamformer-matched RL 尚未實作或執行）
- Version Label: decentralized_active_csi_plan_v13_compacted
- Living Report: `doc/decentralized_active_csi_experiment_report.md`
- Active Detail: `doc/stage5_joint_decentralized_active_csi_experiment_plan.md`

## 1. 核心研究問題與目前 scope

在每個 decision period 的 CSI feedback budget 有限時，各 AP 如何只利用 local stored CSI、CSI age/history、previous update 與 association，決定：

1. 哪些 AP–UE CSI links 需要更新；
2. 哪些 UE 由哪些 AP 服務；
3. 如何在 decentralized execution 下改善 long-term rate，同時控制 switching 與 freshness。

最終方法固定為 **frozen topology-matched GNN + AP-local RL controller**。Primary path 是 RL + decentralized GNN；RL + centralized GNN 是 upper reference；RL + RZF 是 conventional control。三個 policy 必須用各自 matching beamformer reward 訓練。Frozen GNN 不 fine-tune；把 RZF-trained policy 直接換接 GNN 只能算 transfer diagnostic。

Stage 2–4 已降級為 Stage 5 的 causal parent qualifications，不再各自承擔「dynamic association 較好」或「更新 CSI 較好」的一般性研究主張，也不再追加 stage-local method search。

## 2. 凍結的共同 contract

- Topology：200 m × 200 m square torus；5 APs、8 UEs、10 m AP–UE height difference。
- Physical layer：$M=2$、15 dBm/AP、`noise_power=1e-12`、direct Rayleigh fading。
- Time：1 ms channel frame；straight 0/30/80 km/h development trajectories；每條 2000 frames。
- Association：每 50 ms 更新；每個 UE 固定由 top-2 APs 服務；不加 per-AP capacity。
- Feedback：每 1 ms 決策；primary hard budget $B=2$ per AP。
- Beamforming：RZF、frozen centralized GNN 或 frozen decentralized GNN只讀合法 stored CSI 與當期 mask；rate 一律由同一期 true CSI 計算。
- Comparison：同一 cell 內共用 positions、channel innovations、association/update traces、noise 與 power constraints。
- Sampling：development 使用 seed 0；正式 inference 使用至少 5 個 paired seeds，且不把 correlated frames 當獨立樣本。

Topology、channel、budget、action order 或 visibility contract 若改變，必須重新通過 parent boundary；不得沿用不相容 checkpoint 或數值 baseline。

## 3. 階段狀態與唯一用途

| Stage | 唯一用途 | 狀態 | 後續處理 |
|---|---|---|---|
| 0 | RIS-GNN positive control | 完成 | 只保留可學習的正向控制 |
| 1A/1B | no-RIS negative control 與 noise calibration | 完成 | 不作主方法比較 |
| 1C | BPP topology qualification 與 frozen GNN checkpoint | 完成 | checkpoint/layout 凍結 |
| 2 | Mobility/channel environment qualification | 完成 | 不追加 adaptation、ranking 或 mobility matrix |
| 3 | Association action qualification；凍結 H3 | 完成 | SAC 為 negative development result，不再補 seed 或搜尋 |
| 4 | Stored/stale-CSI 與 hard-budget qualification | 完成 | 凍結 priority@B2 與 round-robin control |
| 5A | Joint evaluator、parent boundaries、RZF modular matrix | 完成 | 只作 Stage 5B 起點 |
| 5B | Frozen C/D GNN baseline 與 matching RL integration | **下一個 mandatory gate** | 唯一 active method-development 主線 |
| 6 | C/D attribution 與至多兩項必要 ablations | 待 Stage 5B | 不重開 Stage 2–4 |
| 7 | Paired multi-seed formal evidence | 待方法凍結 | 支持最終主張 |

## 4. 已完成階段的最小 handoff

### 4.1 Stage 0–1C

- Stage 0 證明原 RIS pipeline 位於 meaningful-rate regime。
- Stage 1A 沿用原 noise 後落入 noise floor，作為 negative control。
- Stage 1B 以 `noise_power=1e-12` 恢復 finite、可學習的 no-RIS baseline。
- Stage 1C 完成 200-topology geometry gate、channel/association smoke 與 5-seed snapshot training。Frozen seed-0 checkpoint、AP coordinates 與 config SHA-256 分別為：
  - `16dba87572cd9f9dfc3272b4faeda2d7127dc414945450b856758efdba7bba45`
  - `421b817e0e1e70b88db4e2ef685954b35f2e65432ffcb056190adb881df8b92d`
  - `ace7fbe95ab86dab070de0a04b7517edd1924af1271446fc69d94e72cd98cf0b`

Stage 1C 的 C–D gap只通過 development sign-consistency gate，不作穩健優勢、等效或完整收斂主張。

### 4.2 Stage 2

Square-torus straight/hotspot mobility、periodic kinematics、Gauss–Markov channel、fixed-association full-current-CSI evaluator與 provenance gates 已通過。Legacy ring/disk pilots及逐 run 修復歷史已從 active plan 移除；它們不再產生 rerun 義務。

Stage 2 的唯一輸出是 Stage 5 可重用的 environment 與 full-information anchor。Hotspot/mixed-speed robustness只在最終 claim 需要時由 Stage 7選一個最小 axis。

### 4.3 Stage 3

Top-2 dynamic-association evaluator、causality、mask/power constraints與 moving-action signal已通過。H3在 development matrix提供足夠的 rate–switching handoff，因此凍結為 modular association baseline。

Stage 3 SAC current/history pilots已完成，但在相同 paired development traces上均低於 strong heuristics且 switching 更高。這個 negative result只支持「不再擴張 Stage 3 learned association」；逐 checkpoint、逐 speed與逐 trajectory數據不再保留於 active documents。Stage 5B不沿用這兩個 SAC policy。

### 4.4 Stage 4

Stored CSI、age transition、hard budget、no leakage、B0/B8 boundaries與 33-cell seed-0 development sweep已通過。`mobility_age_priority@B2`以 aggregate rate 作 modular handoff；`round_robin@B2`保留為 fair-refresh control。

Priority 的 aggregate gain伴隨較差 UE tail rate與 weak-link starvation，因此後續必須同報 UE p05、p95/max age與 never-refreshed fraction；不得宣稱 priority 全面較佳。B1/B3/random及完整 budget sweep不進入正式 matrix。

## 5. Stage 5 — Joint active-CSI control

### 5.1 Stage 5A 已完成邊界

Stage 5A 已整合 H3 association 與 B2 feedback scheduling，並通過：

- association → feedback → selected-CSI reveal → stored-CSI beamforming → true-CSI rate 的因果順序；
- top-2、per-AP budget、mask、power與 no-leakage constraints；
- fixed-association Stage 4 boundary、H3+B8 Stage 3 boundary與 0 km/h stationary boundary；
- fixed/H3 × round-robin/priority 的 RZF 2×2 modular matrix。

Seed-0結果只確認兩種 action 在同一 evaluator 中皆非退化。H3+priority有最高 moving-setting mean rate，但仍有 tail/freshness代價；這不是正式 superiority evidence。

### 5.2 Stage 5B 執行順序

1. 建立 stage-local frozen Stage 1C GNN adapter。
2. 在相同 Stage 5 traces完成 C/D GNN的`fixed+RR@B2`、`H3+priority@B2`與`H3+B8` baselines。
3. 驗證 C/D stored-CSI visibility、dynamic mask、power與 B8 full-current reproduction。
4. 以相同 actor architecture、split、transition budget與 selection rule，分別訓練 $\pi_{\mathrm{RZF}}$、$\pi_C$、$\pi_D$。
5. 每個 policy只與相同 beamformer的 no-RL comparator比較；另作 C/D crossed policy evaluation以分離 policy optimization與 inference mode。

### 5.3 Observation、action 與 reward

AP-local actor只可讀本AP的 stored complex CSI或凍結表示、CSI age/history、previous update、previous association、local slow-timescale LSF與 budget。不得讀其他AP raw observation、未選 link 的 current CSI、future channel/position或 test normalization。

Actor為每個AP輸出association bids與feedback scores；UE-side deterministic top-2 arbitration與AP-side top-B projection保證：

$$
\sum_m A_{t,m,k}=2,\qquad U_{t,m,k}\le A_{t,m,k},\qquad \sum_k U_{t,m,k}\le2.
$$

Training可用 centralized critic，但 actor在training/evaluation必須是相同AP-local function。Reward beamformer $f\in\{\mathrm{RZF},C,D\}$時：

$$
r_j^{(f)}=
\frac{1}{50K}\sum_{t=50j}^{50j+49}\sum_{k=1}^{K}R_{t,k}^{(f)}
-0.5\frac{\lVert A_j-A_{j-1}\rVert_1}{5K}.
$$

Primary B2會填滿可用budget，因此不另加feedback penalty；actual usage仍獨立報告。

### 5.4 Mandatory gates

- Frozen checkpoint、model、layout與config hashes匹配 Stage 1C。
- C/D GNN在B2/B8與dynamic masks下finite，未associated weights為0且per-AP power合法。
- 同一controller cell的RZF/C/D paths共用byte-identical state/action/true-channel traces。
- C-GNN不讀未更新current CSI；D-GNN不超出AP-local visibility；future/hidden CSI perturbation不改當期action或beamformer input。
- Three matching policies的loss、Q、entropy、gradients、parameters、raw/projected actions與metrics finite；checkpoint reload deterministic。
- 每個path分開報是否勝random及是否勝matching H3+priority；mechanical completion不等於superiority。

任一 provenance、causality、visibility、constraint或parent boundary失敗時停止performance run，修正 contract並使用fresh root；不得換seed、放寬tolerance或fine-tune GNN掩蓋問題。

## 6. Stage 6 — 最小 attribution

Stage 5B後先做已訓練 $\pi_C/\pi_D$ 的 crossed C/D evaluation。若仍需ablation，至多從下列選兩項：

- 移除 CSI age/history；
- 移除 previous association；
- centralized-information actor upper reference。

Recurrent/GNN actor、alternative projection、fairness/age cap、hierarchical action或end-to-end GNN fine-tuning，只有Stage 5B留下明確failure hypothesis時才可預先登記一項。

## 7. Stage 7 — Formal evidence

- Primary：straight 30/80 km/h、B2；0 km/h只作stationary boundary。
- 每個RZF/C/D beamformer內比較`H3+priority@B2`、matching RL policy與`H3+B8`；`fixed+RR@B2`作simple control。
- Primary estimand是 $\Delta_D$；$\Delta_C$是upper reference，$\Delta_{\mathrm{RZF}}$是control。
- 至少5個paired seeds；使用seed或完整trajectory aggregate作統計單位，報paired effects、confidence intervals與effect sizes，必要時控制multiplicity。
- 不以未顯著宣稱等效；若需要equivalence claim，事前登記practical margin。
- Hotspot、feedback budget、coherence或network size只依最終核心claim選必要的secondary axis，不做完整Cartesian sweep。

主要metrics限制為long-term sum rate、UE p05、association switching、CSI age/freshness、feedback usage、constraint violations、inference time與實際arbitration/message count。

## 8. Artifact 與文件保留規則

每個新run至少保存effective config、random seeds、source/checkpoint hashes、raw per-seed/trajectory metrics、constraint diagnostics、completion status與log。Source snapshot必須在process啟動前固定。

文件只保留會改變後續決策的設定、gate verdict、handoff與claim boundary。已完成stage的逐iteration、逐checkpoint、逐trajectory與重複trend tables不再複製到plan/report；raw artifacts仍保留於原result roots，失敗或被排除的runs不覆寫。

## 9. 下一個 mandatory gate

直接執行 Stage 5B Gate 5.5：完成 frozen C/D GNN B2/B8 baseline compatibility。Gate 5.5通過前，不新增RL architecture、不啟動RL+GNN training，也不得宣稱frozen-GNN integration或RL+GNN improvement。

## References

[A] W.-Y. Ting et al., “Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks,” *IEEE VTC*, 2026.

[B] H. Hojatian et al., “Decentralized Beamforming for Cell-Free Massive MIMO with Unsupervised Learning,” *IEEE Communications Letters*, 2022.

[C] N. X. Tung et al., “Distributed Graph Neural Network Design for Sum Ergodic Spectral Efficiency Maximization in Cell-Free Massive MIMO,” *IEEE Transactions on Vehicular Technology*, 2025.

[D] R. Deng et al., “Intermittent CSI Update for Massive MIMO Systems With Heterogeneous User Mobility,” *IEEE Transactions on Communications*, 2019.

[E] H. A. Ammar et al., “Handoffs in User-Centric Cell-Free MIMO Networks: A POMDP Framework,” *IEEE Transactions on Wireless Communications*, 2024.

[F] E. Björnson and L. Sanguinetti, “Scalable Cell-Free Massive MIMO Systems,” *IEEE Transactions on Communications*, 2020.
