# 從 Snapshot Beamforming 到 Decentralized Active-CSI Control：整體實驗計畫

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-18
- Last Updated: 2026-08-20
- Verification Status: REPLANNED（Stage 0–1B 已完成分析；Stage 2 新版計畫已凍結，source 尚待重寫與重跑）
- Version Label: decentralized_active_csi_plan_v5_rewrite
- Living Report: `doc/decentralized_active_csi_experiment_report.md`
- Stage 2 Detail: `doc/stage2_mobility_implementation_plan.md`

## 1. 核心研究問題

在每個 decision period 的 CSI feedback budget 有限時，各 AP 如何只利用 local observation、CSI age/history 與 previous association，決定：

1. 哪些 AP–UE CSI links 需要更新；
2. 哪些 UE 由哪些 AP 服務；
3. 如何在 decentralized execution 下維持長期 rate，同時控制 feedback 與 association switching。

最終假說是：相較於固定週期或 myopic heuristics，利用 temporal information 的 decentralized policy 能在相同 hard feedback budget 下改善 long-term sum rate 或 rate–overhead trade-off。

這個假說不能由單一 mobility inference、單一 seed，或 centralized/decentralized snapshot gap 證明。整體流程先建立可信 baseline 與 environment，再分離 association、CSI scheduling 與 joint-control effects，最後才進行正式 robustness inference。

## 2. 設計原則

- 每一階段只新增一個主要變因；其餘 channel、power、noise、mask 與 evaluator 儘量固定。
- 「程式/環境可用」與「論文主張成立」使用不同 gates。Development gate 可以小而快；formal evidence 在方法凍結後一次完成。
- 方法比較必須使用相同 channel realizations、mobility traces、association masks 與 power constraints。
- Training、model selection 與 final evaluation 的 seeds/traces 分開；temporally correlated frames 不當作獨立樣本。
- 結果不好不是改環境參數的理由。只有 contract、fairness、constraint 或 provenance 失敗才修正實作。
- 保存所有失敗或被排除的 runs 與原因，不覆寫 raw artifacts。

## 3. 階段總覽與目前狀態

| Stage | 新增的主要變因 | Association | CSI availability | Model/Policy | 目前狀態 |
|---|---|---|---|---|---|
| 0 | 重現原 RIS 系統 | 固定 | current | Snapshot GNN，學習 $(W,\Theta,c)$ | 已完成正向控制 |
| 1A | 移除 RIS | 固定 | current | Snapshot GNN，學習 $(W,c)$ | 已完成；noise-floor negative control |
| 1B | 校準 no-RIS noise scale | 固定 | current | 同一 snapshot GNN | 已完成；`1e-12` gate 通過 |
| 2 | Mobility environment | 固定於 $t=0$ | full current CSI | 凍結 Stage 1B model，inference-only | 新版計畫完成；source 待重寫 |
| 3 | Dynamic association | 動態 | full current CSI | Heuristic / association policy | 待 Stage 2 |
| 4 | Feedback budget 與 CSI aging | 固定 | 部分更新，其餘 stale | Scheduling policy | 待 Stage 2 |
| 5 | Joint association + CSI update | 動態 | 受 budget 限制 | Decentralized temporal RL | 待 Stage 3/4 |
| 6 | Observation/action/model ablation | 動態 | 受 budget 限制 | Snapshot、recurrent、joint variants | 待 Stage 5 |
| 7 | Formal robustness/scalability | 依最終方法 | 依最終問題 | 所有凍結方法 | 最終 evidence stage |

## 4. 已建立的基線

### Stage 0 — 原 RIS GNN 正向控制

**目的**：確認原始 channel、rate、power constraint、local observation 與 centralized-training/decentralized-inference pipeline 可產生 meaningful-rate results。

**固定內容**：5 APs、$K=8$、$M=2$、RIS path、固定 association；GNN 輸出 complex precoder $W$、RIS phase $\Theta$ 與 AP power fraction $c$。

**通關標準**：antenna/power 趨勢合理，power/mask/decentralized inference tests 通過。既有單一-seed結果只作正向控制，不延伸成統計主張。

### Stage 1A — 忠實移除 RIS

**目的**：只移除 reflected path 與 RIS output，觀察原數值設定是否仍適合 direct-only system。

**固定內容**：保留 Stage 0 的 channel scale、`noise_power=4e-4`、optimizer 與 training protocol；GNN 只學習 $(W,c)$。

**已知結論**：signal 落入 noise floor，sum-rate/task-gradient 幾乎消失，optimizer update 被 weight-decay contribution 主導。此結果保留為 negative control，不宣稱 decentralized 等同 centralized。

### Stage 1B — No-RIS 數值尺度校準

**目的**：排除 Stage 1A 的 noise-scale confound，建立後續所有 stages 共用的 snapshot baseline。

**主設定**：5 APs、$K=8$、$M=2$、$P_{\max}=15$ dBm、`noise_power=1e-12`、Stage 1 architecture 與固定 association。

**已通過 gate**：5 seeds、每 seed 2000 iterations 的 gradients、training curves、parameters、artifacts 與 final metrics 均 finite，且未重現 parameter collapse。Centralized/decentralized mean gap 很小，不能據此宣稱方向或等效；GNN 勝過 MRT/RZF 也不是本 gate 的必要條件。

## 5. Stage 2 — Mobility Environment 與 Frozen Inference

### 5.1 Stage 2 要證明什麼

Stage 2 只證明：

1. straight-line 與 hotspot/semi-Markov mobility generators 正確；
2. Stage 1B frozen snapshot model 可在這些新環境 zero-shot inference；
3. fixed-association、full-current-CSI baseline 可供 Stage 3/4 接續使用；
4. 結果 finite、constraint-valid，且能看出大致趨勢。

Stage 2 不重新訓練，不做 matched adaptation，不做多-seed方法推論，也不要求 rate 隨 mobility 增加而單調下降。

### 5.2 共用設定

- 載入 Stage 1B seed-0 final checkpoint，完全凍結參數。
- 5 APs、$K=8$、$M=2$、$P_{\max}=15$ dBm、`noise_power=1e-12`。
- $\Delta t=1$ ms、$f_c=2.6$ GHz、2000 periods（2 s）。
- Position-dependent Stage 1 path loss + first-order Gauss–Markov small-scale fading，one-step coefficient 由 $J_0(2\pi f_D\Delta t)$ 決定。
- 每條 trajectory 只在 $t=0$ 建立 association，之後固定。
- 每一期皆提供 current full CSI；C/D GNN、MRT、RZF 共用同一 frame 與 mask。
- Seed 0、每 setting 10 trajectories、`eval_time_stride=10`。

### 5.3 必跑矩陣

| Substage | Mobility | Speeds | 完整 inference runs |
|---|---|---|---:|
| 2A | Constant-speed straight line | 0、30、80 km/h | 3 |
| 2B | Hotspot semi-Markov | 3、30、80 km/h | 3 |

Straight 3 km/h 只需 channel statistical test；hotspot 0 km/h 因無法合理 transit 而不執行。另跑 low-stickiness/short-dwell 與 high-stickiness/long-dwell 兩組 environment-only diagnostics，不跑完整 beamforming inference。若 Stage 3 需要 heterogeneous mobility，可追加一組 mixed 0/3/30/80 km/h smoke run，但不列為通關條件。

### 5.4 Stage 2 gates

- $t=0$ 與 Stage 1 paired compatibility；0 km/h trajectory 靜止。
- Kinematics、boundary、reproducibility 與 path-loss formula 全部正確。
- Normalized channel mean/variance finite 且符合 convention；empirical lag correlation 對理論 curve 的 maximum absolute error `<=0.02`。
- Hotspot transition、dwell、occupancy、continuity、phase-specific speed/channel diagnostics 通過 tolerance。
- Association 在 clip 內固定；四種方法共用 current CSI/mask；beamformers 符合 association 與 per-AP power limit。
- 六組 inference 都完成並保存 config、compact metrics、diagnostics、source/checkpoint hashes 與 logs。

通過後立即進入 Stage 3 或 Stage 4。多 seeds、full stride、equivalence、method ranking與正式 speed/mobility robustness 統一延到 Stage 7。

完整環境規格、輸出 contract、runtime 與停止規則見 `doc/stage2_mobility_implementation_plan.md`。

## 6. Stage 3 — Dynamic Association with Full Current CSI

### 6.1 目的

在 CSI 完全可用時，單獨量化 mobility 造成的 association mismatch，以及 history/switching cost 對 serving-set decision 的影響。

### 6.2 新增內容

- Association action、previous association、AP capacity 與 switching cost。
- Baselines：fixed-$t=0$、current strongest-link、hysteresis、snapshot learned policy、history-based policy。
- 先固定 beamforming implementation與 full-current-CSI availability，不加入 feedback budget。

### 6.3 Gates

- 所有 serving sets 符合 AP capacity與最低/最高服務限制。
- Switching penalty 增加時，switching rate應呈合理下降趨勢；若沒有，先檢查 reward/action mask。
- Rate 必須使用 action 後同一期 true channel評估；不得讀取 future CSI。
- Stage 2 中舊有的 current-RSSI reassociation、fixed-association regret與 strongest-AP changes在本階段才成為正式 diagnostics。

## 7. Stage 4 — Feedback Budget 與 Stale CSI，固定 Association

### 7.1 目的

單獨回答 active CSI acquisition 是否有效，不讓 association switching 混入。

### 7.2 新增內容

- 每個 AP 每期 hard update budget $B_m$。
- 未更新 links 保留 stored CSI，並更新 age與 update mask；true CSI只供 environment/reward/evaluation。
- Baselines：random、round-robin、fixed-period、age-based、mobility-based scheduling。

### 7.3 Gates

- Budget 足以更新全部 links 時，結果回到對應 Stage 2 full-current-CSI baseline。
- Budget 降低時平均 CSI age 合理上升；每一步 update action符合 hard budget。
- Policy只能讀 stored CSI/age；rate只由 true current CSI計算，禁止 leakage。

## 8. Stage 5 — Joint Decentralized Active-CSI Control

### 8.1 目的

共同控制 CSI updates與 association，直接檢驗核心研究假說。

### 8.2 Observation、action 與 reward

- Local observation：stored local CSI、CSI age/history、previous update mask、previous association 與 local budget。
- Action：選擇更新 links與serving users；透過 mask/projection保證budget與capacity constraints。
- Reward：

$$
r_t=R_{\mathrm{sum},t}-\lambda_{\mathrm{sw}}C_{\mathrm{sw},t}-\lambda_{\mathrm{fb}}C_{\mathrm{fb},t}.
$$

若每期 feedback 是固定 hard budget，可移除 $C_{\mathrm{fb}}$ penalty，避免重複懲罰。先建立「heuristic scheduler + association policy」與「scheduler policy + heuristic association」等 modular baselines，再評估 joint policy。

### 8.3 Gates

- 所有 actions滿足 hard constraints。
- True/stored CSI separation完整，無 future information。
- 在 unlimited budget或zero switching penalty等邊界條件下，能回到對應 Stage 2/3 behavior。

## 9. Stage 6 — Ablations 與 Action Architecture

在主方法與訓練流程穩定後比較：

- snapshot GNN vs. GNN+GRU/RNN；
- 移除 CSI age、history、previous association；
- simultaneous、causal two-stage、delayed或hierarchical actions；
- centralized policy upper reference vs. decentralized execution；
- parameter count/compute-matched variants，避免把更大模型誤認為 temporal information gain。

Stage 6 的目的是找出 gain來源並凍結最終方法，不在此擴大 mobility/network sweep。

## 10. Stage 7 — Formal Robustness、Statistics 與 Scalability

Stage 7 在方法、environment API、metrics與model-selection rule全部凍結後執行。

### 10.1 測試軸

- Straight與hotspot/semi-Markov mobility、代表速度與mixed-speed users。
- Feedback budget、switching cost、channel coherence。
- AP/UE數量、AP capacity與未見過的topologies。
- Frozen Stage 1B snapshot、Stage 3/4 modular baselines、Stage 5 final method與Stage 6 selected ablations。

### 10.2 統計設計

- 至少 5 個獨立、paired seeds；若小樣本 exact test 或 CI precision 不足，依事前 power/precision rule 增加。
- 同一 seed 中所有方法共用 traces 與 innovations；以 seed 或完整 trajectory aggregate 作統計單位。
- 使用 paired effects、confidence intervals與effect sizes；多個主要comparisons作multiplicity control。
- `eval_time_stride=1`只在正式結果確有需要時執行，stride sensitivity明確標示。
- 不以「未顯著」宣稱等效；若需要 robustness/equivalence claim，事前登記 practical-equivalence margin。

### 10.3 主要 metrics與圖表

- Long-term average sum rate、5th-percentile UE rate。
- Association switching rate、平均 CSI age、feedback usage與constraint violations。
- Inference time、communication overhead與scalability。
- Sum rate vs. feedback budget、rate–switching trade-off、speed × budget gain heatmap、network-size performance/cost curve。

## 11. Seeds 與計算資源策略

| 階段 | Seeds/抽樣策略 | 可做的主張 |
|---|---|---|
| Implementation/unit tests | 固定 seed，小樣本 | Contract 與 reproducibility |
| Stage 2 development | seed 0、10 trajectories、stride 10 | Environment可用、inference可跑、大致趨勢 |
| Stage 3–6 development | 1–3 seeds或小型paired pilots | Debug、method selection，明確標為exploratory |
| Stage 7 final evidence | 至少 5 個 paired seeds；必要時 full stride | 正式 method/robustness conclusions |

不要求每個中間 stage 一開始就跑大量 seeds。最終比較仍須把 frozen Stage 1B baseline與所有入選方法放到相同 Stage 7 paired matrix中，確保沒有因延後而遺漏 baseline。

## 12. Artifact 與 Provenance 規則

每個 run 最少保存：

- effective config、random seeds與evaluation indices；
- source commit、dirty status、source SHA-256與checkpoint SHA-256；
- raw per-seed/per-trajectory metrics、summary與completion status；
- environment diagnostics與constraint checks；
- stdout/stderr log與繪圖/彙整script。

Source snapshot 必須在 process 啟動前固定；不能在長時間 process 載入後再以可變動的 working tree hash 聲稱 provenance。被中止、不完整或 contract 版本不一致的 run 保留但排除於 inference，並在 living report 記錄理由。

## 13. 執行路徑與決策點

主開發順序：

1. Stage 0 → 1A → 1B：baseline與數值尺度，已完成。
2. Stage 2：重寫environment/evaluator，跑6組frozen inference，通過後停止擴張。
3. Stage 3 與 Stage 4：分別隔離 association 與 CSI scheduling effects。
4. Stage 5：joint decentralized control。
5. Stage 6：ablation並凍結方法。
6. Stage 7：一次完成paired multi-seed robustness與scalability evidence。

Stage 2 不因多跑幾組只需約一小時，就升級成完整 statistics stage；它可以多覆蓋環境，但不重複支付之後仍須與 final methods 一起重跑的 multi-seed 成本。

## References

[A] W.-Y. Ting, R. Y. Chang, F.-T. Chien, T.-Y. Peng, and P.-H. Lin, “Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks,” *IEEE VTC*, Sep. 2026.

[B] H. Hojatian, J. Nadal, J.-F. Frigon, and F. Leduc-Primeau, “Decentralized Beamforming for Cell-Free Massive MIMO with Unsupervised Learning,” *IEEE Communications Letters*, vol. 26, no. 5, pp. 1042–1046, May 2022. [Paper](https://arxiv.org/abs/2106.16194) · [Reference code](https://github.com/HamedHojatian/CF-mMIMO-HBF)

[C] N. X. Tung, T. V. Chien, H. Q. Ngo, and W. J. Hwang, “Distributed Graph Neural Network Design for Sum Ergodic Spectral Efficiency Maximization in Cell-Free Massive MIMO,” *IEEE Transactions on Vehicular Technology*, vol. 74, no. 3, pp. 5181–5186, Mar. 2025. [Paper](https://arxiv.org/abs/2411.02900)

[D] R. Deng, Z. Jiang, S. Zhou, and Z. Niu, “Intermittent CSI Update for Massive MIMO Systems With Heterogeneous User Mobility,” *IEEE Transactions on Communications*, vol. 67, no. 7, pp. 4811–4824, Jul. 2019. [DOI](https://doi.org/10.1109/TCOMM.2019.2911575)

[E] H. A. Ammar, R. Adve, S. Shahbazpanahi, G. Boudreau, and K. V. Srinivas, “Handoffs in User-Centric Cell-Free MIMO Networks: A POMDP Framework,” *IEEE Transactions on Wireless Communications*, vol. 23, no. 8, pp. 10319–10335, Aug. 2024. [Paper](https://arxiv.org/abs/2403.08900)

[F] W.-J. Hsu, T. Spyropoulos, K. Psounis, and A. Helmy, “Modeling Time-Variant User Mobility in Wireless Mobile Networks,” in *Proc. IEEE INFOCOM*, pp. 758–766, May 2007. [DOI](https://doi.org/10.1109/INFCOM.2007.94)

[G] M. C. González, C. A. Hidalgo, and A.-L. Barabási, “Understanding Individual Human Mobility Patterns,” *Nature*, vol. 453, pp. 779–782, Jun. 2008. [DOI](https://doi.org/10.1038/nature06958)
