# Snapshot Centralized–Decentralized Network-Scaling 實驗計畫

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-20
- Verification Status: UNVERIFIED（新計畫，尚未實作或執行）
- Version Label: snapshot_network_scaling_plan_v1
- Scope: 獨立 exploratory experiment，不隸屬於 active-CSI、mobility 或 feedback-budget 計畫
- Source Baselines: `code/stage0/` 與 `code/stage1/`

## 1. 計畫定位與邊界

本計畫完全獨立於 `doc/decentralized_active_csi_experiment_plan.md`。它不修改原本 active-CSI 的研究問題、stage 編號或執行順序，也不以 mobility、CSI aging、feedback budget 或 dynamic association 為變因。

新實驗只從現有 Stage 0 與 Stage 1 程式出發，做最小限度的參數化，回答 snapshot、full-current-CSI 下的 network-scaling 問題。原始 source behavior 與已完成 results 必須保留；新 runs 寫入獨立 result directories，不覆寫原 artifacts。

第一輪明確不加入：

- strict no-inter-AP-CSI-sharing；
- feedback acquisition budget；
- mobility 或 stale CSI；
- 新 association policy；
- 新 model architecture 或為了放大 C/D gap 而做的 tuning。

## 2. 研究問題與假說

### 2.1 主問題

在 AP/UE 密度、UE/AP ratio、per-AP power、association rule 與 CSI-sharing protocol 儘量固定時，擴大 AP 數、UE 數與服務區域，centralized 與現有 decentralized snapshot inference 的性能差距如何變化？

### 2.2 次問題

1. 這個 scaling trend 在 Stage 0 RIS 系統與 Stage 1 no-RIS 系統中是否都可觀察？
2. 當 network size 增加時，association load、local-to-global visibility ratio、runtime 與 memory 如何變化？
3. C/D gap 若不增加，是穩定的平坦趨勢，還是 training、topology 或小樣本波動造成的不確定？

### 2.3 事前假說

- 當 area 與 AP/UE 數同步增加時，單一 AP 可見 CSI 佔 global CSI 的比例會降低。
- 若非本地 interference coordination 有價值，C/D relative gap 可能隨 scale 增加；由於 path loss 使遠端 links 貢獻降低，gap 也可能飽和或維持平坦。
- 本實驗不將「gap 必須變大」設為成功條件。可重現的平坦或非單調趨勢也是有效結果。

## 3. 共通設計原則

- 第一輪保留現有 C/D 定義。Centralized 使用 association-masked global CSI；decentralized 保留現有對 co-served UEs 的 additional CSI acquisition，標記為 `D-shared`。
- 每個 network size 重新訓練，不將 5-AP checkpoint 直接當成大型 network 的 performance result。
- 同一 setting/seed 中，C/D 共用 model parameters、AP/UE locations、channels、association masks、noise 與 power constraints。
- AP 數、UE 數與 area 一起增加，不把 densification 與 network-size effect 混在一起。
- 保留 per-AP power；因 area 與 AP 數成比例增加，每單位面積的 nominal transmit-power density 大致固定。
- 不用 C/D gap 大小挑 topology、noise、association threshold、training iterations 或 seed。
- Stage 0 與 Stage 1 是兩條獨立的 scaling curves；不直接以兩者 raw rate 差異主張 RIS 的因果效果。

## 4. Experiment A：Stage 0 RIS Fidelity Scaling

### 4.1 目的

在儘量保留原 Stage 0 系統定義的前提下，探索 RIS-assisted snapshot GNN 的 C/D gap 是否隨系統規模變化。

### 4.2 保留不變的內容

- APs 等角度排在圓周的既有 geometry family。
- 每個 AP 與所有 RIS association。
- 現有 AP–RIS–UE/direct channel model、association threshold、continuous/discrete RIS processing、noise、per-AP power、optimizer 與 C/D inference logic。
- Continuous RIS phase 為主結果；discrete/random-phase outputs 可保留為 supplementary diagnostics，不作主趨勢判斷。

### 4.3 Scaling family

以 $s$ 表示 area/network scale：

$$
A_s=5s,\qquad K_s=8s,\qquad L_s=4s,\qquad d_s=\sqrt{s}\,d_0,
$$

其中 $A_s$、$K_s$、$L_s$ 分別為 AP、UE 與 RIS 數，$d_s$ 表示所有 spatial length parameters。

| Scale | AP | UE | RIS | AP radius | UE/RIS spatial scale | 執行定位 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 5 | 8 | 4 | $200$ m | $100$ m | Existing-reference setting |
| 2 | 10 | 16 | 8 | $200\sqrt{2}$ m | $100\sqrt{2}$ m | 必跑 scaling point |
| 4 | 20 | 32 | 16 | $400$ m | $200$ m | Runtime gate 通過後選跑 |

`gen_location()` 的既有 radial sampling 方式在 Experiment A 不變，以維持 fidelity。

### 4.4 必要實作範圍

- 將 AP/UE/RIS counts 與 spatial scale 改為可設定參數。
- 把固定長度的 AP/RIS object construction 改為依數量建立，但不改物理公式與 association semantics。
- 依 AP 數建立現有 AP-specific heads；每個 scale 獨立訓練。
- 增加 effective config 與 topology artifacts，使每個 run 可重現。

## 5. Experiment B：Stage 1 No-RIS CF-mMIMO Scaling

### 5.1 目的

在沒有 RIS 的 snapshot beamforming 系統中，使用較典型的 cell-free mMIMO random deployment，觀察 network size 對 C/D gap 的影響。

### 5.2 Topology model

- 給定確定的 $A_s$ 與 $K_s$，APs 與 UEs 在同一方形區域內 independent uniformly distributed，即 fixed-count binomial point process（BPP）。[^uniform-square]
- 使用 square wrap-around distance 降低 boundary effects。[^uniform-square]
- AP layout 每個 topology/training seed 抽樣一次後固定；UE locations 維持現有每批重新抽樣的訓練方式。
- 第一輪不使用 PPP；PPP 會讓視窗中的 AP/UE counts 也隨機，不適合本計畫的固定-count scaling comparison。[^ppp]

這個選擇與經典 cell-free mMIMO 以 uniform-random AP/UE square deployment 加 wrap-around 的做法一致；PPP 保留為未來 topology-robustness analysis。[^uniform-square] [^ppp]

### 5.3 Scaling family

$$
A_s=5s,\qquad K_s=8s,\qquad D_s=\sqrt{s}\,D_0,
$$

其中 $D_s$ 為 square side length。建議事前設定

$$
D_0=100\sqrt{\pi}\ \text{m}\approx177.25\ \text{m},
$$

即使 baseline square 面積等於原 Stage 1 UE disk（radius $100$ m）面積。這是 geometry-based rule，不依 C/D gap 選擇。

| Scale | AP | UE | Square side | AP density | UE density | 執行定位 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 5 | 8 | $D_0$ | fixed | fixed | 新 topology baseline，必須重訓練 |
| 2 | 10 | 16 | $\sqrt{2}D_0$ | same | same | 必跑 scaling point |
| 4 | 20 | 32 | $2D_0$ | same | same | Runtime gate 通過後選跑 |

### 5.4 保留與必要改動

保留：

- Stage 1B direct-channel model、$M=2$、$P_{\max}=15$ dBm、`noise_power=1e-12`、association threshold $0.1$、optimizer、iterations 與現有 C/D logic。
- MRT/RZF 作 numerical sanity baselines，不將 GNN 必須勝過它們設為通關條件。

必要改動：

- `num_ap` 可設定，並依該數量產生 AP objects 與 model heads。
- 新增 BPP square topology generator 與 wrap-around distance。
- 儲存 AP coordinates、square side、topology seed、AP/UE density 與 wrap-around flag。

由於 APs 由圓周移到服務區域內，received-power distribution 會改變。`noise_power=1e-12` 第一輪仍固定不變，但在 full training 前必須重跑 received-power、initial task-gradient 與 decay/task-ratio diagnostics。只有 numerical/learning gate 失敗才能啟動新的 noise calibration，不得因 C/D gap 太小而改 noise。

## 6. 執行階段與 Seeds 策略

本計畫的目標是找 exploratory trend，不是第一輪就做正式 hypothesis testing。

### Phase 0 — Contract 與 smoke checks

- Seed `0`。
- 每個 branch 先跑 scale 1 與 scale 2 的 short run。
- 檢查 shapes、finite outputs、C/D paired inputs、association mask、per-AP power、topology density、artifact paths 與 runtime/memory。
- Short run 不列入趨勢結果。

### Phase 1 — Minimal trend screen

- Scales 1 與 2：seeds `0,1,2`，每 setting 完整訓練與評估。
- Scale 4：先跑 seed `0`；若無 OOM、timeout 或 numerical failure，再補 seeds `1,2`。
- 圖表顯示每個 seed 原始點、three-seed mean 與 range，不只畫平均線。
- 不執行 p-value 或「統計顯著/不顯著」主張。

### Phase 2 — Conditional seed extension

只在以下任一情況才將相關 settings 從 3 seeds 增加到 5 seeds：

- C/D gap 或 scaling direction 在 3 seeds 間頻繁反轉；
- 不同 seed 的變異與 scale 1→2 或 scale 2→4 的變化量同等或更大；
- training convergence 差異足以影響 trend 判讀；
- 後續要把結果升級為論文正式 quantitative claim 或 reviewer 要求 confidence interval。

若 3 seeds 的趨勢已清楚且 contracts/training 穩定，即停止增加 seeds。不因 gap 沒有變大就自動增加 seeds或改設定。

## 7. Metrics 與分析方法

### 7.1 主要 metrics

1. Centralized mean sum rate $R_C$。
2. Decentralized mean sum rate $R_D$。
3. Absolute gap $\Delta=R_C-R_D$。
4. Relative gap

$$
g_{rel}=\frac{R_C-R_D}{R_C}\times100\%.
$$

5. Mean per-UE rate 與 5th-percentile UE rate。
6. Sum rate per unit area，避免 network size 增加時只因 UE 變多而得到更大 raw sum rate。

### 7.2 Mechanism 與 scalability diagnostics

- 每 UE serving-AP count 的 mean/median/distribution。
- 每 AP associated-UE count 的 mean/median/distribution。
- Local-visible CSI links / global association-masked CSI links。
- Nearest-AP distance 與 strongest-link received-power distribution。
- Training time、inference time、peak GPU memory、parameter count。
- Stage 0 的 AP–RIS degree 必須等於該 scale 的 RIS 總數，確認 all-AP–all-RIS 保留。

### 7.3 主圖表

- Relative C/D gap vs. scale，Stage 0 與 Stage 1 用分開 panels。
- Mean per-UE rate vs. scale，C/D 成對顯示。
- 5th-percentile UE rate vs. scale。
- Runtime 與 peak memory vs. $A\times K$。
- Association load 與 CSI visibility ratio vs. scale。

每張趨勢圖都保留單一-seed points。只有 3 seeds 時報告 descriptive mean/range，不將其包裝成 formal population confidence interval。

## 8. Gates 與停止規則

### 8.1 Implementation gate

- Baseline scale 1 在新參數化路徑下能重現原 branch 的輸入尺寸、constraint behavior 與大致數值區間。
- 所有 beamformers 遵守 association mask 與 per-AP power limit。
- C/D evaluation 每對共用完全相同的 geometry/channel/mask。
- 所有 rates、losses、gradients 與 parameters finite。
- Scale 2/4 的 object counts、tensor shapes、saved config 與 topology files 符合 effective setting。

### 8.2 Trend-screen completion gate

- Stage 0 與 Stage 1 的 scale 1/2 各完成 3 seeds。
- Scale 4 至少完成 seed 0，或有事前規定的 OOM/runtime 排除記錄。
- 主圖表、topology diagnostics、training diagnostics 與 raw per-seed metrics 齊全。
- 結論限定為 exploratory trend：增加、平坦、飽和、反轉或 inconclusive 皆可接受。

### 8.3 不得因結果觸發的動作

- 不因 C/D gap 小而改 CSI visibility、noise、association threshold 或 model capacity。
- 不因 Stage 0/1 的趨勢不同就追溯修改已完成 runs。
- 不在本計畫中自動加入 no-sharing、feedback budget 或 mobility。這些必須成為另一份事前凍結的計畫。

## 9. Artifacts 與預定路徑

本計畫不覆寫 Stage 0/1 既有 results。建議使用：

```text
results_snapshot_scaling/
├── stage0_ris/
│   ├── scale1_A5_K8_L4/
│   ├── scale2_A10_K16_L8/
│   └── scale4_A20_K32_L16/
└── stage1_no_ris/
    ├── scale1_A5_K8/
    ├── scale2_A10_K16/
    └── scale4_A20_K32/
```

每個 run 最少儲存：

- effective config 與 CLI；
- training/topology/channel seeds；
- AP/UE/RIS coordinates 與 spatial-domain parameters；
- source commit、dirty status 與 source hash；
- checkpoint、training curves、final per-sample/per-UE metrics；
- association、power、finite-value 與 pairing checks；
- runtime、peak memory、stdout/stderr log 與 completion status。

## 10. 建議執行順序

1. 凍結本計畫與 output contract。
2. 參數化 Stage 1 AP count，建立 BPP square + wrap-around topology，先完成 scale 1/2 smoke checks。
3. 完成 Stage 1 scales 1/2 的 3-seed trend screen；再依 runtime gate 決定 scale 4 是否補齊 3 seeds。
4. 參數化 Stage 0 AP/UE/RIS counts 與 spatial scale，保留 all-AP–all-RIS 與原 geometry family。
5. 完成 Stage 0 同樣的 minimal trend screen。
6. 分開產生 Stage 0/1 plots 與 descriptive summary。
7. 只在第 6 節條件成立時補到 5 seeds；否則結束本 exploratory plan。

## 11. 實驗產出與可做主張

本計畫完成後可以說：

- 在本計畫的 Stage 0 或 Stage 1 topology family 中，觀察到 C/D relative gap 隨 scale 增加、平坦、飽和或無一致趨勢。
- 該趨勢在 3 個 paired seeds 中的穩定程度如何。
- 目前 model/evaluator 擴大到指定 AP/UE/RIS 數量時的 runtime/memory behavior。

本計畫單獨不能說：

- RIS 造成了 Stage 0/1 scaling trend 的差異；
- decentralized 與 centralized 等效；
- no-sharing、feedback restriction 或 mobility 的效果；
- 3 seeds 的 exploratory trend 代表任意 topology 的統計普遍結論。

## Footnotes

[^uniform-square]: [A] 將 AP/UE 均勻隨機放入方形區域並以 wrap-around 消除邊界效應；[B]、[C] 提供可重現的 random-topology cell-free 模擬框架。此處只借用 deployment geometry，不套用其餘參數。

[^ppp]: [D] 在 user-centric 模型中以獨立 homogeneous PPP 描述 AP/UE；[E] 以 PPP 分析隨機 AP 位置。PPP 的視窗內節點數會隨機，因此不作本 fixed-count scaling 的主設定。

## References

[A] H. Q. Ngo, A. Ashikhmin, H. Yang, E. G. Larsson, and T. L. Marzetta, “Cell-Free Massive MIMO: Uniformly Great Service for Everyone,” *IEEE SPAWC*, 2015. [Author manuscript](https://liu.diva-portal.org/smash/get/diva2%3A935259/FULLTEXT01.pdf)

[B] E. Björnson and L. Sanguinetti, “Scalable Cell-Free Massive MIMO Systems,” *IEEE Transactions on Communications*, vol. 68, no. 7, pp. 4247–4261, Jul. 2020. [Paper](https://arxiv.org/abs/1908.03119) · [Reference code](https://github.com/emilbjornson/scalable-cell-free)

[C] Ö. T. Demir, E. Björnson, and L. Sanguinetti, *Foundations of User-Centric Cell-Free Massive MIMO*, Foundations and Trends in Signal Processing, 2021. [Author book and code](https://github.com/emilbjornson/cell-free-book)

[D] P. Parida and H. S. Dhillon, “Cell-Free Massive MIMO with Finite Fronthaul Capacity: A Stochastic Geometry Perspective,” 2022. [Paper](https://arxiv.org/abs/2202.10542)

[E] A. Papazafeiropoulos, P. Kourtessis, M. Di Renzo, S. Chatzinotas, and J. M. Senior, “Performance Analysis of Cell-Free Massive MIMO Systems: A Stochastic Geometry Approach,” 2020. [Paper](https://arxiv.org/abs/2010.13223)
