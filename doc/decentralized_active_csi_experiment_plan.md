# 從 [A] 重現到 Decentralized Active-CSI RL：實驗計畫

## 1. 核心研究問題

在每個 decision period 的 feedback budget 有限時，各 AP 如何利用 local CSI history、CSI age 與 previous association，控制：

1. 哪些 AP–UE CSI 需要更新；
2. 哪些 UE 由哪些 AP 服務；

使長期 sum rate 提高，同時減少 association switching 與 feedback overhead。

實驗原則：每一階段只加入一個主要變因；前一階段通過驗收後才往下做。

## 2. 實驗順序總覽

| 階段 | 主要變因 | Association | CSI | Policy |
|---|---|---|---|---|
| 0 | 重現 [A] | 固定 | 即時 | Snapshot GNN |
| 1 | 移除 RIS | 固定 | 即時 | Snapshot GNN |
| 2 | UE mobility | 固定 | 全部即時更新 | Temporal environment |
| 3 | 動態 association | 動態 | 全部即時更新 | Heuristic / RL |
| 4 | Feedback budget | 固定 | 部分更新、其餘 stale | Scheduling policy |
| 5 | 完整問題 | 動態 | 部分更新、其餘 stale | Decentralized RL |
| 6 | 模型與 action ablation | 動態 | 受限 | 多種架構比較 |
| 7 | 最終評估 | 動態 | 受限 | Robustness / scalability |

## 3. 各階段工作與理由

### Stage 0 — 重現 [A]

**做什麼**

- 使用 [A] 的完整 RIS 系統、local CSI 定義、固定 association 與 centralized training/decentralized inference。
- 先採主要設定：$L=5$、$K=8$、$M=2$、$R=4$、$N=30$、$P_{\max}=15$ dBm、$\rho=0.1$。
- 訓練先採 6 層 GNN、hidden dimension 64、Adam learning rate $10^{-4}$、batch size 8、2000 iterations。
- 重現 sum rate 對 AP antennas、transmit power 的趨勢，以及 centralized/decentralized 差距。
- 固定資料生成、random seeds、訓練設定與 evaluation script。

**為什麼**

- 確認 channel、rate、power constraint、local observation 與 GNN 實作正確。
- 建立後續修改的可信基準。

**驗收條件**

- 多個 seeds 下趨勢與 [A] 一致；不要求每個數值完全相同。
- Power constraint、association mask 與 decentralized inference 均通過單元測試。

### Stage 1 — 移除 RIS，建立 no-RIS snapshot baseline

**做什麼**

- 只保留 direct AP–UE channel；移除 RIS nodes、phase output 與 CPU aggregation。
- 保留 [A] 的固定 association 與 local-CSI decentralized inference。
- 比較 centralized GNN、decentralized GNN，以及 MRT/RZF 等簡單 beamforming baseline。

**為什麼**

- 隔離 RIS 的影響，確認 [A] 的 decentralized local-CSI 架構可轉移到一般 cell-free 系統。
- 此模型將成為後續所有實驗的 snapshot baseline。

**驗收條件**

- 所有方法使用相同 association、channel samples 與 power constraint。
- Centralized/decentralized gap 穩定，且結果可由多個 seeds 重現。

### Stage 2 — 加入 mobility，但維持固定 association 與 full CSI

**做什麼**

- 定義 decision period $\Delta t$、UE trajectory、large-scale fading 與 temporally correlated small-scale fading。
- 設定 stationary、pedestrian、vehicular 等速度區間。
- 同時保存 true CSI 與 AP 端 stored CSI；本階段每期全部更新，因此兩者相同。

**為什麼**

- 先驗證時間演化，不讓 stale CSI 或 association decision 混入除錯。
- 建立後續 CSI aging 的 ground truth。

**驗收條件**

- 速度為零時，time-average 結果應接近 Stage 1。
- Channel correlation 隨時間差與速度增加而下降，且與選定模型一致。

### Stage 3 — 動態 association，但仍提供 full current CSI

**做什麼**

- 加入 association action、AP capacity、previous association 與 switching cost。
- 比較 strongest-link、hysteresis、snapshot policy 與 history-based RL。
- Beamformer 先固定，避免同時改變太多模組。

**為什麼**

- 單獨驗證 association dynamics 與 switching penalty。
- 判斷 history 在「沒有 CSI budget 問題」時能帶來多少效益。

**驗收條件**

- Switching penalty 增加時，切換次數應下降。
- 所有 association 均符合 AP capacity 與服務限制。

### Stage 4 — 加入 feedback budget 與 stale CSI，但固定 association

**做什麼**

- 對每個 AP 設 hard budget $B_m$；每期只能更新部分 AP–UE CSI。
- 未更新的 CSI 保留舊值，並記錄 age、update mask 與 CSI history。
- 比較 random、round-robin、fixed-period、age-based、mobility-based scheduling。

**為什麼**

- 單獨量化 active CSI acquisition 的價值。
- 排除 association switching，確認效能改善確實來自較好的 CSI 更新分配。

**驗收條件**

- Budget 可更新全部 links 時，結果應回到 Stage 2。
- Budget 降低時，平均 CSI age 上升；所有更新 action 均符合 hard budget。

### Stage 5 — 完整問題：共同控制 CSI updates 與 association

**做什麼**

- 每個 AP 觀察 stored local CSI、CSI age/history、update mask、previous association 與 local budget。
- Policy 共同控制 CSI updates 與 serving UEs；先不預設必須 simultaneous 或 two-stage。
- Reward 使用 true channel 計算實際 sum rate：

  $$r_t=R_{\mathrm{sum},t}-\lambda_{\mathrm{sw}}C_{\mathrm{sw},t}-\lambda_{\mathrm{fb}}C_{\mathrm{fb},t}.$$

- 若每期必須嚴格使用固定 budget，可移除 feedback penalty，只保留 hard constraint。
- 先建立 modular baselines：更新 heuristic + association heuristic/RL，再訓練 joint decentralized RL。

**為什麼**

- 這一階段才直接檢驗核心假說：在相同 feedback budget 下，temporal decentralized policy 是否能維持較好的長期效能。
- Modular baselines 可判斷 gain 來自 CSI scheduling、association，或兩者的交互作用。

**驗收條件**

- Stored CSI 只能供 policy 使用；rate 必須由 true CSI 評估，避免資訊洩漏。
- Budget、AP capacity 與 association constraints 由 action mask/projection 保證。

### Stage 6 — Ablation 與 action architecture 比較

**做什麼**

- 比較 snapshot GNN 與 GNN+GRU/RNN。
- 分別移除 CSI age、history、previous association，確認各資訊的貢獻。
- 比較 simultaneous、delayed、causal two-stage 或 hierarchical action designs。
- 比較 centralized joint policy，作為 decentralized execution 的上界參考。

**為什麼**

- Action 的時間結構是待驗證的設計選項，不應在研究開始前鎖定。
- Ablation 可證明效能不是只來自較大的模型或額外輸入。

### Stage 7 — 最終 robustness 與 scalability

**測試軸**

- UE speed、feedback budget、switching cost。
- AP/UE 數量、AP capacity、channel coherence。
- 訓練 topology 與未見過的測試 topology。

**主要 metrics**

- Long-term average sum rate、5th-percentile UE rate。
- Association switching rate、平均 CSI age、feedback 使用量。
- Constraint violation、inference time、communication overhead。

**主要圖表**

1. Sum rate vs. feedback budget，不同 UE speeds 各一條曲線。
2. Sum rate–switching trade-off。
3. RL 相對 fixed-period/myopic baseline 的 gain heatmap：speed × budget。
4. Network size 增加時的效能與 inference cost。

## 4. 實驗控制規則

- 每階段固定 train/validation/test trajectories，至少使用 5 個 random seeds。
- 方法比較使用相同 channel realizations 與 mobility traces。
- 測 CSI scheduling 時先固定 beamformer；測 association 時先固定 CSI availability。
- 每完成一階段，保存 config、checkpoint、raw logs 與繪圖 script。
- 若某階段未通過 sanity check，不進入下一階段。

## 5. 最小可發表路徑

最小主線為：Stage 0 → 1 → 2 → 4 → 5 → 6 → 7。

Stage 3 是重要的診斷實驗：它可顯示在 full CSI 下，history-based association 本身能帶來多少改善，避免把既有的 mobility-aware association 效果誤認為本題的主要創新。

## Reference

[A] W.-Y. Ting, R. Y. Chang, F.-T. Chien, T.-Y. Peng, and P.-H. Lin, “Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks,” *IEEE VTC*, September 2026.
