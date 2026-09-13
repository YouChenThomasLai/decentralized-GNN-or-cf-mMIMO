# Budgeted Decentralized Active-CSI Control：方法與實驗結果

## 研究背景

本文研究移動式 cell-free massive MIMO 在逐期 feedback budget 下，如何聯合控制 AP–UE CSI 更新與服務關係。既有文獻分別處理有限回授 [1]、intermittent update [2]、mobility handoff [3] 及 decentralized GNN [4], [5]，但較少同時考慮 stale CSI、dynamic association、hard budget 與 local execution，也未解決平均 rate、UE tail、freshness 與 switching 的衝突。

## 所提方法

本文結合 frozen GNN beamformer 與 AP-local RL controller。Actor 僅讀本 AP 的 stored CSI、age/history 與 previous update/association，輸出 association/feedback scores；top-2/top-B projection 保證每 UE 由兩個 AP 服務及每 AP $B=2$。流程為 association、feedback、CSI reveal、stored-CSI beamforming，再以 true CSI 計算 rate。RZF、centralized GNN、decentralized GNN policy 各用 matching reward，並與 `H3+priority@B2` 比較。新穎處是整合 link-level active CSI acquisition、stale-CSI-aware association、decentralized execution 與 no-leakage causal evaluation。

## 實驗結果與分析

No-RL `H3+priority@B2` 在 30 km/h 保留 C/D-GNN B8 rate 的 99.876%/99.927%，80 km/h 保留 96.767%/97.076%，顯示有限 feedback 仍能維持大部分平均 rate。然而，priority 使 UE p05 降低 16.715%/21.005%，never-refreshed links 達 20.576%/34.484%，證明其以公平性與 freshness 為代價。

RL 結果為負：5 seeds × 3 speeds × 3 beamformers，共 45 組比較全無正 delta；mean delta 為 RZF −12.812 至 −11.471、C-GNN −11.797 至 −11.317、D-GNN −11.412 至 −10.630。Projection 後的 action plateau 與高 switching 顯示 continuous-score actor 可能不適合離散 joint action；partial observability、reward/actor 設計尚無 ablation，僅為待驗原因。

## 結論

整體 causal pipeline 可運作，heuristic 亦能在 B2 下保留大部分 GNN rate；但所提 RL policy 未勝 baseline。此失敗只適用於本次 continuous-score projection、五個 seeds、單一 layout/simulator，不能外推為所有 RL 方法均無效。

## 參考文獻

1. Zhang et al., “Joint Port Selection Based Channel Acquisition for FDD Cell-Free Massive MIMO,” *IEEE Transactions on Communications*, 2024. <https://doi.org/10.1109/TCOMM.2024.3356792>
2. Deng et al., “Intermittent CSI Update for Massive MIMO Systems With Heterogeneous User Mobility,” *IEEE Transactions on Communications*, 2019. <https://doi.org/10.1109/TCOMM.2019.2911575>
3. Ammar et al., “Handoffs in User-Centric Cell-Free MIMO Networks: A POMDP Framework,” *IEEE Transactions on Wireless Communications*, 2024. <https://arxiv.org/abs/2403.08900>
4. Hojatian et al., “Decentralized Beamforming for Cell-Free Massive MIMO With Unsupervised Learning,” *IEEE Communications Letters*, 2022. <https://doi.org/10.1109/LCOMM.2022.3157161>
5. Tung et al., “Distributed Graph Neural Network Design for Sum Ergodic Spectral Efficiency Maximization in Cell-Free Massive MIMO,” *IEEE Transactions on Vehicular Technology*, 2025. <https://doi.org/10.1109/TVT.2024.3493235>
