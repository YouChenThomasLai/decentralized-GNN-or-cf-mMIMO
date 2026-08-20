# 從 [A] 重現到 Decentralized Active-CSI RL：實驗計畫

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-18
- Last Updated: 2026-08-19
- Verification Status: ANALYZED（Stage 1A antenna/power sweeps 與 Stage 1B 5-seed full run 已完成；Stage 2A source/checks 已存在，frozen-checkpoint mobility evaluation 與 Stage 2B 尚未完成）
- Version Label: decentralized_active_csi_plan_v4
- Living Report: `doc/decentralized_active_csi_experiment_report.md`

## 1. 核心研究問題

在每個 decision period 的 feedback budget 有限時，各 AP 如何利用 local CSI history、CSI age 與 previous association，控制：

1. 哪些 AP–UE CSI 需要更新；
2. 哪些 UE 由哪些 AP 服務；

使長期 sum rate 提高，同時減少 association switching 與 feedback overhead。

實驗原則：每一階段只加入一個主要變因；前一階段通過驗收後才往下做。

## 2. 實驗順序總覽

| 階段 | 主要變因 | Association | CSI | Policy |
|---|---|---|---|---|
| 0 | 重現 [A] | 固定 | 即時 | Snapshot GNN，學習 $(W,\Theta,c)$ |
| 1A | 忠實移除 RIS | 固定 | 即時 | 沿用 Stage 0 數值設定，學習 $(W,c)$ |
| 1B | 校準 no-RIS 數值尺度 | 固定 | 即時 | Noise sensitivity，學習 $(W,c)$ |
| 2 | UE mobility | 固定 | 全部即時更新 | Temporal environment |
| 3 | 動態 association | 動態 | 全部即時更新 | Heuristic / RL |
| 4 | Feedback budget | 固定 | 部分更新、其餘 stale | Scheduling policy |
| 5 | 完整問題 | 動態 | 部分更新、其餘 stale | Decentralized RL |
| 6 | 模型與 action ablation | 動態 | 受限 | 多種架構比較 |
| 7 | 最終評估 | 動態 | 受限 | Robustness / scalability |

2026-08-19 gate 狀態：Stage 1A 已完成並確認為 noise-floor negative control；Stage 1B 的 `noise_power=1e-12`、5-seed、2000-iteration full run 通過 numerical/learning gate，可作 Stage 2 snapshot baseline。C/D mean gap 僅 0.101%，exact paired sign-flip $p=0.125$，因此不作穩健方向或等效主張；完整數值見 living report。

## 3. 各階段工作與理由

### Stage 0 — 重現 [A]

**做什麼**

- 使用 [A] 的完整 RIS 系統、local CSI 定義、固定 association 與 centralized training/decentralized inference。
- GNN 學習複數 precoder $W$、RIS phase shifts $\Theta$，以及每個 AP 的總功率使用比例 $c$；AP–UE association 不由 GNN 學習。
- 先採主要設定：$L=5$、$K=8$、$M=2$、$R=4$、$N=30$、$P_{\max}=15$ dBm、$\rho=0.1$。
- 訓練先採 6 層 GNN、hidden dimension 64、Adam learning rate $10^{-4}$、batch size 8、2000 iterations。
- 重現 sum rate 對 AP antennas、transmit power 的趨勢，以及 centralized/decentralized 差距。
- 固定資料生成、random seeds、訓練設定與 evaluation script。

**為什麼**

- 確認 channel、rate、power constraint、local observation 與 GNN 實作正確。
- 建立後續修改的可信基準。

**驗收條件**

- 既有單一 seed 結果先作為正向控制並標明 $n=1$；需要論文級比較時再補足至少 5 個 seeds。
- AP antennas 與 transmit power 趨勢須與 [A] 一致；不要求每個數值完全相同。
- Power constraint、association mask 與 decentralized inference 均通過單元測試。

### Stage 1A — 忠實移除 RIS

**做什麼**

- 只保留 direct AP–UE channel；移除 RIS nodes、phase output 與 CPU aggregation。
- GNN 只學習複數 precoder $W$ 與每個 AP 的功率比例 $c$。$W$ 的 real/imaginary outputs 仍包含 beamforming amplitude 與 phase；只有 RIS phase $\Theta$ 被移除。
- 保留 [A] 的固定 association 與 local-CSI decentralized inference。
- 保留 Stage 0 的 channel scaling、`noise_power = 4e-4`、Adam learning rate $10^{-4}$、weight decay $10^{-6}$、batch size 8 與 2000 iterations。
- 比較 centralized GNN、decentralized GNN，以及 MRT/RZF 等簡單 beamforming baseline。

**為什麼**

- 建立「只移除 RIS、其他設定不變」的忠實 code ablation，判斷原設定在失去 reflected channel 後會發生什麼。
- Stage 1A 是負向／失效控制；若落入 noise floor，仍保留結果，不回頭修改或挑選 seeds。

**驗收條件**

- 所有方法使用相同 association、channel samples 與 power constraint。
- 完成預先排定的 5-seed antenna 與 power sweeps，保存 config、checkpoint、raw metrics 與 logs。
- GNN 不必優於 MRT/RZF；MRT/RZF 僅作 numerical sanity check。
- 不要求 centralized/decentralized gap 穩定或非零。若兩者因 task-gradient 消失而相同，結論必須標為 inconclusive，而不是宣稱 decentralized 等同 centralized。

### Stage 1B — 校準 no-RIS 數值尺度

**做什麼**

- 保持 Stage 1A 的 direct-channel model、$L=5$、$K=8$、$M=2$、$P_{\max}=15$ dBm、association mask、GNN architecture 與 centralized-training/decentralized-inference 定義。
- 只將 noise 變成可設定參數，預先測試 `1e-11`、`1e-12`、`1e-13`（在目前程式 power convention 下分別等效為 $-80$、$-90$、$-100$ dBm）。主設定預先指定為中點 `1e-12`；另外兩點作 sensitivity analysis。
- 先保持 Adam learning rate $10^{-4}$ 與 weight decay $10^{-6}$。不預設 loss scaling 或 gradient clipping。
- 每個 noise 先以 seed 0 跑 200 iterations，記錄 task-gradient norm、$\lambda\lVert\theta\rVert_2$、兩者比例、parameter norm、near-zero parameter ratio 與 validation sum rate。
- 通過 numerical gate 後，主設定跑 5 seeds；再進行 antenna 與 power sweeps。

**文獻依據**

- Hojatian et al. [B] 的 no-RIS decentralized beamforming 實驗使用 $-110$ 至 $-130$ dBW noise（即 $-80$ 至 $-100$ dBm），對應平均 SNR 約 3.1–23.1 dB，並使用 weight decay $10^{-6}$。
- Tung et al. [C] 將 noise 正規化為 $\mathcal{CN}(0,1)$，再以 normalized SNR 表示功率，支持 channel、power 與 noise 必須使用一致的 normalization。
- 上述數值是 Stage 1B 的外部先驗，不是跨 channel model 直接複製；最終仍以本程式的 received-power 與 gradient diagnostics 檢查尺度。

**為什麼**

- Stage 1A 的 `noise_power = 4e-4` 約為 MRT desired received power 中位數的 $1.36\times10^8$ 倍，使 sum-rate 與 task gradient 幾乎為零。
- Stage 1B 的目的只是排除 noise-floor confound，使 centralized/decentralized comparison 可辨識；不得為了製造 gap 或讓 GNN 勝過 MRT/RZF 而選參。

**驗收條件**

- Task-gradient norm 不得再落在 Stage 1A 的約 $10^{-7}$ 尺度；相對 Stage 0 初始 task-gradient 的比例應在 $0.1$ 至 $10$ 倍內。
- $\lambda\lVert\theta\rVert_2/\lVert\nabla L_{\text{task}}\rVert_2 \le 10^{-2}$。這是工程 guard，不是文獻常數。
- Training curve 不再維持 noise-floor 平線，模型參數與輸出不發生 Stage 1A 型 collapse。
- GNN 不必優於 MRT/RZF；centralized/decentralized gap 的大小與正負也不是選參條件。
- 所有方法繼續使用相同 channels、association masks、noise 與 per-AP power constraints。

### Stage 2A — Straight-line mobility，維持固定 association 與 full CSI

**做什麼**

- 使用獨立的 `code/stage2/` Stage 1 source-only baseline；不混入 results、checkpoints、logs 或 caches。完整規格見 `doc/stage2_mobility_implementation_plan.md`。
- 主設定採 $\Delta t=1$ ms、$f_c=2.6$ GHz、2000 periods；每個 UE 在一個 episode 內以固定速度、固定方向走 straight-line segment。
- 主速度固定為 0、3、30 km/h，另以 80 km/h 作 stress test；先測 homogeneous speed，再測 heterogeneous users。
- Small-scale fading 使用 Deng et al. [E] 的 first-order stationary Gauss–Markov model，$\rho_k=J_0(2\pi f_{D,k}\Delta t)$；large-scale amplitude 只依 Stage 1 distance/path-loss convention 隨位置更新。
- Association mask 只在 $t=0$ 依 Stage 1 規則建立一次，整條 trajectory 固定。
- 同時保存 true CSI 與 AP 端 stored CSI；本階段每期全部 links 更新，因此兩者逐元素相同。
- 主 gate 不加入新的 log-normal shadowing；通過後才以 Ammar et al. [F], [G] 的 spatially correlated shadowing 作獨立 sensitivity。
- 開發期先凍結 Stage 1B checkpoints，evaluation-only 於 0/30/80 km/h trajectories；3 km/h 與多 seeds 在方法介面凍結後再補。
- Mobility-specific matched retraining 不是主 gate；只在需要量化 domain-adaptation gain 時增加。

**為什麼**

- 先驗證時間演化，不讓 stale CSI 或 association decision 混入除錯。
- 建立後續 CSI aging 的 ground truth。
- 區分 millisecond-level CSI refresh 與 second-level handoff timescale，避免把 handoff 論文的 decision interval 直接套到 small-scale CSI aging。

**驗收條件**

- Stage 2A 的 $t=0$ 與 Stage 1 在相同 positions、initial fading、association、beamformer 下通過 paired rate equivalence；$v=0$ 時單一 episode 的 position/channel 保持不變，跨 trajectories 的 initial distribution 與 Stage 1 相容。
- Normalized small-scale channel 的 empirical lag-$\ell$ correlation 符合理論 $\rho_k^\ell$；主參數範圍內隨 lag 與速度上升而下降。
- 每步位移、distance/path-loss、trajectory boundary、fixed association 與 per-AP power constraint 均通過測試。
- Stored CSI 與 true CSI 每期完全相同；所有方法共用同一 current channel、mask、noise 與 power constraint。
- Train/validation/test 按完整 trajectory 切分，禁止把同一 trajectory 的相鄰 frames 分到不同 splits。
- 凍結 Stage 1B model 能在固定 seed、可重現的 0/30/80 km/h test traces 完成 centralized/decentralized/MRT/RZF evaluation，metrics finite 且 constraints 無違反，即通過 Stage 3 development gate。開發期可每 10 個 periods 評估一次以先看趨勢，但保留完整 traces；這不代表跨 speed traces 已嚴格配對，也不代表已通過多 seed、full-frame 論文 inference gate。

### Stage 2B — Hotspot-aware mobility robustness

**做什麼**

- 不取代 Stage 2A；在相同 channel、power/noise、fixed association、full current CSI 與模型設定下，加入外生 hotspot-aware Markov/semi-Markov mobility。
- Hotspot state 由預先固定的 transition matrix 產生，另顯式保存 dwell duration；UE 以實體速度連續移動到下一 hotspot，不允許 teleportation。
- 保留 1 ms small-scale channel block，但把 hotspot transition/dwell 放在較慢的 macro timescale。先生成 long macro trace，再依自然時間占比抽取 2 s dwell/transit clips。
- 完成 straight/hotspot 的 $2\times2$ train/test matrix，以區分 matched training 與 distribution shift。
- 開發門檻先用 frozen Stage 1B checkpoint 完成 paired straight/hotspot evaluation；$2\times2$ matched-training matrix 與 10 seeds 延後至 evidence gate。
- 詳細模型、data contract、statistics 與 gates 見 `doc/stage2_mobility_implementation_plan.md`。

**為什麼**

- Hsu et al. [I] 的 WLAN trace/model 與 González et al. [J] 的大規模手機資料均支持 frequent locations、location preference 與 revisit；Jiang et al. [K] 支持把 dwell 與 destination transition 放在較慢的 temporal model。
- Stage 2A 的直線軌跡最適合驗證 channel implementation；Stage 2B 則檢查結論是否依賴單一路徑與近乎均質的 occupancy。
- 本階段沒有 UE action/reward，因此正式名稱是 Markov/semi-Markov mobility，不稱 MDP。MDP/POMDP 留給 Stage 3 association 或後續 joint control。

**驗收條件**

- Transition、dwell、long-run occupancy、speed、boundary 與 position/channel continuity 均通過事前 tolerance。
- 每個 clip 的 association mask 只在 $t=0$ 建立一次，`stored_channels == true_channels`；不提前加入 stale CSI、dynamic association 或 history observation。
- Evidence gate 至少 10 個獨立 training seeds；以 seed/trajectory aggregate 作 paired inference，不把 temporally correlated frames 當獨立樣本。此項不阻擋 Stage 3 開發。
- 主要 estimand 為 $(C-D)_{\mathrm{hotspot}}-(C-D)_{\mathrm{straight}}$；另報告方法排名、5th-percentile UE rate、strongest-AP changes 與 frozen-association regret。
- 若 interaction 落在事前 practical-equivalence margin 且排名不變，Stage 2A 作主 baseline、Stage 2B 作 robustness；否則 Stage 3–5 必須共同報告兩種 mobility。

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

- Budget 可更新全部 links 時，結果應回到對應 mobility model 的 Stage 2A/2B full-CSI baseline。
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

- UE speed、straight/hotspot mobility model、feedback budget、switching cost。
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

- Development gate 可使用 seed-0 paired pilot 驗證 execution、constraints 與趨勢，但不作推論性主張。論文 evidence gate 每階段至少使用 5 個 random seeds；Stage 2B mobility-model comparison 至少使用 10 個 training seeds。
- 方法比較使用相同 channel realizations 與 mobility traces。
- 測 CSI scheduling 時先固定 beamformer；測 association 時先固定 CSI availability。
- 每完成一階段，保存 config、checkpoint、raw logs 與繪圖 script。
- Stage 1B 的 noise、optimizer 與 numerical gates 必須在查看 centralized/decentralized gap 前預先登記；不得依 gap、baseline 勝負或個別 seed 選參。
- MRT/RZF 是 sanity-check baselines。GNN 勝過 MRT/RZF 不是 Stage 1 的驗收條件；是否確實收到可學習的 task gradient 才是 numerical gate。
- 每次啟動、完成、中止或排除實驗，都更新 `doc/decentralized_active_csi_experiment_report.md`，記錄日期、設定、seeds、artifact path、狀態與原因，不覆寫 raw artifacts。
- 若某階段未通過 sanity check，不進入下一階段。

## 5. 最小可發表路徑

最小主線為：Stage 0 → 1A → 1B → 2A → 2B → 4 → 5 → 6 → 7。

Stage 1A 保留忠實移除 RIS 的失效結果；Stage 1B 才是後續 Stage 2A/2B 的 no-RIS snapshot baseline。增加 AP/UE 數量只作 robustness test，不作為修復 noise 或 gradient 尺度的方法。

Stage 3 是重要的診斷實驗：它可顯示在 full CSI 下，history-based association 本身能帶來多少改善，避免把既有的 mobility-aware association 效果誤認為本題的主要創新。

## Reference

[A] W.-Y. Ting, R. Y. Chang, F.-T. Chien, T.-Y. Peng, and P.-H. Lin, “Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks,” *IEEE VTC*, September 2026.

[B] H. Hojatian, J. Nadal, J.-F. Frigon, and F. Leduc-Primeau, “Decentralized Beamforming for Cell-Free Massive MIMO with Unsupervised Learning,” *IEEE Communications Letters*, vol. 26, no. 5, pp. 1042–1046, May 2022. [Paper](https://arxiv.org/abs/2106.16194) · [Reference code](https://github.com/HamedHojatian/CF-mMIMO-HBF)

[C] N. X. Tung, T. V. Chien, H. Q. Ngo, and W. J. Hwang, “Distributed Graph Neural Network Design for Sum Ergodic Spectral Efficiency Maximization in Cell-Free Massive MIMO,” *IEEE Transactions on Vehicular Technology*, vol. 74, no. 3, pp. 5181–5186, March 2025. [Paper](https://arxiv.org/abs/2411.02900)

[D] S. Huang, Y. Ye, M. Xiao, H. V. Poor, and M. Skoglund, “Decentralized Beamforming Design for Intelligent Reflecting Surface-enhanced Cell-free Networks,” 2020. [Paper](https://arxiv.org/abs/2006.12238)

[E] R. Deng, Z. Jiang, S. Zhou, and Z. Niu, “Intermittent CSI Update for Massive MIMO Systems With Heterogeneous User Mobility,” *IEEE Transactions on Communications*, vol. 67, no. 7, pp. 4811–4824, July 2019. [DOI](https://doi.org/10.1109/TCOMM.2019.2911575) · [Author manuscript](https://network.ee.tsinghua.edu.cn/niulab/wp-content/uploads/2019/12/Intermittent-CSI-Update-for-Massive-MIMO-Systems-With-Heterogeneous-User-Mobility.pdf)

[F] H. A. Ammar, R. Adve, S. Shahbazpanahi, G. Boudreau, and K. V. Srinivas, “Handoffs in User-Centric Cell-Free MIMO Networks: A POMDP Framework,” *IEEE Transactions on Wireless Communications*, vol. 23, no. 8, pp. 10319–10335, August 2024. [Paper](https://arxiv.org/abs/2403.08900)

[G] H. A. Ammar, R. Adve, S. Shahbazpanahi, G. Boudreau, and I. Bahceci, “Handoff Design in User-Centric Cell-Free Massive MIMO Networks Using DRL,” *IEEE Transactions on Communications*, vol. 73, no. 11, pp. 11368–11384, November 2025. [Paper](https://arxiv.org/abs/2507.20966)

[H] 3GPP, “Study on Channel Model for Frequencies from 0.5 to 100 GHz,” TR 38.901, Release 18, v18.1.0, February 2026. [ETSI PDF](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/18.01.00_60/tr_138901v180100p.pdf)

[I] W.-J. Hsu, T. Spyropoulos, K. Psounis, and A. Helmy, “Modeling Time-Variant User Mobility in Wireless Mobile Networks,” in *Proc. IEEE INFOCOM*, pp. 758–766, May 2007. [DOI](https://doi.org/10.1109/INFCOM.2007.94) · [Author manuscript](https://www.cise.ufl.edu/~helmy/papers/TVC-Infocom-07-published.pdf)

[J] M. C. González, C. A. Hidalgo, and A.-L. Barabási, “Understanding Individual Human Mobility Patterns,” *Nature*, vol. 453, pp. 779–782, June 2008. [DOI](https://doi.org/10.1038/nature06958)

[K] S. Jiang, Y. Yang, S. Gupta, D. Veneziano, S. Athavale, and M. C. González, “The TimeGeo Modeling Framework for Urban Mobility Without Travel Surveys,” *Proceedings of the National Academy of Sciences*, vol. 113, no. 37, pp. E5370–E5378, September 2016. [DOI](https://doi.org/10.1073/pnas.1524261113) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5027456/)
