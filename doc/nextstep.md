# Next steps — novelty 與 contribution 的補強計畫

> **這是暫時性計畫文件。** 依 `CLAUDE.md` 的規範，本檔不是 source of truth：每個項目完成後，
> 其耐久的 setup、結果與限制應併入對應的 E-report 或
> [research positioning](./research_positioning.md) 的 §2／§4／§6／§9，然後刪除本檔。
> 本檔只處理 **novelty 與 contribution**，不處理 seed 數、topology 泛化以外的 robustness 議題。

## 1. Status

- 建立：2026-09-18
- 正確性複核：2026-09-21（統一以 G2 為主線、G1 為 no-context ablation，並保留 prior-art、finite-search oracle 與 Spearman ties／inactive-AP 邊界）
- 範圍：G2 相對既有文獻的貢獻定位，以及把定位推高一級所需的動作
- 依據：[E05](./experiments/e05_energy_consensus.md)、[E06](./experiments/e06_graph_energy_training.md)、
  [E07](./experiments/e07_message_codec.md)、[E10](./experiments/e10_dnn_benchmark.md)、
  [E12](./experiments/e12_model_based_optimization.md)、[E13](./experiments/e13_fixed_ris_beamforming.md)
- 未涵蓋：multi-seed 訓練、load balancing／association 擴展（見 positioning §8.6）

## 2. 現況判斷

目前的成果組合已足以支撐一篇方法論文；若目標是 TWC／TCOM／JSAC 等級，novelty 仍屬可辯護但偏
邊界。原因不在數字，而在 contribution 尚未被抽象成一般方法：目前容易被讀成把 anchor 論文 [1]
的 learned CPU reducer 換成固定規則、把 RIS message-passing node 換成 link memory。要往上一級，
必須附帶下列兩者之一：

1. 解釋「被移除的元件為什麼本來就不必要」的理論或機制論證；或
2. 證明這個移除**解鎖了一個原方法做不到的能力**。

G2 的完整 dataflow 是本專案提出的方法；G1 只保留為移除 context reinjection 的 matched
ablation。既有文獻涵蓋 edge／link information 等個別元素。
項目 A2 已把完整設計寫成可定義的 state/action-factorized framework，並補上融合規則的
objective 與性質。接下來仍須以可採信的同系統外部比較界定 rate–signaling–rounds trade-off。

## 3. 目前站得住的 novelty 與其 prior-art 邊界

| 項目 | 本工作的內容 | 既有邊界 |
|---|---|---|
| 可執行的 phase-proposal payload | AP→CPU 每個 AP–RIS pair 傳 $N$ 個角度加一個 local energy 純量，共 31 個 fp32，取代 R0 的 $4N=120$ | Huang et al. [7]、Xu et al. [8] 已有 BS 端 local RIS 相位與協調；AP 側提案本身不是新的 |
| Parameter-free energy-weighted circular consensus | $\boldsymbol\vartheta_{r,n}=\Pi(\sum_l \bar E_{l,r}\mathbf p_{l,r,n})$，其中 $E_{l,r}=\sum_{k\in\mathcal K_l}\lVert\mathbf H_{(l,r,k)}\rVert_F^2$，CPU 端無可訓練參數 | weighted averaging 與 consensus primitive 已有大量前例；Guo et al. [21] 有相位平均先例 |
| State/action-factorized link-memory backbone（G2；G1 為 ablation） | 移除 RIS recurrent state，保留 $R\times LK\times q$ link-memory tensor 到 late proposal readout，並把固定的 link-derived context 注入每層 AP–UE update；G1 僅移除 context | edge initialization、link information 與 permutation-equivariant wireless GNN 已有前例 [23]–[29]；但目前檢視的文獻沒有重現完整 G2 dataflow，因此 exact architecture 是本工作的設計，不能只把它降成「已知 link token」 |

真正具辨識度的是架構與介面的**組合**：persistent per-RIS link memory、AP-local executable proposal、
單輪 $N+1$ 純量、能量加權、CPU 端無參數，且整條路徑由 end-to-end sum-rate loss 訓練。代表性檢索
尚未發現重覆的完整方法；但這不是 systematic review，因此只能用具邊界的 `to our knowledge`，不能把
「edge/link GNN」本身宣稱為首次提出。

## 4. 貢獻層級的四個缺口

### 4.1 撐起數字的元件缺乏文獻防守（已處理，保留診斷紀錄）

E06 顯示主要差距出現在完整 G2 與 G0 的 representation-package comparison：10k 檢查點的
paper-decentralized rate 由 G0 的 10.7356 上升到 G2 的 18.6514。G1 ablation 為 18.4210，且在
40k 主指標上與 G2 沒有可偵測差異。也就是說，**撐起數字的是整體 backbone package，敘事核心
卻是 consensus 介面**。

當初的問題是：positioning §4 與 §10 的既有文獻全部屬於 RIS／cell-free／壓縮領域，沒有一篇討論
wireless GNN 的**架構設計本身**（node-centric 對 edge/link-centric 表示）；[14] 是 over-the-air
aggregation，不是架構對照。審稿人可以直接主張「以 edge／link token 取代 node bottleneck 在 wireless
GNN 中是已知做法」，而當時沒有段落能回應。項目 A 已補上 positioning §4.4；後續正確性檢查發現，
通用 wireless-GNN 文獻 [23]–[26] 只能證明 edge/channel-aware graph modeling 已知，不能證明完整 G2
已知，因此再加入 RIS-specific [27]–[30]，並把結論修正為「個別元素有前例，完整 dataflow 尚未找到
相同方法」。

### 4.2 「rate 較高」的敘事無法成立

E12 舊協定在 20260915 的四個 primary gate 未通過，數值仍為 diagnostic。另行宣告並在新
seed 20260920 跑完的固定預算、實際部署動作比較通過八個資格 gate，明確顯示 model-based
方法的速率高於 G2。因此 contribution 不能寫成跨方法 rate 優越性，應報告同口徑的
rate–signaling–rounds trade-off：

| 方法 | Paper-dec. rate | 協調位元／sample（fp32） | 協調輪次 | 相對 G2 |
|---|---:|---:|---:|---|
| G2-150k | 22.9587 | $RL(N+1)=620$ fp32 $=19{,}840$ | 1 | — |
| E12 centralized（固定預算） | 28.9463 | 624,640 | 2 | 31.5× 位元，G2 取得 79.3% rate |
| E12 distributed（固定預算） | 28.0170 | 46,595,840 | 1,001 | 2,348.6× 位元，G2 取得 81.9% rate |

**計數邊界**：所有方法均排除 UE→AP CSI acquisition，表中只計 AP→CPU、CPU→AP 與 AP→AP
協調；共同的 CPU→RIS 致動另加每 sample 120 個 fp32 值（3,840 bits）。E12 的原始 ledger
把致動併入 total，因此不能直接拿 total 對 G2 上行訊息相比。E12 新比較只描述固定
online budget 後的可行輸出；精度與最大 consensus residual 仍未達舊版嚴格 gate，不能
宣稱 ADMM 已收斂。

### 4.3 外部比較重新界定 headline

E10 的 DNN 只達到 6.8192 對 G2-150k 的 24.5358，但報告本身註明兩個 arm 在該預算下都未收斂，因此
這個 3.6 倍差距在審稿時可能被讀成 baseline 過弱，不能單靠它強化 contribution。E11 已排除。
E12 的新同系統比較補上外部完整方法，且以 $5.0583\pm0.2265$ bps/Hz 的 paired margin 顯示
distributed optimizer 比 G2 有更高 rate；所以 headline 應落在較低訊令與輪次的 operating point。
原本「+2.2053 ± 0.2841 over R0-500k」仍是同一家族內比較，不能當跨方法優越性的主證據。

另一個需要自行確認的事實：參考文獻 [1] 目前沒有 venue、DOI 或 arXiv ID。若它是同一研究群的前作，
外部審稿人對「相對 anchor 的 delta」會再打一次折扣，敘事必須據此調整。

### 4.4 架構已定義，因果邊界仍在

方法報告現已把 G2 的 tensor 與更新式提升為可重用的 state/action factorization：共享
action 的物理物件不必同時成為 recurrent interaction state。G2 分離 dynamic AP–UE
states、當次 forward 固定的 link evidence 與 shared RIS action readout；G1 僅作移除 context
reinjection 的 ablation。§10.2 給出可驗證的
融合性質。這仍是方法定義與性質，實驗只支持 representation package 的效果。

## 5. 行動項目

依投入產出比排序。每一項都標明前置條件、產出位置與**事前宣告**的判準；判準一旦宣告，依
`stop-on-failed-control` 原則不得在看到結果後放寬。

### A. 補齊 wireless GNN 架構文獻 —— **已完成（2026-09-18）**

- **解決**：§4.1
- **做什麼**：在 positioning §4 新增一小節「Graph architectures for wireless resource allocation」，
  說明 node-centric 與 edge／link-centric GNN 的既有分野，並明確指出 G2 的 link-token
  表示與其差異何在（per-RIS conditioning、shared masked phase head、無 RIS recurrent state）。
  在 §5.1 的表格外，不需要新增欄位。
- **候選文獻（加入 §10 前須逐一核對 title、venue 與識別碼）**：
  - Shen et al., “Graph Neural Networks for Scalable Radio Resource Management: Architecture Design
    and Theoretical Analysis,” IEEE JSAC 2021, arXiv:2007.07632
  - Eisen and Ribeiro, “Optimal Wireless Resource Allocation with Random Edge Graph Neural
    Networks,” IEEE TSP 2020, arXiv:1909.01865
  - Chowdhury et al., “Unfolding WMMSE using Graph Neural Networks for Efficient Power Allocation,”
    IEEE TWC 2021, arXiv:2009.10812
  - Shen et al., “Graph Neural Networks for Wireless Communications: From Theory to Practice,”
    IEEE TWC 2023, arXiv:2203.10800
- **判準**：能寫出一段明確區分「已知的 edge-GNN 做法」與「G2 實際的資料流」的文字；若核對後
  發現既有文獻已涵蓋 link-token 設計，則據實把 backbone 從 novelty 清單移到 §3「不構成貢獻」。
- **結果**：通用文獻 [23]–[26] 與更接近的 RIS-specific 文獻 [27]–[29] 已查證。它們建立了
  edge/channel-aware graph modeling、edge initialization、distributed graph execution 與 detailed
  link information 的 prior art，但沒有重現完整 G2 dataflow。因此 §3 只排除「link／edge features
  本身」作為貢獻；exact architecture 保留為本工作的設計，並受 representative-search 與 package
  ablation 邊界限制。Chowdhury et al. [25] 的 issue 已由誤植的 no. 10 修正為 no. 9，並補 DOI。

### A2. 將 G2 定義為 state/action-factorized framework —— **已完成（2026-09-19）**

- **解決**：§4.4；把 architecture novelty 從特定 implementation 提升為 method principle。
- **一般 formulation**：令 AP $l$ 由 local observation 產生 active action、位於
  $\mathbb T^N$ 的 executable proposal 與 local evidence；以 persistent link memory 保存共享控制物件的
  resolved evidence，只讓 interaction nodes recurrently update，再由 late readout 產生 shared action。
- **G1／G2 的角色**：G2 是完整實例與研究主線；G1 是無 context reinjection 的 ablation。G2 context 在 E06 的
  primary metric 沒有可偵測增益，因此不得獨立列成 performance contribution。
- **至少補三個 proposition**：
  1. energy-weighted circular rule 是 weighted chordal-disagreement problem
     $\min_{|v_n|=1}\sum_l E_l|v_n-p_{l,n}|^2$ 在非零 resultant 下的閉式解；
  2. fusion 對 AP ordering permutation invariant，對所有 proposal 的共同 phase rotation equivariant，
     且輸出始終 unit-modulus feasible；
  3. 線上協定只需一輪與每 AP–RIS pair 的 $N+1$ scalars，CPU 無 trainable parameter。
- **產出**：把一般 framework 與 propositions 寫入 method report；positioning §2／§6 改成
  architecture–interface co-design，而不是只強調 node-free 或 averaging。
- **判準**：移除 G2、G1 ablation、RIS 與專案變數名稱後，formulation 仍能描述一類「local executable
  proposals + evidence + manifold fusion」問題；每個 proposition 都有明確假設與例外（特別是 zero
  resultant），且不把 symmetry／closed form 誤寫成性能保證。
- **結果**：[方法報告 §11](./decentralized_ris_methods.md#state-action-factorization) 先給
  不依賴 RIS 的 agent／shared-object formulation，再精確映射 G2 與 G1 ablation；
  [§10.2](./decentralized_ris_methods.md#energy-consensus) 給 weighted chordal objective、
  非零 resultant 下的唯一閉式解、AP permutation／共同旋轉 symmetry、fallback 例外與
  一輪 $N+1$ scalar 介面計數。相對 Lim 與 Vu 的
  [RIS–UE edge-feature GNN](https://arxiv.org/abs/2501.14987)，差異界定為完整
  fixed link memory → recurrent AP–UE state → late AP-local proposal → evidence fusion 流程，
  不主張「使用 link 資訊」為首次，也不把 manifold consensus 的性質列為獨立 novelty。

### B. 把 claim 改寫成 rate–signaling–rounds 的 trade-off 敘事 —— **已完成（2026-09-19）**

- **解決**：§4.2
- **做什麼**：改寫 positioning §2 與 §6，把主張從跨方法 rate 優越性改為固定單輪、固定 payload 下的
  訊令與協調輪次優勢，並以 §4.2 的同口徑表格作依據。舊 E12 數值維持 diagnostic；
  新 E12 比較只承擔固定預算部署動作的 claim。
- **前置**：項目 E 的新比較已完成。
- **判準**：改寫後的 §6 不含任何未被現有證據支持的跨方法優越性陳述。
- **結果**：positioning §2／§6 已改寫，E12 報告保留嚴格 gate 失敗與新協定的界線。
  有模型速率較高與 G2 訊令／輪次較低兩個同時成立的方向，因此稱為 tested operating
  points 的 trade-off，不宣稱全面 Pareto front 或實測網路 latency。

### C. Energy weight 的理論化與 oracle-weight 診斷 —— **已完成（2026-09-18）**

- **解決**：§2 的第 1 條路徑（解釋被移除的元件為何不必要）
- **做什麼**：
  1. **解析**：嘗試證明 $\bar E_{l,r}$ 加權的 circular mean 是某個明確統計模型下的最佳融合。可行
     方向之一是 von Mises 假設：若 AP $l$ 對 element $(r,n)$ 的相位提案誤差服從 concentration
     $\kappa_{l,r}\propto E_{l,r}$ 的 von Mises 分布，則能量加權 circular mean 即為共同方向的
     MLE。另一方向是證明 $\sum_{k\in\mathcal K_l}\lVert\mathbf H_{(l,r,k)}\rVert_F^2$ 正比於
     AP $l$ 對 $\vartheta_{r,n}$ 的一階 sum-rate sensitivity 量級。
  2. **數值驗證**：原先規劃的「量測 $\lvert\partial R/\partial\vartheta_{r,n}\rvert$ 並與
     $E_{l,r}$ 求相關」在**執行前**被判定為循環論證：權重固定時
     $\partial R/\partial\mathbf p_{l,r,n}=\bar E_{l,r}\,\partial R/\partial(\text{resultant})$，
     相關係數會恆為正而不帶資訊；且 $\vartheta_{r,n}$ 是融合後的共用量，本來就沒有 per-AP 的版本。
     改用 **oracle-style weight 比較**：凍結 beamformer 與 proposals，只把 consensus 權重
     $w_{l,r}$ 當成每個 sample、每個 RIS 的自由變數，以真實 continuous sum rate 做 300-step、
     equal／energy 雙初始化 Adam 搜尋，得到不可部署的 fitted weight $w^{\rm fit}$，再回答兩個問題——
     energy 權重回收了 equal→best-found 差距的多少比例，以及 $\bar E_{l,r}$ 與 $w^{\rm fit}_{l,r}$ 的
     秩相關有多高。這個版本沒有原先的梯度循環性，但有限步數的非凸搜尋不是 global oracle。
- **事前宣告的判準**：見 `experiments/consensus_weight_oracle.py` 的 `DECISION` 常數。主判準是
  gap recovery $\ge 0.50$ 且其 95% cluster CI 下界大於 0；達標時 $E_{l,r}$ 可由 heuristic 升格為
  有依據的 fitted-weight proxy，不能升格為 globally optimal weight。未達標時 §6 的措辭維持
  「經驗上有效的固定規則」。依
  `stop-on-failed-control` 原則，不得在看到結果後調整門檻重跑。
- **產出**：方法報告 §10 新增推導小節；數值診斷併入 E05（同一個研究問題：consensus 權重該用什麼），
  不另開 E-number。
- **結果**：解析部分寫入方法報告
  [§10.1](./decentralized_ris_methods.md#energy-consensus)——在 independent von Mises 模型且權重使用
  各 AP 提案 concentration 時，circular consensus 是共同相位的 MLE；並標示三條邊界（分布與獨立性是假設、
  $\kappa\propto E$ 是代理假設、真正的目標是 sum rate 而非相位 MLE）。數值部分寫入
  [E05 §3.1／§4.1](./experiments/e05_energy_consensus.md)：frozen G2-150k、400 samples、兩個
  evaluation seeds，energy 權重回收 $0.9301\pm0.0047$ 的 equal→best-found 差距；六個宣告控制全數
  通過，兩個判準都達標。有限搜尋找到 $1.2544\pm0.0960$ bps/Hz 的改善，但這是未知 global headroom
  的下界；相對 fitted reference 的 recovery 可能高估相對 global optimum 的 recovery。
- **正確性修正（2026-09-18）**：初版 rank diagnostic 誤用不處理 ties 的 ordinal rank，並替常數
  equal weights 報出沒有統計意義的 Spearman reference。程式已改為只對 active AP 計算 tie-corrected
  Spearman、移除 equal reference，並重新執行兩個 evaluation seeds。2-bit 列也明列為把
  continuous-optimized weights 量化後評估，不是另行最佳化的 2-bit oracle。
- **衍生的下一步（尚未執行）**：energy 比 fitted reference 更集中，暗示同一個 strictly local 純量的
  較平坦轉換可能吃掉部分差距。這必須另行宣告並在新的 evaluation seed 上確認，不得在本 holdout
  上調參。

### D. Message-domain 的位元前緣，並在 matched budget 下比較 —— **已完成（2026-09-19）**

- **解決**：§2 的第 2 條路徑（解鎖新能力）
- **結構性理由**：G2 的訊息**本身就是相位**，因此 $b$-bit 量化有直接物理意義，且 parameter-free
  融合規則不受量化影響；R0 的 $4N$ latent 必須在一個 reducer 從未於量化下訓練過的 latent space
  中壓縮。這是 G2 相對 anchor 的結構性優勢，目前尚未被證據支持。
- **注意用語**：E06 表中的「2-bit」是 **RIS 致動解析度**，不是 AP→CPU 訊息量化。兩者在文稿中
  必須分開命名，避免把 G2-150k 的 2-bit 22.8187 誤讀成訊息壓縮結果（該數值高於 R0-500k 連續相位
  的 22.3305，值得引用，但它講的是致動解析度）。
- **做什麼**：重啟 [E07](./experiments/e07_message_codec.md)。先依 positioning §9 第 3 項決定
  condition-aware 的 identity control 並事前宣告，再對 frozen G2-150k 直接做訊息端量化並重跑
  consensus（免訓練），同時**對 R0 施加同等位元預算的壓縮**，在 matched bit budget 下比較。
  補上目前缺漏的 4／5／6／8-bit 相位格點。
- **判準**：先宣告「G2 在哪一個位元預算下必須不低於 R0 多少」才算通過，並於發現用的 evaluation
  seed 之外，以另一個 seed 確認勝出的操作點。
- **結果**：四條事前判準寫在
  `artifacts/decentralized_ris/e07_message_codec/prereg_matched_budget.json`，結果併入
  [E07](./experiments/e07_message_codec.md)。只有 D2 通過：在 68 bits／AP–RIS pair 下，G2 勝過
  同預算（含更低預算）最佳的 R0 wire format $1.5424\pm0.2363$ bps/Hz，並在新 seed 上重現
  $1.5003\pm0.2472$。D1（對未壓縮 anchor）、D3（壓縮損失較小）、D4（同 backbone 下
  executable-phase 介面較佳）三條都在兩個 seed 上失敗。
- **對本計畫的影響**：§2 的第 2 條路徑（「解鎖新能力」）**不成立**。壓縮優勢不是介面帶來的：
  G2 的壓縮損失比 R0 大 $0.6629\pm0.1191$，且在固定 backbone 下 executable phase 與同位元的
  Cartesian VQ 打平。§7 的目標版本 claim 必須刪掉「在 matched bit budget 下訊息更省」這一層，
  改成「在預先指定的 68-bit 預算，G2 的 paired rate 較高，與 representation-package 差異在壓縮下仍存在相符」。
  62-bit 的區間含零，高相位精度的 global grid control 也失敗，不能推廣為每個預算皆有優勢。真正站得住的較窄結論是
  R0 自己的 $4N$ latent 是最貴的花費方式。
- **附帶結果**：拿掉 G2 唯一的 energy scalar 並改用等權融合，會讓 frozen policy 崩到
  6.72–8.30 bps/Hz，因此 $N+1$ 的 payload 不能再減。另有一條新宣告的 grid-legality control
  失敗（以 grid step 而非弧度為單位），高相位精度端的 frontier 未獲認證，待 owner 決定。

### E. 關閉 E12 的 model-based 缺口 —— **固定預算部署比較已完成（2026-09-19）**

- **解決**：§4.2 的前置與 §4.3
- **狀態**：新 seed 20260920 的 E12/G2 paired 400-sample 比較通過八個可行動作 gate，
  已足以界定固定預算下的 rate–signaling–rounds operating point。舊嚴格 gate 仍在精度與
  最大 copy consensus residual 上失敗，因此沒有收斂 benchmark。
- **做什麼**：依 positioning §9 第 1 項，另行宣告一個新協定或另一個可辨識的 solver，再進行評估。
  不得在既有 holdout 上重新調參。
- **結果**：[E12](./experiments/e12_model_based_optimization.md) 保留舊 diagnostic，另於新 seed
  評估 frozen Huang adaptation 與 G2-150k。Continuous rate 為 28.9463／28.0170／22.9587
  （centralized／distributed／G2）；G2 的 paired deficits 為 $5.9876\pm0.2846$ 與
  $5.0583\pm0.2265$ bps/Hz，卻少 31.5×／2,348.6× 協調位元。結論只屬於可行、固定預算
  的實作，沒有擴張為 Huang 原演算法的收斂或最佳性宣稱。

### F. 維度遷移（較低優先，需誠實設限）

- **做什麼**：以 frozen G2-150k 在不同的 $L$、$R$、$K$ 下評估。
- **必須誠實說明的邊界**：由 `variants.py` 可見，`ApNodeUpdateLayer` 對 node 軸使用
  `_max_excluding_self`，`NodeFreePhaseHead` 對 $K$ 做 masked mean／max 池化，因此 G2 對 $R$ 與
  $LK$ 具結構無關性；但 link encoder 的輸入維度 $2M(N+1)$ 與 phase head 的 $2N$ 輸出使其仍綁定
  $N$ 與 $M$。**R0 對 $R$ 與 $L$ 同樣具備等變性**，所以跨維度不是 G2 獨有的能力，不得當成獨佔賣點。
  這個實驗的價值在於證明優勢的**廣度**，而非新能力。

### G. Uncertainty-aware proposal fusion（條件式方法擴展，不先執行）

- **何時才做**：只有完成 A2 後仍需要更強的 algorithmic novelty，且不是靠改寫就能補足時。
- **方法**：每個 AP 對 element $(r,n)$ 輸出 von Mises expert 的方向 $\mu_{l,r,n}$ 與 concentration
  $\kappa_{l,r,n}$；CPU 的 product-of-experts MAP 仍是
  $\arg\sum_l\kappa_{l,r,n}e^{\mathrm j\mu_{l,r,n}}$，維持單輪與 parameter-free CPU。
- **真正新增之處**：必須讓 $\kappa$ 有 calibration objective 或可檢驗的 uncertainty semantics；若只是
  把現有 energy 改名為 concentration，沒有新增 novelty。
- **代價**：per-element concentration 使 payload 由 $N+1$ 變成 $2N$，仍低於 R0 的 $4N$，但會犧牲
  目前最乾淨的 payload claim。per-RIS scalar $\kappa_{l,r}$ 可維持 $N+1$，但與現有方法差異較小。
- **決策（2026-09-20）**：暫不啟動 G3。E07 沒有證明可執行相位訊息的獨立壓縮優勢，
  E06 的 G1 no-context ablation 顯示 context 對主指標沒有可偵測增益；E12 的主要缺口是較低 rate 與
  G2 shared-UE cross-AP CSI 尚未計入的配送成本。[E14](./experiments/e14_topology_stress.md)
  已完成 T1/T2 拓樸 screen 與明確的直接 AP→AP 單播訊令 ledger：G2 相對 R0 的優勢保留，
  但在此配送模型下 G2 的完整建模位元高於集中式最佳化。這使 G3 更無從以省位元當作動機。
  只有在預先定義 concentration calibration 目標、matched-budget 對照與停止條件後，
  才考慮 uncertainty-aware G3。不把一般 attention、Transformer 或 learned CPU reducer
  當成升級方案；它們增加複雜度，卻破壞目前最有辨識度的 interface semantics。

## 6. 不在本計畫範圍內

- multi-seed 訓練：依 positioning §8.4，延後到 baseline 集合與 claim 凍結之後。
- load balancing、AP／UE service bound、learned association：改變可行集與目標函數，屬於另一個
  研究問題。
- 新的 R0 或 G2 長訓練：E06 已顯示 150k→300k 只增加 $0.1540\pm0.1030$，不值得再投入算力。

## 7. 建議的 claim 措辭

**目前可守版本**：提出 state/action-factorized link-memory GNN，將當次 forward 固定的
逐鏈路證據、遞迴 AP–UE interaction state 與 RIS shared-action readout 分離。各 AP 以一輪
$N+1$ 純量傳送可執行相位提案與 strictly local energy evidence；CPU 以無參數的 circular
rule 融合。此規則的 chordal objective、閉式解與對稱性有明確的非零 resultant 條件，
屬於方法性質而非獨立 novelty。相對自家 R0，G2 的 rate 提升
$2.2053\pm0.2841$ bps/Hz，payload 由 120 降為 31 個 fp32 值／AP–RIS pair；
相對 Huang-adapted 固定預算模型方法，G2 的 rate 較低，但協調位元少 31.5×／2,348.6×，
  輪次為一輪對兩輪／1,001 輪。這些位元比值排除 G2 所需 shared-UE cross-AP CSI 的配送，
  只比較已取得各自觀測後的 modeled coordination payload。E14 的直接 fp32 AP→AP 單播
  情境把 T0 G2 改算為 973,514 bits／sample、兩輪，高於集中式方法的 624,640 bits；
  因此不能主張 G2 在整體傳輸上較省。E12 未證明迭代式方法收斂，E07 的 68-bit matched-budget 差異
主要反映架構差異，不是 executable-phase 訊息本身有較小壓縮損失。

## 8. 與既有文件的關係

- 本檔不取代 positioning §8 的 benchmark plan；§8 處理「G2 是否仍具競爭力」，本檔處理
  「貢獻是否夠大、邊界是否守得住」。
- A／A2／B／C／D／E 的耐久內容已分別歸入方法報告、positioning、E05、E07 與 E12；
  本檔僅保留決策脈絡，不另造永久 framework plan。
- F 若執行，依
  positioning §8.3 的 Stage B 辦理，不另開 E-number。G 只有真正執行時才決定應延伸 E05／E06
  還是建立新 E-number；研究問題分界不清時先問 owner。
