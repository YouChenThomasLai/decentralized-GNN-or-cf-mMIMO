# 壓縮 AP→CPU RIS 介面的貢獻分析與文獻查核

## Material Passport

- Origin Date: 2026-09-15
- Last Updated: 2026-09-16
- Verification Status: PARTIAL（repo 內數字皆已存檔；文獻查核以 WebSearch 與 arXiv／PDF 直取為主，Semantic Scholar API 全程 429，IEEE Xplore 全文未掃）
- Version Label: `ris_contribution_analysis_v2`
- Scope: `r1_ap_ris_mag` 相對 R0／R0c 的貢獻定位、先行工作查核、決定性實驗設計
- Methods: [Decentralized RIS 方法與通訊介面](./decentralized_ris_methods.md)
- Results: [RIS action screening 報告](./ris_action_screening_report.md)
- Current evidence: [Decentralized RIS 現況與證據報告](./decentralized_ris_evidence.md)
- Reference paper: [Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks](<./Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks.pdf>)

本文件只處理「這件事能不能當論文貢獻」這一個問題。完整方法定義仍以方法文件為準，§2
只展開比較四種 AP→CPU RIS 介面的 readout、傳送內容與 aggregation；數值結果以 screening
報告為準。本文件不產生新的實驗數字，只重新組織既有數字並加入外部文獻查核。

## 1. 待答問題與結論

> 把 AP→CPU 的 RIS phase feature 從 $4N$ 壓成「$N$ 個 unit-modulus 提案 + 每 AP–RIS pair
> 一個純量」，並以 parameter-free weighted circular consensus 聚合，能不能當成論文的主要貢獻？

**結論：這條線大方向有道理，但目前證據只支持「固定成熟 R0 policy 的近乎無損結構化壓縮」，
尚未支持真正的 bit-rate 優勢。若公平的 rate--distortion frontier 證明在相同 sum rate 下可省接近
$2\times$ bits，或在相同 bits 下優於最強的 R0c codec，才足以成為 conference paper 的主軸；
若要成為 journal 主軸，還需跨 $N$／AP 數／RIS 數／topology、fronthaul robustness 或進一步理論。**

四個支撐這個結論的判斷：

1. **$4N\to 2N$ 這一半是免費的。** R0c 與 R0 代數等價已在 repo 內驗證（500k 逐 batch 最大差
   $3.8\times10^{-6}$）。論文式 (26) 的 $4N$ readout 本來就含一個可以搬到 AP 端的線性層，
   搬它本身不構成貢獻。
2. **R0c 與 `r1_ap_ris_mag` 屬於同一個 polar bit-allocation family。** 前者保留每個 element
   的方向與 magnitude；後者把同一 AP--RIS pair 的 $N$ 個 radial coefficients 壓成一個共同
   scalar。它不是新的 aggregation primitive，而是 R0c 的 *row-wise constant radial
   approximation*；R0c 在資訊上是其上位集合，CPU 收到完整 R0c message 後也能模擬它。
3. **payload 優勢取決於 wire representation。** 依現行 Cartesian tensor 計數，magnitude
   相對 R0c 是 $2N+1$ 對 $2N$；若數位 fronthaul 以 phase angle／index 傳 unit-modulus
   proposal，則是 $N+1$ 對 $2N$。在 $N=30$ 時為 31 對 60 個 fp32 scalars，名目上少 48%；
   相對 R0 的 120 個則少 74%。後者才反映 intrinsic degrees of freedom，但 repo 尚未實作這個
   AP→CPU wire codec；analog I/Q channel-use 模型也不能直接套用這個結論。
4. **目前不能主張 rate superiority 或 68-bit 實測優勢。** $+0.0419$ bps/Hz（約 +0.2%）只
   是 supporting observation；可能來自移除 element-wise magnitude noise 的 regularization、
   固定 checkpoint 的偶然適配、有限 evaluation distribution 或 post-hoc 規則選擇。repo 內
   `2-bit` 是在 consensus **之後**量化最終 RIS phase，不是量化 AP message
   （[方法文件 §7](./decentralized_ris_methods.md#two-bit)），所以 68 bits/pair 仍是**待測 operating
   point**。
5. **真正應證明的命題是 task-aware compression。** 對固定 policy，R0c 的 element-wise radial
   residual 幾乎沒有 task value，因此可將 $N$ 個 radial coefficients 壓成一個 pair scalar，
   而不造成可觀察的 sum-rate 損失。文獻上尚未找到這個具體組合，但每個構件都有近例（§4.1）。

## 2. 被評估的對象

以 [方法文件 §3](./decentralized_ris_methods.md) 的定義為準。本節只計 AP→CPU 的 RIS-control
message；四種方法最後都要把共同 RIS configuration 由 CPU／controller 傳給 RIS，因此共同的
CPU→RIS 成本不影響方法間比較。以下以 $A$、$R$、$N$ 分別表示 AP 數、RIS 數及每個 RIS 的
element 數；baseline 論文以 $L$ 表示 AP 數，所以本文的 $A$ 對應其 overhead 式中的 $L$。
所有公式均假設每個 AP 都向每個 RIS 送出一則 message，若部署時只保留 active AP–RIS links，
則將 $AR$ 換成 active pair 數即可。

### 2.1 共用的 AP-side readout

四個介面先以相同的 AP-local GNN state 產生兩個 $2N$ 維 branches。對 AP $l$ 與 RIS $r$，
RIS-state branch 將最終 RIS node state $\mathbf r_{l,r}^{(D)}$ 映射為

$$
\mathbf a_{l,r}=f_{l}^{\rm RIS}(\mathbf r_{l,r}^{(D)})\in\mathbb R^{2N},
$$

AP/channel branch 則將 AP $l$ 的 local channel／edge summary $\mathbf e_l$ 映射為

$$
\mathbf c_{l}=f_{l}^{e}(\mathbf e_l)\in\mathbb R^{2N}.
$$

程式將 $\mathbf c_l$ 沿 RIS axis broadcast，再與 RIS-state branch 串接：

$$
\mathbf v_{l,r}
=
[\mathbf a_{l,r};\mathbf c_l]
\in\mathbb R^{4N}.
$$

這裡每個 branch 輸出 $2N$，是目前架構將 readout width 對齊 $N$ 組 Cartesian
real／imaginary channels 的設計選擇，不是 RIS 在物理上必須使用 $2N$ 個自由度；而且
$\mathbf a_{l,r}$ 與 $\mathbf c_l$ 在串接後仍是 latent features，不能各自解讀成可直接執行的
RIS phase。現行程式的
`RIS_readout_AP_list[ap_index]` 使 $f_l^{\rm RIS}$ 與 $f_l^e$ 為 per-AP modules，同一個 AP 的
readout 則沿所有 RIS 共用。

四個介面也共用一個 affine reduction：

$$
\mathbf W_{\rm reduce}\in\mathbb R^{2N\times4N},
\qquad
\mathbf b\in\mathbb R^{2N}.
$$

$\mathbf W_{\rm reduce}$ 與 $\mathbf b$ 是**全域共享參數**，不是 per AP、per RIS 或 per
AP–RIS pair 的參數。R0 將這一層放在 CPU；R0c、R1-Shared 與 `r1_ap_ris_mag` 則讓各 AP
持有同一組參數的副本。以 $N=30$ 為例，此 reduction 含 $60\times120+60=7{,}260$ 個參數，
參數量不乘 $A$ 或 $R$。

### 2.2 R0：先聚合 $4N$ latent，再作 learned reduction

R0 的 AP-side output 到 $\mathbf v_{l,r}$ 為止。每個 AP–RIS pair 傳送
$\mathbf v_{l,r}\in\mathbb R^{4N}$，CPU 對固定 RIS $r$ 先跨 AP 加總，再使用共享 reduction：

$$
\mathbf y_r
=
\mathbf W_{\rm reduce}\sum_{l=1}^{A}\mathbf v_{l,r}
+\mathbf b,
\qquad
\overline{\boldsymbol\theta}_{r,n}^{\rm R0}
=
\Pi(\mathbf y_{r,n}).
$$

因此 R0 的 aggregation 發生在 $4N$ latent space，$\mathbf W_{\rm reduce}$ 與 $\mathbf b$
位於 CPU，unit-modulus projection 則在 aggregation 和 reduction 之後執行。其 AP→CPU
overhead 為每 pair $4N$、全系統 $4ARN$ 個 real coefficients；換回 baseline notation 即
$4RNL$。

### 2.3 R0c：本地 reduction，聚合 raw logits 後再投影

R0c 利用 linear map 與 summation 可交換的性質，把同一個 reduction 搬到 AP。每個 AP 計算

$$
\mathbf z_{l,r}
=
\mathbf W_{\rm reduce}\mathbf v_{l,r}
+\frac{\mathbf b}{A}
\in\mathbb R^{2N}.
$$

將 bias 平分成 $\mathbf b/A$，可確保 CPU 加總後只出現一次完整 bias：

$$
\sum_{l=1}^{A}\mathbf z_{l,r}
=
\mathbf W_{\rm reduce}\sum_{l=1}^{A}\mathbf v_{l,r}
+\mathbf b.
$$

AP 傳送的是 $N$ 組未正規化的二維 logits
$\mathbf z_{l,r,n}=[z_{l,r,n}^{\rm Re},z_{l,r,n}^{\rm Im}]^T$。CPU 不再執行 learned layer，
只計算

$$
\overline{\boldsymbol\theta}_{r,n}^{\rm R0c}
=
\Pi\!\left(\sum_{l=1}^{A}\mathbf z_{l,r,n}\right).
$$

故 R0c 的 AP→CPU overhead 為每 pair $2N$、全系統 $2ARN$ 個 real coefficients。它與 R0
代數等價，但將 learned reduction 的儲存與運算從 CPU 移到 AP；CPU aggregator 本身沒有
trainable parameters。

### 2.4 R1-Shared：本地投影後作等權 circular consensus

R1-Shared 在 AP 端先計算與 R0c 相同的 $\mathbf z_{l,r,n}$，再把每個二維 logit 投影到
unit circle：

$$
\mathbf p_{l,r,n}
=
\Pi(\mathbf z_{l,r,n})
=
\frac{\mathbf z_{l,r,n}}{\lVert\mathbf z_{l,r,n}\rVert_2}.
$$

AP 傳送 $N$ 個 unit-modulus proposals $\{\mathbf p_{l,r,n}\}_{n=1}^{N}$，CPU 對所有 AP
使用等權 circular consensus：

$$
\overline{\boldsymbol\theta}_{r,n}^{\rm shared}
=
\Pi\!\left(\sum_{l=1}^{A}\mathbf p_{l,r,n}\right).
$$

程式實際使用 $1/A$ 加權；因最後仍有 $\Pi(\cdot)$，正的共同尺度不改變輸出方向，故 sum 與
mean 等價。R1-Shared 的 projection 在 AP aggregation **之前**執行，因而丟棄所有
$\lVert\mathbf z_{l,r,n}\rVert_2$。CPU 只做等權加總與 normalization，沒有 trainable
parameters。

### 2.5 `r1_ap_ris_mag`：本地投影加一個 pair-level magnitude

`r1_ap_ris_mag` 使用相同的 unit proposals，但不完全丟棄 magnitude。令

$$
m_{l,r,n}=\lVert\mathbf z_{l,r,n}\rVert_2,
\qquad
s_{l,r}=\frac{1}{N}\sum_{n=1}^{N}m_{l,r,n}.
$$

AP 傳送 $N$ 個 $\mathbf p_{l,r,n}$，以及一個供該 AP–RIS pair 的全部 $N$ 個 elements 共用的
scalar $s_{l,r}$。這個 scalar 直接由 raw logits 的 norms 計算，不是額外 confidence head，
也不增加 trainable parameters。CPU 先轉成相對權重

$$
w_{l,r}=\frac{s_{l,r}}{\sum_{j=1}^{A}s_{j,r}},
$$

再執行 weighted circular consensus：

$$
\overline{\boldsymbol\theta}_{r,n}^{\rm ap\text{-}ris\text{-}mag}
=
\Pi\!\left(\sum_{l=1}^{A}w_{l,r}\mathbf p_{l,r,n}\right).
$$

因最終 projection 對正的共同尺度不敏感，直接以 $s_{l,r}$ 或正規化後的 $w_{l,r}$ 聚合會得到
相同方向。此方法的 CPU 同樣沒有 trainable parameters；它相對 R1-Shared 多保留 pair-level
importance，相對 R0c 則丟棄 element-specific magnitude residual。

### 2.6 R0c 與兩種 circular consensus 的統一表示

將 R0c 的 logit 寫成 polar form

$$
\mathbf z_{l,r,n}=m_{l,r,n}\mathbf p_{l,r,n},
$$

三種 parameter-free CPU aggregators 可統一為

$$
\overline{\boldsymbol\theta}_{r,n}
=
\Pi\!\left(\sum_{l=1}^{A}\alpha_{l,r,n}\mathbf p_{l,r,n}\right),
$$

差別只在 AP vote 的權重粒度：

| 方法 | $\alpha_{l,r,n}$ | 保留的 magnitude information |
|---|---:|---|
| R1-Shared | $1$ | 無 |
| `r1_ap_ris_mag` | $s_{l,r}=\operatorname{mean}_n m_{l,r,n}$ | 每 AP–RIS pair 一個 scalar |
| R0c | $m_{l,r,n}$ | 每 AP–RIS–element 一個 scalar |

因此 `r1_ap_ris_mag` 不是與 R0c 無關的另一種方法，而是把 R0c 的 magnitude field 從每
element 一個權重壓成每 pair 一個權重。R0c message 包含完整的 $m_{l,r,n}$ 與
$\mathbf p_{l,r,n}$，資訊上可以重建另外兩種規則；`r1_ap_ris_mag` 的價值必須來自較低 payload
下仍能維持 task performance，而不是來自新的 weighted-sum primitive。

### 2.7 Signaling overhead 的兩種計數

「real coefficients」與「digital wire payload」必須分開。現行 repo 以 Cartesian tensor
儲存 unit proposal，故每個 $\mathbf p_{l,r,n}$ 計為 real／imaginary 兩個數；但 unit-modulus
proposal 只有一個 phase degree of freedom，數位鏈路可以只傳一個 angle 或 phase index。前者是
目前 `model.describe()` 的實作計數，後者需要明確的 encoder／decoder，repo 尚未實作。

| 方法 | AP 傳送內容 | repo Cartesian reals／pair | angle-aware digital scalars／pair | CPU aggregation | projection 位置 |
|---|---|---:|---:|---|---|
| R0 | $4N$ arbitrary latent features | $4N$ | $4N$ | latent sum + learned reduction | aggregation 後 |
| R0c | $N$ 個 raw 2-D logits | $2N$ | $2N$（$N$ angles + $N$ magnitudes） | raw-logit sum | aggregation 後 |
| R1-Shared | $N$ 個 unit proposals | $2N$ | $N$ angles | equal circular consensus | aggregation前與後 |
| `r1_ap_ris_mag` | $N$ 個 unit proposals + $1$ scalar | $2N+1$ | $N+1$ | pair-weighted circular consensus | aggregation前與後 |

若每個 phase index 使用 $b_p$ bits、每個 magnitude 使用 $b_m$ bits、pair scalar 使用 $b_s$
bits，則 R0c 的 polar codec 為 $N(b_p+b_m)$ bits／pair，R1-Shared 為 $Nb_p$，而
`r1_ap_ris_mag` 為 $Nb_p+b_s$。這些是 wire-format 的理論位元帳，不等同於現有結果中的
`dec 2-bit`；後者量化的是 consensus 後的最終 RIS phase。

以目前 $A=5$、$R=4$、$N=30$ 的 20 個 AP–RIS pairs 為例：

| 方法 | repo Cartesian reals／pair | 全系統 reals | fp32 bits | angle-aware fp32 bits（尚未實作） |
|---|---:|---:|---:|---:|
| R0 | 120 | 2,400 | 76,800 | 76,800 |
| R0c | 60 | 1,200 | 38,400 | 38,400 |
| R1-Shared | 60 | 1,200 | 38,400 | 19,200 |
| `r1_ap_ris_mag` | 61 | 1,220 | 39,040 | 19,840 |

表中的 angle-aware 數字只計 payload value／index，不含封包 header、codebook、量化範圍、同步、
通道編碼與錯誤保護。若 fronthaul 不是數位封包，而是直接以 analog I/Q 傳 phasor，則 phase
proposal 是否能由兩個 channel uses 減為一個不能只靠自由度計數決定，必須另列 physical-link
model。

## 3. Baseline 自己留下的坑（已直接從 PDF 驗證）

作者為 Wen-Yuan Ting、Ronald Y. Chang（Academia Sinica CITI）、Feng-Tsun Chien（NYCU）、
Tzu-Yuan Peng、Pin-Han Lin；PDF producer 為 `IEEE Conference eXpress`，建立日期
2026-07-09，7 頁，故判定為 2026 年 IEEE 會議論文。搜尋該標題、該作者組合與 Ronald Y. Chang
的 2026 年產出，在 arXiv、Google Scholar、Semantic Scholar 上皆無紀錄，**目前零篇引用、
無任何後續文獻**。

論文兩處明確把本文件討論的問題列為 future work（原文照引）：

> §IV：“However, it introduces additional AP-to-CPU overhead for RIS configuration, since RISs
> serve the entire network and cannot be managed by individual APs, necessitating an extra
> coordination mechanism in the decentralized framework. **Mitigating this overhead is a
> worthwhile direction for future work.**”

> §V：“**Future work includes designing a decoupled learning mechanism for RIS configuration to
> further reduce aggregation overhead.**”

其 Table II 的 signaling overhead（單位：real coefficients）為：

| | Centralized | Decentralized |
|---|---|---|
| APs → CPU | $2M\sum_{k}\lvert\mathcal L_k\rvert$ | $4RNL$ |
| CPU → APs and RISs | $2M\sum_{l}\lvert\mathcal K_l\rvert+2RN$ | $2RN$ |

**這一點對定位很重要：在 decentralized 方案中，$4RNL$ 是 AP→CPU 的唯一開銷**（CSI 不上傳
CPU），而且它隨 AP 數 $L$ 線性成長。壓縮它等於壓縮該方案 100% 的 AP→CPU 開銷，不是壓縮一個
邊角項。這是這條線最強的動機陳述，且可直接引用 baseline 自己的 Table II。

同時要注意 baseline 的 ref [15] 就是 Hojatian et al.（§4.1 第 1 項），論文正文已經點名
「unsupervised DNNs for decentralized beamforming with limited AP–CPU communication overhead;
however, RISs were not considered」。**審稿人幾乎確定知道這篇。**

## 4. 文獻查核

三個獨立 subagent 分別查 (a) fronthaul-limited decentralized learned beamforming、
(b) task-oriented feature compression 與 MARL 通訊、(c) distributed RIS phase consensus 與
directional statistics。

### 4.1 直接威脅（會被審稿人拿來說「這已經有人做了」）

| 先行工作 | 撞到的部分 | 查證 |
|---|---|---|
| Hojatian, Nadal, Frigon, Leduc-Primeau, “Decentralized Beamforming for Cell-Free Massive MIMO with Unsupervised Learning,” IEEE COMML 26(5), 2022, [arXiv:2106.16194](https://arxiv.org/abs/2106.16194) | 同樣是 cell-free decentralized beamforming，把網路切開使 AP↔controller 的訊息成為小的 learned representation，原文自述為 autoencoder 式降 signaling | 已取全文 |
| Gu, She, Quan, Qiu, Xu, “GNNs for Distributed Power Allocation: Aggregation Over-the-Air”（Air-MPNN），IEEE TWC 2023, [arXiv:2207.08498](https://arxiv.org/abs/2207.08498) | 「聚合端 parameter-free、learned 參數全在分散節點、目的是降 signaling」＋ permutation invariance 證明＋overhead 由二次降為線性 | 已取全文 |
| Das et al., “TarMAC: Targeted Multi-Agent Communication,” ICML 2019, [arXiv:1810.11187](https://arxiv.org/abs/1810.11187) | per sender–receiver 一個純量權重、parameter-free 加權和；提出動機正是修 CommNet 的等權平均 | 已取全文並核對式子 |
| Chair & Varshney, “Optimal Data Fusion in Multiple Sensor Detection Systems,” IEEE TAES 22(1), 1986 | 傳「本地決策 + 可靠度純量」的加權融合；此模式的原始形式 | snippet 層級 |
| Katsanos & Alexandropoulos, [arXiv:2601.08946](https://arxiv.org/abs/2601.08946)（2026-01）與 [arXiv:2601.13201](https://arxiv.org/abs/2601.13201) | cell-free multi-RIS：每個 BS 持有 RIS configuration 副本，以加權平均融合、**再投影回可行集**；骨架與本方法相同，但權重為 doubly-stochastic 拓樸權重而非節點自己吐出 | 已取全文 |
| Guo, Sun, Tao, “Two-Way Passive Beamforming Design for RIS-Aided FDD,” WCNC 2021, [arXiv:2101.10094](https://arxiv.org/abs/2101.10094) | 已發表「以純量權重合併兩個 RIS phase proposal」，但平均的是**角度**而非 phasor，故非 circular mean | 已取全文 |
| Fernandes & Psaromiligkos, “Model-based DL for Joint RIS Phase Shift Compression and WMMSE Beamforming,” IEEE WCL 2026, [arXiv:2510.05438](https://arxiv.org/html/2510.05438) | RIS phase 壓縮＋**sum-rate vs. control bits 曲線**（AP→RIS controller 那一段鏈路，非 AP→CPU） | 已取全文 |

### 4.2 相鄰工作（應引用，但不直接衝突）

- Xu, An, Li, Gan, Yuen, [arXiv:2301.02360](https://arxiv.org/abs/2301.02360)：RIS-assisted cell-free 的 unrolled distributed ADMM，consensus 於 RIS reflection vector，強調低 signaling 交換。
- Huang, Ye, Xiao, Poor, Skoglund, IEEE WCL 2021, [arXiv:2006.12238](https://arxiv.org/abs/2006.12238)：IRS-enhanced cell-free 的 incremental-ADMM consensus（baseline ref [14]）。
- Shao, Mao, Zhang, IEEE JSAC 2022 [arXiv:2102.04170](https://arxiv.org/abs/2102.04170) 與 IEEE TWC 2023 [arXiv:2109.00172](https://arxiv.org/abs/2109.00172)：task-oriented feature compression 的標準引用，payload-vs-task-performance 取捨的來源。
- Sohrabi, Attiah, Yu, IEEE TWC 2021, [arXiv:2007.06512](https://arxiv.org/abs/2007.06512)：把 limited feedback 當 distributed source coding，分散端學壓縮器、中心端 learned 融合。
- Gao & Gündüz（AirGNN, [arXiv:2302.08447](https://arxiv.org/abs/2302.08447)）、Lee, Yu, Dai（[arXiv:2104.09027](https://arxiv.org/abs/2104.09027)、[arXiv:2208.06963](https://arxiv.org/abs/2208.06963)）：decentralized GNN inference 在真實無線鏈路下的 robustness／privacy，非 payload 壓縮。
- Enqvist, Demir, Cavdar, Björnson, [arXiv:2506.03929](https://arxiv.org/abs/2506.03929)：RIS control signalling 的位元帳。element-wise 量化為 $Nb$ bits（$N=128,b=3\Rightarrow384$ bits）；其 LoS codebook 為 $\lceil1.1746+\log_2 N\rceil$ bits（$N=128\Rightarrow9$ bits）；建議 $\log_2 N+4$ bits。這是本方法位元帳最合適的對照基準。
- Zhu et al., IEEE TVT 2024, [arXiv:2404.14092](https://arxiv.org/abs/2404.14092)：RIS-aided cell-free 的 MARL，各 AP 只用 local CSI 做 precoding，但**RIS 由 CPU 集中控制**——同設定下的相反架構選擇，適合當動機對照。

### 4.3 覆蓋邊界

- Semantic Scholar API 在三個 agent 的整個工作期間皆回 HTTP 429，故本查核建立在 WebSearch
  ＋ arXiv／IEEE 頁面直取之上。
- **IEEE Xplore 全文檢索未執行。** 投稿前應針對 TWC／TVT 2024–2026 的
  “decentralized RIS configuration”、“RIS phase proposal aggregation” 再掃一次，特別是
  Katsanos／Alexandropoulos 該組 2026 年 1 月連發兩篇，正在這個空間高速產出。
- Chair & Varshney 與少數位元帳數字僅到 snippet 層級，引用前需取原文核對。

## 5. 空位分析

三個 agent 各自獨立搜尋，皆未找到以下組合：在 **learned decentralized GNN** 中，以**單次
前饋**（非 ADMM、非多輪 gossip）融合各節點由**純 local CSI** 算出的 **unit-modulus 可行提案**，
權重由節點自己吐出，聚合規則為 $\Pi\!\left(\sum_l w_{l,r}\mathbf p_{l,r,n}\right)$。

此外，**AP→CPU RIS 鏈路的 payload-vs-sum-rate 曲線在文獻中不存在**。Fernandes &
Psaromiligkos 有曲線但在 AP→RIS controller 鏈路；Hojatian 只有兩個離散工作點；Air-MPNN 只
給漸近 scaling。這條曲線是目前最清楚的可佔位置。

## 6. 五個會被打掉的說法

1. **「payload 減半」** — $4N\to2N$ 是代數等價的免費操作，只能寫成 remark。
2. **「parameter-free aggregator 是新的」** — Air-MPNN 與 AirGNN 系列已發表同一設計原則；
   而且 sum／mean readout 本來就是 GNN 預設，baseline 的 learned $\mathbf W_{\rm reduce}$
   才是不尋常的選擇，移除它讀起來是簡化而非發明。
3. **「純量加權是貢獻」** — TarMAC 的 $\alpha_{ji}$ 結構完全相同且動機相同；Chair–Varshney
   在 1986 年已是同一個物件。
4. **「equal consensus 掉 20% 是發現」** — 等權 circular mean 就是 equal-gain combining，
   magnitude 加權就是 maximum-ratio combining。cell-free 幾何下各 AP–RIS 路徑增益差數十 dB，
   掉 20% 是**預期**結果。更具體的風險：[model.py:154](../code/decentralized_ris/model.py#L154)
   的 readout 第二分支輸入就是通道能量 $e(l,r,k)=\operatorname{tr}(\mathbf H^T\mathbf H^*)$，
   因此 $s_{l,r}$ 很可能只是大尺度增益的學習代理。**未排除此解釋前，不得主張這是新發現。**
5. **「可 scale 到任意 AP 數」** — R0 的 $\mathbf W_{\rm reduce}$ 作用在 $\sum_l$ 上，本來就
   permutation-invariant；而 [model.py:255-256](../code/decentralized_ris/model.py#L255-L256)
   顯示 readout 是 per-AP 的 `ModuleList`，兩種介面都不是 AP-count-agnostic。這不是差異點。

## 7. `r1_ap_ris_mag` 與 R0c 的正確關係

### 7.1 現有 rate 沒有 superiority；payload 優勢取決於 encoding

| | R0c | `r1_ap_ris_mag` |
|---|---:|---:|
| repo Cartesian tensor／pair | $2N=60$ | $2N+1=61$ |
| angle-aware digital scalars／pair | $2N=60$ | $N+1=31$（尚未實作） |
| 聚合端可訓練參數 | 0 | 0 |
| decentralized rate @500k | 21.38124 | 21.42311（$+0.0419\pm0.0073$，+0.2%）|
| centralized @500k | 23.58559 | 23.58262 |
| 10k zero-training ladder | 11.29048 | 11.36983（$+0.0794\pm0.0427$，不顯著）|
| 從頭訓練 | 可（$\equiv$ R0） | 失敗（§7.4，$+1.384\pm0.136$）|

screening 報告 §7.6 已自我設限為「未觀察到具實質意義的退化」，不是優越性。依現行 repo 的
Cartesian tensor boundary，`r1_ap_ris_mag` 相對 R0c payload 略大；若以 angle serialization
落實 digital wire interface，則可由 60 降為 31 個 fp32 scalars。後者尚未實作成 AP→CPU codec，
所以現階段只可視為結構上的可壓縮性，不能把 +0.2% rate 差直接標成 31-scalar 或 68-bit 的
實測結果。從頭訓練的可訓練性則仍明確較差。

#### 3,200 samples 與 400 個 batch clusters

§7.6 的 `samples=3200` 搭配 `batch_size=8`，因此 evaluation loop 共執行
$3200/8=400$ 次。第 $g$ 次 loop 先抽一組 user locations，再在相同 locations 與固定 AP／RIS
geometry 下產生八個 small-scale channel realizations；同一 batch 內的八個 samples 因而共享
user geometry 與 large-scale path-loss structure，不能在 uncertainty calculation 中直接視為
3,200 個完全獨立 observations。本文把每個 evaluation batch 當成一個 cluster：

$$
\overline R_{h,g}
=
\frac{1}{8}\sum_{i=1}^{8}R_{h,g,i},
\qquad g=1,\ldots,400,
$$

其中 $h$ 表示被評估的方法。現行 `evaluate.py` 的 `simulator.loss()` 本來就回傳 batch mean，
所以每個方法存入 paired `.npz` 的 `decentralized` array 是
$[\overline R_{h,1},\ldots,\overline R_{h,400}]$，長度為 400。因所有 clusters 大小都等於八，
$400$ 個 batch means 的平均在數值上仍等於全部 3,200 個 channel realizations 的平均；差別只在
standard error 以 400 個 cluster-level observations 計算，而不是錯把 batch 內的共享 geometry
當成額外的獨立證據。

四個介面都以 `eval_seed=20260918` 重播同一序列的 user locations、channels 與 association
masks；同一介面內的 centralized 與 decentralized passes 也藉由
`regenerate_channels=False` 使用相同 channels。因此比較 `r1_ap_ris_mag` 與 R0 時，先對每個
cluster 形成 paired difference

$$
D_g
=
\overline R_{{\rm ap\text{-}ris\text{-}mag},g}
-\overline R_{{\rm R0},g},
$$

再計算

$$
\overline D
=
\frac{1}{400}\sum_{g=1}^{400}D_g,
\qquad
{\rm SE}_{\rm paired}
=
\frac{{\rm sd}(D_1,\ldots,D_{400})}{\sqrt{400}}.
$$

因此表中的 $+0.0419\pm0.0073$ 表示平均 paired difference 為 $+0.0419$ bps/Hz，後面的
$0.0073$ 是 paired standard error，不是 95% confidence interval；normal approximation 對應的
95% interval 約為 $[+0.0275,+0.0562]$。這個 uncertainty 只描述固定 500k R0 checkpoint 與固定
AP／RIS geometry 下的 evaluation Monte Carlo variability，不包含 training seed、checkpoint 或
topology 的不確定性。

### 7.2 兩者不是兩個方法，是同一條 bit-allocation 軸上的兩點

R0c 每個 element 傳的是二維向量 $\mathbf z_{l,r,n}=m_{l,r,n}\mathbf p_{l,r,n}$，方向與長度
都帶資訊。「把 R0c 壓成 2-bit」至少要區分兩種 codec：

- **受限的 phase-only codec**：每 element 總共 2 bit，只使用 4 個等間隔方向，magnitude
  一個 bit 不剩，因而退化為 R1-Shared 的 2-bit 版。現有 $-4.409\pm0.115$ 是 fp32 Shared
  相對 R0c 的差，不是這個 consensus 前 2-bit codec 的實測結果。
- **2-bit 方向 + 每 element $b_m$ bit magnitude**：成本 $N(2+b_m)$，才是真正的 R0c。

一般的 4-codeword vector quantizer 也可讓 codewords 同時具有不同方向與半徑，因此「總共
2 bit 必然只能表示方向」並非無條件成立。§8.1 的公平比較應同時包含 phase-only codec 與較強的
R0c vector／polar codec。

因此正確的問題是：**在固定總位元預算下，magnitude 這個資訊該用什麼粒度買。**
以 $A=5$、$R=4$、$N=30$（20 個 AP–RIS pair）計：

| 每 pair 的配置 | bits/pair | 全系統 bits | 證據狀態 |
|---|---:|---:|---|
| 只有 2-bit 方向（= Shared） | 60 | 1,200 | **未測 AP proposal 量化**；fp32 Shared 掉 $4.409\pm0.115$ |
| + 每 pair 一個 8-bit 純量（= `r1_ap_ris_mag`） | 68 | 1,360 | **未測**；fp32 介面轉換對 R0c $+0.0419\pm0.0073$ |
| + 每 element 1-bit 殘差 | 98 | 1,960 | **未測** |
| + 每 element 4-bit magnitude（R0c polar codec） | 180 | 3,600 | **未測** |
| R0c，fp32 | 1,920 | 38,400 | R0c 本身 |
| R0，fp32 | 3,840 | 76,800 | $\equiv$ R0c |

### 7.3 由此得到的正確主張

[§6.3](./ris_action_screening_report.md) 的 ladder 顯示 per-pair 平均復現 R0c 的 106.6%
（差 $+0.0794\pm0.0427$，不顯著）；§7.6 在 500k 上 per-pair 甚至高出 $+0.0419\pm0.0073$。
在目前固定 R0 checkpoint、固定 topology 與 fp32 proposals 下，結果表示 **per-element
magnitude 相對於 pair 平均的殘差未呈現正的 task value**。這支持把殘差設為 zero-bit 作為
bit-allocation sweep 的一個端點，但尚不足以證明任意 operating point 的 rate-optimal R0c
量化器都不應在該殘差上配置位元。

因此正確的說法是：**`r1_ap_ris_mag` 不是 R0c 的競爭方法，而是 R0c radial field 的
zero-residual 壓縮端點**，並且有機制解釋（magnitude 的變異 68–75% 在 pair 層級，angular
agreement 的變異 74–88% 在 element 層級，見 §6.3）。這個說法比「payload 比較小」強，因為
它是可否證的 rate-allocation 陳述。

更精確地說，對固定 AP--RIS pair，可將 R0c magnitude field 分解為

$$
m_{l,r,n}=s_{l,r}+\varepsilon_{l,r,n},
\qquad
s_{l,r}=\frac{1}{N}\sum_{n=1}^{N}m_{l,r,n},
\qquad
\sum_{n=1}^{N}\varepsilon_{l,r,n}=0.
$$

`r1_ap_ris_mag` 傳送 $s_{l,r}$ 並把所有 $\varepsilon_{l,r,n}$ 設為零。這個「拆掉再合起來」不是
為了改變原本的 $\mathbf z_{l,r,n}=m_{l,r,n}\mathbf p_{l,r,n}$，而是為了把 direction field 與
radial field 分開，辨識 radial field 中哪些成分值得占用 fronthaul bits。使用 arithmetic mean
也不是任意 heuristic；它是 pair 內 constant radial approximation 的 least-squares 最佳解：

$$
s_{l,r}^{\star}
=
\underset{s\in\mathbb R}{\arg\min}
\sum_{n=1}^{N}(m_{l,r,n}-s)^2
=
\frac{1}{N}\sum_{n=1}^{N}m_{l,r,n}.
$$

還可給出一個只控制 RIS phase 擾動、**不直接保證 sum rate** 的 preservation bound。令

$$
\mathbf u_{r,n}=\sum_{l=1}^{A}m_{l,r,n}\mathbf p_{l,r,n},
\qquad
\mathbf v_{r,n}=\sum_{l=1}^{A}s_{l,r}\mathbf p_{l,r,n},
\qquad
\delta_{r,n}=\sum_{l=1}^{A}|m_{l,r,n}-s_{l,r}|.
$$

若 $\mathbf u_{r,n}$ 與 $\mathbf v_{r,n}$ 皆非零，則由 triangle inequality 與 normalization map
的標準界可得

$$
\left\|\Pi(\mathbf u_{r,n})-\Pi(\mathbf v_{r,n})\right\|_2
\le
\frac{2\|\mathbf u_{r,n}-\mathbf v_{r,n}\|_2}{\|\mathbf u_{r,n}\|_2}
\le
\frac{2\delta_{r,n}}{\|\mathbf u_{r,n}\|_2}.
$$

$\delta_{r,n}<\|\mathbf u_{r,n}\|_2$ 是保證 $\mathbf v_{r,n}\neq0$ 的一個充分條件。此界把方法的
成功條件說清楚：被丟棄的 radial residual 要小於 R0c consensus resultant；若 AP votes 嚴重互相
抵消、使 $\|\mathbf u_{r,n}\|_2$ 接近零，界會變鬆，phase 也可能很敏感。因此這是可量測的
approximation mechanism，不是「所有情況都無損」的定理。

最合適的實證措辭是：**在約一半的 angle-aware intrinsic payload 下，compressed interface 對
固定成熟 R0 policy 達到 non-inferior sum rate；觀察到的微小正差異不作 superiority claim。**
在 consensus 前 codec 尚未實測以前，「約一半 payload」只能稱為結構上的名目計數，不能與
$+0.0419$ bps/Hz 拼成一個已完成的 rate--distortion 結果。

## 8. 決定性實驗

兩項皆為純評估，使用現成 500k R0 checkpoint，不需重訓，符合單一 seed 的階段規則。

### 8.1 Bit-allocation sweep（主實驗）

必須把 encoder／quantizer／decoder 放在 consensus **之前**，真正改變 CPU 收到的 AP message。
至少比較以下四條 codec family：

- `r1_ap_ris_mag`：$Nb_p+b_s$ bits／pair；
- R0c polar：$N(b_p+b_m)$ bits／pair；
- R0c progressive：$Nb_p+b_s+Nb_\varepsilon$ bits／pair，其中 residual
  $b_\varepsilon\in\{0,1,2,4\}$，$b_\varepsilon=0$ 即 magnitude endpoint；
- R0c Cartesian／task-aware vector quantization：在相同 total bits 下允許 codewords 同時改變
  方向與半徑，作為不能靠方便限制打敗的強 baseline。

方向解析度至少掃 $b_p\in\{1,2,3,\infty\}$。所有 codec 必須使用相同 calibration samples、
clipping 規則、codebook 訓練資料與 header／scale accounting；若某 codec 需要傳量化範圍或
codebook identifier，也必須計入。橫軸同時報 bits/pair 與全系統 $AR$ pairs 的 AP→CPU bits，
縱軸報 decentralized sum rate。

必須畫完整 rate--distortion frontier 並**掃到效能明確崩潰為止**，不能只報 68-bit 一個工作點。
尤其要含 per-element 1-bit residual：以目前 2-bit direction、8-bit pair scalar 的例子，R0c
progressive 是 98 bits，而 magnitude 是 68 bits，只差 1.4×；若前者即可恢復效能，接近
$2\times$ 的壓縮故事便不成立。

同時此 sweep 會重驗 §6.3 的 granularity 結論——該結論是在 **fp32 方向**下量的，方向一旦量化
到 2 bit，magnitude 加權的相對重要性可能改變，這個交互作用無人量過。

### 8.2 MRC 解釋的排除（防守實驗）

量 $s_{l,r}$ 與 AP–RIS 通道能量 $e(l,r,\cdot)$ 的相關；再以大尺度增益的閉式函數取代學到的
$s_{l,r}$ 重跑評估。

- 若兩者等價：貢獻退化為已知結果，應改主張「不需 global CSI 即可計算的 MRC 權重廉價代理」。
- 若不等價：這是一個可寫的實質發現，且直接反駁 §6 第 4 點的攻擊。

## 9. Go／no-go 判準

先寫下判準，以免用單一漂亮工作點或沉沒成本改變研究問題。

**Go：** 若 §8.1 在一段 operating range 而非單一點上，能相對最強的同位元 R0c codec 滿足
下列至少一項，這條線值得繼續並可成為 conference 主軸：

1. 在 non-inferior sum rate 下，AP→CPU bits 穩定接近減半；或
2. 在相同總 bits 下，得到具統計與實質意義的 sum-rate 改善。

**No-go 或降級為 observation：** 若出現下列任一情況，不應再用「新 aggregation」包裝：

1. magnitude 只勝過受限的 phase-only R0c quantizer，一加入 Cartesian／task-aware VQ 優勢便
   消失；
2. 只有 fp32 的 $+0.2\%$ rate 差，沒有實際 consensus 前的 bit 優勢；
3. per-element magnitude residual 只需 1 bit 即可恢復 R0c，使 bit 優勢只剩約 1.4×；
4. 結果只在單一 checkpoint／topology 成立，此時只能稱為 promising observation；
5. §8.2 顯示 $s_{l,r}$ 可直接由大尺度增益取代，此時 learned-mechanism 主張消失，只剩低成本
   MRC-style weighting 的工程結果。

若目標是 journal 主軸，通過上述 gate 後還需要跨 $N$、$A$、$R$ 與 topology 的驗證，以及
fronthaul quantization／packet error robustness，或更緊的 approximation-to-rate analysis。

## 10. 建議定位

最合理的主題是 **post-training structured compression of the AP→CPU RIS action interface**，
而不是「一個新的 weighted consensus」。完整論證鏈應寫成：

1. 先將 R0 代數等價地交換 reduction 與 summation，得到 R0c 這個完整但更容易編碼的 reference
   interface；
2. 將 R0c polar message 拆成 angle field 與 magnitude field；
3. 以 $m_{l,r,n}=s_{l,r}+\varepsilon_{l,r,n}$ 分解 radial field；
4. 只傳 pair scale $s_{l,r}$，丟棄 element residual $\varepsilon_{l,r,n}$；
5. 證明 arithmetic mean 是 constant radial approximation 的 least-squares 解，並以 §7.3 的
   phase-preservation bound 說明何時近似合理；
6. 以公平的 consensus-before-quantization 實驗證明 residual task value 很低，形成真正的
   bits--sum-rate frontier。

通過 §9 的 gate 後，論文貢獻可整理為三點：R0c 的 exact commuted reference、R0c radial field
的 structured zero-residual compression，以及 AP→CPU RIS control link 的 rate--distortion
frontier。聚合端零參數與 CPU 不需模型更新是實作優點，但不應單獨宣稱 novelty。

誠實引用 Hojatian、Air-MPNN、TarMAC、Chair–Varshney、Katsanos、Fernandes，並把主張寫成
*post-training structured compression ＋ unit-modulus manifold 上的幾何聚合 ＋ 公平量化曲線*。
另需預先準備兩個回應：(a) EGC/MRC 的一行反駁（靠 §8.2）；(b) 為何用數位 weighted circular
consensus 而不是直接在 fronthaul 等效通道上做 AirComp。最需要防守的審稿攻擊不是「它和 R0c
很像」，而是「bit accounting 不公平，而且 68-bit 結果根本還沒有跑」。

## 11. Claim boundary

- repo 內所有相關數字皆為 seed 0 訓練、固定 topology、$A=5$／$R=4$／$N=30$／$M=2$；
  §7.6 的介面轉換為 3,200 samples／400 batch clusters，未跨 training seed 或 topology。
- §7.4 已證明零訓練 counterfactual 不能外推到重訓後，因此 §7.3 的「zero-residual 壓縮端點」主張目前
  只對**固定 R0 checkpoint 的推論端轉換**成立。
- $+0.0419$ bps/Hz 是在同一 evaluation distribution 上比較多個 post-hoc aggregation 規則後的
  supporting observation；不作 superiority claim，也不排除 model-selection bias。
- §7.3 的 preservation bound 控制的是 normalized RIS phase proposal 的距離，不直接推出
  end-to-end sum-rate gap；resultant 接近零時該界可能沒有資訊。
- 本文件的位元帳為理論計數，未計入通道編碼、同步與控制平面封包開銷。
- angle-aware fp32 與 68-bit rows 目前都只是 codec design／待測 operating point，不能和現有
  fp32 rate 數字綁成同一個 empirical claim。
- 文獻查核的覆蓋邊界見 §4.3；IEEE Xplore 全文未掃，不能宣稱窮盡。

## 12. 下一步

1. 實作 §8.1 的 encoder／quantizer／decoder（AP message 解碼後再 consensus），先跑 smoke 驗
   unit-modulus 與 R0c 等價 control 未破，再以 500k checkpoint 跑 3,200-sample 評估。
2. 執行 §8.2。
3. 依 §9 的 go／no-go gate 決定是否繼續；通過後才擴展 topology／robustness，並安排 IEEE
   Xplore 全文查核與 related work 撰寫。
