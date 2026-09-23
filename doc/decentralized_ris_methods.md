# Decentralized RIS 方法與通訊介面

## Material Passport

- Origin Date: 2026-09-15
- Last Updated: 2026-09-21（統一以 G2 為 proposed method，G1 僅作 no-context ablation）
- Verification Status: IMPLEMENTED（定義已與 `model.py`、`variants.py`、`model_based.py` 及 action／graph-energy／model-based tests 對照）
- Version Label: `decentralized_ris_methods_v8`
- Scope: proposed G2、G1 no-context ablation、R0／R0c／R1／G0 controls、local-energy consensus、E12 model-based benchmark pair，以及 2-bit／full-CSI greedy evaluation
- Entry: [Decentralized RIS 研究文件入口](./README.md)
- Reference paper: [Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks](<./Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks.pdf>)

本文件是上述方法定義的單一來源。實驗結果、統計判讀與研究決策分別留在
[實驗總報告](./decentralized_ris_experiments.md)與其連結的詳細報告，不在此重複。

快速對照：

| 名稱 | 研究角色 | RIS representation | AP→CPU message | CPU aggregation |
|---|---|---|---|---|
| R0 | 原論文 baseline | RIS message-passing node | $4N$ learned features | learned reduction，再 projection |
| R0c | algebraic control | 同 R0 | $2N$ raw logits | sum，再 projection；與 R0 代數等價 |
| R1-Shared | interface control | 同 R0 | $N$ phase angles | equal circular consensus |
| R1 AP–RIS magnitude | interface control | 同 R0 | $N$ phase angles + 1 learned scale | learned-scale circular consensus |
| G0 | representation control | 同 R0 | $N$ phase angles + 1 local-energy scalar | energy-weighted circular consensus |
| G1 | G2 no-context ablation | node-free per-RIS link embeddings | $N$ phase angles + 1 local-energy scalar | 同 G0 |
| **G2** | **proposed method／研究主線** | node-free link embeddings + per-RIS context | $N$ phase angles + 1 local-energy scalar | 同 G0 |

<a id="notation"></a>

## 1. 論文系統記號與 CSI input

本文件遵循論文記號：$L$、$R$、$K$ 與 $N$ 分別表示 AP、RIS、UE 與每個 RIS 的 element 數。
現行程式另把 RIS 數命名為參數 `L`，因此對照程式時必須使用下表，不能把兩個 $L$ 混用；
既有結果報告若以 $A$ 表示 AP 數，則 $A\equiv L$。

| 物理量 | 論文 | 現行程式 |
|---|---:|---:|
| AP 數 | $L$ | num_of_AP／AP |
| RIS 數 | $R$ | L |
| UE 數 | $K$ | K／users_per_ap |
| 每 RIS elements | $N$ | N |
| GNN latent width | $q$ | ch |
| Message-passing depth | $D$ | D |

論文先把 AP $l$、RIS $r$、UE $k$ 的複數 CSI 寫成 real-valued vector（式 (9)）：

$$
\widetilde{\mathbf h}_{(l,r,k)}
\triangleq
\begin{bmatrix}
\operatorname{Re}\{\operatorname{vec}(\widetilde{\mathbf H}_{(l,r,k)})\}\\
\operatorname{Im}\{\operatorname{vec}(\widetilde{\mathbf H}_{(l,r,k)})\}
\end{bmatrix}
\in\mathbb R^{2M(N+1)},
$$

其中

$$
\widetilde{\mathbf H}_{(l,r,k)}
=
\begin{bmatrix}
\mathbf H_{(l,r,k)}^{T} & \mathbf h_{D,(l,k)}^{T}
\end{bmatrix}
\in\mathbb C^{M\times(N+1)}.
$$

Paper-decentralized inference 在 AP $l$ 可用的 CSI 集合為論文式 (10)：

$$
\mathcal H_{(l)}
=
\left\{
\widetilde{\mathbf h}_{(l,r,k)}
\mid k\in\mathcal K_l,\ r\in\mathcal R
\right\}
\cup
\left\{
\widetilde{\mathbf h}_{(l',r,k)}
\mid k\in\mathcal K_l,\ l'\in\mathcal L_k,\ r\in\mathcal R
\right\}.
$$

本文使用的三種 input mode 因而可精確區分為：

| Input mode | 模型推論時可見資訊 |
|---|---|
| centralized | 單一 GNN 可見所有相關 global CSI |
| paper-decentralized | AP $l$ 的 GNN copy 只看 $\mathcal H_{(l)}$ |
| own-only | 只保留式 (10) 第一個集合，是額外設計的 ablation，不是論文原方法 |

論文的最佳化變數為 active beamforming
$\mathbf F=[\mathbf F_1,\ldots,\mathbf F_L]$ 與 RIS phase
$\boldsymbol\theta=[\boldsymbol\theta_1^T,\ldots,\boldsymbol\theta_R^T]^T$，其中
$\boldsymbol\theta_r\in\mathbb C^N$。現行程式以 real／imaginary Cartesian pair 儲存每個複數
phase：

$$
\boldsymbol\vartheta_{r,n}
\triangleq
\begin{bmatrix}
\operatorname{Re}\{\theta_{r,n}\}\\
\operatorname{Im}\{\theta_{r,n}\}
\end{bmatrix}
=
\begin{bmatrix}
\cos\phi_{r,n}\\
\sin\phi_{r,n}
\end{bmatrix}
\in\mathbb R^2.
$$

整個 RIS 的 stacked real／imaginary vector 記為

$$
\overline{\boldsymbol\theta}_r
\triangleq
\begin{bmatrix}
\operatorname{Re}\{\boldsymbol\theta_r\}\\
\operatorname{Im}\{\boldsymbol\theta_r\}
\end{bmatrix}
\in\mathbb R^{2N}.
$$

模型內部仍以 $2N$ 個 Cartesian real values 表示 $N$ 個 phase。AP→CPU 的 wire format
則只傳 $N$ 個角度；CPU 以 $(\cos\phi,\sin\phi)$ 重建 unit proposals。2-bit phase 若進一步
放到 consensus 前，則可改傳 $N$ 個 2-bit indices，但那是另一個會改變演算法的介面。

對任意非零 $\mathbf x\in\mathbb R^2$，定義逐 element projection

$$
\Pi(\mathbf x)=\frac{\mathbf x}{\lVert\mathbf x\rVert_2}.
$$

它就是論文 $f_{\rm norm}$ 對每個 real／imaginary pair 執行的 unit-modulus normalization。
R1-Shared／AP–RIS magnitude weighting 對接近零的 pair 回退到 phase $0$，即 $(1,0)^T$；baseline R0／R0c 直接使用
`F.normalize`，所以精確零向量仍會保持為零。這是程式的退化輸入邊界，不是論文式 (8c) 的設計值。

## 2. 論文的 RIS readout 與 CPU aggregation

真正的 GNN latent 是第 $D$ 層 RIS node state
$\mathbf s_r^{(D)}\in\mathbb R^{q(D+1)}$。論文式 (26) 在 AP $l$ 定義 RIS readout；論文沒有
另外命名每個 AP output，以下的 $\mathbf v_{l,r}$ 是本文為簡化後續公式而引入的 shorthand：

$$
f_{\mathrm{RIS},l}^{\mathrm{out}}:
\mathbb R^{q(D+1)}\times\mathbb R^{RK}
\longrightarrow\mathbb R^{4N},
$$

$$
\mathbf v_{l,r}
\triangleq
f_{\mathrm{RIS},l}^{\mathrm{out}}
\left(\mathbf s_r^{(D)},\mathbf e_l\right)
=
\begin{bmatrix}
\mathbf W_{\mathrm{RIS},1}^{\mathrm{out}}\mathbf s_r^{(D)}\\
\mathbf W_{\mathrm{RIS},2}^{\mathrm{out}}\mathbf e_l
\end{bmatrix}
\in\mathbb R^{4N}.
$$

其中 $\mathbf e_l\in\mathbb R^{RK}$ 收集
$\{e(l,r,k)\mid r\in\mathcal R,k\in\mathcal K\}$，
$e(l,r,k)=\operatorname{tr}(\mathbf H_{(l,r,k)}^T\mathbf H_{(l,r,k)}^*)$，且

$$
\mathbf W_{\mathrm{RIS},1}^{\mathrm{out}}
\in\mathbb R^{2N\times q(D+1)},\qquad
\mathbf W_{\mathrm{RIS},2}^{\mathrm{out}}
\in\mathbb R^{2N\times RK}.
$$

所以 $\mathbf s_r^{(D)}$ 才是 GNN latent；$\mathbf v_{l,r}$ 是 readout 後、準備傳給 CPU 的
RIS phase feature。論文中的 $q$ 是 latent width，不是這個 $4N$ 向量的名稱，因此不再用
$q_{l,r}$ 指稱 AP output。預設 $q=64$、$D=6$、$R=4$、$K=8$、$N=30$ 時，
$\mathbf s_r^{(D)}$、$\mathbf e_l$ 與 $\mathbf v_{l,r}$ 的維度依序為 $448$、$32$ 與 $120$。
論文在 decentralized 描述中省略 $\mathbf s_r^{(D)}$ 的 AP index；
因為每個 AP 使用不同的 $\mathcal H_{(l)}$ 跑自己的 GNN copy，必要時本文寫成
$\mathbf s_{l,r}^{(D)}$ 以表示 AP-local state。

論文將所有 AP 對 RIS $r$ 的 phase features 收成

$$
\mathbf R_r
=
\begin{bmatrix}
\mathbf v_{1,r} & \cdots & \mathbf v_{L,r}
\end{bmatrix}
\in\mathbb R^{4N\times L}.
$$

接著依式 (27) 在 CPU 聚合：

$$
f_{\mathrm{RIS}}^{\mathrm{aggre}}:
\mathbf R_r
\longmapsto
f_{\rm norm}
\left(
\mathbf W_{\rm reduce}\mathbf R_r\mathbf 1_L
\right),
\qquad
\mathbf W_{\rm reduce}\in\mathbb R^{2N\times4N},
$$

其中 $\mathbf 1_L$ 是長度 $L$ 的全一向量。式 (28) 再把輸出解讀為

$$
\overline{\boldsymbol\theta}_r
=
f_{\mathrm{RIS}}^{\mathrm{aggre}}(\mathbf R_r).
$$

論文式 (26)–(27) 只顯示矩陣乘法；現行程式的 RisReadoutAp 與 RisMerge 則使用含 bias 的
PyTorch linear layers。因此程式中的兩個 readout branches 是 affine maps，CPU reduction
實際為

$$
\mathbf y_r
=
\mathbf W_{\rm reduce}\sum_{l=1}^{L}\mathbf v_{l,r}
+\mathbf b.
$$

下文的 R0c 會把這個程式中存在的 $\mathbf b$ 平分為 $\mathbf b/L$，以維持與 R0 的精確代數
等價。論文把 readout 記為 $f_{\mathrm{RIS},l}^{\mathrm{out}}$，而矩陣省略 AP index；現行程式則以
ModuleList 為每個 AP 建立一個 RisReadoutAp。

<a id="action-interfaces"></a>

## 3. 四種 action interface

四種介面採用相同的 GNN state 定義、論文式 (26) readout architecture 與 learned
$\mathbf W_{\rm reduce}$；零訓練 control 才進一步共用同一組 checkpoint weights。它們的差別是
AP 傳送的表示和 projection 相對於 aggregation 的位置：

| 方法 | AP-side 流程 | 每個 AP–RIS 傳送 | 聚合端流程 |
|---|---|---:|---|
| R0 | $\mathbf s_{l,r}^{(D)},\mathbf e_l\to\mathbf v_{l,r}$ | $4N$ phase feature | 論文式 (27) 的 learned reduction 與 normalization |
| R0c | $\mathbf v_{l,r}\to\mathbf z_{l,r}$ | $2N$ raw logits | 先加總 $\mathbf z$，再 normalization |
| R1-Shared | $\mathbf v_{l,r}\to\mathbf z_{l,r}\to\mathbf p_{l,r}$ | $N$ phase angles | CPU 重建 unit proposals，再做等權 circular consensus |
| R1 AP–RIS magnitude weighting | $\mathbf v_{l,r}\to\mathbf z_{l,r}\to(\mathbf p_{l,r},s_{l,r})$ | $N+1$ | CPU 重建 unit proposals，再做 importance-weighted circular consensus |

<a id="r0"></a>

### 3.1 R0：論文式 (26)–(28)

R0 是論文方法的程式實作。包含程式 bias 後，其 stacked real／imaginary phase 為

$$
\overline{\boldsymbol\theta}_r^{\rm R0}
=
f_{\rm norm}
\left(
\mathbf W_{\rm reduce}\sum_{l=1}^{L}\mathbf v_{l,r}
+\mathbf b
\right).
$$

所以聚合端仍包含 learned phase component，每個 AP–RIS pair 傳送 $4N$ 個 phase-feature
real values。

### 3.2 R0c：本地 reduction、聚合後 projection

R0c 將 CPU 的 affine reduction 分配到各 AP：

$$
\mathbf z_{l,r}
=
\mathbf W_{\rm reduce}\mathbf v_{l,r}
+\frac{\mathbf b}{L}
\in\mathbb R^{2N},
\qquad
\overline{\boldsymbol\theta}_r^{\rm R0c}
=
f_{\rm norm}
\left(
\sum_{l=1}^{L}\mathbf z_{l,r}
\right).
$$

因為

$$
\sum_{l=1}^{L}\mathbf z_{l,r}
=
\mathbf W_{\rm reduce}\sum_{l=1}^{L}\mathbf v_{l,r}
+\mathbf b,
$$

R0c 與 R0 代數等價，只可能因浮點加總順序產生微小差異。R0 checkpoint 可直接由 R0c
載入，不需重訓。R0c 已把 learned reduction 搬到 AP，但 $\mathbf z_{l,r}$ 仍不是
unit-modulus phase。

把 $\mathbf z_{l,r}$ 寫成前 $N$ 維 real、後 $N$ 維 imaginary 後，element $n$ 的二維 logit 是

$$
\mathbf z_{l,r,n}
=
\begin{bmatrix}
z_{l,r,n}^{\rm Re}\\
z_{l,r,n}^{\rm Im}
\end{bmatrix}
\in\mathbb R^2.
$$

它的方向表示 AP $l$ 建議的 phase，長度則在 R0c 中控制該 AP 對 element $n$ 的聚合影響。

### 3.3 R1-Shared：本地 projection、等權聚合

R1-Shared 將 normalization 提前到每個 AP：

$$
\mathbf p_{l,r,n}
=
\Pi(\mathbf z_{l,r,n}),
\qquad
\boldsymbol\vartheta_{r,n}^{\rm shared}
=
\Pi\left(
\sum_{l=1}^{L}\mathbf p_{l,r,n}
\right).
$$

AP 傳送的是 unit-modulus proposal，但 local projection 使所有非零 proposals 的長度都變成一，
所以 $\lVert\mathbf z_{l,r,n}\rVert_2$ 不再參與 aggregation。最終 RIS phase 仍由所有 AP
proposals 的 parameter-free circular consensus 決定；單一 AP proposal 不是可同時執行的全域
RIS configuration。

<a id="r1-ap-ris-mag"></a>

### 3.4 R1 AP–RIS magnitude weighting：本地 projection、AP–RIS-level magnitude 加權

AP–RIS magnitude weighting 使用相同的 unit proposals，但每個 AP–RIS pair 再傳一個 scalar：

$$
s_{l,r}
=
\frac{1}{N}
\sum_{n=1}^{N}
\lVert\mathbf z_{l,r,n}\rVert_2.
$$

同一個 $s_{l,r}$ 會乘在 AP $l$ 對 RIS $r$ 的全部 $N$ 個 proposals 上：

$$
\boldsymbol\vartheta_{r,n}^{\rm ap\text{-}ris\text{-}mag}
=
\Pi\left(
\sum_{l=1}^{L}s_{l,r}\mathbf p_{l,r,n}
\right).
$$

AP–RIS magnitude weighting 因而不是「完全不傳 magnitude」：它丟棄 $N$ 個 element-specific magnitudes，但傳送
其平均值。Scalar 乘的是二維 unit vector，不是直接乘 phase angle；若所有 $s_{l,r}$ 相等，
該方法就退化為 Shared。此 scalar 沒有被定義成校準後的可靠度機率，因此稱為
importance weight，不稱 confidence。

## 4. 三種聚合規則的直接關係

令

$$
m_{l,r,n}
=
\lVert\mathbf z_{l,r,n}\rVert_2,
\qquad
\mathbf z_{l,r,n}
=
m_{l,r,n}\mathbf p_{l,r,n}.
$$

則 R0c 也可寫成 element-specific magnitude weighting：

$$
\boldsymbol\vartheta_{r,n}^{\rm R0c}
=
\Pi\left(
\sum_{l=1}^{L}m_{l,r,n}\mathbf p_{l,r,n}
\right).
$$

因此三者真正的差別是 AP vote 的權重粒度：

| 方法 | AP $l$ 對 element $n$ 的權重 |
|---|---|
| R1-Shared | $1$ |
| R1 AP–RIS magnitude weighting | $s_{l,r}=\operatorname{mean}_n m_{l,r,n}$ |
| R0c | $m_{l,r,n}$ |

AP–RIS magnitude weighting 可視為 Shared 與 R0c 之間的壓縮介面：只增加每個 AP–RIS pair 一個 scalar，近似
R0c 原本保留的 $N$ 個 magnitudes。

<a id="payload"></a>

## 5. Payload 與 decentralization 用語

以傳輸的 fp32 scalar 數計數；proposal interfaces 使用 phase-angle codec：

| 方法 | 每個 AP–RIS pair | 全部 AP→aggregator payload | 聚合端 trainable phase component |
|---|---:|---:|---:|
| R0 | $4N$ | $4RNL$ | 有 |
| R0c | $2N$ | $2RNL$ | 無 |
| R1-Shared | $N$ | $RNL$ | 無 |
| R1 AP–RIS magnitude weighting | $N+1$ | $RL(N+1)$ | 無 |

在目前 $N=30$ 的設定中，AP–RIS magnitude／energy weighting 是每個 AP–RIS pair 傳 31 個
real values：30 個 phase angles 加一個 scalar。聚合端對 weights 的共同尺度不敏感，實作會先
在 AP 維度正規化，再計算加權 resultant 與最終 unit projection。

R0 的 $4N$ 計數其實是保守的**上界**。由 `model.RisReadoutAp.forward` 可見，$4N$ 特徵由
$[\,f(\mathbf r_l)_{r}\;;\;f_{e}(\mathbf e_l)\,]$ 兩段串接而成，後 $2N$ 維只依賴 AP $l$ 自己的
edge 向量，對 $r$ 完全相同（E07 以 $\max_{r,r'}$ 差值為 $0$ 的 gate 驗證）。因此該段每個 AP 只需
傳一次，攤提後的 payload 是

$$
\frac{2N R + 2N}{R}
=
2N\left(1+\frac{1}{R}\right)
$$

個 real values per AP–RIS pair，在 $N=30$、$R=4$ 時為 75 而非 120。與 R0 做 payload 比較時應採用
這個較寬鬆的計數，否則會高估 anchor 的成本；E07 的所有判準都使用它。

只要最終 consensus 仍由 CPU 或中央 controller 執行，本文稱
`decentralized phase inference with centralized consensus`。只有 AP／RIS controllers 不依賴 CPU、
透過鄰居交換完成協調時，才稱 `fully decentralized`。

## 6. 訓練與 checkpoint 使用方式

R0、R0c、R1-Shared 與 AP–RIS magnitude weighting 共用 RIS backbone 與 $\mathbf W_{\rm reduce}$，state-dict keys 與有效
參數量相容。它們有兩種不同用途：

1. **零訓練介面轉換**：固定同一個 R0 checkpoint，只替換 aggregation／projection 規則，用來
   測量既有 policy 對介面變更的敏感度。
2. **從頭訓練**：用指定介面讓 loss 反向傳播 through projection 與 consensus，得到新的 policy。

兩者回答不同問題，所得結果不能互相外推。

為維持 controlled ablation，R0c、R1-Shared 與 AP–RIS magnitude weighting 都保留 $L$ 個 AP terms。Inactive AP
的 $\mathbf v_{l,r}$ 為零，但仍保留平均分配的 learned bias $\mathbf b/L$。

<a id="two-bit"></a>

## 7. 2-bit phase evaluation

論文將 $Q$-bit phase alphabet 定義為

$$
\Theta_{\rm disc}^{(Q)}
=
\left\{
\exp\left(j\frac{2\pi m}{2^Q}\right)
\;\middle|\;
m=0,\ldots,2^Q-1
\right\}.
$$

當 $Q=2$ 時，相位角集合為
$\mathcal Q_2=\{0,\pi/2,\pi,3\pi/2\}$。

`rounded 2-bit` 或 `dec 2-bit` 是把模型輸出的 continuous phase 四捨五入到最近的
$\mathcal Q_2$ phase。它量測既有 policy 的 post-hoc quantization loss，不是經過 2-bit-aware
retraining 後的最佳表現。

<a id="full-csi-greedy"></a>

## 8. Full-CSI greedy 2-bit reference

Full-CSI greedy 是分析用 oracle，不是 decentralized deployment method。對每個 sample 與指定
input mode，它依序執行：

1. 固定該 input mode 的 GNN active beamforming $\mathbf F$（程式變數 `W`，已包含 power allocation）。
2. 將同一個 GNN continuous phase 量化到 2-bit grid，作為起點。
3. 一次只更新一個 RIS element，固定其餘 elements，枚舉四個候選 phase。
4. 使用 full CSI 計算四個候選的真實 sum rate，保留其中最高者。
5. 掃過全部 $RN$ 個 elements，最多重複四輪；整輪沒有改善時提前停止。

目前 $R=4$、$N=30$，共有 120 個 phase variables。四輪最多評估約
$120\times4\times4=1{,}920$ 個候選／sample，而不是窮舉 $4^{120}$ 個 joint configurations。
所得解只是依賴起點與更新順序的 coordinate-wise local optimum，不是 global optimum。

`centralized greedy`、`paper-decentralized greedy` 與 `own-only greedy` 的 phase search 都使用相同
full-CSI oracle；名稱只表示固定的 $\mathbf F$ 與 phase initialization 來自哪一種 GNN input mode。因此
greedy 後的 mode gap 主要反映 active beamforming／power-allocation quality，但仍混合 initialization 與有限輪
coordinate descent 造成的 local-optimum 差異，不能直接稱為純 precoding gap。

## 9. 實作對照

| 論文／本文物件 | 現行程式 |
|---|---|
| $\mathbf s_{l,r}^{(D)}$ | `ris_state`／`rl` |
| $\mathbf e_l$ | `edge_block`／`e_ap` |
| $\mathbf v_{l,r}=f_{\mathrm{RIS},l}^{\mathrm{out}}(\mathbf s_{l,r}^{(D)},\mathbf e_l)$ | [model.py](../code/decentralized_ris/model.py) 的 `RisReadoutAp.forward`；variant 中稱 `latent` |
| $\mathbf W_{\rm reduce},\mathbf b$ | `RisMerge.f_merge.weight`／`bias` |
| $\mathbf z_{l,r}$ | `RisMerge.local_logits` |
| $\mathbf p_{l,r,n}$、Shared／AP–RIS magnitude consensus | [variants.py](../code/decentralized_ris/variants.py) 的 `_unit_from_pairs`／`circular_consensus` |
| 介面與等價性測試 | [tests/test_action_interface.py](../code/decentralized_ris/tests/test_action_interface.py) |
| $\Theta_{\rm disc}^{(Q)}$ quantization | [rates.py](../code/decentralized_ris/rates.py) |
| Full-CSI greedy coordinate search | [experiments/discrete_cd.py](../code/decentralized_ris/experiments/discrete_cd.py) |
| $E_{l,r}$ | [variants.py](../code/decentralized_ris/variants.py) 的 `local_energy` |
| phase-angle wire codec | [variants.py](../code/decentralized_ris/variants.py) 的 `encode_phase`／`decode_phase` |
| Energy-weighted circular consensus | [variants.py](../code/decentralized_ris/variants.py) 的 `circular_consensus` 與 `VariantNet._merge` |
| G0 | `VariantNet(arch="g0")` |
| G1 no-context ablation path | `LinkEncoder`、`ApNodeUpdateLayer`、`NodeFreePhaseHead`，`arch="g1"` |
| G2 per-RIS context | `VariantNet._backbone` 中 `self.ris_ctx`，`arch="g2"` |
| Screening／long-training evaluation | [experiments/graph_energy_screening.py](../code/decentralized_ris/experiments/graph_energy_screening.py) |

<a id="energy-consensus"></a>

## 10. Parameter-free local-energy consensus

G0／G1／G2 共用同一個 consensus 規則。對 AP $l$ 與 RIS $r$，定義

$$
E_{l,r}
=
\sum_{k\in\mathcal K_l}
\operatorname{tr}\!\left(
\mathbf H_{(l,r,k)}^T\mathbf H_{(l,r,k)}^*
\right).
$$

$\mathcal K_l$ 是 AP $l$ 自己服務的 UE，因此 $E_{l,r}$ 只使用 AP $l$ 的 channel
與 association mask，不使用其他 AP 的 CSI、beamformer 或 ground-truth rate。實作上它由
`edges[:, r, l的K個欄位]` 依 served-user mask 加總，並在進入 consensus 前 detach；
它是 parameter-free weight，不是可訓練的 confidence head。

若 AP $l$ 對 RIS $r$ 提出的 element-$n$ unit proposal 為
$\mathbf p_{l,r,n}\in\mathbb R^2$，則最終 phase 為

$$
\boldsymbol\vartheta_{r,n}^{\rm energy}
=
\Pi\!\left(
\sum_l \bar E_{l,r}\mathbf p_{l,r,n}
\right),
\qquad
\bar E_{l,r}
=
\frac{E_{l,r}}{\sum_{l'}E_{l',r}}.
$$

共同尺度不改變 consensus 方向。若 resultant 幾乎為零，程式回退到 phase 0；
現有正式評估中沒有觸發這個 fallback。AP→CPU 每個 AP–RIS pair 傳送
$N$ 個 phase angles 與一個 $E_{l,r}$，合計 $N+1$ 個 fp32 scalars；$N=30$
時為 31。

Energy consensus 的「local」只指權重 $E_{l,r}$ 的資訊來源。目前 G0／G1／G2
在 paper-decentralized inference 下產生 proposal 時，仍使用式 (10) 允許的 shared-UE
cross-AP links；因此應描述為 `paper-decentralized proposal + strictly AP-local energy
weight + centralized parameter-free consensus`。

### 10.1 權重形式的 MLE 論證與其邊界

上面的規則把 $E_{l,r}$ 當成固定權重使用，但沒有說明為什麼是能量。以下給出一個使這個選擇具有
最大概似解釋的明確統計模型，並標示這個論證不能延伸到哪裡。

設 element $(r,n)$ 存在一個未知的共同最佳相位 $\phi_{r,n}$，AP $l$ 送出的提案角度為
$\psi_{l,r,n}=\phi_{r,n}+\varepsilon_{l,r,n}$，其中各 AP 的誤差彼此獨立，且服從 concentration 為
$\kappa_{l,r}$ 的 von Mises 分布 $\mathcal{VM}(0,\kappa_{l,r})$。給定所有提案，$\phi_{r,n}$ 的
對數概似為

$$
\log p(\{\psi_{l,r,n}\}_l\mid\phi)
=\sum_l \kappa_{l,r}\cos(\psi_{l,r,n}-\phi)+\text{const}.
$$

令 $S=\sum_l\kappa_{l,r}e^{\mathrm j\psi_{l,r,n}}$，則

$$
\log p(\{\psi_{l,r,n}\}_l\mid\phi)
=|S|\cos(\arg S-\phi)+\text{const},
$$

所以當 $S\ne0$ 時，最大概似解為

$$
\hat\phi_{r,n}
=\arg S
=\arg\Big(\sum_l \kappa_{l,r}e^{\mathrm{j}\psi_{l,r,n}}\Big)
=\Pi\Big(\sum_l \kappa_{l,r}\mathbf p_{l,r,n}\Big).
$$

$S=0$ 時所有相位具有相同概似。也就是說，
**在上述 independent von Mises 模型下，以 concentration $\kappa_{l,r}$ 加權的 circular consensus
是共同相位的最大概似估計**。因為 $\Pi(\cdot)$ 對權重的共同尺度不變，只有同一 RIS 內的相對值有意義，
這與前面 $\bar E_{l,r}=E_{l,r}/\sum_{l'}E_{l',r}$ 的正規化一致。

把 $\kappa_{l,r}\propto E_{l,r}$ 代入即得現行規則。這個代入的物理讀法是：AP $l$ 對 RIS $r$ 的相位
提案，其資訊來源是它所服務的 UE 經由 $r$ 的級聯通道，而
$E_{l,r}=\sum_{k\in\mathcal K_l}\lVert\mathbf H_{(l,r,k)}\rVert_F^2$ 正是這些級聯通道的總能量。
$E_{l,r}$ 很小的 AP，既較無依據判斷 $r$ 該用什麼相位，其自身訊號也幾乎不經過 $r$；兩種意義上它都
應該少投票。實作中的 detach 是把這個物理 proxy 固定下來的設計選擇；MLE 模型本身不要求 detach，
也不排除另外學習 concentration。

這個論證的邊界必須明確標示，否則它會被過度引用：

1. von Mises 與跨 AP 獨立性是**建模假設**，不是關於本網路的定理。提案由訓練過的 GNN 產生，其誤差
   分布未經檢驗，也不必然獨立。
2. $\kappa_{l,r}\propto E_{l,r}$ 是一個**代理假設**，不是從通道模型推導出來的結果。
3. 真正的目標是 sum rate，不是相位的 MLE。即使相位估計是最佳的，也不自動等於 rate 最佳。

因此這一小節只把問題從「為什麼用能量」轉換成「能量是否是 concentration 的好代理」。後者是可量測的：
[E05](./experiments/e05_energy_consensus.md) 的 oracle-style weight 診斷凍結 beamformer 與 proposals，
只把權重當成自由變數，以有限步數的雙初始化 Adam 對真實 sum rate 搜尋，並報告 $E_{l,r}$ 回收了
equal 到最佳找到權重之間多少比例的差距，以及它與該權重的秩相關。這個有限預算的非凸搜尋不是
global upper bound。

### 10.2 Manifold consensus 的 objective、閉式解與對稱性

固定 RIS $r$，令 $p_{l,n}=e^{\mathrm j\psi_{l,n}}\in\mathbb T$，
$E_l\geq0$，並將零權重 AP 排除。融合定義為 torus
$\mathbb T^N=\{v\in\mathbb C^N:|v_n|=1\}$ 上的 weighted chordal-disagreement problem：

$$
\min_{v\in\mathbb T^N}J(v),\qquad
J(v)=\sum_l E_l\sum_{n=1}^N|v_n-p_{l,n}|^2.
$$

**Proposition 1（閉式解）。** 若第 $n$ 個 resultant
$S_n=\sum_l E_lp_{l,n}\ne0$，則該座標的唯一 minimizer 是
$v_n^{\rm opt}=S_n/|S_n|$。證明：$|v_n|=|p_{l,n}|=1$ 使
$J_n(v_n)=2\sum_lE_l-2\operatorname{Re}(\overline v_n S_n)$；由
$\operatorname{Re}(\overline v_n S_n)\le|S_n|$，等號只在 $v_n=S_n/|S_n|$
成立。各座標分離，故所有 $S_n\ne0$ 時得到整個 torus 上的唯一解。共同乘上正的權重
尺度不改變解，因而可改用 $\bar E_l=E_l/\sum_jE_j$。

**Proposition 2（symmetry 與可行性）。** 對精確算術的融合映射，若
$\sum_lE_l>0$ 且所有相關座標的 $S_n\ne0$，
(i) 同時重排 $(p_l,E_l)$ 不改變 $v^{\rm opt}$；(ii) 把同一座標的所有提案乘
$e^{\mathrm j\alpha_n}$，輸出也乘同一因子；(iii) 對每個 $n$ 都有
$|v_n^{\rm opt}|=1$。前兩項由 weighted sum 的交換律及
$\sum_lE_le^{\mathrm j\alpha_n}p_{l,n}=e^{\mathrm j\alpha_n}S_n$
直接得到，第三項由正規化得到。若 $S_n=0$，objective 對該座標的所有相位相同，
沒有唯一解。實作先以 $\max(\sum_lE_l,10^{-12})$ 正規化權重，再在所得 resultant
norm 不大於 $10^{-12}$ 時選 phase 0；因此極小但非零的總能量也可能觸發 fallback。
這個固定 fallback 仍保持
AP permutation invariance 與 unit modulus，卻**不**保證共同旋轉 equivariance。
若 $\sum_lE_l=0$，所有座標都落在相同的非唯一情況。
這些 symmetry 只屬於**融合映射**；AP-specific power-control heads、CSI visibility 與整個
G2 policy 並未由此命題獲得 AP permutation equivariance。

**Proposition 3（介面成本）。** 假設每個 AP 已具備其指定的 observation，且每個
$(l,r)$ 只上傳 $N$ 個 phase angles 與一個非負 $E_{l,r}$，CPU 只執行上述
weighted sum 與 projection，則一輪 AP→CPU 上行恰為 $RL(N+1)$ 個 real scalars，
CPU-side trainable parameter 數為零。這是訊息格式的直接計數；它不涵蓋事前 CSI
取得、CPU→RIS 致動或任何重傳，也不表示物理傳播時間只有一個時槽。

三個 proposition 說明融合規則的 objective、可行性、對稱性與協定成本，**不**構成獨立的
演算法 novelty、sum-rate 最佳性或 convergence 保證。Chordal objective 衡量提案間的一致性；
實際訓練目標仍是系統 sum rate，§10.1 的 von Mises 解釋則需要額外統計假設。

<a id="graph-variants"></a>

## 11. G2 graph method and ablations

本節把四個 architecture 寫成同一組 node-level 記號。目的不是另造一套抽象，而是精確說明
`model.py` 與 `variants.py` 實際更新哪些 state、在哪個軸聚合，以及 phase proposal 如何從
最後一層 state 產生。G2 是 proposed method 與後續研究主線；G1 只保留為移除 per-RIS context
的 matched ablation，不作為共同 finalist 或獨立延伸方向。G0 是 representation control，R0 是
原論文 baseline。G0／G1／G2 仍是實驗報告與 CLI 的 canonical 名稱：

| 名稱 | 研究角色 | CLI | GNN state | RIS proposal／aggregation | 有效參數量 |
|---|---|---|---|---|---:|
| R0 | baseline | `--arch r0` | AP–UE nodes + RIS nodes | AP 傳 $4N$ latent；CPU learned reduction | 1,664,445 |
| G0 | representation control | `--arch g0` | 同 R0 | AP local projection；energy consensus | 1,664,445 |
| G1 | G2 no-context ablation | `--arch g1` | AP–UE nodes + per-sample link-embedding tensor | shared masked phase head；energy consensus | 819,269 |
| **G2** | **proposed／finalist** | `--arch g2` | node-free link memory + link-derived context | shared masked phase head；energy consensus | 868,421 |

<a id="state-action-factorization"></a>

### State/action-factorized link-memory framework

此處先定義可脫離 RIS 記號使用的設計類別，再於 §11.6–11.7 指明 G2 及其 G1 ablation 的實例。設多個
agent $l$ 各有局部觀測 $O_l$、局部 active action $a_l$，而共享物件
$o\in\mathcal O$ 要接受一個位於可行集合 $\mathcal M_o$ 的共同 action $v_o$。
$i\in\mathcal I_l$ 是 agent $l$ 在其觀測範圍內可見的 interaction slot；
$\mathcal S_l\subseteq\mathcal I_l$ 是其自己負責的 slots。對每個 $(o,i)$，agent 以共用
encoder 從局部可見的 evidence $\xi_{l,o,i}(O_l)$ 建立
$z_{l,o,i}=f_{\rm link}(\xi_{l,o,i})$。整個 $Z_l=(z_{l,o,i})_{o,i}$ 是**當次輸入**的
link memory：它在一次 forward pass 內固定，跨 channel samples 重新計算，並非跨樣本保存
的狀態或額外 trainable embedding。只讓 interaction state $u_{l,i}^{(t)}$ 遞迴更新：

$$
u_{l,i}^{(0)}=P(\{z_{l,o,i}\}_{o\in\mathcal O}),\qquad
c_{l,i}=C(\{z_{l,o,i},\xi_{l,o,i}\}_{o\in\mathcal O}),\qquad
u_{l,i}^{(t+1)}=U_t(u_{l,i}^{(t)},m_{l,i}^{(t)},c_{l,i}),
$$

其中 $m_{l,i}^{(t)}$ 是其他 interaction slots 的 permutation-invariant message。$C$ 可省略
（G1），或把固定的 link-derived context 在每層重複注入（G2）；兩者都不建立 shared-object
recurrent state。最後，$a_l=A(\{u_{l,i}^{(D)}\})$，而共享 action 的局部提案由**同一份**
link memory 和最終 state 延遲讀出：

$$
t_{l,o,i}=H_o(z_{l,o,i},u_{l,i}^{(D)}),\qquad
p_{l,o}=\operatorname{Proj}_{\mathcal M_o}
 Q_o\bigl(\operatorname{Pool}_{i\in\mathcal S_l}\{t_{l,o,i}\}\bigr),\qquad
e_{l,o}=E_o(O_l)\geq0,\qquad
v_o=F_o(\{(p_{l,o},e_{l,o})\}_l).
$$

$p_{l,o}$ 必須已是可執行的 action；$e_{l,o}$ 只量化 agent 的局部證據，而非由共享
ground-truth reward 偷看得到的 confidence。融合 $F_o$ 僅讀取提案與證據，不再讀取
hidden link states、原始 CSI 或 trainable CPU decoder。所有 learned maps 可用共同系統
目標 end-to-end 訓練；此定義不假設融合能找到全域最佳 action。

在本系統，$o=r$、$\mathcal M_r=\mathbb T^N$、$i=(a,k)$，
$Z_l\in\mathbb R^{R\times LK\times q}$，$a_l$ 是 AP beamformer，
$p_{l,r}\in\mathbb T^N$，而 $e_{l,r}=E_{l,r}$ 是 §10 的 own-served
cascaded-channel energy。完整 G2 的 $C$ 是 §11.7 的固定加權和；G1 ablation 將 $C$ 移除。
§11.6 的 shared masked phase head 實作 $H_r,Q_r$，§10 的 circular rule 實作 $F_r$。
在 paper-decentralized mode，$O_l$ 仍包含式 (10) 允許的 shared-UE cross-AP links；
「局部」不表示只有 AP 自身的 CSI，只有 $E_{l,r}$ 嚴格限於 own served channels。

| Dataflow axis | Lim–Vu multi-RIS GNN | G2（G1 ablation 共用） |
|---|---|---|
| 反射鏈路輸入 | RIS–UE edge 的 cascaded-channel feature | 每個 $(r,i)$ 的 CSI 與 normalized energy 經共用 encoder 成 $z_{r,i}$ |
| 逐層更新 | RIS 與 UE node representation 雙向更新 | 只更新 AP–UE interaction state；$Z_{\rm link}^{(0)}$ 在 forward pass 內固定 |
| RIS action 讀出 | 最後的 RIS node representation 直接映成 phase | 固定 link memory 與最後 AP–UE state 做 late masked readout，先得到每 AP 的可執行提案 |
| 多端協調 | 單 BS 設定，無 AP→CPU 提案融合介面 | 每 AP–RIS 送 $N$ angles 與一個 own-energy，CPU 一輪無參數融合 |

這個分解的可檢驗特徵是：共享物件既不承載 recurrent interaction state，也不在早期被壓成
單一物件向量；其逐鏈路 evidence 保留到 late action readout。它是 G2 的完整
architecture–interface dataflow，而不是「GNN 使用 link 資訊」的同義詞。Lim 與 Vu 的
[multi-RIS GNN](https://arxiv.org/abs/2501.14987) 已把級聯通道放在 RIS–UE edge features，
但其 RIS 與 UE 兩種 node representation 都由雙向 message passing 逐層更新，RIS node 最後
直接輸出相位。故該文是 edge information 的明確前例，卻不是上述固定逐鏈路 memory、
只更新 interaction states、late AP-local executable proposal 與單輪 evidence fusion 的
完整流程。其 single-BS 系統與本專案的 multi-AP cell-free 設定也不同；這個差異只界定
可比性，不能當作架構 novelty 的替代證據。

### 11.1 共用 graph view、edge normalization 與 MLP 記號

一個 AP–UE node 代表一條「AP $a$ 到實體 UE $k$」的 link，記為
$i=(a,k)\in\mathcal I$；因此程式固定配置 $|\mathcal I|=LK$ 個 node slots。對 RIS $r$ 與
node $i$，輸入為 §1 的 real-valued CSI vector $\mathbf x_{r,i}\in\mathbb R^{2M(N+1)}$、
cascaded-channel energy $e_{r,i}\ge 0$，以及 direct-link energy $d_i\ge0$。為了簡化式子，
以下省略 batch index。這裡的 $\mathbf x_{r,i}$ 是 `simulation.py` 產生並送入 GNN 的
`user_feature`：它由原始 $\widetilde{\mathbf h}_{(a,r,k)}$ reshape、normalization 與 visibility
mask 得到，仍是**輸入 CSI feature**，不是 hidden state、link embedding 或 link-embedding tensor。

Centralized path 使用完整 association mask。若 $i=(a,k)$，paper-decentralized path 在 AP $l$
上的 visibility mask 是

$$
m_{l,(a,k)}
=
\mathbb 1\{k\in\mathcal K_l\cap\mathcal K_a\};
$$

own-only path 則只允許 $a=l$。模型先把不可見 slots 的 $\mathbf x_{r,i}$、$e_{r,i}$ 與 $d_i$
乘成零，再跑同一組 shared GNN weights。程式為保持固定 tensor shape，並沒有真的刪掉不可見 nodes；因此
affine layer 的 bias 仍可在 zero-input slot 產生一個與 CSI 無關的常數 state。所有 edge-weighted
sum 都因 edge 為零而忽略這些 slots，最終 beamformer 與 phase pooling 也會再套 served-user
mask，所以不可見 CSI 不會從 placeholder 洩漏。

同一個非負 edge $e_{r,i}$ 依 message 方向使用兩種 L1 normalization：

$$
\widehat e^{\mathrm{node}}_{r,i}
=
\frac{e_{r,i}}{\sum_{j\in\mathcal I}e_{r,j}},
\qquad
\widehat e^{\mathrm{RIS}}_{i,r}
=
\frac{e_{r,i}}{\sum_{s=1}^{R}e_{s,i}},
\qquad
\widehat d_i
=
\frac{d_i}{\sum_{j\in\mathcal I}d_j}.
$$

第一式沿 AP–UE nodes 正規化，用於送進某個 RIS node；第二式沿 RIS 正規化，用於送進某個
AP–UE node。分母為零時，實作的 normalized vector 為零。`simulation.py` 另先在每個 AP block
內，對固定 $(r,\text{feature coordinate})$ 的 $K$ 個 UE slots 做 L2 normalization，再把 $L$ 個
AP blocks 串接；以下的 $\mathbf x_{r,i}$ 已包含這一步與目前 graph view 的 mask。

所有 $q$-dimensional update MLP 都具有相同基本形式，但不同 layer 不共用參數：

$$
\operatorname{MLP}(\mathbf a)
=
\mathbf W_2\,\operatorname{LReLU}(\mathbf W_1\mathbf a+\mathbf b_1)+\mathbf b_2,
$$

為避免和 RIS 實體相位角 $\phi_{r,n}$ 混淆，本節不再用 $\phi$ 或 $\psi$ 表示 neural map。
具名符號 $\operatorname{MLP}_{U,\mathrm{init}}$、
$\operatorname{MLP}_{R,\mathrm{init}}$、$\operatorname{MLP}_{U,t}$、
$\operatorname{MLP}_{R,t}$、$\operatorname{MLP}_{\mathrm{link}}$ 與
$\operatorname{MLP}_{\mathrm{phase}}$ 分別表示 AP–UE initialization、RIS initialization、
第 $t$ 個 AP–UE／RIS update、link encoder 與 phase-token fusion；
$\operatorname{MLP}_{\alpha,l}$ 是 AP $l$ 的 power-control encoder。它們都是不同的 learned
affine–LReLU–affine maps；$\operatorname{MLP}_{U,t}^{\mathrm{G1}}$ 與
$\operatorname{MLP}_{U,t}^{\mathrm{G2}}$ 的 superscript 表示兩個 variant 使用不同 input width 與
獨立參數。

其中 hidden width 為 $2q$、output width 為 $q$。每層不是覆寫舊 state，而是把新產生的
$q$ 維 feature 接在舊 state 前面。故第 $t$ 層 AP–UE state $\mathbf u_i^{(t)}$ 與（若存在）
RIS state $\mathbf s_r^{(t)}$ 都在 $\mathbb R^{q(t+1)}$；預設 $q=64$、$D=6$，最後維度為
$q(D+1)=448$。定義 element-wise other-node max message

$$
\mathbf m_{i,\max}^{(t)}
=
\max_{j\in\mathcal I,\,j\ne i}\mathbf u_j^{(t)}.
$$

若 graph 只有一個 node，程式回傳該 node 自己。這個 max 是 feature-wise max，不是選出一個
完整鄰居向量。AP–UE node 子圖因此是透過 global max pooling 耦合的 complete set graph，而不是
依幾何距離建立的 sparse adjacency graph。

### 11.2 R0／G0 共用的 RIS-node backbone

R0 與 G0 的 GNN backbone 完全相同；兩者只在 RIS action interface 與 CPU aggregation 不同。
初始 AP–UE node 先對每個 RIS 的 CSI encoding 取平均：

$$
\mathbf u_i^{(0)}
=
\frac{1}{R}\sum_{r=1}^{R}
\operatorname{MLP}_{U,\mathrm{init}}(\mathbf x_{r,i})
\in\mathbb R^q.
$$

接著以 channel-energy weights 把 AP–UE states 聚合到每個 RIS，並用一個 shared direct-link
summary 補入不經 RIS 的路徑：

$$
\mathbf a_r^{(0)}
=
\sum_i\widehat e^{\mathrm{node}}_{r,i}\mathbf u_i^{(0)},
\qquad
\mathbf a_D^{(0)}
=
\sum_i\widehat d_i\mathbf u_i^{(0)},
$$

$$
\mathbf s_r^{(0)}
=
\operatorname{MLP}_{R,\mathrm{init}}
\left(
[\mathbf a_r^{(0)}\,\Vert\,\mathbf a_D^{(0)}]
\right)
\in\mathbb R^q.
$$

對 $t=0,\ldots,D-1$，AP–UE node 收到兩類 message：其他 AP–UE nodes 的 max，以及所有 RIS
states 的 channel-weighted average：

$$
\mathbf m_{i,R}^{(t)}
=
\sum_{r=1}^{R}\widehat e^{\mathrm{RIS}}_{i,r}\mathbf s_r^{(t)},
$$

$$
\Delta\mathbf u_i^{(t+1)}
=
\operatorname{MLP}_{U,t}
\left(
[\mathbf u_i^{(t)}\,\Vert\,\mathbf m_{i,\max}^{(t)}\,\Vert\,
\mathbf m_{i,R}^{(t)}]
\right),
\qquad
\mathbf u_i^{(t+1)}
=
[\Delta\mathbf u_i^{(t+1)}\,\Vert\,\mathbf u_i^{(t)}].
$$

RIS node 同時接收目前 AP–UE states 的 cascaded-與 direct-link summaries：

$$
\mathbf m_{r,U}^{(t)}
=
\sum_i\widehat e^{\mathrm{node}}_{r,i}\mathbf u_i^{(t)},
\qquad
\mathbf m_D^{(t)}
=
\sum_i\widehat d_i\mathbf u_i^{(t)},
$$

$$
\Delta\mathbf s_r^{(t+1)}
=
\operatorname{MLP}_{R,t}
\left(
[\mathbf s_r^{(t)}\,\Vert\,\mathbf m_{r,U}^{(t)}\,\Vert\,
\mathbf m_D^{(t)}]
\right),
\qquad
\mathbf s_r^{(t+1)}
=
[\Delta\mathbf s_r^{(t+1)}\,\Vert\,\mathbf s_r^{(t)}].
$$

這是一個雙向 heterogeneous message-passing block：RIS states 將 surface-level context 傳回
AP–UE nodes，AP–UE states 再依 channel energy 更新各 RIS。採用 normalized weighted sum 是為了
保留 channel-strength ordering，同時避免 message scale 隨可見 node 或 RIS 數直接成長；
other-node max 則提供對最強 competing／supporting link 的非平均化摘要。Dense concatenation 保留
早期 CSI encoding，使深層 readout 不必只依賴最後一次 $q$ 維更新。這些是 architecture 的設計
動機；哪一項真正造成效能差異，不能由公式單獨判定。

### 11.3 四個方法共用的 active-beamforming readout

R0、G0、G1、G2 都從最後的 AP–UE state 產生 active beamformer。Shared direction head 先輸出
每個 node 的 $2M$ 個 real／imaginary logits：

$$
\widetilde{\mathbf w}_i
=
\mathbf W_B\mathbf u_i^{(D)}+\mathbf b_B
\in\mathbb R^{2M}.
$$

AP $l$ 另有自己的 power-control head：

$$
\alpha_l
=
\sigma\!\left(
\mathbf w_{\alpha,l}^{T}
\left[
\frac{1}{|\mathcal I|}\sum_i
\operatorname{MLP}_{\alpha,l}(\mathbf u_i^{(D)})
\right]
+b_{\alpha,l}
\right)
\in(0,1).
$$

只保留 AP $l$ 自己服務的 UE columns，將該 block flatten 後做一次 L2 normalization，再乘上
$\sqrt{P_{\max}\alpha_l}$：

$$
\mathbf W_l
=
\sqrt{P_{\max}\alpha_l}\,
\frac{\widetilde{\mathbf W}_l}
{\max(\lVert\widetilde{\mathbf W}_l\rVert_F,\epsilon)}.
$$

因此每個 AP block 滿足 $\lVert\mathbf W_l\rVert_F^2\le P_{\max}$。Direction head 在 AP 間共用，
power heads 則 AP-specific；這個分工讓各 AP 可學不同的使用功率，同時對 UE/link ordering 保持
同一個 direction mapping。實作在 sigmoid 前把 power logit 截到 $[-20,20]$，只用來避免數值
overflow。Centralized path 的各 head 讀取同一份 global states；
paper-decentralized path 則由 AP $l$ 的 local masked graph 產生 $\mathbf u_{l,i}^{(D)}$ 後，只輸出
自己的 block。

### 11.4 R0：RIS-node readout 與 learned CPU reduction

R0 在 AP $l$ 對 RIS $r$ 的輸出由兩個 affine branches 組成。第一個 branch 讀 RIS state，第二個
branch 讀 AP $l$ 對全部 RIS 與 $K$ 個 UE 的 own-link energy vector
$\mathbf e_l\in\mathbb R^{RK}$：

$$
\mathbf v_{l,r}
=
\left[
\mathbf A_l\mathbf s_{l,r}^{(D)}+\mathbf a_l
\;\middle\Vert\;
\mathbf B_l\mathbf e_l+\mathbf b_l
\right]
\in\mathbb R^{4N}.
$$

Decentralized inference 的 $\mathbf s_{l,r}^{(D)}$ 明確帶 AP index，因為每個 AP 都用自己的
masked graph view 跑一次同權重 backbone。AP 傳送 $\mathbf v_{l,r}$ 後，CPU 先沿 AP 加總，
再用 shared learned affine decoder 產生 $2N$ logits：

$$
\mathbf y_r
=
\mathbf W_{\mathrm{reduce}}
\sum_{l=1}^{L}\mathbf v_{l,r}
+\mathbf b_{\mathrm{reduce}},
\qquad
\boldsymbol\vartheta_{r,n}^{\mathrm{R0}}
=
\Pi
\left(
\begin{bmatrix}
y_{r,n}^{\mathrm{Re}}\\
y_{r,n}^{\mathrm{Im}}
\end{bmatrix}
\right).
$$

R0 的設計理由是先讓各 AP 傳較寬的 latent，再由 CPU 在看到所有 AP contributions 後學習如何
解碼 shared RIS action；它保留最大的 aggregation flexibility，也與原論文式 (26)–(28) 對齊。
代價是每個 AP–RIS pair 要傳 $4N$ reals，而且 CPU 上仍有 trainable phase decoder；
$\mathbf v_{l,r}$ 本身也不是可直接執行或量化的 phase proposal。

### 11.5 G0：保留 R0 graph，只替換 phase interface

G0 完整保留 §11.2 的 AP–UE／RIS node updates、§11.3 的 beamforming path、R0 的 AP-specific
$\mathbf v_{l,r}$ readout，以及同一組 $\mathbf W_{\mathrm{reduce}}$。唯一差別是把 shared decoder
線性地搬到 AP，並把 bias 平分到 $L$ 個 AP terms：

$$
\mathbf z_{l,r}
=
\mathbf W_{\mathrm{reduce}}\mathbf v_{l,r}
+\frac{\mathbf b_{\mathrm{reduce}}}{L}
\in\mathbb R^{2N},
\qquad
\mathbf p_{l,r,n}
=
\Pi
\left(
\begin{bmatrix}
z_{l,r,n}^{\mathrm{Re}}\\
z_{l,r,n}^{\mathrm{Im}}
\end{bmatrix}
\right).
$$

AP 傳送 $N$ 個 $\mathbf p_{l,r,n}$ 的 angles 與 §10 的 strictly local $E_{l,r}$；CPU 重建 unit
vectors 後執行

$$
\boldsymbol\vartheta_{r,n}^{\mathrm{G0}}
=
\Pi\!\left(
\sum_l\bar E_{l,r}\mathbf p_{l,r,n}
\right).
$$

G0 的角色是 representation control。它回答「只把 R0 latent interface 改成可解讀的 local phase
proposal，並用物理上可取得的 channel energy 做 parameter-free consensus，是否已足夠？」因此
G0 刻意不改 backbone，讓它能隔離 interface／consensus 變更，並作為完整 G2 representation
package 的對照。Local projection 將每個 AP 的 logit magnitude 丟棄，$E_{l,r}$ 則提供一個不需
學習、只依 own served channels 的 AP–RIS-level importance。這個 energy 是設計 proxy，不是由
公式保證最優的 confidence。

### 11.6 G1：移除 context 的內部 ablation

G1 只用於隔離 G2 的 per-RIS context，不是另一個 proposed method 或後續訓練主線。它與 G2
共用 node-free link-memory 與 action interface，但移除每層 context reinjection。具體而言，G1
移除所有 $\mathbf s_r^{(t)}$ 與 RIS-node update。它不先把同一 RIS 的所有 links 壓成一個
state，而是先把每個 $(r,i)$ 的 CSI input 分別編碼，再把所有 encoder outputs 堆成一個
link-embedding tensor。

對單一 RIS $r$ 與 AP–UE node $i$，link encoder 的完整輸入向量定義為

$$
\boldsymbol\xi_{r,i}
=
[\mathbf x_{r,i}\,\Vert\,
\widehat e^{\mathrm{node}}_{r,i}\,\Vert\,
\widehat e^{\mathrm{RIS}}_{i,r}\,\Vert\,
\widehat d_i]
\in\mathbb R^{2M(N+1)+3}.
$$

其中 $\mathbf x_{r,i}$ 是 $2M(N+1)$ 維 normalized CSI feature，後面三項都是 scalar edge
features。Shared link encoder 將這個**輸入向量**映射成一個 $q$ 維、以 RIS $r$ 的 CSI 與 edge
features 為條件的 AP–UE link embedding：

$$
\mathbf z_{r,i}^{(0)}
=
\operatorname{MLP}_{\mathrm{link}}
\left(\boldsymbol\xi_{r,i}\right)
\in\mathbb R^q.
$$

因此 $\mathbf z_{r,i}^{(0)}$ 是**單一 RIS-conditioned AP–UE link embedding**。把所有
$R\times LK$ 個 embeddings 沿 RIS 與 AP–UE-node axes 堆疊後，得到

$$
\mathbf Z_{\mathrm{link}}^{(0)}
=
\operatorname{stack}_{r,i}\!\left(\mathbf z_{r,i}^{(0)}\right)
\in\mathbb R^{R\times LK\times q},
$$

這就是本文所稱的 **link-embedding tensor**；包含 batch 軸時，實作 tensor `z0` 的 shape 是
$(B,R,LK,q)$。它不是一組跨 samples 固定的 learned embeddings：每個 channel sample、每個
centralized／AP-local graph view 都會由自己的 $\mathbf x$ 與 edges 重新計算一次。這個 tensor
只在同一次 forward pass 的後續 $D$ 個 node-update layers 中保持不變，亦即 embeddings 不具有
自己的 recurrent update。

同一個 raw energy 同時以 node-normalized 與 RIS-normalized scalar 出現，因為兩者回答不同問題：
前者表示 link 在 RIS $r$ 收到的所有 node evidence 中占多少，後者表示 RIS $r$ 在 node $i$ 的
所有 reflected paths 中占多少。G1/G2 的 canonical `identity=none` 不附加 RIS ID；RIS index 仍
是 tensor axis，但 link encoder weights 對所有 $r$ 共用。

AP–UE node 的初始 state 對其 $R$ 個 RIS-conditioned link embeddings 同時取 mean 與
feature-wise max：

$$
\boldsymbol\mu_i^{(0)}
=
\frac{1}{R}\sum_r\mathbf z_{r,i}^{(0)},
\qquad
\boldsymbol\nu_i^{(0)}
=
\max_r\mathbf z_{r,i}^{(0)},
$$

$$
\mathbf u_i^{(0)}
=
\mathbf W_{\mathrm{pool}}
[\boldsymbol\mu_i^{(0)}\,\Vert\,\boldsymbol\nu_i^{(0)}]
+\mathbf b_{\mathrm{pool}}
\in\mathbb R^q.
$$

後續 $D$ 層只在 AP–UE nodes 之間傳遞 other-node max message：

$$
\Delta\mathbf u_i^{(t+1)}
=
\operatorname{MLP}_{U,t}^{\mathrm{G1}}
\left(
[\mathbf u_i^{(t)}\,\Vert\,\mathbf m_{i,\max}^{(t)}]
\right),
\qquad
\mathbf u_i^{(t+1)}
=
[\Delta\mathbf u_i^{(t+1)}\,\Vert\,\mathbf u_i^{(t)}].
$$

$\mathbf Z_{\mathrm{link}}^{(0)}$ 在這 $D$ 層中不會更新；其中每個
$\mathbf z_{r,i}^{(0)}$ 一方面決定 $\mathbf u_i^{(0)}$，另一方面直接保留到 phase head。這個
選擇避免再建立一套 RIS state recurrence，卻不會像 R0 那樣在 backbone 一開始就把所有
$(r,i)$ links 壓成每個 RIS 一個 vector。

G1 的 RIS proposal head 先把保留下來的 link embedding 與最終 node state 重新結合：

$$
\mathbf t_{r,i}
=
\operatorname{MLP}_{\mathrm{phase}}
\left(
[\mathbf z_{r,i}^{(0)}\,\Vert\,
\mathbf W_U\mathbf u_i^{(D)}+\mathbf b_U]
\right)
\in\mathbb R^q.
$$

令 $\mathcal S_l\subset\mathcal I$ 為 AP $l$ 自己服務的 AP–UE nodes；注意它小於或等於 AP $l$
依式 (10) 可見的 slots。Phase head 對 $\mathcal S_l$ 做 masked mean 與 max：

$$
\boldsymbol\mu_{l,r}
=
\frac{1}{|\mathcal S_l|}
\sum_{i\in\mathcal S_l}\mathbf t_{r,i},
\qquad
\boldsymbol\nu_{l,r}
=
\max_{i\in\mathcal S_l}\mathbf t_{r,i},
$$

$$
\mathbf z^{\mathrm{phase}}_{l,r}
=
\mathbf W_P
[\boldsymbol\mu_{l,r}\,\Vert\,\boldsymbol\nu_{l,r}]
+\mathbf b_P
\in\mathbb R^{2N},
\qquad
\mathbf p_{l,r,n}
=
\Pi
\left(
\begin{bmatrix}
z_{l,r,n}^{\mathrm{phase,Re}}\\
z_{l,r,n}^{\mathrm{phase,Im}}
\end{bmatrix}
\right).
$$

若 $\mathcal S_l$ 為空，兩個 pooled vectors 都設為零，且 $E_{l,r}=0$ 使該 AP 不參與 consensus。
最後仍使用 §10 的 energy-weighted circular consensus。Link encoder、node updates、beamformer
direction head 與 phase head 都跨 AP 共用；只有 power-control heads 是 AP-specific。

G1 與 G2 共用的 node-free link-memory 設計有三個動機。第一，per-link embeddings 把 RIS axis 保留到 phase readout，避免 RIS
node 在每一層反覆以 weighted sum 壓縮 link-specific evidence。第二，mean 與 max 分別提供平均
evidence 與 dominant evidence，並對 served-node ordering 保持 permutation invariance。第三，共用
phase head 直接輸出可執行的 local proposal，將 AP-specific parameters 與 CPU trainable decoder
都移出 passive path。代價是 AP–UE updates 在初始化後只看到壓縮過的 RIS summary；這正是 G2
要檢驗的缺口。

### 11.7 G2：完整 proposed method

G2 是本文採用的完整方法與所有後續 benchmark 的唯一 learned finalist。它的 link encoder、初始
node pooling、phase head、beamforming readout 與 energy consensus 全部
與 G1 相同。唯一介入是先從 link-embedding tensor 計算一次 channel-weighted context：

$$
\mathbf c_i
=
\sum_{r=1}^{R}
\widehat e^{\mathrm{RIS}}_{i,r}\mathbf z_{r,i}^{(0)}
\in\mathbb R^q,
$$

並在每一層 AP–UE update 重複提供同一個 $\mathbf c_i$：

$$
\Delta\mathbf u_i^{(t+1)}
=
\operatorname{MLP}_{U,t}^{\mathrm{G2}}
\left(
[\mathbf u_i^{(t)}\,\Vert\,\mathbf m_{i,\max}^{(t)}\,\Vert\,\mathbf c_i]
\right),
\qquad
\mathbf u_i^{(t+1)}
=
[\Delta\mathbf u_i^{(t+1)}\,\Vert\,\mathbf u_i^{(t)}].
$$

$\mathbf c_i$ 不是 attention：weights 是固定的 normalized channel energies，context 也不會隨
layer 更新。它的目的，是讓深層 AP–UE state 在 G1 的初始 mean/max pooling 之後，仍能直接讀到
由 RIS-resolved embeddings 組成的 node-specific summary；同時不恢復 RIS message-passing node、
AP-specific phase head 或 CPU parameters。G2 相對 G1 增加的參數全部來自每層 update MLP 多出的
$q$ 維 input columns。

### 11.8 結構選擇的比較與因果邊界

| 設計問題 | R0 | G0 | G1 | G2 |
|---|---|---|---|---|
| RIS information 在 backbone 中如何保存 | 每個 RIS 一個 recurrent state | 同 R0 | 每個 sample 重算、node updates 期間固定的 link-embedding tensor | 同 G1，另有 weighted context |
| AP–UE node 每層看到什麼 | self、other-node max、RIS-state average | 同 R0 | self、other-node max | G1 inputs + link-derived context |
| Local phase message 是否可直接解讀 | 否，為 $4N$ latent | 是，$N$ angles | 是，$N$ angles | 是，$N$ angles |
| CPU phase component | learned affine decoder | parameter-free energy consensus | 同 G0 | 同 G0 |
| 研究角色 | baseline | representation control | G2 no-context ablation | proposed method／研究主線 |
| 主要設計目的 | 最大化 learned aggregation flexibility | 隔離 interface／consensus 效果 | 歸因 per-RIS context | 保留 node-free link memory 並在深層 update 注入 RIS context |

這組比較有兩條不同的實驗邏輯。R0→G0 固定 graph representation，只改 phase interface
與 aggregation；G2 對 G0 的比較同時移除 RIS nodes、更換 link encoder／phase head、加入 context
並改變容量，因此是 representation package 比較，不是單一 layer-removal ablation。G1 則由 G2
移除 $\mathbf c_i$，是專門歸因 context reinjection 的 matched ablation。

所有 energy weights 都固定且 detach，但 proposals 與 beamformer 仍由 global sum-rate loss 做
centralized training。上述「避免 bottleneck」、「保留 context」與「提供 dominant evidence」是
可由資料流驗證的設計意圖，不等同已證明的性能原因。實際結果只支持 E06 所量到的範圍：主要
差距出現在 G2 相對 G0 的整組 representation change；G1 ablation 顯示 context 在 primary 40k
comparison 沒有可偵測優勢，但改善 own-only robustness。這個 null result 限制 context 的獨立
claim，卻不改變以完整 G2 作為研究主線的選擇。完整數值、seed 與 topology 限制見
[E06 報告](./experiments/e06_graph_energy_training.md)。

<a id="maddpg-arm"></a>

## 12. Archived MADDPG benchmark design（E11）

**本 arm 已停止且從 active code 移除。** 以下只保存被篩除設計的 provenance，不是維護中的
方法定義；不得用來支持 MADDPG 的效能 claim。停止原因與 claim boundary 見
[E11 報告](./experiments/e11_maddpg_baseline.md)。

本節定義的是 benchmark baseline，不是本專案提出的方法。它存在的目的，是檢驗 G2 的優勢
是否只是 weak-baseline 效果；因此除了「policy 如何被改善」之外，其餘條件都與 G2 對齊。

### 12.1 與 G2 共用的部分

Channel generator、fixed topology、$L$／$M$／$R$／$N$／$K$、$P_{\max}$、association mask、
unsupervised sum-rate 目標、式 (10) paper-decentralized observation、unit-modulus 與 per-AP
power constraint，以及 §10 的 AP→CPU interface 全部不變：每個 AP–RIS pair 傳送 $N$ 個 phase
angles 加一個 strictly local $E_{l,r}$，共 $N+1$ 個 fp32 scalars，一個 upstream round，CPU 端
仍是 parameter-free circular consensus。

唯一的介入是 learning rule。G2 直接對 sum rate 做 back-propagation；MADDPG 的 actor 完全看不到
這個 gradient，只透過一個 learned centralized critic 被改善。

### 12.2 Actor、action 與 critic

每個 AP 有一個獨立 actor $\pi_l$，輸入是 AP $l$ 的式 (10) observation，輸出

$$
\mathbf a_l
=
\bigl[
\operatorname{vec}(\mathbf W_l),\;
\operatorname{vec}(\mathbf P_{l,1}),\ldots,\operatorname{vec}(\mathbf P_{l,R})
\bigr],
$$

其中 $\mathbf W_l$ 已經過 L2 normalization 與 $\sqrt{P_{\max}\alpha_l}$ scaling，
$\mathbf P_{l,r}\in\mathbb R^{N\times2}$ 已經過 unit projection。$M=2$、$N=30$、$R=4$、$K=8$
時 $\mathbf a_l$ 為 272 個實數。實作檢查發現 exploration noise 是在 association mask 後加入，
卻沒有重新套用 mask；因此 behaviour policy 可能產生 deployed policy 不允許的未服務 UE 能量。

Critic 是單一 centralized $Q(s,\mathbf a_1,\ldots,\mathbf a_L)$，$s$ 為完整 CSI view。它只在
training 使用，不屬於任何 deployed parameter count。更新 actor $l$ 時，其他 agent 的 action 取自
replay buffer，這正是 MADDPG 與 joint-action ascent 的差別。

### 12.3 $\gamma=0$ 的 one-step terminal reduction

每個 channel realization 是一個獨立的 one-step terminal decision，immediate reward 就是該 sample 的
sum rate，因此

$$
y = r,
\qquad
\mathcal L_{\rm critic} = \mathbb E\bigl[(Q(s,\mathbf a)-r)^2\bigr].
$$

critic target 不含 bootstrapped future return，也因此不需要 target network。這與 deployment 的
one-shot inference 一致，但不是 Zhu et al. [13] 的 temporal MDP（$\gamma=0.99$）。

### 12.4 Actor 為何算 non-graph

Actor trunk 保留 shared per-(RIS, AP-UE) link MLP、masked mean/max pooling，以及兩層讀取
permutation-invariant pooled summary 的 per-node blocks。它移除了 RIS message-passing node、
edge-weighted neighbour aggregation、`_max_excluding_self` 與 depth-$D$ message rounds，
所以是 set-MLP 而非 GNN。

不採用完全 flatten 的 MLP 是因為 parameter budget：式 (10) observation 為
$R\times K_{\rm tot}\times 2M(N+1)=19\,840$ 個實數，在 per-actor 預算下 flatten 版本的 hidden
layer 只剩約 8 units，無法回答 benchmark 問題。代價是 E11 的差距同時混合 learning rule 與
representation；單獨分離 representation 的是 E10 的 DNN control。

### 12.5 與 Lowe et al. [22] 的兩處偏離

- Lowe et al. 每個 agent 各有一個 centralized critic；本 arm 的 reward 是單一 shared sum rate，
  那 $L$ 個 critic 會 regress 完全相同的 target，因此改用一個 shared critic。
- 原構造針對 temporal MDP；本 arm 是 $\gamma=0$ 的 contextual-bandit reduction。

因此它應被描述為 controlled, Zhu-et-al.-motivated MADDPG baseline，而不是 FL-MADDPG 的
faithful reproduction。實際數值、controls 與 limitations 見
[E11 報告](./experiments/e11_maddpg_baseline.md)。

<a id="model-based-pair"></a>

## 13. Model-based benchmark pair（E12）

本節同樣定義的是 benchmark baseline，不是本專案提出的方法。它回答的是 §8.2 item 5
的問題：在完全相同的 objective、constraint 與 channel sample 下，一組**不學習**的
model-based joint active／passive design 能達到多少。兩個 arm 共用同一個 surrogate
與同一組 solver，差別只在 information 與 coordination：

| Arm | Information | Coordination |
|---|---|---|
| `centralized` | CPU 端 full CSI | 一次 CSI upload、一次 precoder download |
| `distributed` | 只有 AP-local CSI | per-AP RIS copy 的 incremental ring ADMM |

### 13.1 來源選擇與 compatibility check

計畫的第一順位是 Xu et al. [8] 的 conventional、non-unrolled ADMM。該檢查**失敗**，
理由必須記錄成結果而不是繞過去：Xu et al. 的 objective（$\omega_k=1$ 的 weighted sum
rate）、per-BS power、unit modulus、multi-RIS、direct link、single-antenna UE 與
coherent cell-free joint transmission 都與本系統相符，但他們的 reflection subproblem
（式 (29)）寫出來之後是交給一個 convolutional neural block 求解，論文本身沒有提供
non-learned solver。要做出「conventional 版 Xu et al.」就必須外掛一個論文沒有的
reflection solver，那正是計畫禁止的 silent hybrid。

因此採用預先登記的 fallback：Huang et al. [7] 的 Algorithm 1。它對同一個問題是完整、
不含學習成分、有 convergence 證明與 complexity 分析的，且同樣有 multi-IRS、per-BS
power、unit modulus、direct link 與 BS-local reflection copy。`centralized` arm 則是
同一套 formulation 的 centralized counterpart，所以兩個 arm 的差異只在 information 與
coordination，而不在 surrogate 或 solver。

### 13.2 共用 surrogate：quadratic transform 與 WMMSE 的等價

令 AP $l$ 對 UE $k$ 的 effective channel 為 $\mathbf g_{l,k}$，並定義唯一耦合 AP 的量

$$
a_{k,j}=\sum_{l}\mathbf g_{l,k}^{H}\mathbf w_{l,j}
=\phi_{k,j}+\psi_{k,j},
$$

其中 $\phi$ 是 direct-link 部分、$\psi$ 是 RIS-reflected 部分。Lagrangian dual
transform 加 quadratic transform 之後，auxiliary 有 closed form（式 (9)、(10)）：

$$
\gamma_k=\mathrm{SINR}_k,
\qquad
\xi_k=\sqrt{1+\gamma_k}\,\frac{a_{k,k}}{\sum_j|a_{k,j}|^2+\sigma^2}.
$$

這與 WMMSE 的 $(\alpha_k,\nu_k)$ 是同一組量，關係為
$|\xi_k|^2=\alpha_k|\nu_k|^2$ 與 $\sqrt{1+\gamma_k}\,\xi_k=\alpha_k\nu_k$，其中
$\alpha_k=1+\gamma_k$、$\nu_k=a_{k,k}/(\sum_j|a_{k,j}|^2+\sigma^2)$。兩種寫法都出現在
文獻裡，因此這個恆等式由
`tests/test_model_based_joint.py::test_fp_auxiliaries_match_the_wmmse_parameterization`
固定，而不是只寫在文字裡。

### 13.3 Beamformer block

固定 $(\gamma,\xi)$ 後，對 AP $l$ 的 stationarity 條件是 Huang et al. 式 (11)：

$$
(\boldsymbol\Phi_l+\mu_l\mathbf I)\,\mathbf w_{l,j}
=\sqrt{1+\gamma_j}\,\xi_j\,\mathbf g_{l,j}-\boldsymbol\Omega_{l,j},
\qquad
\boldsymbol\Phi_l=\sum_k|\xi_k|^2\mathbf g_{l,k}\mathbf g_{l,k}^{H},
$$

$$
\boldsymbol\Omega_{l,j}
=\sum_k|\xi_k|^2\mathbf g_{l,k}\bigl(a_{k,j}-\mathbf g_{l,k}^{H}\mathbf w_{l,j}\bigr),
$$

括號內就是**其他 AP 貢獻的 aggregate**。這是 distributed arm 能只用 AP-local CSI 的關鍵：
AP 只需要自己的 channel 加上 $K^2$ 個 aggregate scalar，永遠不需要別的 AP 的 channel matrix。

$\mu_l$ 由 bisection 求得，滿足 $\lVert\mathbf W_l\rVert_F^2\le P_{\max}$。因為 $M$ 很小，
實作先對 $\boldsymbol\Phi_l$ 做一次 eigendecomposition，之後 block power
$\sum_n E_n/(\lambda_n+\mu)^2$ 是純量算式，bisection 不再需要任何 matrix solve；
$\mu_{\rm hi}=\sqrt{\sum_n E_n/P}$ 是有效上界。若 unconstrained 解已經在預算內則取
$\mu_l=0$，這是論文寫成 $=P_b$ 時略過的 KKT complementary-slackness 情形。有限次 bisection
回傳保證可行的 upper bracket endpoint，而不是可能仍略微超過預算的 midpoint。

### 13.4 Reflection block

固定 $\mathbf W$ 後 surrogate 對 stacked $\mathbf v\in\mathbb C^{RN}$ 是二次式
$\mathbf v^{H}\mathbf Z\mathbf v+2\,\mathrm{Re}(\mathbf v^{H}\mathbf q)$，其中

$$
\mathbf Z=\sum_{k,j}|\xi_k|^2\,\mathbf t_{k,j}\mathbf t_{k,j}^{H},
\qquad
\mathbf q=\sum_{k,j}|\xi_k|^2\,\mathbf t_{k,j}\,\mathrm{base}_{k,j}^{*}
-\sum_k\sqrt{1+\gamma_k}\,\xi_k^{*}\,\mathbf t_{k,k},
$$

$\mathbf t_{k,j}$ 收集 cascade 項、$\mathrm{base}$ 收集 direct-link 項與 peer aggregate。
Huang et al. 印出的 (13) 少了一次共軛與 AP 自身的 direct-link 項；此處採用重新推導的版本，
並與 Xu et al. 式 (31) 對照一致（兩者只差符號約定）。

求解用 MM（式 (15)、(17)）：任何 $\zeta\ge\lambda_{\max}(\mathbf Z)$ 都是合法 majorizer，
故

$$
\mathbf v\leftarrow-\exp\bigl(j\angle((\mathbf Z-\zeta\mathbf I)\mathbf v+\mathbf q)\bigr).
$$

實作取 $\zeta=\max_p\sum_q|Z_{pq}|$，對 Hermitian 矩陣這是 $\lambda_{\max}$ 的保證上界，
成本 $0.17$ ms，而 $R N=120$ 的 batched eigendecomposition 要 $430$ ms。代價是每步步長較小，
以較多 MM iteration 補回；`--exact_curvature` 可切回 $\lambda_{\max}$ 作等價檢查。另外提供
cyclic coordinate descent 作為同一 subproblem 的第二個 monotone solver，只當 control 用，
不是論文的 solver。

### 13.5 Centralized arm

每個 outer iteration：更新 $(\gamma,\xi)$、對各 AP 依固定順序做 exact block
minimization、再對唯一的全域 $\mathbf v$ 做 MM。兩個 block 都是同一 surrogate 的
non-increasing step，因此 sum rate 單調；`monotonicity_violation` 會把實際觀察到的最大
下降量報出來，而不是假設它是零。這是 local-optimization reference，不是 global upper bound。

### 13.6 Distributed arm

AP $l$ 持有自己的 $\mathbf W_l$ 與自己的 copy $\mathbf v_l$，copy 之間以 edge 形式的
consensus 綁定 $\mathbf t=\sum_l\mathbf A_l\mathbf v_l=\mathbf 0$。Augmented Lagrangian 為

$$
\mathcal L=f(\mathbf W,\{\mathbf v_l\})
+\mathrm{Re}(\boldsymbol\lambda^{H}\mathbf t)
+\frac{\rho}{2}\lVert\mathbf t\rVert^2.
$$

每個 activation 只有一個 AP 動作：扣掉自己在 $(\phi,\psi,\mathbf t)$ 的舊貢獻、更新
$(\gamma,\xi)$、更新 $\mathbf W_l$、以 MM 更新 $\mathbf v_l$、更新 multiplier、加回新貢獻，
再把 token 傳給下一個 AP。在 unit modulus 下 $\rho\lVert\mathbf v_l\rVert^2$ 是常數，因此
penalty 只透過線性項進入 local subproblem：$\mathbf Z\!\leftarrow\!\mathbf Z+\rho\mathbf I$、
$\mathbf q\!\leftarrow\!\mathbf q+\tfrac{\rho}{2}\mathbf A_l^{H}(\mathbf t_l+\boldsymbol\lambda/\rho)$。
一個 sweep 定義為 $L$ 次 activation，報告以 sweep 為單位，這樣 round 數才能和 parallel
方案比較。每個 sample 第一次同時滿足 rate／consensus stopping rule 後即凍結其 phase、
beamformer、multiplier 與 penalty；batch 中較慢的 sample 不得繼續改動已收斂 sample。

RIS 實體上只有一組 configuration，所以 deployed phase 取 copy 平均的 unit-modulus
projection，並同時報告 consensus residual：residual 不夠小時，那個 projection 就不是 AP
真正同意的東西。

### 13.7 與原論文的四處偏離

1. **Association mask**：Huang et al. 與 Xu et al. 都沒有 AP–UE association。本專案在
   power bisection 之前把未服務的 column 歸零，closed form 不變，但這是本專案的延伸而非
   論文的問題設定。
2. **$\rho$**：論文未指定。此處以 per-sample 的
   $\mathrm{median}_l\,\lambda_{\max}(\mathbf Z_l)$ 為尺度，並在與所有 evaluation seed
   互斥的 calibration seed 上依預先登記的規則從 $\{0.05,0.1,0.2,0.4,0.8\}$ 選出後凍結。
   用相對尺度而非常數，是因為 surrogate 的曲率帶著本模擬器的 channel scaling。
3. **Multiplier schedule**：式 (8e) 字面上每次 activation 更新一次 multiplier，等於把
   dual step 放大 $L$ 倍；在 calibration seed 上該排程隨 $\rho$ 增大而對越來越多 sample
   發散。因此主結果採「每個 sweep 更新一次」，兩種排程都報告。
4. **Consensus graph**：論文只寫 $\mathbf A_l$ 由某個無向圖決定。此處預先登記 ring
   $e_l=(l,l+1\bmod L)$，故 $|E|=L$ 且 $\mathbf A_l^{H}\mathbf A_l=2\mathbf I$；它連通、
   與 Algorithm 1 的固定啟動順序一致，也是最省的編碼，不會灌水 signaling ledger。

### 13.8 Signaling

以每個 sample 的 real value 計：`centralized` 上行 $2L(RKMN+KM)$、下行 $2LKM$，共兩個
online round；`distributed` 不上傳任何 CSI，但每次 activation 對環上下一個 AP 送
$2(|E|RN+2K^2)$，最後再把談定的 phase 送一次給 controller。UE→AP 的 CSI acquisition 在
計數邊界之外，且對所有 arm 相同。實際數值、controls 與 limitations 見
[E12 報告](./experiments/e12_model_based_optimization.md)。

在固定 sweep 上限下，各 AP copy 未必完全一致。上面的 ring token 包含所有 edge 的
complex residual $t$；在連通 ring 中，最後一個 AP 可用自己的 copy 與這些差分重建其他
copies，形成並投影它們的平均，再把 **$RN$ 個 phase angles** 傳給 controller。
因此報告的 final AP→CPU $RN$ 個 real values 有可實現的資料路徑；若 token 不包含全部
edge residual，就必須另計上傳各 AP copy，不能沿用本 ledger。E12 的 Python 實作以
集中 tensor 計算同一個平均，並以 AP-locality control 檢查更新步驟；ring token 是
通訊計數的抽象，而非經網路實測的封包實作。
