# Decentralized RIS 方法與通訊介面

## Material Passport

- Origin Date: 2026-09-15
- Last Updated: 2026-09-15（補入論文式 (9)、(10)、(26)–(28) 與程式對照）
- Verification Status: IMPLEMENTED（定義已與 `model.py`、`variants.py` 及 action-interface tests 對照）
- Version Label: `decentralized_ris_methods_v2`
- Scope: R0、R0c、R1-Shared、R1 AP–RIS magnitude weighting，以及 2-bit／full-CSI greedy evaluation
- Reference paper: [Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks](<./Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks.pdf>)

本文件是上述方法定義的單一來源。實驗結果、統計判讀與研究決策分別留在
[現況與證據報告](./decentralized_ris_evidence.md)和
[action screening 報告](./ris_action_screening_report.md)，不在此重複。

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

因此 $N$ 個 phase 在目前張量介面中占 $2N$ 個 real values，而不是 $N$ 個。這不是理論上的
最小編碼量；實際系統可改傳 $N$ 個角度，2-bit phase 也可改傳 $N$ 個 2-bit indices。

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

## 3. 四種 action interface

四種介面採用相同的 GNN state 定義、論文式 (26) readout architecture 與 learned
$\mathbf W_{\rm reduce}$；零訓練 control 才進一步共用同一組 checkpoint weights。它們的差別是
AP 傳送的表示和 projection 相對於 aggregation 的位置：

| 方法 | AP-side 流程 | 每個 AP–RIS 傳送 | 聚合端流程 |
|---|---|---:|---|
| R0 | $\mathbf s_{l,r}^{(D)},\mathbf e_l\to\mathbf v_{l,r}$ | $4N$ phase feature | 論文式 (27) 的 learned reduction 與 normalization |
| R0c | $\mathbf v_{l,r}\to\mathbf z_{l,r}$ | $2N$ raw logits | 先加總 $\mathbf z$，再 normalization |
| R1-Shared | $\mathbf v_{l,r}\to\mathbf z_{l,r}\to\mathbf p_{l,r}$ | $2N$ unit proposals | 等權 circular consensus |
| R1 AP–RIS magnitude weighting | $\mathbf v_{l,r}\to\mathbf z_{l,r}\to(\mathbf p_{l,r},s_{l,r})$ | $2N+1$ | importance-weighted circular consensus |

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

## 5. Payload 與 decentralization 用語

以 Cartesian real values 計數：

| 方法 | 每個 AP–RIS pair | 全部 AP→aggregator payload | 聚合端 trainable phase component |
|---|---:|---:|---:|
| R0 | $4N$ | $4RNL$ | 有 |
| R0c | $2N$ | $2RNL$ | 無 |
| R1-Shared | $2N$ | $2RNL$ | 無 |
| R1 AP–RIS magnitude weighting | $2N+1$ | $RL(2N+1)$ | 無 |

在目前 $N=30$ 的設定中，AP–RIS magnitude weighting 是每個 AP–RIS pair 傳 61 個 real values：60 個 Cartesian
proposal coordinates 加一個 scalar。聚合端對 weights 的共同尺度不敏感，實作會先在 AP 維度
正規化，再計算加權 resultant 與最終 unit projection。

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
