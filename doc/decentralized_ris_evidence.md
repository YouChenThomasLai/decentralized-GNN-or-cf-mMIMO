# Decentralized RIS 現況與證據報告

## Material Passport

- Origin Date: 2026-09-13
- Last Updated: 2026-09-15（方法定義集中至獨立文件）
- Verification Status: PARTIAL（單一 seed 與固定 topology；500k 附近已達操作性 plateau，但不是跨 seed 的收斂證明）
- Version Label: `decentralized_ris_evidence_v5`
- Scope: `code/decentralized_ris/`，論文設定 $M=2$、$P_{\max}=15$ dBm
- Methods: [Decentralized RIS 方法與通訊介面](./decentralized_ris_methods.md)
- Reference paper: [Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks](<./Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks.pdf>)

本文件合併並取代舊的 baseline requalification 與 training-budget 報告。它只保留仍會影響目前
研究判斷、重現方式或 claim boundary 的資訊；方法、表示法與通訊介面的完整定義統一由
[方法文件](./decentralized_ris_methods.md)維護。舊 checkpoint 的完整逐點表與執行時間軸可由
Git history 及對應 artifacts 回查。

## 1. 目前結論

1. 現有 baseline 可重現論文結果；原始 2k 訓練預算嚴重不足。
2. 延長至 500k iterations 後，centralized／decentralized sum rate 分別由 7.581／7.436
   提升到 23.590／21.399。400–500k validation 的線性斜率分別為
   $-0.022\pm0.032$ 與 $-0.005\pm0.032$ bps/Hz per 10k；在目前 optimizer、learning rate、
   seed 與 topology 下，繼續延長已看不到可辨識的實質收益。因此把 500k 視為後續比較用的
   **操作性 plateau**，而不是模型已全域收斂的統計證明（§4.1）。
3. **centralized−decentralized gap 在 150k 之後反轉收斂**：由 150k 的 3.697（16.8%）
   降到 500k 的 2.190（9.3%）。150k 之前該 gap 是單調擴大的，因此「絕對差持續增加」
   這個先前讀法只在 150k 之前成立。同一段訓練中 decentralized 的增益（+3.030）是
   centralized（+1.524）的兩倍。
4. **這個收斂未出現在 own-only 推論**：在同一組 800 samples 的 paired trajectory 上，
   decentralized−own-only gap 由 150k 的 6.028 變成 500k 的 6.616（$+0.588\pm0.208$，
   $t=2.8$），並未縮小；cen−own 只縮小 7.3%，對照 cen−dec 的 37.5%（§4.2）。因此論文式 (10)
   在目前 centralized-training／local-inference protocol 下，單純延長訓練沒有縮小約
   6.6 bps/Hz 的 dec−own gap；這仍不能分開 cross-AP feedback 的資訊價值與 train–test mismatch。
   另外 own-only 本身沿 trajectory 不穩定：對 centralized 幾乎等價的權重變化會讓它擺動
   1–1.7 bps/Hz，所以 own-only 的數字必須綁定 checkpoint。
5. 500k paired evaluation 中，改善 RIS phase 仍可補回 decentralized modes 的大部分可測
   缺口：對 paper-decentralized 與 own-only 分別為 90.0% 與 65.1%。但此 greedy search 使用
   full CSI，只是 phase headroom reference，不代表 local AP 可部署地取得同樣增益（§5.3）。
6. Own-only 在 greedy phase 後仍有 3.484 bps/Hz residual；它同時包含 local-information
   limitation、centralized-training/local-inference mismatch 與 beamformer quality，目前不能拆開解讀。
7. 下一條主線是 AP-local feasible phase proposal 加 parameter-free consensus，詳見
   [實驗計畫](./ris_action_experiment_plan.md)。

所有結論仍限於 seed 0 與固定 topology。Greedy／continuous optimization 都需要 full CSI，
只是 feasible reference，不是 upper bound，也不是可部署的 decentralized method。

## 2. 記號與固定設定

Input mode 定義見[方法文件 §1](./decentralized_ris_methods.md#notation)。本報告另使用兩個
結果分析量：

| 名稱 | 定義 |
|---|---|
| phase gain | 同一 input mode 下，`greedy phase − GNN phase` |
| local-view gap | 同一 phase 方法下，`centralized − own-only` |

| 符號 | 意義 | 預設值 | 程式名稱 |
|---|---|---:|---|
| $A$ | AP 數 | 5 | `num_of_AP` |
| $R$ | RIS 數 | 4 | `L`、CLI `--L` |
| $N$ | 每 RIS elements | 30 | `N` |
| $M$ | 每 AP antennas | 2 | `M` |
| $K$ | UE 數 | 8 | `K` |
| $Q$ | phase bits | 2 | `num_bits` |

最容易誤讀的是：程式的 `L` 是 RIS 數，不是論文用來表示 AP 數的 $L$。此外，論文圖例的
C／D 是 continuous／discrete phase，不是 centralized／decentralized。`rounded 2-bit` 與
full-CSI greedy 的精確定義見[方法文件](./decentralized_ris_methods.md)。

## 3. Baseline requalification（2026-09-13 稽核摘要）

| 項目 | 結果 | 邊界 |
|---|---|---|
| 原始碼完整性 | 當時的主線程式只有新增 commit `b712cb1`；後續 experiments 使用獨立複本 | 2026-08-17 cleanup 發生在 Git history 前，無法還原 diff |
| Table I／系統模型 | 幾何、通道、關聯門檻、GNN 與訓練設定吻合 | 論文未列噪聲功率 |
| Fig. 6／7 | 本地存檔與論文各點相差 0.03–0.40 bps/Hz | 論文為單一 seed；Fig. 6 的 $M=4$ 使用 seed 7 |
| 現行程式重跑 | 八個 headline 指標最大偏差 0.16 bps/Hz | 跨 GPU，非 bit-exact |
| Fast evaluator | 與原 `cal_loss` 誤差不超過 $9.5\times10^{-7}$ | 僅驗證現有固定設定 |

一個必須保留的模型風險是噪聲設定：通道先放大 $10^7$，真正進入 SINR 的
`sigma = (2e-2)**2` 寫死於 `utils_return_indivial_rates.py`；宣告的 `background_noise`
沒有被使用。任何跨 stage 或跨 simulator 比較都必須先對齊此設定。

統計單位也不是 individual sample。`gen_location` 每個 batch 只產生一次 UE locations，batch
內 8 個 samples 共用位置，因此有效獨立單位是 batch；本文的 standard error 皆按 batch
cluster 計算。

## 4. 訓練預算結果

下表是 3,200-sample final evaluation；所有 runs 使用 seed 0、batch size 8、Adam
$10^{-4}$、weight decay $10^{-6}$。

| 方法 | 2k | 40k | 100k | 150k | 500k |
|---|---:|---:|---:|---:|---:|
| centralized，GNN phase | 7.581 | 16.101 | 20.415 | 22.066 | **23.590** |
| centralized，rounded 2-bit | 6.996 | 14.954 | 18.961 | 20.587 | 22.035 |
| centralized，random phase | 3.162 | 6.182 | 6.396 | 6.655 | 6.672 |
| decentralized，GNN phase | 7.436 | 13.746 | 17.034 | 18.369 | **21.399** |
| decentralized，rounded 2-bit | 6.837 | 12.841 | 15.811 | 17.122 | 19.876 |
| decentralized，random phase | 3.090 | 5.852 | 5.869 | 6.094 | 6.116 |

`rounded 2-bit` 的 alphabet 與量化 protocol 見
[方法文件 §7](./decentralized_ris_methods.md#two-bit)。

| 區間 | 長度 | Centralized 增益／10k | Decentralized 增益／10k |
|---|---:|---:|---:|
| 2k→40k | 38k | +2.242 | +1.661 |
| 40k→100k | 60k | +0.719 | +0.548 |
| 100k→150k | 50k | +0.330 | +0.267 |
| 150k→500k | 350k | **+0.044** | **+0.087** |

150k→500k 是四段中唯一 decentralized 增益大於 centralized 的一段（+3.030 對 +1.524）。
兩條 random-phase 列在同一段幾乎不動（+0.017 與 +0.022），與先前各段一致：延長訓練的
收益來自 $\theta$ 與 $W$ 的協同，而不是 $W$ 本身。

### 4.1 Plateau 判定（500k）

以下對 400-sample validation 每 2,000 iterations 做線性回歸。多個巢狀窗口是事後的
trajectory summary；$|t|<2$ 只作為停止訓練的操作性訊號，不當作獨立重複實驗的顯著性檢定：

| 窗口 | $n$ | centralized 斜率 | $t$ | decentralized 斜率 | $t$ |
|---|---:|---:|---:|---:|---:|
| 152–500k | 175 | $+0.043\pm0.005$ | 8.4 | $+0.066\pm0.005$ | 12.7 |
| 250–500k | 126 | $+0.032\pm0.008$ | 3.8 | $+0.054\pm0.009$ | 6.2 |
| 300–500k | 101 | $+0.023\pm0.012$ | 2.0 | $+0.054\pm0.012$ | 4.5 |
| 350–500k | 76 | $+0.016\pm0.017$ | 0.9 | $+0.038\pm0.017$ | **2.2** |
| 400–500k | 51 | $-0.022\pm0.032$ | −0.7 | $-0.005\pm0.032$ | −0.2 |
| 450–500k | 26 | $-0.028\pm0.077$ | −0.4 | $+0.023\pm0.075$ | 0.3 |

150k 報告中 122–150k 窗口只有 15 個點、$\mathrm{se}=0.195$，當時無法區分低斜率與量測雜訊；
此處 400–500k 有 51 個點、$\mathrm{se}=0.032$，centralized 斜率的區間已縮小到
$[-0.086,+0.042]$ bps/Hz per 10k。短期自相關修正後 standard error 約為 0.035，判讀不變。
因此，**在現行訓練設定下，centralized 約於 350–400k 後進入操作性 plateau**。

decentralized 的 350–500k 斜率仍為正（$+0.038\pm0.017$），但只略高於操作門檻；最後
100k 已為 $-0.005\pm0.032$。較保守的解讀是 decentralized 可能比 centralized 晚進入平坦區，
但目前單一 training trajectory 不足以精確定位兩者的飽和 iteration。

Centralized−decentralized gap 由 2k 的 0.146（1.9%）增至 40k 的 2.354（14.6%）、100k
的 3.381（16.6%）與 150k 的 3.697（16.8%），**但在 500k 收斂回 2.190（9.3%）**。
先前「絕對差持續增加、相對差在 16–17% 趨平」的讀法只適用到 150k：把訓練推到 plateau 之後，
decentralized 追回了 150k 時約四成的差距。這改變了 gap 的歸因——150k 的 gap 有相當一部分
是 decentralized 尚未訓練到位，而不是 local-input visibility 的結構性限制。

Post-hoc 2-bit quantization loss 仍隨模型變強而增大，且 decentralized 增加得更快：
centralized 1.454（100k）→ 1.479（150k）→ 1.554（500k）；
decentralized 1.223 → 1.247 → **1.524**。兩者在 500k 幾乎相同。

### 4.2 三個 input mode 沿 trajectory 的變化（2026-09-15）

§4／§4.1 的 trajectory 只含 centralized 與 paper-decentralized，own-only 只有 100k 一個點。
本節在 11 個 checkpoint（2k、40k、100k、150k，以及 500k run 每 50k 的中途 checkpoint 與
終點）上各以**同一組 800 samples** 同時評估三個 input mode。三個 mode 與 11 個 checkpoint
共用同一批 channel realization——11 個 cell 的 `visible AP-UE nodes per AP` 完全相同
（17.576／4.790）——所以任意兩個 cell、任意兩個 mode 之間都可以做 paired 比較。

| iterations | centralized | paper-dec. | own-only | cen−dec | dec−own | cen−own |
|---:|---:|---:|---:|---:|---:|---:|
| 2k | 7.325 | 7.199 | 6.736 | 0.126 | 0.464 | 0.590 |
| 40k | 15.657 | 13.279 | 8.815 | 2.378 | 4.465 | 6.842 |
| 100k | 20.325 | 17.126 | 11.497 | 3.199 | 5.629 | 8.828 |
| 150k | 21.959 | 18.552 | 12.523 | **3.407** | 6.028 | **9.435** |
| 200k | 22.842 | 20.091 | 13.529 | 2.751 | 6.563 | 9.313 |
| 250k | 23.240 | 20.457 | 13.511 | 2.783 | 6.946 | 9.729 |
| 300k | 23.313 | 20.673 | 14.324 | 2.640 | 6.349 | 8.989 |
| 350k | 23.274 | 20.823 | 14.233 | 2.450 | 6.590 | 9.040 |
| 400k | 23.717 | 21.525 | 14.114 | 2.192 | 7.411 | 9.603 |
| 450k | 23.870 | 21.552 | 15.801 | 2.318 | 5.751 | 8.069 |
| 500k | 23.549 | 21.419 | 14.803 | **2.130** | 6.616 | 8.746 |

**「paired」的定義。** 每個 batch 只抽一次 UE locations 與 channel，三個 mode 都在這同一組
channel 上前傳（`decentralized_batch(..., regenerate_channels=False)` 重用同一批 channel），
所以 gap 是逐 sample 相減後再平均，不是兩個獨立平均值相減。這一點決定了表的精度：三個 mode
各自的 clustered SE 由 2k 的 0.17 增到後段的 0.34–0.48（centralized 最大、own-only 最小），
因為 channel realization 本身的變異就有這麼大；但 gap 欄的 clustered SE 只有 0.035（2k）到
0.13–0.29（後段），因為兩個 mode 在同一個 channel 上同方向好壞（$\rho=0.93$–$0.98$），相減
時共同的 channel 變異被消掉。以 2k 的 cen−dec 為例，paired SE 0.035 對 unpaired 的 0.241，
精度差 6.9 倍：paired 下 $t=3.6$，unpaired 下只有 $t=0.5$，也就是不 paired 根本量不出那個
0.126 的差。同一個理由也用在下面的相鄰 checkpoint 比較——11 個 checkpoint 共用同一批 channel，
所以 checkpoint 之間也能相減。

相鄰 checkpoint 之間的 paired 變化（continuous phase，每格為 `差值±clustered SE`，t 為差值
除以自身 SE）：

| 窗口 | centralized | paper-decentralized | own-only |
|---|---:|---:|---:|
| 150k→200k | $+0.883\pm0.103$（8.6） | $+1.539\pm0.120$（12.8） | $+1.005\pm0.121$（8.3） |
| 200k→250k | $+0.398\pm0.078$（5.1） | $+0.366\pm0.130$（2.8） | $-0.017\pm0.137$（−0.1） |
| 250k→300k | $+0.073\pm0.081$（0.9） | $+0.216\pm0.121$（1.8） | $+0.813\pm0.132$（6.1） |
| 300k→350k | $-0.039\pm0.085$（−0.5） | $+0.150\pm0.124$（1.2） | $-0.091\pm0.152$（−0.6） |
| 350k→400k | $+0.443\pm0.081$（5.5） | $+0.701\pm0.121$（5.8） | $-0.120\pm0.143$（−0.8） |
| 400k→450k | $+0.154\pm0.092$（1.7） | $+0.028\pm0.108$（0.3） | $+1.687\pm0.155$（**10.9**） |
| 450k→500k | $-0.321\pm0.078$（−4.1） | $-0.133\pm0.118$（−1.1） | $-0.998\pm0.150$（**−6.6**） |

本節的六項結果：

1. **迴歸檢查通過。** 40k 的三個數字（15.65715／13.27942／8.81470）與重構前
   `ablation_local_csi.py` 存檔的 800-sample 結果逐位相同，因此重構後的
   `experiments/local_csi.py` 在另一張 GPU 上重現了舊路徑。2k 的 7.325 與舊表的 7.409
   差 0.084，是因為本節用 `results_repro_check` 的 2k checkpoint，而舊表用 vary_M sweep
   的 2k checkpoint（兩個檔案的 md5 不同）。以舊表那個 checkpoint 重跑得到
   7.40885／7.24902／6.84066，同樣逐位對上舊值，因此這 0.084 是 checkpoint 之差，
   不是實作或 GPU 之差。

2. **cen−dec gap 的收斂獨立確認。** 150k 的 3.407 降到 500k 的 2.130，paired change
   $-1.277\pm0.153$（$t=-8.3$，$-37.5\%$）。與 §4.1 用 3,200-sample final evaluation 得到的
   3.697→2.190 同形，兩套獨立的評估資料給出同一結論。

3. **dec−own gap 沒有跟著收斂，反而略為擴大。** 150k 的 6.028 → 500k 的 6.616，paired
   change $+0.588\pm0.208$（$t=2.8$，$+9.7\%$）。同期 cen−own 只縮小 $7.3\%$
   （$-0.690\pm0.182$，$t=-3.8$），遠小於 cen−dec 的 $37.5\%$。**延長訓練補回的是
   「decentralized 相對 centralized」的差，沒有補回「own-only 相對論文式 (10)」的差。**
   這使 §1 第 3 點的重新歸因只適用於 cen−dec：在目前訓練／推論 protocol 下，500k 時
   dec−own 仍差約 6.6 bps/Hz。這個數字同時包含 feedback 資訊差與 train–test mismatch，
   不能全數歸因於 cross-AP feedback 的必要價值。

4. **own-only 沿 trajectory 不穩定。** 400k→450k 的 paired 變化為 $+1.687\pm0.155$
   （$t=10.9$），450k→500k 為 $-0.998\pm0.150$（$t=-6.6$），而同兩個窗口 centralized 只動
   $+0.154$ 與 $-0.321$。也就是 centralized 已進入操作性 plateau、對 centralized 目標幾乎等價的權重
   變化，仍會讓 own-only 擺動約 1–1.7 bps/Hz。**任何 own-only claim 都必須指明取自哪一個
   checkpoint，或跨數個 checkpoint 平均**；單一終點的 own-only 數字不是穩定量。這也解釋
   為何 own-only 在 200k 之後的「上升」不可直接宣告：400k→500k 的 $+0.690\pm0.140$ 幾乎
   全由 450k 那一個尖峰貢獻。

5. **centralized 的 plateau 位置與 §4.1 一致。** 350k→500k 的 paired 變化仍為
   $+0.275\pm0.081$（$t=3.4$），但 400k→500k 轉為 $-0.168\pm0.083$（$t=-2.0$）。兩套方法
   都把飽和點放在約 350–400k。

6. **own-only 的 2-bit quantization loss 明顯小於另兩者。** 500k 時 cont−round 的 paired
   差為 centralized $1.618\pm0.061$、paper-decentralized $1.586\pm0.065$、own-only
   $0.987\pm0.043$；三者都隨訓練增大（own-only：100k $0.654$ → 150k $0.779$ → 500k
   $0.987$）。own-only 的 phase 本來就較差，可被 quantization 破壞的結構較少，因此不能把
   較小的 quantization loss 讀成 own-only 對 discrete phase 較友善。

圖在 `artifacts/decentralized_ris/mode_trajectory/`：`mode_trajectory.png` 為 log 橫軸
（看得清 2k→100k 的上升段），`mode_trajectory_linear.png` 為線性橫軸（看得清 200k 之後的
plateau）。左圖為三個 mode 的 sum rate（實線 continuous phase、虛線 post-hoc 2-bit），
右圖為三個 paired gap；為保持圖面清楚，兩圖只畫平均值，不顯示 batch-clustered SE。

**邊界。** own-only 在本節仍是「centralized 訓練、own-only input 推論」，因此第 3、4 點的
缺口同時含 local-information limitation 與 centralized-training／local-inference mismatch
（§1 第 6 點）。本節只證明這個合併缺口不隨訓練縮小，沒有把兩者拆開；拆開仍需 §9 第 3 點的
own-only retraining。所有數字限於 seed 0 與固定 topology。

## 5. 可達參考點與缺口歸因

本節只報告 achievable-reference 結果；full-CSI greedy 的演算法、資訊使用與可解讀邊界統一見
[方法文件 §8](./decentralized_ris_methods.md#full-csi-greedy)。

### 5.1 Joint feasible reference

320-sample full-CSI optimization 的結果如下。`joint-rand` 是非凸問題中找到的 feasible point，
不是全域 optimum。

| 方法 | 2k | 40k |
|---|---:|---:|
| centralized GNN | 7.454 | 15.544 |
| 固定 $W$、continuous phase | 16.205 | 24.630 |
| joint $(W,\theta)$，GNN initialization | 26.327 | 28.704 |
| joint $(W,\theta)$，random initialization | **29.536** | **29.536** |

100k／150k 尚未重跑此項，因此不可把 40k 的 52.6% 比例沿用到較新 checkpoint。

### 5.2 100k phase 與 local-input decomposition

以下三列使用相同 320 samples，並各自固定該 input mode 產生的 $W$：

| Fixed $W$／initial phase source | GNN phase | Full-CSI greedy 2-bit | Phase gain |
|---|---:|---:|---:|
| centralized GNN | 20.142 | 23.984 | +3.843 |
| paper-decentralized GNN | 16.876 | 23.764 | +6.888 |
| own-only GNN | 11.078 | 20.726 | +9.648 |

以 centralized greedy 23.984 為共同參考：

| 起點 | 總缺口 | 可由自身 phase search 補回 | Greedy 後 residual |
|---|---:|---:|---:|
| paper-decentralized | 7.108 | 6.888（96.9%） | 0.220（3.1%） |
| own-only | 12.906 | 9.648（74.8%） | 3.258（25.2%） |

Full-CSI greedy 把 own-only 的 local-view gap 由 9.063 降至 3.258，消除觀察到缺口的 64.1%。
這只證明 phase headroom 存在；它沒有證明 local AP 能在部署時取得相同增益。

100k 的 800-sample centralized comparison 另得到 GNN 20.325、2-bit greedy 24.127；GNN／greedy
由 2k 的 52.7%、40k 的 68.9% 升至 84.2%。Centralized phase gap 隨訓練縮小，但 local modes
的 gap 仍大。

### 5.3 500k phase 與 local-input decomposition

500k checkpoint 使用 320 samples／40 個 batch clusters 重跑相同的 2-bit greedy procedure：

| Fixed $W$／initial phase source | GNN phase | Rounded 2-bit | Full-CSI greedy 2-bit | Greedy phase gain |
|---|---:|---:|---:|---:|
| centralized GNN | 23.526 | 21.958 | 24.605 | $+1.079\pm0.113$ |
| paper-decentralized GNN | 21.400 | 19.831 | 24.285 | $+2.885\pm0.218$ |
| own-only GNN | 14.607 | 13.636 | 21.121 | $+6.514\pm0.209$ |

以 centralized greedy 24.605 為共同 reference，paper-decentralized 的總缺口為 3.205，
其中自身 greedy phase 補回 2.885（90.0%），殘差為 $0.320\pm0.109$。Own-only 的總缺口為
9.998，其中補回 6.514（65.1%），殘差為 $3.484\pm0.323$。相較 100k 的 96.9%／74.8%，
phase 仍是主要可測 headroom，但隨 baseline 成熟，占比已下降；剩餘差距仍混合 local-information
limitation、centralized-training/local-inference mismatch 與 beamformer quality。

## 6. Visibility 與 signaling overhead

| Input mode | 每 AP 可見 AP–UE nodes | 全域比例 |
|---|---:|---:|
| centralized | 40 | 100% |
| paper-decentralized | 17.576 | 43.9% |
| own-only | 4.790 | 12.0% |

論文式 (10) 使 UE→AP feedback 由 own-only 的 11,879 增至 43,588 個 real coefficients，約
3.67 倍。使用一致的 CSI 計量後：

| 訊令腿 | Centralized | Paper-decentralized | Own-only |
|---|---:|---:|---:|
| UE→AP | 11,879 | 43,588 | 11,879 |
| AP→CPU | 11,879 | 2,400 | 2,400 |
| CPU→AP/RIS | 336 | 240 | 240 |
| Fronthaul 小計 | 12,215 | **2,640** | **2,640** |
| 全部合計 | 24,094 | 46,228 | **14,519** |

因此原方法的 fronthaul 優勢成立，但若把未列入原 Table II 的 air-interface feedback 一併計入，
paper-decentralized 的總量反而較高。後續方法需同時報 rate 與三段 payload。

## 7. 目前研究決策

| 軸 | 決策 |
|---|---|
| RIS phase representation | 主線；500k 的 greedy phase gain 仍有 1.1–6.5 bps/Hz（§5.3） |
| Local input path | dec−own gap 未隨延長訓練縮小（§4.2），但尚不能稱為結構性限制；仍需 own-only retraining 分離 information limitation 與 mismatch |
| CPU phase aggregation | 改成 AP-local feasible action + parameter-free consensus |
| 2-bit quantization | 併入 phase design，不獨立成題 |
| 延長訓練 | **已結案**：500k 在現行設定下達操作性 plateau（§4.1）。除非方法或 optimizer 改變，不再延長 baseline；後續比較以 500k checkpoint 為基準 |
| Multi-seed | 方法凍結前不執行 |
| own-only 的量測方式 | 不以單一 checkpoint 宣告：它在 50k 間隔上擺動 1–1.7 bps/Hz（§4.2 第 4 點），必須綁定 checkpoint 或跨數個 checkpoint 報告 |

## 8. Reproducibility 與 artifacts

主要程式：

```bash
# Baseline
python trainer_2.py --M 2 --N 30 --L 4 --K 8 --pmax_dbm 15 \
  --batch_size 8 --runs 1 --device cuda:0 --out_dir results_repro_check

# Extend from a resumable checkpoint
python train_fast.py --resume <checkpoint> --n_iter <additional_iterations> \
  --seed 0 --log_eval_interval 2000 --save_every 5000 \
  --test_sample_val 400 --test_sample_final 3200 --device cuda:0 \
  --out_dir <new_output_dir>

# References
python discrete_cd_baseline.py --ckpt <checkpoint> --samples 800 --rounds 4
python continuous_ceiling.py --ckpt <checkpoint> --samples 320 --steps 2000 --restarts 3
python ablation_local_csi.py --ckpt <checkpoint> --samples 320 --greedy

# Three input modes versus training steps (§4.2), from code/decentralized_ris/.
# Stage the trajectory checkpoints as trajectory_ckpts/iter<iteration>.pt first;
# the driver skips cells that already have a summary.json, so it is resumable.
CKPT_DIR=trajectory_ckpts SAMPLES=800 DEVICE=cuda:0 \
  OUT_DIR=../../artifacts/decentralized_ris/mode_trajectory \
  ./scripts/mode_trajectory.sh
python ../plot_mode_trajectory.py \
  --trajectory ../../artifacts/decentralized_ris/mode_trajectory/trajectory.json \
  --output-base ../../artifacts/decentralized_ris/mode_trajectory/mode_trajectory
```

| Evidence | Location |
|---|---|
| 2k／40k／100k／150k training | `artifacts/decentralized_ris/results_repro_check/`、`results_long_training/`、`results_extend_100k/`、`results_extend_150k/` |
| **500k training 與 final evaluation** | `artifacts/decentralized_ris/results_extend_500k/M2_N30_L4_K8_P15.0_iter350000_seed0/run0/` |
| Greedy references | `artifacts/decentralized_ris/results_discrete_cd/` |
| Continuous references | `artifacts/decentralized_ris/results_continuous_ceiling/` |
| Local-CSI ablations | `artifacts/decentralized_ris/results_local_csi_ablation/` |
| **500k R0／R0c／R1-shared／R1 AP–RIS magnitude paired evaluation** | `artifacts/decentralized_ris/evaluation/baseline_500k_*_seed18*` |
| **500k local-CSI＋greedy decomposition（§5.3）** | `artifacts/decentralized_ris/results_local_csi_ablation/trained500k_with_greedy/` |
| **三 input mode trajectory（§4.2）** | `artifacts/decentralized_ris/mode_trajectory/`：11 個 `iter*/` cell、`trajectory.txt`、`trajectory.json`、`mode_trajectory{,_linear}.{pdf,png}`、`mode_trajectory.log` |
| **MRC-style energy proxy 診斷** | `artifacts/decentralized_ris/mrc_proxy_diagnostic/`（本機 WSL2／RTX 5060 Laptop GPU） |
| **energy-weighted consensus 驗證（4 個效能設定、2 種權重情境）** | `artifacts/decentralized_ris/energy_consensus_verification/`：`report.md`、`verification_plan.md`、`summary_tables.md`、`{main,pmax5,pmax25,assoc30}_seed20260922.*`、`reproduction_check.json`、`information_asymmetry.json`（本機 WSL2／RTX 5060 Laptop GPU）；`main`／`pmax5`／`pmax25` 共用逐位元相同的 proposals、$s$、$E$，只有 `assoc30` 改變權重結構 |
| 500k 中途 checkpoints（每 50k） | 僅在 lab301-5090：`~/ThomasLai/code/decentralized_ris/results_extend_500k/.../models/resume_{200,250,300,350,400,450}k.pt` |
| 500k run log、tfevents、`resumable_latest.pt` | 僅在 lab301-5090：`extend_500k.log` 與同目錄 `logs/` |

500k continuation 於 2026-09-14 21:28:22 在 `lab301-5090-tailscale`（單張 RTX 5090）的
tmux session `thomaslai_extend_500k` 起跑，2026-09-15 12:37:17 結束：+350,000 iterations、
wall time 15.15 小時、有效 385.1 it/min。它由 150k 的 `resumable_final.pt` warm start
（log 記為 `optimizer state restored`），只使用 seed 0，175 個 validation 點無缺漏。

換機的理由是 throughput：該機實測純訓練 412 it/min、有效 385 it/min，為 meow2 RTX 4090 的
81–97 it/min 的 4–4.7 倍，且無 meow2 的每日 24 GPU-hour quota——quota 正是把先前每次續訓
壓在 50,000–60,000 iterations 的原因。事前以 benchmark 估的 397 it/min 偏樂觀 3.4%
（ETA 差 31 分鐘）；長 run 的 throughput 略低於短 benchmark，下次估算應留這個餘裕。
該機 `nvidia-smi` 因 NVML 595.91 對上 kernel module 595.84 而不可用，但 CUDA 正常，
`verify_equivalence` 在該機通過（`outputs_match: true`）。

§4.2 的 trajectory 於 2026-09-15 16:57 在同一台 `lab301-5090-tailscale` 的 tmux session
`thomaslai_mode_traj` 執行，11 個 checkpoint 共約 4 分鐘（800 samples、3 個 mode、無 greedy）。
2k／40k／100k／150k 的 checkpoint 由本機上傳，200k–450k 取自該機留存的 `resume_*k.pt`，
500k 取自 `model_final_run0.pt`；每個 bundle 的 `iteration` 欄位都已核對過與檔名一致。
`experiments/local_csi.py` 的 `[verify]` 在 11 個 cell 全部回報
`max|dW|=max|dtheta|=0`，即 `include_cross_ap_csi=True` 路徑與正常
`model(training=False)` 完全相同。

**Checkpoint 相容性**：500k checkpoint 由重構前的 `train_fast.py`／`trainer_2.py`／
`model_2.py` 產生。在重構後的 `model.py` 上以 `strict=False` 載入時 missing keys 為 0、
所有 live parameter 形狀相符（1,700,745 個參數），多出的 48 個 key 只有已被重構移除的
`fc` 與 `edge_update` 兩個 submodule——即先前已知從未被呼叫的 dead weights。因此該
checkpoint 在新程式下仍可用，但載入時必須 `strict=False`。

## 9. 下一次更新條件

只在下列事件發生時更新本文件：

1. ~~500k run 完成：補 final evaluation、長窗口 slope 與是否 plateau。~~ 已完成，見 §4 與 §4.1。
2. ~~在 500k checkpoint 重跑 greedy／local-CSI reference。~~ 已完成，見 §5.3。Continuous
   joint reference 尚未重跑；只有在需要更新 joint $(W,\theta)$ headroom 時才執行。
3. Own-only retraining 能分離 information limitation 與 train–test mismatch。§4.2 第 3 點
   已把「合併缺口不隨訓練收斂」確立為前提，因此這一項現在是要回答缺口屬於哪一半，
   而不是要回答缺口是否存在。
4. 主線 representation 產生足以改變研究決策的新結果。

目前缺口：500k 尚無 continuous joint reference，100k／150k 也沒有新的 continuous joint
reference；§5.3 應依[方法文件 §8](./decentralized_ris_methods.md#full-csi-greedy)
的邊界解讀，只能量測 phase headroom。
