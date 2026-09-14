# Stage 0 訓練預算與可達水準歸因報告

## Material Passport

- Origin Date: 2026-09-13
- Last Updated: 2026-09-14
- Verification Status: PARTIAL（單一 seed 的證據已完成；100k 延長訓練執行中，training curve 仍未飽和）
- Version Label: `stage0_training_budget_v2`
- 前置文件：`doc/stage0_baseline_requalification_report.md`（v2）
- Scope：`code/stage0/`，論文設定 $M=2$、$P_{\max}=15$ dBm

前一份 baseline requalification 報告確認了三件事：stage 0 原始碼未被改動、實驗設定符合論文 Table I，以及現有程式可重現論文 Fig. 6／7。本報告接著回答尚未解決的問題：**GNN 與可達參考點之間的效能差距，有多少來自訓練預算不足？**

主要結果是：將訓練預算由 2,000 iterations 增加至 40,000 iterations 後，GNN 的 sum rate 大幅提升，但訓練曲線仍未飽和。較長訓練也改變了 centralized、decentralized 與 RIS phase quality 之間的相對重要性，因此本報告更正前一份報告的三項方向判斷，詳見 §6。

## 0. 名詞與比較方式

若未另外說明，本文的效能數字均為 sum rate，單位是 bps/Hz。

| 名稱 | 定義 | 對應 artifact |
|---|---|---|
| 2k model | 使用論文預算訓練 2,000 iterations | `results_repro_check/` |
| 40k model | 延長訓練至 40,000 iterations | `results_long_training/` |
| centralized | 一次輸入所有可用 CSI | `cen_*`、`centralized*` |
| decentralized | 使用論文式 (10) 的 local CSI，包含額外跨 AP 回授 | `dec_paper_*`、`decentralized*` |
| decentralized-own | 每個 AP 只看自己服務鏈路的 CSI | `dec_own_only_*` |
| greedy | 固定 GNN 輸出的 $W$，使用全域真實 CSI 在 2-bit phase grid 上做 coordinate descent | `*_greedy`、`*_cd` |

`greedy` 是逐 sample 計算的 full-CSI oracle reference，不是可直接部署的 decentralized 方法。它的用途是量測：在固定 $W$ 後，只改善 RIS phase 最少還能增加多少效能。

本文使用兩種不同的差值，兩者不應混為一談：

- **phase gain**：同一模式內，`greedy phase − GNN phase`。
- **local-view gap**：同一 phase 方法下，`centralized − decentralized-own`。

可見度百分比表示 AP 能觀察到的 AP–UE nodes 比例。它是輸入資訊量，不是 SINR，也不是 sum-rate loss。

## 1. 狀態總表

| 查核項目 | Verdict | 目前可支持的結論 | Claim boundary |
|---|---|---|---|
| 訓練瓶頸定位 | PASS | 95% 的時間用於 forward/backward，而非資料產生 | 單一機器的 profile |
| 向量化等價性 | PASS | 輸出差 $\le 2.5\times10^{-6}$，梯度差 $\le 3.1\times10^{-5}$ | float32 捨入等級 |
| 訓練預算效應 | PASS | centralized 由 7.581 升至 16.101（×2.12） | 單一 seed；40k 尚未飽和 |
| 相對可達水準 | PASS | GNN 由 joint reference 的 25.2% 升至 52.6% | 同上 |
| centralized–decentralized gap | PARTIAL | 3200-sample final evaluation 中由 0.146 升至 2.354 | 單一 seed；量測點本身尚未飽和 |
| local view 與 phase quality | PARTIAL | GNN phase 下的 local-view gap 為 7.053；full-CSI greedy 將它降至 3.230 | 單一 seed；greedy 不是 local 方法 |
| 100k 延長訓練 | IN FLIGHT | 合計 90,000 iterations 時 validation 仍以每 10k 約 +0.85（centralized）上升 | 見 §8 |

## 2. 訓練瓶頸

在本機 RTX 5060 Laptop 上，batch size 8 的單次訓練 profile 如下：

| 階段 | 每次迭代 | 佔比 |
|---|---:|---:|
| 資料產生 | 0.110 s | 4.7% |
| forward + loss | 0.849 s | 36.5% |
| backward + step | 1.368 s | 58.8% |
| 合計 | 2.327 s | — |

模型只有 2,206,479 個參數，輸入也只有 batch 8、40 個 nodes，但每次迭代仍需 2.3 秒。主要原因不是模型規模，而是 `model_2.py` 建立了大量微小運算：

- `initial_layer.forward` 與 `node_update_layer.forward` 將 batch 維度寫死為 1，因此 `node_update.forward` 必須逐 sample 執行。
- `node_update_layer.forward` 內的 `for k in range(self.K)` 會對 40 個 AP–UE nodes 逐一執行小矩陣乘法。

這兩層迴圈使每次迭代建立包含數千個小型 operations 的 autograd graph。換言之，論文的 2,000-iteration 訓練預算主要受實作效率限制，而不是問題本身需要如此高的計算成本。

## 3. 向量化實作與等價性

`code/stage0/fast_forward.py` 將上述運算改寫為 batch operations。它不新增任何參數，而是直接使用原本 `node_update` 的 weights，因此既有 checkpoint 可直接載入，`model_2.py` 也不需修改。

最重要的改寫是式 (14) 的 element-wise max aggregation，其中 $A=1-I$。原實作對每個 node 分別計算「排除自己後的最大值」；向量化版本則先找出每個 feature 的前兩大值。若全域最大值不是由目前 node 提供，就使用全域最大值；否則使用第二大值。這項改寫與原運算精確等價，而非近似。

### 3.1 等價性驗證

| 檢查 | centralized | decentralized |
|---|---:|---:|
| $\max\lvert\Delta W\rvert$ | $1.5\times10^{-8}$ | $1.4\times10^{-7}$ |
| $\max\lvert\Delta\theta\rvert$ | $1.5\times10^{-6}$ | $2.5\times10^{-6}$ |
| $\max\lvert\Delta(\partial L/\partial w)\rvert$ | $1.5\times10^{-5}$ | — |

等價性分別在已訓練 checkpoint 與隨機初始化模型上驗證，並在遠端機器 meow2 重驗。除了 forward output，也必須比較 gradient，因為訓練等價性取決於 backward path。`train_fast.py` 每次啟動都會執行這項檢查並將結果寫入 log；若檢查失敗，訓練會直接中止。

decentralized path 另外使用 `ablation_local_csi.decentralized_forward` 作為 reference。該函式先前已驗證與原實作逐位元相同。

### 3.2 加速結果

| 路徑 | 原實作 | 向量化 | 加速 |
|---|---:|---:|---:|
| training step（batch 8） | 3.709 s | 0.426 s | 8.7× |
| training step（batch 32） | 13.966 s | 0.823 s | 17.0× |
| decentralized inference（batch 8） | 5.229 s | 0.024 s | **216×** |

decentralized inference 的加速最大，因為原實作同時包含逐 AP、逐 sample 與逐 node 三層迴圈。向量化後，3,200 samples 的 final evaluation 由約 28 分鐘縮短至約 8 秒。

## 4. 訓練預算的影響

向量化後重新訓練時，除了 `n_iter` 與 validation frequency，其餘設定均維持論文配置：batch size 8、Adam learning rate $10^{-4}$、weight decay $10^{-6}$、seed 0。下表的 final evaluation 均使用 3,200 samples。

| 方法 | 2,000 iterations | 40,000 iterations | 倍數 |
|---|---:|---:|---:|
| centralized（GNN phase） | 7.581 | **16.101** | ×2.12 |
| centralized（rounded 2-bit phase） | 6.996 | 14.954 | ×2.14 |
| centralized（random phase） | 3.162 | 6.182 | ×1.96 |
| decentralized（GNN phase） | 7.436 | **13.746** | ×1.85 |
| decentralized（rounded 2-bit phase） | 6.837 | 12.841 | ×1.88 |
| decentralized（random phase） | 3.090 | 5.852 | ×1.89 |

增加訓練預算使所有指標明顯提升。然而，40,000 iterations 仍不是收斂點。Training sum rate 的 trailing-500 average 如下：

| iteration | 5,629 | 10,966 | 16,145 | 21,725 | 27,228 | 32,741 | 38,203 |
|---|---:|---:|---:|---:|---:|---:|---:|
| trailing-500 | 10.16 | 11.73 | 12.57 | 13.27 | 14.28 | 14.82 | 15.82 |

最後約 5,000 iterations 仍增加 1.01 bps/Hz。因此，40k model 的結果只能視為目前已達到的水準，不能視為充分訓練後的最終值。

## 5. GNN 與可達參考點的差距

三支 evaluation scripts 都在 2k 與 40k checkpoint 上重新執行。這是必要的，因為 phase-only 與 greedy reference 都固定 GNN 輸出的 $W$；checkpoint 改變時，reference 本身也會改變。

### 5.1 Continuous phase 與 joint reference（320 samples／40 batches）

| 方法 | 2k model | 40k model |
|---|---:|---:|
| centralized GNN | 7.454 | 15.544 |
| centralized $W$ + continuous phase-only | 16.205 | 24.630 |
| joint $(\theta,W)$（GNN initialization） | 26.327 | 28.704 |
| **joint $(\theta,W)$（random initialization）** | **29.536** | **29.536** |
| **centralized GNN ÷ joint-rand reference** | **25.2%** | **52.6%** |

所有 reference 都直接對每個 sample 的真實 sum rate 做最佳化，因此需要 full CSI。它們是已找到的 feasible points，不是非凸問題的全域 optimum 或 upper bound。

`joint-rand` 在兩個 checkpoint 上都得到 29.536。由於它不依賴 GNN output，兩次結果一致表示這個 reference 在目前設定下相當穩定，可作為共同比較基準。

訓練預算帶來的改善為

$$
15.544-7.454=8.090\ \text{bps/Hz}.
$$

因此，GNN 由 joint-rand reference 的 25.2% 提升至 52.6%。剩餘的 13.992 bps/Hz 是 **joint $(W,\theta)$ gap**，不能全部歸因於 RIS phase；其中一部分也可能仍來自訓練不足，因為 §4 的 training curve 尚未飽和。

初始化差異也明顯縮小。`joint_rand − joint` 由 2k model 的 3.209（$t=15.1$，100% batch 勝率），降至 40k model 的 0.833（$t=4.80$，87.5%）。較長訓練讓 GNN output 落入更好的 optimization basin，但仍略差於 random initialization 找到的 reference。

### 5.2 2-bit greedy reference（800 samples／100 batches）

| 方法 | 2k model | 40k model |
|---|---:|---:|
| centralized GNN | 7.409 | 15.657 |
| centralized $W$ + 2-bit greedy | 14.058 | 22.722 |
| greedy gain | +6.649 | **+7.065** |
| centralized GNN ÷ greedy | 52.7% | **68.9%** |

較長訓練讓 GNN 相對於 greedy reference 的比例由 52.7% 提升至 68.9%，但 absolute gap 並未縮小。原因是 $W$ 改善後，greedy 能達到的 reference 也同步由 14.058 升至 22.722。

Post-hoc quantization loss 同樣由 0.621 增至 1.132 bps/Hz。這表示模型效能愈高，直接將 continuous phase 四捨五入至 2-bit grid 的代價也愈大。

## 6. 對前一份報告的三項更正

### 6.1 centralized–decentralized gap 不應直接放棄

前一份報告根據 2k model 的 0.16 bps/Hz gap，判斷這條軸缺乏足夠 headroom。40k model 得到不同結果：

| 訓練預算 | centralized | decentralized | centralized − decentralized | 相對 centralized |
|---|---:|---:|---:|---:|
| 2,000 iterations | 7.581 | 7.436 | 0.146 | 1.9% |
| 40,000 iterations | 16.101 | 13.746 | **2.354** | **14.6%** |

gap 擴大約 16 倍。2k checkpoint 下，兩種 inference modes 的效能都很低，因此額外資訊的價值不明顯。訓練延長後，centralized 能更充分利用完整輸入，兩者差距才顯現。

因此，論文所報告的「decentralized 接近 centralized」不一定能維持到較充分的訓練預算。在 seed 0、40k checkpoint 下，decentralized 的相對差距是 14.6%，而非 1.9%。這項結論目前只有 seed 0 的證據，而且量測點（40k checkpoint）本身尚未飽和。確認方式是在收斂後的 checkpoint 上重新量測，而不是在未飽和的預算下做 seed 掃描；理由見 §8。

### 6.2 greedy 後的排序翻轉只出現在 2k model

前一份報告觀察到，使用 greedy phase 後，2k model 的排序變成 `decentralized-own > decentralized > centralized`。40k model 不再出現這個現象：

| greedy phase 下的方法 | 2k model | 40k model |
|---|---:|---:|
| centralized | 14.003 | **22.396** |
| decentralized | 14.165 | 22.202 |
| decentralized-own | **14.667** | 19.166 |

在 40k model 下，centralized 仍然最高。因此，排序翻轉應標註為訓練不足 checkpoint 的現象，不能作為一般性結論。

### 6.3 local-view gap 也是主要問題

前一份報告使用 2k model，得到的 centralized–decentralized-own gap 只有 0.568 bps/Hz，因此判斷 local CSI 不是主要瓶頸。40k model 顯示這項判斷過早：

| 方法 | 2k model | 40k model |
|---|---:|---:|
| centralized | 7.409 | 15.657 |
| decentralized（43.9% visibility） | 7.249 | 13.279 |
| decentralized-own（12.0% visibility） | 6.841 | **8.815** |
| centralized − decentralized | 0.160 | **2.378**（$t=16.6$，99% batch 勝率） |
| centralized − decentralized-own | 0.568 | **6.842**（$t=31.5$，100%） |
| decentralized − decentralized-own | 0.408 | **4.465**（$t=25.6$，100%） |

最明顯的現象是：由 2k 延長至 40k 後，centralized 由 7.409 升至 15.657，但 decentralized-own 只由 6.841 升至 8.815。這表示目前以 centralized input 訓練的模型，無法在 own-only inference 中獲得同等幅度的訓練收益。

這裡的 6.842 是 **local-view induced sum-rate gap**，不應直接解讀為不可避免的 information-theoretic loss。模型只在 centralized input 上訓練，decentralized-own 則在 inference 時遮蔽輸入，因此差距同時包含有限資訊與 train–test mismatch。針對 decentralized-own 重新訓練後能縮小多少，尚未測量。

在目前結果中，decentralized 相對 centralized 少 15.2%，decentralized-own 少 43.7%。論文式 (10) 的額外回授使 decentralized 比 decentralized-own 多 4.465 bps/Hz，但 UE→AP feedback 也增加 3.67 倍。這項成本未列入論文 Table II。

## 7. local-view gap 與 phase gain

本節要分開回答兩個問題：

1. 在每一種 input mode 下，固定 $W$ 後改善 RIS phase，可以增加多少 sum rate？
2. centralized 與 decentralized-own 的 sum-rate gap，會因此縮小多少？

以下結果使用同一個 40k checkpoint 與同一批 320 samples。

### 7.1 同一模式內的 phase gain

| Input mode | GNN phase | greedy phase | Phase gain |
|---|---:|---:|---:|
| centralized | 15.544 | 22.396 | +6.852 |
| decentralized | 13.292 | 22.202 | +8.910 |
| decentralized-own | 8.492 | **19.166** | **+10.675** |

每一列都固定該模式下 GNN 產生的 $W$，只更換 $\theta$。因此，phase gain 的定義是

$$
R(W_{\mathrm{GNN}},\theta_{\mathrm{greedy}})
-R(W_{\mathrm{GNN}},\theta_{\mathrm{GNN}}).
$$

例如，decentralized-own 的 phase gain 是 $19.166-8.492=10.675$ bps/Hz。這個數字不是 SINR gain，也不是增加 CSI messages 的收益；它表示固定 $W$ 後，以 full-CSI greedy oracle 改善 phase 所增加的 sum rate。

### 7.2 不同 input modes 之間的 local-view gap

| 比較量 | 計算 | 數值 |
|---|---|---:|
| GNN phase 下的 local-view gap | $15.544-8.492$ | 7.053 |
| greedy phase 下的 residual gap | $22.396-19.166$ | 3.230 |
| greedy 使 gap 減少的幅度 | $7.053-3.230$ | **3.823（54.2%）** |
| decentralized-own greedy − centralized GNN | $19.166-15.544$ | **+3.622** |

第二張表是第一張表的跨列比較。7.053 表示 GNN output 在 centralized 與 decentralized-own 之間的 sum-rate gap；改用 greedy phase 後，gap 降至 3.230。兩者的差 3.823 也等於 decentralized-own 與 centralized 的 phase gain 差：

$$
10.675-6.852=3.823.
$$

因此，54.2% 的正確解讀是：**full-CSI greedy oracle 消除了目前觀察到的 local-view sum-rate gap 的 54.2%。** 它不表示恢復了 54.2% 的 CSI、SINR 或 messages，也不證明僅靠 local CSI 的演算法可以取得相同增益。

這組結果支持兩項較保守的結論：

1. decentralized-own 產生的 $W$ 仍保留可被更好 phase 利用的潛力；配上 greedy phase 後可達 19.166 bps/Hz。
2. 當 centralized 與 decentralized-own 都使用相同的 greedy procedure 時，centralized 仍高出 3.230 bps/Hz。這是固定各自 $W$ 並使用特定 2-bit greedy reference 後的 residual gap，不是「phase 永遠無法修復」的理論下界。

此外，greedy 需要 full CSI 與大量逐 sample rate evaluations。它只能作為 headroom 或 teacher reference，不能被稱為已實現的 decentralized RIS configuration。

## 8. 執行中的工作：100k 延長訓練

| 工作 | 位置 | 目的 |
|---|---|---|
| 100k 延長訓練（seed 0；由 40k checkpoint warm start 再跑 60,000 iterations） | meow2，`/tmp2/b12902052/ThomasLai/code/stage0/results_extend_100k/M2_N30_L4_K8_P15.0_iter60000_seed0/run0/`，log 為同機的 `/tmp2/b12902052/ThomasLai/code/stage0/extend.log` | 找出 training curve 的 plateau，取得可供後續比較的收斂 checkpoint |

2026-09-14 08:57 的狀態：本次 run 完成 50,710／60,000 iterations，含 warm start 合計 90,710／100,000。平均約 97 iterations/min，單一 process 佔用一張 RTX 4090。

Warm start 沒有造成退步。40k checkpoint 的 3,200-sample final evaluation 為 centralized 16.101、decentralized 13.746；本次 run 第一個 validation（合計 42,000 iterations，400 samples）為 16.262 與 13.801。`ckpt_40k.pt` 未攜帶 optimizer state（log 記為 `optimizer state NOT available`），Adam moments 由零重建，但在 validation 上看不到可辨識的 transient。

Validation sum rate（400 samples，每 2,000 iterations）以 10k 為區塊平均後如下。區塊平均是為了壓低單點雜訊；單點序列的擺動可達 1–2 bps/Hz。

| 區間（合計 iterations） | centralized | decentralized |
|---|---:|---:|
| 42–52k | 16.740 | 14.366 |
| 52–62k | 17.304 | 14.695 |
| 62–72k | 18.625 | 15.914 |
| 72–82k | 19.208 | 16.221 |
| 82–90k | 19.983 | 16.673 |
| 最新單點（90k） | 21.301 | 18.150 |

42k–90k 全區間的線性斜率為每 10,000 iterations centralized +0.85、decentralized +0.63；最後六個 validation 點（80–90k）分別為 +0.93 與 +1.17，同一段的殘差標準差為 0.77 與 0.68。斜率高於雜訊水準，因此**合計 90,000 iterations 時 training curve 仍未飽和**。§4 對 40k model 的描述（尚未收斂）在更長的預算下依然成立。

每 5,000 iterations 儲存一次含 optimizer state 的 resumable checkpoint（`models/resumable_latest.pt`），中斷後可由該檔續訓。

### 8.1 已取消：六 seed 掃描

原先規劃 seeds 0–5、各 60,000 iterations 的掃描已取消，不再執行。該批 job 於 2026-09-13 20:56:05 全部停止在 iteration 20,000／60,000；六個 process 在同一秒結束，與本次 run 在 23:34–00:04 等待 daily GPU quota reset 的記錄一致，推測是被 meow2 的 quota 機制中止（未直接取得 enforcer log，屬推論）。

取消的理由不是該次中斷，而是量測順序：在 training curve 未飽和時重複 seed，量到的是未收斂 checkpoint 之間的變異，無法回答 §6.1 與 §6.3 的問題。因此把預算集中在單一 run 的收斂上，跨 seed 的統計確認延到方法凍結之後，與其他 stage 文件「paired multi-seed formal evidence 待方法凍結」的排序一致。該批 partial checkpoints 已於 2026-09-14 從 meow2 刪除（`results_seed_sweep/`，201 MB），`run_seed_sweep.sh` 也已從本機與 meow2 移除。

### 8.2 已排程：100k checkpoint 評估與 150k 續訓

`code/stage0/run_chain_150k.sh` 於 2026-09-14 09:41 在 meow2 的 tmux session
`thomaslai_chain_150k` 啟動，log 為 `chain_150k.log`。它等待 100k run 寫出
`resumable_final.pt`，然後依序執行三件事，中間不需人工介入：

1. 由 `resumable_final.pt` 取出兩個檔案：`ckpt_100k.pt`（純 state dict，兩支
   evaluation scripts 只接受這種格式）與 `resume_100k.pt`（含 optimizer state，
   並把 iteration 欄位更正為 100000）。更正是必要的：`ckpt_40k.pt` 沒有 iteration
   欄位，因此 100k run 自己的 bundle 只記到 60000。
2. 在 `ckpt_100k.pt` 上跑兩支 evaluation，設定與 2k／40k checkpoint 完全相同，
   使三個時間點可直接比較：`ablation_local_csi.py --samples 320 --greedy` 與
   `discrete_cd_baseline.py --samples 800 --rounds 4`。較昂貴的
   `continuous_ceiling.py` 不在這次的簡單評估內。
3. 由 `resume_100k.pt` warm start 續訓 50,000 iterations（合計 150,000），寫入
   `results_extend_150k/`。

續訓長度是由 quota 決定的：2026-09-14 09:40 剩餘 47,680 秒，觀測到的扣抵倍率約
1.12×（38,633 秒 quota 對應 34,500 秒 wall time），扣掉 100k run 的收尾與評估後，
50,000 iterations 約需 30,800 秒 quota，仍留約 12,000 秒餘裕，且會在 23:30 的
quota 空窗之前結束。script 在每個階段前重新檢查 quota，不足時等到 00:00 重置再跑，
而不是啟動一個注定被中止的長 job。

## 9. 統計協定補充

前一份報告指出，當 $n=5$ 時，two-sided exact sign test 的最小 $p$ value 為

$$
2/2^5=0.0625>0.05.
$$

因此，無論五個差值有多一致，這項檢定都不可能在 0.05 threshold 下達到 significance。後續實驗至少需要 $n\ge6$。此外，檢定方法也會大幅影響結論。

以 Stage 1C 的實際 paired differences 為例：

| 檢定 | two-sided $p$ value |
|---|---:|
| sign test（5/5 同號） | 0.0625 |
| **paired $t$-test**（$t=11.19$, df=4） | **0.000364** |

兩個 $p$ values 相差 172 倍。Sign test 只使用差值的正負號；paired $t$-test 則同時利用差值的大小與變異。Wilcoxon signed-rank test 在 $n=5$ 時的最小 two-sided $p$ value 同樣是 0.0625，因此無法解決樣本數造成的離散性限制。

Stage 1C 中「無法宣稱方案 C／D 有優勢」的結論，部分來自檢定選擇，而不只是 observed difference 不足。後續實驗應同時報告 paired $t$-test 與 sign test，並附上 effect size、paired differences 及 batch-clustered standard error。

## 10. 研究方向判斷

目前的結果可以用來安排下一輪實驗，但還不足以形成最終結論。原因是 40k training curve 尚未飽和，而且主要結果仍來自單一 model seed。依目前證據，研究軸可整理如下：

| 研究軸 | 目前證據 | 判斷 |
|---|---|---|
| 訓練收斂 | 2k → 40k 增加 8.090 bps/Hz；延長至合計 90k 後 validation 仍在上升 | 優先完成，否則其他 gap 會受 checkpoint quality 影響 |
| RIS phase quality | +7.065 vs 2-bit greedy；+9.086 vs continuous phase-only reference | 主軸 |
| Joint solution quality | +13.992 vs joint-rand reference | 是 $W$ 與 $\theta$ 的共同 gap，不能全歸因於 phase |
| Local input visibility | 同一批 320 samples 中，GNN gap 為 7.053；greedy 後 residual gap 為 3.230 | 主軸之一，但需先加入 decentralized-own training baseline |
| centralized–decentralized | 40k final evaluation 的 gap 為 2.354（14.6%） | 保留；待收斂 checkpoint 重新量測 |
| Aggregation／feedback overhead | decentralized fronthaul 2,640 vs centralized 12,215；論文式 (10) 使 air-interface feedback 由 11,879 增至 43,588 | 應與 sum rate 共同形成 rate–overhead trade-off |
| Discrete quantization | 40k model 的 post-hoc loss 為 1.132 | 併入 phase design，不需獨立成題 |

最值得驗證的問題是：

> **當每個 AP 只觀察自己服務鏈路的 CSI，且 AP→CPU aggregation 維持在明確的 signaling budget 內，是否能設計可部署的 decentralized policy，使 RIS phase 接近 full-CSI greedy reference？**

這個問題合理，但 §7 只證明存在明顯 headroom，尚未證明 decentralized 方法能取得 54.2% 的改善。現有 greedy reference 使用 full CSI；此外，即使是 decentralized-own，程式仍會將每個 AP 的 RIS proposals 送到 CPU 做 `RIS_merge`。因此，「local」描述的是每個 AP 的 input visibility，而不是完全沒有 central aggregation。

要建立完整論證，仍需完成以下工作：

1. 由現有 checkpoint warm start，找出 training curve 的 plateau；若合計 100,000 iterations 仍未飽和，繼續由 resumable checkpoint 續訓（§13）。
2. 使用 own-only input mask 重新訓練模型，分離有限資訊與 train–test mismatch 的影響。
3. 分別量測 masked-max contamination、未被呼叫的 `edge_update` 與 no-op pruning；這些實作問題應在訓練收斂後逐項 ablate。
4. 加入 MRT／RZF 搭配 optimized 或 random RIS 的 classical references。

因此，目前最適合的定位是：**RIS phase quality、local input visibility 與 signaling overhead 構成同一個 rate–overhead research problem；這是一個有實測 headroom 的方向，但仍需可部署方法、以及收斂 checkpoint 上的證據才能形成方法貢獻；跨 seed 的統計確認留到方法凍結之後。**

## 11. Reproducibility 與環境

- 本機：`torch 2.11.0+cu128`，NVIDIA GeForce RTX 5060 Laptop GPU。
- 遠端：meow2，4× RTX 4090，`/tmp2/b12902052/miniforge3/envs/decentralized-inference`，`torch 2.11.0+cu128`、`numpy 2.4.6`、Python 3.11.15。
- 基準 commit：`608e71d`。
- 本報告新增的 `fast_forward.py`、`train_fast.py` 與 `continuous_ceiling.py` 尚未提交。
- meow2 的 launcher：`code/stage0/run_extend.sh`（100k run）與 `code/stage0/run_chain_150k.sh`（評估與 150k 續訓）。兩者都固定單一 GPU，因為 meow2 在使用者持有 $N$ 張 GPU 而閒置少於 $N$ 張時，會以 $(N+0.2)N$ 倍扣抵 quota。

主要指令如下：

```bash
# Extend training; warm start is supported.
python train_fast.py --n_iter 40000 --seed 0 --save_every 5000 \
    --resume <resumable_final.pt or model_final_run0.pt> --out_dir results_long_training

# The 100k continuation reported in §8 (meow2, one RTX 4090).
python train_fast.py --resume ckpt_40k.pt --n_iter 60000 --seed 0 \
    --log_eval_interval 2000 --save_every 5000 \
    --test_sample_val 400 --test_sample_final 3200 \
    --device cuda:0 --out_dir results_extend_100k

# Re-run the three evaluations on any checkpoint.
python discrete_cd_baseline.py --ckpt <ckpt> --samples 800 --rounds 4
python continuous_ceiling.py  --ckpt <ckpt> --samples 320 --steps 2000 --restarts 3
python ablation_local_csi.py  --ckpt <ckpt> --samples 320 --greedy
```

SHA-256：

| Artifact | SHA-256 |
|---|---|
| `fast_forward.py` | `e34aeb0a0df7da663afcff6b630dd9f7e9b0b12445709505dede94f74cc4f216` |
| `train_fast.py` | `a8d81190893dd2e2563434954a8976a8200ad4bc59c0af22f6f1c2e105f9e905` |
| `continuous_ceiling.py` | `82085a5c0013567b59f8803c6fe6a76f2134b21d34a9adc605041eac2ec6c142` |
| 40k checkpoint | `a5646c1dbc9295271b40e3ce1da9f1ad6fc2e2d7b8e2dab3f823b8861dcd9fef` |

## 12. Artifact Index

| Evidence | Location |
|---|---|
| 40k training 與 final evaluation | `code/stage0/results_long_training/M2_N30_L4_K8_P15.0_iter40000/run0/` |
| 2k reference | `code/stage0/results_repro_check/` |
| 兩個 checkpoints 的 greedy reference | `code/stage0/results_discrete_cd/headline_M2_P15/`、`.../trained40k_M2_P15/` |
| 兩個 checkpoints 的 continuous reference | `code/stage0/results_continuous_ceiling/headline_M2_P15/`、`.../trained40k_M2_P15/` |
| Local CSI ablation（兩個 checkpoints，包含 greedy） | `code/stage0/results_local_csi_ablation/`（`headline_M2_P15`、`with_greedy_M2_P15`、`trained40k_M2_P15`、`trained40k_with_greedy`） |
| 向量化實作與 training script | `code/stage0/fast_forward.py`、`train_fast.py` |
| 100k 延長訓練（執行中） | meow2：`/tmp2/b12902052/ThomasLai/code/stage0/results_extend_100k/M2_N30_L4_K8_P15.0_iter60000_seed0/run0/`，log 為同目錄上層的 `extend.log` |

| 100k checkpoint 的評估（排程中） | meow2：`results_local_csi_ablation/trained100k_with_greedy/`、`results_discrete_cd/trained100k_M2_P15/` |
| 150k 續訓（排程中） | meow2：`results_extend_150k/`，log `chain_150k.log` |

各結果目錄的 `paired_sum_rates.npz` 包含每個 sample 的 paired sum rates，可用來重算本報告的統計量。

## 13. 下一次更新條件

§8.2 的 chain 已涵蓋「跑完 100k → 評估該 checkpoint → 續訓到 150k」這三步，因此下一次更新的工作是**讀取結果並回填**，而不是重新排程：

1. 由 `results_local_csi_ablation/trained100k_with_greedy/summary.json` 與
   `results_discrete_cd/trained100k_M2_P15/summary.json` 取得 100k checkpoint 的數字，
   與 §6.3、§5.2 的 2k／40k 欄位並列，確認三項更正在更長預算下是否仍成立。
2. 以 100k run 的 3,200-sample final evaluation 更新 §1、§4 與 §6.1，並把
   `ckpt_100k.pt` 的 SHA-256 加入 §11 的表格。
3. 在本節記錄 150k 續訓的起訖 iteration、wall-clock 與 quota 消耗。

**收斂判準**：取最後六個 validation 點（每 2,000 iterations、400 samples）對 iteration
做線性回歸，若 centralized 與 decentralized 的斜率都落在同一段殘差標準差所對應的雜訊
範圍內，才視為飽和。依 §8 的資料（斜率 +0.93／+1.17 對殘差標準差 0.77／0.68），
合計 100,000 iterations 很可能仍不滿足這個條件，這也是 150k 續訓已先行排程的理由。

**若 150,000 iterations 仍未飽和，繼續由 `resumable_final.pt` warm start，而不是更換
seed 或 operation point。** 續訓沿用同一組參數（seed 0、`--log_eval_interval 2000`、
`--save_every 5000`、`--test_sample_val 400`、`--test_sample_final 3200`），寫入新的
`--out_dir`，不得覆寫既有 output root；每一段都要在本節留下紀錄。續訓長度應先由當時的
quota 反推（§8.2 的算法），寧可切成能完整跑完的區塊，也不要啟動會被 quota 中止的長 job
——被中止的 run 只留下 5,000 iterations 邊界的 `resumable_latest.pt`，不會產生 final
evaluation。

若某一段續訓的 validation 斜率首次落入雜訊範圍，則該段結束時的 checkpoint 就是本報告
所需的收斂點；此後再補 `continuous_ceiling.py` 的 joint reference，並重跑 §5 的完整
比較，之後才進入 §10 清單的第 2 項（own-only 重訓）。

跨 seed 的統計確認不在目前範圍內（§8.1）；待方法凍結後再依 §9 的協定執行，屆時同時
報告 paired $t$-test、sign test、effect size 與 batch-clustered standard error。
