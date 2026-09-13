# Stage 0 訓練預算與可達水準歸因報告

## Material Passport

- Origin Date: 2026-09-13
- Last Updated: 2026-09-13
- Verification Status: PARTIAL（單一 seed 的證據已完成；六 seed 掃描執行中）
- Version Label: `stage0_training_budget_v1`
- 前置文件: `doc/stage0_baseline_requalification_report.md`（v2）
- Scope: `code/stage0/`，論文設定 $M=2$、$P_{\max}=15$ dBm

本報告承接 baseline requalification 報告。前一份確認了 stage 0 未被改動、配置吻合論文 Table I、且可重現論文 Fig. 6／7；本報告處理它留下的最大未解問題：**GNN 與可達水準之間的巨大差距，有多少該歸給訓練預算。**

> **本報告更正了前一份報告的三項結論。** 三項的共同成因相同：它們都建立在訓練不足的 checkpoint 上。詳見 §6。

## 0. 記號約定

沿用前一份報告的約定（集中式／分散式、連續／量化／隨機、貪婪搜尋）。本報告新增兩個對照：

| 概念 | 寫法 | artifact 欄位 |
|---|---|---|
| 論文預算訓練（2000 iterations） | 2k 模型 | `results_repro_check/` |
| 延長預算訓練（40000 iterations） | 40k 模型 | `results_long_training/` |
| 分散式且保留式 (10) 額外回授 | 分散式（論文假設） | `dec_paper_*` |
| 分散式但移除該額外回授 | 分散式（只看自己） | `dec_own_only_*` |

## 1. 狀態總表

| 查核項目 | Verdict | 最小結論 | Claim boundary |
|---|---|---|---|
| 訓練瓶頸定位 | PASS | 95% 時間在 forward/backward，非資料產生 | 單一機器的 profile |
| 向量化等價性 | PASS | 輸出差 $\le 2.5\times10^{-6}$，梯度差 $\le 3.1\times10^{-5}$ | float32 捨入等級 |
| 訓練預算效應 | PASS | 集中式 7.581 → 16.101（×2.12） | 單一 seed，且 40k 仍未平台化 |
| 可達比例歸因 | PASS | GNN 由可達值的 25.2% 升至 52.6% | 同上 |
| 集中−分散 gap | PASS / 前結論更正 | 0.146 → 2.378（擴大 16 倍） | 單一 seed，待六 seed 確認 |
| 資訊落差 vs 相位品質 | PASS / 前結論更正 | 資訊落差 7.05，相位優化可回收 54% | 單一 seed |
| 六 seed 掃描 | IN FLIGHT | — | 見 §8 |

## 2. 訓練瓶頸定位

對 batch 8 的單次訓練迭代做 profile（本機 RTX 5060 Laptop）：

| 階段 | 每次迭代 | 佔比 |
|---|---:|---:|
| 資料產生 | 0.110 s | 4.7% |
| forward + loss | 0.849 s | 36.5% |
| backward + step | 1.368 s | 58.8% |
| 合計 | 2.327 s | — |

模型僅 2,206,479 個參數、batch 8、40 個節點，卻需 2.3 秒一次迭代。成因是 `model_2.py` 的實作方式：

- `initial_layer.forward` 與 `node_update_layer.forward` 將 batch 維度寫死為 1（`torch.zeros((1, self.L, ...))`），因此 `node_update.forward` 必須逐樣本迴圈；
- `node_update_layer.forward` 內有 `for k in range(self.K)`，對 40 個 AP–UE 節點逐一做小矩陣乘法。

兩者合計使每次迭代建出由數千個微小 op 組成的 autograd 圖。**論文的 2000 次迭代預算因此是被實作效率限制的，而非被問題本身的計算需求限制。**

## 3. 向量化與等價性

`code/stage0/fast_forward.py` 以批次化形式重算上述兩層的數學。它**不定義任何參數**，直接讀取既有 `node_update` 實例的權重，因此同一份 checkpoint 可直接載入，`model_2.py` 未被修改。

關鍵一步是 $A = 1 - I$ 的 element-wise max 聚合（式 14）。原本需對每個節點做一次「排除自己取最大」；向量化作法是取每個特徵的前兩大值，每個節點採用全域最大值，除非該最大值由自己貢獻，才改用第二大值。這是**精確等價**，非近似。

### 3.1 等價性驗證

| 檢查 | 集中式 | 分散式 |
|---|---:|---:|
| $\max\lvert\Delta W\rvert$ | $1.5\times10^{-8}$ | $1.4\times10^{-7}$ |
| $\max\lvert\Delta\theta\rvert$ | $1.5\times10^{-6}$ | $2.5\times10^{-6}$ |
| $\max\lvert\Delta(\partial L/\partial w)\rvert$ | $1.5\times10^{-5}$ | — |

在訓練好的 checkpoint 與隨機初始化上各驗證一次，並在遠端機器（meow2）重驗。梯度比對是必要的：訓練等價性取決於梯度，而非僅前向輸出。`train_fast.py` 每次啟動都重跑此檢查並寫入該次 run 的 log，不通過即中止。

分散式路徑另以 `ablation_local_csi.decentralized_forward`（先前已驗證與原實作逐位元相同）為參考再驗一次。

### 3.2 加速

| 路徑 | 原實作 | 向量化 | 加速 |
|---|---:|---:|---:|
| 訓練 step（batch 8） | 3.709 s | 0.426 s | 8.7× |
| 訓練 step（batch 32） | 13.966 s | 0.823 s | 17.0× |
| 分散式推論（batch 8） | 5.229 s | 0.024 s | **216×** |

分散式推論的加速最大，因為原實作在該路徑上同時有逐 AP、逐樣本與逐節點三重迴圈。3200 樣本的最終評估由約 28 分鐘降至約 8 秒。

## 4. 訓練預算效應

以向量化實作重跑，**除 `n_iter` 與驗證頻率外所有設定與論文相同**（batch 8、Adam lr $10^{-4}$、weight decay $10^{-6}$、seed 0）。最終評估皆為 3200 樣本。

| 指標 | 2000 iters（論文預算） | 40000 iters | 倍數 |
|---|---:|---:|---:|
| centralized | 7.581 | **16.101** | ×2.12 |
| centralized_discrete | 6.996 | 14.954 | ×2.14 |
| centralized_random_phase | 3.162 | 6.182 | ×1.96 |
| decentralized | 7.436 | **13.746** | ×1.85 |
| decentralized_discrete | 6.837 | 12.841 | ×1.88 |
| decentralized_random_phase | 3.090 | 5.852 | ×1.89 |

訓練曲線（trailing-500 平均）在 40000 次時**仍未平台化**：

| 迭代 | 5629 | 10966 | 16145 | 21725 | 27228 | 32741 | 38203 |
|---|---:|---:|---:|---:|---:|---:|---:|
| trailing-500 | 10.16 | 11.73 | 12.57 | 13.27 | 14.28 | 14.82 | 15.82 |

最後一段增量仍有 +1.01／5000 iterations，故 16.101 仍是低估。

## 5. 可達水準歸因

三支評估腳本皆在 2k 與 40k 兩個 checkpoint 上重跑，因為貪婪搜尋與連續相位參考點都以 GNN 的 $W$ 為條件，$W$ 改變後參考點本身也會移動。

### 5.1 連續相位可達階梯（320 samples／40 batches）

| 方案 | 2k 模型 | 40k 模型 |
|---|---:|---:|
| GNN | 7.454 | 15.544 |
| 連續相位（$W$ 凍結） | 16.205 | 24.630 |
| 聯合 $(\theta, W)$，由 GNN 解出發 | 26.327 | 28.704 |
| **聯合 $(\theta, W)$，由隨機出發** | **29.536** | **29.536** |
| **GNN ÷ 可達** | **25.2%** | **52.6%** |

由隨機出發的聯合最佳在兩個 checkpoint 上**完全相同（29.536）**。它本就與 GNN 無關，此一致性確認參考點本身穩定，可作為兩次比較的共同分母。

訓練預算解釋 $15.544 - 7.454 = +8.09$ bps/Hz，把 GNN 由可達值的四分之一推進到略過半。**剩餘 13.99 bps/Hz 尚未歸因**，且其中一部分仍屬訓練預算（§4 曲線未飽和）。

另一項變化：`joint_rand − joint`（由隨機出發 vs 由 GNN 解出發）由 $+3.209$（$t=15.1$，100% batch 勝率）降至 $+0.833$（$t=4.80$，87.5%）。訓練足夠後，GNN 的解落在好得多的最佳化盆地，但仍略遜於隨機起點。

### 5.2 貪婪 2-bit 參考點（800 samples／100 batches）

| | 2k 模型 | 40k 模型 |
|---|---:|---:|
| GNN 連續（集中式） | 7.409 | 15.657 |
| 貪婪 2-bit | 14.058 | 22.722 |
| 缺口 | +6.649 | **+7.065** |
| GNN ÷ 貪婪 | 52.7% | **68.9%** |

比值改善，但**絕對缺口未縮小反而略增**，因為 $W$ 變好後貪婪搜尋的天花板同步抬高。多訓練有效，卻未消除相位設計的結構性缺口。

量化損失同步增大（0.621 → 1.132），故模型越好，post-hoc 捨入的代價越高。

## 6. 對前一份報告的三項更正

### 6.1 「放棄集中−分散這條軸」——更正

前報告 §12 依 2k 模型的 0.16 bps/Hz 判定此軸無 headroom。收斂後並非如此：

| 訓練預算 | centralized | decentralized | 集中−分散 | 佔比 |
|---|---:|---:|---:|---:|
| 2000 iters | 7.581 | 7.436 | 0.146 | 1.9% |
| 40000 iters | 16.101 | 13.746 | **2.354** | **14.6%** |

gap 擴大 16 倍。機制直觀：兩個模型都很差時，它們差得很像，資訊多寡無從顯現；模型好到能實際利用資訊後，資訊落差才成為代價。

**這同時意味著論文「decentralized 近乎等同 centralized」的核心主張是訓練不足的產物。** 在相同設定下訓練到接近收斂，分散式的代價是 14.6%，而非 1.9%。

### 6.2 「貪婪相位後集中／分散排序翻轉」——僅適用於 2k 模型

前報告 §9.3 記錄：加上貪婪相位後排序翻轉為 只看自己 > 論文分散 > 集中。40k 模型下不成立：

| 貪婪相位下 | 2k 模型 | 40k 模型 |
|---|---:|---:|
| 集中式 | 14.003 | **22.396** |
| 分散式（論文假設） | 14.165 | 22.202 |
| 分散式（只看自己） | **14.667** | 19.166 |

40k 模型下集中式仍居首。該翻轉現象應標註為僅限訓練不足的 checkpoint。

### 6.3 「資訊落差不是瓶頸，RIS 配置才是」——更正

前報告 §9.3 據 2k 模型判定資訊損失僅 0.41 而相位優化值 +7.87，故資訊不是瓶頸。收斂後資訊落差本身即為主要瓶頸之一：

| | 2k 模型 | 40k 模型 |
|---|---:|---:|
| 集中式 | 7.409 | 15.657 |
| 分散式（論文假設，43.9% 可見度） | 7.249 | 13.279 |
| 分散式（只看自己，12.0% 可見度） | 6.841 | **8.815** |
| 集中 − 論文分散 | 0.160 | **2.378**（$t=16.6$，99% 勝率） |
| 集中 − 只看自己 | 0.568 | **6.842**（$t=31.5$，100%） |
| 論文假設的價值 | 0.408 | **4.465**（$t=25.6$，100%） |

最關鍵的一列是「只看自己」幾乎未從訓練中受益（6.841 → 8.815，僅 +1.97），而集中式翻倍。**訓練再久也無法補足不存在的資訊。**

換算代價：論文的分散式付 15.2%，真正 local 的分散式付 43.7%。論文式 (10) 的額外回授假設在收斂後值 4.465 bps/Hz——它不是技術細節，而是該方法能成立的關鍵，而其代價（UE→AP 回授量乘上 3.67 倍）未出現在 Table II。

## 7. 相位優化能否補償資訊損失

此問題決定「decentralized RIS configuration」方向是否成立。以 40k 模型測量（320 samples）：

| 方案 | GNN 相位 | 貪婪相位 | 增益 |
|---|---:|---:|---:|
| 集中式 | 15.544 | 22.396 | +6.85 |
| 分散式（論文假設） | 13.292 | 22.202 | +8.91 |
| 分散式（只看自己） | 8.492 | **19.166** | **+10.67** |

| 量 | 數值 |
|---|---:|
| 資訊損失（集中 − 只看自己，GNN 相位下） | 7.053 |
| 資訊損失（集中 − 只看自己，貪婪相位下） | 3.230 |
| **相位優化回收的部分** | **3.823（54%）** |
| 相位無法修復的殘餘資訊代價 | 3.230（46%） |
| 只看自己 + 貪婪相位 − 集中式 + GNN 相位 | **+3.62** |

結論分兩層，兩層都需保留：

1. **更好的 RIS 配置價值大於去中心化的資訊代價**：真正 local 的系統配上良好相位（19.166）勝過資訊完整但相位由 GNN 產生的集中式（15.544），差 +3.62。
2. **但相位優化無法完全抵銷資訊落差**：即使雙方都用貪婪相位，集中式仍領先 3.230。前報告「資訊落差不是瓶頸」的說法過強。

## 8. 執行中的工作

| 工作 | 位置 | 目的 |
|---|---|---|
| 六 seed 掃描（seeds 0–5，60000 iterations） | meow2，tmux session `seed_sweep`，`/tmp2/b12902052/ThomasLai/code/stage0/results_seed_sweep/` | 將 §6.1、§6.3 的單 seed 結果提升為可宣稱的統計結果 |

配置：每張 GPU 三個行程，跑在閒置的 GPU 0 與 3。基準測試顯示三行程並行時每個僅慢 14%（單獨 183.9 s／300 iterations，並行 208.8–221.3 s），故牆鐘時間由單一 seed 決定，約 11 小時。每 5000 iterations 存一次含 optimizer 狀態的可續跑 checkpoint。

## 9. 統計協定補充

前報告 §7 指出 $n=5$ 時非參數檢定的最小雙尾 $p$ 值為 $2/2^5 = 0.0625 > 0.05$，故結構上不可能顯著，並建議 $n \ge 6$。此處補充一項同等重要的事實：**檢定方法的選擇本身影響同樣巨大。**

以 Stage 1C 的實際配對差值計算：

| 檢定 | 雙尾 $p$ |
|---|---:|
| sign test（5/5 同號） | 0.0625 |
| **paired $t$-test**（$t=11.19$, df=4） | **0.000364** |

同一組資料相差 172 倍。sign test 僅使用差值的正負號而丟棄大小，$t$ 檢定則利用差值相對於變異的幅度。Wilcoxon signed-rank 在 $n=5$ 時的最小雙尾 $p$ 同樣是 0.0625（符號配置數同為 $2^n$），故改用它無法解決。

因此 Stage 1C「無法宣稱 C／D 優勢」的結論，部分源自檢定選擇而非資料不足。後續實驗應：$n \ge 6$；**同時報告配對 $t$ 檢定與 sign test**；並附上效應量與 batch-clustered 標準誤。

## 10. 方向判斷（更新）

依收斂後的實測 headroom 排序（前報告 §12 的排序建立在 2k 模型上，此處取代）：

| 軸 | 收斂後缺口 | 判斷 |
|---|---|---|
| RIS 相位品質 | +7.07（vs 貪婪）／+13.99（vs 聯合可達） | 主軸 |
| 資訊可見度（真正 local vs 集中） | 6.84，其中 3.23 無法由相位修復 | 升為主軸之一 |
| 集中−分散（論文假設下） | 2.354（14.6%） | 由「放棄」改為**可用** |
| 訓練預算 | 已解釋 +8.09，尚未飽和 | 必須先處理，否則其他歸因不可信 |
| Aggregation／feedback overhead | fronthaul 2,640 vs 12,215；空中介面 11,879 vs 43,588 | 併入主軸 |
| 離散量化損失 | 1.132（模型變好後同步增大） | 併入主軸 |

可定義的開放問題由此收斂為：**在真正 local 的視野（12% 可見度）下，如何取得接近貪婪品質的 RIS 配置。** §7 顯示該路徑可回收 54% 的資訊損失，並使系統勝過「資訊完整但相位由 GNN 產生」的集中式基準 +3.62。

剩餘缺口：

1. 六 seed 掃描完成（執行中）。
2. 訓練曲線平台位置——40000 次仍未飽和，可由既有 checkpoint warm start 延長。
3. 架構與實作瑕疵的責任分離：masked max 汙染、未被呼叫的 `edge_update`、no-op pruning。此三項須待訓練飽和後才能量測。
4. MRT／RZF 搭配優化或隨機 RIS 的古典參考點。

## 11. Reproducibility 與環境

- 本機：`torch 2.11.0+cu128`，NVIDIA GeForce RTX 5060 Laptop GPU。
- 遠端：meow2，4× RTX 4090，`/tmp2/b12902052/miniforge3/envs/decentralized-inference`，`torch 2.11.0+cu128`、`numpy 2.4.6`、Python 3.11.15（與本機版本一致）。
- 基準 commit：`608e71d`。本報告新增的 `fast_forward.py`、`train_fast.py`、`continuous_ceiling.py` 尚未提交。

主要指令：

```bash
# 延長訓練（可 warm start）
python train_fast.py --n_iter 40000 --seed 0 --save_every 5000 \
    --resume <resumable_final.pt 或 model_final_run0.pt> --out_dir results_long_training

# 在任一 checkpoint 上重跑三支評估
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
| 40000-iteration 訓練與最終評估 | `code/stage0/results_long_training/M2_N30_L4_K8_P15.0_iter40000/run0/` |
| 2000-iteration 對照 | `code/stage0/results_repro_check/` |
| 貪婪參考點（兩個 checkpoint） | `code/stage0/results_discrete_cd/headline_M2_P15/`、`.../trained40k_M2_P15/` |
| 連續參考點（兩個 checkpoint） | `code/stage0/results_continuous_ceiling/headline_M2_P15/`、`.../trained40k_M2_P15/` |
| Local CSI ablation（兩個 checkpoint、含貪婪版本） | `code/stage0/results_local_csi_ablation/`（`headline_M2_P15`、`with_greedy_M2_P15`、`trained40k_M2_P15`、`trained40k_with_greedy`） |
| 向量化實作與訓練腳本 | `code/stage0/fast_forward.py`、`train_fast.py` |
| 六 seed 掃描（執行中） | meow2 `/tmp2/b12902052/ThomasLai/code/stage0/results_seed_sweep/` |

各結果目錄的 `paired_sum_rates.npz` 內含每個 sample 的配對 sum rate，可重算本報告所有統計量。

## 13. 下一次更新條件

六 seed 掃描完成時更新 §6.1、§6.3 與 §8，並將單 seed 結果替換為附 $t$ 檢定與 sign test 的多 seed 結果。若訓練曲線在 60000 次仍未飽和，記錄該事實並以 warm start 延長，不因此更換 seed 或操作點。
