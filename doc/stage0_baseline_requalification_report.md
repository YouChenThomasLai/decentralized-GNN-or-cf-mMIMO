# Stage 0 Baseline Requalification 與離散 RIS 相位可達參考點報告

## Material Passport

- Origin Date: 2026-09-13
- Last Updated: 2026-09-13
- Verification Status: VERIFIED（配置比對、論文數值比對、重現性重跑、離散相位貪婪搜尋 baseline 皆完成）
- Version Label: `stage0_baseline_requalification_v2`（v2 新增 §9 local CSI ablation，並更正 §10 的 overhead 結論）
- Scope: `code/stage0/`（唯一含 RIS 的 stage）與論文 `doc/Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks.pdf`
- Reference Paper: Ting, Chang, Chien, Peng, Lin, "Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks"

## 0. 記號約定

論文與本報告各自使用了會互撞的縮寫，為免誤讀，全文一律採下列寫法：

| 概念 | 本報告寫法 | 論文寫法 | artifact 欄位名 |
|---|---|---|---|
| 集中式推論（CPU 持有全域 CSI） | 集中式 | "Centralized" | `C_*`、`centralized*` |
| 分散式推論（每個 AP 只用 local CSI） | 分散式 | "Decentralized" | `D_*`、`decentralized*` |
| 連續相位 | 連續 | "(C)" | `*_cont` |
| 2-bit 量化相位（事後取最近格點） | 量化 | "(D)" | `*_round`、`*_discrete` |
| 隨機 2-bit 相位 | 隨機 | "(R-D)" | `*_random` |
| 逐元素貪婪座標下降（本報告新增的 oracle 參考點） | **貪婪搜尋** | 論文無此對照 | `*_cd`、`*_greedy` |
| 分散式且保留式 (10) 額外回授 | 分散式（論文假設） | 論文預設 | `dec_paper_*` |
| 分散式但移除該額外回授 | 分散式（只看自己） | 論文無此對照 | `dec_own_only_*` |

本報告的參數符號一律沿用論文 Table I，但程式的變數命名與論文不同，對照如下：

| 符號 | 意義 | 預設值 | 程式中的名稱 |
|---|---|---|---|
| $L$ | AP 個數 | 5 | `num_of_AP`（寫死於 `trainer_2.py:70`） |
| $R$ | RIS 個數 | 4 | **`L`**、CLI `--L` |
| $N$ | 每個 RIS 的元素數 | 30 | `N`、CLI `--N` |
| $M$ | 每個 AP 的天線數 | 2 | `M`、CLI `--M` |
| $K$ | UE 個數 | 8 | `K`、CLI `--K` |
| $Q$ | RIS 相位量化位元數 | 2（$2^Q=4$ 個格點） | `num_bits` |
| $P_{\max}$ | 每個 AP 的發射功率上限 | 15 dBm | CLI `--pmax_dbm` |

最容易出錯的是 $L$ 與 $R$：**程式裡的 `L` 是 RIS 數，對應論文的 $R$；論文的 $L$（AP 數）在程式裡是 `num_of_AP`。** 例如 `--L 4` 指的是 4 個 RIS，不是 4 個 AP。

特別注意兩點。第一，論文圖例中括號裡的 C／D 指的是**連續／離散相位**，不是集中式／分散式；本報告不使用單獨的 C、D 字母。第二，artifact 欄位名裡的 `cd` 是 **coordinate descent（貪婪搜尋）的縮寫，與 centralized/decentralized 無關**；集中式與分散式在欄位名中是前綴 `C_` 與 `D_`。

## 1. 報告範圍與動機

本報告回答三個問題，並記錄由此得到的方向判斷：

1. `code/stage0/` 是否被 Stage 1–5 的後續工作意外修改？
2. stage0 的模擬配置是否與論文 Table I／系統模型一致，其結果是否重現論文 Fig. 6／Fig. 7？
3. 在只有「集中式 vs 分散式」對照的情況下，加入第一個 optimization baseline 之後，GNN 的實際品質如何？

動機是先前 Stage 3 與 Stage 5B 的 learned-policy 工作得到負結果，其中一個原因是所選的比較軸（相對強 heuristic 的百分比級差異）headroom 過小。因此在投入新方向前，先用既有 artifact 量測各條軸的實際 headroom。

## 2. 狀態總表

| 查核項目 | Verdict | 最小結論 | Claim boundary |
|---|---|---|---|
| stage0 原始碼完整性 | PASS | git 只有一次新增 commit，工作區乾淨，後續 stage 使用複本 | mtime 顯示 08-17 有一次 cleanup，git 無法還原其 diff |
| 配置 vs Table I | PASS | 幾何、通道、關聯門檻、GNN 超參、訓練設定全部吻合 | 噪聲功率為論文未列、程式未文件化的自由參數 |
| 論文結果重現 | PASS | Fig. 6／Fig. 7 各點差距 0.03–0.40 bps/Hz | 論文為單一 seed，Fig. 6 的 M=4 使用不同 seed |
| 現行程式重現存檔 | PASS | 八個指標最大偏差 0.16 bps/Hz | 跨 GPU，非 bit-exact |
| 離散相位貪婪搜尋參考點 | PASS / GNN 表現為負 | GNN 相位僅達貪婪搜尋的 46–58% | 貪婪搜尋為 oracle，且 $W$ 固定，其值是聯合最佳的**下界**而非上界 |
| Local CSI 假設 ablation | PASS | 移除式 (10) 第二集合後集中−分散差距 ×3.6；改用貪婪相位後排序翻轉 | 單一 checkpoint／seed 0；翻轉成因未釐清 |

## 3. stage0 原始碼完整性

`git log --all -- "code/stage0/*.py"` 只有一個 commit `b712cb1`（2026-08-20），且所有檔案均為新增（`A`）而非修改；`git status code/stage0` 乾淨。Stage 1–5 使用的是獨立複本（`code/snapshot_network_scaling/stage0/`、`code/snapshot_network_scaling/v2_bpp/stage0/`、`code/stage2/` 等，`data.py` 的 md5 各不相同），因此後續工作沒有回寫 stage0。

需留存的疑點：`data.py`、`trainer_2.py`、`utils_return_indivial_rates.py` 的 mtime 為 2026-08-17，晚於結果產生時間 2026-08-13，而 git 歷史自 08-20 才開始，故該次編輯的 diff 無法由版本控制還原。`code/stage0/doc.txt` 記載該 cleanup「yields the exact same result as in the first paper submission」。第 6 節的重現性重跑為此宣稱提供了實證支持。

## 4. 配置比對

| Table I 項目 | 論文 | 程式 | 位置 |
|---|---|---|---|
| APs $L$ | 5 | `num_of_AP = 5` | `trainer_2.py:70` |
| AP 天線 $M$ | 2 | CLI `--M` | `run_exp-v2.sh` |
| RISs $R$ | 4 | CLI `--L`（程式的 `L` 為 RIS 數） | `data.py:98` |
| RIS 元素 $N$ | 30 | CLI `--N` | — |
| UEs $K$ | 8 | CLI `--K` | — |
| Rician $\kappa$ | 10 | `Rician_factor = 10` | `utils_return_indivial_rates.py:103` |
| 大尺度衰落（AP–RIS） | $10^{-0.5}$ | `fading_BS_RIS = 10**(-0.5)` | `utils:107` |
| 路徑損耗指數（AP–RIS） | 2 / 2.5 | `path_loss_exp_LOS/NLOS` | `utils:109-110` |
| 大尺度衰落（RIS–UE） | $10^{-0.5}$ | `fading_RIS_users = 10**(-0.5)` | `utils:108` |
| 大尺度衰落（AP–UE） | $10^{-4.5}$ | `fading_BS_users = 10**(-4.5)` | `utils:105` |
| 路徑損耗指數（AP–UE） | 3.5 | `path_loss_exp_BS_user = 3.5` | `utils:106` |
| 關聯門檻 $\rho$ | 0.1 | `associate_threshold = 0.1` | `trainer_2.py:77-78` |
| 量化位元 $Q$ | 2 | `num_bits = 2` | `trainer_2.py` eval |
| GNN 層數 $D$ | 6 | `node_update(...,6,...)` | `trainer_2.py:71` |
| GNN 潛在維度 $q$ | 64 | `ch=64` | `trainer_2.py:71` |
| 幾何（Fig. 5） | AP r=200、RIS r=100、UE r≤100 | `gen_fixed_location(5,200)` / `(4,100)` / `gen_location(K,100)` | `data.py:95-99` |
| 訓練 | Adam, lr $10^{-4}$, wd $10^{-6}$, batch 8, 2000 iters | 一致 | `trainer_2.py:114, 66` |

通道產生亦與式 (1)–(3) 一致：`gen_LOS` 實作 $\sqrt{\kappa/(\kappa+1)}\,a_R a_A^H$ 的 LoS 分量，AP–UE 直連以 Rician factor 0 實作為 Rayleigh。

**唯一未文件化的自由參數是噪聲。** 論文 Table I 未列 $\sigma^2$。程式的做法是 `scale = -7`，把所有通道振幅放大 $10^7$ 倍；`background_noise` 雖被計算但從未使用，真正進入 SINR 的是 `utils_return_indivial_rates.py:195` 寫死的 `sigma = (2e-2)**2`，原始碼註解為「If too low, varying Pmax seems to be useless」。換算回未縮放尺度約為 $4\times10^{-18}$ W，比程式自己宣告的 `background_noise = 2e-12` 低約 57 dB。這不構成與論文的不一致（論文未給定該值），但它是一個會左右整個 SINR regime 的隱含設定，Stage 1A 落入 noise floor 即與此相關。任何跨 stage 的數值比較都必須先確認此參數。

## 5. 論文結果比對

論文的 Fig. 6／Fig. 7 是 matplotlib 產生的向量圖，可由 PDF 的 Form XObject content stream 解出繪圖座標，再以座標軸刻度校準還原數值。

嵌入圖檔名稱同時揭露了論文的實驗協定：

```
./figs/sum-rate-vs-M-seed-0_M4_uses_seed7_cropped.pdf
./figs/sum-rate_vs_Pmax-seed-0-v3_cropped.pdf
```

即論文為單一 seed（seed 0），且 Fig. 6 的 $M=4$ 點改用 seed 7。

**Fig. 6（$P_{\max}$ = 15 dBm）**

| $M$ | 論文 集中式 | stage0 集中式 | Δ | 論文 分散式 | stage0 分散式 | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.2477 | 6.2193 | −0.028 | 5.8077 | 5.8597 | +0.052 |
| 2 | 7.4729 | 7.6632 | +0.190 | 7.1689 | 7.3577 | +0.189 |
| 3 | 9.2919 | 9.2467 | −0.045 | 8.9135 | 8.8831 | −0.030 |
| 4 | 9.7451 | 10.1429 | +0.398 | 9.4615 | 9.6632 | +0.202 |
| 5 | 11.3724 | 11.1388 | −0.234 | 10.9608 | 10.9131 | −0.048 |

$M=4$ 偏差最大（+0.398），與該點在論文中改用 seed 7 一致。

**Fig. 7（$M$ = 2）**

| $P_{\max}$ | 論文 集中式 | stage0 集中式 | Δ | 論文 集中−分散 | stage0 集中−分散 |
|---:|---:|---:|---:|---:|---:|
| 5 | 4.3558 | 4.5921 | +0.236 | 0.3286 | 0.2742 |
| 10 | 6.1073 | 6.4009 | +0.294 | 0.2555 | 0.3492 |
| 15 | 7.4729 | 7.6632 | +0.190 | 0.3040 | 0.3056 |
| 20 | 8.3589 | 8.5014 | +0.142 | 0.1384 | 0.1631 |
| 25 | 9.2642 | 9.3352 | +0.071 | 0.0639 | 0.0927 |
| 30 | 10.3620 | 10.3643 | +0.002 | 0.0529 | −0.0016 |
| 35 | 11.0040 | 11.0299 | +0.026 | −0.0510 | −0.0217 |

兩項可直接讀出的結論：

- **集中−分散 gap 在高功率區趨近於零並轉負，這是論文自身資料的性質**，不是本地實作的缺陷。論文 Fig. 7 在 $P_{\max}=35$ dBm 的集中−分散 gap 為 −0.0510。
- 論文自身的量化損失（連續相位減 2-bit 量化相位）為 0.290 / 0.542 / 0.918 / 1.812（$P_{\max}$ = 5 / 15 / 25 / 35），與 stage0 的 0.323 / 0.583 / 1.066 / 1.671 同一趨勢。在 $P_{\max}=35$ dBm，論文自身的量化損失是其集中−分散 gap 絕對值的 35 倍。

## 6. 重現性重跑

以現行程式（08-17 cleanup 後）重跑 `--M 2 --N 30 --L 4 --K 8 --pmax_dbm 15 --batch_size 8 --runs 1`，與 2026-08-13 存檔比對：

| 指標 | 重跑 | 存檔 | Δ |
|---|---:|---:|---:|
| centralized | 7.58144 | 7.66323 | −0.082 |
| centralized_discrete | 6.99630 | 7.08062 | −0.084 |
| centralized_random_phase | 3.16156 | 3.03950 | +0.122 |
| centralized_random_phase_discrete | 3.16344 | 3.29986 | −0.136 |
| decentralized | 7.43574 | 7.35767 | +0.078 |
| decentralized_discrete | 6.83669 | 6.80767 | +0.029 |
| decentralized_random_phase | 3.08992 | 3.00848 | +0.081 |
| decentralized_random_phase_discrete | 3.07781 | 3.23741 | −0.160 |

最大偏差 0.16 bps/Hz（≤2.1%）；量化損失幾乎重合（0.58514 vs 0.58261）。final evaluation 使用 3200 samples（400 batches），batch-clustered SE 約 0.086，故上述偏差落在 1–2σ。原始結果跑在 RTX 5090、重跑在 RTX 5060，kernel 選擇差異經 2000 次迭代放大，此量級為預期。

**結論：無證據顯示 08-17 的 cleanup 改變數值行為。**

附帶訊號：集中−分散 gap 在重跑為 0.1457，存檔為 0.3056 —— 僅更換 GPU 即變動約一倍。該指標不具備支撐方法論主張的穩定度。

## 7. 統計協定修正

`data.py:117` 的 `gen_location` **每個 batch 只呼叫一次**，因此同一 batch 內 8 個 sample 共用同一組 UE 位置，僅 fading 不同。sample 之間不獨立，**有效樣本數是 batch 數而非 sample 數**，naive per-sample SE 低估約 $\sqrt{8}$ 倍。本報告所有 SE 均以 batch-clustered 方式計算。

另一項先前 stage 的協定問題：Stage 3／5B 使用 $n=5$ 的雙尾 exact sign test，其最小可能 $p$ 值為 $2/2^5 = 0.0625 > 0.05$，在任何資料下都不可能顯著。$n=6$ 起為 $2/2^6 = 0.031$。後續實驗應使用 $n \ge 6$ 並報告配對差值。

Stage 1C 的資料顯示配對的價值：絕對值跨 seed std 為 1.55，而配對集中−分散差值的 std 僅 0.121。

## 8. 離散相位貪婪搜尋可達參考點（coordinate descent）

**貪婪搜尋不是上界。** 三者的關係是

$$\text{rate}(W_{\text{GNN}}, \theta_{\text{greedy}}) \;\le\; \max_{\theta} \text{rate}(W_{\text{GNN}}, \theta) \;\le\; \max_{W,\theta} \text{rate}(W, \theta),$$

即貪婪搜尋是一個**可達點**，且是聯合最佳的**下界**。它能支持的主張只有「GNN 至少差了這麼多」，真正的差距只會更大。本節所有「缺口」數字都應照此解讀。

### 8.1 方法與驗證

`code/stage0/discrete_cd_baseline.py` 以唯讀方式載入 checkpoint，固定 GNN 輸出的 beamformer $W$，僅對 $\theta$ 在**與論文相同的 2-bit 網格**上執行逐元素貪婪座標下降，每個 sample 獨立選取。

一輪掃描的成本為 $R \times N \times 2^Q = 4 \times 30 \times 4 = 480$ 次 sum rate 評估：共 $R \times N = 120$ 個相位變數，每個變數試遍 $2^Q = 4$ 個候選格點，取使該 sample 的 sum rate 最大者後再處理下一個變數。預設最多 4 輪，單輪改善量低於 $10^{-6}$ 即提早停止。

由於 `cal_loss` 以 Python 迴圈實作（外層 $L \times K$ 個 AP–UE 節點、內層 $R$ 個 RIS 與 $K^2$ 個干擾項），單次呼叫成本高，難以承受上述評估次數，腳本內另實作向量化的等價評估器，並對其做了三重驗證：

| 驗證 | 結果 |
|---|---|
| 對 GNN 解，fast vs `cal_loss` | abs err $4.8\times10^{-7}$ |
| 對貪婪搜尋解，fast vs `cal_loss`（集中式） | abs err $0.0$ |
| 對貪婪搜尋解，fast vs `cal_loss`（分散式） | abs err $9.5\times10^{-7}$ |
| 貪婪搜尋解的相位模長 | $\max\lvert\lVert\theta\rVert-1\rvert = 0$ |
| 貪婪搜尋解是否在 2-bit 網格上 | True |

亦即頭條數字不僅由新評估器產生，也經原始 `cal_loss` 重算確認，且未違反單位模與離散集合約束。$W$ 不變，故 (8b) 功率約束自動維持。

### 8.2 Headline cell（$M=2$、$P_{\max}=15$ dBm、800 samples／100 batches）

| 方案 | 集中式 | 分散式 |
|---|---:|---:|
| GNN 連續相位 | 7.4088 ± 0.173 | 7.2490 ± 0.168 |
| GNN + post-hoc 量化（論文做法） | 6.7874 | 6.6464 |
| **2-bit 貪婪搜尋** | **14.0576 ± 0.236** | **14.1986 ± 0.245** |
| 隨機離散相位 | 2.8968 | 2.9947 |

配對比較（batch-clustered）：

| 比較 | 平均差 | cSE | $t$ | batch 勝率 |
|---|---:|---:|---:|---:|
| 貪婪搜尋 − GNN 連續（集中式） | +6.6488 | 0.140 | 47.4 | 100% |
| 貪婪搜尋 − GNN 連續（分散式） | +6.9496 | 0.155 | 44.8 | 100% |
| 量化損失（集中式） | +0.6215 | 0.036 | 17.2 | 95% |
| 量化損失（分散式） | +0.6026 | 0.029 | 20.9 | 99% |
| 集中−分散 gap | +0.1598 | 0.031 | 5.2 | 68% |
| 貪婪搜尋後的集中−分散 | −0.1410 | 0.033 | −4.3 | 32% |

貪婪搜尋由隨機初始化出發的結果與由量化初始化出發相差 −0.10，顯示結果不依賴初始點。

### 8.3 全操作點結果（集中式，每點 160 samples／20 batches）

| cell | GNN 連續 | GNN 量化 | 貪婪搜尋 | 隨機 | 量化損失 | 貪婪 − GNN | cSE | $t$ | GNN/貪婪 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $P_{\max}$=5 | 4.0897 | 3.8488 | 8.6606 | 1.6484 | 0.2409 | +4.5709 | 0.375 | 12.2 | 47.2% |
| $P_{\max}$=10 | 5.8306 | 5.3927 | 11.4824 | 2.3549 | 0.4380 | +5.6518 | 0.396 | 14.3 | 50.8% |
| $P_{\max}$=15 | 7.0648 | 6.4440 | 13.4411 | 2.8163 | 0.6208 | +6.3762 | 0.337 | 18.9 | 52.6% |
| $P_{\max}$=20 | 8.0558 | 7.1552 | 15.6420 | 3.2198 | 0.9006 | +7.5862 | 0.344 | 22.1 | 51.5% |
| $P_{\max}$=25 | 8.9931 | 7.9158 | 17.4280 | 3.5165 | 1.0772 | +8.4350 | 0.303 | 27.9 | 51.6% |
| $P_{\max}$=30 | 9.9331 | 8.6382 | 17.8724 | 3.9126 | 1.2949 | +7.9394 | 0.337 | 23.5 | 55.6% |
| $P_{\max}$=35 | 10.6870 | 8.9133 | 18.5281 | 4.0387 | 1.7738 | +7.8411 | 0.286 | 27.4 | 57.7% |
| $M$=1 | 6.2089 | 5.8012 | 12.8944 | 2.3396 | 0.4077 | +6.6855 | 0.242 | 27.6 | 48.2% |
| $M$=2 | 7.0648 | 6.4440 | 13.4411 | 2.8163 | 0.6208 | +6.3762 | 0.337 | 18.9 | 52.6% |
| $M$=3 | 8.9485 | 8.3778 | 17.0255 | 4.0352 | 0.5707 | +8.0770 | 0.368 | 22.0 | 52.6% |
| $M$=4 | 9.3990 | 9.0851 | 18.7105 | 6.4366 | 0.3139 | +9.3115 | 0.516 | 18.0 | 50.2% |
| $M$=5 | 9.6574 | 9.4006 | 20.8112 | 6.4026 | 0.2568 | +11.1537 | 0.530 | 21.1 | 46.4% |

（`vary_M/M2` 與 `vary_Pmax/P15` 為同一 checkpoint 與 seed，結果逐位相同，可作為腳本決定性的自我檢查。）

三項結論：

1. **GNN 的 RIS 相位在 13 個操作點全部只達到貪婪搜尋的 46–58%**，$t$ 值介於 12–47，batch 勝率全為 100%。這不是單一操作點的巧合。
2. **量化損失只佔總缺口的 6%–23%。** 即使實作完美的 quantization-aware training，可回收的部分也不到四分之一；主要損失來自 GNN 學到的相位本身。
3. **GNN 未能利用增加的天線數。** $M$=3→4→5 時 GNN 僅從 8.95 增至 9.40、9.66（增量 +0.45、+0.26，已飽和），而貪婪搜尋由 17.03 增至 18.71、20.81（增量 +1.68、+2.10，仍在加速）。論文 Fig. 6 呈現的隨 $M$ 單調成長，相對於可達曲線是被壓平的，且 $M$ 越大浪費越多。

### 8.4 Claim boundary

| Claim | Verdict |
|---|---|
| 貪婪搜尋結果未違反單位模、離散集合與功率約束 | Supported（見 8.1 驗證表） |
| GNN 相位在本模擬設定下遠離可達值 | Supported（13/13 cells，batch 勝率 100%） |
| 貪婪搜尋是可部署的方案 | **Contradicted** — 貪婪搜尋為 oracle，需全域真實 CSI 與數百次 rate 評估 |
| 貪婪搜尋值即為聯合最佳 | **Contradicted** — $W$ 固定為 GNN 輸出，該值仍是聯合最佳的下界 |
| 結論可外推至其他 topology／seed／模擬器 | Not supported — 全部基於 seed 0 的單一 checkpoint 與單一 AP 佈局 |
| 量化損失是 GNN 離散效能的主因 | **Contradicted** — 僅佔 6–23% |

## 9. Local CSI 假設 ablation

### 9.1 動機與方法

論文式 (10) 的第二個集合允許「被多個 AP 共同服務的 UE，把它與其他服務 AP 的 CSI 也回報給本 AP」。若此假設讓各 AP 的視野高度重疊，則論文報告的小幅集中−分散差距可能主要是該假設的產物，而非分散式推論本身的能力。

`code/stage0/ablation_local_csi.py` 以唯讀方式重現 `node_update.forward(training=False)` 的分散式流程並加入開關；`model_2.py` 未被修改。等價性已驗證：開關開啟時與原始前向輸出 `max\lvert\Delta W\rvert = 0.0`、`max\lvert\Delta\theta\rvert = 0.0`。

### 9.2 視野與 rate（800 samples／100 batches）

| 方案 | 每 AP 可見 AP–UE 節點 | 佔全域 40 節點 | GNN 連續相位 |
|---|---:|---:|---:|
| 集中式 | 40 | 100% | 7.4089 |
| 分散式（論文假設） | 17.576 | 43.9% | 7.2490 |
| 分散式（只看自己服務的鏈路） | 4.790 | 12.0% | 6.8407 |

| 比較 | 平均差 | cSE | $t$ | batch 勝率 |
|---|---:|---:|---:|---:|
| 集中 − 分散（論文假設） | +0.1598 | 0.031 | 5.2 | 68% |
| 集中 − 分散（只看自己） | +0.5682 | 0.065 | 8.8 | 86% |
| 論文假設 − 只看自己 | +0.4084 | 0.059 | 7.0 | 80% |

移除該假設後集中−分散差距擴大 3.6 倍，且 batch 勝率由 68% 升至 86%，即效應同時變大且變一致。論文設定下的「分散式」AP 已可見全域圖的 43.9%。

可見節點數具有組合意義：只看自己模式的跨 AP 總和為 $\sum_k\lvert\mathcal{L}_k\rvert = 23.95$，論文模式為 $\sum_k\lvert\mathcal{L}_k\rvert^2 = 87.88$。故本設定（$\rho = 0.1$）下每個 UE 平均被 3.0 個 AP 服務，而該假設使 UE→AP 回授量乘上 $87.88 / 23.95 = 3.67$ 倍。

### 9.3 搭配貪婪相位後排序翻轉（320 samples／40 batches）

| 方案 | GNN 連續相位 | 貪婪相位 |
|---|---:|---:|
| 集中式 | 7.4538 | 14.0031 |
| 分散式（論文假設） | 7.2529 | 14.1648 |
| 分散式（只看自己） | 6.7938 | **14.6672** |

| 比較（貪婪相位下） | 平均差 | cSE | $t$ | batch 勝率 |
|---|---:|---:|---:|---:|
| 集中 − 分散（論文假設） | −0.1616 | 0.049 | −3.3 | 32.5% |
| 集中 − 分散（只看自己） | −0.6641 | 0.083 | −8.0 | 12.5% |
| 分散（只看自己）貪婪 − 集中 GNN 相位 | +7.2134 | 0.222 | 32.4 | 100% |

在 GNN 相位下的排序為 集中 > 論文分散 > 只看自己；改用貪婪相位後**完全翻轉**為 只看自己 > 論文分散 > 集中。資訊不足造成的 rate 損失為 0.41–0.57，而相位優化在同一設定下帶來 +7.87，兩者相差 14 倍以上。

### 9.4 Claim boundary

| Claim | Verdict |
|---|---|
| 論文的小幅集中−分散差距部分來自式 (10) 的額外回授假設 | Supported（移除後差距 ×3.6，勝率 68%→86%） |
| 移除假設會使集中−分散成為值得追求的研究軸 | **Contradicted** — 擴大後仍僅 0.57，比相位缺口小 12 倍 |
| 論文報告的 C／D 排序在相位修正後仍成立 | **Contradicted** — 排序完全翻轉 |
| 分散式系統實際上可勝過集中式 | **Not supported** — 貪婪相位需全域真實 CSI，此為 headroom 陳述而非可部署結論 |
| 翻轉的成因已釐清 | Not established — 推測與 GNN beamformer 對其自身低品質相位的共適應有關，未驗證 |

## 10. 連續相位聯合可達參考點

### 10.1 方法

§8 的貪婪搜尋對聯合最佳做了三重限制：座標式局部最佳、限定 2-bit 網格、$W$ 凍結。`code/stage0/continuous_ceiling.py` 移除後兩項，直接以 Adam 對真實 sum rate 做上升：

- $\theta = e^{j\phi}$ 參數化，單位模由建構保證，不需流形機制；
- $W$ 沿用 `node_update` 相同的 mask-then-normalize 規則，故可行集與 GNN 完全相同；
- 三種模式：`phase_only`（$W$ 凍結於 GNN）、`joint`（$\theta, W$ 皆自由，由 GNN 解出發）、`joint_rand`（同前但由隨機出發，取 3 次重啟最佳）。

此設計與文獻 [3]–[8] 的 WMMSE／manifold 交替優化屬同一類模型式參考，但約束處理更易驗證。可行性在每次優化後檢查：

| 檢查 | phase_only | joint | joint_rand |
|---|---:|---:|---:|
| 功率超出預算 | 0.0 | $-1.5\times10^{-8}$ | $-5.3\times10^{-5}$ |
| 單位模誤差 | $6.0\times10^{-8}$ | $6.0\times10^{-8}$ | $6.0\times10^{-8}$ |
| 遮蔽欄位的 beamformer | 0.0 | 0.0 | 0.0 |

### 10.2 可達階梯（$M=2$、$P_{\max}=15$ dBm、320 samples／40 batches）

| 方案 | sum rate | cSE | 佔 `joint_rand` 比例 |
|---|---:|---:|---:|
| GNN + 2-bit 量化（論文 "D"） | 6.8240 | 0.263 | 23.1% |
| GNN 連續相位（論文 "C"） | 7.4538 | 0.289 | **25.2%** |
| 貪婪 2-bit（$W$ 凍結，§8） | 14.0031 | 0.394 | 47.4% |
| 連續相位解量化為 2-bit（$W$ 凍結） | 13.8294 | 0.369 | 46.8% |
| 連續相位（$W$ 凍結） | 16.2049 | 0.512 | 54.9% |
| 聯合解量化為 2-bit | 24.9408 | 0.601 | 84.4% |
| 聯合（由 GNN 解出發） | 26.3268 | 0.851 | 89.1% |
| **聯合（由隨機出發，3 次重啟）** | **29.5361** | 0.852 | 100% |

| 配對比較 | 平均差 | cSE | $t$ | batch 勝率 |
|---|---:|---:|---:|---:|
| 連續相位 − GNN | +8.7511 | 0.325 | 26.9 | 100% |
| 聯合 − GNN | +18.8730 | 0.647 | 29.2 | 100% |
| 聯合 − 連續相位 | +10.1219 | 0.474 | 21.3 | 100% |
| **聯合（隨機起點） − 聯合（GNN 起點）** | **+3.2093** | 0.213 | 15.1 | 100% |
| 聯合量化 − GNN 量化 | +15.4697 | 0.435 | 35.6 | 100% |

三項結論：

1. **論文的主要數字只達到可達水準的約四分之一**（25.2%）。§8 的「GNN 僅達貪婪搜尋的 46–58%」低估了差距，因為貪婪搜尋本身也只有 2-bit 可達值的 56%。
2. **由 GNN 解出發的聯合優化，結果比由隨機出發更差**（26.33 vs 29.54，$t=15.1$，100% 勝率）。GNN 的解不只次佳，還位於一個比隨機起點更差的盆地。
3. 連續最佳解直接捨入為 2-bit 得 13.83，略**低於**直接在離散空間搜尋的貪婪 14.00，再次顯示事後量化不如量化感知優化，惟幅度僅 0.17。

### 10.3 Claim boundary

| Claim | Verdict |
|---|---|
| 優化過程始終在問題 (8) 的可行集內 | Supported（見 10.1 檢查表） |
| 29.54 為聯合最佳的上界 | **Contradicted** — 非凸問題的局部最佳，仍是**下界** |
| 此參考點為可部署方案 | **Contradicted** — 逐實例優化，需全域真實 CSI 與 2000 次梯度步 |
| GNN 與參考點的差距全部源於架構能力不足 | Not established — 訓練預算的貢獻待 §12 缺口 3 的長訓練實驗釐清 |

## 11. Overhead 分析（解析式）

### 10.1 Table II 的計量不一致

論文 Table II 將集中式的 AP→CPU 訊令量計為 $2M\sum_k\lvert\mathcal{L}_k\rvert$，依據是「第 $k$ 個 UE 的 CSI $h_{\mathrm{eff},(l,k)} \in \mathbb{C}^{1\times M}$」。但集中式 GNN 的輸入是式 (9) 的 $\tilde{h}_{(l,r,k)} \in \mathbb{R}^{2M(N+1)}$：CPU 僅取得 $h_{\mathrm{eff}}$ 可以計算既定 $\theta$ 下的 rate，卻無法最佳化 $\theta$（那需要每個 RIS 的 $H_{(l,r,k)} \in \mathbb{C}^{M\times N}$）。兩者相差 $R(N+1) = 124$ 倍，故該欄低估集中式所需的上行量。

此外 Table II 未計入任何 UE→AP 的空中介面回授，而該腿正是式 (10) 第二集合的成本所在。

### 10.2 一致計量下的三腿比較

以 §9.2 量到的 $\sum_k\lvert\mathcal{L}_k\rvert = 23.95$、$\sum_k\lvert\mathcal{L}_k\rvert^2 = 87.88$，每條服務鏈路的 CSI 為 $2M(N+1)R = 496$ 個實數（$M=2$, $N=30$, $R=4$, $L=5$）：

| 訊令腿 | 集中式 | 分散式（論文） | 分散式（只看自己） |
|---|---:|---:|---:|
| UE→AP（空中介面，Table II 未計） | 11,879 | 43,588 | 11,879 |
| AP→CPU | 11,879 | 2,400 | 2,400 |
| CPU→AP/RIS | 336 | 240 | 240 |
| Fronthaul 小計（AP↔CPU） | 12,215 | **2,640** | **2,640** |
| 全部合計 | 24,094 | 46,228 | **14,519** |

結論分三層：

- **在 fronthaul 上，分散式勝出 4.6 倍。** 論文的核心 overhead 主張在一致計量下成立——其 Table II 反而把自己寫低了。
- **把空中介面計入後，論文版分散式比集中式差 1.9 倍**，因為式 (10) 的額外回授使 UE→AP 腿增至 43,588。
- **只看自己的版本兩腿皆勝**（合計較集中式低 1.66 倍），代價是 §9.2 量到的 0.41–0.57 bps/Hz。

本節為解析計數，不依賴訓練收斂或 seed 統計；唯一的量測輸入是 $\sum_k\lvert\mathcal{L}_k\rvert$ 與 $\sum_k\lvert\mathcal{L}_k\rvert^2$。

> 更正紀錄：本報告 v1 曾依 Table II 字面值宣稱「分散式總訊令量約為集中式的 7 倍」。該結論建立在低估的集中式欄位上，已由本節取代。

## 12. 方向判斷

依實測 headroom 排序：

| 軸 | 實測缺口 | 判斷 |
|---|---|---|
| RIS 相位品質 | 4.57–11.15 bps/Hz（GNN 僅達貪婪搜尋的 46–58%） | 主軸 |
| Aggregation／feedback overhead | fronthaul 2,640 vs 12,215；空中介面 11,879 vs 43,588 | 併入主軸，構成第二條評估軸 |
| 離散量化損失 | 0.24–1.77（佔相位缺口 6–23%） | 併入主軸，不單獨成題 |
| 集中−分散 gap | 0.16（論文假設）／0.57（移除假設後），且跨 GPU 變動一倍、相位修正後排序翻轉 | 放棄 |

建議的評估平面由「集中式 vs 分散式」改為「rate vs 可達參考點」與「rate vs signaling overhead」。兩者的 dynamic range 均為數量級，不需依賴 seed 數量堆砌顯著性。

§9.3 為後續方向提供了具體的立論基礎：在真正 local 的視野（12% 可見度）下，beamformer 的資訊損失為 0.41–0.57 bps/Hz，而同一設定下相位優化帶來 +7.87 bps/Hz。亦即**beamformer 的資訊落差不是瓶頸，RIS 配置才是**。因此可定義的開放問題為：在不取得全域 CSI 的前提下，如何得到接近貪婪搜尋品質的 RIS 配置。

缺口清單：

1. continuous-phase 的 near-optimal ceiling（WMMSE／FP 搭配 manifold optimization，即參考文獻 [3]–[8] 的標準做法），用以界定貪婪搜尋的 14.06 距離真正的聯合最佳還有多遠。
2. §9.3 排序翻轉的成因診斷（是否源於 GNN beamformer 對其自身低品質相位的共適應）。
3. GNN 相位品質不佳的成因分離：訓練未收斂（已有證據）／架構表達力不足／實作瑕疵（masked max、未使用的 `edge_update`、no-op pruning）三者尚未區分。
4. MRT／RZF 搭配優化或隨機 RIS 的古典參考點（`code/stage1/`、`code/stage5/` 已有可移植的實作）。

## 13. Reproducibility 與環境

- 環境：`torch 2.11.0+cu128`，NVIDIA GeForce RTX 5060 Laptop GPU；conda env `decentralized-inference`。
- stage0 原始碼 commit：`b712cb1`（2026-08-20）。
- 重跑指令：`python trainer_2.py --M 2 --N 30 --L 4 --K 8 --pmax_dbm 15 --batch_size 8 --runs 1 --device cuda:0 --out_dir results_repro_check`
- 貪婪搜尋掃描指令：`PY=<env python> bash run_discrete_cd.sh`

SHA-256：

| Artifact | SHA-256 |
|---|---|
| Headline checkpoint（`M2_N30_L4_K8_P15.0/run0/models/model_final_run0.pt`） | `25a58b1a8090541cb2c6cb18d72ee4de1c4aed8a6b02685d2c0d18af73c64474` |
| `model_2.py` | `bc18e7773376611f98f4769398205ca6f10e9055986bb8167ce7ebc673c8853f` |
| `data.py` | `5c409f02c241cd92362680d36eea3f8c26dcfe51f9fd897b27889cba5a33dae5` |
| `utils_return_indivial_rates.py` | `8740c15c5a6a28ae64fb74df861df5a5e1f9a5b56da3c73be55c1e2a04ecee7c` |
| `trainer_2.py` | `74da312f7523b3a96d40e4494fd065955cea87023c388204afdeed609281f0ee` |
| `discrete_cd_baseline.py` | `49488e3a92ee9f63ccfb995dd8d7630072a9196cf2b7a75aa38d50031fdfa110` |
| `ablation_local_csi.py` | `c630e1e788f810f10b306734d5a70ef3c5a8ac6c45a522c84e4b071e187b7908` |
| 重跑 final eval | `4e6294fedb729b97457b1a248fc32c412fec3a1287e60fdbe3098a5486182ebb` |

## 14. Artifact Index

| Evidence | Location |
|---|---|
| 重現性重跑 | `code/stage0/results_repro_check/` |
| 貪婪搜尋 baseline 全部 cells | `code/stage0/results_discrete_cd/`（含每 cell 的 `summary.json`、`summary.txt`、`paired_sum_rates.npz`） |
| 貪婪搜尋彙整表 | `code/stage0/results_discrete_cd/table.txt` |
| 貪婪搜尋評估程式 | `code/stage0/discrete_cd_baseline.py`、`run_discrete_cd.sh`、`summarize_discrete_cd.py` |
| Local CSI ablation 結果 | `code/stage0/results_local_csi_ablation/headline_M2_P15/`（800 samples）、`.../with_greedy_M2_P15/`（320 samples，含貪婪相位） |
| Local CSI ablation 程式 | `code/stage0/ablation_local_csi.py` |
| 原始 stage0 存檔結果 | `code/stage0/results_batch_8_BS-radius_200_RIS-radius_100_vary_M/`、`..._vary_Pmax/` |
| 論文 PDF（Fig. 6／7 向量資料來源） | `doc/Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks.pdf` |

`paired_sum_rates.npz` 內含每個 sample 的配對 sum rate，可直接重算本報告的所有統計量。

## 15. 下一次更新條件

本報告在下列任一情況下更新：加入 continuous-phase near-optimal ceiling（WMMSE／manifold）；stage0 baseline 改以 $n \ge 6$ 獨立 seed 重新量測；或凍結新的 method scope 時。不因單次負結果事後更換 seed、ceiling 或操作點。
