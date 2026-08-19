# Decentralized Active-CSI 實驗報告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate（living experiment report）
- Origin Date: 2026-08-18
- Last Updated: 2026-08-19
- Verification Status: ANALYZED（Stage 0、Stage 1A antenna/power sweeps、Stage 1B calibration 與 5-seed full run）；未進行獨立重新訓練，不標為 VERIFIED
- Version Label: decentralized_active_csi_report_v3
- Plan: `doc/decentralized_active_csi_experiment_plan.md`

## 1. 報告範圍

本報告保存從原版 RIS GNN 到 no-RIS snapshot baseline 的設定、執行狀態、數值診斷與決策。它是 living document：每次實驗啟動、完成、中止或被排除時都必須追加紀錄，不覆寫 raw artifacts。

目前涵蓋：

- Stage 0：原版有 RIS 的正向控制。
- Stage 1A：只移除 RIS、其他數值設定不變的忠實 ablation。
- Stage 1B：三個 noise values 的 seed-0 calibration pilot 與主設定 `1e-12` 的 5-seed full run 均已完成；numerical/learning gate 通過，但 C/D 方向仍不作穩健結論。

## 2. 原版與目前 GNN 學習內容

| 項目 | Stage 0：有 RIS | Stage 1A/1B：No-RIS |
|---|---|---|
| 輸入 | Direct AP–UE CSI + cascaded AP–RIS–UE CSI | Direct AP–UE CSI |
| 複數 precoder | 學習 $W$ | 學習 $W$ |
| Beamforming phase | 包含在 $W$ 的 real/imaginary outputs | 仍包含在 $W$ 的 real/imaginary outputs |
| RIS phase | 學習 $\Theta$ | 不存在 |
| AP power usage | 學習 $c\in(0,1)$ | 學習 $c\in(0,1)$ |
| Association | 固定 threshold mask，不學習 | 固定 threshold mask，不學習 |

簡寫為：

$$
\text{Stage 0 learns }(W,\Theta,c),\qquad
\text{Stage 1 learns }(W,c).
$$

Centralized 與 decentralized 使用同一組已訓練參數。差別只在 inference 時可見的 CSI：centralized 使用所有 association-derived CSI 的聯集；decentralized 由各 AP 使用其 local-visible CSI 產生自己的 precoder block。

## 3. 執行狀態快照

狀態檢查日期：2026-08-19。

| Stage | 狀態 | Seeds | 備註 |
|---|---|---:|---|
| Stage 0 antenna sweep | 已完成並存在本地 artifacts | 1 | 作正向控制，需標明 $n=1$ |
| Stage 0 power sweep | 已完成並存在本地 artifacts | 1 | 作正向控制，需標明 $n=1$ |
| Stage 1A antenna sweep，$M=1,\ldots,5$ | 已完成 | 5/設定 | 5 summaries、25 configs/checkpoints/final evaluations/logs 均完整 |
| Stage 1A power sweep，$P_{\max}=5,\ldots,35$ dBm | 已完成 | 5/設定 | 7 summaries、35 configs/checkpoints/final evaluations/logs 均完整 |
| Stage 1B calibration pilot | 已完成 | seed 0 | `1e-11`、`1e-12`、`1e-13` 各 200 iterations；三者皆通過 numerical gate |
| Stage 1B main full run | 已完成，exit code 0 | 5 seeds（0–4） | `noise_power=1e-12`、每 seed 2000 iterations；5 組 artifacts 完整且 finite |

## 4. Stage 0：原版正向控制

### 4.1 設定

- APs：5；UEs：8。
- RISs：4；每個 RIS 30 elements。
- AP antennas：預設 $M=2$。
- $P_{\max}=15$ dBm；association threshold $\rho=0.1$。
- GNN：6 layers、latent dimension 64。
- Adam：learning rate $10^{-4}$、weight decay $10^{-6}$。
- Batch size：8；training iterations：2000。
- Numerical noise：程式使用 `4e-4`；原版論文只定義 $\sigma_k^2$，simulation table 未公開實際數值。

### 4.2 相對原版程式的修改

Stage 0 保留原版 RIS-GNN 的模型架構、channel/loss 計算、訓練目標、noise、optimizer 與主要超參數，但不是直接執行未修改的初始程式。以下以 GitHub 的初始匯入版本 [`da169b0`](https://github.com/lieDownMan/decentralized-GNN-or-cf-mMIMO/commit/da169b0) 為可追溯基準，整理 Stage 1 code copy 建立前實際使用的修正；[`46f6dc3`](https://github.com/lieDownMan/decentralized-GNN-or-cf-mMIMO/commit/46f6dc3) 之後的 no-RIS 修改不屬於 Stage 0。

| 類別 | Commit | 相對初始版本的修改 | 對 Stage 0 的影響 |
|---|---|---|---|
| Evaluation fairness | [`0338e9a`](https://github.com/lieDownMan/decentralized-GNN-or-cf-mMIMO/commit/0338e9a) | Centralized inference 產生 channel 與 association 後，decentralized inference 以 `regenerate_channels=False` 重用同一批 realization，不再重新抽樣。 | 這是 continuous-phase 主結果的實質修正，使 C 與 D 成為 paired comparison；否則差值同時混入不同 test channels 的 sampling variation。 |
| Random-phase baseline | [`4a32e8b`](https://github.com/lieDownMan/decentralized-GNN-or-cf-mMIMO/commit/4a32e8b) | Continuous random phase 改為在 $[0,2\pi)$ 均勻取樣；2-bit random phase 直接從四個合法 levels 均勻取樣。原作法將正值 random vectors normalization，僅覆蓋第一象限。 | 修正 random/random-discrete controls；不改變第 4.3–4.4 節所報的 learned continuous-phase C/D 數值。 |
| Phase quantization | [`96133a2`](https://github.com/lieDownMan/decentralized-GNN-or-cf-mMIMO/commit/96133a2) | `discrete_mapping()` 改為依輸入 tensor 的 device/dtype 建立 phase levels，以 vectorized、non-mutating nearest-level mapping 取代 hard-coded CUDA 與 in-place loops，並加入四相位測試。 | 修正 CPU/非預設 GPU 執行與輸入被覆寫的風險；影響 discrete variants，不改變 continuous-phase 主表。 |
| Transmit-power interface | [`6766469`](https://github.com/lieDownMan/decentralized-GNN-or-cf-mMIMO/commit/6766469) | CLI 明確改名為 `--pmax_dbm`，在 trainer 中以 $P_{\max}[\mathrm{W}]=10^{(P_{\max}[\mathrm{dBm}]-30)/10}$ 轉換，並保留 `--Pmax` alias；sweep script 同步改名。 | 原始程式已使用同一轉換，因此此修改不意圖改變數值，只消除輸入單位歧義並明確記錄 per-AP power constraint。 |
| Training artifacts | [`f1c28ef`](https://github.com/lieDownMan/decentralized-GNN-or-cf-mMIMO/commit/f1c28ef), [`3a8bc23`](https://github.com/lieDownMan/decentralized-GNN-or-cf-mMIMO/commit/3a8bc23) | 統一 validation array 檔名的 `_run{run_id}` 格式，並在每次 iteration 寫入 `losses_run*.npy` 與 `sumrates_run*.npy`，避免空 arrays。 | 只修正可追溯性與後處理輸出，不改變 forward、loss 或 optimizer update。 |
| Runtime and imports | [`b377afe`](https://github.com/lieDownMan/decentralized-GNN-or-cf-mMIMO/commit/b377afe), [`3f5478d`](https://github.com/lieDownMan/decentralized-GNN-or-cf-mMIMO/commit/3f5478d), [`e66a768`](https://github.com/lieDownMan/decentralized-GNN-or-cf-mMIMO/commit/e66a768) | 以 explicit imports 取代 wildcard imports；註明 `_assert_finite()` 未被呼叫；關閉 expensive autograd anomaly detection。 | 屬於依賴與執行效率整理，沒有預期的數值語義變更。 |
| Plotting and repository hygiene | [`5837b9b`](https://github.com/lieDownMan/decentralized-GNN-or-cf-mMIMO/commit/5837b9b), [`5fd756f`](https://github.com/lieDownMan/decentralized-GNN-or-cf-mMIMO/commit/5fd756f) | 更新 plotting workbook path，並忽略 results、checkpoints、spreadsheets 與 caches。 | 不參與 training/evaluation 計算。 |

因此，Stage 0 應解讀為「保留原版研究方法、修正 evaluation protocol 與工程缺陷後的正向控制」，而非原始 repository 的 byte-for-byte reproduction。對本報告 continuous-phase antenna/power tables 而言，主要會改變比較有效性的修正是 C/D 共用 evaluation realizations；random-phase 與 quantization 修正只影響相應的補充 baselines。

### 4.3 Antenna sweep

以下為 continuous RIS phase、run 0 的 final evaluation：

| $M$ | Centralized | Decentralized | C − D | Relative gap |
|---:|---:|---:|---:|---:|
| 1 | 6.21930 | 5.85967 | 0.35963 | 5.782% |
| 2 | 7.66323 | 7.35767 | 0.30556 | 3.987% |
| 3 | 9.24670 | 8.88309 | 0.36361 | 3.932% |
| 4 | 10.14288 | 9.66318 | 0.47970 | 4.729% |
| 5 | 11.13875 | 10.91305 | 0.22570 | 2.026% |

結果隨 $M$ 增加，centralized/decentralized 維持約 2.0–5.8% 差距，屬於 meaningful-rate regime。

### 4.4 Power sweep

以下為 $M=2$、continuous RIS phase、run 0 的 final evaluation：

| $P_{\max}$ (dBm) | Centralized | Decentralized | C − D | Relative gap |
|---:|---:|---:|---:|---:|
| 5 | 4.59210 | 4.31794 | 0.27416 | 5.970% |
| 10 | 6.40093 | 6.05172 | 0.34921 | 5.456% |
| 15 | 7.66323 | 7.35767 | 0.30556 | 3.987% |
| 20 | 8.50135 | 8.33826 | 0.16309 | 1.918% |
| 25 | 9.33523 | 9.24256 | 0.09267 | 0.993% |
| 30 | 10.36434 | 10.36596 | −0.00162 | −0.016% |
| 35 | 11.02989 | 11.05154 | −0.02165 | −0.196% |

高功率時兩者幾乎重疊且單一 seed 下可輕微反轉，因此 Stage 0 已顯示「centralized/decentralized 接近」不必然是錯誤。這些反轉在補足多 seeds 前不作統計結論。

### 4.5 Stage 0 optimization diagnostics

- Initial task-gradient norm：約 `6.24`。
- Weight-decay gradient proxy $10^{-6}\lVert\theta\rVert_2$：約 `4.55e-5`。
- Decay/task ratio：約 `7.29e-6`。
- Near-zero parameters：約 `0.0026%`。

Stage 0 沒有 weight-decay domination 或 parameter collapse 證據。

## 5. Stage 1A：忠實 no-RIS ablation

### 5.1 固定設定

- 移除 RIS channel、RIS embeddings、RIS readout、phase output $\Theta$ 與 phase aggregation。
- 保留 direct-channel scaling、association mask、local visibility、GNN depth/width、optimizer 與 `noise_power = 4e-4`。
- Centralized GNN、decentralized GNN、MRT 與 local RZF 使用相同 channel samples、association masks、noise 與 per-AP power constraints。
- MRT/RZF 是 sanity-check baselines；GNN 不必勝過它們。

### 5.2 已完成的 antenna sweep

以下皆為 5-seed mean：

| $M$ | Centralized GNN | Decentralized GNN | MRT | RZF | Mean C − D |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.7492994e-8 | 4.7492988e-8 | 9.7781647e-8 | 9.7781647e-8 | 5.5786487e-15 |
| 2 | 7.8809052e-8 | 7.8809057e-8 | 1.8580177e-7 | 1.8580177e-7 | −4.3671733e-15 |
| 3 | 1.1301994e-7 | 1.1301995e-7 | 2.7777482e-7 | 2.7777482e-7 | −8.5140783e-15 |
| 4 | 1.4290322e-7 | 1.4290324e-7 | 3.6742964e-7 | 3.6742964e-7 | −1.4345858e-14 |
| 5 | 1.6774185e-7 | 1.6774186e-7 | 4.4651308e-7 | 4.4651308e-7 | −7.9296569e-15 |

這些數值記錄「沿用 Stage 0 數值設定後的實際結果」。GNN 低於 MRT/RZF 不是 failure criterion；真正的限制是所有 rates 都在約 $10^{-8}$ 至 $10^{-7}$，且 C − D 只剩約 $10^{-15}$ 的浮點尺度，因此無法由 Stage 1A 判斷 local/global CSI 的性能差異。

### 5.3 已完成的 power sweep

以下為 $M=2$、5 seeds 的 final evaluation；`±` 後為程式輸出的 across-seed population standard deviation（`ddof=0`）。RZF 與 MRT 在目前 noise-floor 尺度下至顯示精度相同。

| $P_{\max}$ (dBm) | Centralized GNN | Decentralized GNN | MRT/RZF | Mean C − D |
|---:|---:|---:|---:|---:|
| 5 | 7.0887171e-9 ± 5.4571481e-10 | 7.0887172e-9 ± 5.4571486e-10 | 1.8580178e-8 ± 3.6968601e-10 | −7.2830630e-17 |
| 10 | 2.4753917e-8 ± 8.7414129e-10 | 2.4753918e-8 ± 8.7414145e-10 | 5.8755681e-8 ± 1.1690497e-9 | −7.7493567e-16 |
| 15 | 7.8809052e-8 ± 3.7138001e-9 | 7.8809057e-8 ± 3.7138020e-9 | 1.8580177e-7 ± 3.6968596e-9 | −4.3671733e-15 |
| 20 | 2.6651967e-7 ± 6.1770037e-9 | 2.6651965e-7 ± 6.1770344e-9 | 5.8755674e-7 ± 1.1690495e-8 | 1.5553780e-14 |
| 25 | 8.6500487e-7 ± 2.0426195e-8 | 8.6500474e-7 ± 2.0426099e-8 | 1.8580170e-6 ± 3.6968564e-8 | 1.2586554e-13 |
| 30 | 2.8376655e-6 ± 5.0414164e-8 | 2.8376846e-6 ± 5.0406364e-8 | 5.8755604e-6 ± 1.1690461e-7 | −1.9092710e-11 |
| 35 | 9.4107902e-6 ± 1.4346843e-7 | 9.4108453e-6 ± 1.4282571e-7 | 1.8580100e-5 ± 3.6968236e-7 | −5.5033183e-11 |

Sum rate 每增加約 10 dB 便近似放大一個數量級，與 $S_k\ll\sigma^2$ 的低-SINR 線性區一致。即使在 35 dBm，rate 仍只有 $10^{-5}$ 尺度，C − D 也最多只到 $10^{-11}$；summary 中的 `gap_sign_consistent` 只是對浮點殘差計數，不是中央式優於分散式的證據。

Stage 1A 共完成 60 runs（antenna 25 + power 35）；60 份 config、checkpoint、final evaluation 與 TensorBoard log，以及 420 個 NumPy arrays 均存在且 finite。`run_exp-v2.log` 無 traceback、error、NaN、OOM 或 killed 記錄。原 tmux session 與 process 已結束，但當時未另存 numerical exit-code status file，因此不追溯宣稱其 exit code；completion 以完整 log 與 artifacts 佐證。

### 5.4 Noise 與 received-power 診斷

在 $M=2$、$K=8$、$P_{\max}=15$ dBm、seed 0 的 8192-sample MRT diagnostic：

| Quantity | Value |
|---|---:|
| Median direct-channel magnitude $\lvert h\rvert$ | 2.04155e-6 |
| Median MRT desired received power | 2.93092e-12 |
| 95th-percentile desired power | 3.18318e-11 |
| 99th-percentile desired power | 5.77671e-11 |
| Stage 1A noise | 4e-4 |
| Noise / median desired power | 1.36e8 |
| Noise / 99th-percentile desired power | 6.92e6 |

Stage 0 的 reflected-channel magnitude 中位數約為 direct channel 的 $1.24\times10^4$ 倍，約等於 $1.5\times10^8$ power ratio。移除 reflected path 後仍沿用相同 noise，因而讓 direct-only objective 落入 noise floor。

### 5.5 Optimization diagnostics

在 $M=2$、$K=8$、$P_{\max}=15$ dBm、seed 0 的 initial batch：

- Task-gradient norm：`1.13523e-7`。
- Parameter norm：`28.0321`。
- Weight-decay gradient proxy：`2.80321e-5`。
- Decay/task ratio：`246.93`。
- 既有 checkpoint diagnostic 顯示約 60% parameters 接近零，update layers 明顯 collapse。

`weight_decay=1e-6` 並非在所有設定下都太大；它只是在 Stage 1A task gradient 消失後，相對變成 task gradient 的約 247 倍。

## 6. Stage 1B：no-RIS 數值尺度校準

### 6.1 修改內容與控制變因

Stage 1B 只把 no-RIS pipeline 的 noise 改為可設定參數，目的在排除 Stage 1A 已確認的 noise-floor confound。具體修改如下：

| 修改位置 | 修改內容 | 原因 |
|---|---|---|
| `code/stage1/trainer_2.py` | 新增 CLI `--noise_power`，傳入 training、centralized/decentralized evaluation 與 config；每個 run 保存 effective noise。 | 讓 sensitivity settings 可重現，避免 training 與 evaluation 使用不同 noise。 |
| `code/stage1/data.py`、`utils_return_indivial_rates.py` | `compute_loss()` 與 `cal_loss()` 接受 `noise_power`，SINR denominator 由固定常數改為該 run 的值。 | Noise 必須與 direct-channel power scale 一致，否則 objective 會落入近零梯度區。 |
| `code/stage1/utils_return_indivial_rates.py` | Local RZF 的 regularization 改為 $\alpha=K_a\sigma^2/P_{\max}$，使用同一個 run 的 noise。 | 保持 RZF 與 GNN 在相同 noise 假設下比較；若只修改 loss 而保留舊 RZF regularization，baseline protocol 會不一致。 |

為隔離 noise 的作用，以下項目全部保持不變：direct-channel generator 與 scaling、5 APs、$K=8$、$M=2$、$P_{\max}=15$ dBm、association threshold $0.1$、centralized-training/decentralized-inference 定義、6-layer/64-dimension GNN、batch size 8、Adam learning rate $10^{-4}$、weight decay $10^{-6}$ 與 seed 0。未加入 gradient clipping、loss scaling，亦未為超越 MRT/RZF 而修改模型。

程式驗證已於 2026-08-18 完成：`py_compile`、`code/stage1/test_stage1.py`，以及 `noise_power=1e-12`、2 iterations、8-sample final evaluation 的 CPU smoke check 均通過。Smoke check 只驗證執行路徑與 finite outputs，不納入性能分析。

### 6.2 為何必須修改 noise

No-RIS sum-rate objective 使用

$$
R=\sum_k\log_2\left(1+\frac{S_k}{I_k+\sigma^2}\right).
$$

Stage 1A 的 MRT diagnostic 顯示 desired received power 中位數只有 $2.93092\times10^{-12}$，而沿用自 RIS 程式的 `noise_power=4e-4` 是它的 $1.36\times10^8$ 倍。在 $S_k\ll\sigma^2$ 時，$\log_2(1+S_k/\sigma^2)$ 與其對 beamformer 的梯度都接近零；因此 Stage 1A 的 task-gradient norm 只有 `1.13523e-7`。此時 weight-decay gradient proxy 雖僅約 `2.80321e-5`，相對 task gradient 卻大約是 247 倍，optimizer update 因而主要受到 decay contribution 影響，最後出現 parameter collapse。嚴格而言，是「optimizer update 被 weight decay 主導」，不是記錄的 task loss 本身包含或被替換成 decay loss。

改為主設定 `noise_power=1e-12` 後，noise 約為 MRT desired received power 中位數的 `0.341` 倍，不再比 direct-link signal 高數百萬至上億倍。這不代表 noise 從系統消失，而是讓 signal、interference 與 noise 回到可分辨的尺度，使 beamformer 改變能在 objective 中產生可用梯度。

### 6.3 論文依據與採用邊界

| Source | 可核對證據 | 本研究如何使用 |
|---|---|---|
| [Hojatian et al., *IEEE Communications Letters*, 2022, Sec. V and Fig. 2](https://doi.org/10.1109/LCOMM.2022.3157161)；[arXiv full text](https://arxiv.org/abs/2106.16194) | 論文以 $\sigma^2=-130$ dBW 報告主表，並在 sensitivity experiment 使用 $-110$ 至 $-130$ dBW；考慮 channel attenuation 後，平均 SNR 為 3.1–23.1 dB。論文同節明載 batch size 1000、learning rate $10^{-3}$、weight decay $10^{-6}$。 | 將 $-80$、$-90$、$-100$ dBm（即 $10^{-11}$、$10^{-12}$、$10^{-13}$ W）作為 no-RIS decentralized beamforming 的外部 sensitivity prior。主設定事前固定為中點 $10^{-12}$，不依本研究的 C/D gap 選點。 |
| Hojatian et al. reference code：[training settings](https://github.com/HamedHojatian/CF-mMIMO-HBF/blob/master/Main_CF.py#L51-L59)、[optimizer 與 loss](https://github.com/HamedHojatian/CF-mMIMO-HBF/blob/master/Main_CF.py#L139-L150) | 官方 code 使用 Adam、`lr=0.001`、`wd=1e-6`、batch 1000，並將相同 `Noise_pwr` 傳入 sum-rate loss。 | 支持保留 $10^{-6}$ 作 weight-decay prior，但不跨資料集與架構照抄 learning rate 或 batch size。 |
| [Tung et al., *IEEE Transactions on Vehicular Technology*, 2025, Sec. II](https://doi.org/10.1109/TVT.2024.3493235)；[accepted manuscript](https://pure.qub.ac.uk/files/629167011/Main_short_version.pdf) | Cell-free GNN model 將 additive noise 寫成 $\mathcal{CN}(0,1)$，並以 normalized SNR 表示 pilot/downlink power。 | 作為「channel、power 與 noise 必須使用同一 normalization」的佐證；不提供可直接複製到本 channel generator 的 raw noise。 |
| [Huang et al.](https://arxiv.org/abs/2006.12238) | 以 noise PSD $-174$ dBm/Hz 與 1 GHz bandwidth 建立 physical noise model。 | 僅作 physical-model cross-check，不直接套入目前的 normalized channel。 |

Hojatian et al. 與本研究使用不同 channel model、天線配置與 power normalization，因此上述範圍是校準先驗，不是「該論文證明本研究的最佳 noise」。本研究仍以自己的 received-power、gradient 與 training diagnostics 判定尺度是否合理。

### 6.4 預先登記的 selection rule 與 numerical gates

- Noise 不得依 centralized/decentralized gap 或是否勝過 MRT/RZF 選擇。
- 主設定固定為 `1e-12`；`1e-11` 與 `1e-13` 只作 sensitivity。
- Task-gradient 相對 Stage 0 initial value 必須在 0.1–10 倍內。
- Decay/task ratio 必須小於或等於 `1e-2`。
- Training curve 不得維持 noise-floor 平線，且不得重現 Stage 1A 型 parameter collapse。
- 即使正常學習後 GNN 仍低於 MRT/RZF，也如實報告，不為超越 baseline 而調參。

### 6.5 Initial-batch sensitivity diagnostic

下表使用同一個 seed-0 initial batch；它用來檢查尺度與 gradient，不是 final performance：

| Noise | Equivalent dBm | Initial sum rate | Task-gradient norm | Relative to Stage 0 gradient | Decay/task ratio | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 4e-4 | −4 | 1.65707e-8 | 1.13523e-7 | 1.82e-8 | 246.93 | FAIL：Stage 1A negative control |
| 1e-11 | −80 | 0.466641 | 2.93413 | 0.470 | 9.55381e-6 | PASS |
| **1e-12** | **−90** | **1.65857** | **9.42354** | **1.510** | **2.97469e-6** | **PASS：預先指定主設定** |
| 1e-13 | −100 | 2.71153 | 15.8579 | 2.541 | 1.76771e-6 | PASS |

三個 Stage 1B settings 的 task gradient 都在 Stage 0 的 0.1–10 倍 gate 內，且 decay/task ratio 比 $10^{-2}$ 上限低至少三個數量級。主設定 `1e-12` 的 task gradient 約為 Stage 0 的 1.51 倍，因此保留原 learning rate 與 weight decay，而不新增 gradient clipping 或 loss scaling。

### 6.6 Seed-0 200-iteration pilot 結果

以下為每個 noise 各一個 seed、3200 final samples 的 sum rate；centralized GNN、decentralized GNN、MRT 與 RZF 共用 channels、association masks、noise 與 per-AP power constraints。

| Noise | Centralized GNN | Decentralized GNN | MRT | RZF | C − D | $(C-D)/C$ |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-11 | 3.334029 | 3.343003 | 3.674726 | 3.677043 | −0.008974 | −0.269% |
| **1e-12** | **8.878122** | **8.898171** | **9.415360** | **10.035103** | **−0.020049** | **−0.226%** |
| 1e-13 | 11.612363 | 11.600189 | 12.244229 | 13.639974 | 0.012174 | 0.105% |

三個 noise settings 都已離開 Stage 1A 的 $10^{-8}$–$10^{-7}$ rate scale。C/D gap 的絕對相對值皆小於 0.27%，並在 `1e-13` 改變正負號，因此這個 single-seed pilot 不能支持 centralized 或 decentralized 的穩健優勢。主設定 checkpoint 以相同 BS geometry 與 deterministic final channel sequence 作 read-only replay，可在 $10^{-6}$ 誤差內重現 C/D means，確認 summary 並非 parsing artifact；這項重播沒有重新訓練模型，因此 verification status 仍為 `ANALYZED`，不是完整 reproducibility verification。

GNN 在三個 settings 仍比最佳 MRT/RZF baseline 低約 9.08%、11.33%、14.87%。這不是 Stage 1B failure criterion，但必須保留為後續 full-run 的性能限制。RZF 相對 MRT 的優勢則從 `1e-11` 的 0.06% 增至 `1e-13` 的 11.40%，顯示 noise 降低後 interference management 的影響變得更明顯。

### 6.7 為何改後確實恢復學習，而不只是 rate 分母變小

降低 $\sigma^2$ 會直接提高 SINR，因此不同 noise settings 的 final rates 不能全部歸因於模型學習。Stage 1B 另外使用「同一 noise 內的 training trajectory」與 checkpoint diagnostics 判斷是否恢復可學習性：

| Noise | First-50 train mean | Last-50 train mean | Change | Final parameter norm | Near-zero ratio $(\lvert\theta\rvert<10^{-5})$ |
|---:|---:|---:|---:|---:|---:|
| 1e-11 | 1.029054 | 3.184492 | +209.46% | 29.2736 | 0.02084% |
| **1e-12** | **3.958584** | **8.691864** | **+119.57%** | **29.3503** | **0.02262%** |
| 1e-13 | 6.300444 | 11.490754 | +82.38% | 28.6589 | 0.02003% |

初始化 parameter norm 為 `28.0319`，near-zero ratio 為 `0.02166%`。三個 final checkpoints 的 norm 只改變約 2.24–4.70%，near-zero ratio 也與初始化相當，且所有 parameters 與 training metrics 都是 finite。相較之下，Stage 1A checkpoint 曾有約 60% parameters 接近零。這表示 Stage 1B 的改善包含真正的 task-driven update，而不是單純把相同 collapsed output 放進較小的 noise denominator。

主設定的 last-50 training slope 已接近零，支持 200 iterations 足以作尺度校準；但 training batches 每次重新抽樣且曲線仍有變異，因此不能把 pilot 當成 full convergence 證據。完整性能仍以預定的 2000-iteration runs 為準。

### 6.8 Pilot gate decision 與事前登記的 full run

**Decision：Stage 1B seed-0 numerical calibration 通過；主設定維持預先指定的 `noise_power=1e-12`。** 通過原因是 task-gradient、decay/task ratio、non-flat training curve、finite outputs 與 no-collapse diagnostics 同時符合 gate，而不是因為 C/D gap 或 GNN/baseline 勝負符合期待。

目前限制如下：

1. 每個 noise 只有 seed 0；`final_summary.txt` 的 `std: 0` 只是 $n=1$，不是零變異證據。
2. `gap_sign_consistent: True` 在單一 seed 下由 `1/1` 自動成立，不具有統計意義；正式判斷至少需 5 seeds。
3. Pilot 只有 200 iterations，但 `log_eval_interval=500`，所以 intermediate validation arrays 為空；目前有完整 training arrays 與一次 3200-sample final evaluation，沒有獨立 validation trajectory。
4. 本節證明的是 numerical scale 已可學習，不證明 decentralized 等同 centralized，也不證明 GNN 已優於 conventional beamforming。

下一個 mandatory gate 事前登記為使用 `noise_power=1e-12` 執行 5 seeds、2000 iterations。Noise 不得根據 pilot 的約 0.2% C/D gap、gap sign 或 baseline 排名重新選擇；full run 必須保存跨 seed mean/std、每個 seed 的 paired C − D，以及多個 intermediate validation checkpoints。這個 gate 已於 2026-08-19 完成，結果如下。

### 6.9 Full run 遠端執行記錄

5-seed full run 於 2026-08-18 排到遠端 Stage 1A power sweep 後方。為避免修改當時正在執行的工作目錄，Stage 1B source 被同步到獨立目錄 `~/ThomasLai/code/stage1b_full_source/`，並在遠端 Conda 環境通過 `py_compile` 與 `test_stage1.py`。排程 session 等待 Stage 1A 主工作 PID `2980456` 結束，且啟動前確認 output root 不存在，避免覆寫既有結果。

固定執行參數為：`M=2`、`K=8`、`Pmax=15 dBm`、`batch_size=8`、`seeds=0–4`、`n_iter=2000`、`noise_power=1e-12`、`test_sample_val=128`、`test_sample_final=3200`、`device=cuda:0`。狀態檔記錄 `started_at=2026-08-19T02:44:51+08:00`、`completed_at=2026-08-19T04:54:30+08:00`、`exit_code=0`。

### 6.10 5-seed、2000-iteration full-run 結果

以下為每個 seed 的 3200-sample final evaluation；四種方法在每個 seed 內共用 channels、association masks、noise 與 per-AP power constraints。

| Seed | Centralized GNN | Decentralized GNN | MRT | RZF | C − D |
|---:|---:|---:|---:|---:|---:|
| 0 | 9.904400 | 9.898510 | 9.509532 | 10.065721 | 0.005890 |
| 1 | 9.999804 | 9.986318 | 9.554633 | 10.131932 | 0.013486 |
| 2 | 9.766613 | 9.767665 | 9.393886 | 9.963727 | −0.001052 |
| 3 | 9.880801 | 9.862543 | 9.479312 | 10.109447 | 0.018259 |
| 4 | 9.934161 | 9.920557 | 9.519068 | 10.131453 | 0.013605 |
| **Mean ± SD** | **9.897156 ± 0.076508** | **9.887118 ± 0.072058** | **9.491286 ± 0.054310** | **10.080456 ± 0.063141** | **0.010037 ± 0.006817** |

`final_summary.txt` 的 SD 使用 `np.std` 預設值（`ddof=0`）。C − D 在 4/5 seeds 為正，通過事前登記的 mechanical sign-consistency gate，但 mean gap 只是 centralized mean 的 `0.101%`。對五個 paired seed gaps 作雙尾 exact sign-flip test 得 $p=0.125$；因此不宣稱 centralized 對 decentralized 有穩健優勢，也不以「未顯著」當作二者等效的證據。

Centralized 與 decentralized GNN 對 MRT 的 mean improvement 分別為 4.28% 與 4.17%，五個 seeds 方向一致；對 RZF 則分別低 1.82% 與 1.92%，五個 seeds 也方向一致。這些是 $n=5$ 的 descriptive results：2000 iterations 後 GNN 已超過 MRT，但仍未超過 local RZF。

### 6.11 Full-run learning 與 artifact diagnostics

| Seed | First-50 train mean | Last-50 train mean | Change | Final parameter norm | Near-zero ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.958584 | 9.757540 | +146.49% | 30.4300 | 0.02275% |
| 1 | 6.742641 | 9.896534 | +46.78% | 30.5217 | 0.01948% |
| 2 | 4.846977 | 9.933859 | +104.95% | 30.4074 | 0.02234% |
| 3 | 6.317369 | 9.730798 | +54.03% | 30.4956 | 0.02384% |
| 4 | 4.821261 | 9.757615 | +102.39% | 30.3669 | 0.01962% |

每個 seed 都保存 2000 點 training arrays 與 iterations 500/1000/1500/2000 的四組 intermediate validation metrics。五個 checkpoints 的 parameters 全部 finite，near-zero ratio 只有 0.0195–0.0238%，未重現 Stage 1A 約 60% near-zero 的 collapse。遠端完整性檢查找到 5 configs、5 checkpoints、5 final evaluations、5 TensorBoard logs 與 35 NumPy arrays；全部 arrays/final metrics 均 finite，log 無 traceback、error、NaN、OOM 或 killed 記錄。

來源檔的 read-only SHA-256 比對也確認，遠端 `stage1b_full_source/` 中實際 staged 的 `data.py`、`model_2.py`、`trainer_2.py`、`utils_return_indivial_rates.py` 與 `test_stage1.py` 逐檔匹配目前本地 `code/stage1/`。該 staged directory 沒有 `run_exp-v2.sh`；full-run 參數以每個 run 的 `config.json` 與 completion status 追蹤，不把未 staged 的 script 誤列為執行來源。

**Decision：Stage 1B full-run numerical/learning gate 通過，`noise_power=1e-12` 可作 Stage 2 的 no-RIS snapshot baseline。** 這個決定來自事前固定的 noise、finite/artifact completeness、五個 seeds 的 non-flat learning 與 no-collapse diagnostics，不來自 C/D gap 或 GNN/baseline 排名。

### 6.12 Statistical validation 與 reproducibility boundary

- Numerical/learning gate confidence：`SOLID`；事前登記的設定完成五個 seeds，artifacts 完整，且沒有 non-finite 或 collapse 證據。
- C/D directional claim confidence：`CAUTION`；$n=5$、mean practical gap 只有 0.101%，exact paired test $p=0.125$。
- Fallacy scan coverage：11/11。Simpson、ecological、Berkson 與 collider fallacies 未被這個 paired-seed design 觸發；base-rate neglect、regression to the mean 與 survivorship bias 不適用；所有事前登記 settings/seeds 皆已報告，未發現 look-elsewhere 或 garden-of-forking-paths 的選擇性報告；本報告不作 observational causal 或 reverse-causality 主張。
- Reproducibility method：本次只讀取遠端 raw artifacts、config、log 與 checkpoint diagnostics，未獨立重新訓練；因此狀態是 `ANALYZED`，獨立 reproducibility verdict 為 `CANNOT_VERIFY`。

## 7. Current Interpretation

1. Stage 0 在 meaningful-rate regime 下可觀察約 2–6% centralized/decentralized gap，因此原版 code 並非全面失效。
2. Stage 1A 的 antenna/power sweeps 已全部完成且 artifacts finite。它是忠實且有價值的 negative control：移除主導訊號的 RIS path 後仍沿用 `noise_power=4e-4`，會使 direct-only SINR、sum rate 與 task gradient 一起落入 noise floor，optimizer update 再被 weight decay contribution 主導。
3. Stage 1B 將 noise 調整到與 direct-link received power 一致的量級後，task gradient、五個 seeds 的 training improvement 與 checkpoint distribution 都恢復正常；因此 `1e-12` 通過 numerical/learning gate，可作後續 no-RIS snapshot baseline。
4. `1e-12` 是事前指定的 literature-grounded 中點，不是根據 centralized/decentralized gap 或 baseline 勝負事後挑選。`1e-11` 與 `1e-13` 只證明結論對相鄰 noise scale 的 numerical sensitivity。
5. Stage 1B full run 的 C − D 在 4/5 seeds 為正，但 mean gap 僅 0.0100（0.101%）且 exact paired sign-flip $p=0.125$；這不支持穩健優勢，也不支持等效結論。
6. 2000 iterations 後 centralized/decentralized GNN 的 mean rate 比 MRT 高 4.28%/4.17%，但比 RZF 低 1.82%/1.92%。GNN 是否超越 baselines 不是 Stage 1B gate；增加 AP/UE 數量只能作 robustness test，不能當作修復 noise scale 的方法。

## 8. Artifact Index

| Artifact | Path / Location |
|---|---|
| Stage 0 paper | `doc/Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks.pdf` |
| Stage 0 antenna results | `code/results_batch_8_BS-radius_200_RIS-radius_100_vary_M/` |
| Stage 0 power results | `code/results_batch_8_BS-radius_200_RIS-radius_100_vary_Pmax/` |
| Stage 1 implementation plan | `doc/stage1_no_ris_implementation_plan.md` |
| Stage 1A remote antenna results | `lab301-5090:~/ThomasLai/code/stage1/results_stage1_vary_M/` |
| Stage 1A remote power results | `lab301-5090:~/ThomasLai/code/stage1/results_stage1_vary_Pmax/` |
| Stage 1A remote log | `lab301-5090:~/ThomasLai/code/stage1/run_exp-v2.log` |
| Stage 1B local calibration | `code/stage1/results_stage1b_calibration/`；三個 noise settings 的 config、raw arrays、TensorBoard events、checkpoints、final evaluations 與 summary 均完整 |
| Stage 1B remote staged source | `lab301-5090:~/ThomasLai/code/stage1b_full_source/` |
| Stage 1B remote completion status | `lab301-5090:~/ThomasLai/code/stage1/stage1b_full_noise_1e-12.status`；completed 2026-08-19 04:54:30 +08:00，exit code 0 |
| Stage 1B remote log / results | `lab301-5090:~/ThomasLai/code/stage1/stage1b_full_noise_1e-12.log`；`lab301-5090:~/ThomasLai/code/stage1/results_stage1b_full_noise_1e-12/` |
| Related-work review | `doc/related_work_decentralized_active_csi.md` |

## 9. Experiment Log

| Date | Stage | Event | Status | Evidence / Decision |
|---|---|---|---|---|
| 2026-08-18 | Stage 0 | Existing antenna/power artifacts recorded | Complete, $n=1$ | Positive control；未宣稱多-seed 統計結論 |
| 2026-08-18 | Stage 1A | Antenna sweep $M=1,\ldots,5$ | Complete, 5 seeds/設定 | 所有 rates 約 $10^{-8}$–$10^{-7}$；C − D 約 $10^{-15}$ |
| 2026-08-18 | Stage 1A | Power sweep | In progress | `Pmax=5` seeds 0–2 完成；檢查時 seed 3 iteration 1950/2000 |
| 2026-08-18 | Stage 1B | Noise/gradient/weight-decay plan registered | Not started | 主設定 `1e-12`；sensitivities `1e-11`, `1e-13` |
| 2026-08-18 | Stage 1B | Configurable noise implementation validated | Complete | `py_compile`、`test_stage1.py` 與 `1e-12` 2-iteration CPU smoke check 通過；loss 與 RZF 共用 effective noise |
| 2026-08-18 | Stage 1B | Local 200-iteration noise calibration | Complete, seed 0 | `1e-11`、`1e-12`、`1e-13` 全部完成；artifacts `code/stage1/results_stage1b_calibration/` |
| 2026-08-18 | Stage 1B | Numerical gate analysis | Pass with $n=1$ limitation | 三組 gradient/decay/curve/no-collapse gates 皆通過；主設定維持預先指定的 `1e-12`；C/D 與 baseline 勝負未參與 selection |
| 2026-08-18 | Stage 1B | Remote 5-seed full run queued | Queued / waiting | tmux `thomaslai_stage1b_full` 等待 Stage 1A 主工作 PID `2980456`；5 seeds × 2000 iterations；遠端 compile/test 通過且 output 尚未建立 |
| 2026-08-19 | Stage 1A | Power sweep completion audit | Complete, 5 seeds/設定 | 7/7 summaries、35/35 configs/checkpoints/final evaluations/logs 完整；arrays finite，log 無 anomaly；noise-floor 結論不變 |
| 2026-08-19 | Stage 1B | Remote 5-seed full run | Complete, exit code 0 | 02:44:51 啟動、04:54:30 完成；5 seeds × 2000 iterations；5/5 artifact sets 完整且 finite |
| 2026-08-19 | Stage 1B | Full-run validation | Numerical gate pass; C/D inference caution | C − D mean 0.0100（0.101%）；4/5 正；exact paired sign-flip $p=0.125$；GNN 高於 MRT 但低於 RZF |

## 10. Next Update Checklist

- [x] 補上 Stage 1A 所有 Pmax 的 5-seed mean/std 與 artifact completion status。
- [x] 記錄 Stage 1A 最終狀態：session/process 已結束、artifacts/log 完整、無 anomaly；當時未保存 numerical exit code，不事後推定。
- [x] 實作可設定的 Stage 1B noise，讓 training/evaluation/RZF 共用 effective value；compile、existing Stage 1 test 與 2-iteration smoke check 已通過。
- [x] 執行三個 noise values 的 200-iteration seed-0 pilot，補上 initial-gradient、training-curve、checkpoint 與 final-evaluation diagnostics。
- [x] 依預先登記 gates 判定主設定 `1e-12` 通過 seed-0 numerical calibration。
- [x] 完成固定 `noise_power=1e-12` 的 5-seed、2000-iteration full run，確認 intermediate validation/artifact completeness，並分析跨 seed mean/std 與 paired C − D。
- [ ] 若需將 verification status 從 `ANALYZED` 升為 `VERIFIED`，在保存原 artifacts 的前提下進行一次獨立重新訓練與 tolerance comparison。
