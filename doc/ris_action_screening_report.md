# RIS action representation — 10k screening、diagnostics 與 500k policy conversion

## Material Passport

- Origin Date: 2026-09-15
- Last Updated: 2026-09-15（方法定義集中至獨立文件）
- Verification Status: PARTIAL（單一 training seed 與固定 topology；10k retraining 仍只屬 screening）
- Version Label: `ris_action_screening_v5`
- Experiment plan: [RIS action representation 實驗計畫](./ris_action_experiment_plan.md)
- Methods: [Decentralized RIS 方法與通訊介面](./decentralized_ris_methods.md)
- Current evidence: [Decentralized RIS 現況與證據報告](./decentralized_ris_evidence.md)
- Pre-registration: [prereg_r1_ap_ris_mag_zero_training.md](../artifacts/decentralized_ris/evaluation/prereg_r1_ap_ris_mag_zero_training.md)

方法、表示法、payload 與 `2-bit` protocol 統一見[方法文件](./decentralized_ris_methods.md)。
10k retraining 仍只構成 screening trend；§7.6 另在成熟 500k R0 checkpoint 上使用 3,200 samples
評估不需重訓的介面轉換。

## 1. 結論

1. R0 與 R0c 數值等價。R1-shared 重訓後 decentralized 落後 R0
   **−1.637 ± 0.255（−14.5%）**；centralized 退化較小，且是否可辨識會隨 holdout 改變，
   因此只能說主要缺口出現在 CT→DI transfer，不能說 centralized 完全不受影響。
2. Diagnostics：local CSI 確實使 proposal 分散，但 representation 造成的分散更大；
   magnitude 與 leave-one-AP-out contribution 正相關（+0.423 ± 0.034）卻與 angular
   agreement 負相關（−0.138 ± 0.027），因此只能稱 importance weight，不能稱 confidence。
3. `r1_ap_ris_mag` 在**零訓練**下通過預先登記的
   non-inferiority 檢定：對 R0 的損失 −0.025 ± 0.040，單側 95% 上界 +0.043，遠低於
   δ = 0.5（§7.2）。
4. 但**以同一規則重新訓練失敗**：matched 10k 的 `r1_ap_ris_mag` 對 R0 落後
   +1.384 ± 0.136（上界 +1.612 ≫ 0.5），三項推進門檻全部未過（§7.4）。
5. 兩者的落差顯示：加權規則在**推論端**能近乎無損地讀出 R0 policy，但以同一介面從頭訓練
   會得到較差 policy。證據把問題縮小到含 per-AP projection 的訓練參數化／最佳化路徑，
   尚未單獨證明 projection Jacobian 就是唯一原因。
6. 在 500k R0 checkpoint 上，R1-shared decentralized 損失擴大到 $4.409\pm0.115$，但
   `r1_ap_ris_mag` 與 R0 的差只有 $+0.0419\pm0.0073$（約 +0.2%）。這支持把 AP–RIS magnitude weighting 當成成熟
   R0 policy 的低損失推論轉換；不把這個微小正差解讀為方法優越性。

## 2. 執行摘要

| Step | 內容 | 結果 |
|---|---|---|
| 1 | `py_compile` 與 deterministic tests（本地） | 全數通過 |
| 2 | 同一 R0 checkpoint 評估 r0／r0c／r1_shared-no-retrain | §3 |
| 3 | R1-shared 500-iteration smoke | 通過 |
| 4 | Matched 10k，seed 0 | R0 需補跑，§5 |
| 5 | Paired screening，seed 20260915 | §4 |
| D | Counterfactual diagnostics | §6 |
| A–C | `r1_ap_ris_mag` 凍結、零訓練確認、matched 10k | §7 |

## 3. Step 2：零訓練 control（seed 20260915）

| 指標 | R0 | R0c | R0c−R0 | R1-shared | Δ vs R0 |
|---|---:|---:|---:|---:|---:|
| decentralized | 10.08966 | 10.08966 | 1.9e-06 | 9.55721 | −0.532 ± 0.148 |
| decentralized 2-bit | 9.50079 | 9.50079 | 0.0 | 9.08002 | −0.421 ± 0.117 |
| centralized | 11.18366 | 11.18366 | 1.9e-06 | 11.36839 | +0.185 ± 0.137 |

## 4. Step 5：matched 10k paired screening（seed 20260915）

| 指標 | R0 | R1-shared | diff | SEM | t |
|---|---:|---:|---:|---:|---:|
| decentralized | 11.29048 | 9.65345 | −1.63703 | 0.255 | −6.42 |
| decentralized 2-bit | 10.54693 | 9.30035 | −1.24658 | 0.236 | −5.27 |
| centralized | 12.26384 | 11.95897 | −0.30487 | 0.250 | −1.22 |

## 5. Step 4 的偏離：R0 10k 重跑

遠端 `results_ris_action/r0-baseline_iter10000_seed0` 不符合重用條件：舊的
`trainer_2.eval()` 沒有 re-seed，validation batch 取自進行中的 training RNG stream，
既非固定 validation set，也消耗了 RNG state 使 training sequence 分岔。故以現行程式補跑。
兩個 run 在本機 RTX 5060 Laptop 平行執行。

## 6. Counterfactual diagnostics（seed 20260915，discovery set）

程式：[experiments/proposal_diagnostics.py](../code/decentralized_ris/experiments/proposal_diagnostics.py)。

### 6.1 Proposal concentration

| Policy | centralized | decentralized | drop |
|---|---:|---:|---:|
| r0_policy | 0.9576 | 0.8015 | −0.1561 ± 0.0051 |
| r1_shared | 0.7201 | 0.6275 | −0.0926 ± 0.0044 |

Local CSI 確實降低 concentration，但 centralized 下 r1_shared 已比 r0_policy 低
−0.2376 ± 0.0050：equal consensus 下訓練本身就不迫使 proposal 對齊。

### 6.2 Magnitude 不是 confidence

| Spearman | r0_policy | r1_shared |
|---|---:|---:|
| ‖z‖ vs angular agreement（per AP–RIS） | −0.1382 ± 0.0268 | −0.1253 ± 0.0422 |
| ‖z‖ vs leave-one-AP-out contribution | +0.4230 ± 0.0343 | +0.2080 ± 0.0418 |
| angular agreement vs leave-one-AP-out | −0.0315 ± 0.0417 | +0.0525 ± 0.0391 |

Magnitude 預測邊際貢獻，不預測相位一致度；agreement 本身不預測貢獻。故命名為
importance weight。

### 6.3 Granularity 與 weighting ladder

Agreement 的變異 74–88% 在 element 層級，magnitude 則 68–75% 在 pair 層級。零訓練 ladder
（decentralized，各自用自己的 $W$）：

| 加權 | r0_policy | vs equal | 佔 r0c 差距 |
|---|---:|---:|---:|
| equal（= R1-shared 規則） | 10.08968 | — | — |
| per-element ‖z‖（= R0c） | 11.29048 | +1.2008 ± 0.1843 | 100.0% |
| **per-AP–RIS 平均 ‖z‖** | **11.36983** | **+1.2802 ± 0.1958** | **106.6%** |

`per-element ‖z‖` 復現 R0c 至小數五位，是程式正確性驗證。Per-pair 與 per-element 差
+0.0794 ± 0.0427（不顯著），故 scalar 不會太粗。

### 6.4 2×2 交叉替換

| 對比 | Decentralized | Centralized |
|---|---:|---:|
| 只換 phase（W = R0） | −1.4810 ± 0.2387 | −0.3347 ± 0.2288（不顯著） |
| 只換 beamformer（θ = R0） | −0.5508 ± 0.0991 | −0.5379 ± 0.1001 |
| 全換 | −1.6370 ± 0.2551 | −0.3049 ± 0.2496 |

Phase 是主因，但 beamformer 另有約 −0.54、與 mode 無關的缺口，比較 finalist 時須分開計。

## 7. AP–RIS magnitude weighting experiments

### 7.1 Definition-lock verification（A）

本輪凍結的介面與聚合公式見[方法文件 §3.4](./decentralized_ris_methods.md#r1-ap-ris-mag)。
實作使用 `--arch r1_ap_ris_mag --consensus ap_ris_mag`。
[tests/test_action_interface.py](../code/decentralized_ris/tests/test_action_interface.py)
斷言：strict-load R0 state dict、`effective_parameters` 1,664,445 與 R0 相同、
`cpu_trainable_parameters` 0、輸出 unit modulus、uniform 權重可精確還原 equal 規則，且
`trace["weights"]` 符合凍結定義。

### 7.2 零訓練確認（B，seed 20260916，預先登記）

Arms 與 δ = 0.5 於評估前寫入 pre-registration 檔案。

| arm | centralized | decentralized | dec 2-bit | R0 − arm（dec） | 單側 95% 上界 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| R0 | 12.06821 | 11.23396 | 10.53634 | — | — | — |
| R0c | 12.06821 | 11.23396 | 10.53634 | +0.0000 | +0.0000 | 非劣 |
| r1_equal | 12.18151 | 10.04907 | 9.49288 | +1.1849 ± 0.1409 | +1.4212 | 未過 |
| **r1_ap_ris_mag** | 12.08427 | **11.25916** | 10.57165 | **−0.0252 ± 0.0404** | **+0.0426** | **非劣** |

次要：2-bit +0.0353 ± 0.0519、centralized +0.0161 ± 0.0161，皆無退化；R0c−R0 最大差
1.9e-06；unit-modulus error 1.19e-07。

### 7.3 Matched 10k 訓練（C）

500-step smoke 通過（unit-modulus error 1.19e-07，rate 2.25 → 3.82）。以 training seed 0
跑 10,000 iterations，wall clock 0.96 h。$s_{lr}$ 對 $z$ 可微，故 loss 同時訓練 GNN、
$W_{\rm reduce}$ 與 beamformer。

### 7.4 訓練後評估（seed 20260917）

| run | centralized | decentralized | dec 2-bit |
|---|---:|---:|---:|
| R0 | 11.95278 | 11.17826 | 10.55499 |
| r1_shared | 11.57344 | 9.51626 | 9.06819 |
| r1_ap_ris_mag | 11.06578 | 9.79427 | 9.28186 |

| 門檻 | 判定 |
|---|---|
| 1. decentralized 對 R0 非劣（δ = 0.5） | **未過**：損失 +1.3840 ± 0.1362，上界 +1.6124 |
| 2. 明顯消除 CT→DI gap | **部分**：gap 由 r1_shared 的 2.0572 ± 0.1997 降到 1.2715 ± 0.1596（−0.7857 ± 0.1303，t = −6.03），但仍高於 R0 的 0.7745 ± 0.1371（超出 +0.4970 ± 0.1005，t = 4.94） |
| 3. 2-bit 無新退化 | **未過**：對 R0 −1.2731 ± 0.1198 |

對照 r1_shared：decentralized +0.2780 ± 0.1578（t = 1.76，不顯著）、2-bit
+0.2137 ± 0.1453（不顯著）、centralized **−0.5077 ± 0.1557（t = −3.26，顯著較差）**。
Validation 末四點 decentralized 斜率：R0 +2.018、r1_shared −0.805、r1_ap_ris_mag +0.336
bps/Hz per 10k——加權確實止住了 r1_shared 的下滑，但同時壓低了 centralized 能力，
淨效果在 decentralized 上不顯著。

相對 R0，`r1_ap_ris_mag` 的 centralized 差為 $-0.8870\pm0.1464$，95% CI 約
$[-1.181,-0.593]$。因此這次從頭訓練的問題不只出現在 local inference；global-CSI
training objective 本身也學到較差解。

### 7.5 為什麼這個失敗有資訊量

同一個加權規則，在推論端可以近乎無損讀出 R0 policy（§7.2），在訓練端卻學不出同等 policy
（§7.4）。因此，equal consensus 在推論時丟掉的 magnitude 資訊並非不可傳遞；pair-level
scalar 足以為既有 R0 policy 補回大部分資訊。從頭訓練失敗則把問題指向新的參數化與最佳化
路徑：$\Pi$ 對 $z_l$ 的整體尺度不變，方向與 magnitude 經不同梯度路徑進入 loss，可能改變
conditioning 與 beamformer–phase co-adaptation。這些是與觀察一致的候選機制，仍需 warm-start、
freeze 或 projection schedule ablation 才能區分，不能由本輪結果指定唯一原因。

### 7.6 成熟 500k R0 policy 的零訓練轉換（seed 20260918）

使用同一個 500k R0 checkpoint、3,200 samples／400 個 batch clusters，分別套用四種介面：

| arm | centralized | decentralized | dec 2-bit | 對 R0 的 dec paired difference |
|---|---:|---:|---:|---:|
| R0 | 23.58559 | 21.38124 | 19.80492 | — |
| R0c | 23.58559 | 21.38124 | 19.80492 | $-0.00000003\pm0.00000004$ |
| R1-shared | 23.59403 | 16.97205 | 15.75673 | $-4.4092\pm0.1147$ |
| **r1_ap_ris_mag** | 23.58262 | **21.42311** | **19.83875** | **$+0.0419\pm0.0073$** |

R0c 與 R0 的逐 batch 最大差為 $3.8\times10^{-6}$，再次通過等價 control。R1-shared 在
centralized 幾乎不變，卻在 decentralized 大幅退化，表示成熟 R0 policy 的 local proposals
更加依賴 magnitude weighting。AP–RIS magnitude weighting 將此缺口補回；其 decentralized 95% CI 約為
$[+0.0275,+0.0562]$，2-bit 差為 $+0.0338\pm0.0107$。由於這次 500k 評估是在 10k 結果後安排，
且實質差異只有約 0.2%，本報告只主張**未觀察到具實質意義的退化**，不主張 AP–RIS magnitude weighting 優於 R0。

## 8. Claim boundary

- 從頭訓練比較只有 seed 0、10k budget、400-sample holdout，只能寫 screening trend。
- §6 的 ladder 與相關性都是零訓練 counterfactual；§7.4 已證明它們不能外推到重訓後。
- §7.6 已在 500k R0 checkpoint 使用 3,200 samples；它驗證的是固定 checkpoint 的介面轉換，
  不是新的 training method，也尚未跨 training seed／topology。
- R1-local（`r1`）、node removal、RIS identity 本輪未測。
- Seed 20260915 為 discovery set；20260916 的 zero-training arms 與 margin 有 repository 內的
  pre-registration。20260917 是未重複使用的 training holdout，但 repository 內沒有對應的獨立
  preregistration artifact，因此不稱為預先登記確認。20260918 只用於 500k conversion evaluation。

## 9. 下一步

1. **目前最受支持的部署候選**：R0 目標訓練 + `r1_ap_ris_mag` 推論。10k 的預先登記確認與
   500k 的 3,200-sample evaluation 都未觀察到實質退化，並符合
   [方法文件](./decentralized_ris_methods.md)所定義的 parameter-free consensus interface。
2. **若仍要訓練端方案**，先診斷 §7.5 的 projection 訓練問題（例如 straight-through 或
   逐步加入 projection 的 schedule），而不是再加 gating network——learned gating 是在同一個
   失敗的訓練設定上增加元件，§7.2 已顯示缺的不是資訊。
3. 依原訂規則，`r1_ap_ris_mag` 不安排 40k，也不加 training seeds。
4. 比較任何 finalist 時，用 §6.4 的 2×2 分開 phase 與 beamformer 缺口。

## 10. Artifacts

| Evidence | Location |
|---|---|
| Matched 10k runs（r0／r1_shared／r1_ap_ris_mag）與 log | `artifacts/decentralized_ris/action_10k/` |
| Smoke | `artifacts/decentralized_ris/action_smoke/` |
| Step 2／Step 5 evaluation | `artifacts/decentralized_ris/evaluation/r0*_screen*`、`action_r0_vs_r1s_10k*` |
| Diagnostics | `artifacts/decentralized_ris/evaluation/proposal_diagnostics*` |
| Pre-registration 與零訓練確認 | `artifacts/decentralized_ris/evaluation/prereg_r1_ap_ris_mag_zero_training.md`、`prereg16_*` |
| 訓練後 paired evaluation | `artifacts/decentralized_ris/evaluation/action_10k_seed17*` |
| 500k R0 policy 四介面 paired evaluation | `artifacts/decentralized_ris/evaluation/baseline_500k_*_seed18*` |
| 同步回本機的遠端 R0 baseline | `artifacts/decentralized_ris/action_10k_remote/` |
