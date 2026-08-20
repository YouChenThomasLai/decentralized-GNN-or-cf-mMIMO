# Stage 2 — Mobility Environment 與 Frozen-Model Inference 實驗計畫

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-18
- Last Updated: 2026-08-20
- Verification Status: EVIDENCE-AUDITED（mobility/channel 定義與 fairness boundary 已對照文獻；新版 source 尚待重寫與重跑）
- Version Label: stage2_mobility_plan_v5_evidence_audit
- Parent Plan: `doc/decentralized_active_csi_experiment_plan.md`
- Living Report: `doc/decentralized_active_csi_experiment_report.md`
- Frozen Baseline: Stage 1B，`noise_power=1e-12`、5-seed/2000-iteration gate 已通過

## 1. Stage 2 的唯一任務

Stage 2 不是新的訓練階段，也不是論文的最終 robustness study。它只做兩件事：

1. 驗證新的 mobility/channel environment 符合預先定義的物理與統計 contract。
2. 將 Stage 1B 已訓練完成的 snapshot model 凍結後，放到新環境做 zero-shot inference，確認 pipeline 能正常執行並得到可解讀的大致趨勢。

通過 Stage 2 代表後續 Stage 3–5 已有一個可信的「fixed association + full current CSI」時間環境與基準。不代表 frozen model 已對 mobility 最佳化，也不代表某個方法在不同 mobility 下具有統計優勢。

### 1.1 Stage 2 要回答的問題

> 在不重新訓練、不加入 dynamic association、不限制 CSI feedback 的情況下，Stage 1B frozen snapshot model 能否在正確的 straight-line 與 hotspot/semi-Markov mobility environments 中，以 current CSI 完成 inference，並產生 finite、constraint-valid、可供下一階段比較的 baseline？

### 1.2 Stage 2 不回答的問題

- 不證明「速度越快，所有方法一定越差」。速度改變的是 temporal correlation；normalized fading 的 marginal distribution 並未因此改變，geometry、fixed association 與抽樣也會影響 rate。
- 不比較 matched mobility training、domain adaptation 或 train/test mobility matrix。
- 不宣稱 centralized 優於 decentralized、GNN 優於 MRT/RZF，亦不作 equivalence test。
- 不使用 trajectory frames 冒充獨立統計樣本。
- 不加入 dynamic association、switching cost、feedback budget、stale CSI、CSI age、history input、RNN/GRU 或 RL。
- 不把簡化的 path-loss + first-order Gauss–Markov channel 稱為完整 Jakes simulator 或完整 3GPP channel。

## 2. 為何 hotspot 留在 Stage 2

Hotspot/semi-Markov mobility 和 straight-line mobility 都是 environment generator。兩者共用同一個 channel、fixed association、full current CSI、frozen checkpoint 與 evaluator，因此現在一起實作、測試最省事，也能避免後面才發現 environment contract 不相容。

它在 Stage 2 的角色是「第二種合法環境與 smoke/development evaluation」，不是最終 robustness 證據。等 Stage 3–6 的方法與介面凍結後，Stage 7 再用相同的 straight/hotspot generators、paired traces 與多 seeds，對所有最終方法做正式 robustness 比較。

### 2.1 證據層級與採用邊界

本計畫將「論文有用過」與「本專案為了可控實驗而選定」分開標示：

- **Reference-matched**：直接沿用 reference paper 或 Stage 1 已實作的 geometry、path-loss、power 與 association convention [S2-0]。
- **Literature-supported structure**：論文支持 straight-line、temporally correlated fading、preferred locations、revisit 或 dwell 這種模型結構，但不代表論文支持本計畫的每一個數值 [S2-1]–[S2-6]。
- **Controlled synthetic choice**：為了在現有 100 m disk 內可重現地做 environment qualification 而事前凍結的參數。它們不能寫成由實際人類 mobility trace 校準得到。

### 2.2 Fairness audit 結論

| 比較層次 | Stage 2 狀態 | 理由與可做主張 |
|---|---|---|
| 同一 setting 內的四種 beamforming 方法 | **有條件公平** | 只要 Gate 2.4 通過，四法共用 frame、current true channel、mask、noise、power 與 evaluation indices；centralized/local observation scope 是待比較方法本身的差異。 |
| 0/3/30/80 km/h 的 speed effect | **不是因果比較** | 速度同時改變 Doppler、位移、boundary-conditioned direction 與 fixed-association mismatch；Stage 2 只報環境診斷與 descriptive trend。 |
| Straight 對 hotspot raw rate | **不可直接當 mobility-model effect** | Hotspot 把 spatial occupancy 集中在內圈，straight 的起點來自整個 UE disk；兩者平均 rate 差同時含 geometry shift。 |
| 「真實人類 mobility」或「3GPP channel」 | **不支持** | Hotspot 是 synthetic sensitivity environment；channel 是 Jakes-calibrated AR(1) + Stage 1 path loss，不含完整 multipath clusters、spatial consistency 或 shadowing。 |

## 3. 共用固定設定

| 項目 | Stage 2 固定值 | 來源／設計角色 |
|---|---|---|
| AP/UE/antennas | 5 APs、$K=8$、$M=2$ | [S2-0, Sec. IV, Table I] 的 reference-matched setting。 |
| Power/noise | $P_{\max}=15$ dBm、`noise_power=1e-12` | $P_{\max}$ 來自 [S2-0, Table I]；noise 是已通過 Stage 1B gate 的 project calibration，不聲稱來自 [S2-0]。 |
| Model | Stage 1B seed-0 final checkpoint；參數完全凍結 | Controlled evaluation choice，避免把 environment shift 與 retraining gain 混在一起。 |
| Learned policy | Stage 1B snapshot GNN；每個 frame 獨立 inference | 沿用 [S2-0] 的 snapshot architecture；不聲稱具有 temporal policy。 |
| Comparators | Centralized GNN、decentralized GNN、MRT、RZF | 比較集來自 Stage 1B；在同一 setting 內共用 environment inputs。 |
| Decision period | $\Delta t=1$ ms | [S2-1, Sec. VI] 使用 1 ms LTE subframe；本計畫將它當 CSI/channel update period，不當 handoff period。 |
| Carrier | $f_c=2.6$ GHz | [S2-1, Sec. VI] 的 mobility/CSI-aging simulation setting；只用來算 Doppler。 |
| Episode | $T=2000$ periods，即 2 s | [S2-1] 使用 2000 blocks；本計畫配合 1 ms period 得 2 s controlled clip。 |
| Geometry | 5 APs 均勻放在半徑 200 m 圓上；UE 位於半徑 100 m disk | [S2-0, Sec. IV]；Stage 2 不另換 topology。 |
| Association | 每條 2 s trajectory 僅在 $t=0$ 依 Stage 1 threshold rule 建立，之後固定 | Threshold $0.1$ 來自 [S2-0]；clip 內固定是隔離 dynamic-association effect 的 controlled choice。 |
| CSI | 每期提供全部 current CSI；所有方法共用同一 frame 與 mask | Fairness control，並作為後續 stale-CSI stage 的 full-information anchor。 |
| Evaluation unit | seed 0、每個 setting 10 條 trajectories、`eval_time_stride=10` | Development budget，不是論文 inference sample-size 根據。 |
| Purpose | environment qualification、end-to-end execution、descriptive trend | Scope restriction；正式 paired multi-seed evidence 留到 Stage 7。 |

Stage 2 evaluator 不得建立 optimizer、執行 backpropagation、寫 TensorBoard training log，或產生新的 model checkpoint。速度、方向與 history 不提供給 snapshot GNN；mobility 只透過當期 channel 影響輸入。

## 4. 共用 channel 定義

本階段使用的準確名稱是 **Jakes-calibrated first-order Gauss–Markov (AR(1)) channel**。Deng et al. [S2-1, Eqs. (1)–(2)] 使用 first-order stationary Gauss–Markov process 表示 channel aging，並以 Doppler/Jakes coefficient 連結 user speed；Ammar et al. [S2-2, Eqs. (1)–(2)] 也以 $J_0(2\pi \ell f_DT_s)$ 建模 UC-CF-MIMO 中的 temporal channel covariance。

令 UE $k$ 的速度為 $v_k$，最大 Doppler frequency 與 one-step coefficient 為

$$
f_{D,k}=\frac{f_cv_k}{c},\qquad
\rho_k=J_0(2\pi f_{D,k}\Delta t).
$$

速度會隨 phase 改變時，上式逐期套用為 $f_{D,k,t}=f_cv_{k,t}/c$ 與 $\rho_{k,t}=J_0(2\pi f_{D,k,t}\Delta t)$。

Normalized small-scale fading 使用 first-order stationary Gauss–Markov model：

$$
\mathbf g_{a,k,0}\sim\mathcal{CN}(\mathbf 0,\mathbf I_M),
$$

$$
\mathbf g_{a,k,t+1}
=\rho_{k,t}\mathbf g_{a,k,t}
+\sqrt{1-\rho_{k,t}^2}\,\boldsymbol\epsilon_{a,k,t},
\qquad
\boldsymbol\epsilon_{a,k,t}\sim\mathcal{CN}(\mathbf 0,\mathbf I_M).
$$

不同 AP–UE links、antennas 與 innovation times 使用獨立 innovations；同一 UE 的 links 共用由該 UE instantaneous speed 決定的 $\rho_{k,t}$。Straight-line 的 $\rho_{k,t}$ 在 episode 內固定；hotspot model 則依 dwell/transit 的 instantaneous speed 更新。

這個 AR(1) process 在 constant-speed segment 的 lag-$\ell$ correlation 是

$$
R_{\mathrm{AR(1)}}[\ell]=\rho_k^\ell
=\left[J_0(2\pi f_{D,k}\Delta t)\right]^\ell,
$$

而完整 isotropic-scattering Jakes autocorrelation 是 $R_{\mathrm{Jakes}}[\ell]=J_0(2\pi \ell f_{D,k}\Delta t)$。兩者在多步 lag 不相同；Gate 2.2 必須檢查 $\rho_k^\ell$，不能把實驗結果寫成「完整 Jakes simulator 驗證」。

在 $f_c=2.6$ GHz、$\Delta t=1$ ms 且 $c=3\times10^8$ m/s 下，預先計算的 contract values 為：

| $v_k$ | 每步位移 | $f_{D,k}$ | one-step $\rho_k$ | 角色／來源 |
|---:|---:|---:|---:|---|
| 0 km/h | 0 m | 0 Hz | 1.000000 | Stationary compatibility anchor（derived） |
| 3 km/h | 0.000833 m | 7.222 Hz | 0.999485 | Low-speed example；3GPP 也列 3 km/h mobility cases [S2-3] |
| 30 km/h | 0.008333 m | 72.222 Hz | 0.949178 | 3GPP spatial-consistency calibration 使用 fixed-speed/random-direction 30 km/h example [S2-3] |
| 80 km/h | 0.022222 m | 192.593 Hz | 0.666090 | Deng et al. 的 high-mobility example [S2-1, Figs. 3–4] |

保留 Stage 1 direct-channel amplitude convention：

$$
q_{a,k,t}=\frac{10^{-4.5}d_{a,k,t}^{-3.5}}{10^{-7}},\qquad
\mathbf h_{a,k,t}=q_{a,k,t}\mathbf g_{a,k,t}.
$$

[S2-0, Sec. II, Table I] 將 direct AP–UE link 建模為 Rayleigh fading，使用 large-scale amplitude $10^{-4.5}$ 與 exponent $3.5$；除以 $10^{-7}$ 是 Stage 1 code 的 numerical normalization，不是 [S2-0] 的新物理參數。本階段不額外加入 shadowing，以保持 Stage 1 marginal scale，使 $t=0$ 與 0 km/h case 可作 backward-compatibility check。Ammar et al. [S2-2] 的 correlated-shadowing setting 可在後續作獨立 sensitivity，但不應混入本 gate。

## 5. Stage 2A — Straight-Line Mobility

### 5.1 Environment 定義

直線、固定速度是用來隔離 kinematics 與 channel contract 的最小可驗證模型，不是對完整日常移動的擬真。在 UC-CF-MIMO 文獻中，Ammar et al. [S2-2, Sec. VI] 讓 UE 以 10 m/s（36 km/h）做 1 km straight-line trip；Hsu et al. [S2-4, Sec. III-B] 的 community model 也在每個 movement epoch 抽取速度與均勻方向，並在 epoch 內保持 constant-speed random-direction movement。這些證據支持 **model structure**，不支持將本計畫的 boundary rule 當成標準。

每個 UE 在一條 episode 中使用固定速度與方向：

$$
\mathbf x_{k,t+1}=\mathbf x_{k,t}
+v_k\Delta t[\cos\phi_k,\sin\phi_k]^\mathsf T.
$$

- Initial position 沿用 Stage 1 generator [S2-0]。
- $\phi_k\sim\mathcal U[0,2\pi)$；以 rejection sampling 確保終點仍在半徑 100 m disk。圓盤為 convex，因此整段路徑都在界內。這是 project-specific boundary choice，且接受後的方向分布會依起點與速度而改變，所以 Stage 2 不能把跨速度 rate 差當成純 Doppler effect。
- 每個主 setting 中所有 UE 使用相同速度，避免把 heterogeneous speed composition 混入環境驗證。
- 主 inference speeds 為 0、30、80 km/h：0 是 compatibility anchor；30 是 [S2-3] 使用的 fixed-speed/random-direction calibration example；80 是 [S2-1] 的 high-mobility example。
- 3 km/h 的 $\rho$ 非常接近 1，先由 deterministic/statistical channel test 覆蓋；不再列為 mandatory full inference。
- 若下一階段一開始就需要 heterogeneous users，可追加一組 mixed-speed smoke run，例如 8 個 UE 各兩名使用 0/3/30/80 km/h；它不是 Stage 2 通關條件。

若 Stage 7 要估計 speed effect，straight settings 必須改用 paired design：先以最大速度 80 km/h 的 endpoint constraint 抽取可行的 initial position/direction，再將同一組 geometry draws 與 normalized innovation streams 重用於 0/3/30/80 km/h。如此可避免每個速度各自 rejection 造成不同 direction distribution，但結果仍要標示為「在 80-km/h-feasible 軌跡上的 paired estimand」。

### 5.2 必跑矩陣

| Mobility | Speed | Seeds | Trajectories | Stride | 用途 |
|---|---:|---:|---:|---:|---|
| Straight | 0 km/h | 0 | 10 | 10 | Stationary/Stage 1 compatibility baseline |
| Straight | 30 km/h | 0 | 10 | 10 | Literature-anchored fixed-speed development case |
| Straight | 80 km/h | 0 | 10 | 10 | High-mobility stress case |

這三組只報告 descriptive mean/tail summaries 與環境 diagnostics。不得從三個 means 推論單調 speed effect，除非 Stage 7 使用 paired traces、多 seeds 與預先登記的 estimand 重新測試。

## 6. Stage 2B — Hotspot/Semi-Markov Mobility

### 6.1 Environment 定義

Hotspot mobility 是外生 semi-Markov process，不是 MDP：destination 與 dwell 不受 rate、association 或任何待比較方法控制。Hsu et al. [S2-4] 以 community attraction、local/roaming Markov transitions、movement epochs 與 pause time 表示 preferred-location mobility；González et al. [S2-5] 的軌跡資料顯示個人會重返少數高頻地點；TimeGeo [S2-6] 則將 dwell/burst 時間機制與 spatial destination selection 分開建模。這三篇文獻支持 preferred locations + revisit + explicit dwell 的 **結構**，但不提供本計畫可直接照抄的 4 個 centers、$0.6$ self-transition 或 Gamma$(2,2.5\,\mathrm{s})$ 參數。

主設定固定為：

- 4 個 hotspot centers：$(\pm35,0)$、$(0,\pm35)$ m；hotspot radius 10 m。所有 hotspot points 均位於半徑 45 m 內，因此任兩點間的直線 transit 也位於 100 m convex UE disk。這是對稱、可驗證的 controlled synthetic geometry，不是 trace-calibrated geometry。
- 每個 UE 的 initial hotspot state 從 $\{1,2,3,4\}$ 均勻抽取，initial physical position 在該 hotspot disk 內依面積均勻抽取。同一 multi-UE trajectory 中的 UEs 使用獨立 state/target/dwell RNG substreams，不預設同步轉移。
- Symmetric transition matrix 定義為 $P_{ii}=0.6$、$P_{ij}=0.4/3$ for $i\ne j$；其 event-chain stationary distribution 為 uniform。這是 medium-stickiness diagnostic choice，不是 [S2-4]–[S2-6] 的估計值。
- 每個 macro event 的 dwell duration 使用 Gamma distribution，shape $2$、scale $2.5$ s、mean $5$ s。非 exponential dwell 使 process 成為 semi-Markov；shape $2$ 只是避免 memoryless holding time 的 controlled choice，不宣稱為人類 dwell 的 empirical fit。
- Self-transition 會在原 hotspot 開始新的 dwell event，不產生 transit，也不重抽 physical target。因此「event dwell mean」為 5 s，但連續 self-events 合併後的同一-hotspot residence mean 為 $5/(1-0.6)=12.5$ s；兩個統計量必須分開報告。
- 當 $j\ne i$ 時，在 destination hotspot disk 內依面積均勻抽取 physical target，再以指定 transit speed 沿直線連續前往；不得 teleport。
- Dwell phase **固定為靜止**，不保留「靜止或局部移動」的未定義分支。因此 dwell 時 instantaneous speed $=0$、$\rho=1$，transit 時才使用設定速度。這是對 Hsu et al. pause state [S2-4] 的簡化對應，也意味本模型不含 stationary-UE 時由環境 scatterer 造成的 fading。
- 每條 evaluation trajectory 使用獨立 parent macro trace 與 RNG substream。先捨棄 burn-in，長度至少為 `max(120 s, 10 × configured mean dwell-plus-transit cycle)`；之後保留 300 s macro trace，再從保留區間的合法 start times 均勻抽一個 2 s clip。均勻抽 natural time，不是均勻抽 event，才會保留長 dwell/transit 應有的 time occupancy。
- 每個 2 s clip 在 $t=0$ 建立一次 association，clip 內固定。

獨立 parent traces 是為了避免 10 個 clips 共享同一條 300 s trace 而形成隱性 pseudo-replication。300 s 只是 inference clip pool，不是 transition-matrix/dwell tolerance 的統計樣本；Gate 2.3 必須使用另一條夠長的 environment-only diagnostic stream。

### 6.2 必跑矩陣

| Mobility | Transit speed | Seeds | Trajectories | Stride | 用途 |
|---|---:|---:|---:|---:|---|
| Hotspot semi-Markov | 3 km/h | 0 | 10 | 10 | Slow-transit / long-travel-time stress case |
| Hotspot semi-Markov | 30 km/h | 0 | 10 | 10 | Main case |
| Hotspot semi-Markov | 80 km/h | 0 | 10 | 10 | Fast transit stress case |

Hotspot 0 km/h 不執行：UE 無法合理移往下一 hotspot，會使 transition 與 transit 語意失真。

這三組速度同時改變 Doppler 與 transit duration，所以 natural-time clips 中 dwell/transit 的比例也會跟著改變。特別是 3 km/h 在相距約 50–70 m 的 hotspots 間可能長時間處於 transit，不應再標為 dwell-dominant。Stage 2 只驗證 generator 對這個物理結果的 accounting 是否正確。

### 6.3 額外 hotspot diagnostics

為確認 generator 不是只在單一參數下碰巧正確，再增加兩組 environment-only diagnostics，不跑四種 beamforming inference：

- Low-stickiness / short-dwell：$P_{ii}=0.2$、$P_{ij}=0.8/3$、Gamma shape $2$、mean $2$ s；檢查頻繁 transition、continuity 與 event accounting，merged same-hotspot residence mean target 為 $2/(1-0.2)=2.5$ s。
- High-stickiness / long-dwell：$P_{ii}=0.8$、$P_{ij}=0.2/3$、Gamma shape $2$、mean $10$ s；檢查長 dwell、occupancy 與 clip sampling，merged same-hotspot residence mean target 為 $10/(1-0.8)=50$ s。

這些數值是以主設定為中心的對稱工程 stress choices，不是 [S2-4]–[S2-6] 的 empirical estimates。它們只驗證 state、dwell、position、boundary 與 channel statistics，不作性能比較，也不以結果選擇較好看的主參數。

## 7. 最小實作範圍

新版 `code/stage2/` 應是 evaluation-only package，保留最少責任：

| Component | 責任 |
|---|---|
| Environment | 產生 straight/hotspot positions、phase/state、current channel 與固定 mask |
| Evaluator | 載入 frozen Stage 1B checkpoint，執行 C/D GNN、MRT、RZF |
| Model compatibility | 使用與 Stage 1B 完全相容的 architecture/state dict；不改模型語意 |
| Tests | 驗證 kinematics、channel、hotspot process、mask、power、reproducibility |
| Runner | 只展開本文件的 6 組 mandatory inference 與 environment-only diagnostics |

不要加入 `train.py`、optimizer、matched-training flags、trajectory train/validation split 或 $2\times2$ train/test matrix。若沿用 `trainer_2.py` 檔名以相容既有操作，它的角色仍只能是 evaluation entry point。

## 8. Mandatory Gates

### Gate 2.0 — Provenance 與 Stage 1 相容性

- 啟動前保存 Stage 2 source snapshot、Git commit/dirty status、source SHA-256 與 frozen checkpoint SHA-256。
- 使用相同 positions、initial fading、mask 與 beamformers 時，Stage 2 的 $t=0$ rate 與 Stage 1 在 `atol=1e-6, rtol=1e-6` 內一致。
- 0 km/h 時 position、path loss 與 channel 在同一 trajectory 內不變。

### Gate 2.1 — Mobility kinematics

- Straight-line 每步位移與 $v\Delta t$ 的誤差小於 `1e-10` m。
- 所有 positions 位於半徑 100 m disk；hotspot transition 不可跳躍，step length 不超過 instantaneous-speed limit。
- 相同 seed 的 positions/states/channels 完全可重現；不同 seed 不得意外相同。

### Gate 2.2 — Channel statistics

- Normalized real/imaginary parts 必須 finite；在獨立產生、未經 path loss 加權的 diagnostic sample 上，`abs(mean) <= 0.02` 且 `abs(variance - 0.5) <= 0.02`。
- Constant-speed segments 的 empirical lag correlations 對 **AR(1) contract** $\rho^\ell$ 的 maximum absolute error `<=0.02`；另存 exact-Jakes $J_0(2\pi\ell f_D\Delta t)$ 作為 model-gap reference，但不用它驗收 AR(1) generator。
- 每期 distance 與 path-loss factor 精確符合 Stage 1 公式。
- 0、3、30、80 km/h channel coefficients 均由 unit/statistical test 覆蓋，即使 3 km/h 不跑 straight full inference。

### Gate 2.3 — Hotspot process

- Transition/dwell/occupancy tolerance 使用獨立 environment-only diagnostic stream，不從 10 個 inference clips 估計。Diagnostic stream 至少包含 10,000 個 post-burn-in macro events，且每個 state 至少有 2,000 個 outgoing events；不足就延長 stream，不放寬 tolerance。
- Empirical event-transition matrix 對 configured matrix 的 maximum absolute error `<=0.05`；event-chain occupancy 對 uniform stationary distribution 的 maximum absolute error `<=0.03`。另外報告 natural-time 的 hotspot/transit occupancy，並對照長時間 reference simulation；不把 event occupancy 與 time occupancy 混為同一統計量。
- Empirical **event dwell** mean 對各 variant configured mean 的 relative error `<=5%`；另報告連續 self-events 合併後的 same-hotspot residence mean，main/low/high 的 targets 分別為 12.5/2.5/50 s。保存 event count、mean、variance 與預先指定的 quantiles。
- Main、low-stickiness/short-dwell、high-stickiness/long-dwell 三組都通過 event accounting、boundary 與 physical continuity tests；step-limit maximum violation `<=1e-10` m。
- Dwell/transit channel correlation 分別符合各自 instantaneous-speed coefficient；dwell 的 $\rho=1$ 應使 normalized channel 逐步完全不變。

### Gate 2.4 — Fair inference

- 四種方法在同一 trajectory/frame 使用完全相同的 current channel、association mask、noise 與 power constraint。
- 四種方法使用完全相同的 trajectory order、frame indices 與 `eval_time_stride`，不可因方法而重抽 clips 或 channels。
- Centralized GNN 可讀 global current CSI，decentralized GNN 只讀各 AP 的預定 local current CSI；這是方法定義的 information-structure difference，必須在 config 與報告中明列，不能誤寫成四法 observation 完全相同。
- Association mask 在整個 2 s clip 內不變。
- Beamformers 與 rates 全部 finite；unassociated links 為零，每個 AP 均符合 power limit。
- Rate 由同一期 current true channel 計算，不使用 future channel、mobility state 或隱含 dynamic reassociation。

### Gate 2.5 — End-to-end completion

- Straight 0/30/80 與 hotspot 3/30/80 六組皆完成 seed-0、10-trajectory、stride-10 inference。
- 每組保存完整有效 config、compact raw metrics、environment diagnostics、summary、source/checkpoint hashes 與 log。
- 本 gate 只要求 pipeline 正確與 coarse trend 可讀；不要求 rate 隨速度單調下降，也不要求任何方法獲勝。

## 9. Outputs 與 Metrics

每組 mandatory run 最少保存：

- effective config、seed、source/checkpoint hashes、evaluation indices；
- 每條 trajectory 的 average sum rate；
- 每條 trajectory 的 UE time-average rates 與 descriptive 5th-percentile summary；
- C/D GNN、MRT、RZF 的 finite/power/mask checks；
- trajectory、speed/phase、distance/path-loss 與 selected channel-correlation diagnostics；
- hotspot runs 的 state、dwell、transition、occupancy 與 continuity diagnostics；
- completion status 與 log。

不必為 full-current-CSI 額外保存一份重複的 `stored_channels` 大陣列；以單一 current-channel source 加上 contract test 證明 evaluator 沒有 stale-CSI branch 即可。只有 Stage 4 實作 stale CSI 時，才同時保存 true/stored channels 與 age。

Stage 2 的性能數值只以表格或小圖呈現大致趨勢。`fixed_association_regret`、current-RSSI reassociation 與 strongest-AP switching 屬於 Stage 3 的變因；若保留舊 pilot diagnostic，只能標為歷史觀察，不應留在新版 Stage 2 evaluator。

## 10. 執行順序與時間預算

| 工作 | 組數 | 觀察/估計時間 |
|---|---:|---:|
| Straight 0/30/80 km/h | 3 | 約 30 分 53 秒（既有觀察） |
| Hotspot 3/30/80 km/h | 3 | 約 31–32 分鐘（依 30 km/h 的 10 分 31 秒估計） |
| Mandatory inference 合計 | 6 | 約 62–63 分鐘 |
| Optional mixed-speed smoke run | 1 | 約 10–11 分鐘 |
| Hotspot environment-only variants | 2 | 不執行四方法 inference，應顯著短於完整 run |

建議順序：contract tests → straight 0 → straight 30/80 → hotspot main environment tests → hotspot 30 → hotspot 3/80 → environment-only variants → optional mixed-speed。

## 11. 通關與停止規則

Stage 2 在 Gates 2.0–2.5 全部通過後即結束，可以進入 Stage 3 或 Stage 4。以下都不是阻擋條件：

- 個別速度的 rate 未單調下降；
- centralized/decentralized gap 很小或正負改變；
- GNN 未勝過 MRT/RZF；
- seed-0 不足以支持統計結論。

只有 environment contract、fairness、constraint、provenance 或 end-to-end execution 失敗才需要停下修正。修正後重跑受影響設定，不因性能結果不好而改 mobility 參數。

## 12. 延後到 Stage 7 的正式證據

等 Stage 3–6 的最終方法、observation、action 與 evaluator 凍結後，再一次完成正式 robustness matrix：

- straight 與 hotspot；
- 預先指定的 speeds、feedback budgets、switching costs 與 network sizes；
- 所有最終方法加上 frozen Stage 1B snapshot baseline；
- 至少 5 個 paired seeds，所有方法共用相同 traces/innovations；
- Straight speed comparison 使用第 5.1 節的 max-speed-feasible paired initial positions/directions，不讓每個 speed 各自 rejection；
- Hotspot comparison 預先區分兩種 estimand：自然時間的 **system effect** 可讓 speed 改變 transit 佔比，但要共用 event-state/target/dwell random streams 並報告 phase proportions；若要估計 Doppler-only effect，必須另建 fixed-geometry/channel sensitivity，不可與前者混稱為 speed effect；
- 必要時使用 `eval_time_stride=1`，以 seed 或完整 trajectory aggregate 作統計單位；
- 報告 paired effect、confidence interval、effect size 與多重比較控制。

這樣只在最終需要論文推論時支付多-seed/full-stride 成本，也避免 Stage 2 先替尚未完成的方法做一套之後必須重跑的 evidence sweep。

## 13. 舊版 pilots 的處理

2026-08-19 的 straight 0/30/80 km/h 與 hotspot 30 km/h frozen pilots 保留在 living report，作為新版重寫前的設計依據與 runtime reference。它們支持以下判斷：

- inference 約每組 10–11 分鐘，因此新版可合理擴成 6 組；
- rate 不隨速度單調改變，符合本階段不作 speed-effect inference 的定位；
- hotspot generator 值得留在 Stage 2，但單一 30 km/h case 不足以覆蓋 slow/fast transit；
- 舊 matched-training run 有 provenance anomaly，不納入新版 Stage 2，也不重跑。

舊 pilots 不自動使新版 Stage 2 通關。新版 source 完成後，仍須以新的 provenance snapshot 與 output contract 重跑本文件的 mandatory matrix。

## 14. Environment/channel 定義—證據對照表

| 定義 | 證據 | 本計畫的採用與限制 |
|---|---|---|
| 5 APs、$K=8$、$M=2$、AP radius 200 m、UE disk radius 100 m、$P_{\max}=15$ dBm | [S2-0, Sec. IV, Table I] | Reference-matched；Stage 2 不在新 topology 上同時測 mobility。 |
| Direct Rayleigh link 的 $10^{-4.5}d^{-3.5}$ amplitude convention 與 association threshold $0.1$ | [S2-0, Sec. II, Sec. IV, Table I] | Reference-matched；`/1e-7` 與 `noise_power=1e-12` 是 Stage 1 numerical convention，不假裝是 paper parameter。 |
| $f_c=2.6$ GHz、$\Delta t=1$ ms、2000 blocks | [S2-1, Sec. VI] | 用於 Doppler/channel-update timescale；不將 1 ms 解釋為 handoff decision period。 |
| First-order stationary Gauss–Markov channel aging | [S2-1, Eqs. (1)–(2)] | 直接支持 AR(1) recursion 與 stationary complex-Gaussian marginal。 |
| $J_0(2\pi f_D T_s)$ 將 speed/Doppler 連到 temporal correlation | [S2-1]；UC-CF-MIMO 中的 Jakes covariance 可參照 [S2-2, Eqs. (1)–(2)] | 只用 Jakes 的 one-step value 校準 AR(1)；multi-lag target 是 $\rho^\ell$，不是 exact Jakes curve。 |
| Straight-line constant-speed mobility | [S2-2, Sec. VI] 在 UC-CF-MIMO 使用 36 km/h straight trip；[S2-4, Sec. III-B] 定義 constant-speed random-direction epochs | 支持結構；boundary rejection 是 controlled project choice。 |
| 0/3/30/80 km/h grid | 0 為 derived anchor；3/30 km/h 可見 [S2-3]；80 km/h 可見 [S2-1] | 是 coverage grid，不代表這四點是單一文獻預設的統一 benchmark。 |
| Preferred locations、revisit、pause/dwell 的 hotspot structure | Community/Markov/pause structure [S2-4]；recurrent frequent locations [S2-5]；dwell/burst 與 spatial choice 分離 [S2-6] | 支持 synthetic semi-Markov environment 的結構；不支持本計畫的數值即為真實人類 mobility。 |
| 4 centers at 35 m、radius 10 m、$P_{ii}=0.6$、Gamma shape 2/mean 5 s | 無論文直接校準 | Controlled synthetic main setting；要與 low/high variants 一起做 generator diagnostics，不可以 rate 結果反向選參數。 |
| 300 s parent trace + natural-time 2 s clips | TimeGeo 支持將 macro mobility timing 與 spatial choice 分開 [S2-6]；300 s 本身無文獻校準 | 300 s 是 inference sampling budget；獨立 long diagnostic stream 才能驗證 transition/dwell tolerance。 |
| Fixed-$t=0$ association + full current CSI | Threshold rule 來自 [S2-0]；固定 mask/full CSI 為 project control | 用來隔離 dynamic association 與 stale CSI；不是對 real deployment 會永久固定 association 的主張。 |
| 不加 shadowing | [S2-0] 的 Stage 1 direct-link convention；[S2-2] 顯示 correlated shadowing 是另一個可建模因素 | 為了 backward compatibility 而凍結；對外主張必須承認這是簡化 channel。 |

綜合判定：**Stage 2 作為 environment qualification 與 frozen-model smoke evaluation 是公平且有證據邊界的；作為 speed causality、straight-vs-hotspot robustness 或 realistic-channel evidence 則還不公平，且本計畫已明確將這些主張留到 Stage 7。**

## References

[S2-0] W.-Y. Ting, R. Y. Chang, F.-T. Chien, T.-Y. Peng, and P.-H. Lin, “Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks,” manuscript. [Local reference PDF](<Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks.pdf>)

[S2-1] R. Deng, Z. Jiang, S. Zhou, and Z. Niu, “Intermittent CSI Update for Massive MIMO Systems With Heterogeneous User Mobility,” *IEEE Transactions on Communications*, vol. 67, no. 7, pp. 4811–4824, Jul. 2019. [DOI](https://doi.org/10.1109/TCOMM.2019.2911575) · [Author manuscript](https://network.ee.tsinghua.edu.cn/niulab/wp-content/uploads/2019/12/Intermittent-CSI-Update-for-Massive-MIMO-Systems-With-Heterogeneous-User-Mobility.pdf)

[S2-2] H. A. Ammar, R. Adve, S. Shahbazpanahi, G. Boudreau, and K. V. Srinivas, “Handoffs in User-Centric Cell-Free MIMO Networks: A POMDP Framework,” *IEEE Transactions on Wireless Communications*, vol. 23, no. 8, pp. 10319–10335, Aug. 2024. [arXiv](https://arxiv.org/abs/2403.08900)

[S2-3] 3GPP, “Study on Channel Model for Frequencies from 0.5 to 100 GHz,” TR 38.901, version 19.3.0, Release 19, Apr. 2026. [ETSI PDF](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.03.00_60/tr_138901v190300p.pdf)

[S2-4] W.-J. Hsu, T. Spyropoulos, K. Psounis, and A. Helmy, “Modeling Time-Variant User Mobility in Wireless Mobile Networks,” in *Proc. IEEE INFOCOM*, pp. 758–766, May 2007. [DOI](https://doi.org/10.1109/INFCOM.2007.94) · [Author manuscript](https://bpb-us-w1.wpmucdn.com/sites.usc.edu/dist/b/364/files/2019/05/EE650_hsutimevariant.pdf)

[S2-5] M. C. González, C. A. Hidalgo, and A.-L. Barabási, “Understanding Individual Human Mobility Patterns,” *Nature*, vol. 453, pp. 779–782, Jun. 2008. [DOI](https://doi.org/10.1038/nature06958)

[S2-6] S. Jiang, Y. Yang, S. Gupta, D. Veneziano, S. Athavale, and M. C. González, “The TimeGeo Modeling Framework for Urban Mobility Without Travel Surveys,” *Proceedings of the National Academy of Sciences*, vol. 113, no. 37, pp. E5370–E5378, Sep. 2016. [DOI](https://doi.org/10.1073/pnas.1524261113) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5027456/)
