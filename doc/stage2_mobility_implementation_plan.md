# Stage 2A Straight-Line Mobility and Stage 2B Hotspot Robustness — Implementation Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-18
- Last Updated: 2026-08-19
- Verification Status: PARTIALLY IMPLEMENTED（Stage 2A source/checks 已存在；frozen Stage 1B mobility evaluation、2A evidence sweep 與 Stage 2B 尚未完成）
- Version Label: stage2_mobility_plan_v3
- Parent Plan: `doc/decentralized_active_csi_experiment_plan.md`
- Stage 1 Source: `code/stage1/`
- Stage 1B Gate: ANALYZED / PASSED（`noise_power=1e-12`、5 seeds、2000 iterations；數值與 artifact gate 通過，不代表 C/D 等效）
- Stage 1 Report: `doc/decentralized_active_csi_experiment_report.md`

## 1. Stage 2A 與 Stage 2B 要回答的問題

Stage 2A 回答：在 Stage 1B 的 no-RIS channel/power/noise convention 下，加入可重現的直線 UE mobility 與 temporally correlated channel 後，凍結的 Stage 1B snapshot beamformer 是否能在固定 association、每期 full current CSI 的條件下 zero-shot 運作？Stage 2A 先變動 evaluation environment，不先重新訓練 beamformer。

Stage 2B 回答：把外生直線 mobility 換成具有 hotspot preference、dwell 與 revisit 的外生 Markov/semi-Markov mobility 後，Stage 2A 的 centralized/decentralized 結論、方法排名與 fixed-association regret 是否仍成立？

兩個子階段都不是要證明 temporal policy 優於 snapshot policy，也不加入 dynamic association、feedback budget、stale CSI、CSI age action 或 RL。Stage 2A 是可驗證的 channel/mobility baseline；Stage 2B 是 mobility-model robustness gate。兩者共同提供後續 Stage 3–5 使用的 temporal environments。

Stage 2 將「開發門檻」與「論文證據門檻」分開。開發門檻先以 Stage 1B frozen checkpoint 評估新 traces；只有當 zero-shot 結果顯示明確 distribution-shift loss，或最終論文需要 matched-training comparison 時，才在 mobility snapshots 上重新訓練。

Stage 2B 的 hotspot 選擇若只有狀態轉移機率，正式名稱是 Markov chain；若另顯式建模非 memoryless dwell time，則是 semi-Markov model。只有在 UE 依 action 與 reward 主動選擇下一 hotspot 時才構成 MDP。Ammar et al. [S2-2] 的 POMDP 也把 LSF 當 state、AP association 當 action，而不是把 UE 移動本身當 action；因此本階段不把外生 hotspot trace 誤稱為 MDP。

## 2. Stage 2A 預先固定的主設計

| 項目 | Stage 2A 主設定 | 理由 |
|---|---|---|
| Source isolation | 新建 `code/stage2/`，只複製 `code/stage1/` 的 source/config/test files | 不修改正在使用的 Stage 1 source，也不讓 Stage 2 artifacts 混入 Stage 1。 |
| Decision/channel block | $\Delta t=1$ ms | Deng et al. [S2-1] 使用 1 ms LTE subframe/channel-block timescale；這是 CSI aging/update timescale，不是 handoff timescale。 |
| Carrier frequency | $f_c=2.6$ GHz | 與 [S2-1] 的 mobility/CSI-aging simulation 一致。此參數在 Stage 2 只決定 Doppler correlation，不回頭改 Stage 1 path-loss scale。 |
| Episode length | $T=2000$ periods（2 s） | 與 [S2-1] 的 2000 blocks 對齊，並讓 80 km/h UE 在一個 episode 內移動約 44.4 m。 |
| Controlled speeds | 0、3、30 km/h；80 km/h stress test | 3 與 30 km/h 有 3GPP mobility examples [S2-4]；[S2-1] 使用 20/80 km/h 與 40–80 km/h。 |
| Trajectory | 每個 UE 在一個 episode 內固定速度、固定方向的 straight-line segment | [S2-2] 直接採 straight-line mobility；[S2-3] 也明確允許 straight line 或 random waypoint。主實驗先用最容易驗證的 constant-velocity case。 |
| Association | 只在 $t=0$ 依 Stage 1 規則產生一次，之後整個 episode 固定 | 確保本階段唯一主要變因是 mobility/channel evolution；dynamic association 留到 Stage 3。 |
| CSI availability | 每期所有 AP–UE links 都 refresh，`stored_csi[t] == true_csi[t]` | 建立 Stage 4 stale-CSI 實驗的 full-information ground truth。 |
| Small-scale fading | first-order stationary Gauss–Markov，one-step coefficient 由 Jakes $J_0$ 決定 | 直接採 [S2-1] 的 intermittent-CSI-update channel model。 |
| Large-scale component | 主設定只依 Stage 1 的 distance/path-loss formula 隨位置更新 | 保持 Stage 1 marginal scale；不在 mobility gate 同時加入新的 shadowing distribution。 |
| Shadowing | 不納入主 gate；通過後另作 correlated-shadowing sensitivity | [S2-2], [S2-3] 支持 correlated shadowing，但直接加入會同時改變 Stage 1 的 marginal channel distribution。 |
| Primary model policy | 凍結 Stage 1B `M=2,K=8,Pmax=15,noise=1e-12` checkpoints，evaluation-only | Snapshot GNN 沒有 temporal state；先隔離 environment shift，不把 mobility-specific retraining 混入 Stage 2 gate。 |
| Optional adaptation | 只在需要時訓練 matched straight/hotspot snapshot models | 以 $R_{\mathrm{matched}}-R_{\mathrm{frozen}}$ 量化 retraining gain，不把 matched model 當成進入 Stage 3 的先決條件。 |

## 3. Stage 2A user mobility 的正式定義

### 3.1 位置與速度

令 UE $k$ 在 period $t$ 的二維位置為 $\mathbf{x}_{k,t}$，速度為 $v_k$，方向為 $\phi_k$：

$$
\mathbf{x}_{k,t+1}
=\mathbf{x}_{k,t}
+v_k\Delta t
\begin{bmatrix}\cos\phi_k\\\sin\phi_k\end{bmatrix}.
$$

- $\mathbf{x}_{k,0}$ 沿用 Stage 1 的 UE initial-location generator，故 $t=0$ geometry 不另換分布。
- 每個 episode/UE 獨立抽 $\phi_k\sim\mathcal U[0,2\pi)$；一個 episode 內不改速度與方向。
- 為避免 boundary wraparound 造成不連續 channel，抽方向時使用 rejection：只接受終點 $\mathbf{x}_{k,T-1}$ 仍在既有半徑 100 m UE disk 內的方向。圓盤是 convex set，因此起點與終點都在圓內即可保證整條 straight-line segment 在圓內。
- 這個 boundary rule 是配合現有圓形 geometry 的工程選擇，不宣稱來自 [S2-1]–[S2-4]。Stage 7 才比較 random waypoint、turning 或其他 trajectory models。
- 主實驗每次讓所有 UEs 使用同一速度，以隔離 speed effect；environment gate 通過後，再加入 mixed-speed setting，例如 $K=8$ 時每個 episode 各有兩名 UE 使用 0/3/30/80 km/h，並依 seed permutation。

代表速度在 $f_c=2.6$ GHz、$\Delta t=1$ ms 下的預期 one-step correlation 如下：

| Mobility class | $v_k$ | 每步位移 | $f_{D,k}=f_cv_k/c$ | $\rho_k=J_0(2\pi f_{D,k}\Delta t)$ |
|---|---:|---:|---:|---:|
| Stationary | 0 km/h | 0 m | 0 Hz | 1.000000 |
| Pedestrian | 3 km/h | 0.000833 m | 7.222 Hz | 0.999485 |
| Urban vehicle | 30 km/h | 0.008333 m | 72.222 Hz | 0.949178 |
| High-mobility stress | 80 km/h | 0.022222 m | 192.593 Hz | 0.666090 |

上述每步位移均遠小於 3GPP TR 38.901 spatial-consistency procedure 的 1 m update-distance upper bound [S2-4]；但 Stage 2 仍是簡化的 path-loss + Gauss–Markov model，不應稱為完整 3GPP channel implementation。

### 3.2 Temporally correlated small-scale fading

對 AP $a$、UE $k$，令 $\mathbf{g}_{a,k,t}\in\mathbb C^M$ 為 normalized small-scale fading：

$$
\mathbf{g}_{a,k,0}\sim\mathcal{CN}(\mathbf 0,\mathbf I_M),
$$

$$
\mathbf{g}_{a,k,t+1}
=\rho_k\mathbf{g}_{a,k,t}
+\sqrt{1-\rho_k^2}\,\boldsymbol\epsilon_{a,k,t},
\qquad
\boldsymbol\epsilon_{a,k,t}\sim\mathcal{CN}(\mathbf 0,\mathbf I_M).
$$

不同 AP–UE links、antennas 與 innovation times 使用獨立 innovation；同一 UE 的 links 共用由速度決定的 $\rho_k$。這正是 Deng et al. [S2-1, Eq. (1)–(2)] 用於 heterogeneous-mobility intermittent CSI update 的 first-order stationary Gauss–Markov model。理論上 normalized channel 的 lag-$\ell$ correlation 為 $\rho_k^{\ell}$。

主設定只選擇 $\rho_k>0$ 的速度/period 組合，因此 empirical correlation 應隨 lag 與速度上升而下降。若未來把參數擴展到 $J_0$ 的負值或 oscillatory region，驗收條件必須改成「符合 signed theoretical curve」，不可再要求全域單調。

### 3.3 保留 Stage 1 large-scale convention

Stage 1 的 direct channel 實際使用 amplitude factor

$$
q_{a,k,t}
=\frac{10^{-4.5}d_{a,k,t}^{-3.5}}{10^{-7}},
\qquad
d_{a,k,t}=\lVert\mathbf b_a-\mathbf x_{k,t}\rVert_2,
$$

因此 Stage 2 主 channel 定義為

$$
\mathbf h_{a,k,t}=q_{a,k,t}\mathbf g_{a,k,t}.
$$

這裡刻意保留 Stage 1 的 amplitude/path-loss、channel scale 與 `noise_power=1e-12`，不把其他論文的 $\sqrt{\beta}$、path-loss exponent、height、shadowing 或 noise 數值直接混入。如此 $v=0,t=0$ 可以與 Stage 1 作 paired equivalence check，mobility 才是唯一新增的主要變因。

主 gate 通過後可增加一個明確標為 sensitivity 的 correlated-shadowing variant。Ammar et al. [S2-2], [S2-3] 使用 6 dB shadowing、100 m decorrelation distance，並以位移決定 successive shadowing correlation；該 variant 必須另存 config/result root，不能取代主設定或用來選擇較好結果。

### 3.4 Fixed association 與 full CSI 的精確語意

1. 在每條 trajectory 的 $t=0$，沿用 Stage 1 的 instantaneous-RSSI threshold ratio `0.1` 產生 `association_mask`。
2. `association_mask` 在 $t=1,\ldots,T-1$ 不得重新計算；即使 UE 移動後其他 AP 變強，也保持原 serving set。
3. 每一期先更新位置與 true channel，再把全部 true channel 複製到 stored channel；所以本階段 `stored_csi[t]` 必須逐元素等於 `true_csi[t]`。
4. Centralized/decentralized GNN、MRT 與 RZF 在相同 period 使用相同 true channel、固定 mask、noise 與 power constraint。
5. Beamformer 每期以 current full CSI 重新計算；rate 也以同一期 true CSI 計算。Stage 2 不允許任何 stale channel leakage。

## 4. Stage 2A data contract

令 $B$ 為 trajectory batch size、$T$ 為 periods、$A=5$ 為 APs、$K$ 為 UEs、$M$ 為 antennas/AP。

| Value | Shape | Contract |
|---|---|---|
| `ue_positions` | `[B,T,K,2]` real | 公尺；每條軌跡符合固定速度與直線運動。 |
| `ue_speeds_mps` | `[B,K]` real | 一個 episode 內固定。 |
| `ue_directions_rad` | `[B,K]` real | 一個 episode 內固定。 |
| `true_channels` | `[B,T,A,K,M]` complex | 每期實際 direct AP–UE channel。 |
| `stored_channels` | `[B,T,A,K,M]` complex | Stage 2 必須與 `true_channels` 完全相同。 |
| `association_mask` | `[B,K,A]` boolean | $t=0$ 建立後整條 trajectory 固定。 |
| `centralized_features` | `[B,T,1,A*K,2*M]` real | Stage 1 AP-major layout 加入 time axis。 |
| `decentralized_features` | list of $A$ tensors `[B,T,1,K,2*M]` | 每個 AP 的 current local view。 |

Snapshot GNN 不新增 RNN/GRU。訓練 sampler 可從 training trajectories 抽 `(trajectory_id, t)` frames，但 train/validation/test 必須按完整 trajectory 切分，不能把同一軌跡的相鄰 frames 分到不同 splits，否則 temporal correlation 會造成資料洩漏。

## 5. Stage 2A source layout

Stage 2A source 位於 `code/stage2/`，來源只包含下列 Stage 1 source/config/test files；Stage 2B 直接擴充同一份 source tree，不再複製 trainer/model：

| Stage 2 file | 起點 | Stage 2A 責任 |
|---|---|---|
| `data.py` | `code/stage1/data.py` | 新增 trajectory generation、time-axis views、fixed-$t=0$ mask 與 true/stored CSI accessors。 |
| `utils_return_indivial_rates.py` | Stage 1 同名檔 | 加入 Jakes coefficient、Gauss–Markov transition 與 position-dependent sequence channel generation；保留 rate/MRT/RZF conventions。 |
| `model_2.py` | Stage 1 同名檔 | 初期不改 architecture；只在必要時支援將 `[B,T]` flatten/unflatten。 |
| `trainer_2.py` | Stage 1 同名檔 | 新增 mobility CLI/config、trajectory split/sampling、checkpoint evaluation-only、per-period evaluation 與 temporal diagnostics。 |
| `run_exp-v2.sh` | Stage 1 同名檔 | 開發期執行 frozen Stage 1B checkpoint 的 0/30/80 km/h pilot；full matched-training sweep 延後。 |
| `test_stage2.py` | `code/stage1/test_stage1.py` | 保留 Stage 1 assertions，再加入 mobility/channel/full-CSI/fixed-mask tests。 |

不得複製 `__pycache__/`、`results_stage1*/`、checkpoints、logs 或其他 artifacts。Stage 2 預設 output 必須使用 `results_stage2*`。複製當下記錄 source commit、dirty status 與六個來源檔案的 SHA-256；目前 Stage 1 working tree 有未提交修改，因此不能只記 `git rev-parse HEAD` 就宣稱 source provenance 完整。

目前 development pilot 的 CLI 介面如下；`--checkpoint` 存在時不建立 optimizer、不執行 training loop，只產生 traces 並評估。所有 effective values 與 checkpoint SHA-256 都寫入每個 run 的 config：

```bash
python trainer_2.py \
  --M 2 --K 8 --pmax_dbm 15 --batch_size 8 --runs 1 \
  --n_iter 2000 --noise_power 1e-12 \
  --speed_kmh 30 --decision_period_s 0.001 \
  --carrier_frequency_hz 2.6e9 --episode_steps 2000 \
  --train_trajectories 1 --test_sample_val 1 --test_sample_final 10 \
  --eval_time_stride 10 \
  --checkpoint ../stage1/results_stage1b_full_noise_1e-12/M2_K8_P15.0/run0/models/model_final_run0.pt \
  --device cuda:0 --out_dir results_stage2a_frozen_stage1_seed0_speed_30
```

## 6. Stage 2A mandatory gates

### Gate 2A.0 — Source isolation

- `code/stage2/` 僅含 source/test files，沒有 Stage 1 artifacts。
- 未修改 `code/stage1/`。
- 保存來源檔案 checksums、copy date 與 Stage 1 validation status。

### Gate 2A.1 — Paired backward compatibility

- 以相同 AP/UE positions、initial fast-fading draw、association mask 與 beamformers 比較 Stage 1 和 Stage 2 的 $t=0$ rate，誤差限 `atol=1e-6, rtol=1e-6`。
- $v=0$ 時 positions 與 true channel 在單一 episode 內保持不變；跨多條獨立 trajectories 的 $t=0$ channel/rate distribution 應與 Stage 1 相容。
- 不要求單一 stationary trajectory 的 time average 等於 Stage 1 ensemble mean，因為 $v=0$ 時重複的是同一 channel realization。

### Gate 2A.2 — Mobility kinematics

- 每一步位移誤差小於 `1e-10` m（boundary-free accepted straight segment）。
- 所有 positions 留在半徑 100 m disk 內。
- 固定 seed 產生完全相同的 positions、directions 與 channels；不同 seed 不得意外相同。

### Gate 2A.3 — Channel statistics

- 對 normalized $\mathbf g$ 而非含 path-loss 的 $\mathbf h$ 計算 correlation。
- 每個 speed 使用足夠 links/trajectories，報告 empirical lag-1 與多個 lag correlations、95% bootstrap CI，以及理論值 $\rho_k^{\ell}$。
- 預先 gate：theoretical value 落在 empirical 95% CI，或 absolute error `<= 0.02`；兩者至少符合一項。若樣本不足導致 CI 太寬，增加 trajectories，不調整 $\rho$。
- 每個 $t$ 的 normalized real/imaginary mean 接近 0、variance 接近 Stage 1 complex-normal convention，且所有 values finite。
- Distance/path-loss factor 必須逐期符合 $q_{a,k,t}$ 公式。

### Gate 2A.4 — Fixed association and full CSI

- `association_mask[:,t]` 若在 runtime 展開，必須對所有 $t$ 與 $t=0$ 完全相同。
- `stored_channels` 與 `true_channels` 必須逐元素相同。
- 所有方法每期共用同一 current channel/mask；rate 只由 true current channel 評估。
- 每個 AP 的 beamformer 仍符合 association mask 與 per-AP power limit。

### Gate 2A.5 — Controlled experiment

**Development gate（進入 Stage 3 前）**

- 將 Stage 1B seed-0 frozen checkpoint evaluation-only 於 homogeneous 0/30/80 km/h；3 km/h 延後至 evidence gate。
- 每個 speed 使用固定 seed、可重現且彼此獨立的 test trajectories；pilot 使用 10 條 test trajectories，並以 `eval_time_stride=10` 評估每條軌跡的 200 個等距 frames。完整 2000-step environment/channel traces 與 diagnostics 仍保存。跨 speed 的嚴格 paired innovations 留到 evidence gate，開發期不把 frame 當成配對統計樣本。
- 不要求跨 seed 推論；只檢查 end-to-end execution、finite metrics、constraints 與初步趨勢。
- 現有 speed-0 matched-training run 完成後保留為單一 adaptation diagnostic，不自動繼續後續 19 runs。

**Evidence gate（方法與介面凍結後再補）**

- 凍結的 Stage 1B checkpoints 完成 0/3/30/80 km/h paired evaluation，至少 5 seeds；主結果使用 `eval_time_stride=1`，並可將 stride-10 結果列為計算成本 sensitivity。
- 若需宣稱 mobility-specific adaptation，才對相同 settings 增加 matched-training models，並報告 adaptation gain。

保存：

- long-term average sum rate 與 5th-percentile UE rate；
- centralized/decentralized/MRT/RZF 的 paired per-trajectory metrics；
- empirical channel autocorrelation vs. lag；
- UE trajectories、distance/path-loss traces、fixed association mask；
- config、seed、source checksum、checkpoint、raw metrics 與 logs。

速度上升不預先要求 sum rate 單調下降：Gauss–Markov model 改變 temporal correlation，不改 normalized channel 的 marginal distribution；固定 association 下的 rate 也同時受具體移動方向與 distance evolution 影響。Stage 2A 的 mandatory conclusion 是 environment 是否符合預先定義的 dynamics，而不是某方法是否勝出。

## 7. Stage 2A 與 Stage 2B 明確不做的事

- 不依新位置重新計算 association。
- 不限制 feedback budget，也不保留 stale CSI。
- 不讓 model 讀取速度、方向、history 或 future CSI。
- 不加入 RNN/GRU、RL、switching cost 或 CSI-update action。
- 不把 Ammar et al. 的 1 s/5 s handoff step 當作 CSI channel block。
- 不宣稱 simplified model 為完整 3GPP-compliant channel。
- 不根據 centralized/decentralized gap、GNN 是否勝過 MRT/RZF，或個別 seed 的結果調整 mobility parameters。
- 不讓 rate、AP association 或任何待比較方法的輸出控制 Stage 2B 的 hotspot transition；否則 mobility 會成為內生 policy，破壞方法間的共同環境。
- 沒有實際 mobility trace 校準時，不把 Stage 2B 宣稱為真實人類 mobility ground truth；只稱 synthetic hotspot sensitivity model。

## 8. Stage 2B — Hotspot-aware mobility robustness

### 8.1 Research question、變因與預期作用

Stage 2B 的 independent variable 是 mobility generator：Stage 2A straight line 對 Stage 2B hotspot-aware Markov/semi-Markov。Channel、AP geometry、power/noise convention、fixed association、full current CSI、beamforming methods、training budget 與 evaluation code 都保持不變。

Hotspot model 不預先假設 rate 一定下降。若 hotspot 靠近 AP，平均 rate 可能增加；若 hotspot 位於不同 AP 的優勢區或邊界，association mismatch、handoff pressure 與 tail-rate loss 可能增加。主要 estimand 是 mobility model 是否改變 centralized/decentralized gap：

$$
\Delta_{\mathrm{interaction}}
=
\left(R_C-R_D\right)_{\mathrm{hotspot}}
-
\left(R_C-R_D\right)_{\mathrm{straight}}.
$$

Stage 2B 同時檢查方法排名、5th-percentile UE rate 與 fixed-association regret，不能只比較兩個 environment 的 raw mean rate，因為兩者的 spatial occupancy 本來就可能不同。

### 8.2 兩個 timescales 與連續位置

令 $z_{k,n}\in\{1,\ldots,J\}$ 為 UE $k$ 在第 $n$ 個 macro mobility epoch 的 hotspot state，$q_n$ 為預先固定的 time-period class：

$$
z_{k,n+1}\sim P_k^{(q_n)}\left(z_{k,n},\cdot\right),
\qquad
D_{k,n}\sim F_{z_{k,n},q_n}.
$$

$P_k^{(q)}$ 控制 hotspot transition 與 revisit；$D_{k,n}$ 是該 hotspot 的 dwell duration。若 $D$ 完全由固定 macro tick 的 self-transition 產生，模型可退化為普通 Markov chain；主 sensitivity 使用顯式 dwell distribution，避免把每個 1 ms channel block 當成一次人類 destination decision。

每次 transition 先在下一 hotspot 區域內抽取 physical target $\mathbf y_{k,n+1}$，再沿用 Stage 2A 的 constant-speed integrator：

$$
\mathbf x_{k,t+1}
=
\mathbf x_{k,t}
+
\min\!\left(v_{k,t}\Delta t,
\lVert\mathbf y_{k,n+1}-\mathbf x_{k,t}\rVert_2\right)
\frac{\mathbf y_{k,n+1}-\mathbf x_{k,t}}
{\lVert\mathbf y_{k,n+1}-\mathbf x_{k,t}\rVert_2}.
$$

- 不允許 hotspot 切換時 teleportation、position discontinuity 或跨 boundary 瞬移。
- 若抽到與目前位置相同的 target，直接進入 dwell，不計算零長度方向向量。
- Transit 使用受控速度；dwell 可設為靜止或 hotspot 內局部移動。對應的 instantaneous $v_{k,t}$ 必須用於 $\rho_{k,t}=J_0(2\pi f_c v_{k,t}\Delta t/c)$。
- Fine channel block 保留 $\Delta t=1$ ms；hotspot transition/dwell 位於較慢的 macro timescale。Ammar et al. [S2-3] 也將 5 s mobility/HO decision step 與 $66.7\ \mu$s channel sampling period 分開。
- 現有 2 s episode 中，3/30/80 km/h 只約移動 1.67/16.67/44.44 m，通常不足以觀察多次 hotspot revisit。Stage 2B 先生成足以含多個 transition 的 long macro trace，再按 trace 中實際時間占比抽取 2 s clips；另分開報告 dwell 與 transit clips，不改變主分析的自然時間權重。
- 每個 2 s clip 的 association mask 在 clip 的 $t=0$ 建立後固定，以維持 Stage 2 語意。從 long macro episode 起點一路凍結 association 的結果只能列為 secondary stress test。

### 8.3 Hotspot geometry 與參數凍結

- Hotspot centers、半徑、transition matrices、dwell distributions 與 time-period classes 必須在查看 rate 結果前凍結並寫入 config。
- Hotspot geometry 不得由 centralized/decentralized model、learned beamformer 或 rate surface 產生。
- 每個 UE 可有不同的 frequent hotspots，但在 population level 需用 seed permutation 平衡 hotspot assignment，避免所有 UEs 被人為集中到同一 AP 附近。
- 若沒有實際 trace，至少預先登記 low/medium/high stickiness 或 dwell sensitivity；不可只保留結果最有利的一組參數。
- Hsu et al. [S2-5] 明確指出 community parameters 依 target scenario 而定，且其 trace matching 需要調整 attraction 與 pause time。因此 [S2-5] 支持模型結構，不直接提供本研究可照抄的 transition matrix。

### 8.4 Stage 2B data contract additions

Stage 2A 的所有 arrays 繼續保留，另加入：

| Value | Shape | Contract |
|---|---|---|
| `mobility_model` | scalar string | `straight` 或 `hotspot_semi_markov`；寫入 config 與每個 result root。 |
| `hotspot_centers` | `[J,2]` real | 公尺；固定於該 experiment config，全部位於 UE disk 內。 |
| `hotspot_state` | `[B,T,K]` integer | Dwell 時為所在 hotspot；transit 時為 destination hotspot，並由 phase 明確區分。 |
| `mobility_phase` | `[B,T,K]` enum | `dwell` 或 `transit`。 |
| `instantaneous_speeds_mps` | `[B,T,K]` real | 用於位置更新與 time-varying Jakes coefficient。 |
| `transition_matrix` | config array | 每列總和為 1，元素非負；time-varying 時保存全部 $P^{(q)}$。 |
| `dwell_durations_s` | per-event array | 保存實際抽樣值，用於 distribution recovery 與 reproducibility。 |

實作時直接擴充 `code/stage2/` 的 mobility generator 與測試，新增 `--mobility_model`；不複製另一套 trainer/model tree。Stage 2B artifacts 使用獨立的 `results_stage2b_*` roots。

### 8.5 Controlled comparison

開發門檻先以同一個 frozen Stage 1B checkpoint 評估 paired straight/hotspot test traces，用來驗證 mobility-model shift 的 evaluation pipeline，不需先重新訓練。若初步 interaction 大到可能改變 downstream 設計，Stage 3 就同時保留 straight/hotspot environments；若不大，Stage 3 先以 straight 開發，hotspot 作 regression test。

方法與介面凍結後，Stage 2B evidence gate 再完成下列 train/test matrix：

| Train mobility | Test: straight | Test: hotspot |
|---|---|---|
| Straight | in-distribution baseline | distribution-shift robustness |
| Hotspot | reverse transfer | hotspot matched training |

- Centralized GNN、decentralized GNN、MRT 與 RZF 使用完全相同的 test traces、mask、true CSI、noise 與 power constraint。
- 可共用的 initial positions、normalized small-scale innovations 與 random-number streams 按 seed 配對；mobility-specific draws 分開保存。
- Train/validation/test 必須按 parent long macro trace 分割；同一 long trace 的不同 2 s clips 不得跨 split。
- Evidence gate 至少使用 10 個獨立 training seeds。這是論文 inference requirement，不是 Stage 3 工程開發的 blocker。五個 non-zero paired observations 的雙尾 exact sign-flip test 最小 $p=0.0625$，不足以在 $\alpha=0.05$ 下拒絕虛無假設。
- Statistical unit 是 seed-level 或完整 trajectory-level aggregate；同一 trajectory 的 2000 個 temporally correlated frames 不得當作 2000 個獨立樣本。
- 報告 paired effect、95% CI 與 effect size；多個 method/metric comparisons 使用 Holm correction。Mobility robustness 優先用事前 practical-equivalence margin 判定，不以單一 $p$ value 代替實務差異。

Primary metrics：

1. Long-term average sum rate 與 5th-percentile UE rate。
2. $\Delta_{\mathrm{interaction}}$ 與四個方法的 ranking stability。
3. Strongest-AP change count、原 serving set coverage 與 frozen-association regret。
4. Hotspot occupancy、transition matrix 與 dwell-time distribution recovery。
5. Dwell/transit 分段的 distance、path loss 與 normalized-channel autocorrelation。

### 8.6 Stage 2B mandatory gates 與 downstream decision

#### Gate 2B.1 — Trace validity

- 固定 seed 可重現全部 hotspot states、dwell events、positions 與 channels；不同 seed 不得意外相同。
- Empirical transition rows 與 configured $P^{(q)}$ 相容，empirical dwell distribution 通過預先指定的 distribution/quantile tolerance。
- Long-run hotspot occupancy 與理論或模擬目標 stationary occupancy 相容；若 time-varying，分 $q$ 報告而不假設單一 stationary distribution。

#### Gate 2B.2 — Physical and channel continuity

- 所有 positions 位於 UE disk；每步位移不超過 $v_{k,t}\Delta t+10^{-10}$ m。
- Hotspot transition 前後沒有 position、distance、path-loss 或 true-channel teleportation。
- Time-varying normalized-channel correlation 依 instantaneous speed/dwell regime 符合理論與 Stage 2A Gate 2A.3 的 tolerance。

#### Gate 2B.3 — Fixed association and full CSI

- 每個 evaluation clip 的 mask 只在 $t=0$ 建立一次，之後固定。
- `stored_channels == true_channels`；Stage 2B 不提前加入 stale CSI、dynamic association 或 history observation。

#### Gate 2B.4 — Robustness decision

- 在查看結果前登記 sum-rate、tail-rate 與 C/D-gap 的 practical-equivalence margins。
- Development gate 只要 frozen-checkpoint straight/hotspot evaluation 能成對完成、metrics finite 且 2B.1–2B.3 通過，即可進入 Stage 3；單一 seed 不作 equivalence 或 significance 主張。
- 若 $\Delta_{\mathrm{interaction}}$ 的 95% CI 落在等效界線內且方法排名穩定，Stage 2A 保留為主 baseline，Stage 2B 作 robustness evidence。
- 若 interaction 超出界線、方法排名翻轉或 frozen-association regret 顯著改變，後續 Stage 3–5 必須同時報告 straight-line 與 hotspot results，不得把結論寫成 mobility-model independent。
- 真正由 UE action/reward 控制 destination 的 MDP 若未來要做，必須另立 joint mobility-control research question，不納入 Stage 2B。

## 9. 論文證據與採用邊界

| Source | 可核對證據 | Stage 2 如何使用／不使用 |
|---|---|---|
| [S2-1] Deng et al., 2019 | 使用 first-order stationary Gauss–Markov $\mathbf h_k(t+1)=\rho_k\mathbf h_k(t)+\sqrt{1-\rho_k^2}\mathbf e_k(t)$，並以 Jakes $J_0$ 將 $\rho_k$ 連到速度；simulation 使用 2.6 GHz、1 ms subframe、2000 blocks，例示 20/80 km/h 與 40–80 km/h。 | 作為 core small-scale model、$f_c$、$\Delta t$、episode length 與 high-speed stress prior；不照抄其 single-cell antennas、SNR 或 pilot配置。 |
| [S2-2] Ammar et al., 2024 | 在 UC CF-MIMO 中使用 Jakes channel aging；UE 以 10 m/s、1 s step 走 1 km straight line；LSF 由距離 path loss 與 spatially correlated shadowing組成。 | 支持 straight-line CF mobility、位置驅動 path loss與 later shadowing sensitivity；其 1 s 是 handoff/LSF timescale，不作本研究 CSI block。 |
| [S2-3] Ammar et al., 2025 | 明確允許 straight line 或 random waypoint；simulation 使用 10 m/s、5 s decision step，並比較 direction-assisted 與 LSF-history-assisted observations。 | 支持之後 Stage 3/6 的 trajectory/history choices；不在 Stage 2 提前加入 history observation 或 DRL。 |
| [S2-4] 3GPP TR 38.901 | 列出 3 km/h mobility example、30 km/h fixed-speed/random-direction calibration example，並規範 spatially consistent mobility updates。 | 支持 representative low/urban speeds與位置更新 sanity check；本階段未實作完整 cluster/ray spatial consistency。 |
| [S2-5] Hsu et al., 2007 | WLAN traces 顯示偏斜的 location preference 與週期性返回；TVC model 以 local/roaming Markov chain、community attraction 與 pause time 建模，並提出可用固定 points of interest 作共同 attraction points。 | 支持 Stage 2B 的 hotspot/Markov 結構；其 campus/corporate trace 與參數不直接視為本 CF-mMIMO geometry 的 ground truth。 |
| [S2-6] González et al., 2008 | 分析 100,000 名手機使用者六個月資料，觀察到高度時空規律與返回少數高頻位置的傾向。 | 支持將 revisit/frequent-location 納入 robustness；不提供本研究的 transition matrix、dwell 或 wireless-channel parameters。 |
| [S2-7] Jiang et al., 2016 | TimeGeo 使用 time-inhomogeneous Markov model 與 individual dwell/burst parameters 生成 urban mobility，並區分 temporal destination choice 與 spatial destination selection。 | 支持 Stage 2B 顯式建模 dwell 與較慢 macro timescale；其 10-minute/hundreds-of-meters urban resolution 不套用為 1 ms channel block。 |

## References

[S2-1] R. Deng, Z. Jiang, S. Zhou, and Z. Niu, “Intermittent CSI Update for Massive MIMO Systems With Heterogeneous User Mobility,” *IEEE Transactions on Communications*, vol. 67, no. 7, pp. 4811–4824, Jul. 2019. [DOI](https://doi.org/10.1109/TCOMM.2019.2911575) · [Author manuscript](https://network.ee.tsinghua.edu.cn/niulab/wp-content/uploads/2019/12/Intermittent-CSI-Update-for-Massive-MIMO-Systems-With-Heterogeneous-User-Mobility.pdf)

[S2-2] H. A. Ammar, R. Adve, S. Shahbazpanahi, G. Boudreau, and K. V. Srinivas, “Handoffs in User-Centric Cell-Free MIMO Networks: A POMDP Framework,” *IEEE Transactions on Wireless Communications*, vol. 23, no. 8, pp. 10319–10335, Aug. 2024. [arXiv](https://arxiv.org/abs/2403.08900)

[S2-3] H. A. Ammar, R. Adve, S. Shahbazpanahi, G. Boudreau, and I. Bahceci, “Handoff Design in User-Centric Cell-Free Massive MIMO Networks Using DRL,” *IEEE Transactions on Communications*, vol. 73, no. 11, pp. 11368–11384, Nov. 2025. [arXiv](https://arxiv.org/abs/2507.20966)

[S2-4] 3GPP, “Study on Channel Model for Frequencies from 0.5 to 100 GHz,” TR 38.901, Release 18, v18.1.0, Feb. 2026. [ETSI PDF](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/18.01.00_60/tr_138901v180100p.pdf)

[S2-5] W.-J. Hsu, T. Spyropoulos, K. Psounis, and A. Helmy, “Modeling Time-Variant User Mobility in Wireless Mobile Networks,” in *Proc. IEEE INFOCOM*, pp. 758–766, May 2007. [DOI](https://doi.org/10.1109/INFCOM.2007.94) · [Author manuscript](https://www.cise.ufl.edu/~helmy/papers/TVC-Infocom-07-published.pdf)

[S2-6] M. C. González, C. A. Hidalgo, and A.-L. Barabási, “Understanding Individual Human Mobility Patterns,” *Nature*, vol. 453, pp. 779–782, Jun. 2008. [DOI](https://doi.org/10.1038/nature06958)

[S2-7] S. Jiang, Y. Yang, S. Gupta, D. Veneziano, S. Athavale, and M. C. González, “The TimeGeo Modeling Framework for Urban Mobility Without Travel Surveys,” *Proceedings of the National Academy of Sciences*, vol. 113, no. 37, pp. E5370–E5378, Sep. 2016. [DOI](https://doi.org/10.1073/pnas.1524261113) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5027456/)
