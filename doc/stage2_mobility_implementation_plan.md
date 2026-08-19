# Stage 2 Mobility with Fixed Association and Full CSI — Implementation Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-18
- Last Updated: 2026-08-19
- Verification Status: UNVERIFIED（文獻模型與參數已核對；Stage 2 尚未實作或執行）
- Version Label: stage2_mobility_plan_v1
- Parent Plan: `doc/decentralized_active_csi_experiment_plan.md`
- Stage 1 Source: `code/stage1/`
- Stage 1B Gate: ANALYZED / PASSED（`noise_power=1e-12`、5 seeds、2000 iterations；數值與 artifact gate 通過，不代表 C/D 等效）
- Stage 1 Report: `doc/decentralized_active_csi_experiment_report.md`

## 1. Stage 2 要回答的問題

Stage 2 只回答：在 Stage 1B 的 no-RIS channel/power/noise convention 下，加入可重現的 UE mobility 與 temporally correlated channel 後，既有 snapshot beamforming pipeline 是否仍能在固定 association、每期 full current CSI 的條件下正確運作？

本階段不是要證明 temporal policy 優於 snapshot policy，也不加入 dynamic association、feedback budget、stale CSI、CSI age action 或 RL。Stage 2 的主要產物是後續 Stage 3–5 共用且已通過統計檢查的 temporal environment。

## 2. 預先固定的主設計

| 項目 | Stage 2 主設定 | 理由 |
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

## 3. User mobility 的正式定義

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

## 4. Stage 2 data contract

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

## 5. 新資料夾與 source-copy 計畫

實作獲准後先建立 `code/stage2/`，只複製下列 Stage 1 source files：

| Stage 2 file | 起點 | 預定修改 |
|---|---|---|
| `data.py` | `code/stage1/data.py` | 新增 trajectory generation、time-axis views、fixed-$t=0$ mask 與 true/stored CSI accessors。 |
| `utils_return_indivial_rates.py` | Stage 1 同名檔 | 加入 Jakes coefficient、Gauss–Markov transition 與 position-dependent sequence channel generation；保留 rate/MRT/RZF conventions。 |
| `model_2.py` | Stage 1 同名檔 | 初期不改 architecture；只在必要時支援將 `[B,T]` flatten/unflatten。 |
| `trainer_2.py` | Stage 1 同名檔 | 新增 mobility CLI/config、trajectory split/sampling、per-period evaluation 與 temporal diagnostics。 |
| `run_exp-v2.sh` | Stage 1 同名檔 | 改成 Stage 2 speed sweep 與獨立 output roots。 |
| `test_stage2.py` | `code/stage1/test_stage1.py` | 保留 Stage 1 assertions，再加入 mobility/channel/full-CSI/fixed-mask tests。 |

不得複製 `__pycache__/`、`results_stage1*/`、checkpoints、logs 或其他 artifacts。Stage 2 預設 output 必須使用 `results_stage2*`。複製當下記錄 source commit、dirty status 與六個來源檔案的 SHA-256；目前 Stage 1 working tree 有未提交修改，因此不能只記 `git rev-parse HEAD` 就宣稱 source provenance 完整。

建議 CLI 介面如下；名稱在實作時可微調，但所有 effective values 都必須寫入每個 run 的 config：

```bash
python trainer_2.py \
  --M 2 --K 8 --pmax_dbm 15 --batch_size 8 --runs 5 \
  --n_iter 2000 --noise_power 1e-12 \
  --speed_kmh 30 --decision_period_s 0.001 \
  --carrier_frequency_hz 2.6e9 --episode_steps 2000 \
  --device cuda:0 --out_dir results_stage2_speed_30
```

## 6. 實作順序與 mandatory gates

### Gate 2.0 — Source isolation

- `code/stage2/` 僅含 source/test files，沒有 Stage 1 artifacts。
- 未修改 `code/stage1/`。
- 保存來源檔案 checksums、copy date 與 Stage 1 validation status。

### Gate 2.1 — Paired backward compatibility

- 以相同 AP/UE positions、initial fast-fading draw、association mask 與 beamformers 比較 Stage 1 和 Stage 2 的 $t=0$ rate，誤差限 `atol=1e-6, rtol=1e-6`。
- $v=0$ 時 positions 與 true channel 在單一 episode 內保持不變；跨多條獨立 trajectories 的 $t=0$ channel/rate distribution 應與 Stage 1 相容。
- 不要求單一 stationary trajectory 的 time average 等於 Stage 1 ensemble mean，因為 $v=0$ 時重複的是同一 channel realization。

### Gate 2.2 — Mobility kinematics

- 每一步位移誤差小於 `1e-10` m（boundary-free accepted straight segment）。
- 所有 positions 留在半徑 100 m disk 內。
- 固定 seed 產生完全相同的 positions、directions 與 channels；不同 seed 不得意外相同。

### Gate 2.3 — Channel statistics

- 對 normalized $\mathbf g$ 而非含 path-loss 的 $\mathbf h$ 計算 correlation。
- 每個 speed 使用足夠 links/trajectories，報告 empirical lag-1 與多個 lag correlations、95% bootstrap CI，以及理論值 $\rho_k^{\ell}$。
- 預先 gate：theoretical value 落在 empirical 95% CI，或 absolute error `<= 0.02`；兩者至少符合一項。若樣本不足導致 CI 太寬，增加 trajectories，不調整 $\rho$。
- 每個 $t$ 的 normalized real/imaginary mean 接近 0、variance 接近 Stage 1 complex-normal convention，且所有 values finite。
- Distance/path-loss factor 必須逐期符合 $q_{a,k,t}$ 公式。

### Gate 2.4 — Fixed association and full CSI

- `association_mask[:,t]` 若在 runtime 展開，必須對所有 $t$ 與 $t=0$ 完全相同。
- `stored_channels` 與 `true_channels` 必須逐元素相同。
- 所有方法每期共用同一 current channel/mask；rate 只由 true current channel 評估。
- 每個 AP 的 beamformer 仍符合 association mask 與 per-AP power limit。

### Gate 2.5 — Controlled experiment

依序執行 homogeneous 0/3/30 km/h 與 80 km/h stress test，每個 setting 至少 5 seeds，train/validation/test trajectories 相同且按 trajectory 配對。保存：

- long-term average sum rate 與 5th-percentile UE rate；
- centralized/decentralized/MRT/RZF 的 paired per-trajectory metrics；
- empirical channel autocorrelation vs. lag；
- UE trajectories、distance/path-loss traces、fixed association mask；
- config、seed、source checksum、checkpoint、raw metrics 與 logs。

速度上升不預先要求 sum rate 單調下降：Gauss–Markov model 改變 temporal correlation，不改 normalized channel 的 marginal distribution；固定 association 下的 rate 也同時受具體移動方向與 distance evolution 影響。Stage 2 的 mandatory conclusion 是 environment 是否符合預先定義的 dynamics，而不是某方法是否勝出。

## 7. 本階段明確不做的事

- 不依新位置重新計算 association。
- 不限制 feedback budget，也不保留 stale CSI。
- 不讓 model 讀取速度、方向、history 或 future CSI。
- 不加入 RNN/GRU、RL、switching cost 或 CSI-update action。
- 不把 Ammar et al. 的 1 s/5 s handoff step 當作 CSI channel block。
- 不宣稱 simplified model 為完整 3GPP-compliant channel。
- 不根據 centralized/decentralized gap、GNN 是否勝過 MRT/RZF，或個別 seed 的結果調整 mobility parameters。

## 8. 論文證據與採用邊界

| Source | 可核對證據 | Stage 2 如何使用／不使用 |
|---|---|---|
| [S2-1] Deng et al., 2019 | 使用 first-order stationary Gauss–Markov $\mathbf h_k(t+1)=\rho_k\mathbf h_k(t)+\sqrt{1-\rho_k^2}\mathbf e_k(t)$，並以 Jakes $J_0$ 將 $\rho_k$ 連到速度；simulation 使用 2.6 GHz、1 ms subframe、2000 blocks，例示 20/80 km/h 與 40–80 km/h。 | 作為 core small-scale model、$f_c$、$\Delta t$、episode length 與 high-speed stress prior；不照抄其 single-cell antennas、SNR 或 pilot配置。 |
| [S2-2] Ammar et al., 2024 | 在 UC CF-MIMO 中使用 Jakes channel aging；UE 以 10 m/s、1 s step 走 1 km straight line；LSF 由距離 path loss 與 spatially correlated shadowing組成。 | 支持 straight-line CF mobility、位置驅動 path loss與 later shadowing sensitivity；其 1 s 是 handoff/LSF timescale，不作本研究 CSI block。 |
| [S2-3] Ammar et al., 2025 | 明確允許 straight line 或 random waypoint；simulation 使用 10 m/s、5 s decision step，並比較 direction-assisted 與 LSF-history-assisted observations。 | 支持之後 Stage 3/6 的 trajectory/history choices；不在 Stage 2 提前加入 history observation 或 DRL。 |
| [S2-4] 3GPP TR 38.901 | 列出 3 km/h mobility example、30 km/h fixed-speed/random-direction calibration example，並規範 spatially consistent mobility updates。 | 支持 representative low/urban speeds與位置更新 sanity check；本階段未實作完整 cluster/ray spatial consistency。 |

## References

[S2-1] R. Deng, Z. Jiang, S. Zhou, and Z. Niu, “Intermittent CSI Update for Massive MIMO Systems With Heterogeneous User Mobility,” *IEEE Transactions on Communications*, vol. 67, no. 7, pp. 4811–4824, Jul. 2019. [DOI](https://doi.org/10.1109/TCOMM.2019.2911575) · [Author manuscript](https://network.ee.tsinghua.edu.cn/niulab/wp-content/uploads/2019/12/Intermittent-CSI-Update-for-Massive-MIMO-Systems-With-Heterogeneous-User-Mobility.pdf)

[S2-2] H. A. Ammar, R. Adve, S. Shahbazpanahi, G. Boudreau, and K. V. Srinivas, “Handoffs in User-Centric Cell-Free MIMO Networks: A POMDP Framework,” *IEEE Transactions on Wireless Communications*, vol. 23, no. 8, pp. 10319–10335, Aug. 2024. [arXiv](https://arxiv.org/abs/2403.08900)

[S2-3] H. A. Ammar, R. Adve, S. Shahbazpanahi, G. Boudreau, and I. Bahceci, “Handoff Design in User-Centric Cell-Free Massive MIMO Networks Using DRL,” *IEEE Transactions on Communications*, vol. 73, no. 11, pp. 11368–11384, Nov. 2025. [arXiv](https://arxiv.org/abs/2507.20966)

[S2-4] 3GPP, “Study on Channel Model for Frequencies from 0.5 to 100 GHz,” TR 38.901, Release 18, v18.1.0, Feb. 2026. [ETSI PDF](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/18.01.00_60/tr_138901v180100p.pdf)
