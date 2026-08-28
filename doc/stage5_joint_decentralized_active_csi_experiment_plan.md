# Stage 5 — Joint Decentralized Active-CSI Control 實驗計畫

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-26
- Last Updated: 2026-08-28
- Verification Status: ACTIVE（Stage 5A Gates 5.0–5.4已通過；Stage 5B frozen-GNN baseline與beamformer-matched RL尚未實作或執行）
- Version Label: stage5_joint_decentralized_active_csi_plan_v5_compacted
- Parent Plan: `doc/decentralized_active_csi_experiment_plan.md`
- Living Report: `doc/decentralized_active_csi_experiment_report.md`
- Stage 3 Handoff: `doc/stage3_dynamic_association_experiment_plan.md`
- Stage 4 Handoff: `doc/stage4_feedback_csi_experiment_plan.md`
- Code Target: `code/stage5/`

## 1. 唯一任務

在相同per-AP hard feedback budget下，檢驗只使用AP-local stored CSI、CSI age/history、previous update、previous association與local slow-timescale information的controller，能否改善frozen topology-matched decentralized GNN的long-term rate–switching–freshness trade-off。

- Primary：$\pi_D+$frozen decentralized GNN。
- Upper reference：$\pi_C+$frozen centralized GNN。
- Conventional control：$\pi_{\mathrm{RZF}}+$RZF。
- 每個policy必須用matching beamformer reward訓練；RZF-trained policy換接GNN只算transfer diagnostic。
- Frozen GNN不fine-tune；本方法不能稱為end-to-end jointly trained RL–GNN。

Stage 5A joint evaluator與RZF modular controls已完成。下一個任務只做Stage 5B frozen-GNN baseline與AP-local learned integration。Stage 3 SAC與Stage 4 B1/B3/random sweep已封存，不再補seed或帶入formal matrix。

## 2. Frozen handoff

| Item | Stage 5 setting |
|---|---|
| Topology | 200 m × 200 m square torus；5 APs、8 UEs、10 m height difference |
| Physical layer | $M=2$、15 dBm/AP、`noise_power=1e-12` |
| Channel | 1 ms frame、2.6 GHz、position-dependent path loss + Gauss–Markov fading |
| Trajectory | straight 0/30/80 km/h、2000 frames |
| Association | every 50 ms；top-2 APs/UE；no per-AP capacity |
| Feedback | every 1 ms；primary $B=2$ per AP |
| Modular association | H3 |
| Modular scheduler | `mobility_age_priority` |
| Fair-refresh control | `round_robin` |
| Frozen GNN | Stage 1C seed-0 checkpoint；same parameters for C/D inference |
| Development sample | environment seed 0、10 paired trajectories/speed、stride 1 |

Primary development只用30/80 km/h；0 km/h是stationary boundary。Hotspot、mixed speed、$B\ne2$ performance sweep、association interval與multi-topology不在Stage 5 matrix。

## 3. Code hierarchy

`code/stage5/`以Stage 4為primary base，因true/stored CSI state machine與no-leakage contract比Stage 3 full-current evaluator更關鍵。Stage 3只提供top-2、H3、switching與frozen GNN formatting/loading components。

| Responsibility | Source |
|---|---|
| Mobility/channel/RZF/rate | Stage 4 `environment.py`、`utils_return_indivial_rates.py` |
| Stored CSI/age/budget/evaluator | Stage 4 `feedback.py`、`evaluate.py` |
| Association/switching | Stage 3 `association.py` |
| Frozen C/D GNN | Stage 3 `model_2.py`與dynamic-mask formatting；建立Stage 5 local adapter |
| Existing RL mechanics | Stage 3 training code只參考replay/checkpoint/finite patterns |

Stage 1–4 source不得runtime import；Stage 5使用stage-local copies與source snapshot。Gate 5.5前只需既有`environment.py`、`association.py`、`feedback.py`、`controller.py`、`evaluate.py`、numerical utility、test與runner。Gate 5.5/5.6再最小新增frozen `model_2.py` adapter、AP-local actor/critic、trainer與runner；不建立shared framework或新增dependency。

## 4. Joint causal state

令true channel、stored CSI、age、association與update action為$h_{t,m,k}$、$\hat h_{t,m,k}$、$a_{t,m,k}$、$A_{t,m,k}$、$U_{t,m,k}$。

### 4.1 Initialization

- $t=0$以current LSF top-2建立所有methods共用的$A_0$；initial association不算switching。
- 對全部5×8 candidate links共同bootstrap：$\hat h_0=h_0$、$a_0=0$。
- Bootstrap不計入steady-state B2 usage，但獨立記錄40-link initialization cost。
- Unassociated links仍保留stored CSI與age，使日後rejoin不需偷看current CSI。

### 4.2 Every frame

1. Environment形成true channel，但不向controller揭露。
2. Association epoch時，AP-local bids經UE-side deterministic top-2 arbitration形成$A_t$；其餘frames沿用$A_{t-1}$。
3. 每AP用post-association active set與pre-action legal state選最多2條feedback links。
4. 只揭露selected links的$h_t$；更新stored CSI並reset age，其餘links保留stored value且age加1。
5. Frozen RZF/C/D GNN只用各自合法visibility下的$\hat h_t$與$A_t$；rate用true $h_t$。
6. 保存reward components、raw/projected actions、age、NMSE、usage、power與constraint diagnostics。

固定順序為 **association bids → UE top-2 arbitration → feedback selection → selected CSI reveal → frozen beamformer → true-channel rate**。

## 5. Local observation與 actions

AP $m$ actor只可讀：

- 本AP的stored complex CSI或凍結表示；
- per-link CSI age/history與previous update mask；
- previous association row與local AP load；
- local slow-timescale LSF、predeclared correlation metadata與budget。

Current LSF視為local slow-timescale measurement，不計入instantaneous small-scale CSI budget；此假設必須寫入config與limitations。Actor不得讀其他AP raw observations、未選link current CSI、future channel/position或test normalization statistics。

Parameter-shared actor為每AP輸出8個association bids與8個feedback scores。Projection固定為UE-side top-2 association，再由AP在active set取top-B updates：

$$
\sum_m A_{t,m,k}=2,\qquad U_{t,m,k}\le A_{t,m,k},\qquad \sum_k U_{t,m,k}\le2.
$$

Training可用centralized twin critic與global reward，但evaluation actor不得讀global observation。第一版不加RNN/GNN actor；UE-side arbitration communication必須明示並計數，不能宣稱零協調。

## 6. Reward與 estimands

對reward beamformer$f\in\{\mathrm{RZF},C,D\}$，每50 ms association window：

$$
r_j^{(f)}=
\frac{1}{50K}\sum_{t=50j}^{50j+49}\sum_{k=1}^{K}R_{t,k}^{(f)}
-0.5\frac{\lVert A_j-A_{j-1}\rVert_1}{5K}.
$$

$A_0$不計switching。Primary B2 policies填滿可用budget，因此不加feedback penalty；actual updates/AP/frame與active-link update fraction仍獨立報告。

Primary paired estimands：

- time-average sum rate與relative-to-B8 retention；
- UE time-average rate的5th percentile；
- link toggles/UE/s與serving-set changes；
- active-link mean/p95/max age、never-refreshed fraction與energy-weighted CSI NMSE；
- feedback usage、power/mask/constraint violations與message count。

## 7. Stage 5A completed handoff

Stage 5A已通過Gates 5.0–5.4：provenance、joint-state tests、short smoke、Stage 3/4 parent boundaries與RZF modular matrix。30/80 km/h的H3+priority為2×2 matrix最高mean-rate cell，但priority同時降低UE p05並造成weak-link starvation。

這只證明joint evaluator可用且兩種actions有nonzero signal。逐cell與SAC diagnostic數字不在active plan重複；最小結果見living report，raw values見Stage 5A artifacts。

## 8. Stage 5B matrix

### 8.1 Frozen-GNN no-RL qualification

在相同paired traces完成：

| Beamformer | Simple control | Matching no-RL comparator | Information anchor |
|---|---|---|---|
| RZF | `fixed+RR@B2` | `H3+priority@B2` | `H3+B8` |
| Centralized GNN | `fixed+RR@B2` | `H3+priority@B2` | `H3+B8` |
| Decentralized GNN | `fixed+RR@B2` | `H3+priority@B2` | `H3+B8` |

RZF cells已由Stage 5A提供；C/D GNN尚未執行。B8用於full-current compatibility與information loss，不是B2 method comparator。

### 8.2 Matching RL paths

$\pi_{\mathrm{RZF}}$、$\pi_C$、$\pi_D$共用actor architecture、split、transition ceiling與checkpoint-selection rule，但分別使用matching frozen beamformer reward。每個path只以同beamformer比較：

$$
\Delta_f = R(\pi_f+f)-R(\mathrm{H3+priority@B2}+f).
$$

另做$\pi_C/\pi_D$ crossed C/D evaluation；不同beamformer間的absolute rate差不能歸因為controller gain。

## 9. Mandatory gates

### Gate 5.5 — Frozen-GNN compatibility

- Stage 1C checkpoint、AP coordinates、config與model source hashes一致。
- C/D outputs/rates在B2/B8與dynamic masks下finite；unassociated weights為0；per-AP power合法。
- 同一controller cell的RZF/C/D共用byte-identical association masks、updates、stored CSI與true CSI。
- B8 C/D在相同frames重現Stage 3 full-current dynamic-mask inference至predeclared tolerance。
- C-GNN不讀hidden current CSI；D-GNN不超出AP-local visibility；perturbation與saved-input hashes通過。

### Gate 5.6 — Matching learned policies

- Loss、Q、entropy、gradients、parameters、raw scores、projected actions與metrics finite。
- Reload後fixed validation traces的scores、masks、updates與metrics deterministic。
- Actor locality、no leakage、top-2、B2、mask與power gates通過。
- 三個policy均用matching reward訓練；不得以單一RZF policy替代。
- 是否勝random、是否勝matching H3+priority與C/D transfer分開報；completion不等於superiority。

### Gate 5.7 — Evidence audit

- Eligible methods在straight 0/30/80相同development-test traces作paired evaluation。
- 保存per-trajectory rate/tail/switching/age/NMSE/usage，而非只存aggregate JSON。
- Incomplete、excluded、superseded與negative runs保留但不覆寫。
- 更新claim–evidence map與Stage 6 handoff。

## 10. Execution order

1. 建立Stage 5 local frozen-GNN adapter與source hashes。
2. 加入stored-CSI/dynamic-mask/power/visibility unit tests。
3. 跑2 trajectories × 100 frames CPU smoke，涵蓋至少一次association epoch。
4. 完成C/D B8 reproduction及B2 no-RL baselines；Gate 5.5通過才啟動RL。
5. 依完全相同protocol跑$\pi_{\mathrm{RZF}}$、$\pi_D$、$\pi_C$ smoke與seed-0 pilots。
6. 對matching pairs與C/D crossed paths作paired evaluation，完成Gate 5.6–5.7。

基礎validation：

```bash
cd code/stage5
python -m py_compile *.py
bash -n run_exp-v5.sh
python test_stage5.py
```

Stage 5B CLI在實作時可採最小介面，但不得改action order、all-link bootstrap、B2、top-2、H3、paired traces或boundary tolerances。

## 11. Stop rules

- Provenance、true/stored CSI、causality、visibility、budget、top-2、power或parent boundary失敗：停止performance run，修contract並用fresh root重跑。
- Frozen-GNN adapter失敗：不啟動RL、不fine-tune GNN、不放寬visibility或tolerance。
- H3+priority低於controls或priority造成starvation：保留negative/trade-off result，不事後改channel、noise、B或tail metrics。
- Learned scores變化但projected actions不變：記錄projection plateau；不在同一pilot改action family。
- Policy勝random但不勝matching baseline仍是有效negative result。
- 只有centralized path改善時，decentralized primary claim判為未通過；不得用centralized結果替代。
- 結果不漂亮不是換seed、test traces或training ceiling的理由。

## 12. Expected artifacts

每個run保存effective config、split/RNG hierarchy、input/source/checkpoint hashes、completion status、log、raw association/update traces、stored/true CSI input hashes、per-trajectory metrics與constraint diagnostics。

Stage 5B另保存：

- C/D B2/B8 no-RL baselines；
- 每個reward beamformer的training metrics與validation history；
- best/latest actor/critic checkpoints；
- locality、no-leakage、reload與raw-score/action hashes；
- matching與crossed C/D evaluation。

主圖最多四張：within-beamformer RL gain、rate–feedback retention、rate–switching Pareto、tail-rate/age trade-off。

## 13. Stage 6/7 handoff

Stage 5B後先以$\pi_C/\pi_D$ crossed evaluation分離policy optimization與beamformer visibility。若需ablation，至多從移除age/history、移除previous association、centralized-information actor upper reference中選兩項；不全面搜尋architecture。

Stage 7使用至少5個paired seeds。每個RZF/C/D beamformer內保留matching RL、H3+priority@B2與H3+B8；fixed+RR@B2作simple control。Primary是$\Delta_D$；$\Delta_C$為upper reference，$\Delta_{\mathrm{RZF}}$為control。

Stage 5 seed-0結果不得支持formal superiority、robustness、scalability或equivalence claim。
