# Decentralized RIS action representation 實驗計畫

## Material Passport

- Date: 2026-09-15
- Last Updated: 2026-09-15（方法定義集中至獨立文件）
- Verification Status: EXECUTED（結果見 action screening 報告）
- Version Label: `decentralized_ris_action_plan_compact_v4`
- Anchor paper: [Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks](<./Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks.pdf>)
- Current evidence: [Decentralized RIS 現況與證據報告](./decentralized_ris_evidence.md)
- Methods: [Decentralized RIS 方法與通訊介面](./decentralized_ris_methods.md)
- Results: [RIS action screening 報告](./ris_action_screening_report.md)

本文件保留 RIS action representation 主線的小型、單 seed 執行規格。方法公式與通訊介面由
[方法文件](./decentralized_ris_methods.md)統一維護；本文件不重複其定義。

## 1. 目前只回答的問題

> 保留相同 RIS GNN、相同 learned reduction head 與相同參數量時，把 unit-modulus
> projection 從 aggregation 後移到每個 AP 輸出處，對 decentralized sum rate 有何影響？

這個問題拆成兩個 control：

1. **R0 → R0-commuted**：只移動線性運算的位置，確認代數與程式等價；不預期 rate 改變。
2. **R0-commuted → R1-shared**：只把 unit-modulus projection 提前，測試 feasible AP action
   interface 本身的代價或收益。

目前不先測 node removal、RIS identity 或 confidence weighting。這些都會同時引入其他變因，
等本小實驗確認方向後再進行。

## 2. Controlled comparison arms

精確 forward equations、projection 順序、payload 與 decentralization 用語見
[方法文件](./decentralized_ris_methods.md)。本計畫只記錄各 arm 在實驗中的角色：

| 文件名稱 | CLI `--arch` | 本輪用途 |
|---|---|---|
| R0 | `r0` | baseline |
| R0-commuted | `r0c` | R0 代數等價性 control |
| R1-Shared | `r1_shared` | projection-placement intervention |
| R1-Local | `r1` | AP-specific head capacity control，暫緩 |
| R3a | `r3a` | RIS-node removal，暫緩 |
| R3b | `r3b` | R3a 加 per-RIS context，暫緩 |

R1-Local 同時改變 head sharing、capacity 與 projection placement，不能取代 R1-Shared 作為單變因
control，因此本輪不先執行。

## 3. 固定實驗條件

| 項目 | 設定 |
|---|---|
| AP／RIS／UE | $A=5$、$R=4$、每批 topology 有 $K=8$ users |
| Antennas／RIS elements | $M=2$、$N=30$ |
| Power | $P_{\max}=15$ dBm |
| Backbone | depth 6、channel 64 |
| Optimizer | Adam $10^{-4}$、weight decay $10^{-6}$、batch 8 |
| Training | Global CSI、global sum-rate loss |
| Primary inference | 原論文 Eq. (10) local CSI |
| Primary metric | Decentralized continuous-phase sum rate |
| Secondary metric | Decentralized 2-bit sum rate |
| Seed | 本階段只用 0 |
| Validation | 400 samples，seed `20260913` |
| Screening holdout | 400 samples，seed `20260915` |
| Final test | 3,200 samples，seed `20260914`；本輪不碰 |

50k baseline 可繼續跑，但不能直接與 10k R1-shared 形成 performance claim。小實驗先以 matched
10k runs 判斷趨勢；只有方向成立後，才讓 R0 與 finalist 跑相同完整 budget。

## 4. 小實驗執行步驟

以下命令均從 `code/decentralized_ris/` 執行。先啟用既有環境：

```bash
conda activate decentralized-inference
cd ~/ThomasLai/code/decentralized_ris
```

### Step 1：同步程式並跑 deterministic tests

從本機 repository root 同步；只同步 source/test，不覆蓋 remote results：

```bash
rsync -av code/decentralized_ris/{model.py,variants.py,train.py,evaluate.py} \
  lab301-5090-tailscale:~/ThomasLai/code/decentralized_ris/
rsync -av code/decentralized_ris/tests/test_action_interface.py \
  lab301-5090-tailscale:~/ThomasLai/code/decentralized_ris/tests/
```

在 remote 執行：

```bash
python -m py_compile *.py experiments/*.py tests/*.py
python -m tests.test_forward_equivalence
python -m tests.test_action_interface
python -m tests.test_phase_quantization
python -m tests.test_rate_equivalence
```

`test_action_interface.py` 必須同時證明：

1. R0 與 R0c 在包含 bias split 時數值相等。
2. R0、R0c、R1-shared 可 strict-load 同一份 state dict，且 R0/R1-shared 有相同有效參數量。
3. 有具體例子顯示 projection 前後移不可交換。

### Step 2：用已完成的 R0 checkpoint 做零訓練 control

若已有完成且含 `summary.json` 的 R0 run，以同一 checkpoint 與同一 screening seed 各評估一次。
`evaluate.py --arch_override` 只允許 state-compatible 的 R0c/R1-shared 路徑：

```bash
R0_RUN=<completed_r0_run_directory>

python evaluate.py --runs "$R0_RUN" --checkpoint best.pt --samples 400 \
  --eval_seed 20260915 --out ../../artifacts/decentralized_ris/evaluation/r0_screen.json
python evaluate.py --runs "$R0_RUN" --checkpoint best.pt --samples 400 \
  --eval_seed 20260915 --arch_override r0c \
  --out ../../artifacts/decentralized_ris/evaluation/r0c_screen.json
python evaluate.py --runs "$R0_RUN" --checkpoint best.pt --samples 400 \
  --eval_seed 20260915 --arch_override r1_shared \
  --out ../../artifacts/decentralized_ris/evaluation/r1_shared_no_retrain_screen.json
```

判讀：R0 與 R0c 應只差浮點誤差；否則停止並修正程式。R1-shared-no-retrain 只量測把既有 policy
直接換介面的瞬時影響，不能代表 R1-shared 重訓後的能力。若正在跑的 50k 尚未產生
`summary.json`，先跳過此步，不要中止該 run；Step 1 已涵蓋代數等價性。

### Step 3：500-iteration smoke

先以 `nvidia-smi` 選不會干擾現有 50k job 的 GPU，以下以 `cuda:1` 為例：

```bash
python train.py --arch r1_shared --consensus equal --smoke --device cuda:1 \
  --tag action-r1-shared-smoke \
  --out_dir ../../artifacts/decentralized_ris/action_smoke
```

通過條件：centralized/decentralized forward 正常、loss/gradient finite、unit-modulus error
$<10^{-6}$。Smoke 只驗工程正確性，不比較 500-step rate 高低。

### Step 4：matched 10k screening（同一 seed，各版本一次）

先檢查是否已有相同 code path、seed 0、相同 validation 設定的 completed R0 10k run；若有就直接
重用，不必再訓練 R0。正在執行中的 50k run 不能只憑當前 iteration 當成 frozen 10k checkpoint。
沒有可重用的 completed R0 10k 時，才啟動下列 R0 session。

先檢查 tmux 名稱，避免 duplicate run：

```bash
tmux has-session -t ris_action_r0_10k
tmux has-session -t ris_action_r1s_10k
```

若 session 不存在，依可用 GPU 啟動；同一張 GPU 不足時可依序執行。R0 命令是沒有可重用
10k baseline 時才需要：

```bash
tmux new-session -d -s ris_action_r0_10k \
  '~/miniforge3/bin/conda run -n decentralized-inference --no-capture-output python train.py --arch r0 --consensus wreduce --n_iter 10000 --seed 0 --device cuda:1 --tag action-r0-10k --out_dir ../../artifacts/decentralized_ris/action_10k 2>&1 | tee action-r0-10k.log'

tmux new-session -d -s ris_action_r1s_10k \
  '~/miniforge3/bin/conda run -n decentralized-inference --no-capture-output python train.py --arch r1_shared --consensus equal --n_iter 10000 --seed 0 --device cuda:2 --tag action-r1-shared-10k --out_dir ../../artifacts/decentralized_ris/action_10k 2>&1 | tee action-r1-shared-10k.log'
```

監看：

```bash
tail -f action-r0-10k.log
tail -f action-r1-shared-10k.log
```

兩個 run 都使用 seed 0、相同 validation samples、每 1,000 iterations 評估一次，且
`--test_sample_final` 維持 0。

### Step 5：paired screening holdout

完成後以同一 400 個新 samples 評估兩個 `best.pt`：

```bash
python evaluate.py \
  --runs ../../artifacts/decentralized_ris/action_10k/action-r0-10k_iter10000_seed0 \
         ../../artifacts/decentralized_ris/action_10k/action-r1-shared-10k_iter10000_seed0 \
  --checkpoint best.pt --samples 400 --eval_seed 20260915 \
  --out ../../artifacts/decentralized_ris/evaluation/action_r0_vs_r1s_10k.json
```

保留輸出的 JSON 與 `_paired.npz`，先看：

1. `decentralized` 的 paired mean difference 與 learning-curve trend。
2. `decentralized_discrete` 是否同方向。
3. `centralized` 是否也下降，以區分 action interface 問題與 local-CSI gap。
4. `model.effective_parameters` 是否相同、R1-shared 的 `cpu_trainable_parameters` 是否為 0。

## 5. Screening 決策規則

- **R0 vs R0c 不等價**：停止後續 run，先修 bias split、mask 或 tensor order。
- **R1-shared 在 10k 內與 R0 差距 ≤5%，且後段仍改善**：升到 matched 40k。
- **R1-shared 落後 >5% 且後段已平坦**：不直接跑 50k；先檢查 proposal concentration 與
  magnitude loss，再決定是否測 confidence/gating。
- **R1-shared 勝出**：才將 R1-local 加入，判斷 AP-specific capacity 是否有額外幫助。
- **任何版本只有 10k 結果**：只能寫「screening trend」，不能寫最終優越性。
- 最終只有在方法與 budget 凍結後，才跑共同完整 budget 與 3,200-sample final test。

## 6. 後續實驗（本輪暫緩）

若 R1-shared 通過 screening，再依序做：

| 比較 | 能支持的解讀 |
|---|---|
| R1-shared vs R1-local | shared head 與 AP-specific capacity 的作用 |
| R1-shared equal vs learned confidence | magnitude 被移除後，顯式 reliability weighting 是否補回資訊 |
| R1-shared vs R3a/R3b | RIS node removal 的作用 |
| Best-none vs RIS identity | RIS descriptor 的作用 |

Confidence variant 需另量測 forced-equal counterfactual、weight entropy、effective AP count、proposal
circular concentration，以及 confidence 與 leave-one-AP-out marginal contribution 的 rank correlation。
若只提高 rate、但不對應 reliability proxy，名稱改為 learned gating。

## 7. 文獻定位

- [Distributed user assignment GNN](./Distributed_Unsupervised_Learning_for_Combinatorial_User_Assignment_in_mmWave_Cell-Free_Massive_MIMO_Using_Graph_Neural_Networks.pdf)：支持 agent-aligned graph 與 decentralized local processing 的設計方向。
- [Deep Sets](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) 與 [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)：支持 shared mapping 加 permutation-aware aggregation。
- [Decentralized beamforming for IRS-enhanced cell-free networks](https://doi.org/10.1109/LWC.2020.3045884)：提供 decentralized active/passive beamforming coordination 的先例。

這些文獻支持「shared local policy + permutation-aware aggregation」的設計動機，但不會直接證明
local unit projection 一定提升 sum rate。R0c/R1-shared ablation 才是本研究需要補上的因果證據，
因此目前不宣稱 novelty 或優越性。
