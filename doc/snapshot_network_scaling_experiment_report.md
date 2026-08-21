# Snapshot Centralized–Decentralized Network-Scaling 實驗報告

## Material Passport

- Origin Skill: research-paper-writing
- Origin Mode: results synthesis and mechanism analysis（living experiment report）
- Origin Date: 2026-08-21
- Last Updated: 2026-08-21 22:55 +08:00
- Verification Status: ANALYZED（直接讀取 ws2 的 configs、status、logs、final metrics 與 evaluation details；未獨立重新訓練，因此不標為 VERIFIED）
- Version Label: snapshot_network_scaling_report_v1_partial_scale4
- Plan: `doc/snapshot_network_scaling_experiment_plan.md`
- Artifact Root: `/tmp2/b12902052/snapshot_network_scaling/results_snapshot_scaling/` on `ws2`
- Source Revisions: Stage 0 `388713f`；Stage 1 `d7ce358`；所有已讀取 configs 均記錄 `source_dirty=false`

## 1. 報告範圍與目前結論

本報告分析 snapshot network-scaling 實驗截至 2026-08-21 22:55 的結果。Stage 0 RIS 與 Stage 1 no-RIS 的 scales 1/2 均已完成 seeds 0–2，每個 seed 使用 3200 個 final samples；Stage 1 scale 4 目前只有 seed 0 完成，seed 1 因 non-finite beamformer 失敗，seed 2 仍在執行；Stage 0 scale 4 seed 0 仍在執行。因此，scales 1/2 可作預定的 three-seed descriptive trend screen，scale 4 尚不能形成完整趨勢結論。

目前最重要的結果有三點。第一，Stage 0 的 mean relative C/D gap 從 scale 1 的 `2.408%` 降至 scale 2 的 `0.377%`；三個 scale-1 seed 的 absolute gaps 均大於三個 scale-2 seed，顯示 observed contraction 不是單一 seed 造成。第二，Stage 0 的 local/global visibility ratio 雖按假說從 `0.715` 降至 `0.665`，但每 AP 可見的 association links 從 `17.38` 增至 `58.04`，且任兩 AP 共享 UE 的機率從 `97.95%` 增至 `99.77%`；現有 `D-shared` 並未隨 scale 形成強烈資訊隔離。第三，Stage 1 的 visibility ratio 明顯下降，但 decentralized GNN 反而高於 centralized GNN，而且兩者都遠低於 MRT/RZF；此結果較符合 optimization/architecture failure signal，而不是 decentralized information advantage。

本報告不把 association、RIS aggregation 或 training instability 寫成已證實的因果機制。這些診斷由 artifact statistics 與 code-path audit 支持，但仍需要預先凍結的 ablations 才能區分各機制的獨立效果。

## 2. 實驗設定與執行狀態

### 2.1 共通 protocol

- Scales 1/2/4 分別使用 `(AP, UE)=(5,8),(10,16),(20,32)`。
- Stage 0 同時使用 `RIS=4,8,16`，每個 RIS 有 30 elements，AP antennas `M=2`。
- Stage 1 無 RIS，AP antennas `M=2`，`noise_power=1e-12`。
- Per-AP power 為 `15 dBm = 0.0316228 W`；association threshold 為最強 AP received power 的 `0.1`。
- 每個 run 使用 batch size 8、2000 training iterations、128 validation samples 與 3200 final samples。
- 同一 setting/seed 的 C/D 共用 model parameters、channel snapshots、association masks、noise 與 power constraints。
- 模型只以 centralized forward path 訓練；decentralized 是同一組參數的另一條 inference path，不是獨立最佳化的 policy。
- 本實驗是 exploratory trend screen；依 plan 不作 p-value 或 population-level significance 宣稱。

### 2.2 執行狀態

| Branch | Scale | Seed 0 | Seed 1 | Seed 2 | 可否納入主趨勢 |
|---|---:|---|---|---|---|
| Stage 0 RIS | 1 | Complete | Complete | Complete | Yes |
| Stage 0 RIS | 2 | Complete | Complete | Complete | Yes |
| Stage 0 RIS | 4 | Running，iteration 937/1999 | Gated | Gated | No |
| Stage 1 no-RIS | 1 | Complete | Complete | Complete | Yes |
| Stage 1 no-RIS | 2 | Complete | Complete | Complete | Yes |
| Stage 1 no-RIS | 4 | Complete | Failed：NaN by iteration 450；validation at 500 raised `Non-finite beamformer` | Running，iteration 880/1999 | No |

Stage 0 scale 4 seed 0 在 iteration 500 的 128-sample validation 為 centralized `6.5375`、decentralized `6.6396`。這是未完成 training 中的單一 validation checkpoint，不列入 final scaling curve。

## 3. Stage 0 RIS 結果

### 3.1 Primary C/D results

下表報告 learned continuous RIS phase 的 3200-sample final sum rate。Mean row 是三個 seeds 的算術平均；relative gap 先在每個 seed 計算 `(C-D)/C`，再跨 seeds 平均。

| Scale | Seed | Centralized | Decentralized | C − D | Relative gap |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 7.669895 | 7.493446 | 0.176449 | 2.301% |
| 1 | 1 | 7.604559 | 7.452078 | 0.152482 | 2.005% |
| 1 | 2 | 6.551058 | 6.359845 | 0.191213 | 2.919% |
| **1** | **Mean** | **7.275171** | **7.101790** | **0.173381** | **2.408%** |
| 2 | 0 | 7.668391 | 7.632094 | 0.036297 | 0.473% |
| 2 | 1 | 7.119707 | 7.058295 | 0.061412 | 0.863% |
| 2 | 2 | 7.936000 | 7.952179 | −0.016179 | −0.204% |
| **2** | **Mean** | **7.574699** | **7.547523** | **0.027177** | **0.377%** |

Stage 0 的 absolute C/D gap 從 `0.1734` 降至 `0.0272`，約縮小 `84.3%`。Scale 1 的 observed gap range 為 `[0.1525, 0.1912]`，scale 2 為 `[−0.0162, 0.0614]`，兩個 three-seed ranges 沒有重疊；因此在已執行 seeds 中，gap contraction 是一致的 descriptive finding。Scale 2 seed 2 出現 D 輕微高於 C，進一步顯示 learned C/D policies 沒有保證 `C>=D`。

Raw sum rate 從 scale 1 到 scale 2 大致維持不變，但 UE 數加倍，因此 mean per-UE rate 明顯下降：centralized 從 `0.9094` 降至 `0.4734`，decentralized 從 `0.8877` 降至 `0.4717`。這表示系統擴大並沒有帶來與 UE 數成比例的 aggregate throughput growth；單純觀察 raw sum rate 會掩蓋 per-UE degradation。

### 3.2 RIS-phase diagnostics

| Scale | C learned continuous | D learned continuous | C learned 2-bit | D learned 2-bit | C random continuous | D random continuous |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.275171 | 7.101790 | 6.781011 | 6.617940 | 3.490628 | 3.422712 |
| 2 | 7.574699 | 7.547523 | 6.931314 | 6.939699 | 3.387796 | 3.517651 |

Learned RIS phase 在兩個 scales 都明顯高於 random-phase controls，支持模型確實利用 RIS phase，而不是所有 phase outputs 等效。另一方面，scale 2 的 2-bit 與 random-phase variants 都可出現 D 高於 C，顯示 gap contraction 不只存在於 continuous-phase readout；但 random phases 在 C/D 間是獨立抽樣，因此只能作 expectation-level diagnostic，不能當作逐 snapshot 的 phase-controlled ablation。

### 3.3 Association 與 local visibility

下表由每個 scale 的三個 seeds、共 9600 個 final snapshots 直接重建。`Visible links/AP` 完全依照目前 D-shared rule 計算：若 AP 服務 UE，該 AP 可見所有同樣服務該 UE 的 AP–UE links。

| Metric | Scale 1 | Scale 2 | Scale 1→2 change |
|---|---:|---:|---:|
| AP–UE links/snapshot | 23.810 / 40 | 86.142 / 160 | 3.62× absolute |
| Association density | 0.5953 | 0.5384 | −9.6% relative |
| Serving APs/UE | 2.976 | 5.384 | 1.81× |
| UEs/AP | 4.762 | 8.614 | 1.81× |
| Single-AP UEs | 16.65% | 1.89% | −14.76 pp |
| All-AP UEs | 22.20% | 9.56% | −12.64 pp |
| Visible links/AP | 17.380 | 58.041 | 3.34× |
| Local/global visibility ratio | 0.7150 | 0.6654 | −6.94% relative |
| AP-pair shares at least one UE | 97.95% | 99.77% | +1.82 pp |
| Bipartite graph fully connected | 99.44% | 100.00% | +0.56 pp |

這組結果澄清了「資訊是否更流通」的兩種不同意義。以比例來看，單一 AP 可見的 global associated CSI 確實下降，符合事前假說；以絕對輸入量與 AP overlap 來看，D 的 local view 卻快速擴張，而且 association graph 幾乎完全連通。Stage 0 每個可見 AP–UE node 還包含所有 RIS-specific features；RIS 數從 4 加倍到 8 後，visible AP–UE–RIS feature blocks 約從 `17.38×4=69.5` 增至 `58.04×8=464.3`，約為 6.68 倍。現有 count ratio 因而不足以單獨代表 decentralized inference 的實際資訊量或通信負擔。

### 3.4 Geometry mechanism

Stage 0 將 AP 等角度放在半徑 `2×spatial_scale` 的圓周，UE 則位於半徑 `spatial_scale` 的圓內。雖然 config 中的 nominal AP density `A/(πR_UE^2)` 在 scales 1/2 相同，但 AP 只分布在一維圓周，local geometry 並未保持不變：

| Scale | AP count | AP-ring radius | Mean nearest AP spacing | Ring linear density |
|---:|---:|---:|---:|---:|
| 1 | 5 | 200.0 m | 235.1 m | 0.003979 AP/m |
| 2 | 10 | 282.8 m | 174.8 m | 0.005627 AP/m |

Network area 加倍時，相鄰 AP 的實際距離反而縮短 `25.7%`，ring linear density 增加 `41.4%`。這是 ring geometry 與二維 fixed-density interpretation 的結構性不相容：若 AP 只在圓周上，使用 `radius∝sqrt(A)` 不能同時維持固定 nearest-neighbor spacing。

UE sampling 也保留原 Stage 0 的 fidelity rule：radius 直接由 `Uniform(0,R)` 取樣，而不是 area-uniform 的 `R√U`。前者會讓較高比例 UE 靠近中心；中心 UE 到各圓周 AP 的距離較相近，再配合 `RSSI >= 0.1×strongest RSSI` 的相對 threshold，容易形成多 AP association。這不是 scaling refactor 新增的 bug，但會使 D-shared 在大型 ring topology 中保持高度合作。

### 3.5 D-shared 與 global RIS aggregation

目前 decentralized forward 並非 own-AP-only。對每個 AP，它先找出該 AP 所服務的 UEs，再加入其他 AP 對同一批 co-served UEs 的 CSI。Scale 2 的 serving degree 與 AP-pair overlap 上升，因此這條 additional-CSI path 在絕對量上變得更強。

此外，Stage 0 的 decentralized RIS phase 不是各 AP 獨立使用：所有有服務 UE 的 AP 都輸出 RIS proposal，之後由共享的 `RIS_merge` 以 sum aggregation 合成單一 global RIS phase。AP 數從 5 增至 10 時，merge 的 proposal 數也增加；其後雖有 phase normalization，sum 的方向、bias 相對權重與跨 AP 誤差抵消仍可能隨 AP 數改變。此架構保留了原 Stage 0 semantics，但表示 `D-shared` 具有全域 RIS aggregation，不能解讀為零通信的 fully decentralized baseline。

綜合 geometry、association 與 model path，Stage 0 gap contraction 的合理解釋是：D 缺少的 CSI 比例略增，但 co-served-UE sharing、絕對可見資訊量與 global RIS aggregation 讓關鍵資訊仍高度流通；centralized 額外取得的遠端 links 可能只有有限 marginal value。這是由現有證據支持的 mechanism hypothesis，不是因果證明。

### 3.6 Stage 0 scalability cost

| Metric | Scale 1 | Scale 2 | Growth |
|---|---:|---:|---:|
| Parameters | 2,206,479 | 2,773,944 | 1.26× |
| Mean training time/seed | 2657 s | 13,823 s | 5.20× |
| Centralized inference | 0.0299 s/sample | 0.0947 s/sample | 3.16× |
| Decentralized inference | 0.1430 s/sample | 0.9337 s/sample | 6.53× |
| D/C inference-time ratio | 4.78× | 9.86× | 2.06× |

目前 decentralized implementation 對每個 AP 重新建立 concatenated features 並單獨執行 encoder，因此其 wall-clock cost 比 centralized 更快惡化。這些 CPU timings 不等同 optimized deployment latency，但已構成明確的 implementation scalability limitation。

## 4. Stage 1 No-RIS 結果

### 4.1 Primary results and conventional baselines

| Scale | Seed | Centralized GNN | Decentralized GNN | C − D | Relative gap | MRT | RZF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 5.717460 | 5.755009 | −0.037549 | −0.657% | 26.5445 | 21.9913 |
| 1 | 1 | 10.523437 | 11.146467 | −0.623031 | −5.920% | 27.0890 | 26.6419 |
| 1 | 2 | 20.821745 | 21.409496 | −0.587751 | −2.823% | 28.3682 | 26.3719 |
| **1** | **Mean** | **12.354214** | **12.770324** | **−0.416111** | **−3.133%** | **27.3339** | **25.0017** |
| 2 | 0 | 9.997650 | 11.153522 | −1.155871 | −11.561% | 49.0458 | 44.1705 |
| 2 | 1 | 7.834346 | 8.925817 | −1.091472 | −13.932% | 48.0979 | 39.8119 |
| 2 | 2 | 12.096630 | 12.493850 | −0.397220 | −3.284% | 50.2335 | 44.3935 |
| **2** | **Mean** | **9.976209** | **10.857730** | **−0.881521** | **−9.592%** | **49.1260** | **42.7920** |
| 4 | 0 only | 21.204729 | 21.969229 | −0.764500 | −3.605% | 94.8042 | 80.6692 |

Stage 1 scales 1/2 的每個 seed 都出現 D 高於 C，但這不能被解讀為 decentralized inference 的資訊優勢。首先，C/D 是同一個 learned model 的不同 forward paths，且只以 centralized path 訓練；更多 input information 不保證固定的非最適 policy 產生更好的 output。其次，GNN 明顯低於 MRT/RZF：scale 1 mean centralized GNN 只有 MRT 的 `45.2%`，scale 2 只有 `20.3%`，scale 4 seed 0 只有 `22.4%`。此時 C/D ordering 主要反映不穩定 learned policies 的相對誤差，而不是接近 optimum 時的 information gap。

跨 seed 變異也很大。Scale 1 centralized GNN 從 `5.72` 到 `20.82`，約 3.64 倍；同一批 seeds 的 MRT 只從 `26.54` 到 `28.37`。Scale 2 centralized GNN 從 `7.83` 到 `12.10`，而 MRT 維持在 `48.10–50.23`。這表示 channel/topology 本身並未產生同等幅度的不穩定，主要 variability 來自 GNN training outcome。

Scale 4 目前不能形成趨勢。Seed 0 完成但只是一個 point；seed 1 在 iteration 450 前已持續輸出 NaN，iteration-500 validation 由 finite check 終止；seed 2 尚未完成。依 plan 的 numerical gate，seed-1 failure 必須如實保留，不能從 summary 排除後只報成功 seeds。

### 4.2 Stage 1 association and visibility

| Metric | Scale 1 | Scale 2 | Scale 4 seed 0 |
|---|---:|---:|---:|
| Samples | 9600 | 9600 | 3200 |
| AP–UE links/snapshot | 13.290 | 29.298 | 61.536 |
| Association density | 0.3323 | 0.1831 | 0.09615 |
| Serving APs/UE | 1.661 | 1.831 | 1.923 |
| UEs/AP | 2.658 | 2.930 | 3.077 |
| Single-AP UEs | 58.01% | 53.63% | 50.62% |
| Visible links/AP | 5.805 | 7.565 | 8.490 |
| Local/global visibility ratio | 0.4173 | 0.2501 | 0.1358 |
| AP-pair shares at least one UE | 54.90% | 37.29% | 20.45% |

Stage 1 的 BPP square + wrap-around topology 確實產生預期的 localization：每 AP load 只小幅上升，local/global ratio 約依 scale 持續下降，AP-pair overlap 也降低。因此 Stage 1 的 D>C 不能由「資訊更流通」解釋；association diagnostics 與 performance ordering 反而朝相反方向。

較合理的 architecture hypothesis 是 global max aggregation 與 input-size shift。Centralized encoder 在 scale 1/2/4 分別處理 40、160、640 個 AP–UE slots，固定-depth message passing 的 max statistics 會隨 node count 改變；decentralized path 先將不可見 features 歸零，可能產生類似 relevance filtering 或 regularization 的效果。不過零 feature 通過帶 bias 的 linear layers 後不必保持零，因此這仍不是嚴格的 masked-message-passing ablation。需要專門 instrumentation 或重新設計 mask-aware aggregation 才能確認此機制。

### 4.3 Stage 1 scalability and failure signal

| Metric | Scale 1 | Scale 2 | Scale 4 seed 0 |
|---|---:|---:|---:|
| Parameters | 734,025 | 1,062,990 | 1,720,920 |
| Mean training time | 1451 s | 6829 s | 38,953 s |
| Centralized inference | 0.0157 s/sample | 0.0658 s/sample | 0.4343 s/sample |
| Decentralized inference | 0.0714 s/sample | 0.6001 s/sample | 8.0453 s/sample |
| D/C inference-time ratio | 4.55× | 9.12× | 18.52× |

Stage 1 decentralized inference 從 scale 1 到 scale 4 增加約 113 倍，而 centralized 約增加 27.7 倍。這不是 decentralized algorithm 必然的理論複雜度，而是目前 sequential per-AP implementation 的實際成本。Scale-4 seed-1 NaN 再顯示固定 optimizer、depth 與 aggregation 未能穩健外推到大型 graph；在處理 numerical stability 前，不應用 scale-4 GNN 數值建立方法優劣結論。

## 5. Implementation audit

### 5.1 已通過的 contracts

- ws2 的 Stage 0 `data.py`、`model_2.py`、`evaluate.py`、`trainer_2.py` SHA-256 checksums 與本地 snapshot-scaling source 完全一致。
- Stage 0 scales 1/2 的所有 completed runs 均通過 paired channel/mask、association-mask beamformer、per-AP power、finite output 與 all-AP–all-RIS runtime checks。
- Evaluator 在 centralized 後以 `regenerate_channels=False` 重新格式化同一批資料，並比較 snapshot identifier 與 association mask，未發現 C/D 使用不同 samples。
- 對原 `code/stage0/` 與 parameterized copy 作 whitespace-insensitive diff 後，D-shared 與 RIS merge semantics 沒有被修改。主要 model change 是把 RIS AP head 的 hard-coded `Linear(32, ...)` 改為 `Linear(L×K, ...)`；scale 1 的 `L×K=4×8=32`，所以 baseline input dimension 等價。
- Scaling source 未找到會在 functional path 中固定 AP=5、UE=8 或 total nodes=40 的殘留 hard coding；相關數字只存在 defaults、comments 或 standalone `__main__` examples。

### 5.2 發現的 validation 缺口

`code/snapshot_network_scaling/stage0/trainer_2.py` 在 successful final evaluation 後直接將 `checks.json` 的 `topology_density` 寫成 `true`，沒有計算 nearest-neighbor spacing、ring linear density 或 normalized local geometry。因此，現有 checks 只能證明 config 中 nominal area density 的欄位一致，不能證明 topology 真正保持 fixed local density。這是 reporting/validation bug，不會直接改變已完成 rates，但它未能攔截本報告第 3.4 節的 geometry confound。

Evaluation details 的 Stage 0 `user_locations` 以每個 batch 的 `K×2` locations flatten 儲存，而 association mask 以 `batch_size×K×A` 儲存；因同一 batch 的 8 個 channel samples 共用一組 UE positions，統計 mean distance 仍有效，但 artifact shape 不是直接的 `(samples,K,2)`。任何 snapshot-level topology visualization 都必須先依 batch index 還原位置，不能把兩個 arrays 的第一維直接對齊。

### 5.3 Audit conclusion

目前沒有證據顯示 C/D gap contraction 是由 remote source mismatch、unpaired evaluation、association-mask violation、power violation或 scaling refactor indexing error 造成。較主要的問題是 operational-definition mismatch：`D-shared` 包含 co-served-UE CSI，Stage 0 又包含 global RIS aggregation，而 ring scaling 沒有維持固定 local geometry。Stage 1 則另有明顯的 training instability 與 scale-4 numerical failure。

## 6. 對事前假說的判讀

| Pre-registered question/hypothesis | Evidence | Current decision |
|---|---|---|
| Scale 增加時 local/global visibility ratio 下降 | Stage 0 `0.715→0.665`；Stage 1 `0.417→0.250`，scale-4 seed 0 `0.136` | Supported descriptively |
| 若非本地 coordination 有價值，C/D gap 可能增加 | Stage 0 gap 縮小；Stage 1 D>C 但 GNN failure signals 強 | Not supported by current performance results |
| Gap 也可能飽和、平坦或非單調 | Stage 0 scales 1/2 明顯 contraction；scale 4 pending | Consistent with allowed outcome, but full curve incomplete |
| AP/UE density 與 local environment 儘量固定 | Nominal area density fixed；Stage 0 ring spacing 未固定 | Partially violated for Stage 0 geometry interpretation |
| Stage 0 與 Stage 1 可用相同機制解釋 | Association trends、baseline quality與 numerical stability 不同 | Not supported；兩條 curves 必須分開解讀 |

「gap 必須隨 scale 增加」原本就不是 experiment success criterion。已完成結果否定的是較強的直覺性主張，而不是造成 protocol failure；真正需要修正的是對 Stage 0 fixed-density 與 decentralized semantics 的表述。

## 7. 決策與後續實驗邊界

1. 保留並完成目前已啟動 runs；不得因 gap 方向不符預期而修改 threshold、noise、model capacity 或 seed selection。
2. Stage 0 scale 4 需等待 seed 0 final gate；若通過再依既定流程執行 seeds 1/2。其 intermediate validation 不納入主表。
3. Stage 1 scale-4 seed-1 failure 必須保留。Seed 2 完成後，先診斷 non-finite onset、parameter/gradient trajectory 與 seed-specific topology，再決定是否需要獨立 numerical-stability plan；不能默默重跑到成功為止。
4. 將 `topology_density` check 改成實際計算，而不是 hard-coded boolean。至少保存 area density、nearest-AP spacing、ring linear density與 normalized spacing；這是 future-run validation fix，不回溯改寫 completed artifacts。
5. 若要測試資訊限制的因果效果，另立預先凍結的 follow-up：比較 `D-own-only`、目前 `D-shared` 與 centralized。Stage 0 還需把 local RIS decisions 與 global `RIS_merge` 分開 ablate。
6. 若研究問題要求 fixed local density，Stage 0 應改用 APs 分布於二維 expanding area 的 topology；保留 ring family 時，結論應命名為 ring-topology scaling，而不是一般的 fixed-density network scaling。
7. 若要比較 information sets 的最佳可達表現，應分別訓練 centralized 與 decentralized policies，或使用明確的 joint objective；目前 centralized-only training + alternate decentralized forward 不能提供 information-theoretic ordering。

## 8. Claim–Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| Stage 0 C/D gap 從 scale 1 到 scale 2 縮小 | 每 scale 3 seeds、每 seed 3200 final samples；gap ranges 不重疊 | Supported for observed seeds |
| Stage 0 decentralized 的可見比例下降但絕對資訊量增加 | 9600 snapshots/scale 的 association masks；ratio `0.715→0.665`、visible links `17.38→58.04` | Supported |
| Stage 0 association graph 在 scale 2 高度重疊 | AP-pair overlap `99.77%`、graph connected `100%` | Supported |
| Association overlap 導致 gap contraction | Mechanism statistics 與 code path 一致，但沒有 intervention | Needs ablation |
| Global RIS merge 可能使 C/D 接近 | Decentralized code 確實 sum all AP proposals；未做 local-vs-global merge ablation | Needs ablation |
| Stage 1 D 優於 C | Scales 1/2 所有 observed seeds 的 learned rates皆為 D>C | Supported as a raw model result only |
| Stage 1 證明 decentralized information 優勢 | GNN 遠低於 MRT/RZF、跨 seed 不穩定、scale-4 seed 1 NaN | Unsupported |
| Scale 4 延續 scales 1/2 trend | Stage 0 未完成；Stage 1 只有一個成功 seed | Needs evidence |
| Scaling refactor 引入 C/D pairing 或 indexing bug | Checksum、diff、runtime contracts 與 artifact checks 未發現 | No supporting evidence |

## 9. Adversarial Self-Review

| Dimension | Review question | Assessment | Required action |
|---|---|---|---|
| Contribution | 報告是否提供超出 raw table 的新知？ | Pass：區分 visibility proportion、absolute local information、association overlap 與 global RIS aggregation | 保留 mechanism/causality 邊界 |
| Writing clarity | 設定、D-shared semantics 與目前狀態是否可重現？ | Pass for scales 1/2；scale 4 是 time-stamped partial state | Scale 4 完成或失敗後更新 status |
| Experimental strength | 是否有穩健且 competitive 的 learned baselines？ | Stage 0 descriptive trend 尚可；Stage 1 fail，GNN 明顯低於 MRT/RZF | 不以 Stage 1 C/D sign 作 method claim |
| Evaluation completeness | 是否足以確認 gap contraction 的原因？ | Needs new experiment：缺 D-own-only、RIS-merge與 topology ablations | 另立 follow-up plan，不修改本輪 |
| Method design soundness | Fixed-density 與 decentralized assumptions 是否成立？ | Needs revision：Stage 0 ring geometry 與 global RIS merge 限制外部效度 | 修正命名、computed topology checks 與 baseline definitions |

目前最高風險的 reviewer objection 是：實驗把 nominal area density 當成 fixed local density，並把具有 co-served CSI sharing 與 global RIS aggregation 的 policy 稱為 decentralized，因而不能把 gap trend直接歸因於 network size。本文已縮限 claim，將 Stage 0 結果定位為特定 ring topology 與現有 D-shared architecture 下的 exploratory finding；完成預定 scale 4 與後續 ablations 前，不升級為一般化結論。
