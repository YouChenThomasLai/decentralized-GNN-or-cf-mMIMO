# Decentralized Active-CSI 實驗報告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate（compacted evidence ledger）
- Origin Date: 2026-08-18
- Last Updated: 2026-08-28
- Verification Status: ANALYZED（Stage 0–5A 已完成analysis；Stage 5B尚未實作或執行；目前沒有新版pipeline的獨立完整rerun或formal multi-seed inference）
- Version Label: decentralized_active_csi_report_v18_compacted
- Plan: `doc/decentralized_active_csi_experiment_plan.md`
- Active Detail: `doc/stage5_joint_decentralized_active_csi_experiment_plan.md`

## 1. 報告範圍

本文件只保存會影響後續研究決策的最小 evidence：frozen contract、gate verdict、handoff、主要trade-off與claim boundary。已完成stage的逐iteration、逐checkpoint、逐trajectory、重複sweep表與orchestration敘事已移除；需要重算或稽核時應讀raw artifacts，而不是從本文件還原數據。

目前唯一active method-development工作是Stage 5B：先建立frozen centralized/decentralized GNN的B2/B8 baselines，再訓練RZF/C/D三個beamformer-matched AP-local policies。

## 2. 狀態總表

| Stage | Verdict | 保留的最小結論 | Claim boundary |
|---|---|---|---|
| 0 | Complete | RIS-GNN pipeline在meaningful-rate regime可學習 | 單一seed positive control |
| 1A | Complete | 忠實移除RIS並沿用舊noise後落入noise floor | 不能比較C/D能力 |
| 1B | PASS | `noise_power=1e-12`恢復finite、可學習baseline | 5 seeds仍不足以宣稱C/D方向或等效 |
| 1C | PASS | BPP geometry、channel與5-seed snapshot baseline通過 | Development checkpoint，不是formal method evidence |
| 2-BPP | PASS | Square-torus mobility/channel與full-current fixed-association anchor可用 | Seed-0 environment qualification |
| 3 | PASS / learned negative | H3提供association signal；SAC不勝strong heuristics | 不再擴張Stage 3 learned association |
| 4 | PASS | Stored-CSI、hard budget與B0/B8 boundaries正確；priority/RR trade-off成立 | Seed-0 scheduler handoff，不是superiority claim |
| 5A | PASS | Joint causal evaluator與RZF modular matrix可用 | Seed-0 descriptive result |
| 5B | NOT RUN | Frozen-GNN adapter與matching RL尚不存在 | 不得宣稱RL+GNN improvement |

## 3. Stage 0–1C baseline evidence

### 3.1 Stage 0–1B

Stage 0保留原RIS model的$(W,\Theta,c)$學習與paired centralized/decentralized evaluation，作為positive control。Stage 1A移除RIS後只學$(W,c)$；沿用`noise_power=4e-4`時rate與task gradient落入noise-floor regime，因此該stage只作negative control。

Stage 1B將no-RIS主設定校準為`noise_power=1e-12`。Seeds 0–4各2000 iterations的training curves、parameters、checkpoints與final metrics均finite，沒有parameter collapse或weight-decay domination。Mean C–D gap很小且只有4/5 seeds同向；exact paired sign-flip $p=0.125$。因此只通過numerical/learning gate，不宣稱C/D優勢或等效。

### 3.2 Stage 1C BPP requalification

- TQ0：200 AP topology seeds × 每seed 100 UE drops；wrap、reproducibility、finite與10 m distance floor全部通過。
- TQ1：primary $K=8$ MRT/RZF smoke finite且符合mask/power contract；存在dynamic-association opportunities。
- TQ2：5 effective seeds × 2000 iterations；config、checkpoint、final evaluation與training arrays完整且finite。

Five-seed final means為centralized GNN 27.4597、decentralized GNN 26.7810、MRT 21.6380、RZF 19.8096。C−D mean 0.6787（centralized mean的2.472%），5/5為正，但雙尾exact sign-flip $p=0.0625$；last-500 training slopes仍為正。因此只記為development sign-consistency與可用checkpoint，不宣稱穩健C優勢或完全收斂。

Frozen Stage 1C seed-0 hashes：

| Artifact | SHA-256 |
|---|---|
| Checkpoint | `16dba87572cd9f9dfc3272b4faeda2d7127dc414945450b856758efdba7bba45` |
| AP coordinates | `421b817e0e1e70b88db4e2ef685954b35f2e65432ffcb056190adb881df8b92d` |
| Config | `ace7fbe95ab86dab070de0a04b7517edd1924af1271446fc69d94e72cd98cf0b` |

## 4. Stage 2 environment qualification

Legacy ring/disk mobility pilots、matched-training anomaly與validation-fix chronology已退出active evidence；不再重跑，也不再逐run記錄。

Stage 2-BPP使用Stage 1C seed-0 layout/checkpoint，完成straight 0/30/80、hotspot 3/30/80與兩組environment diagnostics。Runner exit status為0；8/8 settings complete，43個NPZ artifacts finite。Kinematics、square-torus wrap、3D distance、channel lag correlation、fixed association、full-current CSI、mask/power與provenance gates均通過。

此結果只證明Stage 5可重用的environment與full-information anchor可用；方法排名、adaptation、multi-seed mobility inference與robustness不屬Stage 2 evidence。

## 5. Stage 3 association qualification

Stage 3在full-current-CSI RZF下完成top-2 fixed/current/H3/H6與diagnostic heuristics。Stage 2 reproduction error、cardinality、causality、finite-rate、power與switching-semantics gates通過。Moving trajectories存在nonzero association signal；H3在主要development setting提供較好的rate–switching折衷，因此凍結為Stage 5 modular baseline。

SAC current/history的smoke、seed-0 training與paired evaluation均完成，但兩個policy在0/30/80 km/h development traces上都低於fixed/current/H3/H6 strong heuristics，且switching更高。這是negative development result，不是對所有learned association方法的一般否定。

依新版scope：

- 不保留SAC逐checkpoint、逐speed、逐trajectory數字於active docs；
- 不補Stage 3 policy seeds、switching penalty或architecture search；
- 不把SAC帶入Stage 5B或Stage 7 mandatory matrix；
- raw artifacts與completion/provenance仍保留供稽核。

## 6. Stage 4 feedback/CSI-aging qualification

Stage 4固定$t=0$ association，只改feedback scheduling與stored CSI。Straight 0/30/80 km/h、B0/B1/B2/B3/B8與三種causal scheduler共33 cells全部complete；hard budget、age transition、no leakage、B0/B8與Stage 2 reproduction gates通過。

最小B2 handoff evidence：

| Speed | RR retention | Priority retention | Priority vs RR mean-rate gain | UE-tail/freshness verdict |
|---:|---:|---:|---:|---|
| 30 km/h | 83.271% | 95.549% | 15.336% | Priority p05較低，部分弱links長期stale |
| 80 km/h | 71.865% | 84.194% | 16.101% | Priority p05較低，部分弱links長期stale |

依事前B2 aggregate-rate rule，`mobility_age_priority`交接Stage 5；`round_robin`保留為fair-refresh control。這是development selection，不是全面superiority：priority改善energy-weighted CSI quality與mean rate，但犧牲tail/freshness。完整budget sweep與20條paired raw values不再複製到文件。

## 7. Stage 5A joint modular evidence

### 7.1 Mechanical與boundary verdict

`code/stage5/`以Stage 4 stored-CSI state machine為base，整合Stage 3 H3。因果順序為association → feedback → reveal selected CSI → stored-CSI RZF → true-CSI rate。Gates 5.0–5.4通過：

- top-2、B2、mask、power、age與no-leakage constraints無違規；
- Stage 3 H3+B8與Stage 4 fixed-association boundaries在parent-native devices通過；
- 0 km/h H3零switch且B0/B8相等；
- 30/80 km/h兩種actions皆有nonzero signal；
- successful rerun2為8/8 completion、30 NPZ/1092 arrays finite，local/remote 131/131 SHA-256一致。

前兩次boundary attempts依stop rule終止，原因是CPU/CUDA parent數值路徑差異；failed roots保留。Successful rerun2只調整為parent-native device routing，沒有改seed、trace、method、budget或tolerance。

### 7.2 RZF modular matrix

以下為每speed 10條paired trajectories的trajectory-average sum rate；括號為相對H3+B8 full-current anchor的retention。

| Speed | Fixed + RR | Fixed + priority | H3 + RR | H3 + priority |
|---:|---:|---:|---:|---:|
| 30 km/h | 13.5469（76.459%） | 16.9795（95.657%） | 13.7448（77.738%） | 17.2072（97.124%） |
| 80 km/h | 11.0062（60.225%） | 13.5059（74.893%） | 12.6449（69.245%） | 15.7621（87.387%） |

H3+priority是moving 2×2 cells中的最高mean-rate組合，但priority使H3的UE p05相對RR下降16.715%/21.005%，never-refreshed active-link fraction為20.576%/34.484%。因此Stage 5A只支持「joint evaluator有可用action signal」，不支持B2 lossless、fairness改善或正式method superiority。

### 7.3 Stage 5A claim boundary

| Claim | Verdict |
|---|---|
| Joint evaluator遵守causal order與hard constraints | Supported for development implementation |
| Stage 3/4 parent boundaries可回復 | Supported on parent-native devices |
| H3+priority為seed-0 RZF matrix最佳mean-rate cell | Descriptive only |
| Priority改善每個UE/link或公平性 | Contradicted by tail/freshness diagnostics |
| Frozen C/D GNN可在Stage 5 B2運作 | Unsupported；Gate 5.5未執行 |
| AP-local RL改善frozen D-GNN或C-GNN | Unsupported；Gate 5.6未執行 |

## 8. Stage 5B pre-execution record

2026-08-28在任何Stage 5B implementation/training前，scope凍結為：

1. 先完成frozen C/D GNN的`fixed+RR@B2`、`H3+priority@B2`與`H3+B8` baselines；
2. 再以matching RZF/C/D rewards分別訓練$\pi_{\mathrm{RZF}}$、$\pi_C$、$\pi_D$；
3. Primary是$\pi_D+$D-GNN；$\pi_C+$C-GNN是upper reference；$\pi_{\mathrm{RZF}}+$RZF是control；
4. Frozen GNN不fine-tune；RZF-trained policy換接GNN只算transfer diagnostic。

目前Stage 5 stored-CSI GNN adapter、C/D B2/B8 baselines、AP-local actor/critic/trainer與learned artifacts皆不存在。`stage5b_eligible=true`只表示Stage 5A gate通過，不表示Stage 5B完成。

## 9. Statistical與reproducibility boundary

- Stage 1B/1C有5個effective seeds；Stage 2–5A仍主要是單一environment seed與單一AP layout的development evidence。
- Stage 3–5的多cell observations沒有formal multiplicity-controlled inference；不報significance、equivalence或winner claim。
- Local/remote hash一致證明artifact transfer integrity，不等於independent replication。
- Aggregate sum-rate不能下推每個UE/link；Stage 4/5的tail與starvation反例必須隨任何mean-rate結果一起報告。
- Simulator內paired interventions可描述within-simulator effect，但不能外推實際feedback-bit overhead、其他topologies或真實網路效果。

Formal evidence留到Stage 7：至少5個paired seeds，以seed或完整trajectory aggregate作統計單位，報paired effect、confidence interval、effect size與必要的multiplicity control。

## 10. Compact artifact index

| Evidence | Location |
|---|---|
| Stage 1C source/checkpoint handoff | `code/stage1/`及remote `lab301-5090:~/ThomasLai/code/stage1/` |
| Stage 2-BPP results | remote `lab301-5090:~/ThomasLai/code/stage2/` |
| Stage 3 results，包括封存SAC artifacts | remote `lab301-5090:~/ThomasLai/code/stage3/` |
| Stage 4 results | `code/stage4/results_stage4_seed0/`及remote `ws2:~/ThomasLai/code/stage4/` |
| Stage 5A successful results | `code/stage5/results_stage5_seed0_boundary_rerun2/`、`code/stage5/results_stage5_seed0_rerun2/`、`code/stage5/run-exp-v5-rerun2.log` |
| Stage 5A failed boundary attempts | remote `lab301-5090:~/ThomasLai/code/stage5/`；保留原fresh roots/logs |

Artifact路徑是provenance locator，不表示所有remote內容已獨立重跑或本地鏡像完整。

## 11. 下一次更新條件

下一次只在Stage 5B Gate 5.5有新evidence時更新：記錄frozen C/D GNN adapter hashes、B2/B8 compatibility、stored-CSI visibility、dynamic-mask/power checks與paired no-RL baselines。Gate 5.5通過前，不新增RL結果段落。
