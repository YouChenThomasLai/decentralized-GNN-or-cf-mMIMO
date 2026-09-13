# Decentralized Active-CSI 實驗報告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate（compacted evidence ledger）
- Origin Date: 2026-08-18
- Last Updated: 2026-08-30
- Verification Status: ANALYZED（Stage 0–5B Gates 5.5–5.7與5-seed formal evaluation已完成；RL pipeline qualified，但learned performance為negative）
- Version Label: decentralized_active_csi_report_v20_compacted
- Plan: `doc/decentralized_active_csi_experiment_plan.md`
- Active Detail: `doc/stage5_joint_decentralized_active_csi_experiment_plan.md`

## 1. 報告範圍

本文件只保存會影響後續研究決策的最小 evidence：frozen contract、gate verdict、handoff、主要trade-off與claim boundary。已完成stage的逐iteration、逐checkpoint、逐trajectory、重複sweep表與orchestration敘事已移除；需要重算或稽核時應讀raw artifacts，而不是從本文件還原數據。

Stage 5B已完成matching RL qualification與5-seed formal evaluation；目前不再擴張此RL方法。

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
| 5B | Gates 5.5–5.7 PASS / learned negative | Frozen-GNN、RL pipeline與5-seed formal evaluation完成 | 實作qualified；RL不勝matching baseline |

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
| Frozen C/D GNN可在Stage 5 B2運作 | Supported for seed-0 compatibility qualification |
| AP-local RL改善frozen D-GNN或C-GNN | Contradicted by the tested 5-seed formal matrix |

## 8. Stage 5B Gate 5.5 evidence與RL handoff

2026-08-28在任何Stage 5B implementation/training前，scope凍結為：

1. 先完成frozen C/D GNN的`fixed+RR@B2`、`H3+priority@B2`與`H3+B8` baselines；
2. 再以matching RZF/C/D rewards分別訓練$\pi_{\mathrm{RZF}}$、$\pi_C$、$\pi_D$；
3. Primary是$\pi_D+$D-GNN；$\pi_C+$C-GNN是upper reference；$\pi_{\mathrm{RZF}}+$RZF是control；
4. Frozen GNN不fine-tune；RZF-trained policy換接GNN只算transfer diagnostic。

Gate 5.5於2026-08-29完成straight 0/30/80 km/h formal GPU qualification。三個settings與aggregate completion皆為complete；constraints、Stage 5A byte-identical inputs、stored/true CSI visibility與dynamic-mask/power checks全數通過，C/D `H3+B8`對Stage 3 full-current reference的最大絕對誤差為0。Remote artifacts同步後68個檔案逐檔SHA-256一致，79,640,800 bytes已封存於本機。

Matching no-RL `H3+priority@B2`最小evidence：30 km/h C/D sum rate為26.5149/25.8099，對各自B8 retention為99.876%/99.927%；80 km/h為25.8617/24.9061，retention為96.767%/97.076%。這些是compatibility與information-loss描述，不是C/D比較或method superiority。

Gate 5.6–5.7的三條deterministic smoke、seed-0 pilots、matching/crossed evaluation與5-seed formal matrix均完成；finite、reload、locality、no-leakage、top-2/B2、mask/power與protocol gates全數通過。

但learned performance失敗：5 seeds × 3 speeds × 3 matching beamformers共45個paired comparisons沒有正delta。九個formal cells的mean sum-rate delta約落在RZF −12.812至−11.471、D-GNN −11.412至−10.630、C-GNN −11.797至−11.317；各cell皆0/5 positive seeds，雙尾exact sign $p=0.0625$。三actor raw scores雖不同，top-2/B2 projected controls仍出現plateau且switching偏高。因此結論是「RL pipeline qualification成功，但本次最小RL方法不勝matching `H3+priority@B2` baseline」，不再宣稱或追索RL improvement。

## 9. Statistical與reproducibility boundary

- Stage 1B/1C與Stage 5B有5個effective seeds；Stage 2–5A仍主要是單一environment seed與單一AP layout的development evidence。
- Stage 5B formal方向在九個cells一致為負，但$n=5$且未作跨cell multiplicity correction；$p=0.0625$只作descriptive evidence，不宣稱一般性劣勢。
- Local/remote hash一致證明artifact transfer integrity，不等於independent replication。
- Aggregate sum-rate不能下推每個UE/link；Stage 4/5的tail與starvation反例必須隨任何mean-rate結果一起報告。
- Simulator內paired interventions可描述within-simulator effect，但不能外推實際feedback-bit overhead、其他topologies或真實網路效果。

Stage 5B的5-seed formal evidence只適用於目前凍結的單一AP layout、simulator與protocol；不能外推其他actor/action family或真實網路。

## 10. Compact artifact index

| Evidence | Location |
|---|---|
| Stage 1C source/checkpoint handoff | `code/stage1/`及remote `lab301-5090-tailscale:~/ThomasLai/code/stage1/` |
| Stage 2-BPP results | remote `lab301-5090-tailscale:~/ThomasLai/code/stage2/` |
| Stage 3 results，包括封存SAC artifacts | remote `lab301-5090-tailscale:~/ThomasLai/code/stage3/` |
| Stage 4 results | `code/stage4/results_stage4_seed0/`及remote `ws2:~/ThomasLai/code/stage4/` |
| Stage 5A successful results | `code/stage5/results_stage5_seed0_boundary_rerun2/`、`code/stage5/results_stage5_seed0_rerun2/`、`code/stage5/run-exp-v5-rerun2.log` |
| Stage 5A failed boundary attempts | remote `lab301-5090-tailscale:~/ThomasLai/code/stage5/`；保留原fresh roots/logs |
| Stage 5B Gate 5.5 formal evidence | `code/stage5/results_stage5b_gnn_gate_seed0/`及remote `lab301-5090-tailscale:~/ThomasLai/code/stage5/results_stage5b_gnn_gate_seed0/` |
| Stage 5B smoke、pilot與seed-0 evaluation | `code/stage5/results_stage5b_rl_*seed0*/`及對應remote roots |
| Stage 5B 5-seed formal evidence | remote `lab301-5090-tailscale:~/ThomasLai/code/stage5/results_stage5b_rl_formal/` |

Artifact路徑是provenance locator，不表示所有remote內容已獨立重跑或本地鏡像完整。

## 11. 下一次更新條件

Stage 5B以qualified pipeline／learned negative封存。下一次只在formal artifacts完成本機鏡像，或另行凍結新的method scope時更新；不因本次負結果事後更換seed、ceiling或action family。
