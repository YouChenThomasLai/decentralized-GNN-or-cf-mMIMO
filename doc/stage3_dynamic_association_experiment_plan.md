# Stage 3 — Dynamic Association Qualification 結案摘要

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → closure
- Origin Date: 2026-08-24
- Last Updated: 2026-08-28
- Verification Status: ANALYZED
- Version Label: stage3_dynamic_association_closure_v8_compacted
- Parent Plan: `doc/decentralized_active_csi_experiment_plan.md`
- Living Report: `doc/decentralized_active_csi_experiment_report.md`
- Stage 2 Contract: `doc/stage2_mobility_implementation_plan.md`

## 1. 唯一任務與 disposition

Stage 3只需在full-current-CSI條件下確認dynamic-association action非退化，並凍結一個Stage 5可用的strong modular baseline。此任務已完成；Stage 3不再是active method-development stage。

文獻已建立mobility-aware association與handoff-aware sequential control的可行性，本專案不再用Stage 3重證一般性優勢，也不再追加learned association seeds、switching penalties、handoff intervals或architecture search。

## 2. Frozen contract

- 完整沿用Stage 2-BPP topology、channel、noise、power與straight/hotspot environments。
- Primary development使用straight 0/30/80 km/h；full-current CSI。
- Association每50 ms更新；每個UE恰由top-2 APs服務；不加per-AP capacity。
- Primary beamformer為current-CSI RZF；MRT與frozen Stage 1C GNN只作secondary checks。
- Action只能使用當期合法LSF/history與previous association；rate使用action後同一期true channel。
- 所有methods共用positions、channel innovations、decision epochs、noise、power與evaluation frames。

## 3. Frozen handoff

Primary modular association為`hysteresis_lsf_top2_h3_db`（H3）：只有outsider的current local LSF超過weakest incumbent 3 dB時才replacement。

H3保留的理由是它在development matrix中提供非退化moving action與合理rate–switching折衷。這是Stage 5的mechanical handoff，不是跨mobility或formal superiority claim。

## 4. Completed gates

| Gate | Verdict | Minimal evidence |
|---|---|---|
| Stage 2 handoff/provenance | PASS | Fixed-mask RZF重現、source/artifact lineage通過 |
| Association constraint | PASS | 每個UE每期恰有2個serving APs；mask/power合法 |
| Causality/fairness | PASS | Future-channel perturbation不改current action；paired traces共用 |
| Heuristic signal | PASS | Moving setting有nonzero association changes與finite rate/switching metrics |
| Learned implementation | PASS mechanically | SAC training、finite、reload與artifact contracts可運作 |
| Learned performance | Negative development result | SAC current/history均不勝strong heuristics且switching更高 |

## 5. SAC record（壓縮）

`sac_current`與`sac_history`已完成implementation smoke、seed-0 straight-only training與0/30/80 km/h paired evaluation。兩者通過finite、checkpoint reload、top-2 projection與constraint gates，但在相同development traces上都低於fixed/current/H3/H6，且switching更頻繁。

這個結果的處理方式固定為：

- 保留「learned policy未勝strong heuristics」這個negative conclusion；
- 不保留逐checkpoint、逐validation step、逐speed或逐trajectory數字於active docs；
- 不補policy seeds、$\lambda_{\mathrm{sw}}$、categorical fallback或history architecture；
- 不把這兩個global-score SAC policies稱為decentralized，也不帶入Stage 5B/7 mandatory matrix；
- raw checkpoints、metrics、logs與completion/provenance保留在原artifact roots供稽核。

此negative result只針對已測continuous-score SAC設計，不支持「RL不適合dynamic association」的一般結論。

## 6. Stage 5 handoff

Stage 5只從Stage 3移植：

- top-2 projection；
- H3 update rule；
- switching definitions；
- dynamic-mask formatting與Stage 1C frozen GNN loading所需components。

Stage 5不以Stage 3 full-current evaluator為runtime base；stored-CSI causal loop以Stage 4為主。Stage 3 SAC checkpoints只保留為封存diagnostic，不是Stage 5 learned policy。

## 7. Evidence boundary

Stage 3使用單一environment seed與development trajectories，曾比較多個heuristics與metrics；沒有formal multiplicity-controlled inference。可支持的只有implementation、causality、action signal、H3 handoff與SAC negative development verdict。

正式association/feedback method effect必須在Stage 7以matching Stage 5 policies、paired multi-seed traces與預先凍結comparisons檢驗。

## 8. Artifact locator

- Stage 3A heuristics：remote `lab301-5090:~/ThomasLai/code/stage3/results_stage3a_bpp/`
- SAC smoke/training/evaluation：remote `lab301-5090:~/ThomasLai/code/stage3/`下既有`results_stage3b_*` roots
- Source：`code/stage3/`

Artifact保留不代表它們仍屬active experiment matrix；不得因存在checkpoint而自動重新開啟Stage 3。
