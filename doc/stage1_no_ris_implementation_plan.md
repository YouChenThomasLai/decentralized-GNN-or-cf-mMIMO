# Stage 1 No-RIS Snapshot Baseline — 結案與 Handoff 摘要

## Status

- Last Updated: 2026-08-28
- Verification Status: ANALYZED
- Version Label: stage1_no_ris_closure_v2_compacted
- Parent Plan: `doc/decentralized_active_csi_experiment_plan.md`
- Disposition: Stage 1A/1B/1C 已完成；本文件不再是active implementation plan。

## 1. 最終用途

Stage 1只留下三個作用：

1. Stage 1A證明直接移除RIS並沿用原noise會落入noise floor；
2. Stage 1B凍結可學習的no-RIS數值尺度；
3. Stage 1C建立BPP topology-matched frozen GNN checkpoint與AP layout供Stage 2–7使用。

舊版逐檔實作順序、預期輸出與已完成checklist已移除。後續若修改topology、distance、noise、model shape或checkpoint lineage，必須重新qualification，不得把本結案紀錄當成自動通關。

## 2. Frozen no-RIS contract

- 5 APs、8 UEs、$M=2$、15 dBm/AP、association threshold 0.1。
- 200 m × 200 m square torus；AP/UE為fixed-count BPP；wrapped horizontal distance加10 m height difference。
- Direct Rayleigh channel；`noise_power=1e-12`。
- GNN只學complex precoder $W$與AP power fraction $c$；RIS input、embedding、readout與phase output全部移除。
- Centralized/decentralized evaluation共用UE positions、channels、association mask、noise與power constraints；差別只在inference visibility。
- MRT/RZF只作sanity controls；GNN勝過它們不是qualification gate。

## 3. Minimal source scope

| File | Frozen responsibility |
|---|---|
| `code/stage1/environment.py` | BPP AP/UE placement與topology metadata |
| `code/stage1/utils_return_indivial_rates.py` | wrapped 2D distance、10 m 3D distance、channel/rate/MRT/RZF |
| `code/stage1/data.py` | direct-only data contract |
| `code/stage1/model_2.py` | no-RIS GNN $(W,c)$ |
| `code/stage1/trainer_2.py` | training、paired evaluation與provenance |
| `code/stage1/topology_gate.py` | training-free TQ0 geometry harness |
| `code/stage1/test_stage1.py` | geometry、power、mask與reproducibility contracts |

沒有建立shared geometry package或新增dependency；Stage 2另有stage-local copy，以source hashes控制漂移。

## 4. Gate verdicts

### Stage 1A — negative control

沿用Stage 0 `noise_power=4e-4`後，所有rates與C–D gap落入數值noise-floor regime，task gradient不足。此結果不支持C/D等效，只支持必須先校準數值尺度。

### Stage 1B — noise calibration

Seed-0 sensitivity與5-seed主run均支持`noise_power=1e-12`為finite、可學習設定。Seeds 0–4各2000 iterations，artifacts完整且未見collapse。C–D方向不穩健，因此不作method claim。

### Stage 1C — BPP requalification

| Gate | Verdict | Minimal evidence |
|---|---|---|
| TQ0 geometry | PASS | 200 topology seeds × 100 UE drops；wrap、finite、reproducibility與10 m floor通過 |
| TQ1 channel/association | PASS | $K=8$ MRT/RZF finite；mask/power合法；存在moving association opportunities |
| TQ2 snapshot baseline | PASS | 5 effective seeds × 2000 iterations；configs、checkpoints、metrics與training arrays完整且finite |

TQ2 final means：centralized 27.4597、decentralized 26.7810、MRT 21.6380、RZF 19.8096。C−D mean 0.6787（2.472%），5/5同向但exact paired sign-flip $p=0.0625$；last-500 slopes仍為正。這是development gate，不是穩健C優勢、等效或完全收斂證據。

## 5. Frozen handoff artifacts

| Artifact | SHA-256 |
|---|---|
| Stage 1C seed-0 checkpoint | `16dba87572cd9f9dfc3272b4faeda2d7127dc414945450b856758efdba7bba45` |
| AP coordinates | `421b817e0e1e70b88db4e2ef685954b35f2e65432ffcb056190adb881df8b92d` |
| Config | `ace7fbe95ab86dab070de0a04b7517edd1924af1271446fc69d94e72cd98cf0b` |

Stage 2–7必須同時載入這三者及matching model source。Stage 1B ring/disk checkpoint與早期network-scaling pilots只作legacy evidence，不能替代Stage 1C handoff。

## 6. Validation commands

```bash
cd code/stage1
python -m py_compile *.py
python test_stage1.py
python topology_gate.py --topology_seeds 200 --ue_drops_per_topology 100
```

完整training rerun只在frozen contract或artifact lineage改變時需要；一般後續工作直接讀checkpoint，不重跑Stage 1。
