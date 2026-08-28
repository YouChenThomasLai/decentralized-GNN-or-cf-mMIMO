# Stage 2 — Mobility Environment Qualification 結案摘要

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → closure
- Origin Date: 2026-08-20
- Last Updated: 2026-08-28
- Verification Status: ANALYZED
- Version Label: stage2_mobility_closure_v11_compacted
- Parent Plan: `doc/decentralized_active_csi_experiment_plan.md`
- Living Report: `doc/decentralized_active_csi_experiment_report.md`

## 1. 唯一任務與 disposition

Stage 2只負責驗證mobility/channel implementation，並建立fixed-association + full-current-CSI anchor。它不回答matched adaptation、method ranking、dynamic association、feedback scheduling或formal robustness。

Square-torus BPP版本已完成並通關。Legacy ring/disk pilots、matched-training anomaly、source修復chronology與舊矩陣已壓縮為一句provenance boundary，不再產生rerun或方法擴張義務。

## 2. Frozen input

- Stage 1C seed-0 checkpoint SHA-256：`16dba87572cd9f9dfc3272b4faeda2d7127dc414945450b856758efdba7bba45`
- AP coordinates SHA-256：`421b817e0e1e70b88db4e2ef685954b35f2e65432ffcb056190adb881df8b92d`
- Config SHA-256：`ace7fbe95ab86dab070de0a04b7517edd1924af1271446fc69d94e72cd98cf0b`
- 5 APs、8 UEs、$M=2$、15 dBm/AP、`noise_power=1e-12`。
- 200 m square torus；wrapped horizontal distance加10 m height difference。
- $\Delta t=1$ ms、$f_c=2.6$ GHz、2000 periods。
- Position-dependent path loss與first-order Gauss–Markov small-scale fading；coefficient由$J_0(2\pi f_D\Delta t)$決定。
- Association只在$t=0$建立，之後固定；所有methods每期使用current full CSI。

## 3. Completed matrix

| Mobility | Speeds | Role |
|---|---|---|
| Constant-speed straight | 0、30、80 km/h | stationary與mobility anchors |
| Hotspot semi-Markov | 3、30、80 km/h | environment qualification |
| Low/high stickiness and dwell | 2 environment-only settings | transition/dwell diagnostics |

Development使用seed 0、每setting 10 trajectories、`eval_time_stride=10`。Straight 3 km/h只在channel statistics中需要；hotspot 0 km/h不執行。

## 4. Gate verdicts

| Gate | Verdict | Minimal evidence |
|---|---|---|
| Provenance/Stage 1 compatibility | PASS | checkpoint/layout/config與source hashes一致；$t=0$ compatibility通過 |
| Kinematics | PASS | 0 km/h stationary、periodic wrap、displacement與10 m distance floor正確 |
| Channel | PASS | normalized moments finite；empirical lag correlation在tolerance內 |
| Hotspot process | PASS | transition、dwell、occupancy、continuity與phase diagnostics通過 |
| Fair inference | PASS | fixed mask；C/D GNN、MRT、RZF共用frames、noise與power constraints |
| Completion | PASS | runner exit status 0；8/8 settings complete；43 NPZ artifacts finite |

Stage 2結果只支持environment可用與inference可跑。Speed/mobility差異是descriptive trend，不是formal causal、monotonic或robustness evidence。

## 5. Minimal code surface

| File | Responsibility |
|---|---|
| `code/stage2/environment.py` | straight/hotspot trajectories與periodic boundary |
| `code/stage2/utils_return_indivial_rates.py` | wrapped distance、temporal channel與beamforming utilities |
| `code/stage2/trainer_2.py` | frozen-checkpoint evaluator與compact artifacts |
| `code/stage2/test_stage2.py` | kinematics、channel、provenance、mask與power gates |
| `code/stage2/run_exp-v2.sh` | six inference settings與two diagnostics |

## 6. Archived evidence boundary

舊ring/disk source與artifacts可用來解釋設計演進，但不能追認BPP gate，也不能作Stage 3–7數值基準。Stage 2不再新增multi-seed、full-stride、adaptation、equivalence或method-ranking runs。

若最終claim需要hotspot、mixed speed或coherence robustness，Stage 7只選必要的最小paired axis，並沿用本environment contract。

## 7. Validation commands

```bash
cd code/stage2
python -m py_compile *.py
bash -n run_exp-v2.sh
python test_stage2.py
```
