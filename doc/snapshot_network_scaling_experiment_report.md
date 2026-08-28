# Snapshot Network-Scaling Pilot — 封存摘要

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate → archive
- Last Updated: 2026-08-28
- Verification Status: ANALYZED
- Version Label: snapshot_network_scaling_archive_v3_compacted
- Disposition: 舊experiment plan已刪除；本文件只保留影響Stage 1C topology contract的設計教訓。

## 1. Disposition

早期network-scaling工作沒有產生可用的formal scaling claim。Ring/disk v1與BPP v2都只作development evidence；目前主線已由Stage 1C fixed-count BPP qualification與Stage 7預先登記的optional scaling axis取代。

不再續跑舊matrix，也不以舊checkpoint、十個AP layouts或跨scale aggregate tables支持scalability。Raw source/artifacts保留於原位置供provenance稽核。

## 2. 保留的最小證據

### v1 ring topology

- 原RIS Stage 0在scale增加時仍位於meaningful-rate regime，可作positive control。
- No-RIS Stage 1A沿用舊noise後位於noise floor，不能用於network-scaling interpretation。
- Centralized/decentralized gap隨scale的單一seed趨勢不足以支持generalization。

### v2 BPP pilot

- 2D BPP與wrap-around implementation contracts可運作。
- 只用十個AP layouts不足以穩定AP-spacing distribution gate。
- 沒有3D distance floor時，近距離配合高path-loss exponent會造成極端received-power sensitivity。
- 因AP layout、UE drops、model initialization與training randomness同時改變，scale-1/2差異不是pure network-size effect。

## 3. 對目前主線的影響

Stage 1C據此採用：

- 200 m × 200 m square torus；
- fixed-count BPP AP/UE placement；
- wrapped horizontal distance加10 m height difference；
- 至少200 AP topology seeds × 每seed100 UE drops的training-free geometry gate；
- topology-matched checkpoint與AP-layout lineage。

這些implementation lessons已進入`doc/decentralized_active_csi_experiment_plan.md`與Stage 1C code；舊scaling plan不再需要存在。

## 4. Future scaling gate

只有最終paper claim明確包含network-size scalability/generalization時，Stage 7才新增一個secondary axis。新gate必須固定density或明確說明fixed-area scaling、使用多個AP layouts、分離topology與training seeds，並重新訓練或提供可辯護的cross-size model contract。

## 5. Artifact locator

- v1 source：`code/snapshot_network_scaling/`
- v2 source：`code/snapshot_network_scaling/v2_bpp/`
- v1 remote artifacts：`/tmp2/b12902052/snapshot_network_scaling/results_snapshot_scaling/` on `ws2`
- v2 remote staging：`/tmp2/b12902052/snapshot_network_scaling_v2_bpp/`

Artifacts保留不代表舊plan仍active；不得從archive自動生成新實驗義務。
