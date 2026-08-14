# II. RELATED WORK

CF-mMIMO 最早多假設所有 AP 共同服務所有 UE [1]。為改善 scalability，後續工作改採 user-centric clustering，使每個 UE 僅由部分 AP 服務 [2]。本研究進一步關注 mobility 與有限 feedback 下，CSI acquisition、AP–UE association 和 decentralized execution 之間的交互影響。

## A. Limited Feedback and CSI Acquisition in CF-mMIMO

FDD CF-mMIMO 無法直接依賴 uplink–downlink channel reciprocity，因此 downlink CSI acquisition 與 feedback overhead 是重要限制。Yin et al. [3] 分析 quantized limited feedback 對 CF-mMIMO performance 的影響，說明 feedback quality 與 achievable rate 之間的取捨。

Zhang et al. [4] 透過 joint port selection，只取得較重要的 spatial-domain channel components，以降低 FDD channel training 與 feedback overhead；其 learning-based version 也考慮 fast-varying scenarios。不過，該工作的 decision unit 是 channel port，而不是跨時間選擇 AP–UE link 是否更新。

Lee et al. [5] 在每個 AP 的總 feedback-bit budget 下，jointly optimize user-centric association 與 feedback-bit allocation。這是與本題最接近的 limited-feedback CF-mMIMO 工作之一，但其重點是 long-term association 與 quantization bits，並未明確建模：

- 每個 decision period 的 CSI refresh action；
- 未更新 link 繼續使用舊 CSI；
- CSI age、mobility 與 association switching cost；
- decentralized temporal policy。

因此，limited-feedback CF-mMIMO 文獻主要回答「feedback bits 或 channel dimensions 如何分配」，尚未完整回答「有限 budget 下，哪些 AP–UE CSI 現在值得更新」。

## B. CSI Aging and Selective Channel Updating

Deng et al. [6] 指出，不同 mobility 的 users 具有不同 channel coherence，持續更新所有 users 的 CSI 會浪費 pilot resources。他們提出 intermittent channel estimation，保留 aged CSI，並以 MDP 或 convex relaxation 決定 user-level CSI update pattern。這篇工作直接支持本研究的主要假設：當 CSI update 具有資源限制且 channel 隨時間變化時，更新決策具有長期影響。

Chen and Wang [7] 在 ISAC 系統中，讓 policy 為每個 user/target 決定重新估測 CSI 或使用 prediction，並與 beamforming 共同設計。該工作也說明 active CSI acquisition 存在 causality：決策前無法先知道重新估測後的 CSI，因此 update action 必須依據目前可觀察資訊做出。

然而，這些 selective-update 工作並非針對 user-centric CF-mMIMO，通常沒有同時處理：

- 多個 AP 各自持有不同 local CSI；
- AP–UE association；
- association switching cost；
- decentralized multi-agent execution。

## C. Mobility-Aware Dynamic AP–UE Association

Ammar et al. [8] 將 user-centric CF-MIMO handoff 建模為 POMDP，利用 large-scale fading 的時間演化，在維持 rate 的同時減少 handoffs。其後續工作 [9] 使用 SAC，並比較 mobility-direction-assisted 與 channel-history-assisted observations，明確將 handoff penalty 放入長期 objective。

這些研究說明 mobility、channel history 與 previous association 會影響未來 performance，因此 dynamic association 適合使用 sequential decision methods。不過，它們假設 association policy 所需的 channel observations 已經存在，並未讓 policy 決定哪些 AP–UE CSI 要更新。

因此，單純提出 history-based association 或 switching-aware RL 已不足以構成主要 novelty；本研究還必須處理有限 feedback 所造成的 stale CSI 與 active acquisition decision。

## D. Decentralized Learning and Local-CSI Execution

若 beamforming、power allocation 或 association 由 CPU 集中計算，AP 必須回傳大量 CSI，CPU 也必須將決策傳回各 AP。Hojatian et al. [10] 使用 unsupervised DNN 實現 fully/partially decentralized beamforming，降低 AP–network controller 間的通訊與計算成本。Tung et al. [11] 進一步使用 distributed GNN，讓 AP 根據 local CSI 進行 power allocation，並取得接近 centralized learning 的 sum ergodic SE。

Ting et al. [12]（即 [A]）在 multi-RIS-aided CF network 中，讓每個 AP 使用預先定義的 local CSI，獨立產生 beamforming vectors 與 RIS phase estimates。這些工作證明 local-CSI decentralized inference 是可行的，但主要仍屬於 snapshot-based optimization：

- 每次 inference 所需的 local CSI 被假設為可即時取得；
- 沒有 CSI age、update history 或 per-period feedback budget；
- [12] 的 AP–UE association 為預先決定，沒有 switching dynamics。

在 association 方面，Banerjee et al. [13] 將每個 AP 視為 agent，使用 conventional/federated MARL 決定要服務哪些 UEs。其 actors 可 decentralized execution，但訓練使用 centralized critic。APS-GNN [14] 則將 decision 分解到 AP–UE pair agents，使用 RNN 編碼 local channel history，再透過 GNN message passing 進行 distributed association。

上述方法證明 decentralized temporal association 已經可行，但仍假設所需的 current channel observations 已取得，沒有把 CSI acquisition 本身視為 action。因此，這條文獻線提供本研究的 execution architecture，但沒有回答「local CSI 無法全部更新時，各 AP 應優先取得哪些資訊」。

## E. Comparison and Research Gap

| Work | System and decision | Temporal state | CSI refresh action | Dynamic association | Decentralized execution |
|---|---|---:|---:|---:|---:|
| Hojatian et al. [10] | CF beamforming | No | No | No | Yes |
| Tung et al. [11] | CF power allocation | No | No | No | Yes |
| Ting et al. [12] | CF beamforming with RIS | No | No | Fixed | Yes |
| Zhang et al. [4] | FDD CF port selection | Limited | Port selection only | No | No |
| Lee et al. [5] | Association + feedback bits | Long-term statistics | No | Long-term only | No |
| Deng et al. [6] | Massive-MIMO CSI updates | Yes | User-level | No | No |
| Chen and Wang [7] | ISAC CSI re-estimation | Yes | User/target-level | No | No |
| Ammar et al. [8], [9] | CF handoff control | Yes | No | Yes | No |
| Banerjee et al. [13] | CF AP clustering | Mobility | No | Yes | Yes (CTDE) |
| APS-GNN [14] | Distributed CF AP selection | History + RNN | No | Yes | Yes |
| **Proposed direction** | **CF CSI update + association** | **CSI age/history** | **AP–UE link-level** | **Yes** | **Yes** |

CTDE 表示 centralized training with decentralized execution；因此 Banerjee et al. [13] 不應分類為 partially decentralized execution。

### Defensible Gap

現有研究大致分別處理：

1. 在 FDD CF-mMIMO 中分配 feedback bits 或選擇 channel components [3]–[5]；
2. 在其他 MIMO/ISAC 系統中安排 intermittent CSI updates [6], [7]；
3. 在 mobility 下控制 AP–UE association 與 handoffs [8], [9]；
4. 使用 current local CSI 進行 decentralized beamforming 或 association [10]–[14]。

較明確且可防守的缺口，是在 **per-period feedback budget** 下，由各 AP 根據 **stored local CSI、CSI age/history 與 previous association**，共同控制 **AP–UE CSI updates 與 dynamic association**。未更新的 links 明確保留 stale CSI，而 policy 以 decentralized 方式執行。

本研究的 novelty 不應只描述為 non-snapshot RL、limited feedback、dynamic association 或 decentralized learning，而應定位為四條文獻線的交集：

> **Budgeted active AP–UE CSI acquisition + stale-CSI-aware dynamic association + decentralized execution.**

Action 採 simultaneous、delayed、causal two-stage 或 hierarchical design，屬於後續需要實驗比較的 algorithmic choice，不必預先鎖定。

## F. Why the Problem Matters

- FDD CF-mMIMO 的 CSI overhead 會隨 AP–UE links 增加，無法假設每期全部更新。
- Mobility 使不同 links 的 CSI 以不同速度失效，固定週期更新容易浪費 budget。
- CSI acquisition 與 association 相互依賴：可能服務的 link 才值得更新，但是否服務又取決於 CSI 是否可信。
- Association switching 具有後續成本，因此單一 snapshot 的 sum-rate maximization 可能不是最佳長期策略。

# References

[1] H. Q. Ngo, A. Ashikhmin, H. Yang, E. G. Larsson, and T. L. Marzetta, “Cell-free massive MIMO versus small cells,” *IEEE Trans. Wireless Commun.*, vol. 16, no. 3, pp. 1834–1850, Mar. 2017.

[2] E. Björnson and L. Sanguinetti, “Scalable cell-free massive MIMO systems,” *IEEE Trans. Commun.*, vol. 68, no. 7, pp. 4247–4261, Jul. 2020.

[3] X. Yin, J. Dai, and J. Shi, “Performance analysis of cell-free massive MIMO systems under limited feedback,” in *Proc. IEEE Int. Conf. Commun. Workshops (ICC Workshops)*, May 2019, pp. 1–6.

[4] C. Zhang, P. Du, M. Ding, Y. Jing, and Y. Huang, “Joint port selection based channel acquisition for FDD cell-free massive MIMO,” *IEEE Trans. Commun.*, vol. 72, no. 6, pp. 3572–3586, Jun. 2024.

[5] K. Lee, J. H. Lee, and W. Choi, “User-centric association and feedback bit allocation for FDD cell-free massive MIMO,” *IEEE Trans. Wireless Commun.*, vol. 25, pp. 1891–1907, 2026.

[6] R. Deng, Z. Jiang, S. Zhou, and Z. Niu, “Intermittent CSI update for massive MIMO systems with heterogeneous user mobility,” *IEEE Trans. Commun.*, vol. 67, no. 7, pp. 4811–4824, Jul. 2019.

[7] J. Chen and X. Wang, “Learning-based intermittent CSI estimation with adaptive intervals in integrated sensing and communication systems,” *IEEE J. Sel. Topics Signal Process.*, vol. 18, no. 5, pp. 917–932, Jul. 2024.

[8] H. A. Ammar, R. Adve, S. Shahbazpanahi, G. Boudreau, and K. V. Srinivas, “Handoffs in user-centric cell-free MIMO networks: A POMDP framework,” *IEEE Trans. Wireless Commun.*, vol. 23, no. 8, pp. 10319–10335, Aug. 2024.

[9] H. A. Ammar, R. Adve, S. Shahbazpanahi, G. Boudreau, and I. Bahceci, “Handoff design in user-centric cell-free massive MIMO networks using DRL,” *IEEE Trans. Commun.*, vol. 73, no. 11, pp. 11368–11384, Nov. 2025.

[10] H. Hojatian, J. Nadal, J.-F. Frigon, and F. Leduc-Primeau, “Decentralized beamforming for cell-free massive MIMO with unsupervised learning,” *IEEE Commun. Lett.*, vol. 26, no. 5, pp. 1042–1046, May 2022.

[11] N. X. Tung, T. V. Chien, H. Q. Ngo, and W. J. Hwang, “Distributed graph neural network design for sum ergodic spectral efficiency maximization in cell-free massive MIMO,” *IEEE Trans. Veh. Technol.*, vol. 74, no. 3, pp. 5181–5186, Mar. 2025.

[12] W.-Y. Ting, R. Y. Chang, F.-T. Chien, T.-Y. Peng, and P.-H. Lin, “Decentralized graph neural network-based joint beamforming in multi-RIS-aided cell-free networks,” in *Proc. IEEE Veh. Technol. Conf. (VTC)*, Sep. 2026.

[13] B. Banerjee, R. C. Elliott, W. A. Krzymień, and M. Medra, “Access point clustering in cell-free massive MIMO using conventional and federated multi-agent reinforcement learning,” *IEEE Trans. Mach. Learn. Commun. Netw.*, vol. 1, pp. 107–123, 2023.

[14] M. Zangooei, L. Salaün, C. S. Chen, and R. Boutaba, “Graph-neural multi-agent coordination for distributed access-point selection in cell-free massive MIMO,” *arXiv preprint arXiv:2602.17954*, Feb. 2026.
