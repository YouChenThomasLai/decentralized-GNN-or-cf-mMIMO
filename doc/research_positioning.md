# Research positioning and literature notes

## 1. Status

- Updated: 2026-09-17
- Scope: contribution boundary for the G2 graph representation, local-energy consensus, and the
  AP-to-CPU RIS message interface
- Methods: [Method report](./decentralized_ris_methods.md)
- Evidence: [Experiment index](./decentralized_ris_experiments.md)
- Local literature anchors:
  [decentralized multi-RIS GNN](<./Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks.pdf>)
  and
  [load-balanced attention GNN](<./Beamforming_and_Load-Balanced_User_Association_in_RIS-Aided_mmWave_Systems_via_Adaptive_Attention_Graph_Neural_Networks (1).pdf>)
- Verification: the works below were checked against the local PDFs, publisher metadata, or arXiv
  abstracts. The search is representative rather than a systematic review.

## 2. Supported positioning

G2 is best positioned as a communication-aware inference architecture for joint active and passive
beamforming, not as the first use of a GNN, weighted aggregation, or decentralized optimization. Its
system-specific combination is:

1. each AP produces its active beamformer and a feasible per-RIS phase proposal under the
   paper-decentralized CSI view;
2. each AP attaches one strictly local channel-energy scalar to each RIS proposal;
3. the CPU performs parameter-free circular consensus rather than a learned feature reduction or an
   online iterative optimization;
4. a node-free per-RIS link-token graph learns the beamformer and proposals jointly; and
5. the AP-to-CPU RIS message contains $N$ phase angles plus one energy scalar, instead of R0's $4N$
   learned features.

E05 verifies the inference-time energy rule. E06 shows that G2 can be trained end to end and exceeds
the mature R0 reference on the fixed holdout. The large empirical separation occurs in the G0→G1/G2
representation-package change, which includes the node-free per-RIS link tokens, a new encoder/head,
and a capacity change. The G1→G2 context addition is not detectably better in the primary 40k
comparison, although it prevents the G1 own-only collapse. This distinction should remain clear in
any contribution statement.

## 3. What is not a contribution by itself

- **R0→R0c commutation:** moving a shared affine reduction from after the AP sum to before it is
  algebraically exact and should be presented as a control, not novelty.
- **Weighted averaging:** scalar sender–receiver weights and parameter-free aggregation already have
  close precedents in multi-agent communication, distributed GNNs, and consensus optimization.
- **RIS phase compression:** prior work already studies low-bit RIS control and task-aware compression.
- **A small post-hoc rate gain:** E04's $+0.0419$ bps/Hz fixed-checkpoint difference is not evidence
  that learned pair magnitude is superior.
- **The provisional codec frontier:** E07 stopped on a failed control and cannot yet support a
  bits–rate claim.

## 4. Related-work landscape

### 4.1 Centralized and partially distributed model-based design

Model-based RIS-aided cell-free methods commonly formulate joint active/passive beamforming as a
non-convex weighted sum-rate or sum-MSE problem and solve it by alternating optimization (AO), block
coordinate descent, or successive approximation. These methods provide strong numerical references,
but their per-channel online iterations and global state exchange are different from amortized neural
inference. Ni et al. place active beamforming at the APs while retaining CPU-side passive beamforming
inside an iterative AO loop. Zhang et al. similarly decentralize parts of a joint precoding and RIS
beamforming solver. They are therefore useful performance and signaling baselines, rather than prior
instances of G2's one-shot proposal interface.

Recent work also reduces overhead by changing the optimization problem. Xu et al. jointly select AP
clusters and beamformers, while Ke et al. introduce RIS–UE association to avoid acquiring irrelevant
channels. These papers motivate future association or load-balancing extensions, but adding those
variables now would confound the narrower question of whether G2 is already competitive for the
current fixed-association problem.

Representative works:

- Ni et al., “Partially Distributed Beamforming Design for RIS-Aided Cell-Free Networks,”
  [arXiv:2208.05210](https://arxiv.org/abs/2208.05210).
- Zhang et al., “Joint Distributed Precoding and Beamforming for RIS-aided Cell-Free Massive MIMO
  Systems,” [arXiv:2311.13139](https://arxiv.org/abs/2311.13139).
- Xu et al., “Joint AP Clustering and Beamforming Design for RIS-Aided Cell-Free Networks,” IEEE
  TVT 2025, [DOI:10.1109/TVT.2024.3521479](https://doi.org/10.1109/TVT.2024.3521479).
- Ke et al., “Joint RIS-UE Association and Beamforming Design in RIS-Assisted Cell-Free MIMO
  Network,” IEEE TCOM 2025, [arXiv:2506.21690](https://arxiv.org/abs/2506.21690).

### 4.2 Iterative decentralized optimization

Decentralized optimization removes or reduces centralized computation by exchanging local copies,
dual variables, or consensus states. Huang et al. use incremental ADMM for IRS-enhanced cell-free
beamforming and approach the centralized solution as the number of online iterations grows. Xu et
al. unroll a distributed ADMM solver and use directed information exchange. Katsanos and
Alexandropoulos consider a wider multi-RIS, wideband, imperfect-CSI setting and coordinate active and
passive beamforming through consensus updates. Du et al. instead study an active sub-connected RIS
and use a two-stage distributed iterative design. These are the closest algorithmic precedents for
distributed RIS coordination. Their strength is structured optimization; their deployment cost is
iteration-dependent signaling and latency. G2 instead pays a fixed one-message AP→CPU cost and does
not claim the convergence guarantees of those solvers.

- Huang et al., “Decentralized Beamforming Design for Intelligent Reflecting Surface-enhanced
  Cell-free Networks,” IEEE WCL 2021,
  [arXiv:2006.12238](https://arxiv.org/abs/2006.12238).
- Xu et al., “Algorithm Unrolling-Based Distributed Optimization for RIS-Assisted Cell-Free
  Networks,” [arXiv:2301.02360](https://arxiv.org/abs/2301.02360).
- Katsanos and Alexandropoulos, “Robust Consensus-Based Distributed Beamforming for Wideband
  Cell-free Multi-RIS MISO Systems,”
  [arXiv:2601.08946](https://arxiv.org/abs/2601.08946).
- Du et al., “Robust Beamforming Design for Active Sub-Connected RIS Assisted Cell-Free MIMO
  Systems: A Two-Stage Distributed Approach,” IEEE IoT-J 2025,
  [DOI:10.1109/JIOT.2025.3572444](https://doi.org/10.1109/JIOT.2025.3572444).

### 4.3 Learning-based distributed and graph-based design

Hojatian et al. establish that unsupervised DNNs can move cell-free beamforming to APs with zero or
limited AP–controller exchange, but do not jointly coordinate multiple RISs. Chen et al. learn local
BS beamformers and let one BS determine the IRS coefficients; this avoids CPU-heavy iterative
optimization, but centralizes passive control at a designated BS and studies a different interface.
Zhu et al. use centralized-training/distributed-execution multi-agent reinforcement learning for
multi-RIS cell-free precoding, with local AP CSI and CPU-controlled RISs, but do not expose an
AP-proposal/fusion interface or its fixed payload.
The local anchor paper is the closest GNN baseline: it jointly learns AP beamforming and multi-RIS
control, sends $4N$ learned features per AP–RIS pair, and uses a trainable CPU-side affine reducer.

Chan et al. show how an introduction and comparison table can isolate a GNN paper's actual problem
dimensions. Their adaptive-attention GNN jointly handles user association, load balancing,
beamforming, and RIS phases and generalizes across BS/UE counts. It is a strong precedent for graph
generalization and for future load-balancing work, but it is not a decentralized AP→CPU coordination
method for a multi-RIS cell-free network. G2's narrower distinction is the co-design of the local graph
representation and a fixed-size, interpretable proposal-and-confidence message.

- Hojatian et al., “Decentralized Beamforming for Cell-Free Massive MIMO with Unsupervised
  Learning,” IEEE Communications Letters 2022,
  [arXiv:2106.16194](https://arxiv.org/abs/2106.16194).
- Chen et al., “A Distributed Machine Learning-Based Approach for IRS-Enhanced Cell-Free MIMO
  Networks,” [arXiv:2301.08077](https://arxiv.org/abs/2301.08077).
- Zhu et al., “Multi-agent Reinforcement Learning-based Joint Precoding and Phase Shift Optimization
  for RIS-aided Cell-Free Massive MIMO Systems,” IEEE TVT 2024,
  [arXiv:2404.14092](https://arxiv.org/abs/2404.14092).
- Ting et al.,
  [“Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks”](<./Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks.pdf>).
- Chan et al., “Beamforming and Load-Balanced User Association in RIS-Aided mmWave Systems via
  Adaptive Attention Graph Neural Networks,” IEEE TWC 2026,
  [DOI:10.1109/TWC.2025.3621268](https://doi.org/10.1109/TWC.2025.3621268).

### 4.4 Supporting communication and compression precedents

The proposal interface should also be discussed conservatively against adjacent communication work.
Over-the-air GNN aggregation, targeted multi-agent communication, task-oriented feature compression,
and learned distributed source coding all show that low-dimensional or weighted messages are not new
in isolation. RIS control-signaling and phase-compression studies further motivate counting actual
bits instead of only tensor dimensions. The relevant precedents are Gu et al.
([arXiv:2207.08498](https://arxiv.org/abs/2207.08498)), Das et al.
([arXiv:1810.11187](https://arxiv.org/abs/1810.11187)), Shao et al.
([arXiv:2102.04170](https://arxiv.org/abs/2102.04170),
[arXiv:2109.00172](https://arxiv.org/abs/2109.00172)), Sohrabi et al.
([arXiv:2007.06512](https://arxiv.org/abs/2007.06512)), Fernandes and Psaromiligkos
([arXiv:2510.05438](https://arxiv.org/abs/2510.05438)), and Enqvist et al.
([arXiv:2506.03929](https://arxiv.org/abs/2506.03929)). Guo et al.'s two-way RIS design is a more
specific precedent for phase averaging
([arXiv:2101.10094](https://arxiv.org/abs/2101.10094)).

## 5. Comparison with the closest decentralized designs

This table follows the role of Table I in Chan et al.: it separates the dimensions needed for the
paper's claim. It is a feature map, not a performance ranking, because the works use different
channel models, objectives, and information assumptions. “One-shot” means no per-channel AO/ADMM
loop at deployment time.

| Work | RIS setting | AP-side active design | Deployment-time coordination | RIS-coordination message/state | Final RIS fusion |
|---|---|---|---|---|---|
| Huang et al. 2021 | IRS-aided cell-free | Local iterative update | Incremental ADMM rounds | Solver/consensus state | Iterative consensus optimization |
| Hojatian et al. 2022 | Cell-free, no RIS | One-shot DNN | Zero or limited AP–NC exchange | Not applicable | Not applicable |
| Ni et al. 2022 | RIS-aided cell-free | Local optimization | AO between AP and CPU subproblems | Optimization variables/CSI | CPU optimizes passive beamforming |
| Xu et al. 2023 | RIS-assisted cell-free | Distributed unrolled solver | D-ADMM layer exchanges | Directed solver states | Learned iterative consensus |
| Zhang et al. 2023 | RIS-aided cell-free | Local optimization | Distributed AO | Precoder/beamforming variables | CPU-assisted passive optimization |
| Chen et al. 2023 | IRS-enhanced cell-free | One-shot local ML | A designated BS controls the IRS | No common proposal interface | Single-BS passive decision |
| Zhu et al. 2024 | Multi-RIS cell-free | One-shot local MARL actor | CTDE; CPU controls RISs | Not exposed as a fixed interface | CPU-side RIS decision |
| Ting et al. anchor | Multi-RIS cell-free | One-shot paper-decentralized GNN | One AP→CPU message | $4N$ learned features/AP–RIS | Trainable affine CPU reducer |
| Katsanos and Alexandropoulos 2026 | Wideband multi-RIS cell-free | Local cooperative optimization | Consensus rounds | Iterative variable updates | Distributed consensus optimization |
| **G2 (this project)** | **Multi-RIS cell-free** | **One-shot paper-decentralized GNN** | **One AP→CPU message** | **$N$ angles + one local-energy scalar/AP–RIS** | **Parameter-free circular consensus** |

The table supports three specific distinctions. First, G2 combines learned one-shot active/passive
inference with an explicit multi-RIS coordination message. Second, the message is already a feasible
phase proposal plus an interpretable AP-local reliability proxy, rather than a generic latent feature
or iterative solver state. Third, the CPU has no trainable phase-aggregation parameters. None of these
distinctions implies global optimality, full decentralization, or universal novelty of consensus.

## 6. Claims that are currently defensible

- On the fixed 500k R0 checkpoint, AP-local channel energy replaces and exceeds the learned pair
  scale while using the same $N+1$-scalar message.
- The energy weight is strictly local conditional on the shared association mask; the phase proposal
  still uses the broader paper-decentralized CSI view.
- In the fixed-topology seed-0 screen, G2-150k exceeds R0-500k in continuous and rounded 2-bit
  paper-decentralized rate.
- G2 removes trainable CPU-side phase aggregation and reduces the nominal fp32 AP-to-CPU payload
  from $4N$ to $N+1$ values per AP–RIS pair.
- The strongest observed architectural result is the G0→G1/G2 node-free link-token change, not an
  isolated benefit from the G2 context term.

## 7. Claims that are not yet defensible

- “Fully decentralized”: consensus is still performed by a CPU/controller.
- “Globally optimal” or “converged”: training and greedy phase search are non-convex and single-seed.
- “Generalizes across topology, AP/RIS count, or $N$”: none of these axes has been tested for G2.
- “Novel weighted consensus”: close primitives already exist.
- “Proven bits–rate advantage”: E07 remains provisional and undersamples important phase precisions.
- “Energy causes the improvement”: locality and mechanism controls are strong, but no retraining
  intervention isolates causality.
- “Better than optimization-based decentralized methods”: no external method has yet been reproduced
  under the same channel samples and signaling accounting.

## 8. Benchmark plan

### 8.1 Decision to be made

The next experiment should test whether G2 remains attractive against a small set of recognizable
learning and model-based baselines before extending the problem to load balancing, AP/UE service
bounds, or learned association. Those extensions change the feasible set and objective, so they
cannot answer whether the current G2 gain is merely a weak-baseline effect.

The benchmark has three questions:

1. Does G2 outperform a non-graph DNN and simple model-based designs under the current system model?
2. How much performance is lost when each method moves from centralized to its decentralized
   information/coordination mode?
3. Does a frozen G2 policy retain its advantage on deterministic topologies not seen in training?

Multi-seed training is explicitly deferred for this screening stage. Different topologies are stress
conditions, not substitutes for independent training seeds, and their results must be reported
separately.

### 8.2 Baselines, listed one by one

1. **G2-150k (proposed).** Use the frozen E06 finalist. Report centralized,
   paper-decentralized, own-only diagnostic, and rounded 2-bit inference. Its deployed AP→CPU payload
   is $RL(N+1)$ real values and one message round.
2. **R0-500k (closest paper GNN).** Use the mature E01/E06 checkpoint. This is the faithful anchor
   baseline with the same task and simulator, not a reimplementation from a different paper. Report
   the same input modes and its $4RNL$-real AP→CPU payload.
3. **G1-40k (representation ablation).** Reuse E06 rather than retraining. It isolates the G2
   per-RIS context at the matched 40k milestone; it is an ablation, not an external competitor.
4. **Parameter-matched conventional DNN.** Following the DNN control used by Chan et al., replace
   graph message passing with plain MLP blocks while retaining the same input tensors, output
   constraints, unsupervised sum-rate loss, and approximately matched parameter count. Evaluate one
   centralized-input DNN that emits all actions and one AP-shared DNN under the same
   paper-decentralized input view and energy-consensus interface as G2. Treat them as a paired DNN
   family. This tests whether G2's graph/link inductive bias matters; it is not presented as a
   faithful reproduction of Hojatian et al.
5. **Local MRT with random RIS.** Each AP forms maximum-ratio transmission from its direct local
   channel with equal power; each held-out sample receives a random RIS phase vector from the same
   continuous or 2-bit alphabet under one fixed random seed shared by all comparisons. This is the
   transparent low-complexity floor and requires no training.
6. **Local MRT with coordinate-descent RIS.** Keep the same direct-channel local active beamformer,
   but optimize RIS phases using the maintained full-CSI greedy coordinate search. This is an
   analysis oracle for passive-beamforming headroom, not a deployable decentralized method.
7. **Centralized AO and partially distributed AO pair.** Adapt one mathematically compatible pair
   from Ni et al.: centralized joint active/passive optimization and its AP-local-active/CPU-passive
   variant. Use the same initialization, stopping tolerance, and maximum iterations. If the published
   update equations cannot be mapped to the current channel and power model without changing the
   problem, document the mismatch and use Xu et al.'s D-ADMM pair instead; do not silently invent a
   hybrid algorithm.

Chen et al., Huang et al., Zhang et al., and Katsanos–Alexandropoulos remain literature comparators,
but reproducing all of them is unnecessary for the first benchmark. One external optimization pair,
one plain DNN family, the closest GNN, and transparent heuristic floors are sufficient to expose a
weak-baseline problem.

### 8.3 Experimental matrix

Use a staged matrix to avoid a broad sweep:

| Stage | Topology | Methods | Phase settings | Purpose |
|---|---|---|---|---|
| A: same-system screen | T0 canonical training topology | All seven baseline groups | Continuous and rounded 2-bit | Establish rate, gap, payload, and runtime on the existing task |
| B: topology stress test | T1 deterministic jittered layout; T2 deterministic asymmetric/clustered layout | Frozen G2, R0, the best DNN, and the strongest model-based pair | Continuous; 2-bit is included because it is evaluation-only | Test out-of-topology transfer without new training seeds |
| C: conditional follow-up | Only if frozen learned methods all fail on T1/T2 | G2 and strongest DNN with topology-randomized training, seed 0 | Continuous first | Separate architecture failure from a fixed-topology training-distribution failure |

T1 should perturb AP/RIS radii and angles while preserving $L$, $R$, $M$, $K$, coverage area, and
channel law. T2 should use one fixed non-symmetric placement with nonuniform AP/RIS density. Store
both sets of coordinates as machine-readable controls; do not generate a new layout for each method.
A global rotation of the existing rings is not sufficient because it may preserve the same distance
structure.

Stage C is not automatic. It changes the training distribution and should run only when Stage B
shows that the frozen-policy comparison cannot answer the architecture question.

### 8.4 Evaluation protocol and fairness

- Use training seed 0 only. For newly trained DNN arms, use the same optimizer, update budget, batch
  size, and loss as G2; width may be adjusted once to reach within 10% of G2's parameter count.
- Use 400 paired held-out channel samples per topology and a fixed evaluation seed. Every method must
  see the same channel tensors, association mask, power/noise settings, and quantization rule.
- Keep the current $L=5$ APs, $M=2$ antennas/AP, $R=4$ RISs, $N=30$ elements/RIS, and $K=8$
  users setting. Do not repeat the historical antenna and power sweeps in E08 during this screen.
- For G2 and R0, report same-checkpoint centralized rate $R_{\rm cen}$, paper-decentralized rate
  $R_{\rm dec}$, gap $R_{\rm cen}-R_{\rm dec}$, and retention $R_{\rm dec}/R_{\rm cen}$. For the DNN
  and optimization pairs, state explicitly that the centralized/decentralized gap also includes an
  architecture or algorithm difference, not only an information difference.
- For iterative methods, count total transmitted scalars or bits across all online rounds, not only
  one message. State whether CSI acquisition is excluded or included for every arm.
- Measure batch-1 inference/solve time after warm-up on one named device. Report median and 95th
  percentile over the paired samples, plus iteration count for AO/ADMM methods. Runtime is secondary
  to rate and signaling because implementations may have different optimization levels.
- Use a single pre-declared initialization for the main AO result. Check two additional deterministic
  initializations on a small subset only; do not turn initialization sensitivity into a full sweep.
- Do not pool T0/T1/T2 samples to manufacture a narrow confidence interval. Provide paired intervals
  within each topology and display topology-to-topology variation directly.

### 8.5 Required outputs

The primary table should contain, for every retained method and topology:

| Method | $R_{\rm cen}$ | $R_{\rm dec}$ | Cen−dec gap | Dec./cen retention | 2-bit $R_{\rm dec}$ | AP→CPU values or bits | Online rounds | Batch-1 time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

The minimum plots are: decentralized sum rate by method with paired intervals; centralized-to-
decentralized gap; and rate versus total online AP→CPU payload. A topology panel is useful only for
the shortlisted Stage-B methods. Do not add load-balancing, UE-tail-rate, AP/RIS-count scaling, codec
frontiers, or large hyperparameter sweeps to this experiment.

### 8.6 Stop rule and experiment registration

After Stage A, retain at most G2, R0, the stronger DNN, and the strongest model-based pair for Stage B.
The screen is sufficient if it determines whether G2's advantage survives a conventional DNN, a
recognizable distributed optimizer, and two topology shifts. Multi-seed confirmation becomes the
next step only after that method set and claim are frozen.

No new E-number is assigned by this plan alone. Before implementation, confirm whether the benchmark
is registered as the next experiment or attached to an existing question; then place artifacts and
the detailed report under the matching E-folder. If load balancing or AP/UE service-count bounds are
pursued later, they should be a separate research question and experiment because they alter the
optimization target.

## 9. Evidence needed before paper-level claims

1. Execute the minimal Stage-A benchmark and one compatible centralized/distributed optimization
   pair under paired channel samples.
2. Run the Stage-B deterministic topology stress test for the shortlisted methods.
3. Resolve E07's identity control and fill its 4/5/6/8-bit phase grid before any compression claim.
4. Report UE→AP, AP→CPU, and CPU→RIS signaling separately.
5. Add multi-seed training only after the baseline set, topology protocol, and final claim are frozen.
