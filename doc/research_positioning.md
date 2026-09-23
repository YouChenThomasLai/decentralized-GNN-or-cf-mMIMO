# Research positioning and literature notes

## 1. Status

- Updated: 2026-09-21
- Scope: contribution boundary for the G2 graph representation, local-energy consensus, and the
  AP-to-CPU RIS message interface
- Methods: [Method report](./decentralized_ris_methods.md)
- Evidence: [Experiment index](./decentralized_ris_experiments.md)
- Local literature anchors: the decentralized multi-RIS GNN paper [1] and the load-balanced
  attention GNN paper [2]; both PDFs are linked from Section 10.
- Verification: the works listed in Section 10 were checked against the local PDFs, arXiv full text,
  or publisher/Crossref metadata. The search is representative rather than a systematic review.
- Method hierarchy: G2 is the proposed method and sole learned finalist; G1 is retained only as the
  matched no-context ablation for contribution attribution.

## 2. Supported positioning

G2 is best positioned as a state/action-factorized inference architecture for joint active and passive
beamforming with a bounded coordination interface. Its system-specific combination is:

1. each AP produces its active beamformer and a feasible per-RIS phase proposal under the
   paper-decentralized CSI view;
2. each AP attaches one strictly local channel-energy scalar to each RIS proposal;
3. the CPU performs parameter-free circular consensus rather than a learned feature reduction or an
   online iterative optimization;
4. a per-sample $R\times LK\times q$ link memory remains fixed while AP–UE interaction states evolve,
   then participates again in the late local proposal readout; and
5. the AP-to-CPU RIS message contains $N$ phase angles plus one energy scalar, instead of R0's $4N$
   learned features.

E05 verifies the inference-time energy rule and adds a finite-budget weight diagnostic: on the frozen
G2-150k checkpoint the parameter-free energy weight recovers $0.9301\pm0.0047$ of the rate gap between
equal weighting and the best per-sample weight found by a declared two-start Adam search. This is
evidence that energy is a strong proxy within that search, not a global optimality bound. The method
report's [MLE argument](./decentralized_ris_methods.md#energy-consensus) shows that
concentration-weighted circular fusion is the MLE under an independent von Mises model; identifying
concentration with energy remains a proxy assumption. E06 shows that G2 can be trained end to end
and exceeds the mature R0 reference on the fixed holdout. The large empirical separation is between
G2's full representation package and G0; that package includes node-free per-RIS link tokens, a new
encoder/head, context reinjection, and a capacity change. Removing the context in the G1 ablation
causes no detectable primary-metric loss at 40k, although it produces a severe own-only collapse.
G1 therefore limits the context-specific claim but remains an ablation, not an alternative research
track. E12 now supplies a fresh same-system
fixed-budget comparison: the Huang-adapted model-based methods reach higher rate than G2, while G2
uses far fewer coordination bits and rounds. The solver's strict precision and worst-case consensus
checks still fail, so this is a deployed-action trade-off rather than a converged reference.
It compares tested operating points, not a certified Pareto frontier or measured network latency;
G2's shared-UE cross-AP CSI view and the distributed solver's AP-local CSI/iterative aggregate
view also differ. The modeled bit ratios exclude delivery of shared-UE cross-AP CSI needed for G2's
observation, so they do not measure total network traffic. [E14](./experiments/e14_topology_stress.md)
adds one explicit delivery route: direct fp32 AP→AP unicast. Under that route, the T0 G2 input
delivery plus proposal payload averages 973,514 bits/sample, above the centralized arm's 624,640;
distributed ADMM still uses about 47.9× more bits than G2. E14 also retains G2's paired advantage
over R0 on two fixed topology shifts, while both fixed-budget model-based arms remain higher in rate.
Its visibility pair adds the size boundary: across AP ring radii whose visible-link fraction runs
0.2860 to 0.6170, G2 keeps the smaller relative decentralization gap everywhere, but its paired
advantage falls from $+3.6939$ to $+0.4846$ bps/Hz while modeled CSI delivery rises from 0.99× to
2.35× the centralized payload. A rate-matched pair then separates visibility from operating level:
with holdout centralized rates agreeing within 0.8274 bps/Hz, a 0.289 drop in visible fraction
costs G2 $+1.44$ pp of relative gap and R0 $+4.34$ pp.
These transport counts are conditional scenarios, not measured network traffic.

## 3. What is not a contribution by itself

- **R0→R0c commutation:** moving a shared affine reduction from after the AP sum to before it is
  algebraically exact and should be presented as a control, not novelty.
- **Weighted averaging:** scalar sender–receiver weights and parameter-free aggregation already have
  close precedents in multi-agent communication, distributed GNNs, and consensus optimization.
- **Per-AP phase vectors reconciled by consensus:** Huang et al. [7] already have every BS compute a
  local IRS phase vector and impose an explicit consensus constraint across BSs, and Xu et al. [8]
  design the RIS reflection coefficients separately at each BS before reconciling them. Producing a
  full phase vector at each AP and reconciling the copies is therefore established prior art. What is
  specific to G2 is the single-round, $N+1$-scalar, energy-weighted, parameter-free form of that
  reconciliation, not the idea of AP-side phase proposals.
- **Multi-RIS operation:** most of the closest comparators already configure more than one surface,
  so a multi-RIS system model is a precondition of the problem rather than a contribution.
- **RIS phase compression:** prior work already studies low-bit RIS control and task-aware compression.
- **Link or edge features in isolation:** channel-conditioned edges and link-aware message passing are
  established wireless-GNN choices [23]–[30]. That prior art rules out claiming link information or
  edge initialization alone. It does not reproduce G2's exact dataflow: persistent per-RIS link
  memory without recurrent RIS nodes, per-layer context reinjection, and a shared AP-local proposal
  head. Section 4.4 keeps
  that distinction explicit, while E06 measures the whole representation package rather than an
  isolated link-token intervention.
- **A small post-hoc rate gain:** E04's $+0.0419$ bps/Hz fixed-checkpoint difference is not evidence
  that learned pair magnitude is superior.
- **Executable-phase messages as a compression advantage:** E07's matched-budget run shows that on a
  fixed R0 backbone no advantage is detected for the executable-phase interface over a same-bit
  Cartesian vector quantizer on the commuted logits ($+0.0823\pm0.0579$, and $+0.0011\pm0.0300$ on a
  fresh seed), and that G2 loses *more* rate to message compression than R0 does
  ($-0.6629\pm0.1191$). The message being directly executable is therefore not, by itself, a
  compression contribution. What it does beat is R0's own $4N$ latent message at a small budget, and
  even that advantage closes by 250–300 bits per AP–RIS pair.

## 4. Related-work landscape

### 4.1 Centralized and partially distributed model-based design

Model-based RIS-aided cell-free methods commonly formulate joint active/passive beamforming as a
non-convex weighted sum-rate or sum-MSE problem and solve it by alternating optimization (AO), block
coordinate descent, or successive approximation. These methods provide strong numerical references,
but their per-channel online iterations and global state exchange are different from amortized neural
inference. Ni et al. [3] place active beamforming at the APs while retaining CPU-side passive
beamforming inside an iterative AO loop. Zhang et al. [4] use a closely related multi-RIS variable
set—$L$ APs, $R$ RISs, $K$ multi-antenna users, active precoders, receive combiners, and passive
beamforming—but not G2's information pattern: its APs forward CSI and precoders to the CPU, receive
cross-term information, and the CPU updates the receive combiners and RIS phases. These methods are
therefore useful performance and signaling baselines rather than prior instances of G2's one-shot
proposal interface.

Recent work also reduces overhead by changing the optimization problem. Xu et al. [5] jointly select
AP clusters and beamformers, while Ke et al. [6] introduce RIS–UE association to avoid acquiring
irrelevant channels. These papers motivate future association or load-balancing extensions, but
adding those variables now would confound the narrower question of whether G2 is already competitive
for the current fixed-association problem.

### 4.2 Iterative decentralized optimization

Decentralized optimization removes or reduces centralized computation by exchanging local copies,
dual variables, or consensus states. Huang et al. [7] use incremental ADMM for IRS-enhanced cell-free
beamforming and approach the centralized solution as the number of online iterations grows; its model
has $B$ BSs, $R$ IRSs, and $K$ users under per-BS power constraints, and each BS computes its own
local IRS phase vector that the algorithm drives to agreement through an explicit consensus
constraint. Xu et al. [8] unroll a distributed ADMM solver and use directed information exchange,
likewise designing the RIS reflection coefficients separately at each BS and then reconciling them.
These two are the direct precedents for AP-side phase proposals, and the contribution statement must
not claim that idea. Katsanos and
Alexandropoulos [9] consider a wider multi-RIS, wideband, imperfect-CSI setting and coordinate active
and passive beamforming through consensus updates. Du et al. [10] instead study an active
sub-connected RIS and use a two-stage distributed iterative design. These are the closest algorithmic
precedents for distributed RIS coordination. Their strength is structured optimization; their
deployment cost is iteration-dependent signaling and latency. G2 instead pays a fixed one-message
AP→CPU cost and does not claim the convergence guarantees of those solvers.

### 4.3 Learning-based distributed and graph-based design

Hojatian et al. [11] establish that unsupervised DNNs can move cell-free beamforming to APs with zero
or limited AP–controller exchange, but do not jointly coordinate multiple RISs. Chen et al. [12]
learn local BS beamformers and let one BS determine the IRS coefficients; this avoids CPU-heavy
iterative optimization, but centralizes passive control at a designated BS and studies a different
interface. Zhu et al. [13] use centralized-training/distributed-execution FL-MADDPG, where FL denotes
fuzzy logic, for multi-RIS cell-free precoding. APs are the agents, their actor network selects joint
precoding/phase actions, and the CPU physically controls the RISs, but the paper does not expose a
per-AP RIS payload or a rule for reconciling multiple AP-side phase actions.
The local anchor paper [1] is the closest GNN baseline: it jointly learns AP beamforming and
multi-RIS control, sends $4N$ learned features per AP–RIS pair, and uses a trainable CPU-side linear
reducer.

Chan et al. [2] show how an introduction and comparison table can isolate a GNN paper's actual problem
dimensions. Their adaptive-attention GNN jointly handles user association, load balancing,
beamforming, and RIS phases and generalizes across BS/UE counts. It is a strong precedent for graph
generalization and for future load-balancing work, but it is not a decentralized AP→CPU coordination
method for a multi-RIS cell-free network. G2's narrower distinction is the co-design of the local graph
representation and a single-round, interpretable proposal-and-weight message.

### 4.4 Graph architectures for wireless resource allocation

Graph modeling for wireless resource allocation is an established design space. Shen et al. [23]
place each transceiver pair on a node and each interference channel on an edge, Eisen and Ribeiro
[24] convolve over fading-induced random graphs, and Chowdhury et al. [25] parameterize unfolded
WMMSE weights with a permutation-equivariant GNN. Shen et al. [26] provide the broader graph-modeling
and architecture framework. These works establish channel-aware edges, message passing, and
permutation structure; they do not by themselves establish G2's exact link-memory dataflow.

The RIS-specific literature narrows the claim further. Yin et al. [27] introduce an edge-initialized
GNN for joint BS beamforming and RIS phase design but retain RIS and UE nodes in a heterogeneous
graph. Lim and Vu [28] use a distributed BS–user bipartite GNN for multi-RIS multi-cell association
and beamforming, with composite reflected-channel estimation between layers. Their earlier
multi-RIS single-BS design [30] is a direct edge-information precedent: cascaded RIS–UE channels
are edge features in a RIS–UE bipartite graph, while both node types undergo recurrent
bidirectional message passing and the RIS node outputs the phase. Nguyen et al. [29]
design a cell-free heterogeneous GNN that exploits detailed AP/UE/RIS link information for uplink
power and RIS-phase control. Thus edge initialization, detailed link information, distributed graph
execution, and joint active/passive outputs are not individually new.

Against those precedents, G2 implements a more specific
[state/action factorization](./decentralized_ris_methods.md#state-action-factorization). It removes
recurrent RIS states, recomputes one embedding per (RIS, AP–UE link) for every channel sample,
preserves the resulting $R\times LK\times q$ tensor as fixed link memory while only AP–UE states evolve,
reinjects a fixed channel-weighted summary into every AP–UE update layer, and reads the same memory
again through a shared masked head to emit executable AP-local RIS proposals. G1 removes only the
context reinjection and is kept as the matched ablation. None of the reviewed works [23]–[30]
reproduces G2's complete dataflow, so the exact G2 architecture may be claimed as this work's design,
while avoiding a broader “first edge/link GNN” claim. Because the search is
representative rather than systematic, any first-in-literature wording still requires a qualified
“to our knowledge.”

The E06 evidence boundary remains important: G2 versus G0 changes the encoder, phase head, removal
of RIS nodes, context, and effective parameter count together. The measured gain therefore belongs
to the representation package, not to link memory alone. The G1 no-context ablation isolates context
reinjection more cleanly but shows no detectable primary-metric gain at 40k, so the G2 context should
remain a secondary design detail rather than a standalone contribution.

### 4.5 Supporting communication and compression precedents

The proposal interface should also be discussed conservatively against adjacent communication work.
Over-the-air GNN aggregation, targeted multi-agent communication, task-oriented feature compression,
and learned distributed source coding all show that low-dimensional or weighted messages are not new
in isolation. RIS control-signaling and phase-compression studies further motivate counting actual
bits instead of only tensor dimensions. The relevant precedents are Gu et al. [14], Das et al. [15],
Shao et al. [16], [17], Sohrabi et al. [18], Fernandes and Psaromiligkos [19], and Enqvist et
al. [20]. Guo et al.'s two-way RIS design [21] is a more specific precedent for phase averaging.

## 5. Comparison with related distributed and GNN designs

Both tables below are feature maps, not performance rankings, because the works use different channel
models, objectives, and information assumptions. Section 5.1 follows the role of Table I in Chan et
al. [2] and gives the compact per-property summary; Section 5.2 expands the coordination interface,
which is the dimension the contribution actually rests on.

### 5.1 Summary of related works

Rows are ordered by the number of ✓ marks, from fewer to more; ties are ordered by publication year.
This ordering is only a reading aid, not a score. Here, `N/A` means that the criterion is not applicable
because the work has no AP-originated RIS phase-proposal/fusion stage; ✗ means that the criterion is
applicable but the work does not meet it.

| Reference | Problem | Method | Multi-RIS | RIS variable/action at every AP | Non-iterative online action | Explicit one-round AP→CPU RIS interface | Executable phase-proposal payload | Parameter-free CPU fusion |
|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Ni et al. 2022 [3] | AP beamforming, RIS phase shifts | AO | ✗ | ✗ | ✗ | ✗ | N/A | N/A |
| Hojatian et al. 2022 [11] | AP beamforming only (no RIS) | DNN | N/A | N/A | ✓ | N/A | N/A | N/A |
| Chen et al. 2023 [12] | AP beamforming, RIS phase shifts | GNN | ✗ | ✗ | ✓ | N/A | N/A | N/A |
| Zhang et al. 2023 [4] | AP beamforming, receive combining, RIS phase shifts | AO | ✓ | ✗ | ✗ | ✗ | N/A | N/A |
| Chan et al. 2026 [2] | BS beamforming, RIS phase shifts, BS–UE association | GNN | ✗ | ✗ | ✓ | N/A | N/A | N/A |
| Huang et al. 2021 [7] | AP beamforming, RIS phase shifts | ADMM | ✓ | ✓ | ✗ | ✗ | N/A | N/A |
| Xu et al. 2023 [8] | AP beamforming, RIS phase shifts | Unrolled ADMM | ✓ | ✓ | ✗ | ✗ | N/A | N/A |
| Katsanos and Alexandropoulos 2026 [9] | AP beamforming, RIS element tuning (wideband, imperfect CSI) | Consensus optimization | ✓ | ✓ | ✗ | ✗ | N/A | N/A |
| Zhu et al. 2024 [13] | AP beamforming, RIS phase shifts | FL-MADDPG | ✓ | ✓ | ✓ | ✗ | N/A | N/A |
| Ting et al. anchor [1] | AP beamforming, RIS phase shifts | GNN | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| **This work (G2)** | **AP beamforming, RIS phase shifts** | **GNN** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

Column criteria, so that every mark is auditable:

- **Problem column vocabulary:** the cited works call the same AP-side variable *active
  beamforming*, *precoding*, or *transmit beamforming*, and the same RIS-side variable *passive
  beamforming*, *RIS reflection coefficients*, or *RIS phase shifts*. The column normalizes these to
  *AP beamforming* and *RIS phase shifts* so that two rows read differently only when the optimization
  variables genuinely differ, as with Zhang et al. [4]'s receive combining vector and Chan et
  al. [2]'s BS–UE association. Katsanos and Alexandropoulos [9] instead optimize the tunable
  capacitors that induce frequency-dependent RIS responses. Chan et al. [2] keeps *BS* because its
  system is multi-BS cellular rather than cell-free. Each work's own wording is preserved in Section
  4 and Section 5.2.
- **Multi-RIS:** the system model contains more than one RIS whose phases must be configured jointly.
  Every mark in this column was read from the paper's own system model, not from its abstract, because
  abstracts in this literature use a generic singular “RIS” even when the model defines $R > 1$
  surfaces. Huang et al. [7] defines a set of IRSs $\mathcal{R} = \{1,\dots,R\}$, Zhang et al. [4]
  states “$L$ APs, $R$ RISs, and $K$ multi-antenna users,” and Xu et al. [8] deploys “multiple
  energy-efficient RISs.” Ni et al. [3] (“RIS consists of $M$ reflecting elements”) and Chen et
  al. [12] (“the IRS has $L$ passive reflecting elements”) are single-surface.
- **RIS variable/action at every AP:** every AP/BS forms an RIS-related local copy, estimate, or
  action rather than contributing only its active beamformer. Huang et al. [7], Xu et al. [8], and
  Katsanos and Alexandropoulos [9] meet this criterion through distributed local RIS variables;
  Zhu et al. [13] treat APs as agents whose actors select joint precoding/phase actions. Ni et al.
  [3] and Zhang et al. [4] are ✗ because the CPU solves the passive subproblem. Chen et al. [12] are
  also ✗ because only BS 1 determines the passive vector. This is narrower and more relevant than
  generic “decentralized AP-side inference.”
- **Non-iterative online action:** a trained neural readout or actor directly emits the online action,
  with no per-channel AO/ADMM/consensus loop or layer-wise inter-AP exchange. Zhu et al. [13] meet
  this deployment criterion even though its original training problem is a temporal MDP with
  $\gamma=0.99$. The abandoned one-step E11 design was only a controlled benchmark proposal, not a
  faithful reproduction, and produced no valid result.
- **Explicit one-round AP→CPU RIS interface:** the method specifies one bounded AP-originated RIS
  message and one upstream round before the CPU fixes the shared RIS action. Ting et al. [1] and G2
  meet this criterion. Iterative peer-to-peer solver exchanges do not, and Zhu et al. [13] does not
  expose the AP→CPU payload or reconciliation rule even though the CPU controls the RIS hardware.
- **Executable phase-proposal payload:** the AP→CPU message itself contains a feasible per-element
  phase proposal, rather than latent features or an iterative solver state. G2 transmits $N$ angles
  plus one local-energy scalar. Ting et al. [1] is ✗ because its $4N$ values are learned phase features
  that still require the CPU's trainable linear reducer and normalization.
- **Parameter-free CPU fusion:** the controller combines AP messages with a fixed rule that has no
  trainable parameters. Ting et al. [1] is ✗ because of its trainable reducer. `N/A` in this
  or the preceding column means that the work has no applicable AP→CPU proposal/fusion interface,
  not that a parameter-free alternative was tested and lost.

Two readings of this table are wrong and should be avoided. First, multi-RIS operation does not
separate this work: six of the ten comparators already configure more than one surface, so it is a
precondition of the problem rather than a contribution, and it is listed only because a reader
comparing against single-RIS works such as Ni et al. [3] or Chan et al. [2] needs to see which rows
solve the same problem. The sharper separation comes from the last three columns. Second, ✓ is not
uniformly better: the iterative methods buy convergence behavior with their extra rounds, which is
exactly the trade-off Section 4.2 describes. “Fixed-size” is deliberately not a column: for fixed
$N$, both Ting et al.'s $4N$ latent vector and G2's $N+1$ message have fixed dimension. The meaningful
distinctions are whether the interface takes one round, whether its payload is directly executable,
and whether its fusion rule is trainable; Section 5.2 reports the actual payload size.

### 5.2 Coordination-interface comparison

This table expands the message and fusion columns of Section 5.1. “One-shot” again means no
per-channel AO/ADMM loop at deployment time.

| Work | RIS setting | AP-side active design | Deployment-time coordination | RIS-coordination message/state | Final RIS decision |
|---|---|---|---|---|---|
| Ni et al. 2022 [3] | Single-RIS cell-free | Local AO update | AP–CPU AO rounds | Active variables and CSI; passive state remains at CPU | CPU passive optimization |
| Hojatian et al. 2022 [11] | Cell-free, no RIS | Direct local DNN | Zero or limited AP–NC exchange | Not applicable | Not applicable |
| Chen et al. 2023 [12] | Single-IRS cell-free | Direct local GNNs | BS 1 separately controls the IRS | No common proposal interface | Single-BS passive output |
| Zhang et al. 2023 [4] | Multi-RIS cell-free | Local AO update | APs send CSI/precoders; CPU returns cross terms and updates | CSI, cross terms, and precoders | CPU passive optimization |
| Chan et al. 2026 [2] | Single-RIS multi-BS cellular | Joint GNN output | One joint graph-inference pass | No AP→CPU proposal interface | Direct joint-model RIS output |
| Huang et al. 2021 [7] | Multi-IRS cell-free | Local iterative update | Sequential incremental ADMM among BSs | Solver and consensus state | Distributed consensus solution |
| Xu et al. 2023 [8] | Multi-RIS cell-free | Distributed unrolled solver | $D^2$-ADMM block/layer exchanges | Directed solver states | Learned iterative consensus |
| Katsanos and Alexandropoulos 2026 [9] | Wideband multi-RIS cell-free | Local cooperative optimization | Iterative consensus among BSs | Local variables and neighbor updates | Distributed consensus solution |
| Zhu et al. 2024 [13] | Multi-RIS cell-free | Direct FL-MADDPG actor | Actor execution per slot; CPU controls RIS hardware | AP→CPU RIS payload/reconciliation not specified | CPU applies the shared action; fusion unspecified |
| Ting et al. anchor [1] | Multi-RIS cell-free | One-shot paper-decentralized GNN | One AP→CPU message | $4N$ learned features/AP–RIS | Trainable linear CPU reducer |
| **G2 (this project)** | **Multi-RIS cell-free** | **One-shot paper-decentralized GNN** | **One AP→CPU message** | **$N$ angles + one local-energy scalar/AP–RIS** | **Parameter-free circular consensus** |

The table supports three specific distinctions. First, G2 combines learned one-shot active/passive
inference with an explicit multi-RIS coordination message. Second, the message is already a feasible
phase proposal plus an interpretable AP-local importance weight, rather than a generic latent feature
or iterative solver state. Third, the CPU has no trainable phase-aggregation parameters. None of these
distinctions implies global optimality, full decentralization, or universal novelty of consensus.

## 6. Claims that are currently defensible

- On the fixed 500k R0 checkpoint, AP-local channel energy replaces and exceeds the learned pair
  scale while using the same $N+1$-scalar message.
- The energy weight is strictly local conditional on the shared association mask; the phase proposal
  still uses the broader paper-decentralized CSI view.
- On the frozen G2-150k checkpoint and the shared 400-sample holdout, the parameter-free energy
  weight recovers $0.9301\pm0.0047$ of the equal-to-best-found rate gap under the declared finite
  search and has directional rank agreement with the fitted weights. The search improves energy by
  $1.2544\pm0.0960$ bps/Hz. It neither bounds the unknown global weight optimum nor says anything
  about joint retraining, because the proposals are frozen.
- In the fixed-topology seed-0 screen, G2-150k exceeds R0-500k in continuous and rounded 2-bit
  paper-decentralized rate.
- On fixed T1 and T2 geometry shifts, the frozen G2-150k checkpoint still exceeds R0-500k in
  paired paper-decentralized rate under continuous and 2-bit actuation; this is a two-layout
  stress result, not distributional topology generalization (E14).
- G2's decentralized advantage in retention is largest where each AP sees the smallest share of the
  shared-UE links. On E14's first pre-registered visibility pair the paired rate advantage is
  $+3.6939$ bps/Hz at a 0.2860 visible-link fraction and $+0.4846$ bps/Hz at 0.6170, positive
  throughout, but visibility and operating SNR both track AP ring radius there.
- At a matched operating point, reduced visibility costs rate on its own. On E14's rate-matched
  pair, whose holdout centralized rates agree within 0.8274 bps/Hz, a 0.289 lower visible fraction
  raises G2's relative decentralization gap by $2.06$ pp $[1.66,2.46]$ and R0's by $3.59$ pp
  $[2.70,4.48]$ on the confirmation seed. G2 therefore tolerates reduced visibility well, at 2.39%
  against R0's 10.81% of their own centralized rates, but it is not indifferent to it.
- G2 degrades less than R0 as visibility falls, by $+1.53$ pp $[+0.60,+2.45]$ of relative gap. This
  contrast was marginal on its discovery seed and was confirmed once on a fresh seed with 3,200
  samples per layout under the rule fixed beforehand.
- G2 removes trainable CPU-side phase aggregation and reduces the nominal fp32 AP-to-CPU payload
  from $4N$ to $N+1$ values per AP–RIS pair.
- The strongest observed architectural result is G2's full representation-package advantage over
  G0, not an isolated benefit from the context term; G1 is the ablation that establishes this bound.
- In the reviewed literature, G2's exact state/action-factorized dataflow is distinct from prior
  edge-initialized or link-aware GNNs; the defensible contribution is the complete architecture and
  proposal-interface co-design, not link features by themselves.
- The weighted chordal-consensus objective has a coordinatewise closed form when its resultant is
  nonzero and gives AP-order invariance, common-rotation equivariance, and feasible unit-modulus
  output under that condition. These are method properties, not standalone novelty or rate guarantees.
- On the fresh paired T0 seed 20260920, G2-150k retains 79.3% of the Huang-adapted centralized
  fixed-budget rate and 81.9% of its distributed rate, while using 31.5× and 2,348.6× fewer
  modeled coordination bits with one upstream round instead of two and 1,001. These ratios exclude
  G2's shared-UE cross-AP CSI delivery. All eight deployed-action feasibility gates pass; these
  are finite-budget implementation results, not upper bounds (E12).
- Under E14's explicit direct fp32 AP→AP unicast route for the shared-UE CSI, G2's modeled total is
  973,514/926,740/980,173 bits on T0/T1/T2. It exceeds the centralized arm's 624,640 bits on
  every layout and remains 47.5–50.3× below distributed ADMM. Direct delivery adds one sequential
  round before G2's proposal, bringing it to two. This conditional route accounting replaces any
  unqualified whole-network signaling advantage over centralized optimization.
- At a matched AP-to-CPU budget of 68 bits per AP–RIS pair, G2 has a positive paired contrast against the best R0-family wire format
  at the same or lower budget by $1.5424\pm0.2363$ bps/Hz on the shared holdout and
  $1.5003\pm0.2472$ on a fresh confirmation seed, at one fusion round for both (E07). The margin
  grows in point estimate with the budget toward the $2.2053$ bps/Hz uncompressed G2–R0 gap.
  This is consistent with the representation-package difference surviving compression, not an
  isolated advantage of the codec. E07's global grid-legality control failed, so the wider
  frontier remains descriptive.
- Simply deleting the local-energy scalar from frozen G2 and fusing with equal weights collapses
  its rate to 6.72–8.30 bps/Hz at every phase precision. This is a zero-training sensitivity result,
  not a lower bound on the payload of a retrained policy.
- Under G2's own fused RIS phase, its learned active beamformer exceeds RIS-aware local MRT by
  $2.8848\pm0.1685$ and RIS-aware local RZF by $4.7522\pm0.4235$ bps/Hz on the fixed holdout, and
  the margin survives replacing G2's learned per-AP power with full power (E13). This is a component
  control against conventional local precoding, not a comparison against a complete model-based
  design.

## 7. Claims that are not yet defensible

- “Fully decentralized”: consensus is still performed by a CPU/controller.
- “Globally optimal” or “converged”: training and greedy phase search are non-convex and single-seed.
- “Generalizes across arbitrary topology, AP/RIS count, or $N$”: E14 tests only two designed layout
  shifts plus a three-layout visibility pair from one ring family, all at the original AP/RIS
  counts and $N$; no topology distribution or size scaling is tested.
- “G2 degrades far less than R0 as visibility falls”: the confirmed difference is $+1.53$ pp of
  relative gap, not an order-of-magnitude difference in sensitivity; state the size, not a
  qualitative contrast.
- “G2 is insensitive to CSI visibility”: its relative gap is about six times larger on the
  low-visibility member of the rate-matched pair, and the same direction appears in the first pair.
- “Novel weighted consensus”: close primitives already exist.
- “Manifold-consensus closed form or symmetry as independent novelty”: these are properties of an
  established weighted circular rule; [the propositions](./decentralized_ris_methods.md#energy-consensus)
  clarify the interface, subject to nonzero resultant and implementation fallback.
- “G2's message compresses better than the anchor's”: E07 measures the opposite sign for the
  compression penalty, and the interface ties a same-bit Cartesian codec on a fixed backbone.
- “Fewer bits and still better than the uncompressed anchor”: at 68 bits against the anchor's 2400
  generous bits, the paired difference is $+0.1698\ [-0.3603,+0.6999]$ under the declared continuous
  actuation. The same contrast is clearly positive under 2-bit RIS actuation, but that readout was
  declared secondary and must be re-declared and re-confirmed before it can carry a claim.
- “A certified bits–rate frontier at high phase precision”: E07's grid-legality control failed for
  $b_p\ge6$ and was not re-run.
- “Energy causes the improvement”: locality, mechanism, and fitted-weight controls are strong, but no
  retraining intervention isolates causality and the weight search reuses frozen proposals.
- “Energy is the optimal weight”: the MLE argument holds under assumed von Mises, independent
  proposal errors with concentration proportional to $E_{l,r}$; those assumptions are not tested, and
  the finite-budget Adam reference is not a certified global optimum.
- “Higher rate than optimization-based decentralized methods”: the fresh paired E12 comparison shows
  the opposite ordering, with a G2 deficit of $5.0583\pm0.2265$ bps/Hz against the distributed
  Huang adaptation. It supports the rate–signaling–rounds trade-off instead.
- “Converged or precision-stable Huang reproduction”: E12's fixed-budget implementation is feasible,
  but the new seed fails the strict precision and maximum copy-consensus gates.

## 8. Benchmark plan

### 8.1 Decision to be made

The next experiment should test whether G2 remains attractive against a small set of recognizable
learning and model-based baselines before extending the problem to load balancing, AP/UE service
bounds, or learned association. Those extensions change the feasible set and objective, so they
cannot answer whether the current G2 gain is merely a weak-baseline effect.

A complete model-based comparison is mandatory for this decision. Fixed-G2-RIS MRT/RZF controls do
not satisfy that requirement because their passive action still comes from the proposed GNN; they
only test whether G2's learned active beamformer is stronger than conventional local precoding under
the same proposed phase.

The benchmark has four questions:

1. Does G2 outperform a non-graph DNN and recognizable model-based designs under the current system
   model? The proposed MADDPG arm was screened out before a valid holdout and is not part of the claim.
2. With the proposed G2 RIS phase fixed, does its learned active beamformer outperform RIS-aware
   local MRT and local RZF?
3. How much performance is lost when each applicable method moves from centralized to its
   decentralized information/coordination mode?
4. Does a frozen G2 policy retain its advantage on deterministic topologies not seen in training?

Multi-seed training is explicitly deferred for this screening stage. Different topologies are stress
conditions, not substitutes for independent training seeds, and their results must be reported
separately.

### 8.2 Comparison roles and methods

The primary rate comparison should contain complete methods that solve both active and passive
beamforming. Representation ablations and controls that reuse G2's RIS output answer narrower
mechanism questions and must be reported separately rather than ranked as independent competitors.

Primary complete-method comparison:

1. **G2-150k (proposed).** Use the frozen E06 finalist. Report centralized,
   paper-decentralized, own-only diagnostic, and rounded 2-bit inference. Its deployed AP→CPU payload
   is $RL(N+1)$ real values and one message round.
2. **R0-500k (closest paper GNN anchor).** Use the mature E01/E06 checkpoint. This is the faithful
   anchor with the same task and simulator, not a reimplementation from a different paper. Report
   the same input modes and its $4RNL$-real AP→CPU payload.
3. **Parameter-matched conventional DNN.** Following the DNN control used by Chan et al. [2], replace
   graph message passing with plain MLP blocks while retaining the same input tensors, output
   constraints, unsupervised sum-rate loss, and approximately matched parameter count. Evaluate one
   centralized-input DNN that emits all actions and one AP-shared DNN under the same
   paper-decentralized input view and energy-consensus interface as G2. Treat them as a paired DNN
   family. This tests whether G2's graph/link inductive bias matters; it is not presented as a
   faithful reproduction of Hojatian et al. [11]. Completed as
   [E10](./experiments/e10_dnn_benchmark.md): at a matched 150k budget and matched capacity the
   decentralized arm reaches 6.8192 paper-decentralized bps/Hz against G2-150k's 24.5358, although
   neither arm had converged at that budget.
4. **Same-interface MADDPG (skipped).** E11 was stopped before final holdout because its trajectory
   was not competitive enough to justify the remaining compute. Review also found that exploration
   could reintroduce energy on association-masked UE columns, invalidating the collected behaviour
   actions. The implementation was removed and the registration retained as provenance. No claim
   about MADDPG follows from this negative screen.
5. **Mandatory model-based family: centralized and distributed joint optimization.** Include both
   of the following complete active/passive designs in the primary table:
   - **Centralized joint-optimization reference:** optimize the AP beamformers and all RIS phases
     iteratively from full CSI under the same fixed association mask, unweighted sum-rate objective,
     per-AP power constraints, and unit-modulus constraints as G2. This is a local-optimization
     reference, not a global upper bound.
   - **Distributed joint-optimization baseline:** prefer the conventional, non-unrolled ADMM
     formulation in Xu et al. [8], because it jointly updates BS precoders and multi-RIS reflection
     vectors and explicitly reconciles BS-local RIS copies. Compare it with the centralized joint
     design used in the same evaluation frame; the learned $D^2$-ADMM unrolling is not required.

   Use the same initialization, stopping tolerance, maximum iteration count, and feasibility checks
   for the two arms, and count every online exchange. If Xu et al.'s equations cannot be mapped
   without changing the objective or constraints, select Huang et al.'s multi-RIS incremental ADMM
   [7] and corresponding centralized formulation before looking at final holdout results. Ni et al.
   [3] is only a second fallback because its original system has one RIS; any multi-RIS extension
   must be labeled as an adaptation rather than a faithful reproduction. Do not silently invent a
   hybrid algorithm or omit the model-based family.

   Registered as [E12](./experiments/e12_model_based_optimization.md). The compatibility check
   against Xu et al. [8] **failed**, though not for the reason this plan anticipated. Their
   objective, per-BS power constraint, unit-modulus constraint, multi-RIS model, direct link and
   coherent joint transmission all map onto the current system; what does not exist is a
   *conventional* version of their algorithm, because their reflection subproblem (their eq. 29) is
   handed to a learned convolutional block and the paper supplies no non-learned solver for it.
   Building "conventional Xu et al." would therefore require importing a solver the paper does not
   contain, which is the hybrid this plan forbids. The declared fallback to Huang et al.'s
   multi-RIS incremental ADMM [7] was used, with the centralized
   arm as the centralized counterpart of the same formulation. Neither paper has an AP–UE
   association mask, so adding one is an extension of the published problem and is recorded as such.

   The corrected run fixes the original power and stopping defects, uses `matched` initialization,
   and recalibrates the penalty on 400 samples. It reaches 31.8026 centralized and 30.4652
   distributed bps/Hz, but **four of thirteen primary gates still fail**: precision, centralized
   monotonicity, worst-case consensus, and divergence. The rates therefore remain diagnostic. Two
   findings are usable: all-ones materially understates this model-based family, and the distributed
   arm removes the CSI upload only by spending about 74.1 times the centralized arm's counted online
   bits over 1,001 rounds instead of two.

   A new [E12 fixed-budget continuation](./experiments/e12_model_based_optimization.md) was declared
   before a fresh seed 20260920. It evaluates the actual feasible deployed actions after the same
   frozen solver budgets and pairs them with G2-150k on identical channels. All eight eligibility
   controls pass. G2 reaches 22.9587 bps/Hz versus 28.9463 centralized and 28.0170 distributed,
   with 19,840 versus 624,640 and 46,595,840 coordination bits per sample, respectively.
   Precision and worst-case copy-consensus still fail the older strict gates. This closes the
   **finite-budget Stage-A screening comparison**, not a claim about ADMM convergence or a
   paper-exact Huang implementation.

Supporting ablations and component controls:

6. **G1-40k representation ablation.** Reuse E06 rather than retraining. It isolates the G2 per-RIS
   context against G2-40k at the matched milestone; it is not an external competitor and belongs in
   the ablation table.
7. **Fixed-G2-RIS active-beamforming controls.** Registered and completed as
   [E13](./experiments/e13_fixed_ris_beamforming.md). Freeze the G2-150k paper-decentralized phase
   proposals and local-energy consensus, then compare three active designs under exactly the same
   final RIS phase: the native G2 beamformer, RIS-aware local MRT, and RIS-aware local RZF. For the
   two conventional controls, first fuse and set the G2 RIS phase, form each AP's effective channels
   to its associated users, and then compute the local beamformer under the same per-AP power
   constraint. Use
   $\mathbf W_l\propto\mathbf H_l^{\rm eff}$ for MRT and
   $\mathbf W_l\propto(\mathbf H_l^{\rm eff}\mathbf H_l^{{\rm eff},H}
   +\lambda_l\mathbf I)^{-1}\mathbf H_l^{\rm eff}$ for RZF, followed by the common power
   normalization. Pre-register $\lambda_l=|\mathcal K_l|\sigma^2/P_l$ in linear units rather than
   tuning it on the holdout. This is labeled RZF: the experiment defines a regularized downlink
   precoder, whereas an MMSE label would require a separately specified
   estimation/signal model and power-weighting convention. RZF is also preferable to unregularized
   local ZF here because $|\mathcal K_l|$ can exceed $M=2$ under the fixed association mask. These are
   two-stage controls because the fused phase must be available at the APs before their beamformers
   are computed; count the additional CPU→AP phase broadcast and latency. R0 remains the
   complete-method anchor in item 2 and is not substituted for the proposed RIS in this component
   control.

Random or all-zero RIS phases may be retained only as deterministic smoke controls; outperforming
them is not evidence for a paper-level claim. Likewise, the maintained full-CSI greedy
coordinate-descent phase search remains an oracle for passive headroom and is reported separately
from deployable methods. Chen et al. [12], Huang et al. [7], Zhang et al. [4], Zhu et al. [13], and
Katsanos–Alexandropoulos [9] remain literature comparators. The five complete-method groups above are
sufficient for the primary screen; G1 and the fixed-G2-RIS controls answer attribution questions
without being promoted to peer competitors.

### 8.3 Experimental matrix

Use a staged matrix to avoid a broad sweep:

| Stage | Topology | Methods | Phase settings | Purpose |
|---|---|---|---|---|
| A: same-system screen | T0 canonical training topology | Five complete-method groups, including both mandatory model-based arms; G1 and fixed-G2-RIS diagnostics reported separately | Continuous and rounded 2-bit | Establish rate, component attribution, signaling, and runtime on the existing task |
| B: topology stress test | T1 deterministic jittered layout; T2 deterministic asymmetric/clustered layout | Frozen G2, R0, the stronger non-GNN learned baseline, and the strongest model-based pair | Continuous; 2-bit is included because it is evaluation-only | Test out-of-topology transfer without new training seeds |
| C: conditional follow-up | Only if frozen learned methods all fail on T1/T2 | G2 and the strongest non-GNN learned baseline with topology-randomized training, seed 0 | Continuous first | Separate architecture failure from a fixed-topology training-distribution failure |

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
- For the fixed-G2-RIS controls, reuse the exact same fused phase on every paired sample. Recompute
  MRT and RZF from the effective channel after applying the continuous or rounded 2-bit phase,
  use the same association mask, and apply the same per-AP power constraint. Report their two-stage
  CPU→AP signaling and end-to-end latency rather than timing only the final matrix operation.
- For iterative methods, count total transmitted scalars or bits across all online rounds, not only
  one message. State whether CSI acquisition is excluded or included for every arm. Verify that both
  model-based arms optimize the same objective and constraints on identical channel samples; report
  convergence tolerance, realized iteration count, final constraint residuals, and failure rate.
- Measure batch-1 inference/solve time after warm-up on one named device. Report median and 95th
  percentile over the paired samples, plus iteration count for AO/ADMM methods. Runtime is secondary
  to rate and signaling because implementations may have different optimization levels.
- Use a single pre-declared initialization for the main AO/ADMM result. Check two additional
  deterministic initializations on a small subset only; do not turn initialization sensitivity into
  a full sweep.
- Keep random-RIS or all-zero-phase results out of the primary comparison and claims. Use them only
  to catch implementation failures. Label full-CSI coordinate descent as a non-deployable oracle.
- Do not pool T0/T1/T2 samples to manufacture a narrow confidence interval. Provide paired intervals
  within each topology and display topology-to-topology variation directly.

### 8.5 Required outputs

The primary table should contain, for every retained method and topology:

| Method | Role | $R_{\rm cen}$ | $R_{\rm dec}$ | Cen−dec gap | Dec./cen retention | 2-bit $R_{\rm dec}$ | AP→CPU values or bits | CPU→AP values or bits | Online rounds | Batch-1 time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Because CSI acquisition and RIS actuation do not fit cleanly in the already-wide performance table,
also report a companion signaling ledger with one row per method and topology:

| Method | UE→AP | AP→CPU | CPU→AP | CPU→RIS | Total counted online bits | Online rounds | Accounting boundary |
|---|---:|---:|---:|---:|---:|---:|---|

The minimum plots are: decentralized sum rate by method with paired intervals; the fixed-G2-RIS
comparison of G2, local MRT, and local RZF beamforming; centralized-to-decentralized gap where
defined; rate versus total online signaling; and, for the model-based pair, rate versus iteration and
cumulative exchanged bits. A topology panel is useful only for the shortlisted Stage-B methods.
Random-RIS smoke controls and the coordinate-descent oracle should not appear as peer competitors in
the primary rate plot. Do not add load-balancing, UE-tail-rate, AP/RIS-count scaling, codec frontiers,
or large hyperparameter sweeps to this experiment.

### 8.6 Stop rule and experiment registration

After Stage A, retain at most G2, R0, the DNN control, and the strongest
model-based pair for Stage B. The fixed-G2-RIS controls answer component attribution and advance only
if MRT or RZF is close enough to change the active-beamforming claim. The screen is sufficient
if it determines whether G2's advantage survives a conventional DNN, a recognizable
distributed optimizer, and two topology shifts. Multi-seed confirmation becomes the next step only
after that method set and claim are frozen.

Stage A required one centralized and one distributed or partially distributed model-based joint
optimizer. E12 has now evaluated both Huang-adapted arms in a fresh, paired fixed-budget
deployed-action protocol, so that **screening** requirement is met. The older strict protocol
still fails and must not be called a converged optimizer reference. Fixed-G2-RIS MRT/RZF,
random phase, and coordinate descent remain component or smoke controls rather than replacements.

No new E-number is assigned by this plan alone. Before implementation, confirm whether the benchmark
is registered as the next experiment or attached to an existing question; then place artifacts and
the detailed report under the matching E-folder. Item 3 is registered and complete as
[E10](./experiments/e10_dnn_benchmark.md); the DNN family did not come close to G2 and does not
advance to Stage B. Item 4 is recorded as the skipped [E11](./experiments/e11_maddpg_baseline.md),
and item 5 as [E12](./experiments/e12_model_based_optimization.md). E12 now supplies the
finite-budget external comparison, while its convergence-level claim remains excluded. Item 7 is registered as
[E13](./experiments/e13_fixed_ris_beamforming.md); its component controls are complete and neither
MRT nor RZF advanced to Stage B. If load balancing or AP/UE service-count bounds are
pursued later, they should be a separate research question and experiment because they alter the
optimization target. Stage B is now registered as [E14](./experiments/e14_topology_stress.md): the
frozen G2/R0 ordering survives T1/T2, while the fixed-budget E12 arms retain higher rates. Stage C
was not triggered because the learned G2/R0 comparison did not fail on either shift.

## 9. Evidence needed before paper-level claims

1. If a **converged**, precision-stable model-based reference is required, declare a further solver
   study and use fresh data. E12 already closes the finite-budget deployed-action comparison;
   do not retune or promote its failed strict gates.
2. Stage B is complete in E14. A general topology claim would need a declared distribution of
   layouts and multiple training seeds; the two fixed shifts establish only the reported screen.
3. Decide whether E07's phase-grid legality control is restated in radians. Its identity control is
   resolved and the 4/5/6/8-bit phase grid is filled; only the high-precision end of the frontier is
   uncertified. Any compression claim must use the matched-budget wording in §6, not a message-level
   superiority claim.
4. E14 adds a direct AP→AP unicast scenario for G2's shared-UE CSI. A network claim would require
   specifying whether that CSI is fed back by UEs, forwarded by APs, relayed by CPU, or multicast,
   and measuring the corresponding serialization and transport overhead.
5. Add multi-seed training only after the baseline set, topology protocol, and final claim are frozen.

## 10. References

1. Ting et al., “Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided
   Cell-Free Networks,”
   [local PDF](<./Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks.pdf>).
2. Chan et al., “Beamforming and Load-Balanced User Association in RIS-Aided mmWave Systems via
   Adaptive Attention Graph Neural Networks,” IEEE TWC 2026,
   [DOI:10.1109/TWC.2025.3621268](https://doi.org/10.1109/TWC.2025.3621268),
   [local PDF](<./Beamforming_and_Load-Balanced_User_Association_in_RIS-Aided_mmWave_Systems_via_Adaptive_Attention_Graph_Neural_Networks (1).pdf>).
3. Ni et al., “Partially Distributed Beamforming Design for RIS-Aided Cell-Free Networks,”
   [arXiv:2208.05210](https://arxiv.org/abs/2208.05210).
4. Zhang et al., “Joint Distributed Precoding and Beamforming for RIS-aided Cell-Free Massive MIMO
   Systems,” [arXiv:2311.13139](https://arxiv.org/abs/2311.13139).
5. Xu et al., “Joint AP Clustering and Beamforming Design for RIS-Aided Cell-Free Networks,” IEEE
   TVT 2025, [DOI:10.1109/TVT.2024.3521479](https://doi.org/10.1109/TVT.2024.3521479).
6. Ke et al., “Joint RIS-UE Association and Beamforming Design in RIS-Assisted Cell-Free MIMO
   Network,” IEEE TCOM 2025,
   [DOI:10.1109/TCOMM.2025.3585502](https://doi.org/10.1109/TCOMM.2025.3585502),
   [arXiv:2506.21690](https://arxiv.org/abs/2506.21690).
7. Huang et al., “Decentralized Beamforming Design for Intelligent Reflecting Surface-enhanced
   Cell-free Networks,” IEEE WCL 2021, [arXiv:2006.12238](https://arxiv.org/abs/2006.12238).
8. Xu et al., “Algorithm Unrolling-Based Distributed Optimization for RIS-Assisted Cell-Free
   Networks,” [arXiv:2301.02360](https://arxiv.org/abs/2301.02360).
9. Katsanos and Alexandropoulos, “Robust Consensus-Based Distributed Beamforming for Wideband
   Cell-free Multi-RIS MISO Systems,” [arXiv:2601.08946](https://arxiv.org/abs/2601.08946).
10. Du et al., “Robust Beamforming Design for Active Sub-Connected RIS Assisted Cell-Free MIMO
    Systems: A Two-Stage Distributed Approach,” IEEE IoT-J 2025,
    [DOI:10.1109/JIOT.2025.3572444](https://doi.org/10.1109/JIOT.2025.3572444).
11. Hojatian et al., “Decentralized Beamforming for Cell-Free Massive MIMO with Unsupervised
    Learning,” IEEE Communications Letters 2022,
    [arXiv:2106.16194](https://arxiv.org/abs/2106.16194).
12. Chen et al., “A Distributed Machine Learning-Based Approach for IRS-Enhanced Cell-Free MIMO
    Networks,” [arXiv:2301.08077](https://arxiv.org/abs/2301.08077).
13. Zhu et al., “Multi-agent Reinforcement Learning-based Joint Precoding and Phase Shift Optimization
    for RIS-aided Cell-Free Massive MIMO Systems,” IEEE TVT 2024,
    [arXiv:2404.14092](https://arxiv.org/abs/2404.14092).
14. Gu et al., “Graph Neural Networks for Distributed Power Allocation in Wireless Networks:
    Aggregation Over-the-Air,” [arXiv:2207.08498](https://arxiv.org/abs/2207.08498).
15. Das et al., “TarMAC: Targeted Multi-Agent Communication,” ICML 2019,
    [arXiv:1810.11187](https://arxiv.org/abs/1810.11187).
16. Shao et al., “Learning Task-Oriented Communication for Edge Inference: An Information Bottleneck
    Approach,” IEEE JSAC 2022, [arXiv:2102.04170](https://arxiv.org/abs/2102.04170).
17. Shao et al., “Task-Oriented Communication for Multi-Device Cooperative Edge Inference,” IEEE TWC,
    [arXiv:2109.00172](https://arxiv.org/abs/2109.00172).
18. Sohrabi et al., “Deep Learning for Distributed Channel Feedback and Multiuser Precoding in FDD
    Massive MIMO,” IEEE TWC 2021, [arXiv:2007.06512](https://arxiv.org/abs/2007.06512).
19. Fernandes and Psaromiligkos, “Model-based Deep Learning for Joint RIS Phase Shift Compression and
    WMMSE Beamforming,” IEEE WCL 2026,
    [DOI:10.1109/LWC.2026.3683016](https://doi.org/10.1109/LWC.2026.3683016),
    [arXiv:2510.05438](https://arxiv.org/abs/2510.05438).
20. Enqvist et al., “Control Signaling for Reconfigurable Intelligent Surfaces: How Many Bits are
    Needed?,” [arXiv:2506.03929](https://arxiv.org/abs/2506.03929).
21. Guo et al., “Two-Way Passive Beamforming Design for RIS-Aided FDD Communication Systems,”
    IEEE WCNC 2021, [arXiv:2101.10094](https://arxiv.org/abs/2101.10094).
22. Lowe et al., “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments,” NeurIPS
    2017, [arXiv:1706.02275](https://arxiv.org/abs/1706.02275).
23. Shen et al., “Graph Neural Networks for Scalable Radio Resource Management: Architecture Design
    and Theoretical Analysis,” IEEE JSAC 2021, vol. 39, no. 1, pp. 101–115,
    [DOI:10.1109/JSAC.2020.3036965](https://doi.org/10.1109/JSAC.2020.3036965),
    [arXiv:2007.07632](https://arxiv.org/abs/2007.07632).
24. Eisen and Ribeiro, “Optimal Wireless Resource Allocation with Random Edge Graph Neural
    Networks,” IEEE TSP 2020, vol. 68, pp. 2977–2991,
    [DOI:10.1109/TSP.2020.2988255](https://doi.org/10.1109/TSP.2020.2988255),
    [arXiv:1909.01865](https://arxiv.org/abs/1909.01865).
25. Chowdhury et al., “Unfolding WMMSE using Graph Neural Networks for Efficient Power Allocation,”
    IEEE TWC 2021, vol. 20, no. 9, pp. 6004–6017,
    [DOI:10.1109/TWC.2021.3071480](https://doi.org/10.1109/TWC.2021.3071480),
    [arXiv:2009.10812](https://arxiv.org/abs/2009.10812).
26. Shen et al., “Graph Neural Networks for Wireless Communications: From Theory to Practice,”
    IEEE TWC 2023, vol. 22, no. 5, pp. 3554–3569,
    [DOI:10.1109/TWC.2022.3219840](https://doi.org/10.1109/TWC.2022.3219840),
    [arXiv:2203.10800](https://arxiv.org/abs/2203.10800).
27. Yin et al., “Graph Neural Network-Based Energy-Efficient Optimization for RIS-Assisted Wireless
    Networks,” IEEE TWC 2026, vol. 25, pp. 664–680,
    [DOI:10.1109/TWC.2025.3585772](https://doi.org/10.1109/TWC.2025.3585772),
    [institutional record](https://biblio.ugent.be/publication/01KMMRGNQS5SHMN7BMGNR09HJS).
28. Lim and Vu, “Distributed Graph-Based Learning for User Association and Beamforming Design in
    Multi-RIS Multi-Cell Networks,” IEEE TWC 2025, vol. 24, no. 7, pp. 6118–6134,
    [DOI:10.1109/TWC.2025.3551763](https://doi.org/10.1109/TWC.2025.3551763).
29. Nguyen et al., “Optimization of RIS-Assisted Cell-Free Massive MIMO Systems With Heterogeneous
    Graph Neural Networks Under Imperfect Channel Estimation,” IEEE TVT 2026, vol. 75, no. 5,
    pp. 7642–7657,
    [DOI:10.1109/TVT.2025.3624845](https://doi.org/10.1109/TVT.2025.3624845),
    [institutional record](https://ssu.scholarworks.kr/item/17bb3327-7509-40ac-8553-7f77544c756a).
30. Lim and Vu, “Graph Neural Network Based Beamforming and RIS Reflection Design in A Multi-RIS
    Assisted Wireless Network,” [arXiv:2501.14987](https://arxiv.org/abs/2501.14987).
