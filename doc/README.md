# Decentralized RIS research documentation

Use these three entry points:

1. [Method report](./decentralized_ris_methods.md): system notation, the proposed G2 method, its G1
   no-context ablation, R0/R1/G0 controls, and the benchmark arms defined for comparison.
2. [Experiment index](./decentralized_ris_experiments.md): current conclusion and one row per experiment report.
3. [Research positioning](./research_positioning.md): related work, defensible claims, and evidence
   still needed before publication.

The research narrative and follow-up experiments focus on G2. G1 is retained only as an internal
ablation for the contribution of G2's per-RIS context; it is not a co-equal proposed method or a
separate continuation target. For a problem-to-result narrative with the R0 training and G2
ablation figures, see the [progress report](./decentralized_ris_progress_report.md).

## Experiment reports

Each report under [`experiments/`](./experiments/) uses the same format: status, conclusion, setup,
results, interpretation, limitations, and reproduction/artifacts. Detailed numeric audit tables may
remain next to their machine-readable artifacts, but the maintained conclusion lives in the tracked
E-report.

## Reference papers

The PDF files in this directory are source literature, not experiment-status documents. The anchor
paper is [Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free
Networks](<./Decentralized Graph Neural Network-Based Joint Beamforming in Multi-RIS-Aided Cell-Free Networks.pdf>).
