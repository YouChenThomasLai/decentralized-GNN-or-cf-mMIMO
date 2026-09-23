# E11 — Same-interface MADDPG benchmark arm

## 1. Status

- Question: does a same-interface multi-agent actor-critic close the gap to G2?
- Status: **Stopped and excluded from the benchmark**
- Updated: 2026-09-18
- Scope: abandoned Stage-A screening arm; no final holdout result
- Preserved artifact: `artifacts/decentralized_ris/e11_maddpg_baseline/preregistration.json`

## 2. Conclusion

E11 is skipped. The repository owner judged its observed training trajectory not competitive enough
to justify completing a 150k run, and review found that exploration noise was added after the
association mask without reapplying that mask. The behaviour policy could therefore place
beamformer energy on unserved UE
columns. Because the collected transitions were not guaranteed feasible, no intermediate rate is a
valid benchmark result.

The experiment is not repaired or rerun. E10 already supplies the retained learned non-graph
control, while E12 supplies the mandatory conventional model-based comparison. No claim in the
benchmark may cite E11 as either a successful baseline or evidence that MADDPG is intrinsically
weak.

## 3. Setup

The abandoned design used one decentralized actor per AP, a training-only centralized critic, the
same local observation and AP-to-CPU phase interface as G2, and a one-step shared sum-rate reward.
The intended deployed actor count was capacity-matched to G2. These details describe the screened
configuration only; they are not a maintained method definition after the implementation was
removed.

## 4. Results

No final holdout was run and no rate is reported. The run was stopped before it could produce an
admissible comparison. The only durable result is the negative screening decision and the discovery
that the exploration path violated the association constraint.

## 5. Interpretation

Skipping E11 narrows the paper claim. The completed DNN control supports a statement about G2 versus
a parameter-matched conventional DNN under the matched training protocol. It does not support a
general statement about reinforcement learning or MADDPG. Stage A no longer waits for E11 by owner
decision; it still waits for a corrected E12 model-based run.

## 6. Limitations

- The stop decision is based on an uncompleted training trajectory, not a valid final comparison.
- The implementation defect invalidates behaviour-policy data collected before the stop.
- No hyperparameter search or corrected rerun was performed.
- The preserved registration records intent, not a completed or immutable preregistered result.

## 7. Reproduction and artifacts

The unfinished implementation, launch script, and regression test were removed from active code to
avoid presenting an abandoned arm as supported. The registration JSON is retained unchanged as
provenance. Any future revival must be registered as a new run, reapply the association mask after
exploration, select a validation-best checkpoint, and evaluate that checkpoint on a separate final
holdout.
