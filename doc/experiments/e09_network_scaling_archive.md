# E09 — Snapshot network-scaling pilot

## 1. Status

- Question: Did the early network-size pilots support a scalability claim?
- Status: **Archived; no formal claim**
- Updated: 2026-09-16
- Scope: historical ring/disk v1 and BPP v2 prototypes
- Method: frozen prototypes under `code/snapshot_network_scaling/`
- Primary artifacts: remote historical directories listed below

## 2. Conclusion

No. The pilots provide implementation lessons but do not isolate network-size effects or support
generalization. The source remains frozen for provenance and must not be used as the active
experimental path.

## 3. Setup

The v1 pilot used a ring-style topology. The v2 pilot introduced a 2-D binomial point process and
wrap-around geometry. Both varied multiple factors at once and used too few AP layouts for a stable
scaling comparison.

## 4. Results

Retained evidence:

- The original RIS baseline stayed in a meaningful-rate regime as scale increased.
- The no-RIS v1 control used an incompatible noise setting and fell near the noise floor.
- The BPP and wrap-around implementation contracts worked.
- Ten AP layouts were insufficient to stabilize the AP-spacing distribution gate.
- Without a 3-D distance floor, short horizontal distances and high path-loss exponents caused
  extreme received-power sensitivity.

## 5. Interpretation

The observed scale differences mix AP layout, UE drops, model initialization, and training
randomness. They cannot be interpreted as a pure network-size effect. A future scaling experiment
must fix density or explicitly define fixed-area scaling, use many AP layouts, separate topology and
training seeds, and use topology-matched checkpoints.

## 6. Limitations

- Development evidence only; no pre-registered scaling matrix.
- Too few layouts and uncontrolled simultaneous changes.
- Historical source and remote artifacts are not maintained active code.

## 7. Reproduction and artifacts

- Frozen v1/v2 source: `code/snapshot_network_scaling/`
- v1 remote artifacts: `/tmp2/b12902052/snapshot_network_scaling/results_snapshot_scaling/` on `ws2`
- v2 remote staging: `/tmp2/b12902052/snapshot_network_scaling_v2_bpp/`

Artifacts are retained only for provenance; they create no obligation to continue the old matrix.
