# E03 — RIS phase headroom and full-CSI references

## 1. Status

- Question: How much of each inference-mode gap can better RIS phases recover?
- Status: **Completed reference evaluation**
- Updated: 2026-09-16
- Scope: 100k and 500k checkpoints; 320 paired samples for the main decomposition
- Method: [Full-CSI greedy 2-bit reference](../decentralized_ris_methods.md#full-csi-greedy)
- Primary artifacts: `artifacts/decentralized_ris/e03_phase_headroom/`

## 2. Conclusion

At 500k, full-CSI greedy phase search recovers 90.0% of the measurable paper-decentralized gap and
65.1% of the own-only gap. Phase design remains the largest measured headroom, but the oracle uses
full CSI and is not a deployable decentralized method.

## 3. Setup

For each input mode, active beamforming $W$ is fixed to the GNN output. The phase search starts from
the rounded 2-bit GNN phase and performs up to four coordinate-descent sweeps over all $RN=120$
phase variables. Each update tests the four 2-bit phase values with the true full-CSI sum rate.

## 4. Results

At 100k:

| Fixed $W$ / phase source | GNN phase | Greedy 2-bit | Phase gain |
|---|---:|---:|---:|
| Centralized | 20.142 | 23.984 | +3.843 |
| Paper-decentralized | 16.876 | 23.764 | +6.888 |
| Own-only | 11.078 | 20.726 | +9.648 |

At 500k:

| Fixed $W$ / phase source | GNN phase | Rounded 2-bit | Greedy 2-bit | Phase gain |
|---|---:|---:|---:|---:|
| Centralized | 23.526 | 21.958 | 24.605 | $+1.079\pm0.113$ |
| Paper-decentralized | 21.400 | 19.831 | 24.285 | $+2.885\pm0.218$ |
| Own-only | 14.607 | 13.636 | 21.121 | $+6.514\pm0.209$ |

Using centralized greedy 24.605 as the common reference:

| Mode | Total gap | Recovered by own phase search | Residual |
|---|---:|---:|---:|
| Paper-decentralized | 3.205 | 2.885 (90.0%) | $0.320\pm0.109$ |
| Own-only | 9.998 | 6.514 (65.1%) | $3.484\pm0.323$ |

An earlier 320-sample joint continuous optimization found feasible values 26.327/28.704 from GNN
initialization and 29.536 from random initialization at 2k/40k. These are feasible points in a
non-convex problem, not upper bounds.

## 5. Interpretation

- The mature baseline reduces centralized phase headroom, but decentralized proposals still leave
  substantial recoverable phase value.
- After greedy search, own-only retains a much larger residual, consistent with beamforming quality,
  input limitation, train–test mismatch, and local-optimum effects.
- The experiment motivates improved phase proposals; it does not show that local APs can attain the
  oracle result.

## 6. Limitations

- Full CSI is used during search.
- Coordinate descent is initialization- and order-dependent and is not globally optimal.
- The 500k continuous joint reference was not rerun.
- One topology and training seed.

## 7. Reproduction and artifacts

| Evidence | Location |
|---|---|
| Greedy 2-bit references | `artifacts/decentralized_ris/e03_phase_headroom/discrete_cd/` |
| Continuous feasible references | `artifacts/decentralized_ris/e03_phase_headroom/continuous_ceiling/` |
| Input-mode and greedy decomposition | `artifacts/decentralized_ris/e03_phase_headroom/input_mode_decomposition/` |

Maintained entry points are `code/decentralized_ris/scripts/e03_discrete_cd_sweep.sh`,
`experiments/continuous_ceiling.py`, and `experiments/local_csi.py`.
