# E08 — Historical baseline sweeps over antennas and power

## 1. Status

- Question: Does the short-budget baseline reproduce the paper's trends over $M$ and $P_{\max}$?
- Status: **Completed historical reproduction**
- Updated: 2026-09-16
- Scope: original short-budget R0 runs, one seed per operating point
- Method: [R0 baseline](../decentralized_ris_methods.md)
- Primary artifacts: `artifacts/decentralized_ris/e08_baseline_sweeps/`

## 2. Conclusion

The sweeps reproduce the expected monotonic improvement with antennas and transmit power and match
the paper figures closely enough for baseline qualification. They use the old short training budget
and therefore must not be used to support the newer G2 performance claim.

## 3. Setup

The antenna sweep fixes $P_{\max}=15$ dBm and varies $M=1,\ldots,5$. The power sweep fixes $M=2$
and varies $P_{\max}=5,10,\ldots,35$ dBm. Other topology parameters follow the baseline setting.

## 4. Results

| $M$ | Centralized | Paper-dec. | Centralized 2-bit | Paper-dec. 2-bit |
|---:|---:|---:|---:|---:|
| 1 | 6.219 | 5.860 | 5.711 | 5.396 |
| 2 | 7.663 | 7.358 | 7.081 | 6.808 |
| 3 | 9.247 | 8.883 | 8.654 | 8.277 |
| 4 | 10.143 | 9.663 | 9.750 | 9.287 |
| 5 | 11.139 | 10.913 | 10.606 | 10.411 |

| $P_{\max}$ dBm | Centralized | Paper-dec. | Centralized 2-bit | Paper-dec. 2-bit |
|---:|---:|---:|---:|---:|
| 5 | 4.592 | 4.318 | 4.269 | 4.008 |
| 10 | 6.401 | 6.052 | 5.956 | 5.656 |
| 15 | 7.663 | 7.358 | 7.081 | 6.808 |
| 20 | 8.501 | 8.338 | 7.752 | 7.534 |
| 25 | 9.335 | 9.243 | 8.269 | 8.142 |
| 30 | 10.364 | 10.366 | 9.067 | 9.027 |
| 35 | 11.030 | 11.052 | 9.359 | 9.244 |

Across the paper figures and stored points, differences are 0.03–0.40 bps/Hz. A current-code rerun
of the headline setting differs by at most 0.16 bps/Hz.

## 5. Interpretation

- More antennas and transmit power improve both inference modes.
- The centralized–decentralized gap remains small in these short-budget sweeps.
- These are reproduction trends, not fair comparisons with 150k/500k methods.

## 6. Limitations

- One seed and old short-budget training.
- The paper's $M=4$ point used seed 7, unlike the other points.
- Noise power is implicit in implementation rather than specified by the paper.
- No uncertainty intervals were produced for the plotted sweep means.

## 7. Reproduction and artifacts

The two artifact directories contain checkpoints, final-evaluation text/NumPy files, event logs, and
summary workbooks. Plot sources are retained in
`artifacts/decentralized_ris/e08_baseline_sweeps/plots/`. New sweeps should use
`code/decentralized_ris/scripts/e08_baseline_sweeps.sh` and JSON summaries rather than the
historical Excel path.
