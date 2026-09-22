# Stage B Step D: nested sparse-peer imbalance × realization panel

Date: 2026-07-23  
Status: **executed and analyzed**

## Design (simultaneous nested panel)

Imbalance levels and multi-realization variance ran as **one** paid campaign
(not separate P1-sparse / P2). Both acquisition blocks ran in parallel.

| factor | value |
| --- | --- |
| peer count | 16 (`sparse_peer16_*`) |
| imbalances | 8:8, 9:7, 10:6 |
| realizations \(M\) | 6 (partner-paired; leftover on majority when unequal) |
| LLM samples / cell | 4 |
| encodings × offsets × blocks | 3 × 36 × 2 |
| calls | 15,552 |

## Execution

| block | valid | cost |
| --- | ---: | ---: |
| 1 `.../20260723T140327Z` | 7,776/7,776 | $3.160 |
| 2 `.../20260723T140344Z` | 7,776/7,776 | $3.162 |
| **total** | **15,552** | **$6.322** |

## Main findings

### Exact balance (8:8): moments abstain across realizations

Across 6 independent partner-paired realizations (across-block means):

- moments: \(A\simeq0.021\)–\(0.062\), \(R_1\simeq0.015\)–\(0.049\)
- intervals / centers: \(A\simeq0.64\), weak \(R_1\)

Realization scatter for moments activity RMS ≈ 0.015 — abstention is
**stable**, not a one-draw fluke.

### Count imbalance restores moments polar control

| imbalance | moments \(A\) | moments \(a_1\) | moments \(R_1\) |
| --- | ---: | ---: | ---: |
| 8:8 | ~0.047 | ~0 | ~0.033 |
| 9:7 | ~0.992 | ~1.045 | ~1.099 |
| 10:6 | ~0.999 | ~1.045 | ~1.061 |

Sparse count imbalance behaves like the dense \(\varepsilon\) tip: tiny
majority restores a strong signed polar channel for moments.

### Between / within gate

| imbalance | pair | \(d_{C_1}/\)within | \(\Delta A/\)within | mean TV |
| --- | --- | ---: | ---: | ---: |
| 8:8 | intervals–moments | 2.2 | **20.8** | 0.62 |
| 8:8 | moments–centers | 2.1 | **22.1** | 0.62 |
| 8:8 | intervals–centers | 1.9 | 0.07 | 0.40 |
| 9:7 | intervals–moments | **16.0** | 4.8 | 0.49 |
| 10:6 | intervals–moments | **13.3** | 3.2 | 0.51 |

At exact sparse balance, polar \(C_1\) separation is marginal (both histogram
classes weak), but **activity / TV full-operator separation is huge**.
With 9:7 or 10:6, polar-channel separation easily clears a between/within >2
rule for moments vs others.

## Interpretation

1. Nested design succeeded: imbalance effect and realization noise measured
   together.
2. Moments antipodal abstention survives sparse partner-paired 8:8 draws.
3. One-peer majority (9:7) is enough to flip moments into an active polar
   controller — high sparse susceptibility.
4. Intervals vs centers remain harder to separate on \(C_1\) at 8:8; use
   full-operator metrics there.

## Artifacts

- Protocols: `protocol_sparse_peer_v0_1_block_{1,2}.json`
- Cost card: `docs/STAGE_B_D_COST_CARD.md`
- Analysis: `analysis/complex_kernel_sparse_peer/`
- Analyzer: `analysis/analyze_sparse_peer.py`

## Next

Sparse panels are valid Stage 3A training data (with representation-pair
interpretation caveats at exact 8:8). See `docs/STAGE_3A_SURROGATE.md`.
Stage C remains blocked until Stage 3B freeze.
