# Stage B P1: dense antipodal weight-imbalance sweep

Date: 2026-07-23  
Status: **executed and analyzed**

## Design

Signed weight axis on antipodal modes at \(0,\pi\) (κ=6):

\[
\varepsilon\in\{-0.10,-0.05,-0.02,0,0.02,0.05,0.10,0.20\}
\]

3 encodings × 8 ε × 36 offsets × 12 samples × 2 blocks = **20,736** calls.

## Execution

| block | run | valid | cost |
| --- | --- | ---: | ---: |
| 1 | `runs/antipodal_weight/.../20260723T135128Z` | 10,368/10,368 | $4.218 |
| 2 | `runs/antipodal_weight/.../20260723T135129Z` | 10,368/10,368 | $4.216 |
| **total** | | **20,736** | **$8.434** |

Model: `gpt-5.4-mini` @ $0.75 / $4.50 per 1M.

## Primary result

**Moments show a sharp abstention → activation transition under infinitesimal
weight imbalance; histograms retain residual activity at exact balance.**

Across-block means (moments):

| \(\varepsilon\) | \(A\) | \(a_1\) | \(R_1\) |
| ---: | ---: | ---: | ---: |
| −0.10 | 1.000 | −1.129 | 1.135 |
| −0.05 | 0.997 | −1.111 | 1.163 |
| −0.02 | 0.859 | −1.001 | 1.009 |
| **0** | **0.074** | **−0.007** | **0.024** |
| +0.02 | 0.841 | +0.937 | 0.943 |
| +0.05 | 0.988 | +1.111 | 1.160 |
| +0.10 | 1.000 | +1.085 | 1.121 |
| +0.20 | 1.000 | +1.048 | 1.051 |

Signed odd response: \(a_1(-\varepsilon)\approx -a_1(\varepsilon)\).
OLS comparative slope \(\chi_{a_1}\approx 9.17\) (moments), \(5.16\) (centers),
\(1.11\) (intervals). This is **not** a precise local
\(\partial a_1/\partial\varepsilon|_{\varepsilon=0}\) (response jumps between
0 and 0.02 then saturates); use it only as a comparative empirical slope.
Stage 3 learns full trinomials at each \(\varepsilon\), not \(\chi\).

Activity activation scale \(\varepsilon_{1/2}\) (midpoint of \(A(0)\to A_{\max}\))
is **0.02 for all three encodings** as an **upper-bound grid estimate**
(no points in \((0,0.02)\)); safe claim: activation is at or below the
smallest measured nonzero \(|\varepsilon|=0.02\). Baselines differ:

| encoding | \(A(0)\) | \(A_{\max}\) | \(\Delta A\) | near-0 \(\partial A/\partial|\varepsilon|\) |
| --- | ---: | ---: | ---: | ---: |
| moments | 0.074 | 1.000 | 0.926 | ~4.88 |
| centers | 0.326 | 0.997 | 0.670 | (already high by 0.02) |
| intervals | 0.543 | 0.999 | 0.456 | ~0.56 |

Evenness residuals (mean over \(\pm\varepsilon\) pairs) are small for \(A\)
(~0.01), supporting even activity; odd \(a_1\) residuals are also small
(~0.04–0.07).

## Full operator at \(\varepsilon=0\)

Polar \(C_1\) remains collapsed (min pairwise \(d_{C_1}\sim0.07\)–\(0.15\)),
while mean TV stays large (~0.32–0.53). Confirms
**symmetry-selective polar-channel collapse** with persistent full-operator
separation at exact balance.

## Interpretation

1. Exact antipodal balance is a special point for moments: near-complete
   directional abstention.
2. Tiny imbalance (\(|\varepsilon|=0.02\)) restores a strong signed polar
   channel and \(A\simeq0.85\)–\(1\) for moments — encoding-dependent
   **symmetry-breaking susceptibility**.
3. Intervals/centers never fully abstain at \(\varepsilon=0\) (residual
   activity), consistent with serialization / bin pseudo-cues.
4. Shape between 0 and 0.02 is **not resolved**; optional near-zero
   densification top-up remains available (not required for the main claim).

## Artifacts

- Protocols: `protocol_antipodal_weight_v0_1_block_{1,2}.json`
- Cost card: `docs/STAGE_B_P1_COST_CARD.md`
- Power sim: `analysis/stage_b_offline_p3_p4/p1_power_simulation.json`
- Analysis: `analysis/complex_kernel_antipodal_weight/`
  (`epsilon_trajectories.csv`, `susceptibilities.csv`, …)
- Analyzer: `analysis/analyze_antipodal_weight.py`

## Next

- Stage 3A surrogate implementation: `docs/STAGE_3A_SURROGATE.md` (GO).
- Near-zero densification \(|\varepsilon|\in\{0.005,0.01\}\) only under Policy B
  if closed-loop coverage shows the unresolved band matters (Policy A default).
- Stage 3B freeze after grouped validation + closed-loop \(q_{\mathrm{OOD}}\).
