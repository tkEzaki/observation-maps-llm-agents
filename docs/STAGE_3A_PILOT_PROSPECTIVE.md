# Stage 3A pilot — prospective evaluation (pre-refit)

Date: 2026-07-24  
Status: **partial GO (moments only)** — full Stage 3B freeze still blocked  
Artifacts: `analysis/stage3a_artifacts/pilot_prospective_*.{csv,json}`

## Paid acquisition

| block | calls | valid | cost USD |
| --- | ---: | ---: | ---: |
| block_1 | 864 | 864 | 0.3504 |
| block_2 | 864 | 864 | 0.3503 |
| **total** | **1728** | **100%** | **0.7007** |

Model: `gpt-5.4-mini`. No post-hoc field drops.

## Prospective metrics (kernel-NN, frozen \(R\))

| representation | Spearman\((R_{\mathrm{pre}},e_{\mathrm{TV}})\) | enrich \(\Delta\) | mean \(e_{\mathrm{TV}}\) | stay gap obs / hat |
| --- | ---: | ---: | ---: | ---: |
| moments_m1_m3 | **+0.392** | **+0.134** | 0.390 | **1.000 / 0.475** |
| centers_24_standard | +0.039 | +0.049 | 0.329 | 0.250 / 0.066 |
| intervals_24_decimal6 | **−0.023** | +0.008 | 0.286 | 0.312 / 0.006 |

Gates:

- risk Spearman positive on **all** reps → **fail** (intervals)
- high-risk error enrichment → **pass** (all reps, weak on intervals)
- moments stay direction → **pass**
- \(N\in\{9,17\}\) \(Q_{\mathrm{risk}}\) reviewed → **yes**

Closed-loop mean \(Q_{\mathrm{risk}}\) (pre-refit, peer-matched): **≈0.093**  
(contrast: peer-confounded \(N=16\) was ~0.96–0.99)

## Decision

| encoding | freeze? |
| --- | --- |
| **moments_m1_m3** | **partial freeze candidate** after refit + hash-lock |
| centers_24_standard | hold (ranking too weak) |
| intervals_24_decimal6 | hold (risk anti-ranks) |

Full Stage 3B / Stage C remain blocked until:

1. Refit models + recalibrate \(R\) **including** pilot rows (now allowed)
2. Kernel-hurdle grouped CV + moments direction re-check
3. Hash-lock of moments production bundle (optional: keep centers/intervals as exploratory)

## Next

1. Refit + recalibrate \(R\) including pilot → **passed for moments**
2. Stage 3B **partial freeze (moments)** — see `docs/STAGE_3B_PARTIAL_FREEZE.md`
3. Stage C still blocked pending hash-lock; intervals/centers remain exploratory
