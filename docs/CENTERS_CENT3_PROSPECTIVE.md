# Centers CENT-3 — prospective evaluation (before refit)

Date: 2026-07-24  
Status: **PAID COMPLETE; prospective risk PASS; freeze NOT yet**

## Acquisition

| item | value |
| --- | ---: |
| Calls | 1,152 (576×2) |
| Valid rates | 94.4% / 92.9% |
| Actual cost | **\$0.622** |
| Authorized ceiling | \$1.773 |
| Runs | `runs/centers_cent3/centers-cent3-v0-block_{1,2}_*/20260724T04*` |

Hashes verified pre-run (`verify_cent3_hashes.py` ALL_OK).  
Prospective lock unchanged: `cent3_prospective_lock_v0.json`.

## Prospective risk (primary gate)

Using locked \(R_{\mathrm{pre}}\) vs observed TV (no refit):

| metric | value | gate |
| --- | ---: | --- |
| Spearman(\(R_{\mathrm{pre}}\), \(e_{\mathrm{TV}}\)) | **+0.288** | pass (>0) |
| High-risk enrichment (top/bottom 20%) | **1.37×** | pass |
| Risk-bin monotone | yes | pass |

**Conclusion:** centers-specific predictive risk ranks new-field error prospectively. This was the main unpaid→paid hurdle.

## Error levels (descriptive)

| quantity | mean |
| --- | ---: |
| \(e_{\mathrm{TV}}\) | 0.481 |
| activity abs err | 0.161 |
| \(a_0\) abs err | 0.719 |
| stay abs err | 0.161 |
| between-block TV | 0.111 |

Direction / \(a_0\) residual is large: risk ranks uncertainty, but the production hurdle’s signed action on several polar controls is poorly calibrated.

## Control phenotypes (observed)

| control | \(n_{\mathrm{valid}}\) | note |
| --- | ---: | --- |
| unimodal +polar (peer 16) | 32 | **all stay** |
| unimodal sign-reversed | 32 | **all stay** |
| exact antipodal | 21 | stay≈0.24 (not moments-like ~1) |
| small imbalance | 27 | active; \(p_+\approx0.78\) |
| sparse unimodal | 32 | all stay |
| sparse antipodal | **8** | all stay; below min-valid 12 |

Polarity sign-flip on ±unimodal controls: **not observed** (both collapsed to stay).

## Sampling

One control (`cent_ctrl_sparse_antipodal`) has only 8 valid parses (<12 protocol floor). Other fields ≥21. Invalids inflate uncertainty on that cell; do not drop the field post hoc.

## Freeze recommendation

| item | status |
| --- | --- |
| Prospective risk ranking | **PASS** |
| Proceed immediately to `centers_bundle_v1` | **NO** |
| Reason | large \(a_0\) error; unimodal polar controls collapsed to stay; one low-valid control |

Next: CENT-4 checklist with centers-calibrated thresholds; consider model-family revision (direct multinomial / native kernel / regime mixture) before freeze, while keeping the prospective risk calibrator as a positive result.

Artifacts:

- `analysis/centers_branch/cent3_prospective_summary.json`
- `analysis/centers_branch/cent3_prospective_detail.csv`
