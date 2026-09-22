# Stage C v0.2a — results (transition validation)

Date: 2026-07-24  
Status: **PAID COMPLETE**  
Session: `runs/stage_c/stage-c-v0.2a_ec22717957a1/20260724T014052Z`  
Protocol: `experiments/stage_c/protocol_stage_c_v0_2a.json`  
Prediction bundle: `moments_bundle_v2`  
Comparator: `moments_bundle_v1` (offline only)

## Acquisition

| item | value |
| --- | ---: |
| Calls | **27,200 / 27,200** valid |
| Valid rate | **1.000** |
| Actual cost | **\$6.344** (est. \$9.221) |
| Wall time | ≈ 46.7 min |

## Scientific verdict

**v2 prospectively improves the intended axes vs v1**, especially **activity**
and **one-step stay**. Final \(r_1\) alone would have missed this (all cells lock).

### Activity (primary)

| | MAE vs LLM | mean signed err (LLM − surr) |
| --- | ---: | ---: |
| v1 | **0.100** | +0.100 (v1 under-active) |
| v2 | **0.014** | +0.002 |

Teacher-forced stay on LLM trajectories:

| | \(\hat p_{\mathrm{stay}}\) | \(p_{\mathrm{stay}}^{\mathrm{obs}}\) | \|err\| |
| --- | ---: | ---: | ---: |
| v1 | 0.137 | 0.032 | 0.105 |
| v2 | **0.028** | 0.032 | **0.010** |

### By \(K\) (N=17, 4 seeds)

| \(K\) | LLM \(A\) | v1 \(A\) | v2 \(A\) | LLM \(t_{0.9}\) | LLM \(r_1(T)\) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.08 | 0.974 | 0.873 | 0.946 | 32.0 | 0.998 |
| 0.10 | 0.967 | 0.866 | 0.967 | 20.8 | 0.998 |
| 0.12 | 0.967 | 0.868 | 0.976 | 20.8 | 0.998 |
| 0.15 | 0.963 | 0.864 | 0.972 | 15.0 | 0.996 |

All seeds reach \(r_1(T)>0.9\). Surrogates still lag LLM on arrival time
(\(t_{0.9}\)); v2 closes most of the **activity** gap that v1 left open.

### Final \(r_1\) MAE

| vs v1 | vs v2 |
| ---: | ---: |
| 0.060 | **0.042** |

## Interpretation

1. Replay-informed regime-aware stay gate **works prospectively** on new
   transition \(K\) and \(T=100\) fields not used as Stage C v0.1 cells.
2. Binary locking / \(K_{\mathrm{cross}}\) was the wrong sole scorecard — LLM
   and both surrogates lock; the science is in **activity / stay / speed**.
3. Residual: LLM still arrives faster than either surrogate at \(K=0.08\);
   direction/torque structure remains a quantitative gap.

## Artifacts

| file | role |
| --- | --- |
| `analysis/stage_c_artifacts/stage_c_v0_2a/surrogate_lock.json` | pre-paid v1/v2 preds |
| `analysis/stage_c_artifacts/stage_c_v0_2a/actual_cost.json` | tokens + \$ |
| `analysis/stage_c_artifacts/stage_c_v0_2a/v0_2a_verdict.json` | rollup |
| `analysis/stage_c_artifacts/stage_c_v0_2a/llm_vs_surrogate_by_K.csv` | K table |
| `analysis/stage_c_artifacts/stage_c_v0_2a/trajectory_summary_per_run.csv` | per-run endpoints |
| `teacher_v1/`, `teacher_v2/` | one-step calibration |

## Next

Conditional GO for **v0.2b** (\(N=9\), \(K=-0.15\)) after reviewing this card.
Do not treat v0.2a as Stage D production.
