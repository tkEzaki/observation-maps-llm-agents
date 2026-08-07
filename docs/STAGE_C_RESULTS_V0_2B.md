# Stage C v0.2b — results (size / negative replication)

Date: 2026-07-24  
Status: **PAID COMPLETE**  
Session: `runs/stage_c/stage-c-v0.2b_47d97bc0acc3/20260724T023438Z`  
Protocol: `experiments/stage_c/protocol_stage_c_v0_2b.json`  
Depends on: v0.2a success

## Acquisition

| item | value |
| --- | ---: |
| Calls | **28,400 / 28,400** valid |
| Valid rate | **1.000** |
| Actual cost | **\$6.630** (est. \$9.628) |
| Wall time | ≈ 63.0 min |

## Scientific verdict

**Size and negative-\(K\) replication succeed.** v2 remains closer on activity /
stay than v1. Negative \(K\) **suppresses polar order** (not “disordered”):
\(r_1(T)\approx0.03\) with **near-unit activity**.

### Overall (all cells)

| | MAE activity | mean signed (LLM−surr) | teacher \|stay err\| |
| --- | ---: | ---: | ---: |
| v1 | 0.085 | +0.084 | 0.089 |
| v2 | **0.044** | −0.010 | **0.047** |

### \(N=9\) by \(K\)

| \(K\) | LLM \(A\) | v1 \(A\) | v2 \(A\) | LLM \(r_1(T)\) | lock frac |
| ---: | ---: | ---: | ---: | ---: | ---: |
| −0.15 | 0.999 | 0.782 | 0.957 | 0.036 | 0/4 |
| 0 | 0.999 | 0.831 | 0.980 | 0.211 | 0/4 |
| 0.06 | 0.963 | 0.910 | 0.987 | 0.998 | 4/4 |
| 0.08 | 0.940 | 0.923 | 0.989 | 0.998 | 4/4 |
| 0.10 | 0.935 | 0.922 | 0.989 | 0.998 | 4/4 |
| 0.15 | 0.926 | 0.923 | 0.989 | 0.996 | 4/4 |

### \(N=17\), \(K=-0.15\)

| | LLM | v1 | v2 |
| --- | ---: | ---: | ---: |
| \(A\) | 0.996 | 0.875 | 0.938 |
| \(r_1(T)\) | 0.028 | 0.111 | 0.097 |

Polar order stays low; activity stays high — consistent with v0.1 wording
(“suppresses polar order; higher harmonics remain”).

## Notes

1. At strong positive \(K\), v2 slightly **over**-predicts activity vs LLM
   (signed err ≈ −0.05); still far better than v1’s under-activity on
   negative / near-zero \(K\).
2. For \(N=9\), LLM locking onset satisfies
   \(0 < K_{\mathrm{cross}}^{\mathrm{LLM}}\le 0.06\)
   (\(K=0\): 0/4 lock; \(K=0.06\): 4/4). Surrogate offline crossing remains
   **0.08** — LLM locks **earlier** than the attenuated surrogate envelope.
3. Combined with v0.2a: regime-aware v2 generalizes across **size** and
   **sign(\(K\))** on activity/stay; sync-speed / \(K_c\) remain only partially
   predicted.

## Artifacts

| file | role |
| --- | --- |
| `analysis/stage_c_artifacts/stage_c_v0_2b/v0_2b_verdict.json` | rollup |
| `analysis/stage_c_artifacts/stage_c_v0_2b/llm_vs_surrogate_by_K.csv` | per-\((N,K)\) |
| `analysis/stage_c_artifacts/stage_c_v0_2b/actual_cost.json` | tokens + \$ |
| `surrogate_lock.json` | pre-paid preds |

## Next

Stage C v0.2 grid complete for planned cells. Combined report:
`docs/STAGE_C_RESULTS_V0_2_COMBINED.md`. Moments branch closed for further
paid collectives; Stage D / production claims remain **out of scope**.
