# Stage C v0.1 — selection pilot results

Date: 2026-07-24  
Status: **COMPLETE**  
Session: `runs/stage_c/stage-c-v0.1_0291166d2e39/20260723T235620Z`  
Report: `analysis/stage_c_artifacts/stage_c_v0_1_report.json`

## Acquisition

| item | value |
| --- | ---: |
| Calls | 6,240 |
| Valid | 6,240 (100%) |
| Actual cost | **\$1.456** |
| Est. base | \$2.115 |

## Mean final \(r\) by \(K\) (matched init seeds)

| \(K\) | LLM | surrogate | MAE |
| ---: | ---: | ---: | ---: |
| −0.15 | 0.043 | 0.145 | 0.102 |
| 0.0 | 0.188 | 0.188 | 0.000 |
| +0.15 | 0.995 | 0.891 | 0.104 |

Overall MAE(final \(r\)) ≈ **0.069**.

At \(K=0\) dynamics ignore social actions → exact match (seed check).
Attractive \(K=+0.15\): LLM locks harder / faster than the surrogate.
At \(K=-0.15\), the LLM suppresses polar order more strongly than the
surrogate, while substantial higher-harmonic structure remains
(\(r_3\simeq0.39\)–\(0.46\) on several runs).

## Selection reading

- Positive \(K\approx0.15\) is informative (near-complete polar sync).
- Negative \(K\) **suppresses polar order** but is **not** established as a
  disordered phase (see \(r_2/r_3\) in offline diagnostics).
- \(N=9\) and \(N=17\) both usable under peer-matched AD; not interchangeable
  on arrival times or negative-\(K\) harmonics.
- Surrogate v1 is **qualitatively predictive, quantitatively miscalibrated**
  (stay over-prediction). Next: prompt audit → collective-field replay →
  `moments_bundle_v2` (do not overwrite v1).

## Offline follow-up (§5)

See `docs/STAGE_C_OFFLINE_V0_1.md` for \(r_2/r_3\), per-\(N\times\)seed tables,
LLM-field risk, teacher-forced scores, surrogate predictive intervals,
activity/torque, and §7 branch **A**.

## Non-claims

Not Stage D production. Two-body still deferred.
Negative \(K\) is **not** established as a disordered phase (see \(r_2/r_3\)).
