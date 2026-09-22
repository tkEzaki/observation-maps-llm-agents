# Stage 3B — partial freeze (moments)

Date: 2026-07-24  
Status: **HASH-LOCKED selection-grade baseline** (`freeze_ready=true`)  
Bundle: `analysis/stage3a_artifacts/moments_bundle_v1/`  
Decision: `analysis/stage3a_artifacts/stage3b_decision.json`  
Naming: **prospectively frozen Stage-C selection baseline** (not production)

## Scope

| item | decision |
| --- | --- |
| `moments_m1_m3` | **selection-grade frozen (v1)** — keep permanently; do not overwrite |
| model | **kernel_hurdle** (selection baseline) |
| `centers_24_standard` | exploratory (not frozen) |
| `intervals_24_decimal6` | exploratory (not frozen) |
| Stage C collective LLM | **authorized** for v0.1 selection pilot |

## Locked artifacts

Root: `analysis/stage3a_artifacts/moments_bundle_v1/`

| file | role |
| --- | --- |
| `manifest.json` | sha256 lock + aggregate hash |
| `kernel_hurdle/` | fitted stay + direction kernels |
| `applicability_domain/` | shape / orientation / native layers |
| `risk_calibrator.json` | post-pilot calibrated \(R(x)\) |
| `ood_policy.json` | peer hard-gate + risk threshold + \(N\in\{9,17\}\) |
| `hyperparameters.json` | length scales, \(k\), stay features |
| `training_arrays_moments.npz` | moments feature/count slice |
| `training_rows_post_pilot.csv` | full post-pilot row table |
| `provenance.json` | gate metrics + source counts |

Rebuild: `py -m analysis.stage3.freeze_moments_bundle`

## Post-refit gates (moments)

Full **8-gate checklist**: `docs/STAGE_3B_FREEZE_CHECKLIST.md`  
JSON: `analysis/stage3a_artifacts/stage3b_moments_freeze_checklist.json`

| # | gate | pass | key evidence |
| ---: | --- | --- | --- |
| 1 | beats global + near best | yes | kh log_loss 0.381 ≪ global 1.029; within 2% of best |
| 2 | moments abstention/activation | yes | stay gap 0.745 |
| 3 | sparse-realization holdout | yes | kh beats global on sparse scheme |
| 4 | calibrated \(R\) orders error | yes | OOF Spearman 0.484; monotone |
| 5 | peer-matched \(Q_{\mathrm{risk}}\) | yes | closed-loop mean 0.056 at \(N=9,17\) |
| 6 | high-occupancy risk policy | yes | peer hard-reject + OOF-80% \(R\) threshold |
| 7 | ensemble agreement | yes | mean pairwise TV disagree ≈0.18 on train sample |
| 8 | hash-lock | yes | `moments_bundle_v1` `freeze_ready=true` |

## Operating constraints

- Peer hard-gate: only \(\{8,16,240\}\)
- Collective scans at \(N\in\{9,17\}\)
- \(R(x)\) is ranking / applicability, not a precise error predictor
- Reload smoke check on persist (kernel + AD) passed

## Stage C

Hash-lock complete; user GO recorded. See `docs/STAGE_C_PROTOCOL.md`
(`stage_c_authorized=true`; protocol `stage-c-v0.1`, 6,240 calls).
