# Stage 3B — moments_bundle_v2 (regime-aware stay)

Date: 2026-07-24  
Status: **prospectively frozen Stage C v0.2 baseline** (8/8 gates pass);  
moments Stage C paid branch **closed** after v0.2 combined report  
(`docs/STAGE_C_RESULTS_V0_2_COMBINED.md`)  
Bundle: `analysis/stage3a_artifacts/moments_bundle_v2/`  
Decision: `analysis/stage3a_artifacts/stage3b_v2_decision.json`  
Checklist: `docs/STAGE_3B_V2_FREEZE_CHECKLIST.md`  
Parent: `moments_bundle_v1` (**not overwritten**)

## Why v2

Replay branch A showed all 30 Stage-C-sourced fields have
\(p_{\mathrm{stay}}^{\mathrm{obs}}=0\), while v1 still predicted
\(\hat p_{\mathrm{stay}}\approx0.10\)–\(0.17\) by leaking abstention stay into
active collective fields.

v2 adds a **regime-aware stay gate** (descriptors only; no \(K\)/\(t\)/source):

- abstention expert: high `antipodal_balance` and low `|z1|`
- active expert: shorter bandwidth + asymmetric Dirichlet favoring move

Training unifies post-pilot Stage B + Stage 3A pilot + collective-field replay
(`build_unified_dataset_v2`).

## Locked artifacts

| file | role |
| --- | --- |
| `manifest.json` | sha256 lock + aggregate hash |
| `regime_kernel_hurdle/` | dual stay experts + direction kernel |
| `applicability_domain/` | shape / orientation / native layers |
| `risk_calibrator.json` | regime-model OOF TV risk |
| `ood_policy.json` | peer hard-gate; Stage C still not auto-authorized |
| `hyperparameters.json` | regime thresholds + active stay prior |
| `training_rows_v2.csv` | unified row table |
| `replay_stay_check_excl_replay_train.csv` | honest stay check (fit excl. replay) |
| `provenance.json` | holdout + stay diagnostics |

Rebuild: `py analysis/stage3/build_moments_bundle_v2.py`

## Stay diagnostics

| check | v1 | v2 | obs |
| --- | ---: | ---: | ---: |
| collective stay (fit excl. replay) | 0.132 | **0.033** | 0.000 |
| holdout stay gap (exact − small ε) | — | **0.883** | — |

Production v2 (replay in train), mean \(\hat p_{\mathrm{stay}}\) by bucket:

| bucket | v1 | v2 |
| --- | ---: | ---: |
| neg | 0.166 | 0.042 |
| pos | 0.103 | 0.009 |
| zero | 0.121 | 0.014 |
| stage_b_anchor | 0.400 | 0.408 |

Anchors retain abstention; no global \(p_{\mathrm{stay}}\mapsto c\,p_{\mathrm{stay}}\).

## Envelope compare

See `docs/STAGE_C_DENSE_K_V1_VS_V2.md`.  
\(K_{\mathrm{cross}}\) unchanged at \(N=9\!:\!0.08\), \(N=17\!:\!0.10\).

## Stage C v0.2

- Design: **GO** — `docs/STAGE_C_V0_2_PROTOCOL.md`
- Paid v0.2a: **conditional** on cost card + explicit `--yes`
  (`docs/STAGE_C_V0_2A_COST_CARD.md`)
- v2 is the online surrogate; v1 remains the permanent v0.1 comparator
