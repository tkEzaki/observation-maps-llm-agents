# Stage 3B moments freeze checklist (8 gates)

Date: 2026-07-24  
Grade: **selection-grade Stage-C baseline** (not production)  
All gates pass: **True**  
Artifact: `analysis/stage3a_artifacts/stage3b_moments_freeze_checklist.json`  
Bundle v1 is permanent. Successor hash-locked: `moments_bundle_v2`
(`docs/STAGE_3B_MOMENTS_V2.md`).

| # | gate | pass | key evidence |
| ---: | --- | --- | --- |
| 1 | beats_global_and_near_best | yes | kh=0.381 < global=1.029; within_tol=True |
| 2 | moments_abstention_activation | yes | stay_gap=0.745 |
| 3 | sparse_realization_holdout | yes | sparse kh=0.384 < global=1.098 |
| 4 | calibrated_risk_orders_error | yes | Spearman=0.484; monotone=True |
| 5 | peer_matched_Q_risk | yes | Q_risk=0.056; peers={8,16,240} |
| 6 | high_occupancy_risk_policy | yes | threshold=0.458; peer hard-reject |
| 7 | ensemble_agreement | yes | mean_TV_disagree=0.159 |
| 8 | hash_lock | yes | agg=fd61cca598044032… |

Notes:

- Gate 7 is a soft ensemble-agreement diagnostic; production is single-model.
- Gate 6 records explicit policy rather than claiming zero high-R occupancy.
- Intervals/centers remain exploratory (not in this checklist).
