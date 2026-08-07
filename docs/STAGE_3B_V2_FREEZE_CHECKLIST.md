# Stage 3B moments_bundle_v2 freeze checklist (8 gates)

Date: 2026-07-24  
Grade: **prospectively_frozen_stage_c_v0_2_baseline**  
All gates pass: **True**  
Artifact: `analysis/stage3a_artifacts/stage3b_moments_v2_freeze_checklist.json`  
Parent v1: permanent; not overwritten.

| # | gate | pass | key evidence |
| ---: | --- | --- | --- |
| 1 | grouped_cv_beats_global_near_best | yes | regime=0.426 < global=1.004; within_tol=True |
| 2 | moments_abstention_activation | yes | stay_gap=0.883 |
| 3 | sparse_realization_holdout | yes | sparse regime=0.340 < global=1.098 |
| 4 | prospective_replay_excl_train | yes | stay v1=0.132→v2=0.033 (obs=0.000); exact=0.906 |
| 5 | calibrated_risk_orders_error | yes | Spearman=0.218; monotone=True |
| 6 | peer_matched_Q_risk | yes | Q_risk=0.000; peers={8,16,240} |
| 7 | ensemble_agreement | yes | TV_train=0.173; TV_coll=0.156 |
| 8 | hash_lock | yes | agg=e798662e6e65c47a… |

Naming:

- Before this checklist: **hash-locked selection-grade v2 candidate**
- After all gates pass: **prospectively frozen Stage C v0.2 baseline**
- Paid Stage C v0.2 still requires explicit `--yes` + cost-card approval

Notes:

- Gate 4 is v2-specific (fit excluding replay; evaluate on replay fields).
- Gate 5 uses Spearman > 0.15 (v2 training is more heterogeneous).
- Production model is single `regime_kernel_hurdle`.
