# Stage 3A: representation-wise full-trinomial surrogates

Date: 2026-07-23  
Status: **moments hash-locked** — Stage C v0.1 authorized (selection pilot)

## Split

| Track | Scope | Status |
| --- | --- | --- |
| **Stage 3A** | dataset integration, candidate models, grouped CV, OOD, surrogate collective coverage | **start now** |
| **Stage 3B** | freeze model / OOD policy / collective predictions | only after gates below |

Do **not** treat a fitted candidate as production-ready. Stage C remains blocked
until Stage 3B freeze.

## Design locks (from Stage B P1 + Step D)

1. Predict full \(\widehat{\mathbf p}=(p_-,p_0,p_+)\) per representation — not
   mean action \(g\) alone. Moments abstention needs \(p_{\mathrm{stay}}\).
2. Inputs are **field descriptors**, not labeled \(\varepsilon\):
   \(\mathrm{Re}/\mathrm{Im}/|z_m|\) for \(m=1,2,3\), circular entropy,
   antipodal balance, asymmetry, bimodality, sparsity, peer count, plus
   representation-native bin or moment vectors.
3. \(\varepsilon\) may appear in experiment metadata for diagnostics, but the
   production surrogate must predict from field descriptors alone.
4. Near-zero unresolved band \(0<|\varepsilon|<0.02\): **Policy A** — inflate
   OOD / uncertainty; densify only if closed-loop coverage demands it.
5. Do not over-interpret \(\chi_{a_1}\approx 9.17\) (empirical comparative
   slope, not a precise local derivative) or \(\varepsilon_{1/2}=0.02\)
   (upper-bound grid estimate). Neither is a model input.

## Code

| Path | Role |
| --- | --- |
| `circlemap/field_features.py` | common + native field descriptors; Policy-A flag |
| `analysis/stage3/dataset.py` | unify transmutation / manifold / P1 / sparse-peer |
| `analysis/stage3/models.py` | global empirical, kernel-NN, multinomial logistic L2, stump boost |
| `analysis/stage3/validation.py` | grouped schemes + calibration metrics |
| `analysis/stage3/ood.py` | distance-to-manifold OOD + error calibration bins |
| `analysis/stage3/collective_scan.py` | surrogate-only closed-loop \(q_{\mathrm{OOD}}\) |
| `analysis/stage3/run_stage3a.py` | offline pipeline → `analysis/stage3a_artifacts/` |

Run:

```powershell
py -m analysis.stage3.run_stage3a --skip-stump-boost-cv
```

(Omit `--skip-stump-boost-cv` for the full three-candidate grouped CV.)

## Grouped validation (required)

Response-level random splits are forbidden. Implemented schemes:

- leave-one-profile-out (realization suffix collapsed to family)
- leave-one-\(\varepsilon\)-level-out (weight-sweep rows only)
- acquisition-block holdout
- sparse-realization holdout (whole realization)
- offset-group holdout

## Stage 3B freeze gates (conditional GO; revised)

1. Selected model beats **global empirical** and is within tolerance of the
   best grouped-CV candidate. **Kernel-NN may be the production model.**
2. Moments reproduce high \(p_{\mathrm{stay}}\) at exact balance and low
   \(p_{\mathrm{stay}}\) under small imbalance.
3. Sparse-realization holdout performance holds.
4. Calibrated predictive risk \(R(x)\) **positively** orders holdout /
   grouped-OOF TV (raw distance alone is insufficient).
5. Peer-count–consistent collectives (\(N\in\{9,17\}\)) have acceptable
   occupancy-weighted \(Q_{\mathrm{risk}}\); unsupported peers are hard-rejected.
6. High-occupancy OOD regions are cleared or explicitly rejected.
7. Ensemble agreement on major collective predictions.
8. Hash-lock training data, model, hyperparameters, OOD policy, protocol.

Moments partial freeze: `freeze_ready=true`; 8-gate checklist in
`docs/STAGE_3B_FREEZE_CHECKLIST.md`.
Stage C v0.1 complete; offline diagnostics in `docs/STAGE_C_OFFLINE_V0_1.md`.

See `docs/STAGE_3A_COVERAGE_DIAGNOSIS.md` for the 3A.2 diagnosis (peer-count
confound, broken OOD–error correlation).

Applicability-domain rebuild (3A.3): `docs/STAGE_3A_APPLICABILITY_DOMAIN.md`
(calibrated risk \(R(x)\), kernel-hurdle, v2 36-field reselection; Jaccard vs
v1 = 0). Pilot freeze / human review: `docs/STAGE_3A_PILOT_REVIEW.md`.
Cost card: `docs/STAGE_3A_PILOT_COST_CARD.md` (1,728 calls; **paid complete**).
Prospective: `docs/STAGE_3A_PILOT_PROSPECTIVE.md`.  
Post-refit hash-lock: `docs/STAGE_3B_PARTIAL_FREEZE.md`
(moments / `kernel_hurdle` locked; Stage C needs explicit GO).

## First offline run (2026-07-23)

Artifacts: `analysis/stage3a_artifacts/`
(`training_rows.csv`, `training_arrays.npz`, `stage3a_report.json`).

| item | result |
| --- | --- |
| Training rows | 7992 (2664 per representation) |
| Sources | transmutation + stimulus_manifold + antipodal_weight + sparse_peer |
| Provisional model | **kernel_nn** for all three encodings (beats global; logistic does not yet beat kernel on acquisition-block holdout) |
| Moments stay gap (holdout) | exact-balance stay \(\approx0.61\) vs small-imbalance \(\approx0.40\) (direction correct; calibration still soft vs raw LLM) |
| Closed-loop \(q_{\mathrm{OOD}}\) | \(\approx0.96\)–\(0.99\) on \(N=16\) — **confounded by peer=15 mismatch**; see diagnosis |

Interpretation: Stage 3A scaffolding works, but **Stage 3B freeze is not
authorized**. Next is OOD-metric revision under peer-matched \(N\), then a
small coverage-directed acquisition — not undirected sparse-field flooding.

## Sparse-data interpretation

Sparse panels are valid training data. At exact 8:8, do **not** train toward
intervals–centers polar-channel separation as a goal; full trinomials still
differ and remain useful targets.

## Optional matched control

\(|z_1|\)-matched symmetry controls are **not** required before Stage 3A.
Revisit after surrogate feature ablation shows whether \(|z_1|\) alone suffices.
