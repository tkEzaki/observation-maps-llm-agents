# Stage 3A.3: applicability-domain rebuild

Date: 2026-07-23  
Status: **AD + predictive risk rebuilt; v2 field selection ready; paid pilot not authorized**

This supersedes distance-only `ood_combined` as the freeze gate. Companion to
`STAGE_3A_COVERAGE_DIAGNOSIS.md` (3A.2 confound diagnosis).

## What changed

| Old | New |
| --- | --- |
| Single OR-ed OOD flag | Four layers: peer hard gate / shape / native / Policy-A |
| Shape mixed with \(\arg z_m\) | Shape = \((|z_m|, H, S_{\mathrm{anti}}, S_{\mathrm{bimodal}}, \mathrm{sparsity})\) |
| Native Euclidean shared | Hellinger (hist) / standardized kNN (moments) |
| Acquisition-block-only calibration | Grouped OOF residuals (profile / ε / sparse / offset / block) |
| Raw distance as freeze gate | Calibrated predictive risk \(R(x)\) |
| v1 36-field AL list | **v2 reselection** (Jaccard vs v1 = **0**) |

## Layer definitions

1. **\(G_{\mathrm{peer}}\)**: hard reject if \(n_{\mathrm{peer}}\notin\{8,16,240\}\).
   Collective candidates: \(N\in\{9,17\}\) for now.
2. **Shape support**: density-normalized kNN on rotation-invariant features.
3. **Native support**: Hellinger (intervals/centers) or standardized moment NN.
4. **Policy-A**: unresolved near-zero band flag (not a distance).

Freeze-facing rejects are recorded separately:

```text
unsupported_peer
high_predicted_risk
near_zero_policy_A
```

## Predictive risk

From grouped OOF (kernel predictions as primary error):

\[
R(x)=w^\top \phi\bigl(d_{\mathrm{shape}},d_{\mathrm{native}},\rho_{\mathrm{local}},u_{\mathrm{ens}},d_{\mathrm{orient}},\mathrm{PolicyA}\bigr)
\]

fitted by ridge on \(\log(1+\cdot)\) features targeting \(e^{\mathrm{TV}}\).

| representation | Spearman\((R,e_{\mathrm{TV}})\) | monotone bin means |
| --- | ---: | --- |
| intervals | 0.359 | yes |
| moments | **0.484** | yes |
| centers | 0.404 | yes |

Raw canonical distance previously anti-correlated with error; **calibrated risk
orders errors positively** and is the correct freeze-facing score.

Artifacts: `oof_residual_table.csv`, `risk_calibrator_*.json`,
`risk_rediagnosis_*.csv`, `applicability_rebuild_summary.json`.

## Model update: kernel-hurdle

Logistic hurdle remains weak. **Kernel-hurdle** (kernel stay gate on
\(|z_1|\), antipodal balance, sparsity, peer count + kernel direction on full
\(x\)) improves moments substantially on acquisition-block holdout:

| model | moments log_loss | moments stay gap |
| --- | ---: | ---: |
| kernel_nn | 0.492 | 0.209 |
| **kernel_hurdle** | **0.373** | **0.749** |
| stump boost | 0.504 | 0.584 |

Kernel-hurdle is now a serious production candidate for moments; kernel-NN
remains competitive for histogram encodings. Selection rule unchanged: beat
global empirical and stay within tolerance of the best grouped-CV model.

Safe mechanism wording (unchanged in spirit):

> Moments encoding implements a strong activity gate associated primarily with
> the magnitude of the first circular moment, while broader descriptors remain
> necessary for the full action distribution.

Matched-control is still not a Stage 3B blocker.

## Field bank audit

| source | count |
| --- | ---: |
| A_init | 2,016 |
| B_k0_drift | 6,048 |
| C_virtual | 42,336 |

Virtual generators are diverse (polar / 2nd harmonic / negative-odd / soft /
kernel-NN / hurdle / stump). v2 selection is not locked to one surrogate:
selected mix includes K=0 (11), virtual kernels, and a few inits.

## Active-learning v2

File: `active_learning_selection_v2.csv` (36 fields).

| allocation | n |
| --- | ---: |
| high occupancy × high risk | 18 |
| high disagreement | 8 |
| K=0 drift | 6 |
| maximin boundary | 4 |

- Jaccard vs v1 selection: **0** (old list must not be acquired).
- Peer counts in v2: 8 or 16 only.
- Pilot size if authorized: \(36\times3\times8\times2=1728\) calls
  (`docs/STAGE_3A_PILOT_COST_CARD.md`).

## Next (still offline until authorization)

1. Human review of v2 fields for stimulus constructibility.
2. Freeze protocol JSON + hash selection metadata.
3. `--estimate-only` then explicit paid authorization.
4. After pilot: refit, recalibrate \(R\), recompute \(Q_{\mathrm{risk}}\) at
   \(N=9,17\), then Stage 3B freeze decision.

**Stage 3B / Stage C remain blocked.**
