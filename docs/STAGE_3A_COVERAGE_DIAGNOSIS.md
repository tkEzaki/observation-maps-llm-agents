# Stage 3A.2: coverage / OOD diagnosis

Date: 2026-07-23  
Status: **diagnosis complete — do not freeze; do not launch broad paid acquisition yet**

Roadmap position:

```text
3A.1  existing fit                         done
3A.2  OOD cause decomposition              this document
3A.3  model / feature revisit              started (ablation + hurdle)
3A.4  coverage-directed acquisition        selection ready; API not started
3A.5  refit + closed-loop re-eval          blocked on 3A.4
3B    freeze                               blocked
```

Artifacts under `analysis/stage3a_artifacts/`:

| file | content |
| --- | --- |
| `collective_peer_count_scan.csv` | N∈{9,16,17} × (t0 / K=0 / closed-loop) |
| `ood_component_summary.csv` | canonical / native / peer / Policy-A rates |
| `ood_time_trajectories.csv` | \(q_{\mathrm{OOD}}(t)\) components |
| `nearest_training_neighbors.csv` | NN metadata for OOD points |
| `ood_error_calibration.csv` | distance-bin TV / NLL |
| `ood_error_calibration_summary.csv` | Spearman(OOD, error) |
| `feature_ablation_results.csv` | feature masks × models |
| `hurdle_model_comparison.csv` | hurdle vs kernel / logistic |
| `candidate_field_bank.npz` | 50,400 offline candidate fields |
| `active_learning_selection.csv` | 36 coverage-directed picks |

Scripts: `analysis/stage3/diagnose_ood.py`, `calibrate_ood_error.py`,
`feature_ablation.py`, `compare_hurdle_model.py`,
`build_collective_field_bank.py`, `select_active_learning_fields.py`.

---

## 1. Peer-count mismatch is a first-order confounder

Training peer counts in the unified Stage 3A table are **{8, 16, 240}**.
A collective with \(N=16\) observes **15** peers — a value that never appears
in training.

Moments component rates (uniform init):

| \(N\) | peers | regime | \(q_{\mathrm{peer}}\) | \(q_{\mathrm{canonical}}\) | \(q_{\mathrm{combined}}\) (t0 / mean) |
| ---: | ---: | --- | ---: | ---: | --- |
| 9 | 8 | t0 | 0.00 | 1.00 | 1.00 |
| 9 | 8 | K=0 | 0.00 | 0.98 | ~1 |
| 9 | 8 | closed loop | 0.00 | 0.94 | ~1 |
| 16 | 15 | t0 | **1.00** | 0.31 | 1.00 |
| 16 | 15 | closed loop | **1.00** | 0.56 | 1.00 |
| 17 | 16 | t0 | 0.00 | **0.00** | **0.00** |
| 17 | 16 | K=0 | 0.00 | 0.47 | rises |
| 17 | 16 | closed loop | 0.00 | 0.54 | rises |

### Interpretation

1. Previous \(q_{\mathrm{OOD}}\simeq0.96\)–\(0.99\) on \(N=16\) was **inflated by
   peer-count mismatch**, not only by missing field shapes.
2. Matching peers (\(N=17\leftrightarrow 16\)) clears combined OOD at **t=0**.
3. Even with matched peers, **K=0 drift and closed-loop raise canonical OOD**
   (~0.5). That residual is the real manifold-completion target.
4. Matched \(N=9\) (8 peers) is still OOD at t=0 under uniform init: training
   sparse_N8 is structured (unimodal / antipodal), not near-uniform finite-\(N\)
   noise.

**Operational rule:** all future coverage scans and Stage C protocols must use
\(N\in\{9,17,\ldots\}\) so that peer count ∈ {8,16,…}, or else add peer=15
(and other \(N-1\)) explicitly to the training manifold.

---

## 2. Component decomposition

Flags used:

- `ood_canonical`: shape descriptors without peer-size channels
- `ood_native`: representation-native bins / moments
- `ood_peer_count`: nearest training peer count farther than 0.5
- `ood_near_zero`: Policy-A unresolved \(0<|\hat\varepsilon|<0.02\) band
- `ood_combined`: OR of the above

### Time-axis reading (peer-matched \(N=17\))

| pattern | observed |
| --- | --- |
| high at t=0 | **no** (combined = 0) |
| rises under K=0 | **yes** (canonical ~0.47) |
| rises further under closed loop | mild (canonical ~0.54) |
| native-only high | **no** at matched peers (native ≪ canonical) |
| peer-only high | **yes** whenever peers ∉ {8,16,240} |

Conclusion: after fixing peer count, the remaining problem is **canonical field
shape coverage under drift / interaction**, not serialization scale alone.

---

## 3. OOD score is not yet a valid freeze gate

Acquisition-block holdout calibration (kernel-NN predictions):

| representation | distance | Spearman(TV) | Spearman(NLL) |
| --- | --- | ---: | ---: |
| moments | d_canonical | **−0.55** | −0.44 |
| moments | d_native | −0.52 | −0.37 |
| moments | d_peer | **+0.43** | +0.30 |
| intervals | d_canonical | −0.33 | −0.44 |
| intervals | d_peer | +0.33 | +0.22 |

Canonical / native kNN distances currently **anti-correlate** with holdout
error. Peer-count distance correlates positively but is mostly a discrete
size tag.

Therefore:

> \(q_{\mathrm{OOD}}\) from the present combined score must **not** be used as
> a Stage 3B freeze gate until the metric is revised.

Likely causes: duplicate / rotated training mass, thresholding on self-NN
geometry, and mixing peer-size with shape. Next metric work (still offline):

1. representation-specific thresholds;
2. rotation-invariant canonical features only for shape OOD;
3. separate hard reject for unsupported peer counts;
4. recalibrate so \(E[\mathrm{TV}\mid d]\) is monotone increasing.

---

## 4. Model / feature findings (3A.3 start)

### Freeze-gate wording (updated)

Selected model must:

1. beat **global empirical**;
2. be within ~2% log-loss of the best grouped-CV candidate.

**Kernel-NN is an allowed production surrogate**, not a baseline that must be
beaten.

### Hurdle vs kernel (acquisition-block holdout)

| representation | model | log_loss | moments stay gap |
| --- | --- | ---: | ---: |
| moments | kernel_nn | **0.492** | **0.209** |
| moments | hurdle | 0.800 | 0.032 |
| moments | logistic | 0.747 | 0.040 |

Hurdle logistic does **not** yet recover the stay gap; kernel smoothing still
dominates. Keep hurdle as a candidate class (possibly with kernel / stump
stages), but do not replace kernel-NN on this evidence.

### Feature ablation (moments, kernel_nn)

| ablation | log_loss | stay gap |
| --- | ---: | ---: |
| canonical+native | 0.492 | 0.209 |
| drop \(z_2,z_3\) | **0.475** | 0.252 |
| drop \(|z_1|\) block | 0.520 | 0.173 |
| abs_z1 only | 0.680 | **0.710** |
| symmetry only | 0.682 | 0.619 |

Reading:

- \(|z_1|\) carries much of the **abstention** signal (stay gap).
- Full log-loss still improves with broader descriptors; \(|z_1|\) alone is not
  a production model.
- Dropping \(z_2,z_3\) does not hurt holdout log-loss here → matched-control
  for higher harmonics is **not urgent** for predictive performance, but
  remains useful for mechanism claims later.

---

## 5. Coverage-directed field bank (offline)

- Bank size: **50,400** agent-fields from init / K=0 / virtual+surrogate
  kernels at \(N\in\{9,16,17\}\).
- Active-learning pick: **36** fields (maximin + occupancy/OOD/disagreement).
- Selected peers: only **8 or 16** (peer-mismatch down-weighted).
- Near-zero band occupancy among peer-matched bank fields: **~0** in this
  pilot bank → \(\varepsilon\in\{0.005,0.01\}\) top-up is **not first
  priority**.

Recommended first paid pilot (after OOD-metric revision):

\[
24\text{–}48\ \mathrm{fields}\times 3\ \mathrm{encodings}\times 8\ \mathrm{responses}\times 2\ \mathrm{blocks}
=1152\text{–}2304\ \mathrm{calls}.
\]

Do **not** launch that spend until:

1. peer-matched scan protocol is locked (\(N=9/17\));
2. OOD metric shows positive error correlation on holdout;
3. the 36 selected fields are reviewed for stimulus constructibility.

---

## 6. Decision flow (current node)

```text
peer-count matched?  --> partially understood: N=16 was confounded
                         residual canonical OOD remains at N=17 under drift
OOD correlates with holdout error?  --> NO (canonical anti-correlated)
                                         --> fix metric before field spend
```

Next actions (in order):

1. **Revise OOD metric** (shape vs peer hard-gate; recalibrate).
2. Re-run peer-matched closed-loop diagnosis with the revised score.
3. Optionally deepen stump-boost / kernel-hurdle hybrids for moments stay gap.
4. Then run a **1152–2304 call** coverage-directed pilot from
   `active_learning_selection.csv`.
5. Only after coverage + calibration gates pass → Stage 3B freeze.

---

## 3B freeze conditions (revised)

1. Selected model beats global empirical and is ≥ best candidate (within tolerance); kernel-NN allowed.
2. Moments balance / imbalance stay gap adequately reproduced.
3. Sparse-realization holdout passes.
4. OOD score **positively** predicts holdout TV / NLL.
5. Peer-count–consistent collectives (\(N-1\in\) training peers) have acceptable coverage.
6. High-occupancy OOD regions cleared or explicitly rejected.
7. Ensemble agreement on major collective predictions.
8. Hash-lock data, model, OOD policy, collective protocol.
