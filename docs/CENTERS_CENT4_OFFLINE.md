# Centers CENT-4 — offline revision package (A/B/C)

Date: 2026-07-24  
Status: **A/B/C complete; `centers_bundle_v1` freeze NO-GO; no new paid spend**

## Decision locks

| item | decision |
| --- | --- |
| CENT-3 as AD / risk prospective test | **PASS** (unchanged) |
| CENT-3 as surrogate freeze evidence | **FAIL** |
| `centers_bundle_v1` freeze | **NO-GO** |
| Centers collective Stage C | **NO-GO** |
| Additional paid pilot | **conditional** on model revision |
| This package | offline only |

Recommended path remains:

```text
invalid audit → pre-refit bake-off → control comparability
  → revise model family → refit including CENT-3
  → grouped / peer holdouts → recalibrate risk
  → new non-overlapping confirmatory pilot → freeze
```

Do **not** treat in-sample fit on the same 36 CENT-3 fields as freeze evidence.

---

## A. Parse / missingness audit

Scripts / artifacts:

- `analysis/centers/audit_cent3_missingness.py`
- `analysis/centers_branch/cent4a_missingness_summary.json`
- `analysis/centers_branch/cent4a_invalid_by_field_block.csv`
- `analysis/centers_branch/cent4a_call_parse_audit.csv`

### Headline

| quantity | value |
| --- | ---: |
| Calls | 1,152 |
| Strict invalids | **73** (6.3%) |
| `cent_ctrl_sparse_antipodal` valid | **8 / 32** |
| Phenotype gate on that control | **undecidable** (keep field) |

### Raw-text recoverability

All 73 invalids have `raw_response=null`. Token counters show each failed
task used 3 attempts × 16 completion tokens, so the backend produced output
that failed strict JSON parse, but the text was discarded by `run_task`.

**Pipeline fix (forward-only):** `experiments/stage_b_response_law/run.py` now
retains `last_raw_response` on failure. CENT-3 itself cannot be reparsed
without re-acquisition. If raw text later exists, apply the deterministic
`classify_raw` rules uniformly to all fields; keep strict counts; never
hand-rescue one control.

### Invalid structure

| class (recoverable taxonomy) | n |
| --- | ---: |
| `raw_unavailable` | 73 |

Field-level associations:

| correlation | Spearman |
| --- | ---: |
| invalid rate vs bin sparsity | **+0.756** |
| invalid rate vs \(R_{\mathrm{pre}}\) | +0.347 |
| invalid rate vs peer count | +0.284 |
| invalid rate vs \(p_{\mathrm{stay}}^{\mathrm{pre}}\) | ~0 |

Mean invalid rate: peer8 ≈ 0.042, peer16 ≈ 0.101. Missingness is
concentrated on sparse / antipodal controls — not action-independent noise.

---

## B. Pre-refit prospective bake-off

Script / artifacts:

- `analysis/centers/bakeoff_cent3_prospective.py`
- `analysis/centers_branch/cent4b_bakeoff_summary.json`
- `analysis/centers_branch/cent4b_bakeoff_detail.csv`

Fit uses only `training_arrays_centers_v0.npz` (2,736 rows; no CENT-3).
Production hurdle weights loaded from disk when present.

Action channels:

- \(A=p_++p_-\)
- \(a_0=p_+-p_-\)
- \(C_1=|a_0|\)
- \(C_2=p_{\mathrm{stay}}\) (trinomial second channel; \(|e_{C_2}|=|e_A|\) when \(A+p_{\mathrm{stay}}=1\))

### Ranking by mean log-loss (36 fields)

| model | log loss | \(d_{\mathrm{TV}}\) | \(\|e_A\|\) | \(\|e_{a_0}\|\) |
| --- | ---: | ---: | ---: | ---: |
| **global_empirical** | **1.053** | 0.453 | 0.209 | 0.597 |
| **hybrid_circular_emd** | **1.059** | 0.467 | 0.221 | 0.620 |
| hybrid_hellinger | 1.074 | 0.473 | 0.222 | 0.625 |
| softmax_stump_boost | 1.083 | 0.456 | 0.169 | 0.645 |
| activity_direction_hurdle | 1.179 | 0.481 | **0.161** | **0.719** |
| kernel_nn_euclidean | 1.698 | 0.503 | 0.216 | 0.744 |
| soft_regime_mixture | 17.77 | 0.487 | 0.733 | 0.574 |

### Branch read

**Primary:** `coverage_or_feature_insufficiency`  
(`global_empirical` beats every structured model; all structured \(d_{\mathrm{TV}}>0.35\))

**Secondary (also true):**

1. Native / hybrid kernels beat the hurdle → hurdle is not the right freeze candidate.
2. Direction errors remain large (\(\overline{|e_{a_0}|}\approx0.6\)–0.74); hurdle has best activity among structured models but worst \(a_0\) among the competitive set.
3. Peer8 vs peer16 log-loss gap is large on average (driven especially by Euclidean NN / mixture), motivating peer-stratified experts.

Soft regime mixture is numerically unusable on this prospective set (exclude as production candidate).

---

## C. Control comparability

Script / artifacts:

- `analysis/centers/audit_cent3_control_comparability.py`
- `analysis/centers_branch/cent4c_control_comparability.json`
- `analysis/centers_branch/cent4c_control_table.csv`

### Safe wording (locked)

> At peer count 16 and the frozen focal-frame serialization, both
> polarity-reversed unimodal controls collapsed to abstention rather than
> expressing opposite directional responses.

### Comparability classification

| comparison | result |
| --- | --- |
| CENT-3 ±unimodal (peer16, offset=0) vs dense κ=9 sign-reversing (peer240, multi-offset Fourier) | **not same condition** → `new_peer_regime_phenotype` |
| CENT-3 sparse unimodal (peer8) vs historical `sparse_N8_unimodal_k6` | peer-matched; historical stay≈0.79 near mode → **consistent with finite-peer abstention** |
| Dense historical local polarity across mode (±0.13 rad) | sign flip present; stay near 0 |
| CENT-3 polarity pair | both \(p_{\mathrm{stay}}=1\), \(a_0=0\) |

Do **not** mark this as “known dense phenotype gate failure.”

---

## Model-revision priorities (next offline, not yet executed)

Order implied by A–C:

1. **Direct multinomial native kernel** (circular-EMD hybrid is the best structured candidate).
2. **Peer-count stratified experts** (8 / 16 / 240), especially to protect peer16 unimodal abstention from dense averaging.
3. Soft activity-collapse gates using descriptors + native bins only (no field IDs).
4. Recalibrate \(R_{\mathrm{centers}}^{\mathrm{new}}\) from grouped OOF residuals of the chosen family — do not inherit Spearman 0.288.

Then: confirmatory ~768-call pilot on **non-overlapping** fields before any freeze.

## Commands

```powershell
py analysis/centers/audit_cent3_missingness.py
py analysis/centers/bakeoff_cent3_prospective.py
py analysis/centers/audit_cent3_control_comparability.py
```
