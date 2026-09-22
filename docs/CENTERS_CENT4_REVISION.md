# Centers CENT-4 — peer-stratified collective-scope revision

Date: 2026-07-24  
Status: **offline revision complete; confirmatory pilot NO-GO; freeze NO-GO**

## Decision locks

| item | decision |
| --- | --- |
| Scope split peer{8,16} vs peer240 | **adopted** |
| CENT-4 model revision (this package) | **done** |
| Confirmatory paid pilot (768) | **NO-GO** |
| `centers_collective_bundle_v1` freeze | **NO-GO** |
| Centers Stage C | **NO-GO** |

## Scope redesign

```text
centers_collective_bundle_v1
  support: peer ∈ {8,16}
  intended N: {9,17}
  training rows: 1512 (peer8=82, peer16=1430)
  peer240 excluded

centers_dense_exploratory
  support: peer = 240
  rows: 1224
  not used for Stage C / collective surrogate
```

Artifacts:

- `analysis/centers/build_collective_snapshot.py`
- `analysis/centers_branch/collective_scope_v0/`
- `analysis/centers_branch/dense_exploratory_v0/`
- `analysis/centers_branch/collective_scope_split_v0.json`

Missingness is tracked as a separate protocol layer:

\[
P(f\mid \mathrm{valid},x),\qquad p_{\mathrm{fail}}(x)=P(\text{all attempts fail}\mid x)
\]

CENT-3 fieldwise \(p_{\mathrm{fail}}\) is in the revision detail CSV; training-trace
fail rates are joined where recoverable (often 0 on historical Stage B runs).

---

## Candidate set (direct multinomial first)

| model | role |
| --- | --- |
| `peer_specific_global` | required baseline |
| `pooled_global` | old baseline |
| `peer_specific_circular_emd` | primary direct |
| `peer_specific_canonical_emd` | primary direct + descriptors |
| `hierarchical_emd_kernel` | weak peer shrinkage |
| `peer_specific_activity_direction` | hurdle recovery test |
| `descriptor_collapse_mixture` | soft abstention/active gate |
| `pooled_kernel_nn` | reference |

Implementation: `analysis/centers/peer_stratified_models.py`  
Driver: `analysis/centers/run_cent4_revision.py`

---

## Gate results

### Gate 1 — peer-stratified OOF

Completed schemes: profile-family, sparse-realization, offset-group,
acquisition-block, source-family. Peer8/16 reported separately.

### Gate 2 — beat peer-specific global (acquisition-block OOF)

| model | log loss | \(\Delta\)LL vs peer-global (row bootstrap) |
| --- | ---: | --- |
| **peer_specific_activity_direction** | **0.731** | **+0.283** CI [0.264, 0.302] |
| pooled_kernel_nn | 0.771 | — |
| peer_specific_canonical_emd | 0.853 | — |
| peer_specific_circular_emd | 0.903 | — |
| peer_specific_global | 1.034 | baseline |

Peer no-collapse: both peer8 and peer16 improve vs peer-global.  
**Gate 2: PASS**

### Gate 3 — signed action \(a_0\)

| | peer-global | best structured (hurdle) |
| --- | ---: | ---: |
| \(a_0\) MAE | 0.549 | **0.372** |
| sign accuracy | — | **0.854** |
| corr(\(a_0\)) | — | **0.726** |

**Gate 3: PASS** (in-distribution OOF only)

### Gate 4 — finite-peer phenotype (CENT-3 controls)

No candidate reproduces peer16 ±unimodal stay-collapse (observed stay=1;
models typically predict stay≲0.2). Sparse-unimodal high-stay also missed.
Exact antipodal partial activity and 9:7 \(p_+\) are only partially recovered.

`cent_ctrl_sparse_antipodal`: valid=8 → phenotype undecidable (kept).

**Gate 4: FAIL**

### Gate 5 — CENT-3 prospective transfer (required for paid pilot)

| model | CENT-3 log loss | \(a_0\) MAE |
| --- | ---: | ---: |
| **peer_specific_global** | **0.973** | 0.558 |
| peer_specific_circular_emd | 0.996 | 0.530 |
| pooled_global | 1.013 | 0.571 |
| hierarchical_emd_kernel | 1.032 | 0.557 |
| peer_specific_activity_direction (OOF winner) | worse than top-5 | — |

Structured models do **not** clearly beat peer-specific global on CENT-3.  
**Gate 5: FAIL**

### Risk (OOF of OOF-winner)

| metric | value |
| --- | ---: |
| Spearman | +0.193 |
| enrichment | 1.32× |
| peer8 / peer16 Spearman | +0.264 / +0.116 |
| risk≠sparsity-only | yes (proxy) |

Risk gate on OOF: PASS — but does not authorize transfer.

---

## Scientific reading

Hypothesis under test:

> Centers looks unpredictable because different peer-count regimes were
> mixed into one smooth operator.

**Partially supported.** Restricting to peer{8,16} and stratifying experts
produces a large, peer-stable OOF gain over peer-specific global, with
improved \(a_0\). Removing peer240 was necessary for that compressibility.

**Not sufficient for Stage C.** The same OOF winner fails to transfer to
CENT-3 collective-like fields, and finite-peer abstention phenotypes are
not represented in the surrogate. The binding constraint is now
**coverage of finite-peer abstention / collective-like neighborhoods**,
not merely “use circular-EMD instead of hurdle.”

Note: peer8 has only 82 training rows vs 1430 for peer16 — peer8 experts
are fragile; densification remains an offline priority before any paid
pilot.

---

## Confirmatory pilot GO checklist

| # | condition | status |
| ---: | --- | --- |
| 1 | structured ≫ peer-specific global (OOF CI) | PASS |
| 2 | no peer8/16 collapse | PASS |
| 3 | \(a_0\) MAE / sign improved (OOF) | PASS |
| 4 | finite-peer controls reproduced | **FAIL** |
| 5 | CENT-3 prospective transfer | **FAIL** |
| 6 | risk ranks OOF error | PASS |
| 7 | raw retention + uniform reparse tested | forward-fixed; CENT-3 unreparsable |
| 8 | confirmatory fields non-overlapping | not started |

**Confirmatory pilot: NO-GO.** Additional paid spend remains unauthorized.

## Stop rule (unchanged)

The next confirmatory pilot, if eventually authorized, is the last
model-establishment test. A second prospective failure should end the
centers collective-surrogate path rather than indefinite engineering:

> centers representation induces reproducible microscopic transmutation,
> but its response surface was not sufficiently stable or compressible
> for prospective collective prediction

## Commands

```powershell
py analysis/centers/build_collective_snapshot.py
py analysis/centers/run_cent4_revision.py
```

## Next offline (before any paid)

1. Densify peer8 finite-peer abstention neighborhoods (offline synthesis /
   re-weighting — not paid yet).
2. Target stay-collapse regime with descriptor soft gate **only if** direct
   peer-EMD neighborhoods remain insufficient after densification.
3. Re-run Gates 4–5 on CENT-3 without refitting on CENT-3 labels.
4. Only if Gate5+4 pass → draft non-overlapping 768-call confirmatory
   protocol with valid-count top-up rule.
