# Intervals INT-2b — final bounded offline revision (STOP)

Date: 2026-07-24  
Status: **Intervals collective branch STOPPED; INT-3 NO-GO; microscopic archive retained**

## Decision

| item | result |
| --- | --- |
| Full {8,16} model / cluster / \(a_0\) | **PASS** |
| Peer16-only model / cluster / \(a_0\) | **PASS** |
| Locked support gate (full) | **FAIL** |
| Locked support gate (peer16-only) | **FAIL** |
| Discrete-bin features clear support | **NO** |
| INT-3 paid pilot | **NO-GO** |
| Intervals collective surrogate path | **STOP** |

Locked wording:

> Interval-bin serialization yields an in-distribution-compressible
> microscopic response operator, but the available finite-peer training
> manifold does not support reliable prospective transport to
> collective-like fields.

## Locked support gate (frozen; not retuned post-hoc)

From `SUPPORT_GATE` in `analysis/intervals/run_int2b_support_revision.py`:

| parameter | value |
| --- | ---: |
| \(k\) neighbors | 16 |
| coverage \(d_{\mathrm{NN}}\) threshold | 0.35 |
| local dispersion TV threshold | 0.35 |
| max active-feature-gap fraction | 0.35 |
| max local-noncompressible fraction | 0.35 |
| max gap+noncompress | 0.45 |
| min supported+mixed | 0.40 |

## Support decomposition

### Full collective (n=1512)

| diagnosis | fraction |
| --- | ---: |
| **local_noncompressible** | **0.637** |
| active_feature_gap | 0.313 |
| supported | 0.050 |

Peer × family (counts):

| peer × family | noncompress | gap | supported |
| --- | ---: | ---: | ---: |
| peer16 × sparse_antipodal | 874 | 420 | 76 |
| peer8 × sparse_unimodal | 38 | 36 | 0 |
| peer16 × trajectory_al | 45 | 11 | 0 |
| peer8 × trajectory_al | 4 | 4 | 0 |

**Not** a peer8-only artifact. Peer16 sparse-antipodal dominates and is
mostly locally noncompressible (neighbor responses disagree).

### Peer16-only / N=17 (n=1430)

| diagnosis | fraction |
| --- | ---: |
| local_noncompressible | **0.644** |
| active_feature_gap | 0.303 |
| supported | 0.053 |

Support gate: **FAIL** (same pattern). N=17-only pilot is **not** a rescue.

OOF remains strong (`peer16_activity_direction` log-loss 0.834, cluster CI
positive) — failure is **transport/support**, not IID fit.

## Discrete interval features

Added interval-specific features (occupied/zero runs, peaks, antipodal
overlap, focal/boundary mass, edge asymmetry, boundary perturbation
sensitivity, half-turn cummass). Bake-off:

| model | peer16 AB log-loss |
| --- | ---: |
| activity_direction | **0.834** |
| canonical EMD + discrete | 0.893 |
| Hellinger | 0.904 |
| canonical EMD | 0.908 |

On active-feature-gap rows, discrete features improve local-TV on ~44% of
fields but **mean** local-TV does not improve, and support-class fractions
stay ~5% supported. Discrete features do **not** clear the support gate.

## Why STOP (not another offline loop)

Matches pre-registered stop conditions:

1. Peer16-only still support-FAIL (gap+noncompress ≈ **0.95**)
2. Dominant mode is **local neighbor response dispersion**, not missing neighbors
3. Interval discrete features do not repair support classification
4. Structured gains remain in-distribution OOF only

Further surrogate engineering would repeat the Centers failure mode after
the agreed last offline revision.

## Archive layers

| layer | conclusion |
| --- | --- |
| Microscopic interval transmutation (Stage B) | supported (prior) |
| Peer-stratified in-distribution compressibility | **supported** (INT-2/2b) |
| Prospective collective-field transport | **not supported** under available finite-peer labels |

## Artifacts

- `analysis/intervals/run_int2b_support_revision.py`
- `analysis/intervals_branch/int2b_offline_decision.json`
- `analysis/intervals_branch/int2b_support_detail_full.csv`
- `analysis/intervals_branch/int2b_support_detail_peer16.csv`
- `analysis/intervals_branch/final_manifest.json`
- Cross-rep synthesis: `docs/REPRESENTATION_TRANSPORT_SYNTHESIS.md`

## Explicit non-actions

- INT-3 coverage pilot
- INT-4 confirmatory
- `intervals_bundle_v1` freeze
- matched moments↔intervals Stage C
- further feature/mixture chasing on this manifold

## Research implication

Both **centers** and **intervals** histogram encodings show:

1. clear microscopic representation effects, and  
2. peer-stratified in-distribution compressibility, but  
3. **no** reliable finite-peer → collective-like prospective transport under
   current Stage B / sparse-peer label coverage.

Moments remains the only representation with a frozen Stage C path. See
`docs/REPRESENTATION_TRANSPORT_SYNTHESIS.md` for the locked
compressibility / transportability synthesis. Do not open another histogram
collective surrogate without a materially new research plan.
