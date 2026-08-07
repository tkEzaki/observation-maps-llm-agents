# Stage B stimulus-manifold expansion

Date: 2026-07-23 (revised: symmetry-selective interpretation)

## Executive conclusion

Two independent seed blocks measured the three canonical encodings on six
stimulus fields beyond the original unimodal κ-grid.

**Scientific verdict: symmetry-selective polar-channel collapse**
(not “full operator collapse”).

Paper-safe statement:

> Representation-induced response-law transmutation persists across unimodal,
> non-antipodal bimodal, asymmetric and sparse fields. Exact antipodal
> symmetry suppresses the representation-separated polar channels, but does
> not erase representation dependence in activity, even harmonics or the full
> action distribution.

In other words: under non-antipodal fields with polar information,
transmutation is maintained. Under exact antipodal \(\pi\)-rotation
symmetry, first harmonics \(C_1\) vanish together, while \(C_2\), activity,
offset-averaged action bias \(a_0(\mathcal R,\rho)\), and the full trinomial
action law remain representation-dependent.

Stage C remains blocked. Offline surrogate work must incorporate this
symmetry structure before collective runs.

## Frozen design

Representations:

- `intervals_24_decimal6`
- `moments_m1_m3`
- `centers_24_standard`

Stimulus profiles (historical ids; prefer `peer` naming in future protocols):

| id | field | note |
| --- | --- | --- |
| `unimodal_k9` | von Mises κ=9 (anchor) | |
| `bimodal_equal_sep2_k6` | equal bimodal, modes ±1.0, κ=6 | |
| `antipodal_equal_k6` | equal antipodal, κ=6 | \(\pi\)-symmetric |
| `asymmetric_w075_sep2_k6` | weights 0.75/0.25, modes ±1.0, κ=6 | |
| `sparse_N8_unimodal_k6` | unimodal κ=6, **peer_count=8** | rename → `sparse_peer8_*` |
| `sparse_N16_antipodal_k6` | antipodal κ=6, **peer_count=16** | rename → `sparse_peer16_*` |

Sampling: 36 frozen mirror-paired offsets × 12 responses × 2 blocks = 15,552
calls.

## Execution

Primary analyzed pair (used for complex-kernel outputs below):

- Block 1: `.../20260723T130740Z` — 7,776/7,776 valid; $3.1337955; 293.21 s
- Block 2: `.../20260723T131309Z` — 7,776/7,776 valid; $3.1330035; 296.28 s
- Combined analyzed cost: $6.266799

An earlier complete pair also finished (same protocols, independent timestamps):

- Block 1: `.../20260723T125848Z` — $3.1346865
- Block 2: `.../20260723T130338Z` — $3.132954

Total paid stimulus-manifold spend across both complete pairs: about $12.53.
Do not pool the pairs as extra replicates; they are duplicate full blocks.

## Dual-layer gate (polar vs full operator)

The preregistered operational gate used minimum pairwise \(C_1\) chordal
separation (≥0.25) plus within-encoding trajectory replication. That gate
**passed**. Scientifically, \(C_1\) alone mis-labels antipodal fields as
“non-separable” even when the full action operator still differs.

| layer | metric | role |
| --- | --- | --- |
| Polar-channel separability | \(d_{C_1}\) | preregistered; sensitive to polar order |
| Full-operator separability | \(d_{\mathrm{TV}}\), \(d_{\mathrm{JS}}\), and \((C_2,a_0,A)\) | descriptive companion |

Operational gate: **pass**.

Within-encoding field trajectories \((R_1,R_2,a_0)\) correlated across blocks:

- intervals: 0.964
- moments: 0.996
- centers: 0.980

| field | min \(d_{C_1}\) | polar sep. | min mean TV | full-op. sep. (≥0.20 TV) | class |
| --- | ---: | --- | ---: | --- | --- |
| unimodal_k9 | 0.124 | no | (see CSV) | yes* | weak polar / full persists (anchor) |
| bimodal_equal_sep2_k6 | 0.451 | yes | ~0.27+ | yes | polar + full |
| antipodal_equal_k6 | 0.070 | no | ~0.32–0.52 | yes | **symmetry-selective polar collapse** |
| asymmetric_w075_sep2_k6 | 0.317 | yes | ~0.30+ | yes | polar + full |
| sparse_N8_unimodal_k6 | 0.838 | yes | ~0.36+ | yes | strongest polar separation |
| sparse_N16_antipodal_k6 | 0.048 | no | ~0.34–0.60 | yes | **symmetry-selective polar collapse** |

\*Anchor: at unimodal κ=9, intervals and centers both have weak \(R_1\), so
their \(C_1\) distance is small by construction. Phenotype identity remains
in \(R_2\), \(a_0\), activity, and TV/JS.

Non-anchor polar-separable fields: 3 / 5. Antipodal fields fail polar
separability but pass full-operator separability.

## Phenotype picture by field

Across-block means:

### Anchor and sparse unimodal (transmutation persists)

- **unimodal_k9**: moments keep \(R_1\simeq0.95\), \(A=1\); intervals show
  second-harmonic dominance (\(R_2\simeq0.44\)); centers keep negative-odd
  \(a_1\) and negative \(a_0\).
- **sparse_peer8 unimodal** (`sparse_N8_unimodal_k6`): strongest class
  separation (min \(d_{C_1}=0.838\)). Moments: strong polar response
  (\(A=1\)); centers: strong negative odd channel; intervals: intermediate
  \(C_1\) with relatively large \(R_2\). Representation dependence survives
  finite-count sparse observations, not only smooth continuous densities.

### Structured multimode fields

- **bimodal / asymmetric**: all three encodings remain polar-separable.
  Moments stay no-stay directional controllers (\(A=1\)). On equal bimodal,
  intervals \(a_0\) flips **negative**; on asymmetric, centers \(a_0\) is
  strongly negative. Thus \(a_0=a_0(\mathcal R,\rho)\), not an
  encoding-intrinsic torque.
- **antipodal (dense and sparse peer16)**: first-harmonic amplitudes collapse
  for all encodings (\(R_1\lesssim0.13\)). This is the expected consequence
  of \(\pi\)-rotation symmetry of the field
  (\(\rho_\delta=\rho_{\delta+\pi}\)): odd harmonics of a symmetry-respecting
  mean response must vanish. It is **not** evidence that the three operators
  became identical.

### Antipodal endpoints (across-block means)

Dense antipodal:

| encoding | \(R_1\) | \(R_2\) | \(A\) | \(a_0\) |
| --- | ---: | ---: | ---: | ---: |
| intervals | 0.129 | 0.231 | 0.554 | 0.188 |
| moments | 0.023 | 0.106 | 0.075 | 0.016 |
| centers | 0.068 | 0.054 | 0.340 | 0.154 |

Sparse antipodal (peer16): moments \(A\simeq0.04\) vs intervals \(\simeq0.63\)
and centers \(\simeq0.55\); \(R_2\) also differs (centers \(\simeq0.26\),
intervals \(\simeq0.05\)). Mean TV between encodings remains
\(\sim0.32\)–\(0.52\) (dense) and \(\sim0.34\)–\(0.60\) (sparse).

### Moments under antipodal symmetry: abstention, not inactivation

On non-antipodal fields, moments act as a no-stay directional controller
(\(A=1\)). Under equal antipodal fields the first circular moment vanishes and
moments activity falls to \(A\simeq0.04\)–\(0.08\). Prefer the reading
**directional abstention under directional ambiguity** (compress the
symmetry and refuse polar advance/retard) over “response failure.” Histogram
encodings retain substantial activity on the same fields, consistent with
local bin / serialization pseudo-cues.

## Interpretation

1. The object is \(g=g(\mathcal R,\rho)\): encoding and stimulus field
   **interact**, they are not additive corrections.
2. Persistence off unimodal von Mises (bimodal, asymmetric, sparse unimodal)
   argues against a unimodal-only numerical artifact.
3. Antipodal \(C_1\) collapse is a **symmetry-respecting** polar-channel
   shutdown, while full-operator representation dependence persists.
4. \(a_0\) is field-dependent; the unimodal interval+/center− pattern does not
   travel unchanged across \(\rho\).
5. Sparse separation is promising but currently one realization per
   sparsity; realization variance must be measured before strong claims.
6. Collective extrapolation from unimodal-only kernels remains unsafe.

## Research decision

Supported:

- stimulus-manifold dependence of representation-induced response laws;
- persistence of transmutation on non-antipodal polar fields, including sparse
  unimodal;
- symmetry-selective polar-channel collapse under exact antipodal symmetry;
- moments abstention transition as a candidate physics theme.

Not authorized:

- Stage C collective LLM runs;
- claims of macroscopic phase selection;
- “encoding-intrinsic mean torque” independent of \(\rho\);
- treating \(g'(0)\) as established local stability;
- freezing a six-profile-only surrogate as sufficient for arbitrary
  collective trajectories.

Next scientific steps (see `docs/STAGE_B_NEXT_ANALYSIS.md` for revised order):

1. **Offline first (no API):** P3 full-operator gate calibration and P4
   phenotype atlas — `docs/STAGE_B_FULL_OPERATOR_GATE.md`,
   `docs/STAGE_B_PHENOTYPE_ATLAS.md`.
2. **Primary paid:** dense signed weight-imbalance \(\varepsilon\)-sweep
   (not angular detuning first); signed \(C_1\) susceptibility; activity as
   \(|\varepsilon|\) activation.
3. **Integrated sparse:** peer imbalance + multi-realization variance
   (`sparse_peer16_antipodal_*`).
4. Stage 3 surrogate freeze only after (2)–(3); implementation may start now.
5. Stage C only after surrogate OOD policy.

## Artifacts

- Protocols:
  `experiments/stage_b_response_law/protocol_stimulus_manifold_v0_1_block_1.json`,
  `..._block_2.json`
- Stimulus code: `circlemap/stimuli.py`
- Runs (analyzed pair):
  `runs/stimulus_manifold/stimulus-manifold-v0.1-block1_799847a0e9dc/20260723T130740Z`,
  `runs/stimulus_manifold/stimulus-manifold-v0.1-block2_74b53d4dd97b/20260723T131309Z`
- Duplicate complete pair (not pooled):
  `.../20260723T125848Z`, `.../20260723T130338Z`
- Analysis: `analysis/complex_kernel_stimulus_manifold/`
- Tests: `tests/test_stimuli.py`
- Next-analysis plan: `docs/STAGE_B_NEXT_ANALYSIS.md`
