# Stage B next analyses (revised execution order)

Date: 2026-07-23 (updated 2026-07-24)  
Status: **Moments Stage C closed; Centers/Intervals STOP; matched-rep collective preregistered (paid blocked)**

## Goal

Move from “three encodings differ on some fields” to encoding-dependent
symmetry-breaking susceptibility:

> How does representation \(\mathcal R\) read the symmetry of \(\rho\), and
> how does it amplify infinitesimal symmetry breaking into directional
> action?

## Recommended execution order (not the old P-number order)

| Step | Work | API? | Status |
| ---: | --- | --- | --- |
| 0 | Manifold reinterpretation (symmetry-selective polar collapse) | no | done |
| **A** | **P3 dual-layer gates + TV calibration** | no | **done** — see `STAGE_B_FULL_OPERATOR_GATE.md` |
| **B** | **P4 phenotype atlas** \(a_0(\mathcal R,\rho)\), etc. | no | **done** — see `STAGE_B_PHENOTYPE_ATLAS.md` |
| **C** | **P1 dense weight-imbalance \(\varepsilon\)-sweep** (signed) | paid | **done** — `STAGE_B_ANTIPODAL_WEIGHT.md` |
| **D** | **Integrated sparse imbalance + realization variance** (P1 sparse ∪ P2) | paid | **done** — `STAGE_B_SPARSE_PEER.md` |
| **E1** | Stage 3A surrogate implementation (trinomial, grouped CV, OOD, coverage) | offline | **GO** — `STAGE_3A_SURROGATE.md` |
| **E2** | Stage 3B surrogate **freeze** | offline | conditional on E1 gates |
| **F** | Stage C collective | paid | blocked until explicit GO (E2 hash-locked) |

Paid experiments correctly start with dense P1, but **A and B must finish
first** so decision rules and baselines are frozen before spending.

---

## Step A — P3 dual-layer gates (offline; calibrate before freeze)

### Layers

| layer | metric | role |
| --- | --- | --- |
| Polar-channel | min pairwise \(d_{C_1}\) ≥ 0.25 | historical preregistered; keep for v0.1 |
| Full-operator | adaptive between/within TV (see below) | companion; do **not** freeze absolute 0.20 yet |

Antipodal exact-symmetric fields:
`symmetry_selective_polar_channel_collapse` when polar fails and full
operator passes.

### Full-operator calibration (required before locking)

Absolute mean TV = 0.20 is **descriptive only**. Before freezing a
protocol rule, calibrate on existing manifold traces:

1. within-representation TV across the two acquisition blocks;
2. between-representation TV (per field);
3. ratio \(\dfrac{d_{\mathrm{between}}^{\min}}{d_{\mathrm{within}}^{\mathrm{mean}}}\);
4. bootstrap CI for pairwise Δ when available.

**Proposed adaptive rule (to validate in calibration output):**

\[
\frac{d_{\mathrm{between}}^{\min}}{d_{\mathrm{within}}^{\mathrm{mean}}}>2
\]

and (when bootstrap Δ is available) the confidence region excludes zero.

Artifacts: `analysis/stage_b_offline_p3_p4/` from
`analysis/calibrate_full_operator_gate.py`.
Narrative lock: `docs/STAGE_B_FULL_OPERATOR_GATE.md`.

**Locked preference:** between/within TV ratio \(>2\) as the adaptive
full-operator rule; absolute TV = 0.20 remains descriptive only.

Serialization audits for exact antipodal / signed-\(\varepsilon\) covariance:
`circlemap/symmetry_audit.py` + `tests/test_stimuli.py` (green).

Do **not** rewrite the historical operational pass/fail of manifold v0.1;
annotate it with the dual-layer classification.

---

## Step B — P4 phenotype atlas (offline)

Publish across-block means:

\[
a_0(\mathcal R,\rho),\quad A(\mathcal R,\rho),\quad
(R_1,R_2)(\mathcal R,\rho),\quad H(\mathcal R,\rho)
\]

for the six manifold fields. Emphasize:

- \(a_0=a_0(\mathcal R,\rho)\) — not encoding-intrinsic torque;
- unimodal interval+/center− signs do **not** globalize;
- antipodal: polar collapse with residual full-operator separation.

CSV: `analysis/stage_b_offline_p3_p4/phenotype_atlas.csv`.  
Narrative table: `docs/STAGE_B_PHENOTYPE_ATLAS.md` (**done**).

---

## Step C — P1 dense weight-imbalance sweep (primary paid)

### Scientific questions

1. Moments abstention only at exact balance \(\varepsilon=0\)?
2. Intervals/centers residual activity / bias at \(\varepsilon=0\)?
3. Shape of response curves \(C_1(\varepsilon)\), \(A(\varepsilon)\)
   (smooth / saturating / threshold-like) — **not** a discontinuity proof?
4. Encoding-dependent **signed** symmetry-breaking susceptibility?

### Axis choice

**Dense weight imbalance only** for the first paid experiment.
Angular detuning \(\pi-\eta\) is deferred: it mixes \(z_1\), \(z_2\), and
local shape, complicating causal interpretation.

Hold κ, mode locations, and separation fixed; vary only polar weight
asymmetry:

\[
w=\tfrac12+\varepsilon,\qquad 1-w=\tfrac12-\varepsilon.
\]

### Signed \(\varepsilon\) grid

Mode swap implies \(\rho_{-\varepsilon,\delta}=\rho_{\varepsilon,\delta+\pi}\),
so under ideal covariance:

\[
C_m(-\varepsilon)=(-1)^m C_m(\varepsilon),
\qquad
C_1(-\varepsilon)=-C_1(\varepsilon),\quad
C_2(-\varepsilon)=C_2(\varepsilon),
\]

with \(a_0\) and offset-averaged activity approximately even in \(\varepsilon\).
Measuring both signs separates true symmetry-breaking response from
encoding pseudo-cues and one-sided advance/retard bias.

**Default grid (recommended):**

\[
\varepsilon\in\{-0.10,-0.05,-0.02,0,0.02,0.05,0.10,0.20\}
\]

**Near-zero densification (if moments activation is sharp; choose after
power simulation on existing trinomials):**

\[
0,\ 0.005,\ 0.01,\ 0.02,\ 0.05,\ 0.10,\ 0.20
\]

plus signed audit points \(\{-0.02,-0.10\}\) at minimum.

### Susceptibility definitions (corrected)

**Do not** use \(\partial_\varepsilon R_1\) as the primary linear
susceptibility (\(R_1=|C_1|\) is nonnegative and positively biased near 0).

Signed first-harmonic susceptibility:

\[
\chi_1=\left.\frac{\partial C_1}{\partial\varepsilon}\right|_{\varepsilon=0}
\quad\text{or projected}\quad
\chi_\parallel=\left.\frac{\partial}{\partial\varepsilon}
\operatorname{Re}\big(e^{-i\psi}C_1(\varepsilon)\big)
\right|_{\varepsilon=0}.
\]

**Activity is even** under symmetry: \(A(-\varepsilon)=A(\varepsilon)\) and
typically \(A'(0)=0\). Report instead:

- one-sided slope \(\partial A/\partial|\varepsilon|\);
- activation scale \(\varepsilon_{1/2}\) where \(A\) reaches mid-rise;
- \(A(\varepsilon)-A(0)\);
- optional cusp / threshold / saturating curve fits.

Even quantities \(C_2\), \(a_0\): prefer curvature / \(|\varepsilon|\)
dependence over raw first derivatives at 0.

### Exact symmetry & serialization audits (unit tests before API)

\(\varepsilon=0\) must be symmetric **after** encoding serialization.

1. **Histogram (intervals/centers):** serialized mass vectors at \(\delta\)
   and \(\delta+\pi\) related by an exact half-turn bin permutation.
2. **Moments:** after rounding to the prompt decimals,
   \(z_m(\delta+\pi)=(-1)^m z_m(\delta)\) at string level.
3. **Small \(\varepsilon\):** distinct \(\varepsilon\) must not collapse to
   identical prompts after rounding.
4. **Sparse exact balance (when used):** prefer antipodal partner pairing
   so 8:8 is not only count-balanced but shape-partnered.

Failing audits → fix serialization / sampling before paying.

### Decision rules (not jump-only)

Compare curve families (descriptive; not discontinuity claims):

- smooth linear / polynomial;
- saturating;
- Hill-type threshold;
- changepoint.

**Primary decision items:**

1. signed \(C_1\) susceptibility \(\chi_1\) / \(\chi_\parallel\);
2. activity activation scale on \(|\varepsilon|\);
3. full-operator distance at \(\varepsilon=0\);
4. encoding × \(\varepsilon\) interaction on response curves;
5. block-wise trajectory replication.

Outcome vignettes (illustrative, not exclusive gates):

| pattern | reading |
| --- | --- |
| Moments \(A(|\varepsilon|)\) rises sharply from near-zero while histograms vary smoothly | abstention + encoding-dependent susceptibility |
| All encodings share similar signed \(C_1(\varepsilon)\) | shared symmetry restoration |
| Large full-operator distance at \(\varepsilon=0\) with tiny \(C_1\) | residual pseudo-cues under exact symmetry |

### Cost sketch

Dedicated dense weight protocol (not bloating the six-field atlas).
Re-estimate from current token prices ×
`3 encodings × n_ε × 36 offsets × n_samples × 2 blocks` before authorization.
Run a **power simulation** from existing antipodal trinomials to choose
\(n_\varepsilon\), near-zero densification, and samples/offset.

---

## Step D — Integrated sparse imbalance + realization variance

Do **not** run separate P1-sparse and P2 campaigns.

### Nested design (per count imbalance)

Ids (rename; peers ≠ collective \(N\)):

- `sparse_peer16_antipodal_8_8`
- `sparse_peer16_antipodal_9_7`
- `sparse_peer16_antipodal_10_6`

For each imbalance:

- \(M=6\)–\(8\) independent realizations (partner-paired at 8:8 when possible);
- 3–4 LLM samples per (realization, offset) — prefer more realizations over
  12 samples on one realization;
- 2 acquisition blocks.

Variance decomposition:

1. LLM sampling variance;
2. sparse realization variance;
3. between-encoding variance.

Gate: between-encoding separation ≫ within-encoding realization scatter
before using sparse fields as Stage-3 training signal.

---

## Optional follow-up — Moments matched control

After dense P1 (or as a small add-on), compare stimuli with matched
\(|z_1|\) but different \((z_2,z_3)\) / field shape:

- if moments track antipodal symmetry beyond \(|z_1|\) → symmetry reading;
- if moments only threshold on \(|z_1|\) → simpler directional gate
  (still interesting; different claim).

---

## Step E1 — Stage 3A (GO now)

See `docs/STAGE_3A_SURROGATE.md`. Dense P1 and nested sparse panel are in
the training manifold. Implement full-trinomial per-representation
surrogates with field descriptors (not \(\varepsilon\)-only), grouped
validation, Policy-A near-zero OOD inflation, and closed-loop coverage
\(q_{\mathrm{OOD}}\).

## Step E2 — Stage 3B freeze (conditional)

**Done (partial):** moments / `kernel_hurdle` hash-locked in
`analysis/stage3a_artifacts/moments_bundle_v1/`
(`docs/STAGE_3B_PARTIAL_FREEZE.md`). Intervals/centers remain exploratory.

## Step F — Stage C

**v0.1 complete** (qualitative OK; stay miscalibrated).  
Prompt audit **PASS**: `docs/STAGE_C_PROMPT_CONTRACT_AUDIT.md`.  
v1 dense-\(K\) envelope (offline): `docs/STAGE_C_DENSE_K_V1.md`.

**Replay complete (branch A):** `docs/STAGE_C_REPLAY_RESULTS.md`  
**v2 prospectively frozen (8/8):** `docs/STAGE_3B_V2_FREEZE_CHECKLIST.md`  
**v1 vs v2 dense-\(K\):** `docs/STAGE_C_DENSE_K_V1_VS_V2.md` (+ rich compare)  
**v0.2 design locked:** `docs/STAGE_C_V0_2_PROTOCOL.md`  
**v0.2a PAID COMPLETE:** `docs/STAGE_C_RESULTS_V0_2A.md`  
**v0.2b PAID COMPLETE:** `docs/STAGE_C_RESULTS_V0_2B.md`  
**v0.2 COMBINED (moments branch closed):** `docs/STAGE_C_RESULTS_V0_2_COMBINED.md`  

Moments Stage C paid expansion is **not required**.  
**Centers collective path: STOPPED** — `docs/CENTERS_FINAL_DECISION.md`.  
**Intervals collective path: STOPPED** — `docs/INTERVALS_FINAL_DECISION.md`.  

**Synthesis locked:** `docs/REPRESENTATION_TRANSPORT_SYNTHESIS.md`  
(figures: `py analysis/synthesis/build_synthesis_figures.py`).

**Next scientific / paid priority:** surrogate-free matched representation
collective (the study path) — **not** Centers/Intervals surrogate reopen:

| matched-rep step | status |
| --- | --- |
| Paid N=17×4K×6seeds×3reps | **done — Outcome A** |
| Analysis revision v0.2 (\(Q_2\), sustained lock, exact sign, I−C) | **done** — `MATCHED_REP_COLLECTIVE_RESULTS.md` |
| 3×3 cross-encoding replay | **scientific GO; paid pending** — cost/human review |
| Second-model R1/R2 | after replay (or parallel planning) |

| centers step | status |
| --- | --- |
| CENT-0…4 final offline | **done / STOP** — `docs/CENTERS_FINAL_OFFLINE.md` |
| Centers confirmatory / freeze / Stage C | **cancelled** |
| Final manifest | `analysis/centers_branch/final_manifest.json` |

| intervals step | status |
| --- | --- |
| INT-0/1 snapshot + serialization | **done** |
| INT-2 offline gates | model/risk PASS; support FAIL |
| INT-2b final offline revision | **STOP** — `docs/INTERVALS_FINAL_DECISION.md` |
| INT-3/4/5/6 paid / freeze / Stage C | **cancelled** |
| Final manifest | `analysis/intervals_branch/final_manifest.json` |

**Histogram collective paths (centers + intervals): closed.**  
Moments Stage C remains the sole frozen collective representation path.
Do not reopen histogram surrogate engineering on the current finite-peer
label manifold without a materially new research plan.

---

## Explicit non-goals for the next paid step

- Angular detuning as the first paid axis.
- Many unrelated multimodal fields without symmetry control.
- Treating antipodal \(C_1\) collapse as falsification of transmutation.
- Encoding-intrinsic torque from unimodal \(a_0\) alone.
- Jump/discontinuity as the sole preregistered success criterion.
- Freezing absolute TV = 0.20 without between/within calibration.

## Authorization checklist (before any new API spend)

- [x] P3 calibration reviewed; full-operator rule chosen (prefer between/within TV > 2)
- [x] P4 atlas reviewed
- [x] Serialization / antipodal unit tests green (`circlemap/symmetry_audit.py`)
- [x] Signed \(\varepsilon\) grid + sample sizes fixed by power simulation
- [x] Cost card approved (`docs/STAGE_B_P1_COST_CARD.md`, ~$8.36 / 20,736 calls)
- [x] Explicit user authorization to run paid P1 dense weight sweep (2026-07-23)
