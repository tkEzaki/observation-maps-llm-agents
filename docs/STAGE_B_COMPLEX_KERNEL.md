# Stage B complex-kernel reanalysis

Date: 2026-07-23 (revised: phase averaging fix + terminology)

## Executive conclusion

Offline reanalysis of the frozen transmutation and feature-sweep traces
supports a **representation-induced microscopic response-law
transmutation**.

Operational definition used here:

> We use *transmutation* to denote a reproducible change in the complex
> harmonic coefficients and action distribution induced by changing the
> observation encoding while holding the underlying field, model, response
> contract, and generation settings fixed.

The paper-safe claim is:

> Different encodings of the same underlying unimodal von Mises field
> reproducibly transform the amplitude, phase, harmonic content, and
> offset-averaged directional bias of the microscopic response operator
> implemented by a frozen language model and prompt contract.

This is shown at the level of complex harmonics \(C_m=a_m+ib_m\), not
\(a_1\) alone. Different macroscopic collective phases have **not** been
demonstrated. The fitted mean-response derivative at zero offset,
\(g'(0)=\sum_m m a_m\), and fitted fixed-point counts are **diagnostic
only**: they flip with Fourier truncation order and are not primary
endpoints. Moreover, when even harmonics or a nonzero \(a_0\) are present,
\(\delta=0\) need not be a fixed point of the fitted curve, so \(g'(0)\)
is not an aligned-state stability criterion by itself.

No new API calls were made for this report.

## Evidence levels

| Claim | Status |
| --- | --- |
| Phase-dependent response | Established |
| Independent seed-block replication | Established |
| Representation dependence | Established |
| Three distinct complex harmonic trajectories | Established |
| Systematic precision / bin-count control | Strongly supported |
| Representation-separated \(a_0\) sign pattern | Strongly supported as a microscopic operator summary |
| Discontinuous bifurcation | Not established |
| Individual fixed points / local stability | Not established |
| Pairwise nonreciprocity | Not tested |
| Macroscopic phase selection by representation | Not tested |
| Collective rotation / social torque | Not tested |
| Cross-model generality | Not tested |

## Terminology lock

Use:

- odd / alignment channel: \(a_m\)
- even / phase-shifted channel: \(b_m\)
- full harmonic: \(C_m=a_m+ib_m\), or \((R_m,\phi_m)\) with
  \(\phi_m=\operatorname{atan2}(b_m,a_m)\)
- **offset-averaged action bias** (zeroth harmonic / mean-action
  component): \(a_0\) after robustness checks — *not* collective mean torque
- activity: \(A=1-p_{\mathrm{stay}}\)
- action entropy (nats, unnormalized):
  \[
  H=\left\langle
  -\sum_{s\in\{-,0,+\}} p_s(\delta)\log p_s(\delta)
  \right\rangle_\delta
  \]
  with natural log and a uniform average over the design offset grid
  (not density-reweighted; not divided by \(\log 3\))
- encoding phrase: **different encodings of the same underlying unimodal
  von Mises field**
- moments phenotype: **no-stay directional controller with low output
  entropy** (operational \(p_{\mathrm{stay}}=0\); not fully deterministic
  advance/retard — \(H\simeq0.15\)–\(0.22\))

Avoid:

- blanket “semantically equivalent” for bin-count or moment encodings
- negative \(a_1\) as global repulsion
- dominant \(a_2\) as selected two-cluster collective phases
- \(a_1\approx0\) as “no first harmonic” when \(b_1\) is large
- \(g'(0)\) or `n_fixed_points` as established physics
- “aligned-state local slope” for \(g'(0)\)
- treating bootstrap-ellipse non-overlap as a formal pairwise test
- reporting \(\phi_m\) when \(R_m\) is weak (see below)

### Complex averaging rule (mandatory)

Across blocks (and any multi-sample aggregation), average complex
coefficients first:

\[
\bar C_m=\frac{1}{B}\sum_{b=1}^{B} C_{m,b},\qquad
R_m=|\bar C_m|,\qquad
\phi_m=\arg\bar C_m.
\]

Never average phases (or radii) arithmetically across blocks. An earlier
draft incorrectly averaged \(\phi_1\) and \(R_1\) separately, producing an
internal inconsistency at intervals \(\kappa=9\)
(\(a_1,b_1\) in QIII but \(\phi_1\approx49^\circ\)). Correct
\(\arg\bar C_1\approx-124^\circ\); with \(R_1\approx0.05\) the phase is now
**not reported**.

### Weak-\(R\) phase rule

Report \(\phi_m\) only when:

1. \(R_m\ge0.15\), and
2. the block-conditional 95% bootstrap ellipse for \(C_m\) does not contain
   the origin (when an ellipse is available).

The amplitude threshold is a **visualization and reporting convention**
rather than an inferential endpoint. Statistical caution about weak
harmonics is carried primarily by whether the bootstrap ellipse covers the
origin; the \(0.15\) cutoff simply withholds unstable phases from tables and
prose. Otherwise mark phase as undefined / suppressed. Weak-amplitude points
must not be treated as reliable waypoints on a continuous rotation orbit;
show block-wise complex-plane points and ellipses instead.

## Primary phenotype table (complex channels)

Across-block means at Fourier order \(M=6\), using mean-complex-then-polar.
Phases marked `—` fail the weak-\(R\) rule.

| class | \(\kappa\) | \(a_1\) | \(b_1\) | \(R_1\) | \(\phi_1\) | \(R_2\) | \(a_0\) | \(A\) | \(H\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| intervals | 2 | 0.665 | 0.012 | 0.665 | 1° | 0.110 | 0.141 | 0.984 | 0.488 |
| intervals | 4 | 0.421 | 0.037 | 0.423 | 5° | 0.042 | 0.080 | 0.981 | 0.558 |
| intervals | 6 | 0.251 | 0.206 | 0.325 | 39° | 0.080 | 0.180 | 0.978 | 0.546 |
| intervals | 9 | −0.026 | −0.039 | 0.048 | — | 0.435 | 0.249 | 0.970 | 0.486 |
| intervals | 12 | 0.082 | 0.157 | 0.177 | 62° | 0.367 | 0.140 | 0.976 | 0.526 |
| moments | 2 | 0.981 | −0.004 | 0.981 | 0° | 0.449 | 0.028 | 1.000 | 0.202 |
| moments | 4 | 0.984 | −0.220 | 1.008 | −13° | 0.453 | 0.101 | 1.000 | 0.172 |
| moments | 6 | 0.856 | −0.207 | 0.880 | −14° | 0.607 | 0.097 | 1.000 | 0.218 |
| moments | 9 | 0.977 | 0.021 | 0.977 | 1° | 0.459 | −0.033 | 1.000 | 0.156 |
| moments | 12 | 0.998 | −0.018 | 0.998 | −1° | 0.388 | −0.004 | 1.000 | 0.203 |
| centers | 2 | 0.682 | 0.303 | 0.746 | 24° | 0.129 | −0.287 | 0.988 | 0.346 |
| centers | 4 | 0.235 | 0.063 | 0.243 | 15° | 0.204 | −0.368 | 0.991 | 0.457 |
| centers | 6 | −0.045 | 0.092 | 0.102 | — | 0.237 | −0.340 | 0.992 | 0.494 |
| centers | 9 | −0.142 | 0.135 | 0.196 | 136° | 0.108 | −0.253 | 0.982 | 0.487 |
| centers | 12 | −0.392 | 0.153 | 0.421 | 159° | 0.215 | −0.288 | 0.973 | 0.431 |

Bootstrap 95% ellipses for \(C_1\) remain a useful **visual** summary of
within-block multinomial uncertainty. They are **conditional on the two
independent acquisition blocks** and are **not** treated as a formal
pairwise significance test. Prefer the block-stratified \(\Delta C_1\)
bootstrap (below). Outputs: `complex_ellipses.csv`,
`delta_c1_bootstrap_tests.csv`.

Key readings:

- intervals: dominant-harmonic **handover** (\(R_1\) down, \(R_2\) up near
  \(\kappa=9\)); at \(\kappa=6\), \(C_1\) is shrinking and rotating
  (\(\phi_1\simeq39^\circ\)). The \(\kappa=9\) point is near the origin
  (\(R_1\approx0.05\)) — phase suppressed; do not interpolate a continuous
  \(1^\circ\to62^\circ\) rotation through that weak-amplitude sample;
- moments: no-stay directional controller (\(A=1\), never `stay`) with
  low but nonzero output entropy;
- centers: complex-plane rotation from positive-odd toward negative-odd,
  with material \(b_1\) throughout; \(\kappa=6\) phase suppressed
  (\(R_1\approx0.10\)).

## \(a_0\) robustness

Fourier intercepts over \(M\in\{2,4,6,8,12\}\), unweighted means,
arc-length-weighted circular means (periodic trapezoidal quadrature over
\(\delta\); not stimulus-density \(\rho(\delta)\) weighting), and
mirror-even means all preserve:

- interval \(a_0>0\)
- center \(a_0<0\)
- moments: substantially smaller and **sign-inconsistent** \(a_0\)
  (about \(0.10\) at \(\kappa=4,6\); near zero or slightly negative at
  \(\kappa=9,12\)) — not “\(a_0\simeq0\)” in a strong sense, but clearly
  not the signed interval/center pattern

Fourier-order ranges for \(a_0\) stay small (~0.005–0.029). This sign
separation is a robust microscopic **offset-averaged action bias** on the
present unimodal manifold. It is still not a measured collective social
torque: local-field offsets in a collective need not be uniform on the
circle.

## Representation distances

Between-representation distances (per block) now include response \(L^2\),
low-order complex Euclidean, \(C_1\)/\(C_2\) chordal, mean TV, mean
Jensen–Shannon (nats), and symmetric KL (with probability floor).
Within-representation distances across the two acquisition blocks provide
a baseline; ratios are in `distance_between_within_ratios.csv`.

Illustrative across-block picture at selected \(\kappa\) (block-mean of
between-class metrics; see CSV for full detail):

| \(\kappa\) | pair | response \(L^2\) | \(C_1\) chordal | mean TV | mean JS |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | intervals–moments | ~0.58 | ~0.32 | ~0.23 | (see CSV) |
| 9 | intervals–moments | ~1.01 | ~1.01 | ~0.42 | ~0.20 |
| 9 | intervals–centers | ~1.03 | ~0.21 | ~0.45 | ~0.18 |
| 9 | moments–centers | ~1.14 | ~1.13 | ~0.49 | ~0.23 |

At \(\kappa=9\), interval–center \(C_1\) distance is small while response
\(L^2\) and TV/JS remain large: low-order harmonics alone do not capture the
full operator difference. Quadrature for \(L^2\) is an
**arc-length-weighted circular integral over \(\delta\)**, evaluated by
periodic trapezoidal quadrature on the design offset grid. It corrects for
uneven spacing of design offsets; it is **not** a von Mises stimulus-density
\(\rho(\delta)\) weight.

As an **auxiliary** check (not a primary claim of family-wise significance),
a block-stratified bootstrap of
\(\Delta C_1=\bar C_1^{(\mathcal R_1)}-\bar C_1^{(\mathcal R_2)}\)
resamples responses within each acquisition block, forms block-wise \(C_1\),
then averages the two blocks within each replicate. In this dataset, all 15
representation-pair × \(\kappa\) ellipses exclude the origin
(`contains_origin=0`). Interpret as **conditional on the two acquisition
blocks**, without multiple-comparison adjustment; overall representation
dependence is already supported by independent replication and trajectory
separation. Max-statistic / Holm / FDR corrections would be needed before
treating the 15 cells as jointly significant.

Sources: `representation_distances.csv`,
`within_representation_distances.csv`,
`distance_between_within_ratios.csv`,
`delta_c1_bootstrap_tests.csv`.

## Feature sweep (complex form)

Bin count remains the strongest feature axis: at \(\kappa=12\), \(C_1\) moves
from a weak negative-odd channel (12–18 bins) through a weak transitional
24-bin region into a strong positive polar channel (30–48 bins). Precision
reorganizes odd channel, phase, and \(a_0\) together, especially 3→4 decimals.
Origin shift exceeds the pure coordinate-covariance null (max \(7.5^\circ\)) but
is secondary to bin count.

Safe claim: precision and observation resolution control the LLM response
operator largely and reproducibly. Not claimed: a mathematical bifurcation at a
critical bin count or precision.

## Diagnostic demotion: \(g'(0)\) and fixed points

The quantity

\[
g'(0)=\sum_{m\ge1} m a_m
\]

is the **derivative of the fitted mean-response curve at zero offset**.
It is retained only as a truncation diagnostic. With even components and a
nonzero constant term,

\[
g(0)=a_0+\sum_m b_m
\]

need not vanish, so \(\delta=0\) is not generally a fixed point of the fit.

Interval examples (Block 1), \(g'(0)\) by \(M\):

\(\kappa=2\): \(0.845,\ 1.142,\ 0.755,\ -0.178,\ 2.574\)

\(\kappa=9\): \(0.970,\ 0.545,\ 1.303,\ -0.484,\ -3.359\)

Signs flip. Fitted zero counts also rise with \(M\) (typically 2–10),
consistent with jagged finite-order approximation rather than established
multi-stability. Do not conclude aligned-state stability from \(M=6\)
\(g'(0)\) or from `n_fixed_points`. Fixed-point counts stay out of the main
text; supplementary numerical diagnosis only.

Dedicated local stability work would need dense near-\(\delta=0\) measurement,
pair/collective dynamics, or regularized local regression—not claimed here.

Source: `gprime0_order_sensitivity.csv`.

## Research decision

Supported now:

- reproducible microscopic response-law transmutation across three encodings
  of the same unimodal von Mises field;
- complex first-/second-harmonic primary reporting with weak-\(R\) phase
  discipline and mean-complex averaging;
- robust interval-positive / center-negative offset-averaged action bias;
- feature-controlled continuous/sharp crossover, especially by bin count;
- representation distances on \(g\), \(C_m\), and full trinomials, with
  within-class baselines and block-stratified \(\Delta C_1\) auxiliary checks.

Not supported / not authorized:

- macroscopic phase selection;
- discontinuous bifurcation;
- aligned-state stability from current \(g'(0)\);
- Stage C collective LLM runs;
- paid boundary top-up from this pass alone.

## Deferred roadmap (not authorized by this pass)

1. Antipodal symmetry-breaking \(\varepsilon\)-sweep and sparse realization
   variance — see `docs/STAGE_B_NEXT_ANALYSIS.md` (highest priority after
   manifold reinterpretation).
2. Offline stochastic surrogate on full \((p_+,p_0,p_-)\) with frozen
   predictions of \(r_1,r_2,r_3,\Omega_{\mathrm{coll}},\tau_{\mathrm{social}}\),
   gated by leave-one-profile-out and OOD refusal.
3. Small LLM collective: same-\(K\) and gain-matched comparisons.

## Artifacts

- Code: `analysis/complex_kernel.py`, `analysis/analyze_complex_kernel.py`
- Tests: `tests/test_complex_kernel.py`
- Transmutation outputs: `analysis/complex_kernel_transmutation/`
- Feature-sweep outputs: `analysis/complex_kernel_feature_sweep/`
- Key CSVs: `complex_endpoints.csv`, `complex_ellipses.csv`,
  `representation_distances.csv`, `within_representation_distances.csv`,
  `distance_between_within_ratios.csv`, `delta_c1_bootstrap_tests.csv`,
  `gprime0_order_sensitivity.csv`, `a0_robustness.csv`,
  `offset_action_curves.csv`
