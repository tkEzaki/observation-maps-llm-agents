# Stage B full-operator gate calibration (P3 offline)

Date: 2026-07-23  
Source: `analysis/calibrate_full_operator_gate.py`  
CSV: `analysis/stage_b_offline_p3_p4/full_operator_tv_calibration.csv`

## Polar gate (unchanged)

Historical preregistered rule: min pairwise \(d_{C_1}\ge0.25\).
Do not rewrite manifold v0.1 operational pass/fail.

## Absolute TV = 0.20

**Descriptive only — not frozen.** Scale depends on sampling and field.

## Proposed adaptive full-operator rule

\[
\frac{d_{\mathrm{between}}^{\min}}{d_{\mathrm{within}}^{\mathrm{mean}}}>2
\]

where within = same encoding across the two acquisition blocks (mean TV),
between = encoding pairs (across-block mean of per-block TV).

| field | within TV | between min TV | ratio | ratio>2 | polar \(d_{C_1}\) | class |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| unimodal_k9 | 0.111 | 0.420 | 3.78 | yes | 0.124 | weak polar / full persists |
| bimodal_equal_sep2_k6 | 0.105 | 0.257 | 2.45 | yes | 0.451 | polar + full |
| antipodal_equal_k6 | 0.113 | 0.319 | 2.82 | yes | 0.070 | polar collapse / full persists |
| asymmetric_w075_sep2_k6 | 0.102 | 0.315 | 3.09 | yes | 0.318 | polar + full |
| sparse_N8_unimodal_k6 | 0.107 | 0.369 | 3.44 | yes | 0.838 | polar + full |
| sparse_N16_antipodal_k6 | 0.123 | 0.341 | 2.78 | yes | 0.048 | polar collapse / full persists |

On this manifold, the ratio rule and the descriptive TV≥0.20 cutoff agree on
full-operator separability for all six fields. Prefer the **ratio rule** for
future protocols; keep absolute 0.20 as a secondary sanity check only.

Companion (when bootstrap Δ is available): 2D confidence region for
trinomial-distance or \(\Delta C\) should exclude zero — still without
claiming family-wise significance unless adjusted.

## Locked dual-layer antipodal reading

- Polar-channel: collapse under exact antipodal symmetry.
- Full-operator: separable (ratio>2 on present data).
