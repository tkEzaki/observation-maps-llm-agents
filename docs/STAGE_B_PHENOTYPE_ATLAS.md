# Stage B phenotype atlas (P4 offline)

Date: 2026-07-23  
Source: `analysis/complex_kernel_stimulus_manifold/`  
CSV: `analysis/stage_b_offline_p3_p4/phenotype_atlas.csv`

Across-block means. \(a_0=a_0(\mathcal R,\rho)\): encoding-intrinsic torque is
**not** supported.

| \(\rho\) | \(\mathcal R\) | \(a_0\) | \(R_1\) | \(R_2\) | \(A\) | \(H\) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| unimodal_k9 | intervals | +0.181 | 0.074 | 0.440 | 0.969 | 0.543 |
| unimodal_k9 | moments | −0.028 | 0.949 | 0.444 | 1.000 | 0.195 |
| unimodal_k9 | centers | −0.278 | 0.183 | 0.212 | 0.985 | 0.484 |
| bimodal_equal_sep2_k6 | intervals | −0.124 | 0.246 | 0.285 | 0.979 | 0.505 |
| bimodal_equal_sep2_k6 | moments | +0.051 | 1.137 | 0.138 | 1.000 | 0.158 |
| bimodal_equal_sep2_k6 | centers | −0.065 | 0.779 | 0.280 | 0.984 | 0.395 |
| antipodal_equal_k6 | intervals | +0.188 | 0.129 | 0.231 | 0.554 | 0.747 |
| antipodal_equal_k6 | moments | +0.016 | 0.023 | 0.106 | 0.075 | 0.144 |
| antipodal_equal_k6 | centers | +0.154 | 0.068 | 0.054 | 0.340 | 0.525 |
| asymmetric_w075_sep2_k6 | intervals | +0.016 | 0.324 | 0.253 | 0.992 | 0.526 |
| asymmetric_w075_sep2_k6 | moments | +0.022 | 0.996 | 0.452 | 1.000 | 0.171 |
| asymmetric_w075_sep2_k6 | centers | −0.338 | 0.692 | 0.156 | 0.984 | 0.384 |
| sparse_N8_unimodal_k6 | intervals | +0.330 | 0.172 | 0.341 | 0.889 | 0.480 |
| sparse_N8_unimodal_k6 | moments | +0.143 | 1.007 | 0.492 | 1.000 | 0.144 |
| sparse_N8_unimodal_k6 | centers | +0.021 | 0.678 | 0.130 | 0.910 | 0.415 |
| sparse_N16_antipodal_k6 | intervals | +0.022 | 0.078 | 0.047 | 0.634 | 0.744 |
| sparse_N16_antipodal_k6 | moments | −0.006 | 0.015 | 0.062 | 0.039 | 0.094 |
| sparse_N16_antipodal_k6 | centers | −0.129 | 0.039 | 0.261 | 0.552 | 0.607 |

## Readings locked for P1 design

1. Unimodal interval+/center− \(a_0\) does **not** carry to all fields
   (equal bimodal intervals \(a_0<0\); dense antipodal intervals and centers
   both \(a_0>0\)).
2. Moments \(A=1\) except under antipodal symmetry (\(A\simeq0.04\)–\(0.08\)).
3. Antipodal: polar amplitudes small; activity / \(R_2\) / TV still separate
   encodings.

Regenerate: `py analysis/calibrate_full_operator_gate.py`
