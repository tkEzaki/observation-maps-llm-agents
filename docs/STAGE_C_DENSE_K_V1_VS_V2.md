# Dense \(K\) envelope: v1 vs v2 (\(T=100\), offline)

Date: 2026-07-24  
Status: **complete (no API)**  
Artifacts:

- `analysis/stage3a_artifacts/dense_k_v1_vs_v2.csv`
- `analysis/stage3a_artifacts/dense_k_v1_vs_v2_summary.json`

## Protocol

| item | value |
| --- | --- |
| Bundles | `moments_bundle_v1` vs `moments_bundle_v2` |
| \(T\) | 100 |
| Seeds / cell | 12 |
| \(N\) | 9, 17 |
| \(K\) | −0.15; 0; +0.02…+0.30 |

## Crossing threshold

First \(K\) with \(P(r_1(T)>0.9)\ge0.5\):

| \(N\) | \(K_{\mathrm{cross}}^{\mathrm{v1}}\) | \(K_{\mathrm{cross}}^{\mathrm{v2}}\) |
| ---: | ---: | ---: |
| 9 | **0.08** | **0.08** |
| 17 | **0.10** | **0.10** |

## Interpretation

1. **Microscopic stay fix does not move the offline locking envelope.**
   Median \(r_1(T)\) curves and \(K_{\mathrm{cross}}\) are effectively the same
   within seed noise.
2. **Activity / arrival do move** — see rich compare
   `docs/STAGE_C_DENSE_K_RICH_V1_VS_V2.md` (v2 raises \(E[A]\) by ~0.1 on
   N=17 transition \(K\)).
3. Residual v1 Stage-C attenuation was therefore **not solely** a final-\(r_1\)
   story; stay leakage mainly suppressed activity / torque along the path.
4. v2 remains preferred for Stage C v0.2; cell selection reuses the shared
   attenuated crossing as location, but primaries are activity + \(t_{0.5/0.9}\).

## Next

Explicit paid GO for Stage C v0.2a after `--estimate-only`
(`docs/STAGE_C_V0_2_PROTOCOL.md`).
