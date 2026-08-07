# Bundle v1 offline dense \(K\) scan (\(T=100\))

Date: 2026-07-24  
Status: **complete (no API)**  
Artifact: `analysis/stage_c_artifacts/dense_k_v1/dense_k_T100_s16.csv`

## Protocol

| item | value |
| --- | --- |
| Bundle | `moments_bundle_v1` (selection-grade; attenuated) |
| \(T\) | 100 |
| Seeds / cell | 16 |
| \(N\) | 9, 17 |
| \(K\) | −0.20…−0.05; 0; +0.02…+0.30 |

## Envelope (conservative)

First \(K\) with \(P(r_1(T)>0.9)\ge0.5\):

| \(N\) | \(K_{\mathrm{cross}}^{\mathrm{v1}}\) |
| ---: | ---: |
| 9 | **0.08** |
| 17 | **0.10** |

Interpretation: v1 **under-predicts activity**, so these crossings are a
**harder-to-sync / attenuated envelope**, not a fit-free \(K_c\). Do **not**
use them alone to freeze Stage C v0.2 cells.

## Use

- Compared against v2: `docs/STAGE_C_DENSE_K_V1_VS_V2.md`
  (\(K_{\mathrm{cross}}\) unchanged at 0.08 / 0.10).
- Select Stage C v0.2 cells from this shared attenuated envelope after
  explicit GO (surrogate = v2).
