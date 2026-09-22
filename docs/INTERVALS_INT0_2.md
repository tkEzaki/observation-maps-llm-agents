# Intervals INT-0 / INT-1 / INT-2 — offline package

Date: 2026-07-24  
Status: **INT-0/1 done; INT-2 model/risk PASS; support-map FAIL → INT-3 NO-GO**  
Paid: **not authorized**

## Commands

```powershell
py analysis/centers/write_final_manifest.py
py analysis/intervals/snapshot_and_atlas.py
py analysis/intervals/run_int2_offline.py
py -m unittest tests.test_intervals_serialization -v
```

## INT-0 scope lock

| scope | peers | rows |
| --- | --- | ---: |
| `intervals_collective_scope` | {8,16} | **1512** (8:82, 16:1430) |
| `intervals_dense_exploratory` | 240 | **1224** |

- Representation: `intervals_24_decimal6`
- Unique physical / prompt hashes: 1272 / 1272
- Baseline policy: **peer-specific global**
- No cyclic-shift augmentation
- Optional later narrow: `intervals_collective_bundle_v1_N17` (peer=16)

Artifacts: `analysis/intervals_branch/dataset_manifest_intervals_v0.json`

## INT-1 phenotype / serialization

- Atlas: `phenotype_atlas_intervals_v0.csv` (peer × profile_family)
- Serialization unit tests: `tests/test_intervals_serialization.py` (**green**)
  - identical physical → identical prompt
  - half-turn = bin permutation
  - bin-boundary perturbation changes prompt

## INT-2 gates

Acquisition-block OOF (collective scope):

| model | log loss | \(a_0\) MAE | sign acc |
| --- | ---: | ---: | ---: |
| **peer_specific_activity_direction** | **0.803** | **0.370** | 0.734 |
| peer_specific_hellinger | 0.877 | 0.397 | 0.716 |
| peer_specific_canonical_emd | 0.883 | 0.400 | 0.708 |
| peer_specific_global | worse | — | — |

Cluster bootstrap (hurdle vs peer-global): field CI **[0.180, 0.330]**, family CI
**[0.173, 0.378]** — entirely positive (ΔLL≈+0.207).

| gate | result |
| --- | --- |
| Model beats peer-global (cluster) | **PASS** |
| Peer no-collapse | **PASS** |
| Risk positive / enrichment | **PASS** (Spearman +0.15; enrichment 1.26×) |
| Support map not dominated by gap | **FAIL** (62/80 neighbors_active_feature_gap_risk) |

**INT-3 prospective pilot GO: NO**

### Reading

Intervals already shows **in-distribution compressibility** under peer
stratification (same qualitative lesson as centers). The support map warns
that many sparse/unimodal/antipodal neighborhoods look like the centers
failure mode (“neighbors exist but label regime may not match collective
abstention/active mix”). Therefore paid coverage pilot is **not** unlocked
yet — next offline work should diagnose which families drive the gap risk
and whether peer16-only + discrete features reduce it, without spending.

## Explicit non-actions (now)

- No INT-3 paid acquisition
- No freeze
- No Stage C
- No mechanical transplant of centers freeze thresholds

## Next offline (before any auth)

1. Family-level breakdown of support-map diagnoses  
2. Peer16-only INT-2 re-eval  
3. Discrete-bin feature augmentation bake-off  
4. Re-run support gate; only then draft INT-3 protocol + cost card
