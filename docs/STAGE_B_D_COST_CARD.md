# Step D sparse-peer nested panel — cost card

Date: 2026-07-23  
Status: **authorized** (user: proceed with D; imbalance + realizations together; blocks in parallel)

## Nested design (single paid panel)

Imbalance and realization variance are **not** separate campaigns.

| factor | levels |
| --- | --- |
| encodings | intervals / moments / centers |
| imbalance | 8:8, 9:7, 10:6 (`sparse_peer16`) |
| realizations \(M\) | 6 (partner-paired when possible) |
| offsets × LLM samples | 36 × **4** |
| blocks | 2 (parallel) |

Profile count: \(3\times6=18\) per encoding cell  
Calls / block: \(3\times18\times36\times4=7{,}776\)  
Calls total: **15,552**

Ids example: `sparse_peer16_antipodal_8_8_r03_k6`

## Cost estimate

Backend: `gpt-5.4-mini` @ $0.75 / $4.50 per 1M  
Scale from P1 actual (~$4.22 / 10,368 calls):

| | estimate |
| --- | ---: |
| Per block | ~$3.16 |
| Two blocks | **~$6.33** |
| Retry ceiling (×3) | ~$19 |

Exact print from runner estimate.

## Actual spend (completed)

| block | actual USD |
| --- | ---: |
| block_1 | 3.1603725 |
| block_2 | 3.161754 |
| **total** | **6.3221265** |

Results: `docs/STAGE_B_SPARSE_PEER.md`.

## Primary analysis

Variance decomposition per imbalance:

1. LLM sampling (within cell)
2. Realization scatter (within encoding)
3. Between-encoding distance

Gate: between-encoding ≫ within-encoding realization RMS for Stage-3 use.

## Out of scope

- Near-zero dense ε densification (optional separate top-up)
- Stage C
