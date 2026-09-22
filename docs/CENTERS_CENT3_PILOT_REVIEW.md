# Centers CENT-3 coverage pilot — human review & protocol freeze

Date: 2026-07-24  
Status: **human review accepted; paid authorized (CENT-3 spend GO)**

## Decision

Coverage-directed **1,152-call** centers-only fixed-field pilot cleared review.
User authorization (2026-07-24): **YES — CENT-3 pilot spend GO**.

## Frozen artifacts

| artifact | path |
| --- | --- |
| Human review table | `analysis/centers_branch/cent3_human_review_v0.csv` |
| Frozen fields (36) | `analysis/centers_branch/cent3_frozen_fields_v0.json` |
| Stimulus catalog | `analysis/centers_branch/cent3_stimulus_catalog_v0.json` |
| Prospective lock \(\widehat p,R\) | `analysis/centers_branch/cent3_prospective_lock_v0.json` |
| Hash manifest | `analysis/centers_branch/cent3_freeze_manifest_v0.json` |
| Protocol block 1 | `experiments/stage_b_response_law/protocol_centers_cent3_v0_block_1.json` |
| Protocol block 2 | `experiments/stage_b_response_law/protocol_centers_cent3_v0_block_2.json` |
| Cost card | `docs/CENTERS_CENT3_COST_CARD.md` |
| Branch narrative | `docs/CENTERS_BRANCH.md` |

Task count: **576 / block** (= \(36\times1\times16\)), **1,152 total**.

## Field composition (36)

| bucket | n |
| --- | ---: |
| high occupancy × high risk | 16 |
| model disagreement | 8 |
| K=0 / drift | 6 |
| known controls | 6 |

### Controls

| field_id | role |
| --- | --- |
| `cent_ctrl_unimodal_positive_polar` | peer-16 unimodal positive-polar (κ=9) |
| `cent_ctrl_unimodal_sign_reversed` | same shape shifted by π |
| `cent_ctrl_exact_antipodal` | exact antipodal 8+8 |
| `cent_ctrl_small_imbalance` | 9+7 imbalance |
| `cent_ctrl_sparse_unimodal` | sparse unimodal peer 8 |
| `cent_ctrl_sparse_antipodal` | sparse antipodal peer 16 |

All peers ∈ {8, 16}. Dense peer=240 unimodal is excluded.

## Design locks

- Centers representation only (`centers_24_standard`)
- Offset = 0; 16 i.i.d. responses / field / block
- No post-hoc field dropping
- Prospective evaluation **before** refit using locked \(\widehat{\mathbf p}_{\mathrm{pre}}\), \(R_{\mathrm{pre}}\)
- Moments risk calibrator / thresholds are not reused

## Human review checklist

For each row in `cent3_human_review_v0.csv`:

| item | expected |
| --- | --- |
| `constructible` | yes_* |
| `peer_ok` | True |
| `pathology_flag` | ok (f20 repaired to exact family match) |
| controls present | 6 / 6 |

Sign-off:

- Reviewer: author
- Date: 2026-07-24
- Decision: Human review accepted
- Paid authorization: **YES — CENT-3 pilot spend GO**
- Cost ceiling: \$1.773 (retry ×3, `--max-output-tokens 16`)

## Prospective evaluation (after acquisition, before refit)

- Spearman(\(R_{\mathrm{pre}}\), observed TV) > 0
- High-risk enrichment
- Activity / \((C_1,C_2,a_0)\) errors
- Block replication
- Known centers phenotypes (exact balance stay not near 1; polarity / sparse checks)

If prospective risk fails → do **not** freeze `centers_bundle_v1`.
