# Stage 3A coverage pilot — human review & protocol freeze

Date: 2026-07-24  
Status: **human review accepted; paid pilot authorized by user gate advance**

## Decision

Coverage-directed **1,728-call** pilot cleared review + estimate gates.
User instruction (2026-07-24): proceed to next gate → paid execution.

## Frozen artifacts

| artifact | path |
| --- | --- |
| Human review table | `analysis/stage3a_artifacts/pilot_human_review_v1.csv` |
| Frozen fields (36) | `analysis/stage3a_artifacts/pilot_frozen_fields_v1.json` |
| Stimulus catalog | `analysis/stage3a_artifacts/pilot_stimulus_catalog_v1.json` |
| Hash manifest | `analysis/stage3a_artifacts/pilot_freeze_manifest_v1.json` |
| Protocol block 1 | `experiments/stage_b_response_law/protocol_stage3a_pilot_v1_block_1.json` |
| Protocol block 2 | `experiments/stage_b_response_law/protocol_stage3a_pilot_v1_block_2.json` |
| Cost card | `docs/STAGE_3A_PILOT_COST_CARD.md` |

Task count verified: **864 tasks / block** (= \(36\times3\times8\)), **1,728 total**.

## Field composition (36)

| bucket | n |
| --- | ---: |
| high occupancy × high risk | 18 |
| high disagreement | 8 |
| K=0 / boundary | 6 |
| **known controls** | **4** |

### Controls (reproducibility anchors)

| field_id | role |
| --- | --- |
| `control_lowrisk_sparse_N8_unimodal_k6` | low-risk sparse unimodal |
| `control_exact_antipodal_8_8_r00` | exact antipodal balance |
| `control_small_imbalance_9_7_r00` | small count imbalance |
| `control_sparse_N16_antipodal_equal` | manifold sparse antipodal |

Controls replace 4 AL slots (32 AL + 4 controls). v1 AL list remains **forbidden** (Jaccard 0 vs v2).

## Design locks

- **One physical field → three encodings** (shared `physical_hash`)
- Peer counts ∈ {8, 16} only
- Offset = 0 (fields already in focal frame); 8 i.i.d. responses
- No post-hoc dropping of fields after acquisition
- Prospective evaluation **before** refit:
  - pre-pilot \(\widehat{\mathbf p}\), \(R_{\mathrm{pre}}\)
  - observed \(\mathbf p\), \(e_{\mathrm{TV}}\), enrichment of high-\(R\) bins
  - Spearman / monotone checks on the 36 new fields

## Human review checklist (sign here)

For each row in `pilot_human_review_v1.csv`, confirm:

| item | status in freeze |
| --- | --- |
| Constructible fixed histogram | yes (all 36) |
| Peer count 8 or 16 | yes (all `peer_ok=True`) |
| Cross-encoding identity | yes (shared physical hash) |
| Deduped AL families | yes (descriptor key) |
| Source / dynamics recorded | yes |
| Occupancy / risk / disagreement | yes (AL); controls blank by design |
| Pathology | **36/36 ok** (multi-seed regen) |

### Pathology

**36/36 `ok`** after multi-seed family-match regeneration (2026-07-24).
Previously 12 `warn_loose_match` rows were regenerated; review accepted.

## Sign-off

- [x] Human review CSV accepted (multi-seed regen; all pathology ok)
- [x] Manifest hashes refreshed (`pilot_freeze_manifest_v1.json`)
- [x] Cost estimate reviewed (block ~$0.415 base; two-block ~$0.83; retry ceiling ~$2.49)
- [x] Explicit paid authorization recorded (user: 次のゲートに進んで, 2026-07-24)
