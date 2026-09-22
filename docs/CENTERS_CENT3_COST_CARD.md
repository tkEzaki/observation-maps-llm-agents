# Centers CENT-3 coverage-directed pilot — cost card

Date: 2026-07-24  
Status: **PAID COMPLETE** (actual \$0.622; ceiling \$1.773)  
Review: `docs/CENTERS_CENT3_PILOT_REVIEW.md`  
Prospective: `docs/CENTERS_CENT3_PROSPECTIVE.md`

## Scope

| item | value |
| --- | ---: |
| Physical fields | 36 (30 AL + 6 controls) |
| Representations | 1 (`centers_24_standard`) |
| Offsets | 1 (`0`) |
| Responses / condition | 16 |
| Acquisition blocks | 2 |
| **Total calls** | **1,152** |

Protocols:

- `experiments/stage_b_response_law/protocol_centers_cent3_v0_block_1.json`
- `experiments/stage_b_response_law/protocol_centers_cent3_v0_block_2.json`

## Price assumptions

`gpt-5.4-mini`, \$0.75 / \$4.5 per 1M (project standard).

| | |
| --- | ---: |
| Estimate (2026-07-24, character heuristic) | |
| Base / two blocks | **\$0.591** |
| Retry ceiling ×3 / two blocks | **\$1.773** |

## Preconditions

1. Sign-off on `cent3_human_review_v0.csv` / `CENTERS_CENT3_PILOT_REVIEW.md`
2. Manifest hashes in `cent3_freeze_manifest_v0.json` unchanged
3. Offline gates already green (`cent12_decision.json`: Spearman +0.383, enrichment 1.84×)
4. Explicit user authorization to spend
5. Prospective-eval-before-refit rule accepted

## Non-goals

- Moments / intervals acquisition in this pilot
- Stage C collective runs
- Post-hoc dropping of “bad” fields
- Treating offline Spearman alone as freeze
- Copying moments numerical freeze thresholds

## Authorization

Paid run: **not authorized** until user GO after review.
