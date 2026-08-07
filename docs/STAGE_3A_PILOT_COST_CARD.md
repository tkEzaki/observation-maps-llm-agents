# Stage 3A coverage-directed pilot — cost card

Date: 2026-07-24  
Status: **PAID COMPLETE** (actual \$0.7007; valid 1728/1728)  
Review: `docs/STAGE_3A_PILOT_REVIEW.md`  
Prospective: `docs/STAGE_3A_PILOT_PROSPECTIVE.md`

## Scope

| item | value |
| --- | ---: |
| Physical fields | 36 (32 AL + 4 controls) |
| Representations | 3 |
| Offsets | 1 (`0`) |
| Responses / condition | 8 |
| Acquisition blocks | 2 |
| **Total calls** | **1,728** |

Protocols:

- `experiments/stage_b_response_law/protocol_stage3a_pilot_v1_block_1.json`
- `experiments/stage_b_response_law/protocol_stage3a_pilot_v1_block_2.json`

## Price assumptions

`gpt-5.4-mini`, \$0.75 / \$4.5 per 1M (project standard).

| | |
| --- | ---: |
| Estimate (2026-07-24 `--estimate-only`) | |
| Base / block | **\$0.4149** |
| Base / two blocks | **\$0.8298** |
| Retry ceiling ×3 / two blocks | **\$2.4896** |

## Preconditions

1. Sign-off on `pilot_human_review_v1.csv` / `STAGE_3A_PILOT_REVIEW.md`
2. Manifest hashes in `pilot_freeze_manifest_v1.json` unchanged
3. Explicit user authorization to spend
4. Prospective-eval-before-refit rule accepted

## Non-goals

- Using v1 36-field list
- Near-zero \(\varepsilon\) top-up in this pilot
- Post-hoc dropping of “bad” fields
- Treating Spearman alone as Stage 3B freeze (also require enrichment / \(Q_{\mathrm{risk}}\))
