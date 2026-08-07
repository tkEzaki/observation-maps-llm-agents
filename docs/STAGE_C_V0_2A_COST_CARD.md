# Stage C v0.2a — cost card (transition validation)

Date: 2026-07-24  
Status: **PAID COMPLETE** (actual **\$6.344**; 27200/27200 valid)  
Results: `docs/STAGE_C_RESULTS_V0_2A.md`  
Protocol: `experiments/stage_c/protocol_stage_c_v0_2a.json`  
Design: `docs/STAGE_C_V0_2_PROTOCOL.md`

## Design

| item | value |
| --- | ---: |
| Representation | `moments_m1_m3` |
| Prediction bundle | `moments_bundle_v2` |
| Comparator bundle | `moments_bundle_v1` (offline only) |
| \(N\) | **17** |
| \(T\) | 100 |
| \(K\) | 0.08, 0.10, 0.12, 0.15 |
| Seeds | 4 |
| **Expected calls** | **27,200** |

Calls = \(17\times4\times100\times4 = 27200\).

## Price assumptions

`gpt-5.4-mini`, \$0.75 / \$4.5 per 1M. Moments prompts ~same as Stage C v0.1.

Scaled from v0.1 `--estimate-only` (\$2.1154 / 6240 calls):

| | |
| --- | ---: |
| Est. base (`--estimate-only` 2026-07-24) | **\$9.2208** |
| Retry ceiling ×3 | **\$27.6624** |
| Cost ceiling (protocol) | **\$12.00** |
| **Actual** | **\$6.344** |
| Sample prompt chars | 789 |

## Preconditions (all required)

1. [x] `docs/STAGE_3B_V2_FREEZE_CHECKLIST.md` — all 8 gates pass  
2. [x] `docs/STAGE_C_DENSE_K_RICH_V1_VS_V2.md` — activity / \(t_{0.5/0.9}\) reviewed  
3. [x] Fresh `--estimate-only` under ceiling (**\$9.2208** < \$12)  
4. [x] Explicit user authorization (`--yes`) — 2026-07-24  
5. [x] v1 left untouched; v2 hash-locked / checklist-frozen  

## Non-goals for v0.2a

- \(N=9\) and \(K=-0.15\) (deferred to v0.2b)
- Intervals / centers encodings
- Claiming production Stage D
- Reusing v0.1 init seeds
