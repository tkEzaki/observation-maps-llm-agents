# Stage C v0.2b — cost card (size / negative replication)

Date: 2026-07-24  
Status: **PAID COMPLETE** (actual **\$6.630**; 28400/28400 valid)  
Results: `docs/STAGE_C_RESULTS_V0_2B.md`  
Protocol: `experiments/stage_c/protocol_stage_c_v0_2b.json`  
Depends on: v0.2a success (`docs/STAGE_C_RESULTS_V0_2A.md`)

## Design

| item | value |
| --- | ---: |
| Representation | `moments_m1_m3` |
| Prediction bundle | `moments_bundle_v2` |
| \(N=9\) \(K\) | −0.15, 0, 0.06, 0.08, 0.10, 0.15 |
| \(N=17\) \(K\) | −0.15 only |
| \(T\) | 100 |
| Seeds | 4 |
| **Expected calls** | **28,400** |

Breakdown: \(9\times6\times100\times4 + 17\times1\times100\times4 = 21600+6800\).

## Price assumptions

`gpt-5.4-mini`, \$0.75 / \$4.5 per 1M.

| | |
| --- | ---: |
| Est. base (`--estimate-only` 2026-07-24) | **\$9.6276** |
| Retry ceiling ×3 | **\$28.8828** |
| Cost ceiling (protocol) | **\$12.00** |
| **Actual** | **\$6.630** |

## Preconditions

1. [x] v0.2a complete and activity/stay win for v2  
2. [x] v2 checklist frozen  
3. [x] Fresh `--estimate-only` under ceiling (**\$9.6276** < \$12)  
4. [x] Explicit continue authorization (進めて)  
5. [x] Surrogate preds locked before paid LLM  
