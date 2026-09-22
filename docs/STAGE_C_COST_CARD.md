# Stage C selection pilot — cost card

Date: 2026-07-24  
Status: **PAID COMPLETE** (actual \$1.456; 6240/6240 valid)  
Results: `docs/STAGE_C_RESULTS_V0_1.md`  
Protocol: `experiments/stage_c/protocol_stage_c_v0_1.json`

## Design (peer-matched)

| item | value |
| --- | --- |
| Representation | `moments_m1_m3` only |
| \(N\) | **9, 17** (peers 8, 16) |
| \(T\) | 40 (selection; \(T=100\) deferred) |
| \(K\) | −0.15, 0.0, +0.15 |
| \(\omega\) half-width | 0.05 (equally spaced) |
| Seeds | 2 |
| **Expected calls** | **6,240** |

Calls = \(2\times3\times(9+17)\times40 = 6240\).

Two-body (\(N=2\), peer=1) is **deferred** (outside train peers).

## Price assumptions

`gpt-5.4-mini`, \$0.75 / \$4.5 per 1M. Moments prompts ~800 chars.

| | |
| --- | ---: |
| Est. base (2026-07-24 `--estimate-only`) | **\$2.1154** |
| Retry ceiling ×3 | **\$6.3461** |
| Cost ceiling (protocol) | **\$2.50** |

## Preconditions

1. Moments bundle hash-locked (`moments_bundle_v1`, `freeze_ready=true`)
2. `stage_c_authorized=true` in bundle manifest / OOD policy
3. Offline surrogate baseline written
4. `--estimate-only` reviewed
5. Explicit `--yes` for paid openai run

## Non-goals

- Production Stage D claims
- Intervals / centers encodings
- \(N\in\{8,16\}\) (peer confound)
- Reusing these runs as production replicates
