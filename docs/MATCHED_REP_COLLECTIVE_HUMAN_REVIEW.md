# Matched representation collective — human review

Date: 2026-07-24  
Status: **PAID AUTHORIZED by user (2026-07-24); acquisition pending API key in runner env**

## Decision

Surrogate-free matched multi-representation collective cleared cost estimate
display + explicit user GO:

> 次の実験を行ってください。コスト見積もりを表示した後に実行で。

| item | value |
| --- | ---: |
| Est. base (3 reps) | \$58.7826 |
| Retry ceiling ×3 | \$176.3478 |
| Ceiling | \$180 |
| Calls | 122,400 |
| Model | `gpt-5.4-mini` |

`analysis/matched_rep_collective/auth_go.json` → `paid_authorized: true`.

## Frozen artifacts

| artifact | path |
| --- | --- |
| Hypothesis | `docs/MATCHED_REP_COLLECTIVE_HYPOTHESIS.md` |
| Protocol narrative | `docs/MATCHED_REP_COLLECTIVE_PROTOCOL.md` |
| Cost card | `docs/MATCHED_REP_COLLECTIVE_COST_CARD.md` |
| Replay lock (pre-outcome) | `docs/MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md` |
| Protocol JSON ×3 | `experiments/stage_c/protocol_matched_rep_collective_v0_1_*.json` |
| Match manifest | `analysis/matched_rep_collective/match_manifest.json` |
| Auth sidecar | `analysis/matched_rep_collective/auth_go.json` |

## Human review checklist

- [x] Matching lock verified (`match_manifest.json` three-way IC identity)
- [x] Surrogate-free auth path smoke-tested (mock)
- [x] Success criteria Outcome A / B / null frozen
- [x] Three per-rep `--estimate-only` reviewed and logged on cost card
- [x] Cost sum within ceiling (\$58.78 < \$180)
- [x] Replay lock document reviewed (no outcome peeking)
- [x] Explicit user paid authorization recorded (2026-07-24)
- [x] `auth_go.json` → `paid_authorized: true`

## Sign-off

| role | name | date | decision |
| --- | --- | --- | --- |
| Scientific lead | user | 2026-07-24 | GO after estimate |
| Cost reviewer | author (estimate display) | 2026-07-24 | within ceiling |

**Current auth state:** `paid_authorized = true`  
Orchestrator: `run_paid_all.py` (operational launcher; not included in this release)
