# Matched representation collective — cost card v0.1

Date: 2026-07-24  
Status: **PAID GO — acquisition launched after estimate display**  
Hypothesis: `docs/MATCHED_REP_COLLECTIVE_HYPOTHESIS.md`  
Protocol: `docs/MATCHED_REP_COLLECTIVE_PROTOCOL.md`

## Design

| item | value |
| --- | ---: |
| Representations | moments / centers / intervals |
| \(N\) | 17 |
| \(T\) | 100 |
| \(K\) | −0.15, 0, 0.08, 0.15 |
| Seeds (paired) | 6 |
| **Calls per representation** | **40,800** |
| **Total expected calls** | **122,400** |

Calls = \(17\times4\times100\times6\times3 = 122{,}400\).

## Price assumptions

Default card prices: `gpt-5.4-mini`, \$0.75 / \$4.5 per 1M (same as Stage C v0.2a).  
Prompt length **varies by representation**.

### Pre-GO character-based estimates (2026-07-24)

Logged in `analysis/matched_rep_collective/estimate_only_log.json`.

| representation | sample prompt chars | est. base USD | retry ×3 |
| --- | ---: | ---: | ---: |
| moments_m1_m3 | 789 | 13.8312 | 41.4936 |
| centers_24_standard | 1429 | 21.6648 | 64.9944 |
| intervals_24_decimal6 | 1562 | 23.2866 | 69.8598 |
| **sum** | | **58.7826** | **176.3478** |

| | USD |
| --- | ---: |
| Protocol cost ceiling (all three reps) | **180.00** |
| Per-rep ceiling (protocol JSON) | 60.00 |

Optional: refresh with `--backend openai --model …` before flipping `auth_go` if provider tokenization differs from the character heuristic.

```powershell
py analysis/matched_rep_collective/run_estimate_only.py `
  --backend openai --model MODEL
```

## Preconditions for paid GO

1. [x] Three per-rep `--estimate-only` logged (character heuristic)  
2. [x] Sum base \$58.78 < ceiling \$180  
3. [x] `docs/MATCHED_REP_COLLECTIVE_HUMAN_REVIEW.md` checklist complete  
4. [x] `analysis/matched_rep_collective/auth_go.json` flipped to `paid_authorized: true`  
5. [x] Explicit user `--yes` authorization recorded (2026-07-24)  
6. [x] Replay field rules already locked (`MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md`)

## Non-goals

- Surrogate bundle freeze or predictive MAE as a spend justification  
- \(N=9\) in this card  
- Cross-encoding replay calls (separate 4,608-call card after collectives)  
- Second-model replication (conditional; separate card)
