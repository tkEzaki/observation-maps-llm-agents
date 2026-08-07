# Matched representation collective — branch README

Surrogate-free causal intervention: vary only serialization \(\mathcal R\)
in LLM Kuramoto collectives. **Not** a reopen of Centers/Intervals surrogate
Stage C.

## Docs

| doc | role |
| --- | --- |
| `docs/MATCHED_REP_COLLECTIVE_HYPOTHESIS.md` | Outcome A/B/null + journal submission gates |
| `docs/MATCHED_REP_COLLECTIVE_PROTOCOL.md` | matching lock + endpoints |
| `docs/MATCHED_REP_COLLECTIVE_COST_CARD.md` | 122,400-call cost |
| `docs/MATCHED_REP_COLLECTIVE_HUMAN_REVIEW.md` | GO checklist |
| `docs/MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md` | 3×3 replay rules (pre-outcome) |

## Artifacts here

| path | role |
| --- | --- |
| `auth_go.json` | `paid_authorized=false` until human GO |
| `match_manifest.json` | 24 physical cells × 3 reps |
| `estimate_only_log.json` | character-heuristic cost |
| `paid_gate_status.json` | remaining gates |
| `smoke/` | mock 3-rep IC-match smoke |

## Offline commands

```powershell
py analysis/matched_rep_collective/write_match_manifest.py
py analysis/matched_rep_collective/run_estimate_only.py --backend mock
py analysis/matched_rep_collective/run_mock_smoke.py
py analysis/matched_rep_collective/write_paid_gate_status.py
```

Paid acquisition stays blocked until `auth_go.json` flips and the user
passes `--yes`.

## Windows launchers (independent PC)

| bat | role |
| --- | --- |
| `run_matched_rep_collective_PARALLEL.bat` | estimate → confirm → 3 windows |
| `run_matched_rep_collective_ONE.bat` | single rep |
| `run_matched_rep_collective_RESUME.bat` | resume latest incomplete sessions |
| `run_matched_rep_collective_STATUS.bat` | progress / PID / log tails |

`.env` is loaded automatically (same order as legacy):

1. `pilot4_kuramoto/.env`
2. `llm_emergent/.env`
3. `research_current/gen_ai_logi/rq1/.env` (usual shared Dropbox key)
4. `research_current/gen_ai_logi/.env`

No `MATCHED_REP_ENV_FILE` setup required if Dropbox layout is intact.
