# Matched collective R2 — cost card

Date: 2026-07-25  
Status: **estimate-only pending review; paid NOT authorized**

## Budget

| panel | formula | calls |
| --- | --- | ---: |
| Core 6 | \(6\times3K\times3\mathcal R\times17\times100\) | 91,800 |
| Held-out 4 | \(4\times2K\times3\mathcal R\times17\times100\) | 40,800 |
| **Total** | | **132,600** |

## Pricing (R1 Haiku freeze)

| item | value |
| --- | ---: |
| model | `claude-haiku-4-5-20251001` |
| input \$/1M | 1.00 |
| output \$/1M | 5.00 |

## Heuristic vs protocol ceiling

| item | USD |
| --- | ---: |
| R1 Claude actual \$1.279 / 2,304 calls → linear extrapolate | ~\$74 |
| ×2 safety | ~\$148 |
| **Protocol ceiling (expected spend)** | **\$175** |

## Frozen `--estimate-only` (2026-07-25)

From `analysis/matched_rep_collective/r2_macro/estimate_only.json`:

| quantity | value |
| --- | ---: |
| planned calls | 132,600 |
| estimated input tokens | 68,996,200 |
| maximum output tokens | 2,652,000 |
| **estimated base cost** | **\$82.26** |
| naive retry ceiling (×3 every call) | \$246.77 |

Base is within the \$175 protocol ceiling. The ×3 retry figure is a pathological bound (every call exhausts `max_attempts`); it is **not** the operational authorization threshold. Authorize on base ≤ \$175 and monitor actual spend; abort if cumulative actual cost approaches the ceiling.

Linear R1 extrapolation is superseded by this frozen-task estimate.

## Commands

```powershell
py -m analysis.matched_rep_collective.freeze_r2_artifacts
py -m experiments.stage_c.run_r2_claude_macro --estimate-only
```

## Paid authorization checklist

- [x] `estimate_only.json` written; base \$82.26 ≤ \$175
- [ ] Human review (`MATCHED_REP_COLLECTIVE_R2_HUMAN_REVIEW.md`) complete
- [ ] `auth_go.json` → `paid_authorized=true`
- [ ] Explicit user GO + `--yes`
