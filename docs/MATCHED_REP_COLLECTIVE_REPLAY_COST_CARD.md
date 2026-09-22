# Matched representation collective — 3×3 replay cost card

Date: 2026-07-24  
Status: **PAID COMPLETE** — 4,608/4,608 valid; primary report generated  
Rules: `docs/MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md` (**frozen pre-outcome**)  
Results: `docs/MATCHED_REP_COLLECTIVE_REPLAY_RESULTS.md`  
Protocol: `experiments/stage_c/protocol_matched_rep_collective_replay_v0_1.json`

## Call budget (locked)

\[
48\ \text{fields}\times 3\ \text{target encodings}\times 32\ \text{responses}
= 4{,}608\ \text{calls}.
\]

Storage: \(16\times 2\) acquisition blocks (same call total).

## Estimate vs actual

| quantity | USD |
| --- | ---: |
| Estimate base | \$2.213 |
| Estimate ×3 retry | \$6.639 |
| Protocol ceiling | \$12.00 |
| **Actual total (trace tokens)** | **\$1.876** |
| Session log (post-preflight) | \$1.873 |
| Preflight | 12 in-budget tasks (inside 4608) |

## Preconditions (closed)

1. [x] Field strata quotas unchanged since pre-outcome lock  
2. [x] Source × stratum balance; no hand edits  
3. [x] Physical / prompt / protocol hashes frozen  
4. [x] Analysis plan locked  
5. [x] Retention + mock smoke  
6. [x] Estimate under ceiling  
7. [x] Runner source hashes  
8. [x] Explicit user authorization  
9. [x] `replay_auth_go.json` → true  
10. [x] Paid acquisition complete  

## Explicit non-actions (still binding)

- Retuning strata after Outcome A  
- Surrogate refit before primary 3×3 report (primary now frozen first)  
- Dropping fields post-acquisition without preregistered pathology rule
