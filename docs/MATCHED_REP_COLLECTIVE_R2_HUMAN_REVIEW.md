# Matched collective R2 — human review checklist

Date: 2026-07-25  
Status: **OPEN** (do not flip paid auth until all boxes are true)

## Science / design

- [ ] Claude selected as **targeted** R1 micro replication (documented; not random family)
- [ ] Primary question is Claude-internal representation effect (GPT numerical match not required)
- [ ] Claim tiers A/B/C understood; Gate C optional
- [ ] \(K=-0.15\) intentionally dropped
- [ ] Core 6 physical seeds match GPT `match_manifest.json`
- [ ] Held-out seeds 6–9 frozen **before** seeing Claude outcomes
- [ ] Phenotype rules identical to GPT v0.2
- [ ] Primary statistic \(T_L\) + exact \((3!)^6\) permutation accepted
- [ ] Inference unit = physical seed

## Freeze artifacts present

- [x] `experiments/stage_c/protocol_matched_rep_collective_r2_claude_v0_1.json`
- [x] `analysis/matched_rep_collective/r2_macro/physical_seed_manifest.json`
- [x] `analysis/matched_rep_collective/r2_macro/task_manifest.json`
- [x] `analysis/matched_rep_collective/r2_macro/prompt_hashes.json`
- [x] `analysis/matched_rep_collective/r2_macro/source_hashes.json`
- [x] `analysis/matched_rep_collective/r2_macro/estimate_only.json`
- [x] Docs: hypothesis / protocol / analysis plan / cost card

## Cost / ops

- [x] `--estimate-only` run after freeze (base **\$82.26**)
- [x] Expected-spend ceiling \$175 covers base; naive ×3 retry bound \$246 is pathological only
- [ ] Round-robin acquisition schedule understood
- [ ] \(K=0\) control abort policy understood
- [ ] No seed substitution on failure
- [ ] Anthropic credentials available; ResearchBot kill-safety noted (PID-only stop)

## Authorization

| field | required |
| --- | --- |
| `paid_authorized` | `true` only after this checklist |
| `estimate_only_reviewed` | `true` |
| `human_review_complete` | `true` |
| `explicit_user_go` | recorded in sidecar notes |

Sidecar: `analysis/matched_rep_collective/r2_macro/auth_go.json`

**Current decision:** protocol/analysis freeze **GO**; paid acquisition **NO-GO**.
