# Matched collective R1 — second-model replication results

Date: 2026-07-24 (claim wording revised 2026-07-25)  
Status: **COMPLETE — Claude PASS, Gemini PASS**  
Protocol: `docs/MATCHED_REP_COLLECTIVE_R1_PROTOCOL.md`  
Decision: `analysis/matched_rep_collective/r1_primary/decision.json`  

## Design freeze

- Same 48 physical fields / hashes / encodings as GPT replay  
- \(n=16\) responses/condition (8×2 blocks) → **2,304 calls/model**  
- Offline downsample: n=8/12/16 detection_rate=1.0; froze 16 for CI stability  
- Downsample stop log: `analysis/matched_rep_collective/r1_downsample/stop_log.json`

| arm | model | calls | valid | cost (session) |
| --- | --- | ---: | ---: | ---: |
| GPT (reference) | gpt-5.4-mini | 4608@32 | 1.000 | \$1.876 |
| Claude | claude-haiku-4-5-20251001 | 2304 | **1.000** | \$1.279 |
| Gemini | gemini-3.5-flash | 2304 | **0.990** | \$0.521 |

## Replication PASS gates (all met)

| gate | Claude | Gemini |
| --- | --- | --- |
| Global target \(p<0.05\) | **0.0002** | **0.0002** |
| Between TV > within block TV | yes | yes |
| Between−within positive (paired) | yes | yes |
| ≥2/3 pairs > noise floor | yes | yes |
| Invalid rate OK / not skewed | yes | yes |

**Overall:** `both_second_families_pass = true`

## Effect sizes (mean pairwise TV)

| pair | GPT (32) | Claude (16) | Gemini (16) |
| --- | ---: | ---: | ---: |
| Moments–Centers | 0.335 | 0.393 | 0.155 |
| Moments–Intervals | 0.408 | 0.411 | 0.213 |
| Centers–Intervals | 0.290 | 0.310 | 0.114 |
| Mean pairwise / within-block ratio | 3.76× | 20.4× | 20.6× |

Replicated quantity is \(d_{\mathrm{between}}>d_{\mathrm{within,block}}\), not shared ranking or absolute effect size.  
Gemini: smaller magnitude, still well above its acquisition-noise floor.

## Safe claim (frozen; split macro vs micro)

> Observation representations causally select distinct collective phases in the
> matched GPT system. On identical physical fields, representation-dependent
> response operators generalize across three model families, and even
> information-preserving changes in presentation alter the operator.

| layer | status |
| --- | --- |
| Macro phase selection | GPT only |
| Micro operator generality (cross-rep) | 3 families |
| Serialization sensitivity (same-info) | GPT omap control |
| Macro generality across families | **not tested** |

Do **not** claim: length-alone causality; full mediation; source/feedback established;
collective phases replicated on Claude/Gemini.

## Deferred / next

- Same-information observation-map control (preregistered; paid hold)  
- Optional second-family collective (only if needed after control + manuscript)  

Unpaid stratum anchors: `analysis/matched_rep_collective/anchors/`

## Run directories

- Claude: `runs/matched_rep_collective_r1/matched-rep-r1-cross-model-v0.1_claude_claude-haiku-4-5-20251001/20260724T142109Z`  
- Gemini: `runs/matched_rep_collective_r1/matched-rep-r1-cross-model-v0.1_gemini_gemini-3.5-flash/20260724T142109Z`
