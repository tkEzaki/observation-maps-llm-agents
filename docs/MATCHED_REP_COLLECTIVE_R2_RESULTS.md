# Matched collective R2 — Claude macroscopic results

Date: 2026-07-25  
Status: **ACQUISITION COMPLETE + PRIMARY INFERENCE COMPLETE**  
Session: `runs/stage_c/r2_claude_macro/matched-rep-collective-r2-claude-v0.1_ff2ef27f0dfe/20260725T050234Z`  
Inference: `.../r2_inference/primary_inference.json`

## Verdict

| Gate | Result |
| --- | --- |
| \(K=0\) engine control | **PASS** (\(\max\|\Delta r_m\|=0\)) |
| Gate A — core macro effect | **PASS** |
| Gate B — qualitative phase separation | **PASS** |
| Gate C — GPT phenotype-map replication | **FAIL** (hierarchy reversed) |

**Claim tier supported:** Macro representation effect + qualitative phase replication in Claude.  
**Not supported:** GPT moments-lock hierarchy (Gate C).

Frozen reading (preregistered reverse-hierarchy case):

> Macroscopic representation dependence generalized across model families, whereas the mapping from representation to collective phenotype was model-family dependent.

## Primary inference (core 6 seeds; exact \((3!)^6=46656\))

| endpoint | \(T_{\mathrm{obs}}\) | \(p_{\mathrm{perm}}\) | reject \(H_0\) |
| --- | ---: | ---: | --- |
| Sustained-lock score \(L\) | 0.449 | **0.00103** | yes |
| Final-\(r_1\) mean \(Y\) | 0.0827 | **0.000386** | yes |

Mean \(L\) (positive \(K\)):

| representation | \(\overline L\) |
| --- | ---: |
| moments | 0.00 |
| centers | 0.67 |
| intervals | 0.92 |

Mean \(Y\) (avg final \(r_1\) at \(K\in\{0.08,0.15\}\)):

| representation | \(\overline Y\) |
| --- | ---: |
| moments | 0.598 |
| centers | 0.921 |
| intervals | 0.973 |

Direction vs GPT: **reversed** (\(r_1^I \gtrsim r_1^C > r_1^M\)).

## Core phenotype matrix (polar_locked counts)

| \(K\) | moments | centers | intervals |
| ---: | ---: | ---: | ---: |
| 0 | 0/6 | 0/6 | 0/6 |
| 0.08 | **0/6** | 2/6 | **5/6** |
| 0.15 | **0/6** | **6/6** | **6/6** |

Gate B at \(K=0.15\): intervals/centers \(\ge5/6\) lock vs moments \(\le1/6\).  
Held-out (4 seeds) same direction: moments 0/4 lock; centers/intervals 4/4 at \(K=0.15\).

## Pairwise contrasts (seed-level)

Sustained-lock \(L\):

| contrast | mean \(\Delta\) | bootstrap 95% CI | sign-flip \(p\) |
| --- | ---: | --- | ---: |
| M−C | −0.667 | [−0.833, −0.50] | 0.031 |
| M−I | −0.917 | [−1.00, −0.75] | 0.031 |
| I−C | +0.25 | [+0.083, +0.417] | 0.25 |

Final \(r_1\) \(Y\): M−C and M−I significantly negative (moments lower); I−C small positive (n.s. at exact sign-flip \(p=0.094\)).

## Relation to GPT matched collective

| family | moments | centers | intervals |
| --- | --- | --- | --- |
| GPT (prior) | polar lock 6/6 | nonlock | nonlock |
| Claude R2 | nonlock 0/6 | lock (esp. \(K=0.15\)) | lock |

Representation still selects collective phase; **which** representation locks is family-specific.

## Non-claims (unchanged)

- Universal phase diagram across all LLM families  
- Identical Claude/GPT operators  
- Thermodynamic \(K_c\) / source-manifold feedback  

## Artifacts

- `r2_inference/primary_inference.json`
- `r2_inference/trajectory_rows.json`
- `r2_inference/k0_control.json`
