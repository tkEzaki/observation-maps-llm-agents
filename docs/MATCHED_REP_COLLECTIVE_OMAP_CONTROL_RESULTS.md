# Observation-map control — results

Date: 2026-07-25  
Status: **COMPLETE** — 2,304/2,304 valid; claim hierarchy frozen  
Run: `runs/matched_rep_collective_omap/matched-rep-omap-control-v0.1_gpt-5.4-mini/20260724T161234Z`  
Decision: `analysis/matched_rep_collective/omap_primary/decision.json`  

## Acquisition

| item | value |
| --- | --- |
| Calls | 2304 |
| Valid | **2304 (1.000)** |
| Cost (session) | **\$0.610** |
| Ceiling | \$6 |

## Pairwise same-info (lead with these)

| pair | mean \(d_{\mathrm{TV}}\) | 95% CI |
| --- | ---: | --- |
| original vs reformatted | **0.145** | [0.079, 0.219] |
| original vs length_matched | **0.414** | [0.311, 0.517] |
| reformatted vs length_matched | **0.374** | [0.273, 0.475] |

Field-blocked variant-label permutation on mean pairwise TV: \(p=0.0002\).

## Scale bar (Result 6)

```text
block noise                 0.069
original–reformatted        0.145
same-info mean              0.311
cross-representation mean   0.344
original–padded             0.414
```

Same-info mean is inflated by the padding arm; do not lead with 0.311 alone.

## Three-step conclusion

1. Table reformatting alone changes the operator (0.145 > block noise 0.069).  
2. Neutral padding / context manipulation produces a larger separation (0.414).  
3. Cross-representation differences cannot be reduced to retained numerical information alone.

## Verdict & claim (frozen)

**SERIALIZATION_SENSITIVE** (preregistered branch).

> The operator is sensitive not only to retained information but also to its serialization.

Safe padding wording: “neutral-padding manipulation produced the largest separation” / “sensitive to added neutral context and prompt length.”  
**Not:** “prompt length alone caused the effect.”

## Does not overturn

- Macro Outcome A (observation-map intervention, not serialization-alone)  
- Cross-rep operator effect / 3-family replication  
- Source/feedback remain suggested / not established; mediation not claimed
