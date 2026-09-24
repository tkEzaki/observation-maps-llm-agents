# Matched collective R1 — second-model-family micro replication

Date: 2026-07-24  

## Design

Same frozen 48 physical fields × 3 target encodings. Only LLM family changes.

\[
48\times 3\times 16 = 2{,}304\ \text{calls per model (8×2 blocks)}.
\]

Arms:

| family | backend | model ID (frozen) |
| --- | --- | --- |
| claude | anthropic | `claude-haiku-4-5-20251001` |
| gemini | google | `gemini-3.5-flash` |

## Primary hypothesis

\[
H_0: p(f\mid\rho,R_M)=p(f\mid\rho,R_C)=p(f\mid\rho,R_I)
\]

Field-blocked target-label permutation on mean pairwise \(d_{\mathrm{TV}}\).

## Replication PASS gates (preregistered)

1. Global target effect \(p<0.05\)  
2. Between-target TV exceeds within-target block TV  
3. Bootstrap CI of (between − within) entirely positive  
4. ≥2/3 pairs exceed noise floor  
5. Invalid rate acceptable / not representation-skewed  

Non-requirements: exact M–I>M–C>C–I ranking; source/interaction replication.

## n_response choice

Offline GPT downsample: n=8 and n=12 already detection_rate=1.0 with noise ratio>2.  
Freeze **n=16** for CI stability (`analysis/matched_rep_collective/r1_downsample/`).
