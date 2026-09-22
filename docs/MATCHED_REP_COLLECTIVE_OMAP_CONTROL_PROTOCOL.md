# Same-information observation-map control (preregistered)

Date: 2026-07-25  
Status: **PREREGISTERED — paid NOT authorized**  
Parent: GPT replay + R1 operator replication  
Purpose: Test whether representation effects reduce to format / length artifacts

## Design

Same frozen 48 physical fields. **Moment information fixed**; only serialization changes.

| Condition | Information | Display |
| --- | --- | --- |
| `moments_original` | moments m1–m3 | original line list |
| `moments_reformatted` | same values | table (harmonic \| cos \| sin) |
| `moments_length_matched` | same values | original + neutral padding → ~1496 chars (centers/intervals mid-length) |

\[
48\times 3\times 16 = 2{,}304\ \text{calls (8×2 blocks)}.
\]

Model: **gpt-5.4-mini** (same family as primary GPT replay; isolates serialization vs information).  
No tool use / reasoning mode. Same action contract, temperature 0.7, raw retention.

## Primary question

Is

\[
\overline{d_{\mathrm{TV}}}(\text{same-information moment variants})
\]

much smaller than

\[
\overline{d_{\mathrm{TV}}}(\text{moments vs centers/intervals})
\]

and near the within-target block-noise floor?

## PASS / interpret readout (preregistered)

| Outcome | Interpretation |
| --- | --- |
| Same-info TVs ≈ block noise; ≪ cross-rep TVs | Effect not explained by reformatting/length alone |
| Same-info TVs large | Operator also sensitive to serialization; weakens “information content only” reading |

## Explicit non-actions

- Choosing padding text after seeing results  
- Adding centers/intervals arms into this control budget  
- Upgrading source/feedback claims from this control  

## Auth

`analysis/matched_rep_collective/omap_auth_go.json` remains `paid_authorized: false` until explicit user GO.
