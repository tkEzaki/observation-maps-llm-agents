# Matched representation collective — cross-encoding replay lock

Date: 2026-07-24  
Purpose: mechanistic decomposition after surrogate-free matched collectives

> Do **not** revise strata quotas, sampling rules, or call budgets after
> inspecting matched collective macroscopic results (Outcome A confirmed
> 2026-07-24). Exploratory extras must be labeled post-hoc.

## Scientific target

Decompose collective representation differences into:

\[
\text{representation effect}
=
\underbrace{\text{operator effect}}_{g_{\mathcal R}}
+
\underbrace{\text{endogenous field-distribution effect}}_{\mu_{\mathcal R}}
+
\underbrace{\text{feedback interaction}}_{g_{\mathcal R}\leftrightarrow\mu_{\mathcal R}}.
\]

On identical physical fields \(\rho\), measure

\[
g_{\mathcal R}(\rho)=\mathbb{E}[f\mid\mathcal R(\rho)]
\]

and the transfer matrix

\[
g_{\mathcal R_{\mathrm{target}}}\bigl(\rho_{\mathcal R_{\mathrm{source}}}\bigr).
\]

## Call budget (frozen)

\[
48\ \text{fields}
\times 3\ \text{representations}
\times 32\ \text{responses}
= 4{,}608\ \text{calls}.
\]

Separate cost card to be issued after matched collectives complete; this
document freezes **selection rules only**.

## Field strata (48 fields)

| stratum | n fields | definition (algorithmic) |
| --- | ---: | --- |
| early_transient | 8 | \(t\in[5,15]\), \(r_1<0.35\) |
| pre_onset | 8 | within 10 steps before first \(r_1\ge0.5\) (if exists); else mid-run low-\(r_1\) |
| post_onset | 8 | within 10 steps after first \(r_1\ge0.5\); else high-activity mid-run |
| ordered_state | 8 | \(r_1\ge0.8\) for ≥5 consecutive steps |
| negative_K | 8 | from \(K=-0.15\) runs only; prefer high \(\lvert\tau\rvert\) |
| high_r2 | 8 | \(r_2\ge0.45\) and \(r_2>r_1\) |

If a stratum undersupplies, fill from the nearest stratum with documented
shortfall — **do not** invent new strata after outcomes.

## Source-encoding balance

Among the 48 fields, target approximately equal contribution from each
collective source encoding:

| \(\mathcal R_{\mathrm{source}}\) | target n |
| --- | ---: |
| moments_m1_m3 | 16 |
| centers_24_standard | 16 |
| intervals_24_decimal6 | 16 |

Within each source, spread strata as evenly as integer arithmetic allows
(at least one field per stratum per source when possible).

## Selection algorithm (frozen)

1. After all matched runs finish, load `phases.npy` / metrics for every
   `(representation, K, seed, t)`.
2. Tag candidate frames by stratum predicates above.
3. Deduplicate by physical histogram hash (`physical_hash` of 24-bin
   relative-phase histogram in focal frame, peer=16).
4. Greedy fill quotas: round-robin sources × strata; prefer median-\(r_1\)
   exemplars within stratum (avoid extreme outliers unless stratum is
   `high_r2` / `negative_K`).
5. Freeze selected field list + hashes to
   `analysis/matched_rep_collective/replay_fields_v0_1.json`
   **before** launching replay API calls.
6. Present each frozen field under **all three** target encodings with
   32 i.i.d. responses (offset 0; peer_count=16).

## Primary replay endpoints

Per (field, target encoding):

- \(\hat p_-,\hat p_0,\hat p_+\), activity \(A\), signed mean action \(a_0\);
- complex harmonics \(C_1,C_2\) of the response curve when offsets are not
  used (here offset-0 multinomial only — report \(p,A,a_0\) as primary);
- pairwise TV / \(\Delta a_0\) across target encodings on the same field;
- 3×3 mean TV matrix averaged within stratum and overall.

## Explicit non-actions

- Retuning strata after seeing which representation “won” macroscopically
- Mixing Stage B training fields into this 48 without labeling them as
  controls (controls, if any, are extra and outside the 4,608 budget)
- Using surrogates to choose fields

## Implementation note

Field selection audited in
`analysis/matched_rep_collective/select_replay_fields.py` →
`replay_fields_v0_1.json` (AUDIT_PASS; 0 fallbacks).

Runner: `experiments/stage_c/run_matched_rep_collective_replay.py`  
(freeze re-verify, exact 4608=48×3×16×2, auth gate, raw retention,
primary 3×3 before surrogate). Mock smoke:
`analysis/matched_rep_collective/run_replay_mock_smoke.py`.

Source reproducibility (no git): `replay_source_hashes_v0_1.json`
(`runner_source_sha256`, `encoder_sources_sha256`, `selector_source_sha256`,
`dependency_snapshot_sha256`).

Paid acquisition **COMPLETE** (2026-07-24): 4,608/4,608 valid;
actual \$1.876; primary inference in
`analysis/matched_rep_collective/replay_primary/decision.json`.
Selection rules herein remain binding and must not be retuned.
