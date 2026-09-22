# Matched collective R2 — Claude macroscopic replication hypothesis

Date: 2026-07-25  
Status: **preregistered / estimate-only (paid NOT authorized)**  
Parent macro: GPT matched collective v0.2 (`docs/MATCHED_REP_COLLECTIVE_RESULTS.md`)  
Parent micro: R1 Claude PASS (`docs/MATCHED_REP_COLLECTIVE_R1_RESULTS.md`)

## Selection rationale (not random)

Claude Haiku (`claude-haiku-4-5-20251001`) is the second family by **targeted replication** from R1 micro evidence on the frozen 48-field panel:

- valid rate 1.000
- target operator effect \(p=0.0002\)
- pairwise \(d_{\mathrm{TV}}\) comparable to GPT
- between/within ratio \(20.4\times\)

This is **not** a random draw among all available families.

## Primary scientific question

> Does GPT’s observation-representation selection of collective phenotype replicate in an independently developed Claude family under matched physical dynamics?

GPT positive-\(K\) outcome (6 seeds):

| representation | polar lock |
| --- | ---: |
| moments | 6/6 |
| centers | 0/6 |
| intervals | 0/6 |

R2 tests the **Claude-only** representation effect. Numerical match to GPT and exact ranking are **not** primary requirements.

## Claim hierarchy

| Tier | Meaning |
| --- | --- |
| Macro representation effect | Claude: representation changes collective outcome |
| Qualitative phase replication | ≥1 representation locks; another does not |
| GPT phenotype-map replication | moments lock → centers partial → intervals high-\(Q_2\) hierarchy |

Tiers 1–2 are the the study primary gates. Tier 3 is strong but optional.

## Design summary

- Family: Anthropic / `claude-haiku-4-5-20251001` (R1-identical fixed ID; no `latest`)
- \(N=17\), \(T=100\); \(\mathcal R\in\{\mathrm{moments},\mathrm{centers},\mathrm{intervals}\}\)
- Core 6 seeds: \(K\in\{0,0.08,0.15\}\) (same physical ICs as GPT matched collective)
- Held-out 4 seeds: \(K\in\{0.08,0.15\}\) only
- Drop \(K=-0.15\) (secondary torque-sign; not needed for positive-\(K\) phase gate)
- Budget: **132,600** calls; protocol ceiling **\$175**

## Frozen claim templates

**Core PASS:**

> In a second, independently developed language-model family, changing only the observation representation again selected different collective outcomes under matched physical dynamics.

**Same hierarchy:**

> The moments-specific polar-locking phenotype replicated in an independent model family.

**Effect without GPT hierarchy:**

> Macroscopic representation dependence generalized across model families, whereas the mapping from representation to collective phenotype was model-family dependent.

## Non-claims

- Same phase diagram for all LLM families
- Identical Claude/GPT operators
- Complete mediation of the macro effect
- Thermodynamic phase transition / critical \(K_c\)
- Established source manifold / feedback (replay supports **target** operator only)
