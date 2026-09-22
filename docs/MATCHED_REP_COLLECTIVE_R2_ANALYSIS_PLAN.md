# Matched collective R2 — analysis plan (preregistered)

Date: 2026-07-25  
Status: **frozen before paid acquisition**  
Inference unit: **physical seed** (not agents, timesteps, or API calls).

## Primary endpoint

For each seed \(s\) and representation \(R\),

\[
L_{sR}
=
\frac12
\sum_{K\in\{0.08,0.15\}}
\mathbf{1}[\text{sustained polar lock}],
\qquad
L_{sR}\in\{0,\tfrac12,1\}.
\]

Sustained polar lock = finite \(t\) such that \(r_1(s)>0.9\) for all \(s\in[t,T]\) (GPT v0.2 definition).

### Primary global test (core 6 seeds)

Exact seed-blocked permutation of representation labels. Within each seed, the pair of positive-\(K\) outcomes travels with the label; permute \((M,C,I)\).

\[
(3!)^6 = 46{,}656 \quad\text{permutations (exhaustive)}.
\]

Statistic:

\[
T_L
=
\sum_R
\bigl(\overline L_R - \overline L\bigr)^2.
\]

\[
H_0:\ L_M=L_C=L_I,\qquad
p_{\mathrm{perm}} = \Pr(T_L^{\mathrm{null}}\ge T_L^{\mathrm{obs}}).
\]

PASS threshold: \(p_{\mathrm{perm}}<0.05\).

Implementation: `analysis/matched_rep_collective/analyze_r2_macro.py`.

## Key secondary inference

### Final polar order

\[
Y_{sR}
=
\frac{r_1(T;K{=}0.08)+r_1(T;K{=}0.15)}{2}.
\]

Same exact seed-blocked permutation.

### Pairwise contrasts

Report \(M{-}C\), \(M{-}I\), \(I{-}C\) with seed-level paired points, mean difference, bootstrap CI, exact sign-flip permutation. Primary remains the global test.

### Phenotype hierarchy (descriptive)

- phenotype matrix (core + held-out)
- lock / sustained-lock fractions
- \(Q_2\), final/mean \(r_1\)

GPT ranking is **not** required for PASS.

## Replication gates

### Gate A — Core macro effect (**required**)

1. Sustained-lock global perm \(p<0.05\)
2. Final-\(r_1\) global perm \(p<0.05\)
3. Operational phenotype difference across \(\mathcal R\) at ≥1 positive \(K\)
4. Phase difference not driven by a single seed
5. Seed-paired effect direction broadly consistent

### Gate B — Qualitative phase separation (**near-required**)

At ≥1 positive \(K\):

- one representation: \(\ge 5/6\) core seeds `polar_locked`
- another: \(\le 1/6\) `polar_locked`

Held-out seeds must show the **same direction**.

### Gate C — GPT map replication (**optional strong**)

\[
r_1^{M} > r_1^{C} > r_1^{I}
\quad\text{and}\quad
Q_2^{I}>Q_2^{M},Q_2^{C}
\]

or phenotype hierarchy moments lock / centers partial / intervals high-\(Q_2\).

## \(K=0\) control

Exact \(r_m\) coincidence (tol \(10^{-12}\)). Failure ⇒ invalid experiment.

## Result reading guide

| outcome | interpretation |
| --- | --- |
| Lock/nonlock splits by \(\mathcal R\) | Macro phase selection replicated in independent family |
| Ordering effect, all nonlock | Macro effect yes; qualitative phase replication no |
| All lock; paths/\(Q_2\) differ | Pathway dependence; phase-selection claim weakened |
| No Claude difference | GPT macro effect does not generalize under this protocol |
| Reverse hierarchy vs GPT | Representation dependence generalizes; phenotype map family-specific (**not a failure**) |
| \(K=0\) mismatch | Engine / matching failure |

## Held-out panel

Secondary confirmation only; does **not** enter the primary \(T_L\) exact test on core 6. Report lock fractions and direction consistency for Gate B.
