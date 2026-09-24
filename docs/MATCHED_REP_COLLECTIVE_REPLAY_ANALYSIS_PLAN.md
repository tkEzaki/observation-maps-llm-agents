# Matched 3×3 replay — primary analysis plan (frozen pre-acquisition)

Date: 2026-07-24  

## Inference unit

**48 physical fields** (not 4,608 calls). Use field-blocked permutation /
cluster bootstrap / field random intercepts. Stratum and source are explicit
design factors.

## Acquisition blocks (optional, same call budget)

Prefer storing 32 responses as \(16\times 2\) acquisition blocks for drift /
reproducibility checks. Total calls remain 4,608. This is **not** a change to
field selection or scientific hypotheses.

## Layer 1 — Target representation effect (operator)

On identical physical field \(f\), compare

\[
\widehat{\mathbf p}_{f,R}=(\hat p_-,\hat p_0,\hat p_+)
\]

across target encodings \(R\).

Primary:

\[
d_{\mathrm{TV}}(\widehat{\mathbf p}_{f,R_1},\widehat{\mathbf p}_{f,R_2}),
\qquad
\Delta a_0,\qquad
\Delta A.
\]

## Layer 2 — Source-field effect (\(\mu_{\mathcal R_{\mathrm{source}}}\))

After averaging or modeling over targets, test whether action distributions
depend on which collective trajectory produced \(\rho\) (moments / centers /
intervals source).

## Layer 3 — Source × target interaction (feedback)

Test whether

\[
\Delta_{R_1,R_2}(\rho_S)
\]

varies with source \(S\). Interaction supports representation-dependent
feedback.

## Mechanistic readout (preregistered)

| Replay pattern | Conclusion |
| --- | --- |
| Target main only | Operator difference on shared fields |
| Source main only | Endogenous field-distribution difference |
| Target + source | Additive operator + state-distribution |
| Source × target | Representation-dependent feedback |
| Target ≈ 0 | Macro difference mainly manifold branching |
| All weak | 48-field / offset-0 panel insufficient |


## Explicit non-actions before primary report

- Surrogate refit before 3×3 primary tables  
- Post-hoc field dropping (except preregistered pathology)  
- Replacing frozen fields after hash freeze without full reselection
