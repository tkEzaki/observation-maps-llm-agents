# Matched representation collective — results v0.1 (analysis revision v0.2)

Date: 2026-07-24  
Status: **COMPLETE — Outcome A (macroscopic transmutation)**  
Analysis revision: **v0.2** (phenotype / \(Q_2\) / sustained-lock / exact sign tests)  
Family: `matched_representation_collective` (surrogate-free)

## Causal claim (scoped)

> Holding initial phases, natural frequencies, \(N\), \(K\), \(T\), system
> prompt, action set, and integrator fixed, an intervention on the
> **observation representation** \(\mathcal R(\rho)\) selects different
> collective phenotypes of frozen language-model agents.

We do **not** claim a pure string-format effect independent of informational
content; moments / centers / intervals also differ in which features of
\(\rho\) are retained.

## Sessions

| representation | session | valid calls |
| --- | --- | ---: |
| moments_m1_m3 | `…/matched-rep-collective-v0.1-moments_ed41bff41a9a/20260724T065125Z` | 40800/40800 |
| centers_24_standard | `…/matched-rep-collective-v0.1-centers_65c4f06d18a1/20260724T065125Z` | 40800/40800 |
| intervals_24_decimal6 | `…/matched-rep-collective-v0.1-intervals_1d533f018b96/20260724T065128Z` | 40800/40800 |

Design: \(N=17\), \(K\in\{-0.15,0,0.08,0.15\}\), \(T=100\), 6 paired seeds.  
Tables: `analysis/matched_rep_collective/paired_analysis/`

## Verdict

**Outcome A — macroscopic transmutation supported.**

At positive \(K\), every paired seed that polar-locks under moments fails to
lock under centers and under intervals (\(\Delta\mathrm{lock}=-1\) per seed).

Exact two-sided sign test (moments locks & other does not vs reverse), each of
\(K\in\{0.08,0.15\}\), each of centers/intervals vs moments:

\[
p = \frac{2}{2^6} = 0.03125.
\]

Primary narrative endpoints: locking / sustained lock, final & mean \(r_1\),
\(Q_2=r_2-r_1\), phenotype labels. Bootstrap CIs on many secondary deltas are
archived but **not** centered in the text (avoid “many significant events”
multiplicity framing).

---

## Phenotypes (revised labels)

| label | rule |
| --- | --- |
| `polar_locked` | \(r_1(T)\ge 0.9\) |
| `high_r2_nonpolar` | \(r_2(T)\ge 0.5\) and \(r_2(T)>r_1(T)\) |
| `partial_polar_order` | \(0.35\le r_1(T)<0.9\) and not high-\(r_2\) nonpolar |
| `low_polar_active` | \(r_1(T)<0.35\) (includes active polar-order suppression at \(K<0\)) |

Counts (n=24 runs / encoding):

| \(\mathcal R\) | polar_locked | partial_polar_order | high_r2_nonpolar | low_polar_active |
| --- | ---: | ---: | ---: | ---: |
| moments | 12 | 0 | 0 | 12 |
| centers | 0 | 11 | 1 | 12 |
| intervals | 0 | 4 | **7** | 13 |

Hierarchy (positive \(K\)):

\[
\text{moments polar lock}
\;\rightarrow\;
\text{centers partial polar order}
\;\rightarrow\;
\text{intervals multicluster-leaning (high }Q_2\text{) order}.
\]

---

## Positive \(K\): qualitative phase separation

### \(K=0.08\) (means over 6 seeds)

| \(\mathcal R\) | \(r_1(T)\) | \(r_2(T)\) | \(Q_2=r_2-r_1\) | lock | sustained lock |
| --- | ---: | ---: | ---: | ---: | ---: |
| moments | 0.999 | 0.995 | −0.004 | 1.00 | 1.00 |
| centers | 0.710 | 0.606 | −0.103 | 0.00 | 0.00 |
| intervals | 0.382 | 0.510 | **+0.128** | 0.00 | 0.00 |

### \(K=0.15\)

| \(\mathcal R\) | \(r_1(T)\) | \(r_2(T)\) | \(Q_2\) | lock | sustained lock |
| --- | ---: | ---: | ---: | ---: | ---: |
| moments | 0.996 | 0.982 | −0.013 | 1.00 | 1.00 |
| centers | 0.727 | 0.616 | −0.110 | 0.00 | 0.00 |
| intervals | 0.507 | 0.720 | **+0.213** | 0.00 | 0.00 |

**\(r_2\) wording (corrected).** Absolute \(r_2\) is highest under moments because
polar lock raises all harmonics. The intervals signature is **elevated second
harmonic relative to polar order** (\(Q_2>0\)), consistent with
multicluster-leaning partial states — not “higher \(r_2\) than moments.”

First-passage times \(t_{r_1>0.9}\) are **not** locking times (transient
crossings can occur without sustained lock). Prefer `sustained_lock` /
`t_r1_gt_0.9_sustained`. Singleton uncensored first-passage bootstrap CIs are
not used as primary evidence.

---

## \(K=0\): negative control

Order parameters \(r_1,r_2\) (final and mean) are **identical** across the three
encodings (shared free evolution of matched ICs). Activity and action bias still
differ. This supports:

> \(\mathcal R\) changes microscopic actions, but without coupling those actions
> cannot divert the macroscopic phase trajectory.

This strongly constrains “different IC / engine bug per representation”
alternative explanations.

---

## \(K=-0.15\): active polar-order suppression with torque sign structure

All encodings remain non-locked (`low_polar_active`). Mean social torque:

\[
\tau_{\mathrm{moments}}\simeq -0.335,\quad
\tau_{\mathrm{centers}}\simeq -0.033,\quad
\tau_{\mathrm{intervals}}\simeq +0.062.
\]

Intervals reverse the moments torque sign; collective frequency shifts
accordingly. Separate mechanistic signature from positive-\(K\) locking.

---

## Secondary contrast: intervals − centers

| \(K\) | \(\Delta r_1(T)\) | \(\Delta \overline{r_1}\) | \(\Delta Q_2(T)\) |
| ---: | ---: | ---: | ---: |
| 0.08 | −0.328 | −0.270 | +0.232 |
| 0.15 | −0.220 | −0.230 | +0.323 |

Intervals are weaker in polar order than centers and higher in \(Q_2\),
supporting the three-step phenotype hierarchy.

---

## Statistics policy (n=6 paired seeds)

- Inference unit = paired seed (correct).  
- **Primary:** exact sign test on locking; all paired points for \(r_1\), \(Q_2\).  
- Percentile bootstrap CIs: descriptive for continuous deltas; unstable when
  few finite first-passage pairs — not primary for \(t_{0.9}\).  
- Full paired tables: `paired_contrasts.csv`, `endpoints_long.csv`.

---

## Relation to surrogate archive

Moments’ Stage C surrogate success and centers/intervals transport failure
remain valid **auxiliary** results: the encodings that fail prospective
surrogate transport also fail to realize the moments locking phenotype under
matched endogenous dynamics.

---

## Nat Commun gates

| Gate | Status |
| --- | --- |
| Matched collective comparison | **PASS** |
| Robust macroscopic effect | **PASS — Outcome A** |
| Causal claim within tested system | **supported** |
| 3×3 cross-encoding replay | **scientific GO** (paid after cost/protocol review) |
| Second-model micro replication | required before submission |
| Observation-map / length controls | recommended |
| \(N=9\) size replication | after replay |

Replay field-selection rules remain those locked **before** these outcomes
(`docs/MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md`) — not retuned post hoc.

## Next

1. Cost card + human review for 4,608-call 3×3 replay → paid GO.  
2. Analyze source × target × interaction on frozen fields.  
3. Conditional R1 second-model micro panel.
