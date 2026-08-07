# Matched 3×3 replay — primary results (frozen)

Date: 2026-07-24  
Status: **PAID COMPLETE** — 4,608/4,608 valid; primary field-blocked inference frozen  
Parent macro: Outcome A (`docs/MATCHED_REP_COLLECTIVE_RESULTS.md`)  
Artifacts: `analysis/matched_rep_collective/replay_primary/`

## Acquisition integrity

| item | value |
| --- | --- |
| Run directory | `runs/matched_rep_collective_replay/matched-rep-collective-3x3-replay-v0.1_1e976b79b71d/20260724T114515Z` |
| Calls | **4,608 / 4,608** |
| Valid | **4,608** (rescued 0, unrecoverable 0) |
| Blocks | 16×2 (stable) |
| Field selection | pre-Outcome-A lock; 48 unique; source 16/16/16; stratum 8×6; fallback 0 |
| Actual total cost | **\$1.876** (all 4608 calls from trace tokens) |
| Session cost (post-preflight accounting) | \$1.873 |
| Preflight | 12 in-budget tasks (included in 4608) |
| Ceiling | \$12 |

Hashes: see `analysis/matched_rep_collective/replay_primary/decision.json` → `hashes`.

## Inference unit

**48 physical fields** (not 4,608 independent calls).  
Primary globals only (α = 0.05):

1. Target main effect (operator)  
2. Source main effect (endogenous manifold)  
3. Source × target interaction (feedback)

Pairwise TV / ΔA / Δa0 and stratum cells are **effect-size decompositions**, not a family of primary tests.

## Primary global tests

### 1. Target main effect — **supported**

Field-blocked permutation of target labels across the 96 responses within each field  
(statistic: mean pairwise \(d_{\mathrm{TV}}\)).

| quantity | value |
| --- | ---: |
| Observed mean pairwise TV | 0.344 |
| Null mean (perm) | ≪ observed |
| \(p\) (one-sided) | **0.0002** |
| Multinomial deviance \(p\) | **0.0002** |
| Noise-floor paired \(p\) (between−within) | **0.0002** |

Field-cluster bootstrap means (95% CI):

| pair | \(\overline{d_{\mathrm{TV}}}\) | 95% CI |
| --- | ---: | --- |
| Moments–Centers | 0.335 | [0.251, 0.423] |
| Moments–Intervals | 0.408 | [0.323, 0.490] |
| Centers–Intervals | 0.290 | [0.221, 0.363] |

Noise floor:

| | mean TV | 95% CI |
| --- | ---: | --- |
| Between-target | 0.344 | [0.298, 0.392] |
| Within-target, between-block | 0.092 | [0.076, 0.109] |
| Ratio | **3.76×** | |

→ Operator separation ≫ acquisition drift.

### 2. Source main effect — **suggested (not established at α=0.05)**

Stratum×\(K\)-blocked permutation of source labels; statistic = SSD of source-mean \(a_0\)  
(target-averaged within field).

| quantity | value |
| --- | ---: |
| \(p\) (one-sided) | **0.077** |
| Moments mean \(a_0\) | −0.115 |
| Centers mean \(a_0\) | −0.119 |
| Intervals mean \(a_0\) | +0.244 |

Effect-size contrast (bootstrap; not a primary test):  
intervals − moments \(\Delta a_0 = 0.359\), CI [0.021, 0.686] (excludes 0).  
Global blocked test remains the primary call → **suggested, not established**.

### 3. Source × target interaction — **suggested (not established at α=0.05)**

Same blocking; statistic = SSD across sources of mean \(\Delta A^{M-I}\).

| quantity | value |
| --- | ---: |
| \(p\) (one-sided) | **0.066** |

\(\Delta A^{M-I}\) by source (effect size):

| source | mean \(\Delta A^{M-I}\) | 95% CI |
| --- | ---: | --- |
| Moments | 0.191 | [0.060, 0.355] |
| Centers | 0.053 | [0.016, 0.100] |
| Intervals | 0.049 | [−0.059, 0.170] |

Asymmetry (moments-source × intervals-target lower activity) is visible as effect size but does **not** pass the primary global interaction test.

## Mechanistic verdict (preregistered readout)

| mechanism | verdict |
| --- | --- |
| Operator effect (target) | **Supported** |
| Endogenous manifold (source) | **Suggested** (\(p=0.077\)) |
| Representation-dependent feedback | **Not established** (\(p=0.066\)) |

Frozen claim (feedback withheld):

> Same physical fields elicit representation-dependent operators (strong target main effect). Together with Outcome A (moments-only polar lock at positive \(K\)), this supports operator-driven macroscopic transmutation. Endogenous source and feedback remain secondary pending stronger panels or second-model replication.

Full feedback claim is **not** asserted.

## Figures

- `replay_primary/figures/fig6a_target_pair_tv.png`  
- `replay_primary/figures/fig6b_action_matrix.png`  
- `replay_primary/figures/fig6c_interaction_contrasts.png`  
- `replay_primary/figures/fig6d_noise_floor.png`

## Explicitly deferred / exploratory

- Surrogate refit  
- Cell-wise multiplicity of stratum×source×target p-values  
- Post-hoc field dropping  
- Second-model-family micro replication  
- Observation-map / length controls  

## Machine-readable freeze

`analysis/matched_rep_collective/replay_primary/decision.json`

## Next Nat Commun gates

1. Second-model-family micro replication  
2. One observation-map / length control  
3. Open-data manifest + figure integration  
4. Manuscript draft  

No new large matched-collective acquisition is required for this mechanistic gate.
