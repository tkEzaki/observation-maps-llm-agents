# Centers branch (second representation)

Date: 2026-07-24  
Status: **STOPPED (collective surrogate closed); microscopic archive retained**

## Formal close

See `docs/CENTERS_FINAL_DECISION.md` and
`analysis/centers_branch/final_manifest.json`:

```text
status = stopped
paid_authorized = false
freeze_ready = false
stage_c_authorized = false
```

| layer | conclusion |
| --- | --- |
| Microscopic response transmutation | supported |
| Peer-stratified in-distribution compressibility | supported |
| Prospective collective-field surrogate | not supported |

Next representation priority: **intervals** (`docs/INTERVALS_BRANCH.md`).


> Can a center-bin encoding of the same physical field yield a predictable
> microscopic response operator, and does that operator select different
> macroscopic dynamics from moments?

Two stages (do not skip ahead):

1. Centers-alone micro→macro predictability (`centers_bundle_v1` freeze)
2. Moments vs centers matched Stage C (after freeze + offline disagreement scan)

## Roadmap

| step | work | status |
| --- | --- | --- |
| CENT-0 | hash-lock centers training snapshot | **done** |
| CENT-1 | centers model / native-distance bake-off | **done** |
| CENT-2 | centers AD + predictive risk offline gates | **done** (pass) |
| CENT-3 | 36-field coverage-directed fixed-field pilot | **paid complete; prospective risk PASS** |
| CENT-4A/B/C | missingness + pre-refit bake-off + control audit | **done** — `docs/CENTERS_CENT4_OFFLINE.md` |
| CENT-4 (model revision) | peer{8,16} stratified direct/EMD revision | **done** — `docs/CENTERS_CENT4_REVISION.md` |
| CENT-4 final offline | cluster bootstrap + local support + peer16-only | **done — STOP** — `docs/CENTERS_FINAL_OFFLINE.md` |
| CENT-4 (freeze) | `centers_collective_bundle_v1` / `_N17` | **NO-GO** (stop rule) |
| CENT-5 | offline moments↔centers disagreement scan | **cancelled** (no centers freeze) |
| CENT-6 | matched moments vs centers Stage C | **cancelled** |

## Design locks

- Representation id: `centers_24_standard`
- Collective-scope bundle: **peer ∈ {8,16} only** (intended \(N\in\{9,17\}\))
- Dense peer=240 held in `centers_dense_exploratory` (not Stage C training)
- No moments / intervals response rows in centers targets
- No collective_replay in centers training (moments-only)
- No cyclic-shift augmentation (serialization effects preserved)
- Rotation-invariant features for shape support; native 24 bins keep orientation
- Moments thresholds are **not** copied; centers noise calibrates freeze gates
- Response operator and protocol reliability (\(p_{\mathrm{fail}}\)) are separate layers
- Next paid step is a **non-overlapping confirmatory fixed-field pilot** only after Gates 4–5 pass — not Stage C

## CENT-0 snapshot

| item | value |
| --- | --- |
| Path | `analysis/centers_branch/` |
| Rows | 2736 |
| Aggregate sha256 | `2a60c21be2f6145f3761337b8239884d4d87be58b925919f78fe8e328eb08454` |
| Exact-balance mean stay | ≈0.44 (not moments-like near-total abstention) |
| Mean \(p_{\mathrm{stay}}\) | ≈0.136 |

Sources: transmutation, stimulus manifold, signed antipodal weight, sparse-peer, Stage 3A pilot (centers responses only).

## CENT-1 / CENT-2 results

Acquisition-block holdout ranking (log-loss):

| model | log_loss | mean TV |
| --- | ---: | ---: |
| **activity_direction_hurdle** | **0.714** | **0.286** |
| kernel_nn_euclidean | 0.721 | 0.295 |
| hybrid_hellinger | 0.754 | 0.326 |
| softmax_stump_boost | 0.813 | 0.351 |
| global_empirical | 0.944 | 0.401 |
| soft_regime_mixture | 3.681 | 0.314 |

Hurdle was a **candidate**, not the default: it wins grouped CV over direct multinomial / hybrid kernels, so it is production for CENT-3.

Native distance holdout (hybrid kernels only): circular EMD slightly beats Hellinger
(log-loss 0.735 vs 0.736). Production remains the hurdle model.

Offline risk gates (`cent12_decision.json`):

| gate | result |
| --- | --- |
| OOF risk–TV Spearman | **+0.383** pass |
| Risk-bin monotone | pass |
| High-risk enrichment (top/bottom 20%) | **1.84×** pass |
| Beats global on holdout | pass |

## CENT-3 pilot (complete)

See `docs/CENTERS_CENT3_PILOT_REVIEW.md`, `docs/CENTERS_CENT3_PROSPECTIVE.md`,
and `docs/CENTERS_CENT3_COST_CARD.md`.

- 36 physical fields × 1 representation × 16 responses × 2 blocks = **1152** calls
- Actual cost **\$0.622** (ceiling \$1.773)
- Prospective risk **PASS**; response accuracy insufficient for freeze

## CENT-4 offline (complete → STOP)

See `docs/CENTERS_CENT4_OFFLINE.md`, `docs/CENTERS_CENT4_REVISION.md`, and
`docs/CENTERS_FINAL_OFFLINE.md`.

- Peer stratification restores **cluster-robust** in-distribution OOF gains
- CENT-3 is a **development-transfer benchmark** only
- Local support: peer16 ±unimodal = **neighbors_active_feature_gap**
- Peer16-only / N=17 bundle: OOF PASS, CENT-3 transfer FAIL
- Stay-collapse unreproducible under family holdout
- **Stop rule fired** — no further centers collective surrogate engineering;
  confirmatory paid / freeze / Stage C remain NO-GO

## Non-claims

- Centers is not “better” than moments a priori
- Encoding differences are not claimed as semantic understanding
- Freeze is not automatic from offline Spearman alone
- CENT-3 risk PASS does not authorize a collective bundle
- In-distribution OOF gains do not authorize confirmatory spend without
  development-transfer + phenotype gates
- Offline “densification” cannot create new LLM labels
