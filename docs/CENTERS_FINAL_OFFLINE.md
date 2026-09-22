# Centers — final bounded offline attempt

Date: 2026-07-24  
Status: **STOP centers collective-surrogate path; confirmatory paid NO-GO; freeze NO-GO**

CENT-3 is treated hereafter as a **development-transfer benchmark**, not as
final prospective evidence.

## Decision

| item | decision |
| --- | --- |
| Cluster-bootstrap OOF gain | **PASS** (strong) |
| Peer16-only in-distribution OOF | **PASS** |
| CENT-3 development transfer | **FAIL** |
| Stay-collapse reproduction | **FAIL** |
| Confirmatory paid (any scope) | **NO-GO** |
| `centers_collective_bundle_v1` / `_N17` freeze | **NO-GO** |
| Centers Stage C | **NO-GO** |
| Further centers surrogate engineering | **STOP** |

Locked wording:

> Center-bin serialization induces reproducible microscopic transmutation
> and peer-regime dependence, but its finite-peer response surface was not
> sufficiently stable or compressible for prospective collective prediction
> under the tested descriptors and coverage.

Recommended scientific next step: **intervals** representation branch (or
archive centers as microscopic-only).

---

## A. Cluster bootstrap (Gate 2 re-check)

Winner: `peer_specific_activity_direction` vs `peer_specific_global`
(acquisition-block OOF, peer{8,16} train).

| clustering | mean ΔLL | 95% CI | entirely >0? |
| --- | ---: | --- | --- |
| row (reference) | +0.283 | — | — |
| **physical-field** (56 clusters) | +0.283 | **[0.247, 0.429]** | **yes** |
| **profile-family** (41 clusters) | +0.283 | **[0.246, 0.488]** | **yes** |
| acquisition-group (2 blocks) | +0.283 | n/a (only 2 clusters) | point >0 |

In-distribution compressibility after peer stratification is **not** an
artifact of row-level dependence.

Artifacts: `analysis/centers_branch/cent4_final_offline_decision.json`

---

## B. Local-support diagnosis (CENT-3)

Script writes `cent4_final_local_support.csv` and
`cent4_final_controls_support.json`.

### Unimodal / sparse controls

| field | diagnosis | reading |
| --- | --- | --- |
| `cent_ctrl_unimodal_positive_polar` | **neighbors_active_feature_gap** | peer16 neighbors exist but respond actively; stay-collapse not in training support |
| `cent_ctrl_unimodal_sign_reversed` | **neighbors_active_feature_gap** | same |
| `cent_ctrl_sparse_unimodal` | **local_noncompressibility** | neighbor responses disperse (high TV) |

So the binding failure is **not** “no nearest neighbors.” It is that the
presented finite-peer abstention phenotype is **absent or non-unique** in the
peer16 training manifold under native/canonical distance — a
coverage/representation gap that offline reweighting cannot invent labels for.

---

## C. Discrete bin features

Added from the serialized 24-bin vector (no field IDs):

occupied / zero counts, max zero-run, max mass, top-2 gap, peak distance,
antipodal pairing, focal mass, boundary mass.

On peer16-only AB OOF, `canonical_emd + discrete` improves over plain
canonical EMD (0.853 vs 0.878 log-loss) but remains behind the hurdle and
**does not** repair CENT-3 transfer or stay-collapse.

---

## D. Peer16-only / N=17 bundle

```text
centers_collective_bundle_v1_N17
  support: peer=16 (n=1430)
  intended N: 17
```

### In-distribution (PASS)

| model | AB log-loss | \(a_0\) MAE | sign acc |
| --- | ---: | ---: | ---: |
| **peer16_activity_direction** | **0.761** | 0.382 | 0.853 |
| peer16_canonical_emd+discrete | 0.853 | 0.432 | 0.823 |
| peer16_canonical_emd | 0.878 | 0.451 | 0.767 |
| peer16_global | worse | — | — |

Field-cluster ΔLL vs peer16-global: CI entirely positive.

### CENT-3 peer16 development (FAIL)

Only 13 peer16 CENT-3 fields. Ranking:

| model | log-loss | \(a_0\) MAE |
| --- | ---: | ---: |
| **peer16_global / peer_specific_global** | **1.087** | **0.473** |
| peer16_activity_direction | 5.38 | 0.651 |
| peer16_canonical_emd | ~5.5 | ~0.60 |

Structured models collapse on the development set (near-zero probability on
observed actions). Peer16-only does **not** rescue collective transport.

### Stay-collapse family holdout (FAIL)

Holding out high-\(|z_1|\) / low-antipodal peer16 rows (n=16): observed mean
stay ≈0.10 (training “unimodal-like” peer16 is **not** the CENT-3 all-stay
regime). All models predict stay≈0. Offline holdout therefore cannot certify
the CENT-3 abstention phenotype — confirming the local-support feature gap.

---

## GO checklist (final)

| # | condition | status |
| ---: | --- | --- |
| 1 | cluster bootstrap structured ≫ peer-global | **PASS** |
| 2 | peer16-only OOF improves | **PASS** |
| 3 | CENT-3 development ≫ peer-global | **FAIL** |
| 4 | stay-collapse family holdout / controls | **FAIL** |
| 5–8 | a0 / risk / non-overlap pilot prep | blocked |

**Confirmatory pilot GO: NO**  
**Peer16-only pilot GO: NO**  
**Stop rule: FIRED**

---

## Commands

```powershell
py analysis/centers/run_cent4_final_offline.py
```

## What remains scientifically valuable

1. Representation-induced microscopic transmutation (Stage B) for centers.
2. Peer-regime dependence: peer240 dense directional vs peer16 finite-peer
   abstention are different phenotypes.
3. Peer stratification restores in-distribution compressibility (cluster-robust).
4. Collective Stage C for centers is **not** supported under tested coverage
   and descriptors — without further paid surrogate chasing.
