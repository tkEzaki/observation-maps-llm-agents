# Intervals branch (third representation — collective path)

Date: 2026-07-24  
Status: **STOPPED (collective surrogate closed); microscopic + IID-compressibility archive retained**

## Scientific question

> Can an interval-bin encoding of the same physical field yield a predictable
> microscopic response operator that prospectively transports to
> collective-like finite-peer fields, and—only after freeze—does that
> operator select different macroscopic dynamics from moments?

Separate from day one:

1. Microscopic representation-induced response law
2. Prospective collective-field compressibility / Stage C readiness

## Lessons locked from Centers (do not repeat)

- Split collective peer{8,16} from dense peer=240 exploratory scope
- Baseline = **peer-specific global**, not pooled global
- Separate response operator from \(p_{\mathrm{fail}}\)
- Require collective-like prospective transfer (not only OOF)
- At most one coverage pilot + one confirmatory pilot
- Second prospective failure → STOP
- Freeze metrics include activity, signed action, torque — not \(r_1\) alone
- Do **not** transplant the centers model family mechanically

## Roadmap

| step | work | status |
| --- | --- | --- |
| INT-0 | dataset snapshot + scope lock | **done** |
| INT-1 | phenotype / serialization / missingness audit | **done** |
| INT-2 | model / AD / risk / support-map offline gates | **done** — model/risk PASS; support FAIL |
| INT-2b | support decompose + peer16-only + discrete bake-off | **done — STOP** (`docs/INTERVALS_FINAL_DECISION.md`) |
| INT-3 | coverage-directed prospective pilot | **cancelled** |
| INT-4 | confirmatory (max 1) | **cancelled** |
| INT-5 | `intervals_bundle_v1` freeze | **cancelled** |
| INT-6 | moments↔intervals matched Stage C | **cancelled** |

## Formal close

```text
status = stopped
paid_authorized = false
freeze_ready = false
stage_c_authorized = false
```

Manifest: `analysis/intervals_branch/final_manifest.json`  
Decision: `docs/INTERVALS_FINAL_DECISION.md`


## Scope locks

```text
intervals_collective_scope
  peer ∈ {8,16}
  intended N ∈ {9,17}
  optional later narrow: intervals_collective_bundle_v1_N17 (peer=16)

intervals_dense_exploratory
  peer = 240
  not used for collective freeze
```

Representation id: `intervals_24_decimal6`

## Stop rule

After INT-3/INT-4, if peer-global remains best on prospective fields, signed
direction collapses, or collective-like regimes remain unreproducible →
**STOP** intervals collective surrogate path and archive as microscopic-only.

## Artifacts

- `analysis/intervals_branch/`
- Drivers under `analysis/intervals/`
- Decision docs under `docs/INTERVALS_*.md`
