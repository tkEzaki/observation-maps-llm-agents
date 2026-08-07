# Serialization-binding ladder (SBC v0.1)

## Why this exists

The serialization-length control (`SLC v0.1`, Supplementary Fig. S19) refuted a
monotone prompt-length account of the encoding effect. It left one thing open.

Its compact histogram condition sits far from every other condition in response,
and that condition differs from the standard histogram condition in two ways at
once: it is 458 characters shorter, **and** it binds each mass to its bin
positionally rather than by an explicit label. Only the length half of that was
excluded. The two single-class reference contrasts in that figure cannot settle
it either, because their character gaps are 37 and 458, so their ratio is also
what a length account predicts.

This ladder holds the information fixed and varies the binding.

## Design

All four conditions serialize **the same 24 bin masses at the same precision**,
so the retained information is identical throughout. Only the binding of each
mass to its bin changes.

| condition | binding | form | chars |
|---|---|---|---|
| `centers_compact` | positional, grid stated by rule | one bare row | 971 |
| `centers_indexed_row` | explicit bin index per mass | one row, `b<k>=mass` | 1055 |
| `centers_centered_row` | explicit bin centre per mass | one row, `centre:mass` | 1114 |
| `centers_standard` | explicit index and centre | one labelled line per bin | 1429 |

The two anchor conditions are imported unchanged from the length control, so
`centers_standard` stays byte identical to the primary `centers_24_standard`
encoding and `centers_compact` stays byte identical to the arm already acquired.
The offline audit asserts both.

Both anchors are **re-acquired inside this run**, so every comparison is within
one acquisition and their agreement with the earlier acquisition is reported as a
run-to-run reproducibility check rather than assumed.

## The crossing

`centers_indexed_row` is the pivot. It sits **84 characters** from the positional
condition and **374** from the fully labelled one, a ratio of 4.4 to 1. It is
also the shortest condition in which every mass carries an explicit label.

- A prompt-length account predicts it responds like `centers_compact`.
- A binding account predicts it responds like `centers_standard`.

The length gaps favour the length account a priori, so the sign of the
prespecified contrast separates the two with no calibration between characters
and response distance.

## Prespecified primary contrast

```
Delta_binding = TV(centers_compact, centers_indexed_row)
              - TV(centers_indexed_row, centers_standard)
```

`Delta_binding > 0` favours binding, `< 0` favours length.

Judged on its **field-cluster bootstrap 95% interval**, not on a permutation p.
A field-blocked permutation p is reported for completeness and explicitly
labelled as not the basis of the verdict, because its null is the absence of any
condition effect rather than `Delta_binding = 0`. This is the same rule the
length control settled on after its p and its interval disagreed.

Secondary: the same contrast with `centers_centered_row` as pivot, where the gaps
are 143 and 315 and so nearly balanced; whether the three distances from
`centers_standard` are monotone in prompt length, in paired form; anchor
reproducibility against the earlier acquisition; rank correlation of the six
distances against absolute character difference and against
positional-versus-labelled mismatch; and the within-condition between-block noise
floor.

## Scope

The moment order does not appear anywhere in this ladder. Every condition carries
the full 24-bin histogram: this is a control on serialization at fixed
information, not on how much of the field is retained.

It is a control on the frozen 48-field replay panel. It does not re-run the
collective experiments and makes no claim about closed-loop outcomes.

## Files

| purpose | path |
|---|---|
| encoders and decoders | `circlemap/serialization_binding_controls.py` |
| offline audit | `analysis/matched_rep_collective/verify_sbc_information_equivalence.py` |
| audit artifact | `analysis/matched_rep_collective/sbc_information_audit.json` |
| protocol | `experiments/stage_c/protocol_matched_rep_collective_sbc_v0_1.json` |
| acquisition runner | `experiments/stage_c/run_sbc_control.py` |
| authorization sidecar | `analysis/matched_rep_collective/sbc_auth_go.json` |
| one-shot Windows pipeline | `run_sbc_all.bat` (operational launcher; not included in this release) |
| analysis | `analysis/matched_rep_collective/analyze_sbc_control.py` |
| decision artifact | `analysis/matched_rep_collective/sbc_primary/decision.json` |

## How to run

```
analysis\matched_rep_collective\run_sbc_all.bat
```

Four steps: offline audit, cost estimate, paid acquisition, inference. It asks
once before spending and stops at the first failure. `run_sbc_all.bat /y` skips
the prompt. Re-running resumes an unfinished acquisition rather than paying twice.

Estimate: 3,072 calls, $1.3668 base, $4.1005 at the retry ceiling, protocol
ceiling $8.

No SI figure step. The length control taught us to design the figure around the
result rather than around the anticipated result, so the figure and the
manuscript text are written after the numbers exist.

## Status

Complete. Acquisition 2026-07-30, 3,072 calls, 100% valid, $1.11. The figure
appears as **Supplementary Figure S20** in the document (the script series number
`figS23` is a file name and differs from the document number).

The contrast designated primary turned out **not to be diagnostic**: its first term
is an index-only change across a small character gap and its second a centre-only
change across a large one, so a prompt-length account and a centre-printing account
predict the same sign for it. It is reported for transparency and not used for
inference. The interpretation rests instead on the crossed 2x2 structure of the
acquired conditions, which is a post hoc reading of the design. See
`docs/LENGTH_CONFOUNDING_MEMO.md`.
