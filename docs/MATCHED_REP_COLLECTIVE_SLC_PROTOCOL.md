# Serialization-length control (SLC v0.1)

## Why this exists

The same-task-information control (`omap_control_v0_1`, main Fig. 5) varies
layout and neutral padding at fixed moment values. Its padded condition moved
the operator a lot, but that condition changes context volume, the position of
the relevant numbers and prompt length together, so it does not isolate length.
A remaining question is whether the primary encoding effect is a
prompt-length effect in disguise.

This control answers that question with no filler text anywhere.

## Design

On the frozen 48-field replay panel, each of two content classes is written in
a standard and a compact form:

| condition | content class | form | chars | tokens |
|---|---|---|---|---|
| `moments_compact` | 3 circular moments | compact | 752 | 317 |
| `moments_standard` | 3 circular moments | standard | 789 | 332 |
| `centers_compact` | 24 bin masses | compact | 971 | 405 |
| `centers_standard` | 24 bin masses | standard | 1429 | 588 |

Within a content class the two forms carry the same numbers at the same
precision: nothing is added, dropped, rounded or derived. Only labelling and
layout change. The two standard forms are byte-identical to the primary
`moments_m1_m3` and `centers_24_standard` prompts of the main design, which the
offline audit asserts.

The compact histogram form omits the list of bin centres and states the fixed
bin grid exactly in its header instead. The grid is identical in every field,
so it carries no field information either way.

**The length axis crosses the content axis.** `centers_compact` sits 458
characters from its own content-class partner but only 182 characters from
`moments_standard`. A prompt-length account predicts responses group by
position on the length axis; a content account predicts they group by content
class. One acquisition separates them.

## Prespecified primary contrast

```
Delta = TV(centers_compact, moments_standard) - TV(centers_compact, centers_standard)
```

Field-blocked permutation of the condition labels within each field, 5,000
resamples, one-sided, alpha 0.05, plus-one estimator. `Delta > 0` favours
retained information over prompt length. Field-cluster bootstrap gives the
interval.

Secondary: the layout-only reference within the moments class, the
information-only reference within the histogram class, the rank correlation of
the six pairwise distances with content mismatch versus with absolute length
difference, and a global any-difference test.

## Scope, and why the moment order is fixed at three

The moment order is held at three in every condition and is **not** a length
knob.

Binned circular moments are a discrete Fourier transform of the 24 bin masses.
Harmonics 1 to 12 therefore invert the histogram exactly: the audit measures a
maximum reconstruction error of 1.4e-15 on this field panel. A twelfth-order
moments condition would be the centers condition in another notation, not a
longer summary of the same field, and the centers condition is already an arm of
the primary design. Raising the order changes the retained information, which is
exactly what this control holds fixed.

This control is a serialization manipulation on the frozen replay panel. It does
not re-run the collective experiments and makes no claim about closed-loop
outcomes.

## Files

| purpose | path |
|---|---|
| encoders and decoders | `circlemap/serialization_length_controls.py` |
| offline audit | `analysis/matched_rep_collective/verify_slc_information_equivalence.py` |
| audit artifact | `analysis/matched_rep_collective/slc_information_audit.json` |
| protocol | `experiments/stage_c/protocol_matched_rep_collective_slc_v0_1.json` |
| acquisition runner | `experiments/stage_c/run_slc_control.py` |
| authorization sidecar | `analysis/matched_rep_collective/slc_auth_go.json` |
| analysis | `analysis/matched_rep_collective/analyze_slc_control.py` |
| decision artifact | `analysis/matched_rep_collective/slc_primary/decision.json` |
| figure | `analysis/figures_si/figS22.py` -> `figures/si/figS22_serialization_length.pdf` |

## Status

Complete. Acquisition 2026-07-29, 3,072 calls, 100% valid, $0.93. The figure appears as **Supplementary
Figure S19** in the document (the script series number `figS22` is a file name and
differs from the document number).

The prespecified contrast came out inconclusive: 0.049 with a bootstrap interval of
-0.043 to 0.134. The interpretation rests on the anchored paired contrasts added
after acquisition, which are labelled post hoc and diagnostic in the caption. See
`docs/SERIALIZATION_VALIDATION.md` for the statistical interpretation.
