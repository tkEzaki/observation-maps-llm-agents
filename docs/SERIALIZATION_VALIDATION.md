# Serialization controls: design, results and limitations

## 3. Serialization-length control (Supplementary Fig. S19)

`docs/MATCHED_REP_COLLECTIVE_SLC_PROTOCOL.md`
`analysis/matched_rep_collective/slc_primary/decision.json`
Acquisition 2026-07-29, 3,072 calls, 100% valid, $0.93.

### Design

Each of two content classes written in a standard and a compact form. Within a
class the two forms carry the same numbers at the same precision: no filler, no
digit dropped, no derived quantity introduced. Only labelling and layout change.
The two standard forms are byte-identical to the primary encodings, which the
offline audit asserts.

| condition | content | form | chars |
|---|---|---|---|
| `moments_compact` | 3 circular moments | compact | 752 |
| `moments_standard` | 3 circular moments | standard | 789 |
| `centers_compact` | 24 bin masses | compact | 971 |
| `centers_standard` | 24 bin masses | standard | 1429 |

The length axis crosses the content axis: `centers_compact` sits 458 characters
from its own content-class partner but only 182 from `moments_standard`.

Offline audit: information gap between the compact and standard form of a class
exactly 0.0 for both the moment components and the bin masses.

### Results

Block-noise floor 0.105 (95% CI 0.087-0.126). Global mean pairwise distance
0.369, *p* = 0.0002, 3.5 times the floor.

| pair | content | Δchars | mean TV | 95% CI |
|---|---|---|---|---|
| moments std / cmp | same | 37 | 0.074 | 0.040-0.113 |
| centers std / cmp | same | 458 | 0.448 | 0.372-0.525 |
| moments std / centers std | diff | 640 | 0.353 | 0.271-0.440 |
| moments cmp / centers std | diff | 677 | 0.339 | 0.262-0.422 |
| moments std / centers cmp | diff | 182 | 0.497 | 0.424-0.568 |
| moments cmp / centers cmp | diff | 219 | 0.504 | 0.434-0.573 |

**Prespecified primary contrast.**
Δ = TV(centers cmp, moments std) − TV(centers cmp, centers std) = **0.049**,
95% CI **−0.043 to 0.134**, permutation *p* = 0.0026.

Directional toward content, not established. The permutation *p* is **not** read
as support for Δ: its null is exchangeability of the condition labels, that is
the absence of any condition effect, which the global test already rejects for
reasons unrelated to Δ. The interval is the honest statement about Δ. (The
analysis script's verdict logic was corrected to judge on the interval after it
first reported a `PASS` that the interval did not support.)

**Rank correlation.** −0.09 against absolute character difference, +0.41 against
content mismatch.

---

## 4. A reading that was withdrawn, and the objection that caught it

From the two same-content rows above the memo originally read that serialization sensitivity
"depends on the content class": re-serializing the moments moved the operator by
0.074, below the noise floor, while re-serializing the bin masses moved it by
0.448, 4.2 times the floor.

**That reading is confounded and was withdrawn.** The two manipulations differ by
37 and 458 characters. Their ratio is *also* exactly what a prompt-length account
predicts, so the pair cannot be read as evidence about content class at all. The
objection was raised in these terms: the moments forms barely differ in length,
the centers forms differ by a lot, so of course the second distance is the larger
one.

Both rows are now marked `confounded with length` in the figure's verdict strip
and carry a `reference_contrast_caveat` in the decision artifact.

### The replacement: an anchored contrast immune to the confound

Hold one moments prompt fixed as an anchor and compare it with the two histogram
prompts, which carry identical information and differ only in serialization.
**Under any monotone increasing function of prompt length, the histogram prompt
further from the anchor in characters must be at least as distant in response.**
Paired within field, so the same physical field supplies both distances.

| anchor | to `centers_compact` | to `centers_standard` | paired excess (near − far) | exact sign test |
|---|---|---|---|---|
| `moments_standard` (789) | 0.497 at 182 chars | 0.353 at 640 chars | **+0.145** [+0.027, +0.259] | *p* = 0.026, 31 of 46 |
| `moments_compact` (752) | 0.504 at 219 chars | 0.339 at 677 chars | **+0.165** [+0.049, +0.283] | *p* = 0.040, 31 of 47 |

A monotone length account requires both excesses to be at most zero. Both are
positive with intervals excluding zero, in two independent anchors, with
supporting sign tests. **A monotone prompt-length account is refuted.**

The results are therefore **inconsistent with a simple monotone relationship
between the character-count difference and the response distance**, and prompt
length does not work as a dose variable. Two supporting descriptions, neither of
which is load-bearing: `centers_compact` at 971 characters sits *between* the
moments conditions (752, 789) and `centers_standard` (1429) in length yet is the
condition furthest from both moments conditions in response, which no simple or
threshold-like length rule produces; and a best-fit one-dimensional embedding of
the four conditions leaves 35% of the variance unexplained with a latent order
that is not the length order.

**What is not claimed.** This does not exclude every possible function of prompt
length. With finitely many prompts, an unconstrained non-monotone mapping from
length to response distribution could describe almost any arrangement, so no
finite experiment of this kind can rule out length-only accounts in general. The
defensible statement is the one above: prompt length alone does not provide a
simple dose-response explanation of the observed operator differences.

What stayed open: `centers_compact` differs from `centers_standard` in **two**
ways at once, being 458 characters shorter *and* binding each mass to its bin
positionally rather than by an explicit label. Only the length half was excluded.

---

## 5. Serialization-binding ladder (SBC v0.1)

`docs/MATCHED_REP_COLLECTIVE_SBC_PROTOCOL.md`
`analysis/matched_rep_collective/sbc_primary/decision.json`
Acquisition 2026-07-30, 3,072 calls, 100% valid, $1.11.

### Design

Four serializations of **the same 24 bin masses at the same precision**, so
retained information is identical throughout (audited mass gap exactly 0.0). Both
anchors re-acquired inside this run, so every comparison is within one
acquisition.

| condition | binding | chars |
|---|---|---|
| `centers_compact` | positional, grid stated by rule | 971 |
| `centers_indexed_row` | explicit bin index per mass, `b00=0.062500,...` | 1055 |
| `centers_centered_row` | explicit bin centre per mass, `-3.010693:0.062500,...` | 1114 |
| `centers_standard` | explicit index and centre, one line per bin | 1429 |

The pivot `centers_indexed_row` sits 84 characters from the positional condition
and 374 from the fully labelled one, a ratio of 4.4 to 1. Prespecified contrast:
Δ_binding = TV(compact, indexed) − TV(indexed, standard); length predicts
negative, binding predicts positive.

### Results

Block-noise floor 0.139 (95% CI 0.118-0.160), higher than the previous run's
0.105 and heterogeneous by condition (compact 0.180, indexed 0.122, centred
0.107, standard 0.146). Global mean pairwise distance 0.395, *p* = 0.0002.

| pair | Δchars | mean TV |
|---|---|---|
| indexed / centred | 59 | 0.448 |
| compact / indexed | 84 | 0.290 |
| compact / centred | 143 | 0.417 |
| centred / standard | 315 | 0.312 |
| indexed / standard | 374 | 0.440 |
| compact / standard | 458 | 0.460 |

Δ_binding = **−0.150**, 95% CI **−0.259 to −0.039**, excludes zero. The pivot
groups with the positional condition it is 84 characters from.

---

## 6. The prespecified contrast turned out not to be diagnostic

Δ_binding cannot separate the two accounts, and this is visible from its own
construction rather than from the data. Its first term, compact vs indexed, is a
change of the **index** feature only across an 84-character gap; its second term,
indexed vs standard, is a change of the **centre** feature only across a
374-character gap. A length account and a centre-printing account therefore
predict the **same** sign for Δ_binding. Its value is consistent with both.

The error was in the operationalisation: I called explicit index labelling
"binding", and index labelling turns out to be the weak feature. The decision
artifact now computes a `diagnosticity` flag for each pivot contrast, and the
verdict logic refuses to read a non-diagnostic contrast as settling anything.

The **secondary** contrast was the diagnostic one. Δ_centred = TV(compact,
centred) − TV(centred, standard) pits a centre-only change across 143 characters
against an index-only change across 315: length predicts negative, centre
printing predicts positive. Observed **+0.104**, 95% CI **−0.017 to +0.223**.
Right sign, not resolved on its own.

---

## 7. What the design actually answers: a crossed 2x2

The four conditions cross two binary features.

|  | index not printed | index printed |
|---|---|---|
| **centre not printed** | `compact` | `indexed_row` |
| **centre printed** | `centered_row` | `standard` |

Decomposing the six distances by which feature differs:

| what differs | pairs (Δchars → TV) | cell mean | 95% CI | × noise |
|---|---|---|---|---|
| index only | 84 → 0.290, 315 → 0.312 | **0.301** | 0.251-0.357 | 2.2 |
| centre only | 143 → 0.417, 374 → 0.440 | **0.428** | 0.372-0.487 | 3.1 |
| both | 59 → 0.448, 458 → 0.460 | **0.454** | 0.396-0.510 | 3.3 |

Centre feature minus index feature = **+0.127**, 95% CI **+0.047 to +0.214**,
exact sign test *p* = 0.011.

**The decisive observation is within-cell length insensitivity.** Each cell holds
the feature pattern fixed while its two pairs differ greatly in character gap.

| cell | character gap | ratio | distance difference | |
|---|---|---|---|---|
| index only | 84 → 315 | 3.8× | +0.022 [−0.074, +0.125] | covers zero |
| centre only | 143 → 374 | 2.6× | +0.023 [−0.086, +0.130] | covers zero |
| both | 59 → 458 | **7.8×** | +0.012 [−0.083, +0.112] | covers zero |

The "both" cell carries it: a 7.8-fold difference in length gap produces a 0.012
difference in response distance, and the interval covers zero.

**Within this tested 2x2 ladder, the pair category was more informative about
response distance than the magnitude of the character-count difference.** Response differences tracked the type of textual binding
manipulation more closely than the size of the character-count gap. This is a
statement about the two features this ladder crosses, not a general claim about
serialization.

Note also that no serialization change is inert: even changing only the index
label moves the operator by 0.301, 2.2 times the block noise.

---

## 8. Where the answer stands

**Established.**

- The operator effect survives with retained information held exactly fixed, in
  three independent designs: centers vs intervals in the replay (0.290, 3.2× the
  floor), the compact-versus-standard re-serializations (up to 0.448), and every
  cell of the binding ladder (0.301 to 0.454, 2.2× to 3.3× the floor).
- A **monotone prompt-length account is refuted**, by two independent anchored
  paired contrasts with intervals excluding zero and supporting sign tests.
- **Within the tested ladder, the pair category was more informative about
  response distance than the magnitude of the character-count difference**, from
  the within-cell length insensitivity, with the strongest case at a 7.8-fold gap
  ratio. Response differences tracked the type of
  textual binding manipulation more closely than the size of the character-count
  gap.
- Of the two features tested, **printing the numeric bin centre next to each mass
  dominates printing the bin index**, +0.127 [+0.047, +0.214].
- **Moment-only layout changes produced substantially smaller and less
  consistently resolved effects than the histogram manipulations.** Fig. 5 gave
  0.145, 2.1× the floor with an interval overlapping it; the length control gave
  0.074 across a different layout change, below the floor. Note this is *smaller
  and less consistently resolved*, not absent: the 0.145 interval does not exclude
  the floor but the point estimate is above it. This finding is supported by two acquisitions.

**Not established.**

- The prespecified Δ of the length control: 0.049 [−0.043, 0.134], directional
  toward content, unresolved.
- The prespecified Δ_binding of the ladder: not diagnostic by construction, so it
  settles nothing regardless of its value.
- Δ_centred, the diagnostic contrast: +0.104 [−0.017, +0.223], right sign, not
  resolved on its own. The within-cell evidence is what carries the conclusion.

**Explicitly not claimed.**

- That prompt length can never affect language-model responses. No finite set of
  prompts can exclude an unconstrained length-only mapping; what is excluded is
  length as a simple dose variable.
- That the ladder identifies a universal serialization mechanism. It crosses two
  features and separates those two.
- That serialization alone changes the macroscopic outcome. Both controls are on
  the frozen replay panel.

**Open.**

- Whether printing the phase value is the *general* feature or a specific case of
  something broader. The 2x2 tests two features; there are others (precision,
  ordering, delimiter choice, whether the axis is monotone in the printed order).
- Why the positional condition is also the noisiest (block noise 0.180 against
  0.107 for the centred row). Positional binding may destabilise responses as
  well as shift them, which the design was not built to test.
- Whether any of this transports to the closed loop. Both controls are on the
  frozen 48-field replay panel; neither re-runs the collective experiments, and
  neither licenses a claim about collective outcomes.

**Caveats to carry.**

- Run-to-run drift. The two re-acquired anchors, one day apart, gave between-run
  distances of 0.165 and 0.188 against a within-run block floor of 0.139, ratios
  1.19 and 1.35. Drift is comparable to, and if anything slightly above,
  within-run noise. Every primary comparison here is within a single acquisition,
  so nothing above is affected, but this caps how finely any cross-run comparison
  can be read. The analysis script's `agrees_within_block_noise` boolean uses a
  generous criterion (the intervals touch); read the numbers, not the flag.
- The 2x2 reading of the ladder is a **post hoc** reframing. It is analysis of a
  crossed design rather than a search over candidate features, since both main
  effects are carried equally by the design, but the fact remains that the
  contrast designated primary was built on the wrong feature.
- The anchored monotonicity contrast in the length control was also added after
  acquisition, and is labelled as such in the figure caption.
- Noise floors differ between the two acquisitions (0.105 and 0.139), so ratios
  to the floor are comparable within a run and only roughly so across runs.

---

