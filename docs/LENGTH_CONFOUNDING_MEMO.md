# Is the observation-map effect a prompt-length effect?

Working memo on the length-confounding question: what was asked, what was
measured, which two readings had to be withdrawn and why, and where the answer
now stands.

Project: `pilot4_kuramoto`. All figures are Supplementary Figures of the NMI
submission. Every number below is read from a frozen artifact; the artifact path
is given for each block so any figure can be regenerated from it.

---

## 1. The question

The paper's central claim is that the observation map, the code that turns a
physical state into the text an agent reads, is part of the effective policy: the
same relative-phase field elicits different action distributions under different
encodings, and in the closed loop that difference selects a different collective
outcome.

The three primary encodings differ in prompt length as well as in content:

| encoding | chars | est. tokens | lines |
|---|---|---|---|
| `moments_m1_m3` | 789 | 332 | 16 |
| `centers_24_standard` | 1429 | 588 | 34 |
| `intervals_24_decimal6` | 1562 | 641 | 34 |

So the obvious deflationary reading is available: perhaps nothing about the
*mapping* matters and the model simply responds differently to longer prompts.
Closing that off is what this line of work is for.

---

## 2. What was already in hand, and why it was not enough

**Identical-field replay (main Fig. 3).**
`analysis/matched_rep_collective/replay_primary/decision.json`

48 endogenous fields, each re-encoded under all three encodings and presented to
GPT for 32 responses. Mean pairwise total-variation distance 0.344
(*p* = 0.0002, 5,000 within-field permutations), 3.76 times the within-encoding
between-block noise of 0.092. Pairwise:

| pair | mean TV | 95% CI |
|---|---|---|
| moments vs centers | 0.335 | 0.251-0.423 |
| moments vs intervals | 0.408 | 0.323-0.490 |
| **centers vs intervals** | **0.290** | 0.221-0.363 |

That third row is the first piece of leverage. Centers and intervals carry the
*same* 24 bin masses in the *same* 24 rows and differ only in how each bin is
labelled, and they still separate by 0.290, 3.2 times the block noise. So
serialization moves the operator with the retained information held exactly
fixed. But their lengths differ too (1429 vs 1562), so this alone does not
isolate length.

**Same-task-information control (main Fig. 5).**
`analysis/matched_rep_collective/omap_primary/decision.json`

Three GPT inputs carrying identical circular-moment values: the original moments
text, the same numbers as a table, and a neutral-padding version. Global mean
pairwise distance 0.311 (*p* = 0.0002) against a within-version block distance of
0.069.

| manipulation | mean TV | 95% CI |
|---|---|---|
| re-laid out as a table | 0.145 | 0.079-0.219 |
| neutral context added | 0.414 | 0.311-0.517 |
| table vs padded | 0.374 | 0.273-0.475 |

Two readings follow, and only one of them is safe. Re-laying out identical
numbers is at most a weak effect: 2.1 times the block noise with an interval that
overlaps the noise floor. Adding neutral context is large, exceeding even the
0.344 between genuinely different encodings. But the padded condition changed
context volume, the position of the task-relevant numbers, and length **all at
once**, so it is not a prompt-length mechanism and cannot be used as one. The
sensitivity reported in the paper rests on that condition, so the length question
was still open.

---

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
response distance than the magnitude of the character-count difference.** Put for
the manuscript: response differences tracked the type of textual binding
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
  ratio. Stated for the manuscript: response differences tracked the type of
  textual binding manipulation more closely than the size of the character-count
  gap.
- Of the two features tested, **printing the numeric bin centre next to each mass
  dominates printing the bin index**, +0.127 [+0.047, +0.214].
- **Moment-only layout changes produced substantially smaller and less
  consistently resolved effects than the histogram manipulations.** Fig. 5 gave
  0.145, 2.1× the floor with an interval overlapping it; the length control gave
  0.074 across a different layout change, below the floor. Note this is *smaller
  and less consistently resolved*, not absent: the 0.145 interval does not exclude
  the floor but the point estimate is above it. This is the reading the paper
  wanted preserved and it now rests on two acquisitions.

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

## 9. Prepared answer to the reviewer question

> **Is this simply a prompt-length effect?**

We tested this directly. First, we fixed a moments prompt and compared it with two
histogram prompts that retained identical bin masses but differed in serialization
and length. Under a monotone prompt-length account, the histogram prompt further
away in character count should be at least as distant in response. The opposite
occurred for two independent anchors: the closer prompt was more distant by 0.145
[0.027, 0.259] and 0.165 [0.049, 0.283], with paired exact sign tests *p* = 0.026
and *p* = 0.040 (Supplementary Fig. S19e).

Second, a four-condition serialization ladder crossed whether bin indices and
numerical bin centres were printed. Within each manipulation class, character-count
gaps varied by factors of 2.6 to 7.8 while response distances changed by only 0.012
to 0.023 and the corresponding intervals included zero (Supplementary Fig. S20f).
The operator differences therefore do not follow a simple monotone dose-response
relationship with prompt length; they depend instead on which textual binding
features are changed.

We do not claim that prompt length can never affect language-model responses, or
that the ladder identifies a universal serialization mechanism. The controls were
performed on the frozen replay panel and were not re-run in the collective loop.

### The position the manuscript fixes on

> The microscopic encoding effect cannot be reduced to a simple monotone
> prompt-length difference: it persists with retained information fixed, and
> response distance tracks the type of serialization change more closely than the
> size of the character-count gap.

> These replay controls do not identify a universal textual mechanism and were not
> tested for macroscopic transport.

### Claim matrix

| claim | status | placement |
|---|---|---|
| encoding changes the operator on the same field | established | main text |
| serialization matters with retained information fixed | established | main Fig. 5 and SI |
| a simple monotone prompt-length account | refuted | SI, one or two sentences in main text |
| any length-only mechanism whatsoever | not refuted | not claimed |
| the type of textual feature matters | supported within the tested ladder | SI |
| centre printing exceeds index printing | secondary, post hoc | SI, limited mention in Discussion |
| serialization alone changes the macroscopic outcome | untested | not claimed |
| the neutral-padding condition is a length-only test | no | explicitly denied |
| closed-loop transport of these controls | untested | limitation |

### Whether a further experiment is needed before submission

No. The deflationary reading a reviewer will actually raise is answered by
multiple independent controls with retained information fixed, and by paired
evidence running against a monotone length account. Going further would move the
paper's centre of gravity onto the details of serialization mechanism. A further
experiment is worth running only if the independent claim *numerical coordinate
binding is the mechanism* is to be made, and that would need a prespecified
factorial design with token length matched as closely as possible.

## 10. Methodological lessons worth keeping

1. **A contrast that two competing accounts predict the same sign for is not
   evidence, whatever its interval.** Check diagnosticity at design time, not
   after. The analysis script now computes the flag and refuses to read a
   non-diagnostic contrast as decisive.
2. **A permutation *p* whose null is "no effect at all" is not a statement about a
   derived contrast.** Judge derived contrasts on their bootstrap intervals. The
   first version of the length-control verdict reported a `PASS` that its own
   interval did not support.
3. **Comparing two manipulations of unequal magnitude tells you about the
   magnitudes, not the kinds.** The 0.074-versus-0.448 reading failed for exactly
   this reason. An anchored paired contrast, where one side is held fixed and the
   two comparisons differ only in the factor of interest, does not have this
   failure mode.
4. **Design the figure after the result.** The first figure for the length control
   was laid out around the anticipated finding and had to be rebuilt, both in its
   panels and in its caption, once the numbers arrived. The ladder's pipeline
   deliberately has no figure step.
5. **Hash text artifacts with newlines normalized.** Python's text writer
   translates `\n` to `\r\n` on Windows, so a "frozen" JSON artifact's digest was
   platform-dependent and the integrity check failed when the artifact and the
   decision were produced on different machines.

---

## 11. Artifact index

| what | path |
|---|---|
| Replay target effect (Fig. 3) | `analysis/matched_rep_collective/replay_primary/decision.json` |
| Same-task-information control (Fig. 5) | `analysis/matched_rep_collective/omap_primary/decision.json` |
| Length control protocol | `experiments/stage_c/protocol_matched_rep_collective_slc_v0_1.json` |
| Length control audit | `analysis/matched_rep_collective/slc_information_audit.json` |
| Length control decision | `analysis/matched_rep_collective/slc_primary/decision.json` |
| Length control figure (document S19) | `analysis/figures_si/figS22.py` → `figures/si/figS22_serialization_length.pdf` |
| Binding ladder figure (document S20) | `analysis/figures_si/figS23.py` → `figures/si/figS23_serialization_binding.pdf` |

The script series (`figS22`, `figS23`) and the document numbering (S19, S20) differ; the script numbers are file names and the document numbers come from the position of each figure in the Supplementary Information.
| Binding ladder protocol | `experiments/stage_c/protocol_matched_rep_collective_sbc_v0_1.json` |
| Binding ladder audit | `analysis/matched_rep_collective/sbc_information_audit.json` |
| Binding ladder decision | `analysis/matched_rep_collective/sbc_primary/decision.json` |
| Encoders, length control | `circlemap/serialization_length_controls.py` |
| Encoders, binding ladder | `circlemap/serialization_binding_controls.py` |
| Protocol documents | `docs/MATCHED_REP_COLLECTIVE_SLC_PROTOCOL.md`, `docs/MATCHED_REP_COLLECTIVE_SBC_PROTOCOL.md` |

Acquisitions: `runs/matched_rep_collective_slc/matched-rep-slc-v0.1_gpt-5.4-mini/20260729T215934Z`
and `runs/matched_rep_collective_sbc/matched-rep-sbc-v0.1_gpt-5.4-mini/20260730T000855Z`.
Both 3,072 calls at 100% valid, $0.93 and $1.11, GPT `gpt-5.4-mini`, temperature
0.7, on the frozen 48-field replay panel.

## 12. State of the write-up

Both pages are in the Supplementary Information and both are wired into the main
text.

- **Supplementary Fig. S19**, serialization-length control, eight panels. Title
  and caption recentred on the single message that response distance does not
  follow a monotone function of the character-count difference. The prespecified
  contrast is labelled inconclusive; the anchored contrasts are labelled post hoc
  and diagnostic; the two single-class references are labelled confounded with
  length; the rank-correlation panel is labelled descriptive.
- **Supplementary Fig. S20**, serialization-binding ladder, seven panels plus a
  verdict strip. Centred on the crossed 2x2 reading. The contrast designated
  primary is marked non-diagnostic and is reported for transparency only.
- **Main text.** Two sentences in the Fig. 5 Results subsection and an extension of
  the same-task-information Discussion paragraph, both hedged as above. Nothing
  added to the Abstract. Two Methods subsections carry the detail, which is outside
  the word cap. Main text 3,499 words against the 3,500 limit.
- **Supplementary Note 1** now opens with a paragraph recording that the
  prespecified contrast of S19 was inconclusive and the designated pivot of S20
  non-diagnostic, and that the interpretation rests on diagnostic paired contrasts
  and the crossed structure of the acquired conditions, labelled post hoc.

Nothing further is planned for this line of work before submission.
