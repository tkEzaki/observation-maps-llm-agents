# Stage B representation-invariance screen

Date: 2026-07-23

## Executive conclusion

The concentration-induced harmonic switch measured with the frozen 24-bin,
six-decimal interval prompt is **not invariant to equivalent encodings of the
same relative-phase field**.

The replicated baseline passed the preregistered switching rule with bootstrap
probability 1.000. None of seven alternate representations reached the 0.95
probability gate. This is not a null result caused by invalid responses or low
execution throughput: all 20,160 new responses were valid, and the principal
differences are much larger than their bootstrap uncertainty.

Stage C is therefore blocked under observation protocol v0.1. The current
evidence does not support describing the measured law as a property of the
underlying continuous relative-phase distribution alone. The serialized
language representation is part of the interaction operator.

## Protocol

The same continuous von Mises fields were presented at broad
\(\kappa=2\) and narrow \(\kappa=12\) concentration using:

1. 24 bin-center values;
2. half-bin-shifted 24 bin-center values;
3. 12 interval bins;
4. 48 interval bins;
5. 24 interval bins rounded to four decimals;
6. 24-bin integer counts with fixed total 100;
7. circular moments \(m=1,2,3\).

Each representation used 40 independent responses at every offset and
concentration. The first screen sampled 12 offsets. It was then completed to a
uniform 36-offset grid after an aliasing diagnostic described below.

The primary switching rule was fixed before the API run:

- broad \(a_1 \ge 0.2\);
- \(|a_1|\le 0.2\) under narrow concentration;
- broad \(a_2>0\);
- narrow \(a_2-\) broad \(a_2\ge0.1\);
- bootstrap probability of satisfying all four conditions at least 0.95.

The replicated 24-bin, six-decimal baseline used its original 48-offset grid
and 100 pooled responses per condition.

## Execution and cost

Initial screen:

- 6,720 / 6,720 valid;
- actual cost: $3.2212755;
- API execution time: 247.44 s;
- configured concurrency: 20;
- effective parallelism: 19.70;
- throughput: 27.16 calls/s.

Aliasing top-up:

- 13,440 / 13,440 valid;
- actual cost: $6.480621;
- API execution time: 496.62 s;
- configured concurrency: 20;
- effective parallelism: 19.82;
- throughput: 27.06 calls/s.

Combined:

- 20,160 / 20,160 valid;
- actual cost: $9.7018965;
- measured API execution time: 744.06 s.

## Aliasing correction

The planned 12-offset screen was insufficient for the primary Fourier
endpoints because the original response curve contains a large \(m=23\)
component. On a 12-point grid, \(m=23\) aliases to \(m=-1\).

This was visible directly in the replicated baseline:

- full 48-point narrow \(a_1=-0.0385\);
- 12-point baseline subset narrow \(a_1=+0.3195\).

No scientific decision was made from the aliased analysis. The seven
representation curves were completed to 36 offsets, where the known \(m=23\)
component maps to \(m=-13\), not to either primary endpoint. The 36-point
screen resolves the known alias, although it is not a general proof of
continuous-grid convergence for arbitrary higher harmonics.

## Primary results

Replicated baseline, 24 intervals at six decimals:

- broad: \(a_1=0.571\), \(a_2=0.114\);
- narrow: \(a_1=-0.039\), \(a_2=0.332\);
- switching-rule probability: 1.000; pass.

Alternate representations:

- Standard bin centers: \(a_1: 0.681\to-0.242\),
  \(a_2: 0.154\to-0.094\); probability 0.000.
- Half-shifted centers: \(a_1: 0.608\to-0.318\),
  \(a_2: 0.022\to0.283\); probability 0.000.
- 12 intervals: \(a_1: 0.352\to-0.175\),
  \(a_2: 0.410\to0.475\); probability 0.137.
- 48 intervals: \(a_1: 1.038\to0.902\),
  \(a_2: 0.198\to0.325\); probability 0.000.
- Four-decimal intervals: \(a_1: 0.800\to0.259\),
  \(a_2: 0.183\to0.228\); probability 0.002.
- Integer counts: \(a_1: 0.378\to-0.383\),
  \(a_2: 0.266\to0.308\); probability 0.000.
- Circular moments: \(a_1: 1.027\to1.048\),
  \(a_2: 0.364\to0.338\); probability 0.000.

The closest alternate encoding was the 12-bin representation. It retained a
small-magnitude narrow \(a_1\), but \(a_2\) was already dominant in the broad
field and its concentration increment was only 0.065, below the fixed 0.1
criterion.

## Interpretation

The failure is structured rather than a uniform loss of responsiveness.

- The 48-bin and moment representations retained a strong positive first
  harmonic at both concentrations.
- Center, count, and 12-bin encodings often changed the narrow first harmonic
  into a negative, anti-aligning component instead of suppressing it near zero.
- Changing only numeric precision from six to four decimals removed the
  baseline switching pattern.
- Shifting the bin origin changed both first- and second-harmonic behavior.
- Alternate encodings introduced large mean-action biases. For example,
  standard centers produced \(a_0=-0.329\) broad and \(-0.343\) narrow,
  whereas the baseline values were \(0.018\) and \(0.012\).

These effects show that semantically equivalent phase summaries are not
interchangeable interventions for this model. Tokenization, table geometry,
numeric precision, and the explanatory language surrounding the numbers can
change both reciprocal and non-reciprocal components of the effective law.

## Research decision

Do not proceed to a two-body or collective \(K\)-sweep using observation
protocol v0.1.

Two defensible paths remain:

1. Treat the textual representation as part of the physical operator and make
   representation-induced coupling laws the central phenomenon. This requires
   independent replication and multiple models before collective claims.
2. Redesign the observation/task contract to seek a law stable across a
   predefined equivalence class, then repeat Stage B before Stage C.

The current data do not justify calling the six-decimal harmonic switch an
intrinsic response to the continuous phase field.

## Reproducibility artifacts

- Protocol:
  `experiments/stage_b_response_law/protocol_representation_v0_1.json`
- Top-up protocol:
  `experiments/stage_b_response_law/protocol_representation_topup_v0_1.json`
- Initial run:
  `runs/response_law_representation/representation-v0.1_b339d0b23c2c/20260723T105325Z`
- Top-up run:
  `runs/response_law_representation/representation-topup-v0.1_827c00e5594f/20260723T110118Z`
- Machine-readable analysis:
  `analysis/representation_grid_real/representation_grid_analysis.json`
- Endpoint export:
  `analysis/representation_grid_real/representation_grid_endpoints.csv`
- Analysis code:
  `analysis/analyze_representation_grid.py`
