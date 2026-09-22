# Stage B representation feature sweep

Date: 2026-07-23

## Executive conclusion

Precision and bin count produced large, reproducible changes in the first
harmonic, but the frozen high-confidence adjacent-jump and sign-transition
gate did not pass.

This distinction is important. The two blocks strongly agree on broad feature
trajectories, and 10 of 12 feature-by-concentration series span at least 0.25
in both blocks. However, with six responses per offset, no adjacent pair
simultaneously reached the preregistered 0.95 jump probability in both blocks.
The data therefore support systematic feature dependence, but not a claim of
a discontinuous representation-induced bifurcation.

No automatic boundary top-up or collective Stage C run is authorized by this
result.

## Frozen design

Fifteen new representation conditions were measured.

Precision axis:

- 24 interval bins;
- 2, 3, 4, 5, or 6 decimals;
- identical prompt header at every precision.

Bin-count axis:

- 12, 18, 24, 30, 36, or 48 interval bins;
- six decimals.

Origin-shift axis:

- 24 center bins;
- shifts of 0, 0.125, 0.25, 0.375, or 0.5 bin widths.

The 24-bin, six-decimal interval condition was shared by the precision and
bin-count axes. Each block used:

- \(\kappa=2,6,9,12\);
- 36 mirror-paired stratified-jitter offsets;
- six responses per cell;
- 12,960 API calls.

Uncertainty used a multinomial bootstrap with a frozen 0.5 pseudocount because
six observations per offset otherwise create zero-probability action cells.
Primary Fourier order was \(M=6\), with \(M=2,4,8,12\) sensitivity analyses.

## Execution

Block 1:

- 12,960 / 12,960 valid;
- actual cost: $6.7170105;
- API execution time: 467.18 s;
- effective parallelism: 19.82 / 20;
- throughput: 27.74 calls/s.

Block 2:

- 12,960 / 12,960 valid;
- actual cost: $6.7155165;
- API execution time: 463.06 s;
- effective parallelism: 19.53 / 20;
- throughput: 27.99 calls/s.

Combined:

- 25,920 / 25,920 valid;
- actual cost: $13.432527;
- measured API execution time: 930.24 s.

## Precision trajectory

Across-block mean \(a_1\) for precisions 2 through 6 was:

- \(\kappa=2\): -0.190, 0.264, 0.758, 0.602, 0.535;
- \(\kappa=6\): -0.183, -0.214, 0.286, 0.131, 0.197;
- \(\kappa=9\): -0.138, -0.291, 0.124, 0.081, 0.170;
- \(\kappa=12\): -0.153, -0.255, 0.115, -0.129, 0.114.

Block-to-block \(a_1\) trajectory correlations were 0.952, 0.986, 0.953,
and 0.921, respectively. All four series had a point-estimate span above 0.25
in both blocks.

The response is not a smooth monotone function of decimal precision. In
particular, the transition from three to four decimals changes the sign and
magnitude of \(a_1\) across all four concentrations. It did not pass the
two-block 0.95 adjacent-jump probability gate because of the deliberately
small per-offset sample size.

Precision also changed the fitted torque component. At \(\kappa=12\), mean
\(a_0\) was 0.715, 0.666, -0.086, -0.229, and -0.178 from two through six
decimals. Thus numeric precision reorganized both coupling symmetry and mean
action bias even though prompt wording was held fixed.

## Bin-count trajectory

At \(\kappa=12\), across-block mean \(a_1\) for 12, 18, 24, 30, 36, and 48
bins was:

\[
-0.228,\ -0.166,\ 0.114,\ 0.605,\ 0.760,\ 0.861.
\]

The other concentrations also showed large ranges:

- \(\kappa=2\): 0.409, 0.107, 0.535, 0.592, 0.811, 1.092;
- \(\kappa=6\): 0.087, -0.168, 0.197, 0.411, 0.721, 0.910;
- \(\kappa=9\): -0.170, -0.093, 0.170, 0.352, 0.533, 0.926.

The \(a_1\) trajectory correlations were 0.956, 0.986, 0.997, and 0.981.
Every concentration had a span greater than 0.96 in both blocks.

The strongest unresolved boundary is between 24 and 30 bins at narrow
concentrations. The point estimates support a coarse-to-fine transition from
suppressed or negative coupling to strong polar alignment, but the frozen
adjacent-pair confidence gate was not reached in both blocks.

## Origin-shift trajectory

Origin shift was less stable than precision or bin count.

At \(\kappa=12\), mean \(a_1\) over shifts 0 through 0.5 was:

\[
-0.302,\ -0.190,\ -0.348,\ 0.024,\ -0.502.
\]

The two blocks reproduced the near-zero response at shift 0.375 and negative
response at shift 0.5. The 0.375-to-0.5 jump probability was 0.845 in Block 1
and 0.985 in Block 2, so it failed the requirement that both exceed 0.95.

The \(a_1\) trajectory correlation was 0.952 at \(\kappa=2\), 0.590 at
\(\kappa=6\), 0.648 at \(\kappa=9\), and 0.840 at \(\kappa=12\). Origin effects
at intermediate concentration should therefore be treated as unresolved.

## Statistical decision

Preregistered outcomes:

- replicated operational adjacent jumps: 0;
- replicated sign-transition brackets: 0;
- overall feature gate: fail.

Secondary diagnostics:

- 10 of 12 feature-by-concentration trajectories spanned at least 0.25 in both
  blocks;
- maximum pointwise overdispersion ratio: 0.877;
- maximum Fourier-order sensitivity: 0.084 for \(a_1\), 0.058 for \(a_2\).

The negative gate result is a power-qualified finding, not evidence of feature
invariance. The broad precision and bin-count trajectories are reproducible,
but six samples per offset are insufficient for the deliberately strict local
transition criterion.

## Research decision

The present data justify these claims:

- numeric precision can reorganize both the first-harmonic channel and mean
  action bias while prompt wording is fixed;
- bin count is the strongest feature axis: it converts \(C_1\) in amplitude and
  phase (coarse negative-odd → fine strong positive polar);
- representation features also alter the offset-averaged action bias \(a_0\);
- precision and bin-count effects replicate at the trajectory level.

They do not justify:

- a mathematical bifurcation or discontinuity;
- a unique critical precision, bin count, or origin shift;
- origin shift as a primary axis (secondary; exceeds pure coordinate-shift null
  but is less stable than bin count);
- collective-phase predictions.

Complex-kernel reanalysis and origin-covariance residuals are in
`docs/STAGE_B_COMPLEX_KERNEL.md`.

Under the frozen stopping rule, no automatic adaptive top-up is run. If a
separate follow-up is approved, prioritize precision 3–4 and bin count 24–30 at
\(\kappa=9,12\) over origin shift, and only pursue high-\(n\) discontinuity
tests if that claim itself becomes the target.

## Reproducibility artifacts

- Block 1 protocol:
  `experiments/stage_b_response_law/protocol_feature_sweep_v0_1_block_1.json`
- Block 2 protocol:
  `experiments/stage_b_response_law/protocol_feature_sweep_v0_1_block_2.json`
- Block 1 run:
  `runs/feature_sweep/feature-sweep-v0.1-block1_89f0bcad5ae2/20260723T120058Z`
- Block 2 run:
  `runs/feature_sweep/feature-sweep-v0.1-block2_eb1adc1ccf75/20260723T120913Z`
- Analysis:
  `analysis/feature_sweep_real/feature_sweep_analysis.json`
- Endpoint export:
  `analysis/feature_sweep_real/feature_sweep_endpoints.csv`
- Analysis code:
  `analysis/analyze_feature_sweep.py`
- Complex-kernel reanalysis:
  `analysis/complex_kernel_feature_sweep/` and
  `docs/STAGE_B_COMPLEX_KERNEL.md`
