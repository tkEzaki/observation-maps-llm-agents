# Stage B representation-induced transmutation map

Date: 2026-07-23

## Executive conclusion

**Representation-induced microscopic response-law transmutation** passed its
preregistered two-block replication gate.

The same unimodal relative-phase field generated three reproducible
concentration trajectories:

1. the 24-bin, six-decimal interval encoding suppressed its first odd harmonic
   and developed a dominant positive second harmonic;
2. the circular-moment encoding preserved a strong positive polar channel
   (no-stay directional controller with low output entropy; see complex-kernel);
3. the 24-bin center encoding changed from a positive to a negative first odd
   harmonic, with a material even component \(b_1\) at several \(\kappa\).

All three phenotype probabilities exceeded 0.95 in both independent seed
blocks. This supports a structured representation-by-concentration interaction,
not a representation-independent response law.

Cosine coefficients \(b_m\) are not negligible. Primary reporting should use
complex harmonics \(C_m=a_m+ib_m\) or \((R_m,\phi_m)\); see
`docs/STAGE_B_COMPLEX_KERNEL.md`. The fitted mean-response derivative
\(g'(0)\) is demoted there as a Fourier-order unstable diagnostic (not an
aligned-state stability criterion).

The result remains microscopic. It does not yet demonstrate that the three
measured laws select different collective phases.

## Frozen design

Three representation classes were measured:

- switching: `intervals_24_decimal6`;
- polar-preserving: `moments_m1_m3`;
- sign-reversing: `centers_24_standard`.

Each independent block measured:

- concentrations \(\kappa=2,4,6,9,12\);
- 36 frozen, mirror-paired, stratified-jitter offsets;
- 24 responses per representation, concentration, and offset;
- 12,960 API calls.

The offset design is nonuniform, so no high harmonic aliases exactly into a
low harmonic. Primary coefficients were estimated by Fourier regression at
order \(M=6\). Orders \(M=2,4,8,12\) were frozen sensitivity analyses.

Tasks were deterministically shuffled before concurrent submission. Sample
seeds were hash-derived from block, representation, concentration, offset, and
repetition identifiers. The two blocks used disjoint seeds and separate API
runs.

## Execution

Block 1:

- 12,960 / 12,960 valid;
- actual cost: $5.2128315;
- API execution time: 439.36 s;
- effective parallelism: 19.80 / 20;
- throughput: 29.50 calls/s.

Block 2:

- 12,960 / 12,960 valid;
- actual cost: $5.2111845;
- API execution time: 423.69 s;
- effective parallelism: 19.84 / 20;
- throughput: 30.59 calls/s.

Combined:

- 25,920 / 25,920 valid;
- actual cost: $10.424016;
- measured API execution time: 863.05 s.

## Mean harmonic trajectories

### Switching interval representation

Across-block mean \(a_1\) at increasing concentration:

\[
0.665,\ 0.421,\ 0.251,\ -0.026,\ 0.082.
\]

Across-block mean \(a_2\):

\[
0.066,\ -0.038,\ 0.008,\ 0.432,\ 0.365.
\]

The transition is localized between \(\kappa=6\) and \(9\). At
\(\kappa=9\), the first harmonic is near zero and the second harmonic is
dominant. The frozen switching phenotype passed with probability 1.000 in
Block 1 and 0.9888 in Block 2.

### Polar-preserving moment representation

Across-block mean \(a_1\):

\[
0.981,\ 0.984,\ 0.856,\ 0.977,\ 0.998.
\]

The first harmonic remains strongly positive across the full concentration
range. The phenotype probability was 1.000 in both blocks. Mean activity was
1.000 at every concentration: the model never selected `stay` in this arm.

### Sign-reversing center representation

Across-block mean \(a_1\):

\[
0.682,\ 0.235,\ -0.045,\ -0.142,\ -0.392.
\]

The sign change occurs between \(\kappa=4\) and \(6\), earlier than the
interval representation's suppression. The narrow-field sign-reversal
phenotype passed with probability 1.000 in both blocks.

The corresponding \(a_2\) remained negative or near zero:

\[
-0.052,\ -0.189,\ -0.181,\ -0.035,\ -0.057.
\]

Thus this class is not the interval representation with a stronger second
harmonic. It belongs to a different interaction-symmetry trajectory.

## Representation-specific torque

The fitted constant component \(a_0\) also separated the classes.

- Interval representation: generally positive, from 0.080 to 0.249.
- Moment representation: close to zero except at intermediate concentrations.
- Center representation: consistently negative, from -0.368 to -0.253.

The encoding therefore controls both the alignment symmetry and the sign of
the mean social torque. The magnitude of the new interval \(a_0\) estimates is
larger than in the historical regular-grid baseline; this difference may
reflect nonuniform-grid projection of unresolved high harmonics or run-level
change and must not yet be interpreted as a calibrated collective rotation
rate.

## Replication and uncertainty

Block-to-block trajectory correlations were:

- interval: \(r(a_1)=0.939\), \(r(a_2)=0.971\);
- moments: \(r(a_1)=0.972\), \(r(a_2)=0.986\);
- centers: \(r(a_1)=0.994\), \(r(a_2)=0.868\).

All six correlations exceeded the frozen 0.80 threshold. All three classes
passed their phenotype rule in both blocks, and the overall replication gate
passed.

Thirty-nine of 45 block-difference intervals for \(a_0,a_1,a_2\) included
zero. The six exceptions were:

- interval \(a_1\) at \(\kappa=4\);
- interval \(a_0\) and \(a_1\) at \(\kappa=9\);
- center \(a_1\) at \(\kappa=6\);
- center \(a_0\) at \(\kappa=9\);
- center \(a_2\) at \(\kappa=12\).

These deviations did not change any qualitative phenotype. The maximum
pointwise Pearson overdispersion ratio was 0.920, so this two-block dataset did
not show excess pointwise variation relative to its within-cell trinomial
scale. With only two blocks, this is not evidence that API-level random effects
are absent.

Changing Fourier order across \(M=2,4,8,12\) changed \(a_1\) by at most 0.056
and \(a_2\) by at most 0.033. The class assignments are therefore not an
artifact of the selected primary regression order.

## Research decision

The data support adopting representation-induced interaction transmutation as
the central microscopic phenomenon.

The next justified experiment is a parameterized feature map:

- precision \(2,3,4,5,6\);
- bin-origin shift from zero to half a bin;
- bin count \(12,18,24,30,36,48\).

That experiment should localize whether coupling-sign changes are smooth
crossovers or representation-induced bifurcations. It should use the same
paired nonuniform offset design and independent seed blocks.

Collective Stage C remains deferred. Before direct LLM collective runs, the
measured response manifold must be expanded to bimodal, asymmetric, and sparse
small-\(N\) histograms and converted into frozen offline surrogate predictions.

## Claim boundary

Supported:

- representation and concentration jointly select distinct low-order
  interaction symmetries for this model and prompt contract;
- the three selected trajectories replicate across independent seed blocks.

Not yet supported:

- generality across model families;
- a causal attribution to one representation feature;
- phase selection in an interacting LLM population;
- quantitative collective rotation predicted from the fitted \(a_0\);
- continuous-grid convergence for all unresolved harmonics.

## Reproducibility artifacts

- Block 1 protocol:
  `experiments/stage_b_response_law/protocol_transmutation_v0_1_block_1.json`
- Block 2 protocol:
  `experiments/stage_b_response_law/protocol_transmutation_v0_1_block_2.json`
- Block 1 run:
  `runs/transmutation/transmutation-v0.1-block1_2bf5f01c6698/20260723T113147Z`
- Block 2 run:
  `runs/transmutation/transmutation-v0.1-block2_d8eb58f5e0d6/20260723T113939Z`
- Analysis:
  `analysis/transmutation_real/transmutation_analysis.json`
- Endpoint export:
  `analysis/transmutation_real/transmutation_endpoints.csv`
- Analysis code:
  `analysis/analyze_transmutation.py`
- Complex-kernel reanalysis:
  `analysis/complex_kernel_transmutation/` and
  `docs/STAGE_B_COMPLEX_KERNEL.md`
