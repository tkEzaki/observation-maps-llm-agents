# Stage B independent replication report

Date: 2026-07-23  
Status: completed  
Model: `gpt-5.4-mini`  
Prompt version: `response-law-v0.1`

## 1. Executive conclusion

The microscopic response measurement is operationally and statistically
replicable under an independent sampling-seed block.

The preregistered provisional gate required a response-curve correlation of at
least 0.8 for both concentration profiles. The observed correlations were:

- broad field, \(\kappa=2\): 0.949;
- narrow field, \(\kappa=12\): 0.899.

The primary replication gate therefore passed. At offsets where both runs had
a response magnitude of at least 0.2, the sign of the response agreed for
100% of offsets in both profiles.

The replicated result is not a simple universal Kuramoto-like law:

- the broad field is dominated by a positive first sine harmonic;
- the narrow field has almost no first sine harmonic and is dominated by the
  second harmonic;
- both profiles contain substantial even components;
- sharp changes between adjacent offsets are themselves reproducible.

The appropriate interpretation is that this model implements a reproducible,
concentration-dependent response operator for the frozen prompt. It is not yet
established that the irregular fine structure is invariant to histogram
binning or language representation.

## 2. Experimental separation

The first and replication runs used identical:

- model and prompt;
- 24-bin relative-phase representation;
- 48 circular offsets;
- concentration profiles \(\kappa=2\) and \(\kappa=12\);
- 50 responses per condition;
- temperature, output schema, and retry policy.

The only scientific sampling change was the seed block:

- first run base seed: `2026072301`;
- replication base seed: `2026077301`.

The seed ranges are disjoint. Protocol names and versions were also changed so
that the run directories and provenance hashes remain distinct. No response
from the first run was used in the replication run.

## 3. Execution and cost

First run:

- API responses: 4,800;
- valid responses: 4,800;
- wall-clock time: 193.3 seconds;
- reported input tokens: 2,899,200;
- reported output tokens: 65,740;
- cost: USD 2.47023.

Independent replication:

- API responses: 4,800;
- valid responses: 4,800;
- wall-clock time: 183.7 seconds;
- reported input tokens: 2,899,200;
- reported output tokens: 65,339;
- cost: USD 2.46843.

Combined acquisition:

- API responses: 9,600;
- valid responses: 9,600;
- reported input tokens: 5,798,400;
- reported output tokens: 131,079;
- total cost: USD 4.93866;
- total API-run wall time: approximately 6 minutes 17 seconds.

Both runs used 20 concurrent requests. Every run first executed one serial
response as a schema and API-parameter preflight. That response was retained
as the first scientific sample, so the preflight did not add calls or cost.

## 4. Curve-level replication

### Broad field: \(\kappa=2\)

- curve correlation: 0.949;
- RMSE between runs: 0.183;
- mean absolute difference: 0.136;
- direction agreement at jointly informative offsets: 100% across 30 offsets.

### Narrow field: \(\kappa=12\)

- curve correlation: 0.899;
- RMSE between runs: 0.203;
- mean absolute difference: 0.159;
- direction agreement at jointly informative offsets: 100% across 27 offsets.

The narrow response is somewhat less reproducible in amplitude than the broad
response, but both exceed the 0.8 correlation gate with consistent response
direction at strong-response offsets.

## 5. Pooled response estimates

Because the runs differ only in independent sampling seeds, pooling them gives
100 responses per offset and concentration.

### Broad field pooled structure

- constant component \(a_0=0.0179\);
- first sine harmonic \(a_1=0.5714\);
- second sine harmonic \(a_2=0.1140\);
- correlation with \(\sin\delta\): 0.753;
- odd-component RMS: 0.476;
- even-component RMS: 0.249;
- mean absolute change between adjacent offsets: 0.347;
- maximum adjacent change: 1.27.

The positive first harmonic is the dominant large-scale structure and is
consistent with an attractive response. The non-negligible even component and
higher harmonics show that the response is not well described by a single
odd sinusoid.

### Narrow field pooled structure

- constant component \(a_0=0.0121\);
- first sine harmonic \(a_1=-0.0385\);
- second sine harmonic \(a_2=0.3318\);
- correlation with \(\sin\delta\): -0.063;
- odd-component RMS: 0.361;
- even-component RMS: 0.243;
- mean absolute change between adjacent offsets: 0.427;
- maximum adjacent change: 1.46.

The first harmonic is near zero while the second harmonic is stable and large.
This is compatible with a response that can support two-cluster structure, but
that collective consequence has not yet been tested.

The pooled constant terms are small in both profiles. There is therefore no
evidence here for a large offset-independent advance or retard bias. This does
not rule out a nonzero social torque in asymmetric collective distributions.

## 6. Fourier coefficient compatibility

Raw actions were resampled independently within each offset and run for 2,000
bootstrap replicates. A coefficient was called compatible when the 95%
interval for `replication - first run` included zero.

- broad field: 12 of 13 coefficients compatible;
- narrow field: 11 of 13 coefficients compatible.

The primary broad first harmonic and narrow second harmonic were compatible
between runs. The nominal incompatibilities were:

- broad \(b_5\);
- narrow \(b_1\);
- narrow \(a_6\).

These are secondary coefficients. The 26 intervals were inspected without a
multiplicity correction, so isolated nominal exclusions are not strong
evidence of drift. The low-order scientific conclusion is stable.

## 7. Why the jagged curve matters

The first run raised the possibility that adjacent-offset jumps were sampling
noise. Independent replication makes that explanation insufficient:

- full curves correlate strongly across disjoint seed blocks;
- all jointly strong responses retain their sign;
- pooled curves remain sharply irregular rather than becoming smooth.

Possible mechanisms include:

1. a genuinely thresholded categorical response by the model;
2. dependence on higher moments or detailed shape of the phase density;
3. sensitivity to which histogram bins contain the modal mass;
4. sensitivity to decimal or textual representation of nearly adjacent
   distributions.

Only the first two represent a robust physical interaction operator. The last
two are representation effects. The current experiment cannot yet separate
them.

## 8. Research decision

Stage B has passed the microscopic-response replication gate.

This supports the claim:

> Under a frozen relative-phase prompt, the model implements a reproducible
> phase-dependent response whose harmonic structure changes qualitatively with
> concentration.

It does not yet support:

- a representation-independent coupling law;
- pairwise nonreciprocity;
- a predicted two-cluster collective phase;
- frequency arrest;
- generality across model families.

Proceeding immediately to a large collective phase diagram would therefore be
premature. The next experiment should isolate representation sensitivity.

## 9. Recommended next experiment

Use a diagnostic subset of offsets selected from:

- stable high-magnitude regions;
- stable sign reversals;
- the largest reproducible adjacent jumps;
- symmetry pairs \((\delta,-\delta)\).

For the same underlying continuous fixed fields, compare:

1. the current bin origin;
2. a half-bin-shifted discretization;
3. an equivalent numeric encoding with frozen precision.

The primary question is whether response direction and low-order Fourier
structure survive these equivalent representations. If they do, expand the
field manifold over additional concentrations and bimodal profiles. If they
do not, revise and version the observation protocol before any collective
experiment.

Direct two-body and small-\(N\) collective experiments remain the following
stage, conditional on this representation-invariance test.

## 10. Parallel execution policy

The response runs already dispatch independent calls with concurrency 20.
Future simulations will use:

- process-level parallelism across independent offline runs;
- vectorized agent updates within offline runs;
- concurrent LLM calls within each collective time step;
- a strict synchronous barrier before applying phase updates;
- one global in-flight request limit across concurrently active runs;
- immutable seed derivation independent of scheduling order;
- per-run atomic output directories and resume by completed run ID.

This preserves the synchronous dynamics while avoiding unnecessary serial API
or simulation work.

## 11. Reproducibility files

First run:

`runs/response_law/0.1_b0208be93331/20260723T101434Z`

Replication run:

`runs/response_law/0.1-rep1_9e7d68b5e354/20260723T102637Z`

Replication protocol:

`experiments/stage_b_response_law/protocol_v0_1_replication_1.json`

Machine-readable comparison:

`analysis/stage_b_replication_1.json`

Comparison implementation:

`analysis/compare_stage_b_runs.py`

Structural diagnostic implementation:

`analysis/stage_b_status.py`
