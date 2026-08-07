# System specification: language-mediated circular map

Status: v0.2, microscopic response-law transmutation adopted after Stage B  
Date: 2026-07-23

## 1. Scientific question

The primary question is:

> How does the representation of a relative phase distribution transmute the
> microscopic response operator implemented by a language model (complex
> harmonics and action distribution), and—only after that is established—how
> might those laws select collective phase order and frequency?

The system is not assumed to be a Kuramoto model. Kuramoto-like coupling,
threshold responses, higher harmonics, asymmetry, and zero-response regions
are alternative empirical outcomes.

## 2. State and update rule

Each agent has an unwrapped continuous phase \(x_i(t)\in\mathbb{R}\), with
wrapped phase

\[
\theta_i(t)=x_i(t)\bmod 2\pi.
\]

All agents update synchronously:

\[
x_i(t+1)=x_i(t)+\omega_i+K f_i(t).
\]

- \(\omega_i\): natural phase increment per step, fixed within a run.
- \(K\): numerical coupling scale applied by the engine, in radians per step.
- \(f_i(t)\in\{-1,0,+1\}\): normalized social action returned by the LLM.
- Mapping: `retard = -1`, `stay = 0`, `advance = +1`.

The LLM never receives \(K\), an absolute phase, a clock label, a preferred
phase, or a target trajectory. Negative coupling is implemented only by the
engine through the sign of \(K\); the prompt and response law remain unchanged.

At \(K=0\), the engine must satisfy

\[
x_i(t)=x_i(0)+t\omega_i
\]

up to floating-point tolerance, independently of LLM output.

## 3. Representation-aware observation operator

Agent \(i\) observes relative phases

\[
\delta_{ij}(t)=\operatorname{wrap}(\theta_j(t)-\theta_i(t))
\in[-\pi,\pi).
\]

Let \(\rho_i(t)\) denote the relative-phase distribution available to agent
\(i\). A representation operator \(\mathcal R_\eta\) converts that field into
the text supplied to the LLM, where \(\eta\) includes bin count, bin origin,
numeric precision, interval versus center semantics, count normalization, and
moment order. The response law is

\[
g_{\mathcal R_\eta}(\rho)
=\mathbb E\!\left[f\mid\mathcal R_\eta(\rho)\right].
\]

Thus representation is an externally controlled component of the interaction
operator, analogous to an observation channel or sensor encoding. It is not
assumed to be a neutral readout of \(\rho\).

The historical v0.1 representation is a fixed 24-bin histogram. Counts exclude
the focal agent, and serialization contains every bin label in canonical order.
It remains a named switching reference, not a representation-independent
default.

For the primary all-to-all experiments, counts are normalized to fractions
before presentation so that prompt values have a fixed scale across \(N\).
Raw peer count \(N-1\) is not stated in the prompt. Population-size effects
are therefore physical finite-size effects rather than prompt-length effects.

Required properties of the observation operator:

1. no absolute phase information;
2. invariant to peer permutation;
3. fixed number and order of fields;
4. deterministic boundary convention;
5. global-rotation invariance before serialization.

The exact representation specification, bin edges, decimal precision, and
prompt text must be frozen before each paid response measurement. Semantically
equivalent encodings are distinct interventions and may not be pooled.

The transmutation-v0.1 canonical classes are:

1. `intervals_24_decimal6`: concentration-induced harmonic switching;
2. `moments_m1_m3`: polar-preserving response;
3. `centers_24_standard`: narrow-field sign-reversing response.

## 4. LLM response contract

The initial portable response contract is:

```json
{"social_action": "advance"}
```

where the only valid values are `advance`, `stay`, and `retard`.

The backend is stateless. It receives no previous answer, chain of thought,
agent identity, round number, \(K\), \(\omega_i\), or current absolute phase.
Invalid output is never replaced by a social action. Response-law measurements
record a failed call as missing. In a collective run, a call that remains
invalid after the frozen retry policy aborts and invalidates that run before
the phase update; no partially updated round is written as scientific data.

Sampling seed, initial-condition seed, and frequency-allocation seed are
separate. Backend model identifier, checkpoint/revision, tokenizer, generation
parameters, and software versions are recorded with every dataset.

If a backend exposes stable token log probabilities for the three actions, a
secondary response can be defined as

\[
f_i=p(\mathrm{advance})-p(\mathrm{retard}).
\]

This continuous-probability arm is not pooled with the sampled three-action
arm unless specified before collection.

## 5. Sign convention and one-body response measurement

For a consensus located at relative offset \(\delta\):

- positive \(\delta\) means peers are ahead of the focal phase;
- `advance` is \(f=+1\);
- an attractive first harmonic therefore has positive
  \(g(\delta)=E[f\mid\delta]\) for small positive \(\delta\).

Measure \(g_{\mathcal R}(\delta)\) before collective experiments using
synthetic fixed fields. The historical v0.1 grid used 48 equally spaced
offsets. New primary measurements use a frozen nonuniform, mirror-paired,
stratified-jitter design. Fourier coefficients are obtained by regression on
the actual offsets rather than by regular-grid projection. This prevents a
single high harmonic from aliasing exactly into \(m=1\) or \(m=2\).

Estimate:

\[
g(\delta),\quad
g_{\mathrm{odd}}(\delta)=\frac{g(\delta)-g(-\delta)}{2},\quad
g_{\mathrm{even}}(\delta)=\frac{g(\delta)+g(-\delta)}{2},
\]

and a Fourier expansion with coefficients reported through at least harmonic
\(m=6\). Report multinomial uncertainty, not only the mean action.

Primary transmutation measurements use independent seed blocks with identical
offsets and different hash-derived sample seeds. Tasks are deterministically
shuffled before concurrent submission so representation, concentration, and
offset are not confounded with API execution order. Block-specific estimates
are retained; pooling may not erase excess run-to-run variation.

A second fixed-field grid varies concentration at fixed mean offset. This
tests whether the response is a function only of consensus direction or is a
genuine many-body/density-dependent operator.

## 6. Two-body tests

Before \(N>2\), measure the two directed responses for mirrored configurations.
Do not label a result nonreciprocal solely because collective social torque is
nonzero. Pairwise nonreciprocity requires direct evidence that

\[
T_{i\leftarrow j}(\delta)+T_{j\leftarrow i}(-\delta)\ne0.
\]

If this pairwise sum vanishes but many-body configurations generate net torque,
the candidate mechanism is non-conservative many-body language coupling.

## 7. Collective observables

Primary observables are computed from unwrapped phases:

\[
r_m(t)=\left|\frac{1}{N}\sum_i e^{im\theta_i(t)}\right|,
\qquad m=1,2,3.
\]

\(r_1\) measures polar order, \(r_2\) detects antipodal two-cluster or nematic
order, and \(r_3\) detects threefold cluster structure. None may be substituted
post hoc as the sole order parameter after observing a trajectory.

\[
\Omega_i^{\mathrm{eff}}
=\frac{x_i(T)-x_i(T_0)}{T-T_0},
\qquad
\Omega_{\mathrm{coll}}
=\frac{1}{N}\sum_i\Omega_i^{\mathrm{eff}},
\]

\[
\tau_{\mathrm{social}}
=\left\langle\frac{1}{N}\sum_i f_i(t)\right\rangle_t.
\]

The engine identity

\[
\Omega_{\mathrm{coll}}
=\bar\omega+K\tau_{\mathrm{social}}
\]

must hold numerically for the same analysis window. This identity is a
bookkeeping validation; a nonzero torque is a scientific result only after
response validity and symmetry controls pass.

Frequency-locking fraction is

\[
q_{\mathrm{lock}}=
\frac{1}{N}\sum_i
\mathbf{1}(|\Omega_i^{\mathrm{eff}}-\Omega^\ast|<\epsilon).
\]

\(\Omega^\ast\), the analysis window, and \(\epsilon\) must be fixed using
pilot data and then frozen before production. No post-hoc binary `sync`
classifier is used as a primary endpoint.

## 8. Initial pilot sequence

### Stage A: engine validation, no paid LLM calls

1. Exact free drift at \(K=0\).
2. Global rotation equivariance for trajectories.
3. Peer-permutation invariance of the observation.
4. Histogram boundary and normalization tests.
5. Synchronous-update test.
6. Exact collective-frequency torque identity.

Any failure blocks subsequent stages.

### Stage B: response-law pilot

- 24 bins;
- 48 consensus offsets;
- balanced repeated samples at each offset;
- at least two concentration profiles;
- one frozen model/checkpoint and generation configuration.

Proceed only if:

1. valid structured-response rate is at least 99%;
2. repeated calls are operationally reproducible under recorded seeds or
   sufficiently sampled when deterministic seeding is unavailable;
3. the response has detectable phase dependence beyond a constant action bias;
4. mirrored-offset measurements are sufficient to estimate odd and even parts.

If criterion 3 fails, do not run a \(K\)-sweep; revise the observation/task
definition as a new versioned protocol.

Before Stage C, the low-order response structure must pass a representation
test. Primary endpoints are complex harmonics
\(C_m=a_m+ib_m\) (equivalently \((R_m,\phi_m)\)) for \(m=1,2\), together with
activity, action entropy, and offset-averaged action bias (\(a_0\))
robustness checks—not \(a_1\) alone and not pointwise curve correlation.
The fitted mean-response derivative at zero offset,
\(g'(0)=\sum m a_m\), and fitted fixed-point counts are diagnostic only
until a dedicated local-stability or pair/collective protocol exists; they
are not primary endpoints and are not aligned-state stability claims. The diagnostic set includes bin-origin shifts,
bin-count changes, decimal or count encodings, and a circular-moment
representation. High-frequency grid fingerprints are reported separately from
the low-order response.

Status as of 2026-07-23: protocol v0.1 failed this gate. The replicated
24-bin, six-decimal baseline passed its frozen harmonic-switching rule, but
none of seven alternate encodings passed. Stage C is blocked until the
observation contract is revised and versioned, or representation dependence is
adopted explicitly as the object of study. See
`docs/STAGE_B_REPRESENTATION_INVARIANCE.md`.

Representation dependence was subsequently adopted as the object of study.
Transmutation-v0.1 mapped three canonical classes over
\(\kappa=2,4,6,9,12\) using two independent seed blocks and nonuniform Fourier
regression. All three representation-specific phenotype trajectories passed
the frozen replication gate. The supported claim is **microscopic
response-law transmutation**, not macroscopic interaction-phase selection.
See `docs/STAGE_B_TRANSMUTATION_MAP.md` and `docs/STAGE_B_COMPLEX_KERNEL.md`.

This pass does not itself authorize Stage C. The response manifold must first
cover bimodal, asymmetric, and representative sparse small-\(N\) fields, and
offline surrogate predictions must be frozen before direct collective calls.

Feature-sweep v0.1 subsequently varied numeric precision, bin count, and
center-bin origin over two independent blocks. Precision and bin count produced
large reproducible complex first-harmonic trajectories, but no adjacent
parameter pair passed the frozen 0.95 jump criterion in both blocks. The
result supports systematic feature control (continuous or sharp crossover) but
not a discontinuous bifurcation claim. No automatic boundary top-up or Stage C
progression is authorized. See `docs/STAGE_B_FEATURE_SWEEP.md` and
`docs/STAGE_B_COMPLEX_KERNEL.md`.

Stimulus-manifold v0.1 measured the three canonical encodings on bimodal,
antipodal, asymmetric, and sparse fields (15,552 calls, two seed blocks).
Scientific verdict: **symmetry-selective polar-channel collapse**.
Transmutation persists on unimodal, non-antipodal bimodal, asymmetric, and
sparse fields; exact antipodal symmetry suppresses representation-separated
\(C_1\) phenotypes while activity, even harmonics, and full action
distributions remain representation-dependent. Stage C remains blocked until
Stage 3B freezes an offline surrogate with an OOD policy on the expanded
manifold (Stage 3A implementation is authorized now; see
`docs/STAGE_3A_SURROGATE.md`). See `docs/STAGE_B_STIMULUS_MANIFOLD.md` and
`docs/STAGE_B_NEXT_ANALYSIS.md`.
The signed dense weight-imbalance sweep (P1) is documented in
`docs/STAGE_B_ANTIPODAL_WEIGHT.md`: moments abstain at exact antipodal balance
and activate by the smallest measured nonzero \(|\varepsilon|=0.02\)
(comparative slope \(\chi_{a_1}\) is not a precise local derivative).
Nested sparse peer imbalance + realization variance (Step D) is documented in
`docs/STAGE_B_SPARSE_PEER.md`.

### Stage C: two-body and small-ensemble pilot

- two-body directed response test;
- \(N=8\) and \(N=16\);
- \(T=100\);
- equally spaced natural frequencies in a prespecified positive interval;
- \(K=0\) plus a coarse signed \(K\) grid;
- no production claims.

Small-\(N\) histograms are sparse and reveal count granularity even after
normalization. Stage C therefore records the distance from every realized
histogram to the Stage B stimulus manifold and includes direct response
measurements for representative sparse histograms. Any smoothing or
pseudocount rule must be frozen and applied identically in measurement and
collective runs.

This stage selects the informative ranges of \(K\), frequency width, \(T_0\),
and locking tolerance without reusing these runs as production replicates.

### Stage D: frozen production design

After Stage C, freeze:

- primary \(K\)-by-frequency-width grid;
- \(N\), \(T\), number of independent runs;
- \(\mu_\omega\) translation grid;
- exclusion and missing-response rules;
- uncertainty intervals and multiplicity handling;
- response-kernel estimator;
- fit-free prediction procedure for collective observables.

Production data are kept separate from pilot data.

## 9. Model-derived prediction

The response kernel is fit only to Stage B single-agent data. Its parameters
are frozen before predicting collective runs. Simulations driven by that
measured stochastic response must predict, without refitting to collective
data:

- \(\bar r(K,\Delta)\);
- \(\Omega_{\mathrm{coll}}(K,\Delta,\mu_\omega)\);
- frequency-locking fraction;
- cluster structure where identifiable.

Prediction failure is reported as a failure of the chosen coarse-graining, not
repaired by fitting an unrelated minimal model to the collective results.

## 10. Data layout and provenance

New data use a new dataset namespace and never write inside `legacy/`.

```text
runs/
  response_law/<protocol_id>/<run_id>/
  two_body/<protocol_id>/<run_id>/
  pilot_collective/<protocol_id>/<run_id>/
  production/<protocol_id>/<run_id>/
```

Every run stores:

- immutable resolved configuration;
- protocol and prompt hashes;
- model/checkpoint and generation metadata;
- all seeds;
- initial unwrapped phases and natural frequencies;
- serialized observations;
- raw and parsed backend responses;
- social actions;
- unwrapped and wrapped phase trajectories;
- validity and retry records;
- software version or source-tree hash.

No failed response is converted to `stay` in scientific data.

## 11. Deliberate exclusions from the new system

The following concepts are not carried over:

- preferred slots or habits;
- \(\mu_i+\omega_i t\) as an external target;
- natural-language percentage weights;
- absolute clock labels;
- integer internal phase;
- visible-peer valleys as a primary claim;
- post-hoc synchronization labels;
- ad hoc minimal models fit to collective outcomes.

The `legacy/` directory is an archive and is not imported by the new package.

## 12. Decisions still required before paid calls

1. Primary open-weight model and exact checkpoint/revision.
2. Inference runtime and whether stable three-token log probabilities exist.
3. Exact prompt wording and histogram numeric format.
4. Sampling temperature and seed support.
5. Initial \(\omega\), \(K\), and concentration grids after unit tests.

These decisions do not block implementation of the deterministic engine and
its invariance tests.

## 13. Parallel execution policy

Independent simulations and LLM calls should run concurrently without changing
their scientific random streams.

- Offline simulations parallelize independent runs with a process pool and
  vectorize agents within each run.
- LLM collective dynamics retain a synchronous barrier at every time step:
  all observations come from one immutable snapshot, then the \(N\) calls are
  dispatched concurrently, and no phase is updated until all valid responses
  return.
- Multiple LLM runs may overlap, but a single global in-flight request limit
  prevents accidental nested oversubscription.
- Worker count and request concurrency are configurable. The default CPU worker
  count leaves one logical core free; the API default is selected below the
  measured provider rate limit.
- Simulation, sampling, and run seeds are derived from immutable run IDs before
  dispatch, so scheduling order cannot change results.
- Each run writes to its own temporary directory and is atomically finalized.
  Completed run IDs are skipped on resume.
- Every paid execution prints its base and retry-ceiling cost before backend
  initialization and still requires explicit authorization.
