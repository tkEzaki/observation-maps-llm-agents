# Matched representation collective — protocol v0.1

Date: 2026-07-24  
Status: **preregistered; surrogate-free; paid not authorized**  
Protocols:

- `experiments/stage_c/protocol_matched_rep_collective_v0_1_moments.json`
- `experiments/stage_c/protocol_matched_rep_collective_v0_1_centers.json`
- `experiments/stage_c/protocol_matched_rep_collective_v0_1_intervals.json`

## Experiment identity

| field | value |
| --- | --- |
| `experiment_family` | `matched_representation_collective` |
| `surrogate_free` | `true` |
| Prompt contract | `response-law-v0.1` |
| Action set | \(\{-1,0,+1\}\) (retard / stay / advance) |
| Integrator | `circlemap.engine.step` (unchanged) |

## Design grid

| item | value |
| --- | ---: |
| \(N\) | 17 |
| peer count | 16 |
| \(T\) | 100 |
| \(K\) | \(-0.15,\,0,\,0.08,\,0.15\) |
| paired seeds | 6 |
| \(\omega\) halfwidth | 0.05 (equally spaced, symmetric) |
| representations | 3 (moments / centers / intervals) |
| expected calls | \(17\times4\times100\times6\times3 = 122{,}400\) |

Moments trajectories are **fresh rematches** under this grid (do not graft
Stage C v0.2a).

## Matching lock (causal identification)

Held identical across \(\mathcal R\):

1. `base_seed` (shared);
2. physical `init_seed` formula (representation-free):
   `base_seed + 1000*N + 10*seed_i + round(|K|*1000)`;
3. \(\theta_i(0)\sim\mathrm{Unif}(-\pi,\pi)\) from that seed;
4. \(\omega_i\) linspace on \([-\omega_{\mathrm{hw}},+\omega_{\mathrm{hw}}]\);
5. \(N\), \(K\), \(T\), `run_id` scheme `N17_K{…}_s{i}`;
6. system prompt contract, action parse, integrator.

**Only** the histogram→text serialization \(\mathcal R\) changes.

### LLM sample-seed policy

API sample seeds hash **include** `protocol_version` (which embeds the
representation). Physical ICs remain paired; LLM stochastic draws are
**independent** across encodings (avoids accidental coupling of decoding
noise).

Match manifest: `analysis/matched_rep_collective/match_manifest.json`
(written by `write_match_manifest.py`).

## Primary endpoints (preregistered)

Per run \((s,K,\mathcal R)\):

| endpoint | definition |
| --- | --- |
| `mean_r1`, `final_r1` | time-average / terminal polar order |
| `mean_r2`, `final_r2` | second harmonic |
| `mean_r3`, `final_r3` | third harmonic |
| `mean_activity` | \(\mathbb{E}_{t,i}|a_{i,t}|\) |
| `mean_tau_social` | mean signed social torque |
| `t_r1_gt_0.5`, `t_r1_gt_0.9` | first-passage times (NaN if never) |
| `omega_coll` | collective frequency |
| `locking_fraction` | indicator \(r_1(T)>0.9\) (aggregated as rate) |
| `cluster_phenotype` | see below |
| paired \(\Delta_{\mathcal R}\) | centers−moments, intervals−moments on same \((s,K)\) |

### Cluster phenotype (minimal preregistered rule)

Before acquisition, lock:

1. If `final_r1 ≥ 0.9` → `polar_locked`;
2. Else if `final_r2 ≥ 0.5` and `final_r2 > final_r1` → `high_r2_multicluster`;
3. Else if `mean_activity < 0.15` → `abstention_dominated`;
4. Else → `partial_or_disordered`.

Refinements after seeing outcomes are **exploratory only**.

## Inferential model

Treat seed \(s\) as a paired block:

\[
Y_{s,\mathcal R,K}
=
\alpha_s
+\beta_{\mathcal R}
+\gamma_K
+(\beta\gamma)_{\mathcal R,K}
+\varepsilon_{s,\mathcal R,K}.
\]

Report seed-block contrasts vs moments reference with nonparametric CIs
(bootstrap over seeds). Primary multiplicity: family-wise attention on
`mean_r1`, `final_r1`, `mean_activity`, `t_r1_gt_0.5` first; remaining
endpoints secondary / pathway.

## Success mapping

See `docs/MATCHED_REP_COLLECTIVE_HYPOTHESIS.md` Outcomes A / B / null.

## Authorization

Paid OpenAI runs require:

1. Cost card estimate reviewed (`docs/MATCHED_REP_COLLECTIVE_COST_CARD.md`);
2. Human review checklist complete;
3. Sidecar `analysis/matched_rep_collective/auth_go.json` with
   `paid_authorized: true` (currently **false**);
4. CLI `--yes`.

`--estimate-only` is allowed **without** paid auth for this family.

No `moments_bundle_*` is required or consulted for primary analysis.

## Size replication

\(N=9\) is **blocked** until \(N=17\) shows Outcome A or B.

## Cross-encoding replay (Phase 2) — rules locked pre-outcome

Field-selection strata, quotas, and call budget are frozen in
`docs/MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md` **before** inspecting matched
collective outcomes. Acquisition of the 3×3 replay panel occurs **after**
matched collectives finish, under those frozen rules.

## Analysis entrypoint

```powershell
py analysis/matched_rep_collective/analyze_paired_endpoints.py --sessions ...
```

Primary path is LLM-vs-LLM; surrogate columns are out of scope.

## Explicit non-actions

- INT-3 / CENT confirmatory / histogram surrogate freeze
- Bundle-based Stage C auth for this family
- Post-hoc retuning of cluster phenotype thresholds after Outcome A/B calls
