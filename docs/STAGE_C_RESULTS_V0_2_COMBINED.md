# Stage C v0.2 — combined report (moments branch close)

Date: 2026-07-24  
Status: **COMPLETE — moments Stage C branch closed for further paid collectives**  
Panels: v0.2a + v0.2b (44 runs; 55,600 valid calls; ≈ \$12.97)

| panel | session | calls | cost |
| --- | --- | ---: | ---: |
| v0.2a | `runs/stage_c/stage-c-v0.2a_ec22717957a1/20260724T014052Z` | 27,200 | \$6.344 |
| v0.2b | `runs/stage_c/stage-c-v0.2b_47d97bc0acc3/20260724T023438Z` | 28,400 | \$6.630 |

Artifacts: `analysis/stage_c_artifacts/stage_c_v0_2_combined/`  
Rebuild: `py analysis/stage_c/build_v0_2_combined.py`

---

## Supported claim (locked wording)

> A replay-informed, regime-aware microscopic surrogate prospectively
> corrected the abstention bias of the original model and improved
> collective **activity** and **stay** predictions across unseen coupling
> strengths, longer horizons, population sizes, and coupling signs.

**Not** claimed: accurate sync-speed / \(K_c\) / torque prediction;
negative \(K\) = disordered; arbitrary-\(N\) or cross-representation
macroscopic phase selection.

---

## Causal chain (prospective)

```text
v1 prospectively frozen
  → Stage C v0.1: stay over-prediction under collective dynamics
  → prompt / OOD excluded (audit PASS)
  → independent collective-field replay: stay=0 on all 30 Stage-C fields
  → regime-aware v2 built + 8-gate freeze (v1 untouched)
  → v0.2a/b: new K, T=100, seeds, N, sign(K)
  → activity / stay improve prospectively
```

---

## 1. Preregistered primary endpoints

Per-run table: `combined_endpoints_per_run.csv`  
Per-cell means: `combined_endpoints_by_cell.csv`

Columns include LLM / v1 / v2 for:

| endpoint | available |
| --- | --- |
| mean activity \(A\) | yes |
| empirical / teacher \(p_{\mathrm{stay}}\) | yes |
| \(r_1(T)\), time-averaged \(r_1\) | yes |
| \(t_{0.5}\), \(t_{0.9}\) | yes (NaN if never hit) |
| mean social torque \(\tau\) | yes |
| predictive interval coverage | **not computed** in v0.2 (deferred) |

### Panel rollups (activity / stay)

| panel | activity MAE v1 | v2 | teacher stay \|err\| v1 | v2 |
| --- | ---: | ---: | ---: | ---: |
| v0.2a (\(N=17\), \(K>0\)) | 0.100 | **0.014** | 0.105 | **0.010** |
| v0.2b (size + \(K=-0.15\)) | 0.085 | **0.044** | 0.089 | **0.047** |

---

## 2. Error reduction \(\Delta e = e_{\mathrm{v1}}-e_{\mathrm{v2}}\)

Seed-level absolute errors; positive \(\Delta e\) ⇒ v2 closer to LLM.  
Bootstrap: 2000 resamples of run-level \(\Delta e\); 95% percentile CI.

### Overall (44 runs)

| quantity | mean \(\Delta e\) | 95% CI | interpret |
| --- | ---: | --- | --- |
| \(\Delta e_A\) | **+0.057** | [0.035, 0.080] | v2 wins activity |
| \(\Delta e_{\mathrm{stay}}\) (teacher) | **+0.061** | [0.037, 0.084] | v2 wins stay |
| \(\Delta e_{r_1(T)}\) | +0.013 | [0.004, 0.023] | modest \(r_1\) gain |
| \(\Delta e_{t_{0.9}}\) | +0.92 | [−2.0, +4.0] | **not significant** |
| \(\Delta e_{\tau}\) | −0.023 | [−0.041, −0.005] | v1 slightly closer on \(\|\tau\|\) |

### By panel

| panel | \(\Delta e_A\) CI | \(\Delta e_{\mathrm{stay}}\) CI |
| --- | --- | --- |
| v0.2a | [0.080, 0.092] | [0.092, 0.099] |
| v0.2b | [0.010, 0.073] | [0.005, 0.076] |

Sync speed is **not** rescued by the stay-gate fix alone. Residual candidates:
active advance/retard law, torque time-correlation, non-independent sampling,
local direction response under evolving fields.

---

## 3. Generalization axes

| axis | novelty vs training / v0.1 |
| --- | --- |
| \(K\in\{0.08,0.10,0.12\}\) | not used as Stage C v0.1 collective cells |
| \(T=100\) | v0.1 / replay source used \(T\le40\) |
| new init seeds | independent of v0.1 |
| \(N=9\) | size generalization (v0.2b) |
| \(K=-0.15\) | sign generalization; active polar suppression |

Replay fields informed v2 training, but v0.2 cells use new \(K\), longer \(T\),
and new seeds. Improvement is therefore not a replay-field lookup.

---

## 4. Transition bounds (coarse grid)

Safe statements (not precision \(K_c\)):

\[
0 < K_{\mathrm{cross}}^{\mathrm{LLM}}(N=9) \le 0.06
\]

(\(K=0\): 0/4 lock; \(K=0.06\): 4/4).

\[
K_{\mathrm{cross}}^{\mathrm{LLM}}(N=17) \le 0.08
\]

(smallest measured positive \(K\) in v0.2a already 4/4 lock).

Offline surrogate crossings (v1 = v2):

\[
K_{\mathrm{cross}}^{\mathrm{surr}}(N=9)=0.08,\qquad
K_{\mathrm{cross}}^{\mathrm{surr}}(N=17)=0.10.
\]

**Correction:** LLM locking onset is **earlier** than the attenuated surrogate
envelope. Do **not** write that N=9 onset “matches” 0.08.

Negative \(K\): \(A\simeq0.996\)–\(0.999\) with \(r_1(T)\simeq0.028\)–\(0.036\) —
active polar-order suppression, not inactivation / disorder.

---

## 5. Supported vs not supported

### Strongly supported

1. Moments has a localized antipodal-abstention regime and an active directional regime.
2. A single smooth surrogate leaks stay across regimes and weakens collective action.
3. Replay-informed regime-aware v2 prospectively improves activity and stay.
4. Improvement replicates across \(N\in\{9,17\}\), both signs of \(K\), and \(T=100\).
5. Positive \(K\) induces polar synchronization; negative \(K\) suppresses polar order at near-unit activity.

### Not supported

1. Surrogate fully predicts sync speed.
2. Surrogate predicts true \(K_c\).
3. Negative \(K\) creates a disordered phase.
4. Arbitrary-\(N\) generalization.
5. Representation-dependent macroscopic phase selection.
6. Same coarse-graining for non-moments encodings.

---

## 6. Moments branch status

| item | status |
| --- | --- |
| Further moments-only paid collectives | **not required** |
| Moments v3 (direction/torque polish) | optional; not central claim |
| `moments_bundle_v1` | permanent v0.1 prospective baseline |
| `moments_bundle_v2` | frozen Stage C v0.2 baseline |
| Next scientific priority | **second representation → Stage C** |

Recommended next path (no moments paid expansion):

1. Close moments narrative with this combined report (done).
2. Choose one histogram representation (prefer improving **centers** given prior weak positive risk ranking; do not force moments model form).
3. Build representation-specific AD + surrogate.
4. Coverage-directed replay → freeze.
5. Matched-seed dual Stage C: moments vs histogram under identical ICs.

---

## References

- v0.2a: `docs/STAGE_C_RESULTS_V0_2A.md`
- v0.2b: `docs/STAGE_C_RESULTS_V0_2B.md` (K_cross wording corrected)
- Protocol: `docs/STAGE_C_V0_2_PROTOCOL.md`
- v2 freeze: `docs/STAGE_3B_V2_FREEZE_CHECKLIST.md`
- Replay branch A: `docs/STAGE_C_REPLAY_RESULTS.md`
