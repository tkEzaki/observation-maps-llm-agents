# Stage C v0.1 — offline diagnostics (§5)

Date: 2026-07-24  
Status: **complete (no new API spend)**  
Artifacts: `analysis/stage_c_artifacts/offline_v0_1/`  
Summary JSON: `offline_diagnostics_summary.json`

Branch heuristic (§7): **A** — LLM fields are low-risk; the one-step model
is **structurally informative but systematically miscalibrated in activity**
(stay over-prediction), so closed-loop magnitude / speed are attenuated.

---

## 5.1 / 5.2 Trajectory harmonics (per \(N\times\)seed)

Source: `trajectory_summary_per_run.csv`, `trajectory_harmonics_timeseries.csv`.

### \(K=+0.15\) (attractive)

| run | final \(r_1\) | final \(r_2\) | final \(r_3\) | \(t(r_1>0.5)\) | \(t(r_1>0.9)\) |
| --- | ---: | ---: | ---: | ---: | ---: |
| N9 s0 | 0.993 | 0.971 | 0.936 | 2 | 7 |
| N9 s1 | 0.994 | 0.974 | 0.943 | 7 | 21 |
| N17 s0 | 0.996 | 0.985 | 0.966 | 3 | 11 |
| N17 s1 | 0.997 | 0.988 | 0.973 | 6 | 19 |

All four runs reach near-complete **polar** sync. Seed variance is mainly in **arrival time** (\(t_{0.9}\in[7,21]\)), not final phase. Surrogate–LLM final-\(r_1\) gap is therefore not “different attractor” but **faster / stronger locking for LLM**.

### \(K=-0.15\) (repulsive)

| run | final \(r_1\) | final \(r_2\) | final \(r_3\) |
| --- | ---: | ---: | ---: |
| N9 s0 | 0.082 | 0.148 | **0.457** |
| N9 s1 | 0.049 | 0.184 | **0.389** |
| N17 s0 | 0.011 | **0.268** | 0.161 |
| N17 s1 | 0.030 | 0.147 | **0.411** |

**Do not call this disordered.** Polar order is suppressed, but \(r_2\)/\(r_3\) remain \(O(0.15\)–\(0.46)\). Safe claim: **negative \(K\) suppresses polar order**; cluster / higher-harmonic structure remains possible.

### \(K=0\)

| run | final \(r_1\) | final \(r_2\) | final \(r_3\) |
| --- | ---: | ---: | ---: |
| N9 s0 | 0.347 | 0.128 | 0.406 |
| N9 s1 | 0.103 | 0.138 | 0.159 |
| N17 s0 | 0.278 | 0.134 | 0.162 |
| N17 s1 | 0.022 | 0.270 | 0.347 |

Large seed / \(N\) scatter under pure drift — expected; not a surrogate test.

**\(N=9\) vs \(N=17\):** both usable at \(K=\pm0.15\) for the qualitative regime; they are **not** interchangeable on arrival times or negative-\(K\) harmonic pattern.

---

## 5.3 AD / risk on LLM fields

Source: `llm_field_risk_teacher.csv`.

| quantity | value |
| --- | ---: |
| mean \(Q_{\mathrm{risk}}\) (all runs) | **0.0** |
| mean \(\bar R\) | ≈0.12–0.13 |
| \(R_{0.90}\) (typical) | ≈0.13–0.15 |
| \(R_{0.99}\) (typical) | ≈0.14–0.16 |
| \(q_{\mathrm{peer\_reject}}\) | 0 |

LLM collective fields stay **inside** the moments applicability domain. The \(r_1\) magnitude error is **not** explained by high-risk / OOD fields.

---

## 5.4 Teacher-forced one-step

Same file. Overall mean log-loss ≈ **0.735**.

| \(K\) | log-loss | pred stay | obs stay | pred \(\langle f\rangle\) | obs \(\langle f\rangle\) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| −0.15 | 0.877 | 0.188 | **0.003** | −0.13 | **−0.34** |
| 0 | 0.812 | 0.172 | 0.008 | −0.12 | −0.01 |
| +0.15 | 0.516 | 0.107 | 0.032 | −0.02 | −0.05 |

Dominant failure mode: **stay over-prediction** and under-prediction of action extremity, worst at negative \(K\). This matches a closed-loop “shrink toward baseline” bias.

Time-bin detail: `teacher_forced_by_tbin.csv`.

---

## 5.5 Surrogate predictive replicates (\(n=200\))

Source: `surrogate_replicates.csv`. Fixed LLM init; resample actions only.

| | count |
| --- | ---: |
| LLM final \(r_1\) ∈ \([p_{05},p_{95}]\) | **7 / 12** |
| below \(p_{05}\) | 3 (all \(K=-0.15\)) |
| above \(p_{95}\) | 2 (both \(N=17\), \(K=+0.15\)) |

So the MAE≈0.07 is not just noise: LLM sits in the **tails** of the surrogate predictive distribution for nonzero \(K\).

---

## 5.6 Activity, torque, engine identity

| check | result |
| --- | --- |
| \(\Omega_{\mathrm{coll}}-\bar\omega-K\tau_{\mathrm{social}}\) | \(\sim10^{-17}\) (identity holds) |
| mean activity \(\lvert f\rvert\) | ≈0.94–1.00 (LLM rarely stays) |
| mean \(\tau_{\mathrm{social}}\) | signed; drives \(\Omega_{\mathrm{coll}}\) at \(K\neq0\) |

---

## §7 branch decision (from this offline pass)

| branch | meaning | evidence |
| --- | --- | --- |
| **A (selected)** | low-risk fields; one-step usable; closed-loop magnitude bias | \(Q_{\mathrm{risk}}=0\); log-loss finite; stay bias; LLM outside surrogate PI at nonzero \(K\) |
| B | high-risk LLM fields | rejected |
| C | one-step structure broken | rejected as primary (stay bias is calibration / hurdle, not AD failure) |

**Next paid run is still not authorized by this note.** Recommended offline-next / paid-next if A holds:

1. Frozen-surrogate dense \(K>0\) scan at \(T=100\), many seeds (no API).
2. Stage C v0.2 LLM only on transition / control cells with ≥4–8 seeds.

Do **not** expand paid \(T\) before that surrogate scan.

---

## Artifact index

| file | content |
| --- | --- |
| `trajectory_summary_per_run.csv` | per-run \(r_{1,2,3}\), arrival, torque |
| `trajectory_harmonics_timeseries.csv` | \(r_k(t)\), activity, \(\tau\) |
| `llm_field_risk_teacher.csv` | \(\bar R\), quantiles, \(Q_{\mathrm{risk}}\), teacher scores |
| `teacher_forced_by_tbin.csv` | calibration by time bin |
| `surrogate_replicates.csv` | 200-rep predictive intervals |
| `offline_diagnostics_summary.json` | rollup + branch |
