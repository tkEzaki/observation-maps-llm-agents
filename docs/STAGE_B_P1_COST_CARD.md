# P1 dense antipodal weight-imbalance — cost card

Date: 2026-07-23  
Status: **authorized to run** (user: proceed with revised plan)  
Protocol: `protocol_antipodal_weight_v0_1_block_{1,2}.json`

## Design lock (from power simulation)

| item | value |
| --- | --- |
| Axis | dense antipodal weight \(w=\tfrac12+\varepsilon\) only |
| \(\varepsilon\) | \(-0.10,-0.05,-0.02,0,0.02,0.05,0.10,0.20\) |
| Encodings | intervals / moments / centers |
| Offsets × reps | 36 × 12 (same as manifold v0.1) |
| Blocks | 2 independent seed blocks |
| Calls / block | \(3\times8\times36\times12=10{,}368\) |
| Calls total | \(20{,}736\) |

Near-zero densification (\(0.005,0.01\)) deferred unless \(A(|\varepsilon|)\)
between 0 and 0.02 is unresolved after this panel.

Power simulation: `analysis/stage_b_offline_p3_p4/p1_power_simulation.json`

## Cost estimate

Backend: `gpt-5.4-mini`  
Prices (same as prior Stage B paid runs): input **$0.75 / 1M**, output **$4.50 / 1M**  
Env: `<HOME>\Dropbox\research_current\gen_ai_logi\rq1\.env`

Scaling from analyzed manifold pair ($6.266799 / 15,552 calls):

| | estimate |
| --- | ---: |
| Two blocks (scaled) | **~$8.36** |
| Per block | ~$4.18 |
| Worst-case ×3 retries (upper) | ~$25 |

Exact gate print comes from `--estimate-only` / run start `cost_estimate.json`.

## Actual spend (completed)

| block | actual USD |
| --- | ---: |
| block_1 | 4.2177915 |
| block_2 | 4.216392 |
| **total** | **8.4341835** |

Results: `docs/STAGE_B_ANTIPODAL_WEIGHT.md`.

## Primary endpoints (analysis)

- Signed \(C_1(\varepsilon)\), \(\chi_1\) / projected \(\chi_\parallel\)
- \(A(|\varepsilon|)\), \(\varepsilon_{1/2}\), \(A-A(0)\)
- Even checks: \(C_2(-\varepsilon)\approx C_2(\varepsilon)\), \(a_0\) evenness
- Full-operator TV ratio at \(\varepsilon=0\)
- Block trajectory replication

## Out of scope this run

- Angular detuning
- Sparse peer nested panel (Step D)
- Stage C
