# Dense \(K\) rich compare: v1 vs v2 (\(T=100\))

Date: 2026-07-24  
Status: **complete (no API)**  
Artifacts:

- `analysis/stage3a_artifacts/dense_k_rich_runs_v1_vs_v2.csv` (864 runs)
- `analysis/stage3a_artifacts/dense_k_rich_v1_vs_v2.csv`
- `analysis/stage3a_artifacts/dense_k_rich_deltas_focus.csv`

Seeds / cell: 12. Metrics beyond crossing:
\(P(r_1>0.9)\), \(E[r_1]\), \(E[t_{0.5}]\), \(E[t_{0.9}]\), \(E[A]\), \(E[\tau]\).

## Headline

**Crossing \(K\) is unchanged, but activity rises under v2** — especially on the
N=17 transition neighborhood that v0.2a targets.

| \(N=17\) \(K\) | \(\Delta A\) (v2−v1) | \(A_{\mathrm{v1}}\) | \(A_{\mathrm{v2}}\) | \(\Delta E[r_1]\) | \(\Delta E[t_{0.9}]\) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| −0.15 | **+0.062** | 0.870 | 0.932 | −0.018 | — |
| 0.00 | **+0.076** | 0.885 | 0.962 | +0.000 | — |
| 0.08 | **+0.097** | 0.874 | 0.971 | +0.009 | −3.4 |
| 0.10 | **+0.093** | 0.870 | 0.963 | +0.035 | +10.2 |
| 0.12 | **+0.116** | 0.866 | 0.982 | −0.019 | −8.8 |
| 0.15 | **+0.105** | 0.871 | 0.976 | −0.007 | −6.1 |

(`—` / noisy \(t_{0.9}\): some seeds never hit 0.9 within \(T=100\).)

## Interpretation

1. Binary locking / \(K_{\mathrm{cross}}\) alone **hides** the stay-gate fix.
2. Offline v2 raises mean activity by ~0.09–0.12 on transition \(K\) at \(N=17\).
3. Final \(r_1\) deltas are small / seed-noisy; **arrival time and activity are
   the right primaries for paid v0.2a**.
4. This does **not** prove LLM collectives will match v2 — only that the
   surrogate envelope now moves on the scientifically intended axes.

## N=9 focus (abbrev.)

See `dense_k_rich_deltas_focus.csv`. Same pattern: activity up; \(r_1\) quiet.

## Implication for Stage C v0.2a

Design GO stands. Paid v0.2a should preregister:

- primary: activity, \(t_{0.5/0.9}\), torque, \(r_1\), coverage  
- cells: \(N=17\), \(K\in\{0.08,0.10,0.12,0.15\}\), \(T=100\), 4 seeds  

Do not judge v0.2a success by final-\(r_1\) alone.
