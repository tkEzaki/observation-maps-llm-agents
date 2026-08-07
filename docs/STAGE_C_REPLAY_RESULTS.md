# Collective-field replay — prospective results

Date: 2026-07-24  
Status: **COMPLETE**  
Session: `runs/collective_replay/replay-v0.1_a3b41651b3c4/20260724T005703Z`  
Smoke: **PASS** (30/30 Stage-C field prompt hashes)  
Cost: **\$0.269** (1152/1152 valid)

## Branch verdict

**A — replay confirms near-zero stay on Stage-C-realized fields; v1 over-smooths stay.**

All **30** collective-sourced fields (neg / pos / zero) have

\[
p_{\mathrm{stay}}^{\mathrm{replay}}=0
\]

(32 responses each across 2 blocks). Locked v1 predictions still give
\(\hat p_{\mathrm{stay}}\approx0.10\)–\(0.17\).

This matches Stage C teacher-forced low stay and is **not** an OOD / prompt issue.

## By bucket

| bucket | n | \(\bar p_{\mathrm{stay}}^{\mathrm{obs}}\) | \(\bar{\hat p}_{\mathrm{stay}}\) | \(\overline{\Delta p_{\mathrm{stay}}}\) | \(\overline{e_{\mathrm{TV}}}\) |
| --- | ---: | ---: | ---: | ---: | ---: |
| neg | 12 | **0.000** | 0.166 | −0.166 | 0.391 |
| pos | 12 | **0.000** | 0.103 | −0.103 | 0.295 |
| zero | 6 | **0.000** | 0.121 | −0.121 | 0.409 |
| stage_b_anchor | 6 | 0.500 | 0.400 | +0.100 | 0.365 |

Anchors still show the intended abstention structure:

- exact 8:8 / sparse antipodal equal → \(p_{\mathrm{stay}}=1\)
- small imbalance 9:7 → \(p_{\mathrm{stay}}=0\)

So the microscopic antipodal stay gate is real; v1 wrongly **exports that stay mass into active collective fields**.

## Implication for v2

Proceed to `moments_bundle_v2` with a **regime-aware stay gate**:

- exact / near-antipodal abstention regime (preserve anchors)
- active collective-field regime (low stay)

Do **not** apply a global \(p_{\mathrm{stay}}\mapsto c\,p_{\mathrm{stay}}\) rescale.
Do **not** overwrite v1.

## Artifacts

| file | role |
| --- | --- |
| `prompt_hash_smoke.json` | pre-paid wiring proof |
| `v1_predictions_locked.json` | prospective \(\hat{\mathbf p}^{v1}\) |
| `replay_vs_v1_prospective.csv` | per-field \(e_{\mathrm{TV}}\), \(\Delta p_{\mathrm{stay}}\) |
| `replay_prospective_summary.json` | rollup |

## Next

1. ~~Build `moments_bundle_v2`~~ → `docs/STAGE_3B_MOMENTS_V2.md`
2. ~~Grouped CV / AD / hash-lock v2~~
3. ~~Compare v1 / v2 dense-\(K\) envelopes~~ → `docs/STAGE_C_DENSE_K_V1_VS_V2.md`
4. Explicit GO → Stage C v0.2 (surrogate = v2; shared attenuated \(K\) grid)
