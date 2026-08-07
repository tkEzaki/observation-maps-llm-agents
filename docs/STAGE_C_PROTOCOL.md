# Stage C — selection pilot protocol

Date: 2026-07-24  
Status: **v0.1 complete** (see `docs/STAGE_C_RESULTS_V0_1.md`)  
Cost card: `docs/STAGE_C_COST_CARD.md`

## Amendments vs SYSTEM_SPEC §Stage C

| SYSTEM_SPEC | Stage C v0.1 |
| --- | --- |
| \(N\in\{8,16\}\) | **\(N\in\{9,17\}\)** (peer-matched) |
| two-body first | **deferred** (peer=1 unsupported) |
| \(T=100\) | **\(T=40\)** selection; expand later |
| encodings unspecified | **moments only** (hash-locked) |

## Bundle

`analysis/stage3a_artifacts/moments_bundle_v1/`  
Production model: `kernel_hurdle`. OOD: peer hard-gate + calibrated \(R\).

## Commands

```powershell
# Offline surrogate predictions (locked bundle)
py -m analysis.stage_c.run_surrogate_stage_c
py -m analysis.stage_c.run_surrogate_stage_c --n-steps-override 100

# Cost estimate
py -m experiments.stage_c.run_collective --estimate-only `
  --backend openai --model gpt-5.4-mini `
  --input-price-per-million 0.75 --output-price-per-million 4.5

# Paid (after estimate review)
py -m experiments.stage_c.run_collective --backend openai --model gpt-5.4-mini `
  --input-price-per-million 0.75 --output-price-per-million 4.5 `
  --env-file "<HOME>\Dropbox\research_current\gen_ai_logi\rq1\.env" --yes
```

## Outputs

- Surrogate: `analysis/stage_c_artifacts/`
- LLM: `runs/stage_c/stage-c-v0.1_<hash>/<stamp>/`
