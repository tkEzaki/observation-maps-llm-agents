# Collective-field replay panel — cost card

Date: 2026-07-24  
Status: **PAID COMPLETE** (\$0.269; 1152/1152 valid; branch A)  
Results: `docs/STAGE_C_REPLAY_RESULTS.md`  
Depends on: prompt audit **PASS** (`docs/STAGE_C_PROMPT_CONTRACT_AUDIT.md`)

## Purpose

Re-measure single-agent response law on **Stage-C-realized** low-risk fields
with independent seeds (do **not** reuse Stage C actions as labels). Keeps
Stage C v0.1 as a prospective test of `moments_bundle_v1`.

## Design

| item | value |
| --- | ---: |
| Representation | `moments_m1_m3` only |
| Fields | **36** (12 neg / 12 pos / 6 K=0 / 6 Stage B anchors) |
| Responses / field / block | 16 |
| Blocks | 2 |
| **Expected calls** | **1,152** |
| Rough cost (`gpt-5.4-mini`) | **≈ \$0.28** |

Artifacts:

- Fields: `analysis/stage_c_artifacts/replay_panel_v1/replay_fields_v1.json`
- Protocol: `experiments/stage_c/protocol_collective_replay_v0_1.json`

## Preconditions

1. [x] Stage B ↔ Stage C prompt contract audit PASS  
2. [x] Explicit user authorization (Conditional GO for replay panel)  
3. [x] Runner wiring smoke: 30/30 Stage-C fields
   `hash(replay)=hash(Stage C observation)`;
   artifact `analysis/stage_c_artifacts/replay_panel_v1/prompt_hash_smoke.json`  
4. [x] Estimate: **\$0.3905** base / \$1.17 retry ceiling (`--estimate-only`)  
5. [x] v1 predictions locked prospectively before acquisition  
6. [x] v1 bundle left untouched; v2 only after replay

## After replay

Build `moments_bundle_v2` from fresh response-law data (no constant \(p_{\mathrm{stay}}\) rescaling). Compare v1 vs v2 dense-\(K\) envelopes before Stage C v0.2.

**Done:** v2 checklist-frozen; rich dense compare shows activity lift;
v0.2a design locked (`docs/STAGE_C_V0_2_PROTOCOL.md`). Paid still needs
`--estimate-only` + explicit GO.
