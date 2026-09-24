# Stage C v0.2 — prospective validation protocol

Date: 2026-07-24  

## Scientific purpose

Not “does it lock again,” but:

> Does replay-informed `moments_bundle_v2` prospectively improve the
> activity / sync-speed / collective-response magnitude that v1 missed?

Three-way comparison on every cell:

1. frozen v1 prediction  
2. frozen v2 prediction (pre-registered online surrogate)  
3. new LLM collective outcome  

v1 remains the permanent v0.1 prospective baseline; v0.2’s prediction model is **v2**.

## Primary endpoints

| endpoint | why |
| --- | --- |
| \(r_1(T)\) | order |
| time-averaged \(r_1\) | path |
| \(t_{0.5}\), \(t_{0.9}\) | sync speed (v2 may show here even if \(K_{\mathrm{cross}}\) matches) |
| **mean activity** | direct stay-gate target — **required primary** |
| mean social torque | response magnitude |
| predictive interval coverage | calibration |

## Secondary endpoints

\(r_2,r_3\), \(\Omega_{\mathrm{coll}}\), locking fraction, teacher-forced one-step
stay/direction, trajectory risk.

## Two-stage design

### v0.2a — transition validation 

Protocol: `experiments/stage_c/protocol_stage_c_v0_2a.json`

| item | value |
| --- | --- |
| \(N\) | **17** only |
| \(K\) | **0.08, 0.10, 0.12, 0.15** |
| \(T\) | 100 |
| Seeds | 4 (new; not v0.1 seeds) |
| Calls | \(17\times4\times100\times4=\) **27,200** |

Unmeasured transition \(K\) + longer \(T\) reduce field-distribution proximity
to the Stage C v0.1 → replay training slice.

### v0.2b — size / negative replication

Protocol: `experiments/stage_c/protocol_stage_c_v0_2b.json`  

| item | value |
| --- | --- |
| \(N=9\) | \(K\in\{-0.15,0,0.06,0.08,0.10,0.15\}\) |
| \(N=17\) | \(K=-0.15\) only |
| \(T\) | 100 |
| Seeds | 4 |
| Calls | **28,400** |

## Full grid (reference)

| \(N\) | \(K\) |
| ---: | --- |
| 9 | 0, 0.06, 0.08, 0.10, 0.15 [, −0.15] |
| 17 | 0, 0.08, 0.10, 0.12, 0.15 [, −0.15] |

## Naming

| object | name |
| --- | --- |
| v1 | prospectively frozen Stage C v0.1 baseline (permanent) |
| v2 before checklist | hash-locked selection-grade v2 candidate |
| v2 after checklist pass | prospectively frozen Stage C v0.2 baseline |
