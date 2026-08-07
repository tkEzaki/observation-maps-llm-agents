# Matched collective R2 — Claude macroscopic protocol

Date: 2026-07-25  
Status: **frozen for estimate-only; paid acquisition gated**  
Protocol: `experiments/stage_c/protocol_matched_rep_collective_r2_claude_v0_1.json`

## Model / contract (match R1 + GPT collective)

| item | value |
| --- | --- |
| family | anthropic |
| model_id | `claude-haiku-4-5-20251001` |
| prompt_version | `response-law-v0.1` |
| temperature | 0.7 |
| max_output_tokens | 20 |
| max_attempts | 3 |
| parser | strict `parse_social_action` + deterministic rescue (identical to replay/R1) |
| encodings | `moments_m1_m3`, `centers_24_standard`, `intervals_24_decimal6` |

No `latest` alias. System prompt, action contract, integrator match GPT matched collective.

## Physical system

\[
N=17,\quad T=100,\quad
\mathcal R\in\{\mathrm{moments},\mathrm{centers},\mathrm{intervals}\}.
\]

| panel | seeds | \(K\) |
| --- | --- | --- |
| core (GPT-matched) | seed_index \(0\ldots5\) | \(\{0, 0.08, 0.15\}\) |
| held-out | seed_index \(6\ldots9\) | \(\{0.08, 0.15\}\) |

Physical IC formula (identical to GPT matched collective):

```text
init_seed = base_seed + 1000*N + 10*seed_i + round(|K|*1000)
theta0 ~ Uniform(-π, π) via RNG(init_seed)
omega = linspace(-0.05, 0.05, N)
```

`base_seed = 2026072420`. Legacy cells must match `analysis/matched_rep_collective/match_manifest.json` hashes.

## LLM sampling seeds (Claude-independent)

```text
sha256(family | model_id | representation | init_seed | K | agent_id | t)
→ uint63 sample_seed
```

Physical ICs are GPT-matched; API draws are family-specific.

## Acquisition schedule

Cannot shuffle tasks across time within a trajectory. Execute **seed × K matched triplets** with round-robin representations:

```text
for cell in frozen_task_order:          # seed × K
  init identical (θ0, ω) for all 3 reps
  for t in 0..T-1:
    for R in (moments, centers, intervals):   # round-robin
      call N agents for R at state_R(t)
      update state_R ← step(...)
```

Purpose: reduce representation × wall-clock confounds and provider drift skew.

Frozen order: `analysis/matched_rep_collective/r2_macro/task_manifest.json`.

## Call budget

| panel | calls |
| --- | ---: |
| core \(6\times3K\times3\mathcal R\times17\times100\) | 91,800 |
| held-out \(4\times2K\times3\mathcal R\times17\times100\) | 40,800 |
| **total** | **132,600** |

## Phenotypes (GPT v0.2 — frozen unchanged)

| label | rule |
| --- | --- |
| `polar_locked` | \(r_1(T)\ge 0.9\) |
| `high_r2_nonpolar` | \(r_2(T)\ge 0.5\) and \(r_2(T)>r_1(T)\) |
| `partial_polar_order` | \(0.35\le r_1(T)<0.9\), not high-\(r_2\) |
| `low_polar_active` | \(r_1(T)<0.35\) |

Also record: sustained lock, mean/final \(r_1\), final \(r_2\), \(Q_2=r_2-r_1\), activity, social torque, collective frequency. First passage \(r_1>0.9\) is **not** called locking time.

## Validity / failure policy

- Retain raw responses for every attempt
- Separate strict vs rescued parse
- **Never** replace unrecoverable with `stay`
- **Never** write a step with only a subset of agents updated
- **Never** replace a failed seed with a new seed
- Incomplete positive-\(K\) matched cube → reacquire same (seed, \(\mathcal R\), \(K\))

## \(K=0\) negative control (engine gate)

On core 6 seeds, require

\[
\max_{R,R',s,t,m\in\{1,2,3\}} \lvert r_m^R - r_m^{R'} \rvert < 10^{-12}.
\]

Activity/action bias may differ across \(\mathcal R\). Failure ⇒ abort as **engine / matching failure** (not a scientific null).

## Paid gates

1. Freeze manifests + hashes written  
2. `--estimate-only` reviewed (`estimate_only.json`)  
3. Human review checklist signed  
4. `auth_go.json` → `paid_authorized=true`  
5. Explicit `--yes` on paid launch  

Until then: protocol/analysis freeze **GO**; paid acquisition **NO-GO**.
