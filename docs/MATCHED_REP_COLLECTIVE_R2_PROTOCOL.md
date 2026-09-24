# Matched collective R2 — Claude macroscopic protocol

Date: 2026-07-25  
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


## Selection rationale (not random)



Claude Haiku (`claude-haiku-4-5-20251001`) is the second family by **targeted replication** from R1 micro evidence on the frozen 48-field panel:



- valid rate 1.000

- target operator effect \(p=0.0002\)

- pairwise \(d_{\mathrm{TV}}\) comparable to GPT

- between/within ratio \(20.4\times\)



This is **not** a random draw among all available families.



## Primary scientific question



> Does GPT’s observation-representation selection of collective phenotype replicate in an independently developed Claude family under matched physical dynamics?



GPT positive-\(K\) outcome (6 seeds):



| representation | polar lock |

| --- | ---: |

| moments | 6/6 |

| centers | 0/6 |

| intervals | 0/6 |



R2 tests the **Claude-only** representation effect. Numerical match to GPT and exact ranking are **not** primary requirements.



## Claim hierarchy



| Tier | Meaning |

| --- | --- |

| Macro representation effect | Claude: representation changes collective outcome |

| Qualitative phase replication | ≥1 representation locks; another does not |

| GPT phenotype-map replication | moments lock → centers partial → intervals high-\(Q_2\) hierarchy |



Tiers 1–2 are the primary scientific criteria. Tier 3 is optional.



## Design summary



- Family: Anthropic / `claude-haiku-4-5-20251001` (R1-identical fixed ID; no `latest`)

- \(N=17\), \(T=100\); \(\mathcal R\in\{\mathrm{moments},\mathrm{centers},\mathrm{intervals}\}\)

- Core 6 seeds: \(K\in\{0,0.08,0.15\}\) (same physical ICs as GPT matched collective)

- Held-out 4 seeds: \(K\in\{0.08,0.15\}\) only

- Drop \(K=-0.15\) (secondary torque-sign; not needed for positive-\(K\) phase gate)

- Budget: **132,600** calls; protocol ceiling **\$175**



