# Stage B ↔ Stage C prompt contract audit

Date: 2026-07-24  
Verdict: **PASS_prompt_contract_identical**  
Artifact: `analysis/stage_c_artifacts/prompt_contract_audit.json`

## Generation settings

- model match: `True` (`gpt-5.4-mini`)
- temperature match: `True` (`0.7`)
- max_output_tokens match: `True` (`20`)
- max_attempts match: `True`
- prompt_version match: `True` (`response-law-v0.1`)

## Serializer / contract

- Shared builder: `circlemap.representations.build_representation_prompt_from_histogram`
- Stage C reconstructed prompt hash match rate: **1.0000** (400/400)
- Observation-only (no K/t/agent id): `True`
- `PROMPT_VERSION`: `response-law-v0.1`

## Implication

Stay bias is surrogate smoothing / coverage, not Stage B↔C prompt mismatch.
