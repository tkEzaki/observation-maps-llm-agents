# Observation maps select collective outcomes in language-model agents

Code, frozen results, figure sources and acquisition traces for

> Ezaki, T., Imura, N. & Nishinari, K. *Same physical state, different collective dynamics:
> state encodings select synchronization outcomes in language-model agents.*
> TODO_JOURNAL (TODO_YEAR). doi:TODO_PAPER_DOI

A population of phase oscillators is driven by a pretrained language model.
Each agent sees only a text description of its peers' relative phases and
chooses to advance, stay or retard; a deterministic engine applies the
coupling. The only thing that differs between conditions is how the same
physical state is encoded into text. That choice, which we call the
**observation map**, decides whether the population synchronizes.

---

## Quick start

```bash
git clone https://github.com/tkEzaki/observation-maps-llm-agents
cd observation-maps-llm-agents
python -m pip install -r requirements.txt

python tools/restore_frozen_results.py   # frozen artefacts -> analysis/
python tools/build_all_figures.py        # rebuild the figures
python tools/verify_frozen_results.py    # nothing should differ yet
```

No API key is required for any of this. The model calls were made once; their
outcomes are frozen in this repository.

---

## What is where

| Path | Contents |
|---|---|
| `circlemap/` | The library. Deterministic engine, observation-map encoders, prompt contract, backends, metrics. |
| `experiments/` | Acquisition runners and the versioned, hash-locked protocol JSON for each experiment. |
| `analysis/` | Statistical analyses and figure sources. **Code only** in git; its outputs are generated. |
| `results_frozen/` | **The frozen results.** A byte-for-byte copy of every analysis artefact the paper reports. No pipeline script writes here. |

| `figures/` | The rendered display items, as PDF at the final 180 mm width. |
| `docs/` | Protocols, cost cards, decision records, and the memos cited in the Methods. |
| `runs/` | Per-run state and the complete raw acquisition traces (`trace.jsonl`) for all 15 acquisition groups. |
| `tools/` | Restore, verify, figure driver, hygiene check. |

### Why `results_frozen/` exists

Running an analysis script overwrites its output under `analysis/`. If the
published numbers lived only there, a single exploratory re-run would silently
replace them and nothing would notice.

`results_frozen/` is therefore a separate, immutable copy: **no script in
`analysis/` or `experiments/` ever writes to it.** `tools/restore_frozen_results.py`
copies it into place so the figures build; `tools/verify_frozen_results.py`
compares the working tree back against it and prints, key by key, what moved.

```bash
python tools/verify_frozen_results.py --glob '*decision.json'
```

Both directions of the comparison are useful. If you re-run an analysis and
nothing differs, the pipeline reproduces the paper. If something differs, the
tool tells you which JSON key changed and by how much, which is exactly the
check that caught the two defects recorded in `docs/STATS_AUDIT_MEMO.md`.

Comparison normalises two things that are not scientific differences: personal
absolute paths, replaced by `<PROJECT_ROOT>` and `<HOME>` in the published copy
(see `tools/scrub_paths.py`), and CRLF line endings.

---

## Reproducing the results

### From the frozen artefacts (no data archive, no API key)

```bash
python tools/restore_frozen_results.py
python tools/build_all_figures.py
```

### From the raw responses

Some analyses re-derive their statistics from the large microscopic sweeps
rather than from a stored summary, which is the point of them: they audit the
acquisition rather than restate its conclusion. Nothing extra has to be
downloaded: the raw responses are in `runs/**/trace.jsonl` in this repository.

`DATA_ARCHIVE_MANIFEST.json` records a `sha256` and a byte count for every
trace file and for the eight large analysis artefacts, so a clone can be
checked for completeness with `tools/verify_frozen_results.py --traces`.

### Re-running an analysis end to end

```bash
python -m analysis.matched_rep_collective.run_replay_primary_inference
python tools/verify_frozen_results.py --glob '*replay_primary/decision.json'
```

Every random operation is seeded, so a re-run on the same inputs reproduces the
frozen artefact exactly.

### Placing new model calls

Acquisition is the only part that spends money and the only part that needs
credentials.

```bash
python -m pip install -r requirements-acquisition.txt
cp .env.example .env        # then fill in the provider keys you need
```

Each paid runner refuses to start unless an authorization sidecar next to it
records an explicit human go, together with the cost card it was approved
against. Those sidecars are kept in the repository as provenance records; they
are what makes the ordering of design and acquisition auditable.

---

## Figure coverage

Every figure module is self-contained in this repository: the raw acquisition
traces the four audit figures read (`figS02`, `figS03`, `figS04`, `figS06`) are
tracked here rather than held in a separate archive, so a clean `restore`
followed by `build_all_figures` rebuilds all 28 modules and reproduces the 42
shipped PDFs.

Figure numbers in the manuscript and the script names do not always agree: the
supplementary figures were reordered into narrative order late, and the scripts
keep their original numbers. The complete mapping is:

| Manuscript | Released file | Manuscript | Released file |
|---|---|---|---|
| S1  | `figS01_engine_validation` | S14 | `figS17_three_family_replication` |
| S2  | `figS02a/b_observation_maps` | S15 | `figS19a-c_claude_trajectories` |
| S3  | `figS08a-c_gpt_trajectories` | S16 | `figS20_claude_r2_gates` |
| S4  | `figS09_phenotype_robustness` | S17 | `figS21_gpt_claude_reversal` |
| S5  | `figS10_negative_coupling` | S18 | `figS18a/b_omap` |
| S6  | `figS03_offset_design_aliasing` | S19 | `figS22_serialization_length` |
| S7  | `figS04_representation_screen` | S20 | `figS23_serialization_binding` |
| S8  | `figS05_transmutation_map` | S21 | `figS_surrogate_logic_schematic` |
| S9  | `figS06_antipodal_imbalance` | S22 | `fig6_compressibility_transportability` |
| S10 | `figS07_sparse_finite_peer` | S23 | `figS11a/b_moments_surrogate` |
| S11 | `figS14_replay_field_audit` | S24 | `figS12_centers_stop_rule` |
| S12 | `figS15a-f_replay_fields` | S25 | `figS13_intervals_stop_rule` |
| S13 | `figS16_replay_robustness` | | |

Two quantities that a reader might want without running the rendering layer
also have a standalone module:

| Module | What it reports |
|---|---|
| `analysis/matched_rep_collective/phenotype_rule_sensitivity.py` | Lock fraction against the final-`r1` threshold and against the trailing-window length, with the exact paired sign test recomputed at every swept threshold. Checks the classification at the prespecified setting against the frozen table before sweeping. Writes a JSON artefact; `figS09` draws the same sweeps. |
| `analysis/stage3/spearman_rpre_vs_etv.py` | Spearman rho between the pre-acquisition predictive risk and the observed response error for the moments pilot, with tertile medians. `figS11` plots the same quantity. |

---

## Notes for a reader checking the statistics

- **Repeated model calls are not replicates.** They estimate a stochastic
  action distribution. The independent unit is the physical seed for the
  collective experiments, the physical field for the replay and presentation
  controls, and the complete run for prospective surrogate evaluation. The
  Methods open with a table that states this per analysis.
- **Prespecification is internal.** Protocols, endpoints and stopping rules
  were versioned and hash-locked in this repository before the acquisition they
  govern, but were not deposited with a public registry. `docs/` holds those
  records with their dates.
- **Two implementation defects were found and fixed**, and the affected
  artefacts were re-frozen with the current code. `docs/STATS_AUDIT_MEMO.md`
  records what changed and by how much; nothing changed a conclusion.

---

## What the traces contain

Each line of `runs/**/trace.jsonl` is one model call: the task and field
identifiers, the physical hash of the field, the encoding, the acquisition
block, the sampling seed, the SHA-256 of the prompt, the raw response text,
which parser stage read it, token counts and elapsed time.

The prompt *body* is not stored per call --- only its hash --- because every
prompt is a deterministic function of the physical field and the encoder, both
of which are in the repository; `analysis` regenerates a prompt and checks it
against the stored hash. The raw responses themselves are the one-field JSON
objects the contract asks for, a few tens of bytes each.

The three `*_mock` acquisition groups are pipeline smoke tests against a
synthetic backend and are not released; no result in the paper depends on them.

---

## Requirements

Python 3.11 or newer, with `numpy`, `pandas` and `matplotlib`. The analyses use
no other third-party libraries: the permutation tests, bootstraps, exact sign
tests and rank correlations are implemented directly against `numpy`, so there
is no hidden dependence on a statistics package's defaults.

Developed and frozen with Python 3.11, numpy 2.4.6, pandas 3.0.3. The figures
in the manuscript were rendered with matplotlib 3.10.9; the rendering version
is recorded in the metadata of every released PDF.

---

## Citation

See `CITATION.cff`.

## Archiving

The released contents are also deposited on Zenodo with a permanent DOI; the
GitHub repository is the working copy and the Zenodo record is the citable
archive. The two hold the same files, raw acquisition traces included.

## Licence

MIT. See `LICENSE`.

The frozen results under `results_frozen/` and the model outputs they derive
from are released under the same terms; the underlying provider models are not
redistributed here.
