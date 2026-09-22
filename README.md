# How a shared state is described determines whether AI agents synchronize

Research data and code by **Takahiro Ezaki, Naoto Imura and Katsuhiro Nishinari**.
**Version v1.1.1**, September 22, 2026. Earlier manuscript:
[arXiv:2608.06968](https://arxiv.org/abs/2608.06968).

This release adds the September Claude controlled-field follow-up, current figure
scripts, and an audit of **507,112 valid responses in the reported analyses**.
See [release notes](docs/RELEASE_v1.1.1.md).

## Offline verification and reproduction

Use Python 3.11 or later. On Windows, use a short checkout path or Git long-path support.

```bash
git clone https://github.com/tkEzaki/observation-maps-llm-agents.git
cd observation-maps-llm-agents
git checkout v1.1.1
python -m pip install -r requirements-figures.txt
python -B tools/verify_release.py
python -B tools/restore_frozen_results.py
python -B -m analysis.sciadv_revision.count_reported_responses_20260921
python -B -m analysis.sciadv_revision.integrate_claude_figures
python -B tools/build_current_figures.py
```

These commands make **no provider API calls**. Verify the manifest before regeneration:
PDF metadata and font rendering can differ across machines. The shipped
`figures/sciadv_current/` PDFs are the reference artwork. The current driver redraws
the main figures and S27 and converts historical SI frozen vector inputs to uppercase.
SI plotting sources are in `analysis/natcomm_si_opus/`; the current Claude SI
audit and plots are in `analysis/sciadv_revision/integrate_claude_figures.py`.

To rerun the complete Claude bootstrap analysis offline, in a disposable clone:

```bash
python -B -m experiments.stage_b_response_law.run_claude_kappa_v2 --mode analyze
```

This rewrites derived run summaries. Preserve the original attempt journals.
Do not run the live `execute_claude_kappa_v2` acquisition driver for offline reproduction.

## Files

| Path | Contents |
|---|---|
| `circlemap/` | Engine, encoders, parsers and provider backends |
| `experiments/stage_b_response_law/` | Protocols and acquisition/analysis runners |
| `runs/` | Original acquisitions plus Claude v1 preflight and v2 follow-up |
| `runs/claude_kappa_v2/anthropic/` | Two blocks, attempt journals, accepted traces, stimuli, task manifests and analysis |
| `results_frozen/` | Original frozen artifacts plus current figure inputs and Claude snapshots |
| `analysis/sciadv_revision/` | Current figures, persistence and response-count audits |
| `analysis/natcomm_figures_opus/`, `analysis/natcomm_si_opus/` | Main and SI source modules |
| `figures/sciadv_current/` | 44 PDF assets: five main and 27 supplementary figures, some in multiple parts |
| `docs/response_count_audit.json` | Count definition, groups, exclusions and source hashes |
| `RELEASE_MANIFEST.json` | SHA-256 checks for raw data, frozen results and current artwork |
| `DATA_ARCHIVE_MANIFEST.json` | Acquisition and large-artifact completeness manifest |

The original studies comprise 15 acquisition groups. The new concentration sweep
adds **25,920 accepted responses**. Not every archived line contributes to the
reported count: duplicates, failed attempts and superseded acquisitions are excluded.

The Claude experiment is a descriptive follow-up after observing the collective
reversal. Controlled-field curves alone do not establish a general stability
criterion. Model family and acquisition date were not independently randomized.

The original version remains available at tag v1.0.0. Its old layout uses
`tools/build_all_figures.py`; use the current driver above for this release.
Historical instructions are retained in `docs/README_v1.0.0.md`.

## Citation and license

See `CITATION.cff`. The **MIT License** is retained. The Zenodo concept DOI is
[10.5281/zenodo.21834781](https://doi.org/10.5281/zenodo.21834781); cite the
version-specific DOI associated with the release for an exact snapshot.
This release does not imply journal acceptance.
