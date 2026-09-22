# How a shared state is described determines whether AI agents synchronize

Research data and code by Takahiro Ezaki, Naoto Imura and Katsuhiro Nishinari.
Version v1.2.0. Associated preprint: https://arxiv.org/abs/2608.06968.

## Offline reproduction

Use Python 3.11 or later. No provider API calls are made by these commands.

```bash
python -m pip install -r requirements-figures.txt
python -B tools/verify_release.py
python -B tools/restore_frozen_results.py
python -B -m analysis.publication.count_reported_responses_20260921
python -B -m analysis.publication.integrate_claude_figures
python -B tools/build_current_figures.py
```

The archive contains 507,112 valid responses used in the reported analyses,
including 25,920 accepted responses from the later Claude controlled-field study.
The analysis excludes duplicates, failed attempts and superseded acquisitions;
the original records of those attempts are retained for auditability.

## Contents

- `circlemap/`: engine, encoders, parsers and model-provider backends.
- `experiments/`: acquisition protocols and analysis runners.
- `runs/`: recorded prompts, responses, stimuli and acquisition metadata.
- `results_frozen/`: analysis inputs and derived tables preserved for verification.
- `analysis/publication/`: current figures, response-count and persistence audits.
- `analysis/plot_main/`, `analysis/plot_supplement/`: figure rendering helpers.
- `analysis/figure_assets/`: generation of numerical figure inputs.
- `figures/publication/`: 44 reference PDF assets for five main and 27 supplementary figures.
- `figures/source_main/`, `figures/plot_supplement/`, `figures/figure_work/`: frozen vector inputs required by the current supplementary rendering pipeline.
- `docs/`: scientific protocols, decision records and response-count audit.

Run verification before regenerating outputs. PDF metadata and fonts can vary
across machines. The current driver regenerates main figures and S27 from data,
and converts retained supplementary vector inputs to the current panel labeling.
Additional supplementary plotting sources are included for reproducibility.

This release consolidates the figure pipeline and uses neutral directory names.
Unused earlier plotting implementations and outdated release instructions have
been removed. Raw acquisition files, accepted responses and reference artwork
are unchanged from v1.1.1. Scientific model identifiers and provider metadata are
retained; they identify the systems actually studied.

The Claude controlled-field experiment is a descriptive follow-up after the
collective reversal was observed. Its response curves do not establish a general
stability criterion. Model family and acquisition date were not independently randomized.

## Citation and license

See CITATION.cff. MIT License. Concept DOI: https://doi.org/10.5281/zenodo.21834781.
Cite the version-specific DOI for an exact snapshot. See docs/RELEASE_v1.2.0.md.
