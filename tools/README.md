# Tools

| Script | Purpose |
|---|---|
| `restore_frozen_results.py` | Copy `results_frozen/` into the analysis tree. Run once after cloning. `--force` overwrites, `--dry-run` lists. |
| `verify_frozen_results.py` | Compare the analysis tree against `results_frozen/`. Reports differing JSON keys individually. Exit status 1 if anything differs. |

| `build_all_figures.py` | Rebuild every figure. Takes optional module-name fragments, e.g. `figS16`. |
| `check_repo_hygiene.py` | Refuse to publish if the tree holds an API key, a personal absolute path, an oversized file or a `__pycache__`. |
| `scrub_paths.py` | Replace personal absolute paths with `<PROJECT_ROOT>` and `<HOME>`. Applied to the published copy; `verify_frozen_results.py` applies the same rule before comparing, so a local re-run still diffs cleanly. |
