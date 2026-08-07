# Frozen results

Every analysis artefact the manuscript reports, exactly as it stood when the
numbers were written.

**No script under `analysis/` or `experiments/` writes anything here.** Those
scripts write to `analysis/`, which is where a re-run lands. This directory is
the reference the re-run is compared against.

```bash
python tools/restore_frozen_results.py   # copy into analysis/ so figures build
python tools/verify_frozen_results.py    # diff analysis/ back against this
```

`MANIFEST.json` gives a `sha256` for every file. The digest is taken after the
normalisation in `tools/verify_frozen_results.py`, which replaces personal
absolute paths with `<PROJECT_ROOT>` and `<HOME>` and folds CRLF to LF, so the
same digest is obtained on any platform.

Artefacts larger than 2 MB, and the raw acquisition traces, are hosted in the
data archive instead; see `DATA_ARCHIVE_MANIFEST.json`.

## The load-bearing artefacts

| File | What the manuscript takes from it |
|---|---|
| `analysis/matched_rep_collective/replay_primary/decision.json` | The identical-field replay: the presented-encoding effect, the two source tests, the pairwise distances and the noise floor |
| `analysis/matched_rep_collective/omap_primary/decision.json` | The same-task-information control |
| `analysis/matched_rep_collective/slc_primary/decision.json` | The serialization-length control |
| `analysis/matched_rep_collective/sbc_primary/decision.json` | The serialization-binding control |
| `analysis/matched_rep_collective/r2_macro/r2_inference/primary_inference.json` | The Claude matched-macro replication |
| `analysis/complex_kernel_*/complex_kernel_analysis.json` | The microscopic response sweeps and their Fourier coefficients |
| `analysis/stage3a_artifacts/`, `analysis/centers_branch/`, `analysis/intervals_branch/` | The surrogate compressibility and transportability analysis |

Two defects were found in the code that produced some of these and were fixed;
the affected artefacts were then re-frozen with the current code.
`docs/STATS_AUDIT_MEMO.md` records what moved.
