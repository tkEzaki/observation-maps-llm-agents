# Collective forecasts: procedure, checks and figures

Offline code for the additional experiments in `independent_studies/`. No provider API calls are made.

```bash
python -m pip install -r requirements-collective.txt
python -B tools/restore_frozen_results.py
python -B collective_forecast/verify_counts.py
python -B collective_forecast/verify_protocol.py
python -B collective_forecast/response_model.py
python -B collective_forecast/recompute_from_raw.py
python -B collective_forecast/build_figures.py
python -B collective_forecast/build_fig1.py
python -B collective_forecast/build_figS5.py
```

- `protocol.py`: reference implementation of both tasks, the state descriptions, the reflection and label
  interventions and the update rules.
- `verify_protocol.py`: rebuilds every stored prompt, re-parses every response and replays every state
  transition of the collective experiments with `protocol.py`.
- `response_model.py`: response models and closed-loop forecasts. Refits the primary model and regenerates the
  frozen forecasts of the 60-step and 200-step cohorts from the frozen coefficients, designs and seeds.
- `recompute_from_raw.py`: recomputes the main reported numbers from stored states, raw responses and frozen
  forecasts and compares them with the reported values (`source_data/recomputed_from_raw.csv`).
- `verify_counts.py`, `dataset_index.json`: acquisition sets, journals and valid-response counts.
- `build_figures.py`, `build_fig1.py`, `build_figS5.py`: figures written to `figures/collective_forecast/`.
- `mechanism_model.py`: mean-field map, Jacobian, linear-noise approximation and flat-region model of the frozen response model.
- `source_data/`: numerical values behind every panel.
- `FIGURE_INPUTS.json`: SHA-256 of every file read by `build_figures.py`.

Requirements: Python 3.11 or later with numpy, scipy, pandas and matplotlib.
