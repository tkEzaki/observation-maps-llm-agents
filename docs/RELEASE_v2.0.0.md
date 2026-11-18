# v2.0.0

Adds the additional experiments under `independent_studies/` and the offline tools under `collective_forecast/`:
frozen response-model forecasts of new populations, reflection interventions with independent and shared coins,
fixed-state one-step tests and the resource-allocation task. Reference figures are in `figures/collective_forecast/`.

`collective_forecast/` contains a reference implementation of the procedure (`protocol.py`), the response models and
closed-loop forecasts (`response_model.py`) and scripts that replay the recorded experiments, regenerate the frozen
forecasts and recompute the reported numbers from the raw records.

Raw acquisitions, numerical results and reference figures from v1.2.1 are retained. Package documentation, metadata and dependency instructions are updated.
