# Additional experiments

Each folder holds one acquisition or one offline analysis. Typical contents:

- `protocol.json`, `design.json`, `prompts.json`, `tasks.json`: fixed inputs of the acquisition.
- `freeze.json`, `prediction_freeze.json`, `input_freeze.json`: SHA-256 hashes and UTC times recorded before the corresponding step.
- `runs/real/`: append-only journal of every dispatched request and returned response (`attempts.jsonl`), stored states and prompts.
- `analysis_real/`, `analysis/`, `predictions/`: derived tables and frozen forecasts.
- `*.py`: scripts that define the response models, forecasts and analyses.

`collective_forecast/dataset_index.json` maps each acquisition set to its journal and its number of valid responses.
`ORIGINAL_HASHES.json` lists the files edited for release together with their acquisition-time hashes.
`LARGE_FILES.json` lists journals stored gzip-compressed; `tools/restore_large_files.py` restores them.

Synthetic pipeline tests, acquisition runners and report generators are not part of this archive.
The task logic of the runners (descriptions, interventions, parsing and updates) is implemented in
`collective_forecast/protocol.py` and checked there against every stored prompt, response and transition.
The freeze files are local records of chronology and are not public preregistrations.
