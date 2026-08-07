# Acquisition traces

This directory is intentionally empty in the repository.

Every model call is recorded in a `trace.jsonl` under
`runs/<experiment>/<protocol-hash>/<timestamp>/`, together with the resolved
configuration, the protocol, and a session summary. The complete set is about
370 MB and is hosted in the data archive rather than in git; see
`DATA_ARCHIVE_MANIFEST.json` for the DOI and the file list.

The traces are needed only to re-run an *acquisition-level* analysis from raw
responses. Every number and figure in the paper can be reproduced from the
frozen artefacts in `results_frozen/` without them.
