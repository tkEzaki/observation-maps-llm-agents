"""Freeze reproducible source SHAs for matched 3×3 replay (no git required)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "matched_rep_collective"
FREEZE = OUT / "replay_hash_freeze_v0_1.json"

RUNNER = (
    ROOT / "experiments" / "stage_c" / "run_matched_rep_collective_replay.py"
)
SELECTOR = OUT / "select_replay_fields.py"
ENCODER = ROOT / "circlemap" / "representations.py"
PARSER = ROOT / "circlemap" / "response.py"
BACKENDS = ROOT / "circlemap" / "backends.py"
OBSERVATION = ROOT / "circlemap" / "observation.py"
COSTING = ROOT / "circlemap" / "costing.py"
PROTOCOL = (
    ROOT
    / "experiments"
    / "stage_c"
    / "protocol_matched_rep_collective_replay_v0_1.json"
)
PRIMARY = OUT / "analyze_replay_primary_3x3.py"


def _sha_concat(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(
        paths, key=lambda p: str(p.relative_to(ROOT)).replace("\\", "/")
    ):
        if not path.exists():
            raise FileNotFoundError(path)
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def compute_source_hashes() -> dict:
    dependency = [
        RUNNER,
        SELECTOR,
        ENCODER,
        PARSER,
        BACKENDS,
        OBSERVATION,
        COSTING,
        PROTOCOL,
    ]
    if PRIMARY.exists():
        dependency.append(PRIMARY)
    return {
        "runner_source_sha256": _sha_concat([RUNNER]),
        "encoder_sources_sha256": _sha_concat([ENCODER, OBSERVATION]),
        "selector_source_sha256": _sha_concat([SELECTOR]),
        "dependency_snapshot_sha256": _sha_concat(dependency),
        "files": {
            "runner": str(RUNNER.relative_to(ROOT)).replace("\\", "/"),
            "encoder": str(ENCODER.relative_to(ROOT)).replace("\\", "/"),
            "selector": str(SELECTOR.relative_to(ROOT)).replace("\\", "/"),
            "parser": str(PARSER.relative_to(ROOT)).replace("\\", "/"),
            "backends": str(BACKENDS.relative_to(ROOT)).replace("\\", "/"),
            "protocol": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        },
        "runner_commit_hash": None,
        "note": "Git unavailable; source-file aggregates used for reproducibility.",
    }


def main() -> int:
    hashes = compute_source_hashes()
    out = OUT / "replay_source_hashes_v0_1.json"
    out.write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    freeze.update(
        {
            "runner_commit_hash": None,
            "runner_source_sha256": hashes["runner_source_sha256"],
            "encoder_sources_sha256": hashes["encoder_sources_sha256"],
            "selector_source_sha256": hashes["selector_source_sha256"],
            "dependency_snapshot_sha256": hashes["dependency_snapshot_sha256"],
            "source_hashes_path": str(out.relative_to(ROOT)).replace("\\", "/"),
        }
    )
    FREEZE.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps(hashes, indent=2))
    print(f"Wrote {out}")
    print(f"Updated {FREEZE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
