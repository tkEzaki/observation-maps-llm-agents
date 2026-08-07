"""Shared CENT-3 offline I/O helpers (no paid calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.coverage_utils import feature_builder  # noqa: E402
from analysis.stage3.dataset import load_run_counts  # noqa: E402
from circlemap.observation import RelativePhaseHistogram  # noqa: E402
from circlemap.stimuli import STIMULUS_CATALOG, StimulusSpec  # noqa: E402

REP = "centers_24_standard"
BRANCH = ROOT / "analysis" / "centers_branch"


def latest_cent3_runs(root: Path | None = None) -> list[Path]:
    base = (root or ROOT) / "runs" / "centers_cent3"
    runs: list[Path] = []
    for experiment_dir in sorted(base.iterdir()):
        if not experiment_dir.is_dir():
            continue
        if "centers-cent3-v0-block_" not in experiment_dir.name:
            continue
        stamped = [
            path
            for path in experiment_dir.iterdir()
            if path.is_dir() and (path / "trace.jsonl").exists()
        ]
        if stamped:
            runs.append(max(stamped, key=lambda path: path.name))
    return sorted(runs)


def load_catalog(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, data in payload.items():
        data = dict(data)
        data.pop("id", None)
        data.pop("derived_from_catalog", None)
        data.pop("shift_bins", None)
        if "fixed_fractions" in data:
            data["fixed_fractions"] = tuple(float(v) for v in data["fixed_fractions"])
        STIMULUS_CATALOG[str(name)] = StimulusSpec(**data)  # type: ignore[arg-type]


def histogram_from_field(field: dict) -> RelativePhaseHistogram:
    stim = field["stimulus"]
    fractions = np.asarray(stim["fixed_fractions"], dtype=np.float64)
    edges = np.linspace(-np.pi, np.pi, int(stim["n_bins"]) + 1, dtype=np.float64)
    return RelativePhaseHistogram(
        edges=edges,
        fractions=fractions,
        peer_count=int(stim["peer_count"]),
    )


def field_feature_row(field: dict) -> tuple[np.ndarray, bool, int, float]:
    hist = histogram_from_field(field)
    x, unresolved, desc = feature_builder(REP, hist)
    sparsity = float(np.mean(hist.fractions > 0))
    return x, bool(unresolved), int(hist.peer_count), sparsity


def pooled_and_block_counts(
    run_dirs: list[Path],
) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
    block_counts: list[dict[str, np.ndarray]] = []
    for run_dir in run_dirs:
        _protocol, counts = load_run_counts(run_dir)
        by_profile = counts[REP]
        block_counts.append(
            {pid: np.asarray(cell[0], dtype=np.float64) for pid, cell in by_profile.items()}
        )
    pooled: dict[str, np.ndarray] = {}
    for block in block_counts:
        for pid, vec in block.items():
            pooled[pid] = pooled.get(pid, np.zeros(3, dtype=np.float64)) + vec
    return pooled, block_counts


def action_channels(p: np.ndarray) -> dict[str, float]:
    """Trinomial channels used in CENT-3/4 offline reports.

    A = activity = p+ + p-
    a0 = signed mean action = p+ - p-
    C1 = |a0| (polar magnitude on the action simplex)
    C2 = p_stay (second reporting channel; not a field Fourier mode)
    """
    p_m, p0, p_p = float(p[0]), float(p[1]), float(p[2])
    a0 = p_p - p_m
    activity = p_p + p_m
    return {
        "A": activity,
        "a0": a0,
        "C1": abs(a0),
        "C2": p0,
        "p_minus": p_m,
        "p_stay": p0,
        "p_plus": p_p,
    }
