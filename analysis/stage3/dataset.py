"""Unify Stage B response-law traces into Stage 3 training tables."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from circlemap.field_features import extract_field_descriptors
from circlemap.response import fixed_field_histogram
from circlemap.stimuli import STIMULUS_CATALOG, StimulusSpec, build_stimulus_histogram

ACTION_INDEX = {"retard": 0, "stay": 1, "advance": 2}
SPARSE_REALIZATION_RE = re.compile(r"_r(\d{2})_")
EPSILON_RE = re.compile(r"antipodal_weps_([mp])(\d+)_")


@dataclass
class Stage3Row:
    representation: str
    profile: str
    profile_family: str
    source_family: str
    acquisition_block: str
    run_dir: str
    offset_index: int
    offset_radians: float
    epsilon_level: str
    sparse_realization: str
    offset_group: str
    count_retard: int
    count_stay: int
    count_advance: int
    estimated_eps: float
    unresolved_near_zero: bool


def _latest_run_dirs(root: Path) -> list[Path]:
    """Pick the newest timestamped child under each experiment hash folder."""
    if not root.exists():
        return []
    chosen: list[Path] = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        stamped = [
            path
            for path in experiment_dir.iterdir()
            if path.is_dir() and (path / "trace.jsonl").exists()
        ]
        if not stamped:
            continue
        chosen.append(max(stamped, key=lambda path: path.name))
    return chosen


def _latest_pilot_run_dirs(project_root: Path) -> list[Path]:
    root = project_root / "runs" / "response_law_representation"
    if not root.exists():
        return []
    chosen: list[Path] = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        if "stage3a-pilot-v1" not in experiment_dir.name:
            continue
        stamped = [
            path
            for path in experiment_dir.iterdir()
            if path.is_dir() and (path / "trace.jsonl").exists()
        ]
        if stamped:
            chosen.append(max(stamped, key=lambda path: path.name))
    return chosen


def load_stimulus_catalog_file(project_root: Path, catalog_file: str | None) -> None:
    """Register protocol-bundled fixed stimuli into STIMULUS_CATALOG."""
    if not catalog_file:
        return
    path = Path(catalog_file)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, spec_payload in payload.items():
        data = dict(spec_payload)
        data.pop("id", None)
        data.pop("derived_from_catalog", None)
        for key in ("mode_offsets", "weights", "fixed_fractions"):
            if key in data:
                data[key] = tuple(float(v) for v in data[key])
        if "mode_counts" in data:
            data["mode_counts"] = tuple(int(v) for v in data["mode_counts"])
        STIMULUS_CATALOG[str(name)] = StimulusSpec(**data)  # type: ignore[arg-type]


def default_source_runs(project_root: Path) -> dict[str, list[Path]]:
    """Canonical Stage-3A training sources (canonical three encodings only)."""
    return {
        "transmutation": _latest_run_dirs(project_root / "runs" / "transmutation"),
        "stimulus_manifold": _latest_run_dirs(
            project_root / "runs" / "stimulus_manifold"
        ),
        "antipodal_weight": _latest_run_dirs(
            project_root / "runs" / "antipodal_weight"
        ),
        "sparse_peer": _latest_run_dirs(project_root / "runs" / "sparse_peer"),
    }


def post_pilot_source_runs(project_root: Path) -> dict[str, list[Path]]:
    """Pre-pilot Stage B sources plus Stage 3A coverage pilot blocks."""
    sources = default_source_runs(project_root)
    pilot = _latest_pilot_run_dirs(project_root)
    if pilot:
        sources["stage3a_pilot"] = pilot
    return sources


def _latest_replay_run_dirs(project_root: Path) -> list[Path]:
    root = project_root / "runs" / "collective_replay"
    if not root.exists():
        return []
    chosen: list[Path] = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        stamped = [
            path
            for path in experiment_dir.iterdir()
            if path.is_dir() and (path / "trace.jsonl").exists()
        ]
        if stamped:
            chosen.append(max(stamped, key=lambda path: path.name))
    return chosen


def rows_from_replay_run(
    run_dir: Path,
    *,
    fields_path: Path,
    source_family: str = "collective_replay",
    representation: str = "moments_m1_m3",
) -> tuple[list[Stage3Row], dict[str, list]]:
    """Materialize Stage3 rows from collective-field replay traces."""
    fields_payload = json.loads(fields_path.read_text(encoding="utf-8"))
    field_map = {f["field_id"]: f for f in fields_payload["fields"]}
    counts: dict[str, np.ndarray] = {
        fid: np.zeros(3, dtype=np.int64) for fid in field_map
    }
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not record.get("valid"):
                continue
            profile = str(record["profile"])
            if profile not in counts:
                continue
            counts[profile][ACTION_INDEX[record["action_label"]]] += 1

    config = {}
    cfg_path = run_dir / "resolved_config.json"
    if cfg_path.exists():
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
    block = str(
        config.get("protocol", {}).get("protocol_version", run_dir.parent.name)
    )
    rows: list[Stage3Row] = []
    feature_store: dict[str, list] = {representation: []}
    for profile, cell in counts.items():
        if int(np.sum(cell)) <= 0:
            continue
        field = field_map[profile]
        fr = np.asarray(field["fixed_fractions"], dtype=np.float64)
        edges = np.linspace(-np.pi, np.pi, fr.size + 1, dtype=np.float64)
        from circlemap.observation import RelativePhaseHistogram

        histogram = RelativePhaseHistogram(
            edges=edges,
            fractions=fr,
            peer_count=int(field["peer_count"]),
        )
        descriptors = extract_field_descriptors(representation, histogram)
        feature_store[representation].append(descriptors)
        rows.append(
            Stage3Row(
                representation=representation,
                profile=profile,
                profile_family=profile_family(profile),
                source_family=source_family,
                acquisition_block=block,
                run_dir=str(run_dir),
                offset_index=0,
                offset_radians=0.0,
                epsilon_level="na",
                sparse_realization="na",
                offset_group="g0",
                count_retard=int(cell[0]),
                count_stay=int(cell[1]),
                count_advance=int(cell[2]),
                estimated_eps=float(descriptors.estimated_eps),
                unresolved_near_zero=bool(descriptors.unresolved_near_zero),
            )
        )
    return rows, feature_store


def v2_source_runs(project_root: Path) -> dict[str, list[Path]]:
    """Post-pilot sources plus collective-field replay (moments v2 training)."""
    sources = post_pilot_source_runs(project_root)
    replay = _latest_replay_run_dirs(project_root)
    if replay:
        sources["collective_replay"] = replay
    return sources


def build_unified_dataset_v2(
    project_root: Path,
    *,
    replay_fields_path: Path | None = None,
) -> tuple[list[Stage3Row], dict[str, dict[str, np.ndarray]]]:
    """Unified dataset including replay rows (moments-focused)."""
    sources = v2_source_runs(project_root)
    fields_path = replay_fields_path or (
        project_root
        / "analysis"
        / "stage_c_artifacts"
        / "replay_panel_v1"
        / "replay_fields_v1.json"
    )
    all_rows: list[Stage3Row] = []
    descriptors_by_rep: dict[str, list] = {}
    for source_family, run_dirs in sources.items():
        for run_dir in run_dirs:
            if source_family == "collective_replay":
                rows, feature_store = rows_from_replay_run(
                    run_dir,
                    fields_path=fields_path,
                    source_family=source_family,
                )
            else:
                rows, feature_store = rows_from_run(
                    run_dir,
                    source_family=source_family,
                    project_root=project_root,
                )
            all_rows.extend(rows)
            for representation, descriptors in feature_store.items():
                descriptors_by_rep.setdefault(representation, []).extend(descriptors)

    packed: dict[str, dict[str, np.ndarray]] = {}
    for representation, descriptors in descriptors_by_rep.items():
        rep_rows = [row for row in all_rows if row.representation == representation]
        if len(rep_rows) != len(descriptors):
            raise RuntimeError("row / descriptor alignment mismatch")
        common = np.asarray([item.common for item in descriptors], dtype=np.float64)
        native = np.asarray([item.native for item in descriptors], dtype=np.float64)
        features = np.concatenate([common, native], axis=1)
        counts = np.asarray(
            [
                [row.count_retard, row.count_stay, row.count_advance]
                for row in rep_rows
            ],
            dtype=np.float64,
        )
        packed[representation] = {
            "features": features,
            "counts": counts,
            "unresolved_near_zero": np.asarray(
                [row.unresolved_near_zero for row in rep_rows], dtype=bool
            ),
            "estimated_eps": np.asarray(
                [row.estimated_eps for row in rep_rows], dtype=np.float64
            ),
        }
    return all_rows, packed


def profile_family(profile: str) -> str:
    """Strip sparse realization suffix so families group across realizations."""
    return SPARSE_REALIZATION_RE.sub("_rXX_", profile)


def epsilon_level_of(profile: str, epsilon_by_profile: dict | None) -> str:
    if epsilon_by_profile and profile in epsilon_by_profile:
        return f"{float(epsilon_by_profile[profile]):+.3f}"
    match = EPSILON_RE.search(profile)
    if not match:
        return "na"
    sign = -1 if match.group(1) == "m" else 1
    value = sign * int(match.group(2)) / 1000.0
    return f"{value:+.3f}"


def sparse_realization_of(profile: str) -> str:
    match = SPARSE_REALIZATION_RE.search(profile)
    return match.group(1) if match else "na"


def offset_group_of(offset_index: int, n_offsets: int, *, n_groups: int = 6) -> str:
    width = max(1, int(np.ceil(n_offsets / n_groups)))
    return f"g{offset_index // width}"


def _histogram_for_profile(
    protocol: dict,
    profile: str,
    offset_radians: float,
):
    if "stimulus_profiles" in protocol:
        if profile not in STIMULUS_CATALOG:
            raise KeyError(f"profile {profile!r} missing from STIMULUS_CATALOG")
        return build_stimulus_histogram(profile, offset_radians)
    concentrations = protocol["concentration_profiles"]
    kappa = float(concentrations[profile])
    return fixed_field_histogram(offset_radians, kappa, n_bins=24)


def load_run_counts(run_dir: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    protocol = config["protocol"]
    offsets = list(protocol["offset_radians"])
    n_offsets = len(offsets)
    if "stimulus_profiles" in protocol:
        profiles = list(protocol["stimulus_profiles"])
    else:
        profiles = list(protocol["concentration_profiles"])
    representations = list(protocol["representations"])
    counts = {
        representation: {
            profile: np.zeros((n_offsets, 3), dtype=np.int64)
            for profile in profiles
        }
        for representation in representations
    }
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not record.get("valid"):
                continue
            counts[record["representation"]][record["profile"]][
                int(record["offset_index"])
            ][ACTION_INDEX[record["action_label"]]] += 1
    return protocol, counts


def rows_from_run(
    run_dir: Path,
    *,
    source_family: str,
    project_root: Path | None = None,
) -> tuple[list[Stage3Row], dict[str, list]]:
    """Materialize training rows and aligned feature tensors per representation."""
    protocol, counts = load_run_counts(run_dir)
    if project_root is None:
        # .../project/runs/<family>/<exp>/<stamp>
        project_root = Path(run_dir).resolve().parents[3]
    load_stimulus_catalog_file(project_root, protocol.get("stimulus_catalog_file"))
    offsets = np.asarray(protocol["offset_radians"], dtype=np.float64)
    n_offsets = int(offsets.size)
    block = str(protocol.get("seed_block_id", run_dir.parent.name))
    epsilon_by_profile = protocol.get("epsilon_by_profile")
    feature_store: dict[str, list] = {
        representation: [] for representation in protocol["representations"]
    }
    rows: list[Stage3Row] = []
    if "stimulus_profiles" in protocol:
        profiles = list(protocol["stimulus_profiles"])
    else:
        profiles = list(protocol["concentration_profiles"])

    for representation in protocol["representations"]:
        for profile in profiles:
            for offset_index, offset in enumerate(offsets):
                cell = counts[representation][profile][offset_index]
                if int(np.sum(cell)) <= 0:
                    continue
                histogram = _histogram_for_profile(protocol, profile, float(offset))
                descriptors = extract_field_descriptors(representation, histogram)
                feature_store[representation].append(descriptors)
                rows.append(
                    Stage3Row(
                        representation=representation,
                        profile=profile,
                        profile_family=profile_family(profile),
                        source_family=source_family,
                        acquisition_block=block,
                        run_dir=str(run_dir),
                        offset_index=int(offset_index),
                        offset_radians=float(offset),
                        epsilon_level=epsilon_level_of(profile, epsilon_by_profile),
                        sparse_realization=sparse_realization_of(profile),
                        offset_group=offset_group_of(offset_index, n_offsets),
                        count_retard=int(cell[0]),
                        count_stay=int(cell[1]),
                        count_advance=int(cell[2]),
                        estimated_eps=float(descriptors.estimated_eps),
                        unresolved_near_zero=bool(descriptors.unresolved_near_zero),
                    )
                )
    return rows, feature_store


def build_unified_dataset(
    project_root: Path,
    *,
    source_runs: dict[str, list[Path]] | None = None,
) -> tuple[list[Stage3Row], dict[str, dict[str, np.ndarray]]]:
    """Return rows and per-representation feature matrices / labels / groups."""
    sources = source_runs or default_source_runs(project_root)
    all_rows: list[Stage3Row] = []
    descriptors_by_rep: dict[str, list] = {}
    for source_family, run_dirs in sources.items():
        for run_dir in run_dirs:
            rows, feature_store = rows_from_run(
                run_dir,
                source_family=source_family,
                project_root=project_root,
            )
            all_rows.extend(rows)
            for representation, descriptors in feature_store.items():
                descriptors_by_rep.setdefault(representation, []).extend(descriptors)

    packed: dict[str, dict[str, np.ndarray]] = {}
    for representation, descriptors in descriptors_by_rep.items():
        rep_rows = [row for row in all_rows if row.representation == representation]
        if len(rep_rows) != len(descriptors):
            raise RuntimeError("row / descriptor alignment mismatch")
        common = np.asarray([item.common for item in descriptors], dtype=np.float64)
        native = np.asarray([item.native for item in descriptors], dtype=np.float64)
        features = np.concatenate([common, native], axis=1)
        counts = np.asarray(
            [
                [row.count_retard, row.count_stay, row.count_advance]
                for row in rep_rows
            ],
            dtype=np.float64,
        )
        packed[representation] = {
            "features": features,
            "counts": counts,
            "unresolved_near_zero": np.asarray(
                [row.unresolved_near_zero for row in rep_rows],
                dtype=bool,
            ),
            "estimated_eps": np.asarray(
                [row.estimated_eps for row in rep_rows],
                dtype=np.float64,
            ),
        }
    return all_rows, packed


def write_dataset_csv(rows: Iterable[Stage3Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_packed_npz(
    packed: dict[str, dict[str, np.ndarray]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for representation, arrays in packed.items():
        for key, value in arrays.items():
            payload[f"{representation}__{key}"] = value
    np.savez_compressed(path, **payload)
