"""Calibrate OOD distance against grouped-holdout prediction error."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.coverage_utils import ComponentOOD, tv_distance  # noqa: E402
from analysis.stage3.dataset import build_unified_dataset  # noqa: E402
from analysis.stage3.models import KernelNeighborBaseline, counts_to_probs  # noqa: E402
from analysis.stage3.ood import ood_error_calibration  # noqa: E402
from analysis.stage3.validation import multinomial_log_loss  # noqa: E402


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if denom <= 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, packed = build_unified_dataset(ROOT)
    calib_rows: list[dict] = []
    summary_rows: list[dict] = []

    for representation, arrays in packed.items():
        rep_rows = [r for r in rows if r.representation == representation]
        x = arrays["features"]
        counts = arrays["counts"]
        unresolved = arrays["unresolved_near_zero"]
        blocks = sorted({r.acquisition_block for r in rep_rows})
        if len(blocks) < 2:
            continue
        test_block = blocks[-1]
        test_mask = np.asarray(
            [r.acquisition_block == test_block for r in rep_rows], dtype=bool
        )
        train_mask = ~test_mask
        model = KernelNeighborBaseline.fit(x[train_mask], counts[train_mask])
        component = ComponentOOD.fit(x[train_mask])
        pred = model.predict_proba(x[test_mask])
        empir = counts_to_probs(counts[test_mask], alpha=0.0)
        tv = tv_distance(pred, empir)
        nll_row = -np.sum(
            counts[test_mask] * np.log(np.clip(pred, 1e-12, 1.0)),
            axis=1,
        ) / np.maximum(np.sum(counts[test_mask], axis=1), 1.0)
        dists = component.distances(
            x[test_mask],
            unresolved_near_zero=unresolved[test_mask],
        )
        for key in ("d_canonical", "d_native", "d_combined", "d_peer"):
            tv_bins = ood_error_calibration(dists[key], tv, n_bins=8)
            nll_bins = ood_error_calibration(dists[key], nll_row, n_bins=8)
            for tv_row, nll_bin in zip(tv_bins, nll_bins):
                calib_rows.append(
                    {
                        "representation": representation,
                        "distance": key,
                        "mean_ood": tv_row["mean_ood"],
                        "mean_tv": tv_row["mean_tv"],
                        "mean_nll": nll_bin["mean_tv"],
                        "n": tv_row["n"],
                        "bin_left": tv_row["bin_left"],
                        "bin_right": tv_row["bin_right"],
                    }
                )
            summary_rows.append(
                {
                    "representation": representation,
                    "distance": key,
                    "spearman_tv": spearman(dists[key], tv),
                    "spearman_nll": spearman(dists[key], nll_row),
                    "holdout_log_loss": multinomial_log_loss(pred, counts[test_mask]),
                    "holdout_mean_tv": float(np.mean(tv)),
                }
            )

    _write_csv(out_dir / "ood_error_calibration.csv", calib_rows)
    _write_csv(out_dir / "ood_error_calibration_summary.csv", summary_rows)
    print(f"Wrote {out_dir / 'ood_error_calibration.csv'}")
    for row in summary_rows:
        print(
            f"  {row['representation']} {row['distance']}: "
            f"spearman_tv={row['spearman_tv']:.3f} "
            f"spearman_nll={row['spearman_nll']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
