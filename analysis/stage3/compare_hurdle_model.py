"""Compare hurdle multinomial vs softmax / kernel on moments stay gap."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.dataset import build_unified_dataset  # noqa: E402
from analysis.stage3.models import fit_candidate_models  # noqa: E402
from analysis.stage3.validation import (  # noqa: E402
    action_metrics,
    evaluate_grouped_schemes,
    moments_abstention_activation_check,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _group_maps(rows: list) -> dict[str, list[str]]:
    return {
        "leave_one_profile_out": [row.profile_family for row in rows],
        "acquisition_block_holdout": [row.acquisition_block for row in rows],
        "sparse_realization_holdout": [row.sparse_realization for row in rows],
        "offset_group_holdout": [row.offset_group for row in rows],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    parser.add_argument(
        "--include-stump-boost",
        action="store_true",
        help="Include stump-boost in grouped CV (slower).",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    rows_all, packed = build_unified_dataset(ROOT)
    out_rows: list[dict] = []

    model_names = [
        "global_empirical",
        "kernel_nn",
        "multinomial_logistic_l2",
        "hurdle_multinomial",
    ]
    if args.include_stump_boost:
        model_names.append("softmax_stump_boost")

    for representation, arrays in packed.items():
        rep_rows = [r for r in rows_all if r.representation == representation]
        x = arrays["features"]
        counts = arrays["counts"]
        print(f"=== {representation} ===")
        summary = evaluate_grouped_schemes(
            x,
            counts,
            _group_maps(rep_rows),
            model_names=tuple(model_names),
        )
        blocks = sorted({r.acquisition_block for r in rep_rows})
        test_mask = np.asarray(
            [r.acquisition_block == blocks[-1] for r in rep_rows], dtype=bool
        )
        train_mask = ~test_mask
        fitted = fit_candidate_models(x[train_mask], counts[train_mask])
        for name in model_names:
            if name not in fitted and name != "global_empirical":
                continue
            model = fitted[name]
            pred = model.predict_proba(x[test_mask])
            metrics = action_metrics(pred, counts[test_mask])
            stay = moments_abstention_activation_check(
                [rep_rows[i] for i, flag in enumerate(test_mask) if flag],
                pred,
                representation=representation,
            )
            block_key = f"acquisition_block_holdout::{name}"
            grouped = summary["by_scheme_model"].get(block_key, {})
            out_rows.append(
                {
                    "representation": representation,
                    "model": name,
                    **{f"holdout_{k}": v for k, v in metrics.items()},
                    **stay,
                    **{f"cv_{k}": v for k, v in grouped.items()},
                }
            )
            print(
                f"  {name}: log_loss={metrics['log_loss']:.3f} "
                f"stay_gap={stay.get('moments_stay_gap', float('nan'))}"
            )

    _write_csv(out_dir / "hurdle_model_comparison.csv", out_rows)
    print(f"Wrote {out_dir / 'hurdle_model_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
