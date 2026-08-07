"""Feature ablations for Stage 3A surrogates (acquisition-block holdout)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.coverage_utils import N_COMMON  # noqa: E402
from analysis.stage3.dataset import build_unified_dataset  # noqa: E402
from analysis.stage3.models import (  # noqa: E402
    GlobalEmpiricalBaseline,
    HurdleMultinomial,
    KernelNeighborBaseline,
)
from analysis.stage3.validation import (  # noqa: E402
    action_metrics,
    moments_abstention_activation_check,
)
from circlemap.field_features import COMMON_FEATURE_NAMES  # noqa: E402


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mask_features(name: str, x: np.ndarray) -> np.ndarray:
    common = x[:, :N_COMMON]
    native = x[:, N_COMMON:]
    idx = {key: i for i, key in enumerate(COMMON_FEATURE_NAMES)}
    if name == "canonical_only":
        return common
    if name == "native_only":
        return native
    if name == "canonical_plus_native":
        return x
    if name == "drop_abs_z1_block":
        # Drop re/im/abs of z1 from common; keep native.
        keep_c = [i for i in range(N_COMMON) if i not in {idx["re_z1"], idx["im_z1"], idx["abs_z1"]}]
        return np.concatenate([common[:, keep_c], native], axis=1)
    if name == "drop_z2_z3":
        drop = {
            idx["re_z2"],
            idx["im_z2"],
            idx["abs_z2"],
            idx["re_z3"],
            idx["im_z3"],
            idx["abs_z3"],
        }
        keep_c = [i for i in range(N_COMMON) if i not in drop]
        return np.concatenate([common[:, keep_c], native], axis=1)
    if name == "drop_peer_count":
        keep_c = [i for i in range(N_COMMON) if i not in {idx["peer_count"], idx["sparsity"]}]
        return np.concatenate([common[:, keep_c], native], axis=1)
    if name == "symmetry_only":
        keep = [
            idx["abs_z1"],
            idx["abs_z2"],
            idx["abs_z3"],
            idx["antipodal_balance"],
            idx["asymmetry"],
            idx["bimodality"],
            idx["circular_entropy"],
        ]
        return common[:, keep]
    if name == "abs_z1_only":
        return common[:, [idx["abs_z1"]]]
    raise KeyError(name)


ABLATIONS = (
    "canonical_only",
    "native_only",
    "canonical_plus_native",
    "drop_abs_z1_block",
    "drop_z2_z3",
    "drop_peer_count",
    "symmetry_only",
    "abs_z1_only",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    rows_all, packed = build_unified_dataset(ROOT)
    out_rows: list[dict] = []

    for representation, arrays in packed.items():
        rep_rows = [r for r in rows_all if r.representation == representation]
        x_full = arrays["features"]
        counts = arrays["counts"]
        blocks = sorted({r.acquisition_block for r in rep_rows})
        test_mask = np.asarray(
            [r.acquisition_block == blocks[-1] for r in rep_rows], dtype=bool
        )
        train_mask = ~test_mask
        y_test_rows = [rep_rows[i] for i, flag in enumerate(test_mask) if flag]

        for ablation in ABLATIONS:
            x = _mask_features(ablation, x_full)
            for model_name, factory in (
                ("global_empirical", lambda xt, ct: GlobalEmpiricalBaseline.fit(ct)),
                ("kernel_nn", lambda xt, ct: KernelNeighborBaseline.fit(xt, ct)),
                ("hurdle_multinomial", lambda xt, ct: HurdleMultinomial.fit(xt, ct)),
            ):
                model = factory(x[train_mask], counts[train_mask])
                pred = model.predict_proba(x[test_mask])
                metrics = action_metrics(pred, counts[test_mask])
                stay = moments_abstention_activation_check(
                    y_test_rows, pred, representation=representation
                )
                out_rows.append(
                    {
                        "representation": representation,
                        "ablation": ablation,
                        "model": model_name,
                        **{f"m_{k}": v for k, v in metrics.items()},
                        **stay,
                    }
                )
                print(
                    f"{representation} {ablation} {model_name}: "
                    f"log_loss={metrics['log_loss']:.3f}"
                )

    _write_csv(out_dir / "feature_ablation_results.csv", out_rows)
    print(f"Wrote {out_dir / 'feature_ablation_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
