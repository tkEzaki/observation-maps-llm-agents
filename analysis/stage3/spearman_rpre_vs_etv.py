"""Calibration of the fixed applicability risk against the observed response error.

For the moments prospective pilot, each evaluated field carries a predictive
risk ``R_pre`` computed before the acquisition from descriptors alone, and an
observed response error ``e_tv`` measured afterwards. This module reports the
Spearman rank correlation between the two, per representation, together with
the tertile medians that summarise the trend. It is the calibration evidence
behind the applicability-domain claim, and it is descriptive: no threshold is
fitted here and nothing is selected on the outcome.

Spearman rho is computed from ranks against numpy alone; scipy is not a
dependency of this repository.

Usage
-----
    python -m analysis.stage3.spearman_rpre_vs_etv

Writes ``analysis/stage3a_artifacts/spearman_rpre_vs_etv.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
S3A = ROOT / "analysis" / "stage3a_artifacts"
DETAIL = S3A / "pilot_prospective_detail.csv"
OUT = S3A / "spearman_rpre_vs_etv.json"

REP_ORDER = (
    "moments_m1_m3",
    "centers_24_standard",
    "intervals_24_decimal6",
)


def spearman(a, b) -> float:
    """Spearman rho via ranks (average ranks for ties)."""
    ra = pd.Series(np.asarray(a, dtype=float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, dtype=float)).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    det = pd.read_csv(DETAIL)
    out: dict = {"source": str(DETAIL.relative_to(ROOT)), "per_representation": {}}

    for rep in REP_ORDER:
        sub = det[det["representation"] == rep]
        if sub.empty:
            continue
        r = sub["R_pre"].to_numpy(dtype=float)
        e = sub["e_tv"].to_numpy(dtype=float)
        order = np.argsort(r)
        r_s, e_s = r[order], e[order]
        thirds = np.array_split(np.arange(r_s.size), 3)
        out["per_representation"][rep] = {
            "n": int(r.size),
            "spearman_rho": spearman(r, e),
            "tertile_median_R_pre": [float(np.median(r_s[t])) for t in thirds],
            "tertile_median_e_tv": [float(np.median(e_s[t])) for t in thirds],
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    for rep, v in out["per_representation"].items():
        print(f"  {rep}: rho = {v['spearman_rho']:+.3f}  (n = {v['n']})")


if __name__ == "__main__":
    main()
