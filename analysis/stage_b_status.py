"""Summarize structural diagnostics of a Stage B response curve."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    profiles = {}
    curves = {}
    for name in sorted({row["profile"] for row in rows}):
        selected = [row for row in rows if row["profile"] == name]
        delta = np.asarray(
            [float(row["offset_radians"]) for row in selected]
        )
        g = np.asarray([float(row["g_mean"]) for row in selected])
        odd = np.asarray([float(row["g_odd"]) for row in selected])
        even = np.asarray([float(row["g_even"]) for row in selected])
        standard_error = np.asarray(
            [float(row["g_standard_error"]) for row in selected]
        )
        stay = np.asarray([float(row["p_stay"]) for row in selected])
        adjacent_change = np.abs(np.roll(g, -1) - g)
        profiles[name] = {
            "correlation_with_sin_delta": float(
                np.corrcoef(g, np.sin(delta))[0, 1]
            ),
            "mean_absolute_adjacent_change": float(
                np.mean(adjacent_change)
            ),
            "maximum_adjacent_change": float(np.max(adjacent_change)),
            "odd_rms": float(np.sqrt(np.mean(odd**2))),
            "even_rms": float(np.sqrt(np.mean(even**2))),
            "mean_standard_error": float(np.mean(standard_error)),
            "mean_stay_probability": float(np.mean(stay)),
        }
        curves[name] = g

    profile_names = sorted(curves)
    cross_profile_correlation = (
        float(np.corrcoef(curves[profile_names[0]], curves[profile_names[1]])[0, 1])
        if len(profile_names) == 2
        else None
    )
    return {
        "source": str(path),
        "profiles": profiles,
        "cross_profile_correlation": cross_profile_correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response_curve", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = summarize(args.response_curve)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
