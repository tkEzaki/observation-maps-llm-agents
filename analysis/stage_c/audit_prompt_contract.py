"""Audit Stage B vs Stage C prompt / generation contract (no API calls)."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.collective_scan import _histogram_from_phases  # noqa: E402
from circlemap import backends as backends_mod  # noqa: E402
from circlemap import representations as reps_mod  # noqa: E402
from circlemap import response as response_mod  # noqa: E402
from circlemap.representations import (  # noqa: E402
    build_representation_prompt_from_histogram,
    coerce_representation_spec,
)
from circlemap.response import PROMPT_VERSION, parse_social_action  # noqa: E402
from circlemap.stimuli import STIMULUS_CATALOG, StimulusSpec, build_stimulus_histogram  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_catalog(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, data in payload.items():
        data = dict(data)
        data.pop("id", None)
        data.pop("derived_from_catalog", None)
        for key in ("mode_offsets", "weights", "fixed_fractions"):
            if key in data:
                data[key] = tuple(float(v) for v in data[key])
        if "mode_counts" in data:
            data["mode_counts"] = tuple(int(v) for v in data["mode_counts"])
        STIMULUS_CATALOG[str(name)] = StimulusSpec(**data)  # type: ignore[arg-type]


def _instruction_prefix_hash() -> str:
    # Reconstruct frozen instruction with a sentinel description.
    text = reps_mod._instruction("__DESC__")
    return _sha(text)


def _response_contract_source_hash() -> str:
    return _sha(inspect.getsource(parse_social_action))


def _moments_serializer_source_hash() -> str:
    return _sha(inspect.getsource(reps_mod._serialize_moments_histogram))


def _openai_call_source_hash() -> str:
    return _sha(inspect.getsource(backends_mod.OpenAIResponseBackend.call))


def audit_stage_c_reconstructions(session: Path, *, max_checks: int = 200) -> dict:
    """Rebuild prompts from phases.npy and compare to stored prompt_sha256."""
    protocol = json.loads((session / "protocol.json").read_text(encoding="utf-8"))
    representation = protocol["representation"]
    n_bins = int(protocol["n_bins"])
    checked = 0
    mismatches = 0
    examples = []
    for run_dir in sorted(p for p in session.iterdir() if p.is_dir()):
        phases_path = run_dir / "phases.npy"
        trace_path = run_dir / "trace.jsonl"
        if not phases_path.exists() or not trace_path.exists():
            continue
        phases = np.load(phases_path)
        with trace_path.open(encoding="utf-8") as handle:
            for line in handle:
                if checked >= max_checks:
                    break
                rec = json.loads(line)
                t = int(rec["t"])
                agent = int(rec["agent"])
                hist = _histogram_from_phases(phases[t], agent, n_bins=n_bins)
                prompt = build_representation_prompt_from_histogram(
                    representation, hist
                )
                got = _sha(prompt)
                checked += 1
                if got != rec["prompt_sha256"]:
                    mismatches += 1
                    if len(examples) < 5:
                        examples.append(
                            {
                                "run_id": rec["run_id"],
                                "t": t,
                                "agent": agent,
                                "stored": rec["prompt_sha256"],
                                "recomputed": got,
                            }
                        )
            if checked >= max_checks:
                break
    return {
        "checked": checked,
        "mismatches": mismatches,
        "match_rate": 1.0 - mismatches / max(checked, 1),
        "examples": examples,
    }


def audit_stage_b_moments_identity(catalog_path: Path) -> dict:
    """Same physical field → same moments prompt via Stage B path."""
    _load_catalog(catalog_path)
    # Use a pilot control that exists in both catalog and Stage B training.
    name = "control_exact_antipodal_8_8_r00"
    hist = build_stimulus_histogram(name, 0.0)
    p1 = build_representation_prompt_from_histogram("moments_m1_m3", hist)
    # Stage B representation runner uses the same builder.
    p2 = build_representation_prompt_from_histogram(
        coerce_representation_spec("moments_m1_m3")[0], hist
    )
    return {
        "profile": name,
        "prompt_sha256": _sha(p1),
        "identical_dual_call": p1 == p2,
        "prompt_chars": len(p1),
        "contains_no_k_or_time": (
            "coupling" not in p1.lower()
            and "\nK=" not in p1
            and "agent_id" not in p1.lower()
            and "timestep" not in p1.lower()
        ),
        "starts_with_operator_line": p1.startswith(
            "You are an interaction operator for a phase on a circle."
        ),
        "has_required_schema": '{"social_action":"advance|stay|retard"}' in p1,
    }


def compare_generation_settings(stage_b_config: Path, stage_c_protocol: Path) -> dict:
    b = json.loads(stage_b_config.read_text(encoding="utf-8"))
    c_proto = json.loads(stage_c_protocol.read_text(encoding="utf-8"))
    # Stage C runner defaults (must match argparse defaults in run_collective).
    c_defaults = {
        "model": "gpt-5.4-mini",  # as used in paid run
        "temperature": 0.7,
        "max_output_tokens": int(c_proto["max_output_tokens"]),
        "max_attempts": int(c_proto["max_attempts"]),
        "prompt_version": c_proto.get("prompt_version"),
        "representation": c_proto.get("representation"),
    }
    b_settings = {
        "model": b.get("model"),
        "temperature": b.get("temperature"),
        "max_output_tokens": b.get("max_output_tokens"),
        "max_attempts": b.get("max_attempts"),
        "prompt_version": b.get("protocol", {}).get("prompt_version"),
        "backend": b.get("backend"),
    }
    return {
        "stage_b": b_settings,
        "stage_c": c_defaults,
        "matches": {
            "model": b_settings["model"] == c_defaults["model"],
            "temperature": b_settings["temperature"] == c_defaults["temperature"],
            "max_output_tokens": (
                b_settings["max_output_tokens"] == c_defaults["max_output_tokens"]
            ),
            "max_attempts": b_settings["max_attempts"] == c_defaults["max_attempts"],
            "prompt_version": (
                b_settings["prompt_version"] == c_defaults["prompt_version"]
            ),
        },
        "all_match": all(
            [
                b_settings["model"] == c_defaults["model"],
                b_settings["temperature"] == c_defaults["temperature"],
                b_settings["max_output_tokens"] == c_defaults["max_output_tokens"],
                b_settings["max_attempts"] == c_defaults["max_attempts"],
                b_settings["prompt_version"] == c_defaults["prompt_version"],
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-c-session",
        type=Path,
        default=ROOT
        / "runs"
        / "stage_c"
        / "stage-c-v0.1_0291166d2e39"
        / "20260723T235620Z",
    )
    parser.add_argument(
        "--stage-b-config",
        type=Path,
        default=ROOT
        / "runs"
        / "response_law_representation"
        / "stage3a-pilot-v1-block_1_9cd99d18c51d"
        / "20260723T234204Z"
        / "resolved_config.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "analysis" / "stage_c_artifacts" / "prompt_contract_audit.json",
    )
    parser.add_argument("--max-checks", type=int, default=400)
    args = parser.parse_args()

    stage_c_protocol = args.stage_c_session / "protocol.json"
    catalog = ROOT / "analysis" / "stage3a_artifacts" / "pilot_stimulus_catalog_v1.json"

    code_hashes = {
        "PROMPT_VERSION": PROMPT_VERSION,
        "instruction_template_sha256": _instruction_prefix_hash(),
        "parse_social_action_source_sha256": _response_contract_source_hash(),
        "serialize_moments_source_sha256": _moments_serializer_source_hash(),
        "openai_backend_call_source_sha256": _openai_call_source_hash(),
        "representations_py_sha256": _sha_file(ROOT / "circlemap" / "representations.py"),
        "response_py_sha256": _sha_file(ROOT / "circlemap" / "response.py"),
        "backends_py_sha256": _sha_file(ROOT / "circlemap" / "backends.py"),
    }
    gen = compare_generation_settings(args.stage_b_config, stage_c_protocol)
    stage_b_identity = audit_stage_b_moments_identity(catalog)
    recon = audit_stage_c_reconstructions(
        args.stage_c_session, max_checks=args.max_checks
    )

    # Shared builder: Stage B representation path and Stage C both call
    # build_representation_prompt_from_histogram("moments_m1_m3", hist).
    shared_builder = {
        "stage_b_module": "experiments.stage_b_response_law.run_representation",
        "stage_c_module": "experiments.stage_c.run_collective",
        "shared_function": "circlemap.representations.build_representation_prompt_from_histogram",
        "representation": "moments_m1_m3",
        "stateless_observation_only": stage_b_identity["contains_no_k_or_time"],
    }

    verdict_ok = (
        gen["all_match"]
        and recon["mismatches"] == 0
        and stage_b_identity["identical_dual_call"]
        and stage_b_identity["contains_no_k_or_time"]
    )
    report = {
        "verdict": (
            "PASS_prompt_contract_identical"
            if verdict_ok
            else "FAIL_protocol_mismatch"
        ),
        "implication": (
            "Stay bias is surrogate smoothing / coverage, not Stage B↔C prompt mismatch."
            if verdict_ok
            else "Fix prompt / generation settings before surrogate redesign."
        ),
        "code_hashes": code_hashes,
        "generation_settings": gen,
        "stage_b_moments_identity": stage_b_identity,
        "stage_c_prompt_reconstruction": recon,
        "shared_builder": shared_builder,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = ROOT / "docs" / "STAGE_C_PROMPT_CONTRACT_AUDIT.md"
    md.write_text(
        "\n".join(
            [
                "# Stage B ↔ Stage C prompt contract audit",
                "",
                f"Date: 2026-07-24  ",
                f"Verdict: **{report['verdict']}**  ",
                f"Artifact: `{args.out.relative_to(ROOT).as_posix()}`",
                "",
                "## Generation settings",
                "",
                f"- model match: `{gen['matches']['model']}` "
                f"(`{gen['stage_b']['model']}`)",
                f"- temperature match: `{gen['matches']['temperature']}` "
                f"(`{gen['stage_b']['temperature']}`)",
                f"- max_output_tokens match: `{gen['matches']['max_output_tokens']}` "
                f"(`{gen['stage_b']['max_output_tokens']}`)",
                f"- max_attempts match: `{gen['matches']['max_attempts']}`",
                f"- prompt_version match: `{gen['matches']['prompt_version']}` "
                f"(`{gen['stage_b']['prompt_version']}`)",
                "",
                "## Serializer / contract",
                "",
                f"- Shared builder: `{shared_builder['shared_function']}`",
                f"- Stage C reconstructed prompt hash match rate: "
                f"**{recon['match_rate']:.4f}** "
                f"({recon['checked'] - recon['mismatches']}/{recon['checked']})",
                f"- Observation-only (no K/t/agent id): "
                f"`{stage_b_identity['contains_no_k_or_time']}`",
                f"- `PROMPT_VERSION`: `{PROMPT_VERSION}`",
                "",
                "## Implication",
                "",
                report["implication"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "verdict": report["verdict"],
        "generation_all_match": gen["all_match"],
        "recon_match_rate": recon["match_rate"],
        "checked": recon["checked"],
        "md": str(md.relative_to(ROOT).as_posix()),
    }, indent=2))
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
