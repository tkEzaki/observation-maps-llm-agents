"""Load OPENAI_API_KEY the same way legacy/src/llm/base.py does.

Search order (existing process env always wins; utf-8-sig for BOM):
  <pilot4_kuramoto>/.env
  <llm_emergent>/.env
  <research_current>/gen_ai_logi/rq1/.env
  <research_current>/gen_ai_logi/.env
"""

from __future__ import annotations

from pathlib import Path

_PILOT_ROOT = Path(__file__).resolve().parents[1]  # pilot4_kuramoto/
_EMERGENT_ROOT = _PILOT_ROOT.parent  # llm_emergent/
_RESEARCH_ROOT = _EMERGENT_ROOT.parent  # research_current/

ENV_CANDIDATES = (
    _PILOT_ROOT / ".env",
    _EMERGENT_ROOT / ".env",
    _RESEARCH_ROOT / "gen_ai_logi" / "rq1" / ".env",
    _RESEARCH_ROOT / "gen_ai_logi" / ".env",
)


def load_project_env() -> list[str]:
    """Load dotenv candidates. Returns paths that existed (loaded)."""
    loaded: list[str] = []
    try:
        from dotenv import load_dotenv
    except ImportError:
        return loaded
    for path in ENV_CANDIDATES:
        if path.is_file():
            load_dotenv(path, encoding="utf-8-sig", override=False)
            loaded.append(str(path))
    # cwd-upward fallback (same spirit as legacy controls/engine.py)
    try:
        from dotenv import load_dotenv

        load_dotenv(encoding="utf-8-sig", override=False)
    except Exception:
        pass
    return loaded
