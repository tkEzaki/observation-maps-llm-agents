"""Fresh Stage B response backends; no legacy imports."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BackendResult:
    raw_text: str
    input_tokens: int
    output_tokens: int


class ResponseBackend(Protocol):
    name: str
    model: str

    def call(self, prompt: str, *, seed: int) -> BackendResult:
        """Return one raw response without parsing or fallback."""


class QuotaError(RuntimeError):
    """Non-retryable provider quota or billing failure."""


@dataclass
class MockResponseBackend:
    """Seeded stochastic attractive response used only for pipeline tests."""

    name: str = "mock"
    model: str = "attractive-v1"

    def call(self, prompt: str, *, seed: int) -> BackendResult:
        values = []
        centers = []
        for line in prompt.splitlines():
            interval = re.fullmatch(
                r"bin_(\d{2}) "
                r"\[([+-]?\d+\.\d+),([+-]?\d+\.\d+)\): "
                r"(\d+(?:\.\d+)?)",
                line,
            )
            center = re.fullmatch(
                r"bin_(\d{2}) center ([+-]?\d+\.\d+): "
                r"(\d+(?:\.\d+)?)",
                line,
            )
            if interval:
                centers.append(
                    (float(interval.group(2)) + float(interval.group(3))) / 2.0
                )
                values.append(float(interval.group(4)))
            elif center:
                centers.append(float(center.group(2)))
                values.append(float(center.group(3)))

        if values:
            value_array = np.asarray(values, dtype=np.float64)
            value_array /= np.sum(value_array)
            center_array = np.asarray(centers, dtype=np.float64)
            vector = np.sum(value_array * np.exp(1j * center_array))
        else:
            cosine_match = re.search(
                r"moment_1_cos: ([+-]?\d+\.\d+)",
                prompt,
            )
            sine_match = re.search(
                r"moment_1_sin: ([+-]?\d+\.\d+)",
                prompt,
            )
            if not cosine_match or not sine_match:
                raise ValueError("mock backend could not parse representation")
            vector = complex(
                float(cosine_match.group(1)),
                float(sine_match.group(1)),
            )
        direction = float(np.angle(vector))
        concentration = min(1.0, float(abs(vector)))
        mean_action = concentration * np.tanh(2.0 * np.sin(direction))
        stay_probability = 0.15
        moving_probability = 1.0 - stay_probability
        plus_probability = moving_probability * (1.0 + mean_action) / 2.0
        minus_probability = moving_probability - plus_probability

        rng = np.random.default_rng(seed)
        label = rng.choice(
            ["retard", "stay", "advance"],
            p=[minus_probability, stay_probability, plus_probability],
        )
        raw = json.dumps({"social_action": str(label)}, separators=(",", ":"))
        return BackendResult(
            raw_text=raw,
            input_tokens=max(1, (len(prompt) + 3) // 4),
            output_tokens=max(1, (len(raw) + 3) // 4),
        )


@dataclass
class OpenAIResponseBackend:
    """OpenAI or OpenAI-compatible chat-completions backend."""

    model: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: float
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    name: str = "openai"

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI backend requires: py -m pip install -e .[openai]"
            ) from exc

        # Legacy-compatible Dropbox-relative .env search when key not yet set.
        if not os.environ.get(self.api_key_env, "").strip():
            from circlemap.envload import load_project_env

            load_project_env()

        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key and not self.base_url:
            raise RuntimeError(
                f"{self.api_key_env} is not configured "
                "(searched pilot/.env, llm_emergent/.env, "
                "research_current/gen_ai_logi/rq1/.env, "
                "research_current/gen_ai_logi/.env)"
            )
        self._client = OpenAI(
            api_key=api_key or "local-openai-compatible",
            base_url=self.base_url,
        )

    def call(self, prompt: str, *, seed: int) -> BackendResult:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_completion_tokens=self.max_output_tokens,
                reasoning_effort="none",
                response_format={"type": "json_object"},
                seed=seed,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            message = str(exc).lower()
            if any(
                marker in message
                for marker in (
                    "insufficient_quota",
                    "billing",
                    "exceeded your current quota",
                )
            ):
                raise QuotaError(str(exc)) from exc
            raise

        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return BackendResult(
            raw_text=text,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(
                getattr(usage, "completion_tokens", 0) or 0
            ),
        )


@dataclass
class AnthropicResponseBackend:
    """Anthropic Messages API backend (Claude family)."""

    model: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: float
    api_key_env: str = "ANTHROPIC_API_KEY"
    name: str = "anthropic"

    def __post_init__(self) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic backend requires: py -m pip install anthropic"
            ) from exc
        if not os.environ.get(self.api_key_env, "").strip():
            from circlemap.envload import load_project_env

            load_project_env()
        if not os.environ.get(self.api_key_env, "").strip():
            raise RuntimeError(f"{self.api_key_env} is not configured")

    def call(self, prompt: str, *, seed: int) -> BackendResult:
        # Anthropic Messages API has no request seed; task_id/sample_seed
        # remain for bookkeeping only.
        del seed
        import anthropic

        client = anthropic.Anthropic(
            api_key=os.environ[self.api_key_env].strip(),
            timeout=self.timeout_seconds,
        )
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=max(self.max_output_tokens, 64),
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            message = str(exc).lower()
            if any(
                m in message
                for m in ("insufficient_quota", "billing", "credit", "rate_limit")
            ):
                # rate_limit is often transient; only hard-quota-like strings
                # become QuotaError. Keep rate_limit retryable via attempts.
                if "rate_limit" not in message:
                    raise QuotaError(str(exc)) from exc
            raise
        text = "".join(
            getattr(block, "text", "")
            for block in resp.content
            if getattr(block, "type", None) == "text"
        )
        usage = getattr(resp, "usage", None)
        return BackendResult(
            raw_text=text,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )


@dataclass
class GoogleResponseBackend:
    """Google Gemini backend via google-genai (thinking disabled)."""

    model: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: float
    api_key_env: str = "GOOGLE_API_KEY"
    name: str = "google"

    def __post_init__(self) -> None:
        if not os.environ.get(self.api_key_env, "").strip():
            from circlemap.envload import load_project_env

            load_project_env()
        if not os.environ.get(self.api_key_env, "").strip():
            raise RuntimeError(f"{self.api_key_env} is not configured")
        try:
            from google import genai  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Google backend requires: py -m pip install google-genai"
            ) from exc

    def call(self, prompt: str, *, seed: int) -> BackendResult:
        from google import genai
        from google.genai import types

        api_key = os.environ[self.api_key_env].strip()
        client = genai.Client(api_key=api_key)
        max_out = max(int(self.max_output_tokens), 256)
        base = {
            "temperature": self.temperature,
            "max_output_tokens": max_out,
            "response_mime_type": "application/json",
        }
        resp = None
        last_exc: Exception | None = None
        for cfg in (
            {**base, "seed": int(seed) % 2_147_483_647,
             "thinking_config": types.ThinkingConfig(thinking_budget=0)},
            {**base, "thinking_config": types.ThinkingConfig(thinking_budget=0)},
            {**base, "seed": int(seed) % 2_147_483_647},
            base,
        ):
            try:
                resp = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**cfg),
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
        if resp is None:
            assert last_exc is not None
            message = str(last_exc).lower()
            if any(
                m in message
                for m in (
                    "resource_exhausted",
                    "exceeded your current quota",
                    "insufficient_quota",
                    "billing",
                )
            ):
                raise QuotaError(str(last_exc)) from last_exc
            raise last_exc
        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        return BackendResult(
            raw_text=text,
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        )
