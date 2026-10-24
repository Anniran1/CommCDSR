"""Thread-safe accounting for OpenAI-compatible API usage payloads."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from threading import Lock
from typing import Any, Mapping


class TokenUsageTracker:
    def __init__(self, path: str | Path | None = None) -> None:
        self._lock = Lock()
        self._path = Path(path) if path is not None else None
        self._values: dict[str, dict[str, int | str]] = defaultdict(
            lambda: {
                "model": "",
                "operation": "",
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        )
        self._session_values: dict[str, dict[str, int | str]] = defaultdict(
            lambda: {
                "model": "", "operation": "", "requests": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            }
        )
        if self._path is not None and self._path.exists():
            for key, value in json.loads(self._path.read_text(encoding="utf-8")).items():
                self._values[key].update(value)

    @staticmethod
    def _add(value: dict[str, int | str], *, model: str, operation: str,
             prompt: int, completion: int, total: int) -> None:
        value["model"] = model
        value["operation"] = operation
        value["requests"] = int(value["requests"]) + 1
        value["prompt_tokens"] = int(value["prompt_tokens"]) + prompt
        value["completion_tokens"] = int(value["completion_tokens"]) + completion
        value["total_tokens"] = int(value["total_tokens"]) + total

    def record(self, response: Mapping[str, Any], *, model: str, operation: str) -> None:
        usage = response.get("usage") or {}
        prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        completion = int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
        total = int(usage.get("total_tokens", prompt + completion) or 0)
        key = f"{operation}:{model}"
        with self._lock:
            self._add(
                self._values[key], model=model, operation=operation,
                prompt=prompt, completion=completion, total=total,
            )
            self._add(
                self._session_values[key], model=model, operation=operation,
                prompt=prompt, completion=completion, total=total,
            )
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self._path.with_suffix(self._path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(self._values, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                temporary.replace(self._path)

    def snapshot(self) -> dict[str, dict[str, int | str]]:
        with self._lock:
            return {key: dict(value) for key, value in self._values.items()}

    def session_snapshot(self) -> dict[str, dict[str, int | str]]:
        with self._lock:
            return {key: dict(value) for key, value in self._session_values.items()}
