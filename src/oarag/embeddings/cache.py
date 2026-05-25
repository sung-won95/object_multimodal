from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlEmbeddingCache:
    """Append-only JSONL cache keyed by provider/model/dimension/text hash."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.hits = 0
        self.misses = 0
        self._records: dict[str, list[float]] = {}
        if path is not None and path.exists():
            self._records = _load_cache(path)

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def get(self, key: str) -> list[float] | None:
        if key in self._records:
            self.hits += 1
            return list(self._records[key])
        self.misses += 1
        return None

    def set(self, key: str, vector: list[float], *, metadata: dict[str, Any]) -> None:
        self._records[key] = list(vector)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": key, "vector": vector, "metadata": metadata}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "entries_loaded": len(self._records),
            "file": self.path.name if self.path is not None else None,
        }


def _load_cache(path: Path) -> dict[str, list[float]]:
    records: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"embedding cache entry must be an object at {path}:{line_number}")
            key = payload.get("key")
            vector = payload.get("vector")
            if not isinstance(key, str) or not isinstance(vector, list):
                raise ValueError(f"embedding cache entry is missing key/vector at {path}:{line_number}")
            records[key] = [float(value) for value in vector]
    return records
