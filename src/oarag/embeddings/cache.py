from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oarag.embeddings.providers import EmbeddingDimensionError, validate_embedding_vector


EMBEDDING_CACHE_ENTRY_SCHEMA_VERSION = "oarag-embedding-cache-entry-v1"


@dataclass
class CacheStats:
    enabled: bool
    hits: int = 0
    misses: int = 0
    writes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }


@dataclass(frozen=True)
class EmbeddingCacheKey:
    provider: str
    model: str
    dimensions: int
    text_hash: str

    def fingerprint(self) -> str:
        payload = {
            "schema_version": EMBEDDING_CACHE_ENTRY_SCHEMA_VERSION,
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "text_hash": self.text_hash,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class EmbeddingCache:
    def __init__(self, cache_dir: Path | None) -> None:
        self.cache_dir = cache_dir.expanduser().resolve() if cache_dir is not None else None

    @property
    def enabled(self) -> bool:
        return self.cache_dir is not None

    def get(self, key: EmbeddingCacheKey) -> list[float] | None:
        if self.cache_dir is None:
            return None
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EmbeddingDimensionError(f"embedding cache entry is not valid JSON: {path.name}") from exc
        if not isinstance(payload, dict):
            raise EmbeddingDimensionError(f"embedding cache entry must be an object: {path.name}")
        for field_name, expected in (
            ("schema_version", EMBEDDING_CACHE_ENTRY_SCHEMA_VERSION),
            ("provider", key.provider),
            ("model", key.model),
            ("dimensions", key.dimensions),
            ("text_hash", key.text_hash),
        ):
            if payload.get(field_name) != expected:
                raise EmbeddingDimensionError(
                    f"embedding cache metadata mismatch for {path.name}: {field_name}"
                )
        vector = payload.get("vector")
        if not isinstance(vector, list):
            raise EmbeddingDimensionError(f"embedding cache entry is missing vector: {path.name}")
        return validate_embedding_vector(
            vector,
            dimensions=key.dimensions,
            context=f"cache entry {path.name}",
        )

    def set(self, key: EmbeddingCacheKey, vector: list[float]) -> None:
        if self.cache_dir is None:
            return
        normalized = validate_embedding_vector(
            vector,
            dimensions=key.dimensions,
            context=f"cache write {key.text_hash}",
        )
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": EMBEDDING_CACHE_ENTRY_SCHEMA_VERSION,
            "provider": key.provider,
            "model": key.model,
            "dimensions": key.dimensions,
            "text_hash": key.text_hash,
            "vector": normalized,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    def _path_for_key(self, key: EmbeddingCacheKey) -> Path:
        digest = key.fingerprint()
        assert self.cache_dir is not None
        return self.cache_dir / digest[:2] / f"{digest}.json"
