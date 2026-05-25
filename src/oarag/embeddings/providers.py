from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from oarag.integrations.meili import normalize_query_vector
from oarag.retrieval.vectors import deterministic_text_vector


DEFAULT_EMBEDDING_API_BASE = "https://api.openai.com/v1"
DEFAULT_EMBEDDING_API_KEY_ENV = "OARAG_EMBEDDING_API_KEY"
FALLBACK_EMBEDDING_API_KEY_ENV = "OPENAI_API_KEY"
ENV_EMBEDDING_API_BASE = "OARAG_EMBEDDING_API_BASE"
ENV_EMBEDDING_MODEL = "OARAG_EMBEDDING_MODEL"
DEFAULT_SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingProvider(Protocol):
    provider_id: str
    model: str
    dimensions: int | None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def public_config(self) -> dict[str, Any]:
        ...


@dataclass
class OpenAICompatibleEmbeddingProvider:
    model: str
    api_base: str = DEFAULT_EMBEDDING_API_BASE
    api_key: str | None = None
    dimensions: int | None = None
    timeout_seconds: float = 60.0
    user_agent: str = "object-aligned-rag/0.1"

    provider_id: str = "openai_compatible"

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        api_base: str | None = None,
        api_key_env: str = DEFAULT_EMBEDDING_API_KEY_ENV,
        dimensions: int | None = None,
    ) -> "OpenAICompatibleEmbeddingProvider":
        resolved_model = _non_empty(model) or _non_empty(os.environ.get(ENV_EMBEDDING_MODEL))
        if resolved_model is None:
            raise ValueError(
                "embedding model is required; pass --model or set OARAG_EMBEDDING_MODEL"
            )
        resolved_base = _non_empty(api_base) or _non_empty(os.environ.get(ENV_EMBEDDING_API_BASE))
        if resolved_base is None:
            resolved_base = DEFAULT_EMBEDDING_API_BASE
        key = _non_empty(os.environ.get(api_key_env)) or _non_empty(
            os.environ.get(FALLBACK_EMBEDDING_API_KEY_ENV)
        )
        if key is None:
            raise ValueError(
                f"embedding API key is required; set {api_key_env} or "
                f"{FALLBACK_EMBEDDING_API_KEY_ENV}"
            )
        return cls(
            model=resolved_model,
            api_base=resolved_base,
            api_key=key,
            dimensions=dimensions,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        request = urllib.request.Request(
            _join_url(self.api_base, "embeddings"),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key or ''}",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"embedding provider request failed with HTTP {exc.code}: {_redact(detail)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"embedding provider request failed: {exc.reason}") from exc
        loaded = json.loads(body)
        data = loaded.get("data") if isinstance(loaded, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError("embedding provider response count did not match input count")
        vectors: list[list[float]] = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"embedding provider response item {index} must be an object")
            vector = item.get("embedding")
            vectors.append(
                normalize_query_vector(
                    vector,
                    dimensions=self.dimensions,
                    field_name=f"embedding provider response item {index}",
                )
            )
        return vectors

    def public_config(self) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "model": self.model,
            "dimensions": self.dimensions,
            "api_base": _public_api_base(self.api_base),
        }


@dataclass
class DeterministicFixtureEmbeddingProvider:
    model: str = "deterministic_fixture_v1"
    dimensions: int = 8
    provider_id: str = "deterministic_fixture"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [
            deterministic_text_vector(text, dimensions=self.dimensions)
            for text in texts
        ]

    def public_config(self) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "model": self.model,
            "dimensions": self.dimensions,
            "quality_claim": "none",
            "purpose": "test_fixture",
        }


@dataclass
class SentenceTransformersEmbeddingProvider:
    model: str = DEFAULT_SENTENCE_TRANSFORMERS_MODEL
    dimensions: int | None = None
    local_files_only: bool = False
    normalize_embeddings: bool = True
    provider_id: str = "sentence_transformers"

    def __post_init__(self) -> None:
        self._model = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        encoded = model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = encoded.tolist()
        return [
            normalize_query_vector(
                vector,
                dimensions=self.dimensions,
                field_name=f"sentence-transformers embedding {index}",
            )
            for index, vector in enumerate(vectors)
        ]

    def public_config(self) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "model": self.model,
            "dimensions": self.dimensions,
            "local_files_only": self.local_files_only,
            "quality_claim": "provider_embedding",
        }

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers provider requires the optional "
                "`sentence_transformers` package"
            ) from exc
        self._model = SentenceTransformer(
            self.model,
            local_files_only=self.local_files_only,
        )
        return self._model


def cache_key_for_text(
    *,
    provider: EmbeddingProvider,
    text_hash: str,
    embedder: str,
) -> str:
    dimensions = provider.dimensions if provider.dimensions is not None else "provider_default"
    raw = "|".join(
        [
            provider.provider_id,
            provider.model,
            str(dimensions),
            embedder,
            text_hash,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _join_url(api_base: str, suffix: str) -> str:
    return f"{api_base.rstrip('/')}/{suffix.lstrip('/')}"


def _public_api_base(api_base: str) -> str:
    parsed = urllib.parse.urlsplit(api_base)
    if parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return "<custom>"


def _redact(value: str) -> str:
    for env_name in (DEFAULT_EMBEDDING_API_KEY_ENV, FALLBACK_EMBEDDING_API_KEY_ENV):
        secret = os.environ.get(env_name)
        if secret:
            value = value.replace(secret, "<redacted>")
    return value


def _non_empty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
