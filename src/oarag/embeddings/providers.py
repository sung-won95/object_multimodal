from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


OPENAI_COMPATIBLE_PROVIDER = "openai_compatible_http"


class EmbeddingError(RuntimeError):
    """Base class for embedding pipeline errors."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when a provider request or response fails."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when vectors do not match the configured dimensions."""


class EmptyEmbeddingTextError(EmbeddingError, ValueError):
    """Raised when a record has no embeddable text."""


class EmbeddingProvider(Protocol):
    @property
    def provider_name(self) -> str:
        ...

    @property
    def model(self) -> str:
        ...

    @property
    def dimensions(self) -> int:
        ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def public_metadata(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class OpenAICompatibleEmbeddingProvider:
    model: str
    dimensions: int
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    timeout_seconds: float = 60.0

    @property
    def provider_name(self) -> str:
        return OPENAI_COMPATIBLE_PROVIDER

    def __post_init__(self) -> None:
        if not str(self.model).strip():
            raise ValueError("embedding model must not be empty")
        _validate_dimensions(self.dimensions)
        if not str(self.base_url).strip():
            raise ValueError("embedding base_url must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("embedding provider timeout_seconds must be > 0")

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        input_texts = [str(text) for text in texts]
        if not input_texts:
            return []

        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_texts,
            "dimensions": self.dimensions,
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "oarag-embedding-pipeline/1",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            _embedding_endpoint(self.base_url),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _safe_http_error_detail(exc)
            raise EmbeddingProviderError(
                f"{self.provider_name} request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise EmbeddingProviderError(
                f"{self.provider_name} request failed before receiving a response"
            ) from exc
        except json.JSONDecodeError as exc:
            raise EmbeddingProviderError(
                f"{self.provider_name} returned a non-JSON embedding response"
            ) from exc

        return _vectors_from_openai_response(
            response_payload,
            expected_count=len(input_texts),
            dimensions=self.dimensions,
            provider_name=self.provider_name,
        )

    def public_metadata(self) -> dict[str, Any]:
        return {
            "name": self.provider_name,
            "model": self.model,
            "dimensions": self.dimensions,
        }


def validate_embedding_vector(
    vector: Sequence[Any],
    *,
    dimensions: int,
    context: str,
) -> list[float]:
    expected_dimensions = _validate_dimensions(dimensions)
    if len(vector) != expected_dimensions:
        raise EmbeddingDimensionError(
            f"embedding dimension mismatch for {context}: "
            f"expected {expected_dimensions}, got {len(vector)}"
        )
    normalized: list[float] = []
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingDimensionError(
                f"embedding vector for {context} has a non-numeric value at offset {index}"
            )
        normalized.append(float(value))
    return normalized


def _vectors_from_openai_response(
    payload: Any,
    *,
    expected_count: int,
    dimensions: int,
    provider_name: str,
) -> list[list[float]]:
    if not isinstance(payload, dict):
        raise EmbeddingProviderError(f"{provider_name} response must be a JSON object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise EmbeddingProviderError(f"{provider_name} response is missing a data array")
    if len(data) != expected_count:
        raise EmbeddingProviderError(
            f"{provider_name} returned {len(data)} embeddings for {expected_count} inputs"
        )

    ordered_items = sorted(data, key=_response_item_index)
    vectors: list[list[float]] = []
    for fallback_index, item in enumerate(ordered_items):
        if not isinstance(item, dict):
            raise EmbeddingProviderError(
                f"{provider_name} response data[{fallback_index}] must be an object"
            )
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise EmbeddingProviderError(
                f"{provider_name} response data[{fallback_index}] is missing embedding"
            )
        vectors.append(
            validate_embedding_vector(
                embedding,
                dimensions=dimensions,
                context=f"{provider_name} response item {fallback_index}",
            )
        )
    return vectors


def _response_item_index(item: Any) -> int:
    if isinstance(item, dict) and isinstance(item.get("index"), int):
        return item["index"]
    return 0


def _embedding_endpoint(base_url: str) -> str:
    normalized = str(base_url).strip().rstrip("/")
    if normalized.endswith("/embeddings"):
        return normalized
    return f"{normalized}/embeddings"


def _safe_http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read(512).decode("utf-8", errors="replace")
    except Exception:
        return "provider returned an error body that could not be read"
    compact = " ".join(raw.split())
    if not compact:
        return "provider returned an empty error body"
    return compact[:240]


def _validate_dimensions(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("embedding dimensions must be a positive integer")
    return value
