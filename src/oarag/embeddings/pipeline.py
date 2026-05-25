from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oarag.core.io import write_json, write_jsonl
from oarag.embeddings.cache import CacheStats, EmbeddingCache, EmbeddingCacheKey
from oarag.embeddings.manifest import (
    VECTOR_RECORD_SCHEMA_VERSION,
    build_vector_manifest,
    public_embedding_summary,
)
from oarag.embeddings.providers import (
    EmbeddingDimensionError,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmptyEmbeddingTextError,
    validate_embedding_vector,
)


DEFAULT_EMBEDDING_SOURCE_FIELDS = [
    "semantic_text",
    "transcript_window_text",
    "transcript_text",
    "normalized_text",
    "text",
    "visual_entities.text",
    "visual_entities.visual_description",
    "visual_description",
]


@dataclass
class _PreparedEmbeddingRecord:
    record_id: str
    text: str
    text_hash: str
    source_fields: list[str]
    cache_status: str
    vector: list[float] | None = None


def build_embedding_vectors(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    provider: EmbeddingProvider,
    embedder: str = "default",
    id_field: str = "segment_id",
    source_fields: Sequence[str] | None = None,
    cache_dir: Path | None = None,
    batch_size: int = 64,
    limit: int | None = None,
) -> dict[str, Any]:
    resolved_input_path = input_path.expanduser().resolve()
    resolved_output_path = output_path.expanduser().resolve()
    resolved_manifest_path = manifest_path.expanduser().resolve()
    resolved_source_fields = _source_fields(source_fields)
    resolved_embedder = _non_empty(embedder, field_name="embedder")
    resolved_id_field = _non_empty(id_field, field_name="id_field")
    resolved_batch_size = _positive_int(batch_size, field_name="batch_size")
    resolved_limit = None if limit is None else _positive_int(limit, field_name="limit")

    provider_metadata = provider.public_metadata()
    provider_name = _non_empty(provider_metadata.get("name"), field_name="provider name")
    provider_model = _non_empty(provider_metadata.get("model"), field_name="provider model")
    dimensions = _positive_int(provider_metadata.get("dimensions"), field_name="dimensions")
    cache = EmbeddingCache(cache_dir)
    cache_stats = CacheStats(enabled=cache.enabled)

    prepared_records: list[_PreparedEmbeddingRecord] = []
    misses: list[_PreparedEmbeddingRecord] = []
    for line_number, document in _iter_jsonl_documents(resolved_input_path, limit=resolved_limit):
        record_id = _record_id(document, id_field=resolved_id_field, line_number=line_number)
        text, used_source_fields = _embedding_text_and_fields(document, resolved_source_fields)
        if not text:
            raise EmptyEmbeddingTextError(
                f"embedding text is empty for record '{record_id}' using source fields "
                f"{resolved_source_fields}"
            )
        text_hash = stable_text_hash(text)
        cache_key = EmbeddingCacheKey(
            provider=provider_name,
            model=provider_model,
            dimensions=dimensions,
            text_hash=text_hash,
        )
        cached = cache.get(cache_key)
        if cached is None:
            cache_stats.misses += 1
            record = _PreparedEmbeddingRecord(
                record_id=record_id,
                text=text,
                text_hash=text_hash,
                source_fields=used_source_fields,
                cache_status="miss",
            )
            misses.append(record)
        else:
            cache_stats.hits += 1
            record = _PreparedEmbeddingRecord(
                record_id=record_id,
                text=text,
                text_hash=text_hash,
                source_fields=used_source_fields,
                cache_status="hit",
                vector=cached,
            )
        prepared_records.append(record)

    provider_batches = 0
    provider_records = 0
    for batch in _batches(misses, resolved_batch_size):
        provider_batches += 1
        provider_records += len(batch)
        try:
            vectors = provider.embed_batch([record.text for record in batch])
        except EmbeddingDimensionError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError(
                f"embedding provider '{provider_name}' failed for a batch of {len(batch)} records"
            ) from exc
        if len(vectors) != len(batch):
            raise EmbeddingProviderError(
                f"embedding provider '{provider_name}' returned {len(vectors)} vectors "
                f"for {len(batch)} records"
            )
        for record, vector in zip(batch, vectors, strict=True):
            normalized = validate_embedding_vector(
                vector,
                dimensions=dimensions,
                context=f"record '{record.record_id}'",
            )
            record.vector = normalized
            cache.set(
                EmbeddingCacheKey(
                    provider=provider_name,
                    model=provider_model,
                    dimensions=dimensions,
                    text_hash=record.text_hash,
                ),
                normalized,
            )
            if cache.enabled:
                cache_stats.writes += 1

    output_rows = [
        _vector_output_record(
            record,
            provider=provider_name,
            model=provider_model,
            dimensions=dimensions,
            embedder=resolved_embedder,
        )
        for record in prepared_records
    ]
    write_jsonl(resolved_output_path, output_rows)

    manifest_records = [
        {
            "record_id": record.record_id,
            "text_hash": record.text_hash,
            "source_fields": record.source_fields,
            "cache": record.cache_status,
        }
        for record in prepared_records
    ]
    manifest = build_vector_manifest(
        provider={
            "name": provider_name,
            "model": provider_model,
            "dimensions": dimensions,
        },
        embedder=resolved_embedder,
        source_fields=resolved_source_fields,
        records=manifest_records,
        cache_stats=cache_stats.to_dict(),
        provider_batches=provider_batches,
        provider_records=provider_records,
    )
    write_json(resolved_manifest_path, manifest)
    return public_embedding_summary(
        manifest=manifest,
        output_name=resolved_output_path.name,
        manifest_name=resolved_manifest_path.name,
    )


def stable_text_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector_output_record(
    record: _PreparedEmbeddingRecord,
    *,
    provider: str,
    model: str,
    dimensions: int,
    embedder: str,
) -> dict[str, Any]:
    if record.vector is None:
        raise EmbeddingProviderError(f"record '{record.record_id}' does not have a vector")
    return {
        "schema_version": VECTOR_RECORD_SCHEMA_VERSION,
        "record_id": record.record_id,
        "embedder": embedder,
        "provider": provider,
        "model": model,
        "dimensions": dimensions,
        "text_hash": record.text_hash,
        "source_fields": record.source_fields,
        "_vectors": {
            embedder: record.vector,
        },
    }


def _iter_jsonl_documents(path: Path, *, limit: int | None) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        yielded = 0
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at input line {line_number}")
            yield line_number, payload
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _embedding_text_and_fields(
    document: dict[str, Any],
    source_fields: list[str],
) -> tuple[str, list[str]]:
    pieces: list[str] = []
    used_source_fields: list[str] = []
    for source_field in source_fields:
        values = [
            _compact_text(str(value))
            for value in _flatten_field_values(document, source_field.split("."))
            if value not in (None, "")
        ]
        values = [value for value in values if value]
        if not values:
            continue
        pieces.extend(values)
        used_source_fields.append(source_field)
    return _compact_text(" ".join(pieces)), used_source_fields


def _flatten_field_values(value: Any, path: list[str]) -> list[Any]:
    if not path:
        if isinstance(value, list):
            flattened: list[Any] = []
            for item in value:
                flattened.extend(_flatten_field_values(item, []))
            return flattened
        return [value]
    if isinstance(value, dict):
        return _flatten_field_values(value.get(path[0]), path[1:])
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(_flatten_field_values(item, path))
        return flattened
    return []


def _record_id(document: dict[str, Any], *, id_field: str, line_number: int) -> str:
    values = _flatten_field_values(document, id_field.split("."))
    for value in values:
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"embedding record at input line {line_number} is missing id field '{id_field}'")


def _batches(records: Sequence[_PreparedEmbeddingRecord], batch_size: int) -> Iterator[list[_PreparedEmbeddingRecord]]:
    for offset in range(0, len(records), batch_size):
        yield list(records[offset : offset + batch_size])


def _source_fields(fields: Sequence[str] | None) -> list[str]:
    candidates = list(fields or DEFAULT_EMBEDDING_SOURCE_FIELDS)
    normalized: list[str] = []
    seen: set[str] = set()
    for field in candidates:
        value = _non_empty(field, field_name="source field")
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    if not normalized:
        raise ValueError("at least one source field is required")
    return normalized


def _compact_text(value: str) -> str:
    return " ".join(value.split())


def _non_empty(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _positive_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value
