from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from oarag.core.io import write_json
from oarag.embeddings.cache import JsonlEmbeddingCache
from oarag.embeddings.providers import EmbeddingProvider, cache_key_for_text
from oarag.retrieval.vectors import text_from_document_fields


VECTOR_MANIFEST_SCHEMA_VERSION = "oarag-vector-manifest-v1"
QUERY_VECTOR_MANIFEST_SCHEMA_VERSION = "oarag-query-vectors-v1"
SUPPORTED_MANIFEST_KINDS = ("document", "query")


def load_input_records(path: Path, *, input_format: str = "auto") -> list[dict[str, Any]]:
    resolved_format = _resolve_input_format(path, input_format=input_format)
    if resolved_format == "jsonl":
        return list(_iter_jsonl(path))
    if resolved_format == "json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            return [_ensure_dict(row, path=path, index=index) for index, row in enumerate(loaded)]
        if isinstance(loaded, dict):
            for key in ("records", "documents", "queries"):
                value = loaded.get(key)
                if isinstance(value, list):
                    return [
                        _ensure_dict(row, path=path, index=index)
                        for index, row in enumerate(value)
                    ]
            raise ValueError("JSON input must be a list or contain records/documents/queries list")
        raise ValueError("JSON input must be an object or array")
    if resolved_format == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"unsupported input format: {input_format}")


def build_vector_manifest(
    *,
    records: Iterable[dict[str, Any]],
    provider: EmbeddingProvider,
    output_path: Path,
    id_field: str,
    text_fields: list[str],
    kind: str = "document",
    embedder: str = "default",
    batch_size: int = 64,
    cache: JsonlEmbeddingCache | None = None,
    fail_on_empty_text: bool = True,
    source_label: str | None = None,
) -> dict[str, Any]:
    if kind not in SUPPORTED_MANIFEST_KINDS:
        raise ValueError(f"manifest kind must be one of: {', '.join(SUPPORTED_MANIFEST_KINDS)}")
    if not id_field.strip():
        raise ValueError("id_field must not be empty")
    if not text_fields:
        raise ValueError("at least one text field is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    cache = cache or JsonlEmbeddingCache(None)
    prepared = [
        _prepare_record(
            record,
            index=index,
            id_field=id_field,
            text_fields=text_fields,
            fail_on_empty_text=fail_on_empty_text,
        )
        for index, record in enumerate(records)
    ]
    vectors_by_position: dict[int, list[float]] = {}
    pending: list[dict[str, Any]] = []
    for item in prepared:
        cached = cache.get(cache_key_for_text(provider=provider, text_hash=item["text_hash"], embedder=embedder))
        if cached is not None:
            vectors_by_position[item["position"]] = cached
        else:
            pending.append(item)

    for batch in _batches(pending, batch_size=batch_size):
        vectors = provider.embed_texts([item["text"] for item in batch])
        if len(vectors) != len(batch):
            raise ValueError("embedding provider returned an unexpected number of vectors")
        for item, vector in zip(batch, vectors, strict=True):
            dimensions = _validate_vector(vector, expected_dimensions=provider.dimensions)
            cache_key = cache_key_for_text(
                provider=provider,
                text_hash=item["text_hash"],
                embedder=embedder,
            )
            cache.set(
                cache_key,
                vector,
                metadata={
                    "provider": provider.provider_id,
                    "model": provider.model,
                    "dimensions": dimensions,
                    "text_hash": item["text_hash"],
                    "record_id": item["record_id"],
                },
            )
            vectors_by_position[item["position"]] = vector

    vector_records = [
        _manifest_vector_record(item, vectors_by_position[item["position"]], embedder=embedder)
        for item in prepared
    ]
    manifest = _manifest_payload(
        kind=kind,
        embedder=embedder,
        provider=provider,
        id_field=id_field,
        text_fields=text_fields,
        source_label=source_label,
        vector_records=vector_records,
        cache_summary=cache.summary(),
    )
    write_json(output_path, manifest)
    return public_manifest_summary(manifest, output_path=output_path)


def public_manifest_summary(manifest: dict[str, Any], *, output_path: Path | None = None) -> dict[str, Any]:
    vectors = _manifest_records(manifest)
    provider = manifest.get("provider") if isinstance(manifest.get("provider"), dict) else {}
    dimensions = manifest.get("dimensions")
    summary = {
        "schema_version": manifest.get("schema_version"),
        "kind": manifest.get("kind"),
        "embedder": manifest.get("embedder"),
        "provider": {
            "provider": provider.get("provider"),
            "model": provider.get("model"),
            "dimensions": provider.get("dimensions"),
        },
        "dimensions": dimensions,
        "record_count": len(vectors),
        "vector_count": len(vectors),
        "text_hashes_sha256": _hash_text_hashes(vectors),
        "cache": manifest.get("cache"),
    }
    if output_path is not None:
        summary["output_file"] = output_path.name
    return summary


def _manifest_payload(
    *,
    kind: str,
    embedder: str,
    provider: EmbeddingProvider,
    id_field: str,
    text_fields: list[str],
    source_label: str | None,
    vector_records: list[dict[str, Any]],
    cache_summary: dict[str, Any],
) -> dict[str, Any]:
    dimensions = _single_dimensions(vector_records)
    provider_config = provider.public_config()
    base = {
        "schema_version": VECTOR_MANIFEST_SCHEMA_VERSION,
        "kind": kind,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "embedder": embedder,
        "provider": provider_config,
        "dimensions": dimensions,
        "source": {
            "label": source_label,
            "id_field": id_field,
            "text_fields": text_fields,
        },
        "embedders": {
            embedder: {
                "source": "userProvided",
                "dimensions": dimensions,
                "provider": provider_config.get("provider"),
                "model": provider_config.get("model"),
            }
        },
        "counts": {
            "input_records": len(vector_records),
            "vector_records": len(vector_records),
        },
        "cache": cache_summary,
    }
    if kind == "query":
        base["schema_version"] = QUERY_VECTOR_MANIFEST_SCHEMA_VERSION
        base["queries"] = {
            str(record["id"]): {
                "vector": record["vector"],
                "embedder": record["embedder"],
                "dimensions": record["dimensions"],
                "text_hash": record["text_hash"],
                "source_fields": record["source_fields"],
            }
            for record in vector_records
        }
    else:
        base["records"] = vector_records
    return base


def _prepare_record(
    record: dict[str, Any],
    *,
    index: int,
    id_field: str,
    text_fields: list[str],
    fail_on_empty_text: bool,
) -> dict[str, Any]:
    record_id = record.get(id_field)
    if record_id is None or str(record_id).strip() == "":
        raise ValueError(f"record {index} is missing id field '{id_field}'")
    text = text_from_document_fields(record, text_fields)
    if not text and fail_on_empty_text:
        raise ValueError(f"record '{record_id}' has no text in fields: {', '.join(text_fields)}")
    return {
        "position": index,
        "record_id": str(record_id),
        "text": text,
        "text_hash": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_fields": list(text_fields),
    }


def _manifest_vector_record(
    item: dict[str, Any],
    vector: list[float],
    *,
    embedder: str,
) -> dict[str, Any]:
    dimensions = _validate_vector(vector)
    return {
        "id": item["record_id"],
        "embedder": embedder,
        "dimensions": dimensions,
        "vector": vector,
        "text_hash": item["text_hash"],
        "source_fields": item["source_fields"],
    }


def _manifest_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = manifest.get("records")
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]
    queries = manifest.get("queries")
    if isinstance(queries, dict):
        return [record for record in queries.values() if isinstance(record, dict)]
    return []


def _single_dimensions(records: list[dict[str, Any]]) -> int:
    dimensions = {record.get("dimensions") for record in records}
    if len(dimensions) != 1:
        raise ValueError("all vectors in a manifest must have the same dimensions")
    value = next(iter(dimensions))
    if not isinstance(value, int) or value <= 0:
        raise ValueError("manifest vector dimensions must be positive")
    return value


def _validate_vector(vector: list[float], *, expected_dimensions: int | None = None) -> int:
    if not isinstance(vector, list) or not vector:
        raise ValueError("embedding vector must be a non-empty list")
    normalized = [float(value) for value in vector]
    if expected_dimensions is not None and len(normalized) != expected_dimensions:
        raise ValueError(
            f"embedding vector dimension mismatch: got {len(normalized)}, expected {expected_dimensions}"
        )
    return len(normalized)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def _resolve_input_format(path: Path, *, input_format: str) -> str:
    if input_format != "auto":
        return input_format
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"cannot infer input format from suffix: {path}")


def _ensure_dict(value: Any, *, path: Path, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path} item {index}")
    return value


def _batches(records: list[dict[str, Any]], *, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def _hash_text_hashes(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record.get("text_hash") or "").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
