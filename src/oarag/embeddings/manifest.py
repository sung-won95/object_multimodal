from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from oarag.core.io import write_json
from oarag.embeddings.cache import JsonlEmbeddingCache
from oarag.embeddings.providers import EmbeddingProvider, cache_key_for_text
from oarag.retrieval.vectors import text_from_document_fields


VECTOR_MANIFEST_SCHEMA_VERSION = "oarag-vector-manifest-v1"
QUERY_VECTOR_MANIFEST_SCHEMA_VERSION = "oarag-query-vectors-v1"
SUPPORTED_MANIFEST_KINDS = ("document", "query")


@dataclass(frozen=True)
class LoadedVectorRecord:
    record_id: str
    embedder: str
    dimensions: int
    vector: list[float]
    text_hash: str | None


@dataclass(frozen=True)
class LoadedVectorManifest:
    path: Path
    content_hash: str
    schema_version: str | None
    kind: str | None
    embedder: str | None
    provider: dict[str, Any]
    dimensions_by_embedder: dict[str, int]
    records_by_embedder: dict[str, dict[str, LoadedVectorRecord]]
    record_count: int

    @property
    def embedder_names(self) -> list[str]:
        return sorted(self.records_by_embedder)

    def get(self, *, embedder: str, record_ids: Iterable[str]) -> LoadedVectorRecord | None:
        records = self.records_by_embedder.get(embedder, {})
        for record_id in record_ids:
            vector_record = records.get(record_id)
            if vector_record is not None:
                return vector_record
        return None

    def public_summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "source": "manifest",
            "provider": _public_provider_config(self.provider),
            "dimensions_by_embedder": dict(self.dimensions_by_embedder),
            "embedder_names": self.embedder_names,
            "record_count": self.record_count,
            "hash": self.content_hash,
        }


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
        "provider": _public_provider_config(provider),
        "dimensions": dimensions,
        "record_count": len(vectors),
        "vector_count": len(vectors),
        "text_hashes_sha256": _hash_text_hashes(vectors),
        "cache": manifest.get("cache"),
    }
    if output_path is not None:
        summary["output_file"] = output_path.name
    return summary


def _public_provider_config(provider: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "provider",
        "model",
        "dimensions",
        "api_base",
        "local_files_only",
        "quality_claim",
        "purpose",
    )
    return {key: provider.get(key) for key in keys if key in provider}


def load_vector_manifest(path: Path) -> LoadedVectorManifest:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"vector manifest not found: {resolved_path}")
    raw_text = resolved_path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse vector manifest JSON at {resolved_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"vector manifest must be a JSON object: {resolved_path}")

    default_embedder = _non_empty_string(payload.get("embedder"))
    default_dimensions = _optional_dimensions(payload.get("dimensions"))
    provider = payload.get("provider") if isinstance(payload.get("provider"), dict) else {}
    records_by_embedder: dict[str, dict[str, LoadedVectorRecord]] = {}
    dimensions_by_embedder: dict[str, int] = {}
    for raw_record in _manifest_records(payload):
        record_id = _record_id(raw_record)
        embedder = _non_empty_string(raw_record.get("embedder")) or default_embedder
        if embedder is None:
            raise ValueError(f"vector manifest record '{record_id}' is missing embedder")
        expected_dimensions = _optional_dimensions(raw_record.get("dimensions")) or default_dimensions
        vector = _normalize_vector(
            raw_record.get("vector"),
            dimensions=expected_dimensions,
            field_name=f"vector manifest record '{record_id}' for embedder '{embedder}'",
        )
        dimensions = len(vector)
        previous_dimensions = dimensions_by_embedder.get(embedder)
        if previous_dimensions is not None and previous_dimensions != dimensions:
            raise ValueError(
                f"vector manifest dimension mismatch for embedder '{embedder}': "
                f"expected {previous_dimensions}, got {dimensions}"
            )
        dimensions_by_embedder[embedder] = dimensions
        embedder_records = records_by_embedder.setdefault(embedder, {})
        if record_id in embedder_records:
            raise ValueError(
                f"duplicate vector manifest record id '{record_id}' for embedder '{embedder}'"
            )
        text_hash = _non_empty_string(raw_record.get("text_hash"))
        embedder_records[record_id] = LoadedVectorRecord(
            record_id=record_id,
            embedder=embedder,
            dimensions=dimensions,
            vector=vector,
            text_hash=text_hash,
        )
    record_count = sum(len(records) for records in records_by_embedder.values())
    if record_count == 0:
        raise ValueError(f"vector manifest contains no records: {resolved_path}")
    return LoadedVectorManifest(
        path=resolved_path,
        content_hash=content_hash,
        schema_version=_non_empty_string(payload.get("schema_version")),
        kind=_non_empty_string(payload.get("kind")),
        embedder=default_embedder,
        provider=dict(provider),
        dimensions_by_embedder=dimensions_by_embedder,
        records_by_embedder=records_by_embedder,
        record_count=record_count,
    )


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
    if isinstance(records, dict):
        return [
            {"id": str(record_id), **record}
            for record_id, record in records.items()
            if isinstance(record, dict)
        ]
    queries = manifest.get("queries")
    if isinstance(queries, dict):
        return [
            {"id": str(record_id), **record}
            for record_id, record in queries.items()
            if isinstance(record, dict)
        ]
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


def _normalize_vector(
    vector: Any,
    *,
    dimensions: int | None,
    field_name: str,
) -> list[float]:
    if not isinstance(vector, list) or isinstance(vector, str | bytes) or not vector:
        raise ValueError(f"{field_name} must be a non-empty JSON array")
    normalized: list[float] = []
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{field_name} item {index} must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field_name} item {index} must be a finite number")
        normalized.append(number)
    if dimensions is not None and len(normalized) != dimensions:
        raise ValueError(
            f"{field_name} dimension mismatch: expected {dimensions}, got {len(normalized)}"
        )
    return normalized


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


def _record_id(record: dict[str, Any]) -> str:
    for field in ("id", "record_id", "segment_id", "window_id", "entity_id", "local_entity_id", "query_id"):
        value = record.get(field)
        if value not in (None, ""):
            return str(value)
    raise ValueError("vector manifest record is missing id")


def _optional_dimensions(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("vector manifest dimensions must be a positive integer")
    return value


def _non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
