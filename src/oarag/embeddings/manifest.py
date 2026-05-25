from __future__ import annotations

from typing import Any


VECTOR_MANIFEST_SCHEMA_VERSION = "oarag-vector-manifest-v1"
VECTOR_RECORD_SCHEMA_VERSION = "oarag-vector-record-v1"


def build_vector_manifest(
    *,
    provider: dict[str, Any],
    embedder: str,
    source_fields: list[str],
    records: list[dict[str, Any]],
    cache_stats: dict[str, Any],
    provider_batches: int,
    provider_records: int,
) -> dict[str, Any]:
    dimensions = provider["dimensions"]
    return {
        "schema_version": VECTOR_MANIFEST_SCHEMA_VERSION,
        "provider": provider["name"],
        "model": provider["model"],
        "dimensions": dimensions,
        "embedder": embedder,
        "text_hash": {
            "algorithm": "sha256",
            "prefix": "sha256:",
        },
        "source_fields": source_fields,
        "cache": cache_stats,
        "counts": {
            "records": len(records),
            "vectors": len(records),
            "provider_batches": provider_batches,
            "provider_records": provider_records,
            "cache_hits": cache_stats["hits"],
            "cache_misses": cache_stats["misses"],
        },
        "records": records,
    }


def public_embedding_summary(
    *,
    manifest: dict[str, Any],
    output_name: str,
    manifest_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "provider": manifest["provider"],
        "model": manifest["model"],
        "dimensions": manifest["dimensions"],
        "embedder": manifest["embedder"],
        "source_fields": manifest["source_fields"],
        "counts": manifest["counts"],
        "cache": manifest["cache"],
        "artifacts": {
            "vectors_jsonl": output_name,
            "manifest": manifest_name,
        },
        "privacy": {
            "raw_text": "excluded",
            "api_key": "excluded",
            "absolute_paths": "excluded",
            "raw_vectors": "excluded_from_summary",
        },
    }
