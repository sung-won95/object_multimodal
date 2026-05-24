from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any


DEFAULT_LOCAL_VECTOR_DIMENSIONS = 384
LOCAL_HASH_VECTOR_SOURCE = "local_hash_v1"
QUERY_VECTOR_MANIFEST_SCHEMA_VERSION = "oarag-query-vectors-v1"

_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣_]+")


def deterministic_text_vector(
    text: str,
    *,
    dimensions: int = DEFAULT_LOCAL_VECTOR_DIMENSIONS,
) -> list[float]:
    """Build a deterministic, dependency-free vector for reproducible smoke tests."""
    dims = _validate_dimensions(dimensions)
    tokens = _tokens(text)
    if not tokens:
        tokens = ["empty"]

    vector = [0.0] * dims
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for salt in range(4):
            offset = salt * 4
            bucket = int.from_bytes(digest[offset : offset + 4], "big") % dims
            sign = -1.0 if digest[16 + salt] & 1 else 1.0
            vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        bucket = int.from_bytes(hashlib.sha256(str(text).encode("utf-8")).digest()[:4], "big") % dims
        vector[bucket] = 1.0
        norm = 1.0
    return [round(value / norm, 6) for value in vector]


def text_from_document_fields(document: Mapping[str, Any], fields: list[str]) -> str:
    pieces: list[str] = []
    for field in fields:
        pieces.extend(_flatten_field_values(document, field.split(".")))
    return " ".join(piece for piece in (_compact_text(str(value)) for value in pieces) if piece)


def _flatten_field_values(value: Any, path: list[str]) -> list[Any]:
    if not path:
        if isinstance(value, list):
            flattened: list[Any] = []
            for item in value:
                flattened.extend(_flatten_field_values(item, []))
            return flattened
        return [value] if value not in (None, "") else []
    if isinstance(value, Mapping):
        return _flatten_field_values(value.get(path[0]), path[1:])
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(_flatten_field_values(item, path))
        return flattened
    return []


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def _compact_text(value: str) -> str:
    return " ".join(value.split())


def _validate_dimensions(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("vector dimensions must be a positive integer")
    return value
