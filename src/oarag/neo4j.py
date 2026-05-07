from __future__ import annotations

import socket
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

from .config import Neo4jConfig, load_neo4j_config


DEFAULT_HEALTH_TIMEOUT_SECONDS = 2.0


class Neo4jUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class Neo4jEndpoint:
    host: str
    port: int


def endpoint_from_uri(uri: str) -> Neo4jEndpoint:
    parsed = urllib.parse.urlparse(uri)
    if not parsed.scheme:
        raise ValueError(f"Neo4j URI must include a scheme, for example bolt://127.0.0.1:7687: {uri}")
    if not parsed.hostname:
        raise ValueError(f"Neo4j URI must include a host: {uri}")
    return Neo4jEndpoint(host=parsed.hostname, port=parsed.port or default_port(parsed.scheme))


def default_port(scheme: str) -> int:
    normalized = scheme.lower()
    if normalized in {"bolt", "neo4j"}:
        return 7687
    if normalized == "http":
        return 7474
    if normalized == "https":
        return 7473
    raise ValueError(f"Unsupported Neo4j URI scheme: {scheme}")


def check_neo4j_health(
    config: Neo4jConfig | None = None,
    *,
    timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    active_config = config or load_neo4j_config()
    endpoint = endpoint_from_uri(active_config.uri)
    started_at = time.monotonic()
    try:
        with socket.create_connection((endpoint.host, endpoint.port), timeout=timeout_seconds):
            pass
    except OSError as exc:
        raise Neo4jUnavailableError(
            "Neo4j is not reachable at "
            f"{active_config.uri} ({endpoint.host}:{endpoint.port}). "
            "Start the local runtime with `docker compose up neo4j` or set "
            "OARAG_NEO4J_URI, OARAG_NEO4J_USER, OARAG_NEO4J_PASSWORD, and "
            "OARAG_NEO4J_DATABASE for your environment."
        ) from exc
    latency_ms = round((time.monotonic() - started_at) * 1000, 2)
    return {
        "ok": True,
        "uri": active_config.uri,
        "user": active_config.user,
        "database": active_config.database,
        "endpoint": {"host": endpoint.host, "port": endpoint.port},
        "latency_ms": latency_ms,
    }
