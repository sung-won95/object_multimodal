import pytest

from oarag.config import Neo4jConfig
from oarag.neo4j import (
    Neo4jUnavailableError,
    check_neo4j_health,
    default_port,
    endpoint_from_uri,
)


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


def test_endpoint_from_uri_uses_explicit_port() -> None:
    endpoint = endpoint_from_uri("bolt://graph.local:17687")

    assert endpoint.host == "graph.local"
    assert endpoint.port == 17687


def test_endpoint_from_uri_uses_scheme_default_port() -> None:
    assert endpoint_from_uri("bolt://127.0.0.1").port == 7687
    assert endpoint_from_uri("http://127.0.0.1").port == 7474
    assert endpoint_from_uri("https://127.0.0.1").port == 7473


def test_default_port_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError, match="Unsupported Neo4j URI scheme"):
        default_port("ftp")


def test_check_neo4j_health_returns_status_for_reachable_socket(monkeypatch) -> None:
    calls = []

    def fake_create_connection(address, timeout):
        calls.append((address, timeout))
        return FakeConnection()

    monkeypatch.setattr("socket.create_connection", fake_create_connection)
    config = Neo4jConfig(
        uri="bolt://127.0.0.1:17687",
        user="neo4j",
        password="dev-password",
        database="neo4j",
    )
    status = check_neo4j_health(config, timeout_seconds=1.0)

    assert status["ok"] is True
    assert status["uri"] == "bolt://127.0.0.1:17687"
    assert status["user"] == "neo4j"
    assert status["database"] == "neo4j"
    assert status["endpoint"] == {"host": "127.0.0.1", "port": 17687}
    assert calls == [(("127.0.0.1", 17687), 1.0)]


def test_check_neo4j_health_raises_clear_error_when_unreachable(monkeypatch) -> None:
    def fake_create_connection(address, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr("socket.create_connection", fake_create_connection)
    config = Neo4jConfig(
        uri="bolt://127.0.0.1:17687",
        user="neo4j",
        password="dev-password",
        database="neo4j",
    )

    with pytest.raises(Neo4jUnavailableError, match="docker compose up neo4j"):
        check_neo4j_health(config, timeout_seconds=0.1)
