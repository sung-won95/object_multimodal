from pathlib import Path

from oarag.config import (
    DEFAULT_NEO4J_DATABASE,
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USER,
    dotenv_default,
    load_neo4j_config,
    normalize_env_value,
)


def test_dotenv_default_reads_matching_key(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        """
        # Local defaults
        OARAG_STT_LANGUAGE="en"
        OTHER=value
        """,
        encoding="utf-8",
    )

    assert dotenv_default("OARAG_STT_LANGUAGE", env_path=env_path) == "en"


def test_normalize_env_value_ignores_blank_values() -> None:
    assert normalize_env_value("  ") is None
    assert normalize_env_value(" ko ") == "ko"


def test_load_neo4j_config_uses_defaults(monkeypatch) -> None:
    for name in (
        "OARAG_NEO4J_URI",
        "OARAG_NEO4J_USER",
        "OARAG_NEO4J_PASSWORD",
        "OARAG_NEO4J_DATABASE",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_neo4j_config()

    assert config.uri == DEFAULT_NEO4J_URI
    assert config.user == DEFAULT_NEO4J_USER
    assert config.password == DEFAULT_NEO4J_PASSWORD
    assert config.database == DEFAULT_NEO4J_DATABASE


def test_load_neo4j_config_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("OARAG_NEO4J_URI", "bolt://neo4j.local:7687")
    monkeypatch.setenv("OARAG_NEO4J_USER", "reader")
    monkeypatch.setenv("OARAG_NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("OARAG_NEO4J_DATABASE", "lectures")

    config = load_neo4j_config()

    assert config.uri == "bolt://neo4j.local:7687"
    assert config.user == "reader"
    assert config.password == "secret"
    assert config.database == "lectures"
    assert config.redacted()["password"] == "***"
