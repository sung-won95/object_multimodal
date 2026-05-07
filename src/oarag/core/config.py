from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MEILI_URL = "http://127.0.0.1:7700"
DEFAULT_MEILI_API_KEY = "dev-master-key"
ENV_STT_LANGUAGE = "OARAG_STT_LANGUAGE"
DEFAULT_NEO4J_URI = "bolt://127.0.0.1:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "dev-password"
DEFAULT_NEO4J_DATABASE = "neo4j"
ENV_NEO4J_URI = "OARAG_NEO4J_URI"
ENV_NEO4J_USER = "OARAG_NEO4J_USER"
ENV_NEO4J_PASSWORD = "OARAG_NEO4J_PASSWORD"
ENV_NEO4J_DATABASE = "OARAG_NEO4J_DATABASE"


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str

    def redacted(self) -> dict[str, str]:
        return {
            "uri": self.uri,
            "user": self.user,
            "password": "***",
            "database": self.database,
        }


def default_paths() -> Paths:
    repo_root = Path(__file__).resolve().parents[2]
    return Paths(repo_root=repo_root, artifacts_dir=repo_root / "artifacts")


def env_default(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        value = dotenv_default(name)
    return normalize_env_value(value)


def dotenv_default(name: str, env_path: Path | None = None) -> str | None:
    path = env_path or default_paths().repo_root / ".env"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == name:
            return value.strip().strip("\"'")
    return None


def normalize_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def load_neo4j_config() -> Neo4jConfig:
    return Neo4jConfig(
        uri=env_default(ENV_NEO4J_URI) or DEFAULT_NEO4J_URI,
        user=env_default(ENV_NEO4J_USER) or DEFAULT_NEO4J_USER,
        password=env_default(ENV_NEO4J_PASSWORD) or DEFAULT_NEO4J_PASSWORD,
        database=env_default(ENV_NEO4J_DATABASE) or DEFAULT_NEO4J_DATABASE,
    )
