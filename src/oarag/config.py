from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MEILI_URL = "http://127.0.0.1:7700"
DEFAULT_MEILI_API_KEY = "dev-master-key"
ENV_STT_LANGUAGE = "OARAG_STT_LANGUAGE"


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    artifacts_dir: Path


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
