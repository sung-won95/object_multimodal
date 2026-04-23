from pathlib import Path

from oarag.config import dotenv_default, normalize_env_value


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
