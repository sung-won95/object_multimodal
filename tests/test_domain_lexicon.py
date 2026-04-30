from __future__ import annotations

import json
from pathlib import Path

import pytest

from oarag.domain_lexicon import load_domain_lexicon


def test_missing_domain_lexicon_is_disabled_by_default(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    lexicon = load_domain_lexicon(project_dir=project_dir)

    assert lexicon.enabled is False
    assert lexicon.metadata() == {
        "enabled": False,
        "source_path": None,
        "canonical_term_count": 0,
        "alias_count": 0,
        "term_count": 0,
    }
    assert lexicon.expand_query("wager sizing") == "wager sizing"


def test_domain_lexicon_loads_aliases_and_expands_queries(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    lexicon_path = project_dir / "domain_lexicon.json"
    lexicon_path.write_text(
        json.dumps({"aliases": {"bet": ["wager", "벳"], "size": ["sizing", "사이즈"]}}),
        encoding="utf-8",
    )

    lexicon = load_domain_lexicon(project_dir=project_dir)

    assert lexicon.enabled is True
    assert lexicon.canonicalize("벳은") == "bet"
    assert lexicon.canonicalize("sizing") == "size"
    expanded = lexicon.expand_query("wager sizing")
    assert expanded == "wager sizing size 사이즈 bet 벳"
    metadata = lexicon.metadata()
    assert metadata["enabled"] is True
    assert metadata["source_path"] == str(lexicon_path.resolve())
    assert metadata["canonical_term_count"] == 2
    assert metadata["alias_count"] == 4
    assert "wager" not in json.dumps(metadata)


def test_domain_lexicon_rejects_invalid_alias_shape(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    lexicon_path = project_dir / "domain_lexicon.json"
    lexicon_path.write_text(json.dumps({"aliases": ["bet"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="aliases"):
        load_domain_lexicon(project_dir=project_dir)
