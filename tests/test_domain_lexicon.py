from __future__ import annotations

import json
from pathlib import Path

import pytest

from oarag.domain_lexicon import load_domain_lexicon

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
    assert expanded == "wager sizing bet 벳 size 사이즈"
    assert lexicon.query_expansion_metadata("wager sizing") == {
        "enabled": True,
        "applied": True,
        "added_term_count": 4,
        "expanded_query": "wager sizing bet 벳 size 사이즈",
        "source_path": str(lexicon_path.resolve()),
        "terms": [
            {
                "term": "bet",
                "source": "domain_lexicon",
                "canonical": "bet",
                "matched_terms": ["wager"],
            },
            {
                "term": "벳",
                "source": "domain_lexicon",
                "canonical": "bet",
                "matched_terms": ["wager"],
            },
            {
                "term": "size",
                "source": "domain_lexicon",
                "canonical": "size",
                "matched_terms": ["sizing"],
            },
            {
                "term": "사이즈",
                "source": "domain_lexicon",
                "canonical": "size",
                "matched_terms": ["sizing"],
            },
        ],
    }
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


def test_domain_lexicon_expands_korean_english_and_numeric_variants(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    lexicon_path = project_dir / "domain_lexicon.json"
    lexicon_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "aliases": {
                    "3bet": {"aliases": ["3-bet", "3 bet", "쓰리벳"]},
                    "bet sizing": ["벳사이즈", "벳 사이즈", "sizing"],
                },
            }
        ),
        encoding="utf-8",
    )

    lexicon = load_domain_lexicon(project_dir=project_dir)

    assert lexicon.canonicalize("쓰리벳은") == "3bet"
    assert lexicon.expand_query("쓰리벳은 벳사이즈를 봅니다") == (
        "쓰리벳은 벳사이즈를 봅니다 3bet 3-bet 3 bet bet sizing 벳 사이즈 sizing"
    )
    expansion = lexicon.query_expansion_metadata("3-bet sizing")
    assert expansion["applied"] is True
    assert expansion["added_term_count"] == 6
    assert expansion["terms"] == [
        {
            "term": "3bet",
            "source": "domain_lexicon",
            "canonical": "3bet",
            "matched_terms": ["3 bet", "3-bet", "3bet"],
        },
        {
            "term": "3 bet",
            "source": "domain_lexicon",
            "canonical": "3bet",
            "matched_terms": ["3 bet", "3-bet", "3bet"],
        },
        {
            "term": "쓰리벳",
            "source": "domain_lexicon",
            "canonical": "3bet",
            "matched_terms": ["3 bet", "3-bet", "3bet"],
        },
        {
            "term": "bet sizing",
            "source": "domain_lexicon",
            "canonical": "bet sizing",
            "matched_terms": ["bet sizing", "sizing"],
        },
        {
            "term": "벳사이즈",
            "source": "domain_lexicon",
            "canonical": "bet sizing",
            "matched_terms": ["bet sizing", "sizing"],
        },
        {
            "term": "벳 사이즈",
            "source": "domain_lexicon",
            "canonical": "bet sizing",
            "matched_terms": ["bet sizing", "sizing"],
        },
    ]


def test_domain_lexicon_matches_plural_aliases_without_overstemming(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "domain_lexicon.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "aliases": {
                    "gradient": ["slope"],
                    "loss": ["objective"],
                    "class": ["category"],
                },
            }
        ),
        encoding="utf-8",
    )

    lexicon = load_domain_lexicon(project_dir=project_dir)

    assert lexicon.matched_canonical_terms("gradients point downhill") == {
        "gradient": ("gradients",)
    }
    assert lexicon.matched_canonical_terms("los claz") == {}


def test_domain_lexicon_rejects_conflicting_aliases(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "domain_lexicon.json").write_text(
        json.dumps({"aliases": {"bet": ["wager"], "stake": ["wager"]}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="maps to both"):
        load_domain_lexicon(project_dir=project_dir)


def test_domain_lexicon_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "domain_lexicon.json").write_text(
        json.dumps({"schema_version": 2, "aliases": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_domain_lexicon(project_dir=project_dir)


def test_public_domain_lexicon_fixtures_load() -> None:
    lecture_lexicon = load_domain_lexicon(
        project_dir=FIXTURES_DIR / "public_lecture_semantic_project"
    )
    poker_lexicon = load_domain_lexicon(
        project_dir=FIXTURES_DIR / "public_retrieval_ablation_project"
    )

    assert lecture_lexicon.query_expansion_terms("로스 커브") == ["loss curve", "loss-curve"]
    assert poker_lexicon.query_expansion_terms("쓰리벳") == ["3bet", "3-bet", "3 bet"]
