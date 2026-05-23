from __future__ import annotations

import json

import pytest

from oarag.meili import (
    HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE,
    LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
    LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE,
    LECTURE_SEGMENT_PRE_SEMANTIC_SETTINGS_PROFILE,
    MeiliClient,
    LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE,
    VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
    hybrid_embedder_settings,
    hybrid_embedder_settings_hash,
    hybrid_embedder_settings_profile_names,
    hybrid_embedder_settings_snapshot,
    normalize_hybrid_embedder_settings,
    lecture_segment_settings,
    lecture_segment_settings_hash,
    lecture_segment_settings_profile_names,
    lecture_segment_settings_snapshot,
    lecture_window_settings,
    lecture_window_settings_hash,
    lecture_window_settings_profile_names,
    lecture_window_settings_snapshot,
    visual_entity_settings,
    visual_entity_settings_hash,
    visual_entity_settings_profile_names,
    visual_entity_settings_snapshot,
)
from oarag.schemas import (
    LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
    LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
)


EXPECTED_DEFAULT_SETTINGS_HASH = "f0a9515744b51f74ba52310b07940174f99f040f7f7a6935e3678a0445e95917"
EXPECTED_VISUAL_ENTITY_SETTINGS_HASH = "5cb877a0006ff93448b2c06680fc566f41bd3d54cd527b6b2282c0e1b158ccdc"
EXPECTED_WINDOW_SETTINGS_HASH = "c42ed8c97d77620012254b45e2d6b3961f12e2211b0f8ac78759f844f07503e8"
EXPECTED_HYBRID_EMBEDDER_SETTINGS_HASH = "d7e2a5244531c3c149b0415962e6b114cc8b7a39df3eb24b5acbbaf14f12cab3"


class RecordingMeiliClient(MeiliClient):
    def __init__(self) -> None:
        super().__init__("http://localhost:7700")
        self.requests: list[tuple[str, str, dict]] = []

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        self.requests.append((method, path, payload or {}))
        return {"hits": [], "processingTimeMs": 1}


def test_lecture_segment_settings_payload_is_domain_agnostic_and_multilingual_safe() -> None:
    settings = lecture_segment_settings()

    assert settings["searchableAttributes"] == [
        LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
        "transcript_text",
        "normalized_text",
        "mention_candidates",
        "video_name",
        "video_id",
        "slide_id",
        "sample_id",
    ]
    assert settings["rankingRules"] == [
        "words",
        "typo",
        "proximity",
        "attribute",
        "sort",
        "exactness",
    ]
    assert settings["stopWords"] == []
    assert settings["synonyms"] == {}
    assert settings["typoTolerance"] == {
        "enabled": True,
        "minWordSizeForTypos": {
            "oneTypo": 5,
            "twoTypos": 9,
        },
        "disableOnAttributes": [
            "video_id",
            "slide_id",
            "sample_id",
        ],
        "disableOnWords": [],
    }
    assert settings["displayedAttributes"] != ["*"]
    assert settings["displayedAttributes"].index(LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD) < (
        settings["displayedAttributes"].index("slide_id")
    )
    assert LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD in settings["displayedAttributes"]
    assert LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD in settings["filterableAttributes"]
    assert "frame_refs" in settings["displayedAttributes"]
    assert "visual_entities" in settings["displayedAttributes"]
    assert "source_video_path" not in settings["displayedAttributes"]


def test_meili_search_accepts_hybrid_payload() -> None:
    client = RecordingMeiliClient()

    response = client.search(
        "segments",
        "bet size",
        limit=3,
        hybrid={"embedder": "default", "semanticRatio": 1.0},
        show_ranking_score_details=True,
    )

    assert response["hits"] == []
    assert client.requests == [
        (
            "POST",
            "/indexes/segments/search",
            {
                "q": "bet size",
                "limit": 3,
                "showRankingScore": True,
                "hybrid": {"embedder": "default", "semanticRatio": 1.0},
                "showRankingScoreDetails": True,
            },
        )
    ]


def test_hybrid_embedder_profile_builds_meili_settings_payload() -> None:
    settings = hybrid_embedder_settings(
        HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE,
        embedder_name="lecture_embedder",
        dimensions=768,
    )
    snapshot = hybrid_embedder_settings_snapshot(
        settings,
        profile=HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE,
    )

    assert settings == {
        "embedders": {
            "lecture_embedder": {
                "source": "userProvided",
                "dimensions": 768,
            }
        }
    }
    assert hybrid_embedder_settings_hash(settings) == EXPECTED_HYBRID_EMBEDDER_SETTINGS_HASH
    assert snapshot == {
        "profile": HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE,
        "hash": EXPECTED_HYBRID_EMBEDDER_SETTINGS_HASH,
        "settings": settings,
    }
    assert HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE in hybrid_embedder_settings_profile_names()


def test_hybrid_embedder_custom_settings_snapshot_redacts_credentials() -> None:
    settings = normalize_hybrid_embedder_settings(
        {
            "embedders": {
                "default": {
                    "source": "openAi",
                    "model": "text-embedding-3-small",
                    "apiKey": "sk-private-test-key",
                    "documentTemplate": "{{doc.semantic_text}}",
                }
            }
        }
    )

    snapshot = hybrid_embedder_settings_snapshot(settings)

    assert settings["embedders"]["default"]["apiKey"] == "sk-private-test-key"
    assert snapshot["settings"]["embedders"]["default"]["apiKey"] == "<redacted>"
    assert "sk-private-test-key" not in json.dumps(snapshot)


def test_hybrid_embedder_user_provided_rejects_document_template() -> None:
    with pytest.raises(ValueError, match="source 'userProvided'"):
        normalize_hybrid_embedder_settings(
            {
                "embedders": {
                    "default": {
                        "source": "userProvided",
                        "dimensions": 384,
                        "documentTemplate": "{{doc.semantic_text}}",
                    }
                }
            }
        )


def test_lecture_segment_settings_hash_is_a_regression_guard() -> None:
    snapshot = lecture_segment_settings_snapshot()

    assert lecture_segment_settings_hash() == EXPECTED_DEFAULT_SETTINGS_HASH
    assert snapshot["profile"] == LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE
    assert snapshot["hash"] == EXPECTED_DEFAULT_SETTINGS_HASH
    assert snapshot["settings"] == lecture_segment_settings()


def test_lecture_segment_settings_returns_defensive_copy() -> None:
    settings = lecture_segment_settings()
    settings["searchableAttributes"].append("domain_specific_alias")

    assert "domain_specific_alias" not in lecture_segment_settings()["searchableAttributes"]


def test_legacy_profile_is_available_for_rollback() -> None:
    settings = lecture_segment_settings(LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE)

    assert settings["displayedAttributes"] == ["*"]
    assert LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE in lecture_segment_settings_profile_names()


def test_pre_semantic_profile_remains_available_for_settings_rollback() -> None:
    settings = lecture_segment_settings(LECTURE_SEGMENT_PRE_SEMANTIC_SETTINGS_PROFILE)

    assert settings["searchableAttributes"] == [
        "transcript_text",
        "normalized_text",
        "mention_candidates",
        "video_name",
        "video_id",
        "slide_id",
        "sample_id",
    ]
    assert LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD not in settings["searchableAttributes"]
    assert LECTURE_SEGMENT_PRE_SEMANTIC_SETTINGS_PROFILE in lecture_segment_settings_profile_names()


def test_unknown_lecture_segment_settings_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown lecture segment settings profile"):
        lecture_segment_settings("pilot_video_synonyms")


def test_visual_entity_settings_payload_indexes_direct_visual_evidence() -> None:
    settings = visual_entity_settings()

    assert settings["searchableAttributes"] == [
        "text",
        "visual_description",
        "entity_type",
        "frame_id",
        "source_model",
    ]
    assert settings["filterableAttributes"] == [
        "project_id",
        "frame_id",
        "entity_id",
        "entity_type",
        "source",
        "source_model",
        "semantic_source_fields",
        "timestamp",
    ]
    assert "semantic_source_fields" in settings["displayedAttributes"]
    assert "frame_path" in settings["displayedAttributes"]
    assert "source_video_path" not in settings["displayedAttributes"]
    assert settings["rankingRules"] == [
        "words",
        "typo",
        "proximity",
        "attribute",
        "sort",
        "exactness",
    ]


def test_visual_entity_settings_hash_is_a_regression_guard() -> None:
    snapshot = visual_entity_settings_snapshot()

    assert visual_entity_settings_hash() == EXPECTED_VISUAL_ENTITY_SETTINGS_HASH
    assert snapshot["profile"] == VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE
    assert snapshot["hash"] == EXPECTED_VISUAL_ENTITY_SETTINGS_HASH
    assert snapshot["settings"] == visual_entity_settings()
    assert VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE in visual_entity_settings_profile_names()


def test_unknown_visual_entity_settings_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown visual entity settings profile"):
        visual_entity_settings("visual_entities_experimental")


def test_lecture_window_settings_payload_indexes_window_context() -> None:
    settings = lecture_window_settings()

    assert settings["searchableAttributes"] == [
        LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
        "transcript_window_text",
        "transcript_text",
        "visual_entities.text",
        "visual_entities.visual_description",
        "video_name",
        "video_id",
        "sample_id",
    ]
    assert "window_id" in settings["filterableAttributes"]
    assert "target_segment_id" in settings["filterableAttributes"]
    assert "source_segment_ids" in settings["filterableAttributes"]
    assert "evidence_window" in settings["displayedAttributes"]
    assert "source_video_path" not in settings["displayedAttributes"]
    assert settings["rankingRules"] == [
        "words",
        "typo",
        "proximity",
        "attribute",
        "sort",
        "exactness",
    ]


def test_lecture_window_settings_hash_is_a_regression_guard() -> None:
    snapshot = lecture_window_settings_snapshot()

    assert lecture_window_settings_hash() == EXPECTED_WINDOW_SETTINGS_HASH
    assert snapshot["profile"] == LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE
    assert snapshot["hash"] == EXPECTED_WINDOW_SETTINGS_HASH
    assert snapshot["settings"] == lecture_window_settings()
    assert LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE in lecture_window_settings_profile_names()


def test_unknown_lecture_window_settings_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown lecture window settings profile"):
        lecture_window_settings("lecture_windows_experimental")
