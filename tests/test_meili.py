from __future__ import annotations

import pytest

from oarag.meili import (
    LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
    LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE,
    LECTURE_SEGMENT_PRE_SEMANTIC_SETTINGS_PROFILE,
    VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
    lecture_segment_settings,
    lecture_segment_settings_hash,
    lecture_segment_settings_profile_names,
    lecture_segment_settings_snapshot,
    visual_entity_settings,
    visual_entity_settings_hash,
    visual_entity_settings_profile_names,
    visual_entity_settings_snapshot,
)
from oarag.schemas import (
    LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
    LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
)


EXPECTED_DEFAULT_SETTINGS_HASH = "cf3664e1c618e4b81fdfc7099cf9f2f6da7e552e48479e9127a55546af7a1abd"
EXPECTED_VISUAL_ENTITY_SETTINGS_HASH = "bbda5dcb692e2d42d935f085654934584d4ec66ae7ce40238cb44541806770e9"


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
    assert "frame_refs" in settings["displayedAttributes"]
    assert "source_video_path" not in settings["displayedAttributes"]


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
        "timestamp",
    ]
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
