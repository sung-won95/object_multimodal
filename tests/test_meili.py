from __future__ import annotations

import pytest

from oarag.meili import (
    LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
    LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE,
    lecture_segment_settings,
    lecture_segment_settings_hash,
    lecture_segment_settings_profile_names,
    lecture_segment_settings_snapshot,
)


EXPECTED_DEFAULT_SETTINGS_HASH = "f5164705dfc709f02399414cd0cb6856013185aeabc61e74fdabf3ec4af27352"


def test_lecture_segment_settings_payload_is_domain_agnostic_and_multilingual_safe() -> None:
    settings = lecture_segment_settings()

    assert settings["searchableAttributes"] == [
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


def test_unknown_lecture_segment_settings_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown lecture segment settings profile"):
        lecture_segment_settings("pilot_video_synonyms")
