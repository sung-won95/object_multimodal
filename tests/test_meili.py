from __future__ import annotations

import json
import os
import urllib.error
import uuid
from pathlib import Path

import pytest

from oarag.core.config import DEFAULT_MEILI_API_KEY, DEFAULT_MEILI_URL
from oarag.meili import (
    HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE,
    LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
    LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE,
    LECTURE_SEGMENT_PRE_SEMANTIC_SETTINGS_PROFILE,
    MeiliClient,
    EVIDENCE_UNIT_DEFAULT_SETTINGS_PROFILE,
    LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE,
    VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
    evidence_unit_settings,
    hybrid_embedder_settings,
    hybrid_embedder_settings_hash,
    hybrid_embedder_settings_profile_names,
    hybrid_embedder_settings_snapshot,
    normalize_hybrid_embedder_settings,
    normalize_query_vector,
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
from oarag.retrieval.project_index import index_project_segments, index_project_windows
from oarag.retrieval.project_query import query_project
from oarag.schemas import (
    LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
    LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
)


EXPECTED_DEFAULT_SETTINGS_HASH = "f0a9515744b51f74ba52310b07940174f99f040f7f7a6935e3678a0445e95917"
EXPECTED_VISUAL_ENTITY_SETTINGS_HASH = "2f8f2529ea5760191cc3f14551d1befd98873f4fa765420f082e72e9f5fc8bd6"
EXPECTED_WINDOW_SETTINGS_HASH = "c42ed8c97d77620012254b45e2d6b3961f12e2211b0f8ac78759f844f07503e8"
EXPECTED_HYBRID_EMBEDDER_SETTINGS_HASH = "d7e2a5244531c3c149b0415962e6b114cc8b7a39df3eb24b5acbbaf14f12cab3"
PUBLIC_USER_PROVIDED_VECTOR_PROJECT = (
    Path(__file__).parent / "fixtures" / "public_userprovided_vector_project"
)


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


def test_meili_search_accepts_user_provided_vector_payload() -> None:
    client = RecordingMeiliClient()

    response = client.search(
        "segments",
        "opaque probe",
        limit=1,
        hybrid={"embedder": "default", "semanticRatio": 1.0},
        vector=[0.1, 0.2, 0.3],
    )

    assert response["hits"] == []
    assert client.requests == [
        (
            "POST",
            "/indexes/segments/search",
            {
                "q": "opaque probe",
                "limit": 1,
                "showRankingScore": True,
                "hybrid": {"embedder": "default", "semanticRatio": 1.0},
                "vector": [0.1, 0.2, 0.3],
            },
        )
    ]


def test_normalize_query_vector_validates_dimensions_and_values() -> None:
    assert normalize_query_vector([1, 0.5, 0], dimensions=3) == [1.0, 0.5, 0.0]

    with pytest.raises(ValueError, match="dimension mismatch: expected 3, got 2"):
        normalize_query_vector([1.0, 0.0], dimensions=3)

    with pytest.raises(ValueError, match="item 1 must be a finite number"):
        normalize_query_vector([1.0, "private-not-a-number"])


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
        "local_entity_id",
        "frame_id",
        "entity_id",
        "video_id",
        "segment_id",
        "entity_type",
        "source",
        "source_model",
        "semantic_source_fields",
        "timestamp",
    ]
    assert "local_entity_id" in settings["displayedAttributes"]
    assert "video_id" in settings["displayedAttributes"]
    assert "segment_id" in settings["displayedAttributes"]
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


def test_evidence_unit_settings_payload_indexes_enriched_search_fields() -> None:
    settings = evidence_unit_settings(EVIDENCE_UNIT_DEFAULT_SETTINGS_PROFILE)

    assert settings["searchableAttributes"][:6] == [
        "semantic_text",
        "evidence_text",
        "concept_search_text",
        "concept_labels",
        "concept_aliases",
        "concept_relation_text",
    ]
    for field in [
        "transcript_keywords",
        "visual_state_text",
        "visual_entity_text",
        "visual_entities.detected_text",
        "visual_states.detected_text",
        "candidate_link_signal_summary",
        "verified_link_signal_summary",
    ]:
        assert field in settings["searchableAttributes"]
    assert "source_video_path" not in settings["displayedAttributes"]
    assert settings["rankingRules"] == [
        "words",
        "typo",
        "proximity",
        "attribute",
        "sort",
        "exactness",
    ]


def test_unknown_lecture_window_settings_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown lecture window settings profile"):
        lecture_window_settings("lecture_windows_experimental")


def test_public_userprovided_vector_fixture_live_smoke() -> None:
    if os.environ.get("OARAG_RUN_LIVE_MEILI_VECTOR_SMOKE") != "1":
        pytest.skip("set OARAG_RUN_LIVE_MEILI_VECTOR_SMOKE=1 to run live Meilisearch vector smoke")

    client = _live_meili_client()
    _skip_unless_vector_store_enabled(client)
    segment_index = f"oarag_userprovided_vector_segments_{uuid.uuid4().hex[:12]}"
    window_index = f"oarag_userprovided_vector_windows_{uuid.uuid4().hex[:12]}"

    try:
        segment_summary = index_project_segments(
            client,
            index_uid=segment_index,
            project_dir=PUBLIC_USER_PROVIDED_VECTOR_PROJECT,
            batch_size=2,
            reset=True,
            hybrid_embedder_profile=HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE,
            hybrid_embedder_dimensions=3,
            hybrid_embedder_live_smoke=True,
            allow_local_hash_vectors=True,
        )
        window_summary = index_project_windows(
            client,
            index_uid=window_index,
            project_dir=PUBLIC_USER_PROVIDED_VECTOR_PROJECT,
            batch_size=2,
            reset=True,
            hybrid_embedder_profile=HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE,
            hybrid_embedder_dimensions=3,
            hybrid_embedder_live_smoke=True,
            allow_local_hash_vectors=True,
        )

        assert segment_summary["hybrid_embedder_live_smoke"]["ok"] is True
        assert window_summary["hybrid_embedder_live_smoke"]["ok"] is True

        for index_uid, index_kind in (
            (segment_index, "segment"),
            (window_index, "window"),
        ):
            response = query_project(
                client=client,
                index_uid=index_uid,
                retrieval_index_kind=index_kind,
                project_dir=PUBLIC_USER_PROVIDED_VECTOR_PROJECT,
                query="zzzz_opaque_probe",
                limit=1,
                neighbor_count=0,
                hybrid_retrieval=True,
                hybrid_query_vector_name="gradient_direction",
            )

            assert response["counts"]["bundles"] == 1
            assert response["bundles"][0]["candidate"]["segment_id"] == "vector_semantic_seg_001"
            assert response["retrieval_context"]["hybrid_retrieval"]["query_vector"] == {
                "used": True,
                "embedder": "default",
                "dimensions": 3,
                "source": "manifest",
                "name": "gradient_direction",
            }
    finally:
        for index_uid in (segment_index, window_index):
            try:
                client.wait_task(client.delete_index(index_uid))
            except Exception:
                pass


def _live_meili_client() -> MeiliClient:
    url = os.environ.get("OARAG_MEILI_URL", DEFAULT_MEILI_URL)
    api_key = os.environ.get("OARAG_MEILI_API_KEY", DEFAULT_MEILI_API_KEY)
    client = MeiliClient(base_url=url, api_key=api_key, timeout=2.0)
    try:
        client.health()
    except urllib.error.HTTPError as exc:
        pytest.skip(
            "Meilisearch is reachable but rejected the configured credentials; "
            f"set OARAG_MEILI_API_KEY or start docker compose Meilisearch. reason: {exc}"
        )
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        pytest.skip(
            "Meilisearch is unavailable; start it with `docker compose up -d meilisearch` "
            f"or set OARAG_MEILI_URL. reason: {exc}"
        )
    return client


def _skip_unless_vector_store_enabled(client: MeiliClient) -> None:
    try:
        features = client._request("GET", "/experimental-features")
    except urllib.error.HTTPError as exc:
        pytest.skip(
            "Meilisearch experimental-features endpoint is unavailable; "
            f"cannot run userProvided vector smoke. reason: {exc}"
        )
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        pytest.skip(
            "Meilisearch experimental-features endpoint is unavailable; "
            f"cannot run userProvided vector smoke. reason: {exc}"
        )
    if features.get("vectorStore") is not True:
        pytest.skip(
            "Meilisearch vector store experimental feature is disabled; "
            "enable vectorStore to run userProvided vector smoke"
        )
