from __future__ import annotations

import os
import urllib.error
import uuid
from pathlib import Path

import pytest

from oarag.core.config import DEFAULT_MEILI_API_KEY, DEFAULT_MEILI_URL
from oarag.core.schemas import (
    LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
    LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
    ensure_lecture_segment_semantic_contract,
)
from oarag.integrations.meili import MeiliClient
from oarag.retrieval.project_index import index_project_segments, iter_jsonl_documents
from oarag.retrieval.project_query import query_project


FIXTURE_PROJECT_DIR = Path(__file__).parent / "fixtures" / "public_lecture_semantic_project"
SMOKE_QUERY = "slope information lower loss"
TARGET_SEGMENT_ID = "synthetic_semantic_lecture_seg_001"


def test_public_fixture_builds_semantic_contract_from_visual_fields() -> None:
    rows = list(iter_jsonl_documents(FIXTURE_PROJECT_DIR / "segments" / "lecture_segments.jsonl"))
    target = ensure_lecture_segment_semantic_contract(rows[0])

    assert target["segment_id"] == TARGET_SEGMENT_ID
    assert "slope information points toward lower loss" in target[
        LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD
    ]
    assert "visual_entities.visual_description" in target[
        LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD
    ]


def test_public_fixture_meilisearch_semantic_retrieval_smoke() -> None:
    client = _live_meili_client()
    index_uid = f"oarag_semantic_smoke_{uuid.uuid4().hex[:12]}"

    try:
        summary = index_project_segments(
            client,
            index_uid=index_uid,
            project_dir=FIXTURE_PROJECT_DIR,
            batch_size=2,
            reset=True,
        )

        settings = client.get_settings(index_uid)
        assert settings["searchableAttributes"][0] == LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD
        assert LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD in settings["displayedAttributes"]
        assert summary["indexed_documents"] == 2

        response = query_project(
            client=client,
            index_uid=index_uid,
            project_dir=FIXTURE_PROJECT_DIR,
            query=SMOKE_QUERY,
            limit=2,
            neighbor_count=0,
        )

        assert response["counts"]["bundles"] >= 1
        top_bundle = response["bundles"][0]
        assert top_bundle["candidate"]["segment_id"] == TARGET_SEGMENT_ID
        assert "visual_entities.visual_description" in top_bundle["candidate"][
            "semantic_source_fields"
        ]

        search_hit = client.search(index_uid, SMOKE_QUERY, limit=1)["hits"][0]
        assert search_hit["segment_id"] == TARGET_SEGMENT_ID
        assert "slope information points toward lower loss" in search_hit[
            LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD
        ]
        assert "visual_entities.visual_description" in search_hit[
            LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD
        ]
    finally:
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
