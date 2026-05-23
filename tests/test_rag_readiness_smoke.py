from __future__ import annotations

import os
import urllib.error
import uuid
from pathlib import Path

import pytest

from oarag.core.config import DEFAULT_MEILI_API_KEY, DEFAULT_MEILI_URL
from oarag.integrations.meili import MeiliClient
from oarag.retrieval.answer import ask_project
from oarag.retrieval.project_index import index_project_segments
from oarag.retrieval.project_query import query_project


FIXTURE_PROJECT_DIR = Path(__file__).parent / "fixtures" / "public_lecture_semantic_project"
SMOKE_QUERY = "slope information lower loss"
TARGET_SEGMENT_ID = "synthetic_semantic_lecture_seg_001"


def test_public_fixture_rag_readiness_query_and_ask_paths() -> None:
    client = _live_meili_client()
    index_uid = f"oarag_readiness_{uuid.uuid4().hex[:12]}"

    try:
        index_summary = index_project_segments(
            client,
            index_uid=index_uid,
            project_dir=FIXTURE_PROJECT_DIR,
            batch_size=2,
            reset=True,
        )
        assert index_summary["indexed_documents"] == 2

        query_response = query_project(
            client=client,
            index_uid=index_uid,
            project_dir=FIXTURE_PROJECT_DIR,
            query=SMOKE_QUERY,
            limit=2,
            neighbor_count=0,
        )
        _assert_readiness_bundle_contract(query_response)

        answer_response = ask_project(
            client=client,
            index_uid=index_uid,
            project_dir=FIXTURE_PROJECT_DIR,
            query=SMOKE_QUERY,
            limit=2,
            neighbor_count=0,
        )
        _assert_readiness_bundle_contract(answer_response)
        assert answer_response["answer_text"]
        assert "negative gradient" in answer_response["answer_text"]
        assert "loss curve annotation" in answer_response["answer_text"]
    finally:
        try:
            client.wait_task(client.delete_index(index_uid))
        except Exception:
            pass


def _assert_readiness_bundle_contract(response: dict) -> None:
    assert response["counts"]["bundles"] >= 1
    top_bundle = response["bundles"][0]
    assert top_bundle["candidate"]["segment_id"] == TARGET_SEGMENT_ID
    assert top_bundle["evidence_window"]["target_segment"]["segment_id"] == TARGET_SEGMENT_ID
    assert top_bundle["evidence_window"]["frame_refs"]
    assert top_bundle["linked_entities"]
    assert top_bundle["linked_entities"][0]["entity"]["text"] == "loss curve annotation"


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
