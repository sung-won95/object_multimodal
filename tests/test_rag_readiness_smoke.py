from __future__ import annotations

import argparse
import json
import os
import urllib.error
import uuid
from pathlib import Path
from typing import Any

import pytest

from oarag import cli
from oarag.core.config import DEFAULT_MEILI_API_KEY, DEFAULT_MEILI_URL
from oarag.integrations.meili import MeiliClient
from oarag.retrieval.project_index import index_project_segments
from oarag.retrieval.project_query import query_project


SMOKE_QUERY = "gradient loss curve"
TARGET_SEGMENT_ID = "readiness_seg_001"


class FakeClient:
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self.hits = hits
        self.searches: list[tuple[str, str, int]] = []

    def search(self, index_uid: str, query: str, limit: int = 10) -> dict[str, Any]:
        self.searches.append((index_uid, query, limit))
        return {
            "hits": self.hits[:limit],
            "indexUid": index_uid,
            "query": query,
            "processingTimeMs": 1,
        }


def test_rag_readiness_query_project_contract_without_meilisearch(tmp_path: Path) -> None:
    project_dir = _write_public_safe_project(tmp_path)
    response = query_project(
        client=FakeClient([_target_hit()]),
        index_uid="readiness_segments",
        project_dir=project_dir,
        query=SMOKE_QUERY,
        limit=1,
        neighbor_count=0,
    )

    _assert_query_contract(response)


def test_rag_readiness_ask_project_contract_when_command_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli.build_parser()
    if "ask-project" not in _command_names(parser):
        pytest.skip("ask-project is not available on current develop; #88 owns that command/schema.")

    project_dir = _write_public_safe_project(tmp_path)
    args = parser.parse_args(
        [
            "ask-project",
            "--index",
            "readiness_segments",
            "--project-dir",
            str(project_dir),
            "--query",
            SMOKE_QUERY,
            "--limit",
            "1",
            "--neighbor-count",
            "0",
        ]
    )
    monkeypatch.setattr(cli, "client_from_args", lambda _: FakeClient([_target_hit()]))

    args.func(args)

    payload = json.loads(capsys.readouterr().out)
    _assert_query_contract(payload)
    assert payload.get("answer_text") or payload.get("answer")


def test_rag_readiness_live_meilisearch_smoke_or_clear_skip(tmp_path: Path) -> None:
    client = _live_meili_client()
    project_dir = _write_public_safe_project(tmp_path)
    index_uid = f"oarag_readiness_{uuid.uuid4().hex[:12]}"

    try:
        summary = index_project_segments(
            client,
            index_uid=index_uid,
            project_dir=project_dir,
            batch_size=2,
            reset=True,
        )
        assert summary["indexed_documents"] == 2

        response = query_project(
            client=client,
            index_uid=index_uid,
            project_dir=project_dir,
            query=SMOKE_QUERY,
            limit=1,
            neighbor_count=0,
        )
        _assert_query_contract(response)
    finally:
        try:
            client.wait_task(client.delete_index(index_uid))
        except Exception:
            pass


def _assert_query_contract(response: dict[str, Any]) -> None:
    assert response["counts"]["bundles"] == 1
    bundle = response["bundles"][0]
    assert bundle["candidate"]["segment_id"] == TARGET_SEGMENT_ID
    assert bundle["evidence_window"]["target_segment"]["segment_id"] == TARGET_SEGMENT_ID
    assert bundle["evidence_window"]["frame_refs"] == [
        {
            "frame_id": "readiness_frame_001",
            "timestamp": 4.0,
            "frame_path": "fixtures/readiness/readiness_frame_001.png",
        }
    ]
    assert bundle["linked_entities"][0]["entity"]["text"] == "loss curve annotation"


def _write_public_safe_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "public_readiness_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": TARGET_SEGMENT_ID,
                "project_id": "public_readiness_project",
                "video_id": "synthetic_gradient",
                "sample_id": TARGET_SEGMENT_ID,
                "sample_index": 1,
                "start_time": 0.0,
                "end_time": 8.0,
                "timestamp_center": 4.0,
                "transcript_text": "A synthetic gradient update lowers the loss curve.",
                "frame_refs": ["readiness_frame_001"],
            },
            {
                "segment_id": "readiness_seg_002",
                "project_id": "public_readiness_project",
                "video_id": "synthetic_gradient",
                "sample_id": "readiness_seg_002",
                "sample_index": 2,
                "start_time": 10.0,
                "end_time": 18.0,
                "timestamp_center": 14.0,
                "transcript_text": "A separate public-safe segment talks about study notes.",
                "frame_refs": ["readiness_frame_002"],
            },
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {
                "frame_id": "readiness_frame_001",
                "timestamp": 4.0,
                "frame_path": "fixtures/readiness/readiness_frame_001.png",
            },
            {
                "frame_id": "readiness_frame_002",
                "timestamp": 14.0,
                "frame_path": "fixtures/readiness/readiness_frame_002.png",
            },
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "readiness_entity_loss_curve",
                "project_id": "public_readiness_project",
                "frame_id": "readiness_frame_001",
                "timestamp": 4.0,
                "frame_path": "fixtures/readiness/readiness_frame_001.png",
                "bbox": None,
                "text": "loss curve annotation",
                "entity_type": "synthetic_visual_note",
                "confidence": 1.0,
                "source": "public_synthetic_fixture",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "readiness_link_loss_curve",
                "project_id": "public_readiness_project",
                "segment_id": TARGET_SEGMENT_ID,
                "entity_id": "readiness_entity_loss_curve",
                "frame_id": "readiness_frame_001",
                "link_type": "synthetic_alignment",
                "score": 1.0,
                "evidence": ["public_synthetic_fixture"],
                "time_overlap": True,
                "lexical_match": ["loss", "curve"],
                "mention_candidate": ["curve"],
            }
        ],
    )
    return project_dir


def _target_hit() -> dict[str, Any]:
    return {
        "segment_id": TARGET_SEGMENT_ID,
        "sample_id": TARGET_SEGMENT_ID,
        "video_id": "synthetic_gradient",
        "start_time": 0.0,
        "end_time": 8.0,
        "timestamp_center": 4.0,
        "transcript_text": "A synthetic gradient update lowers the loss curve.",
        "_rankingScore": 0.91,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _command_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()


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
