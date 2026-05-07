from __future__ import annotations

import json
from pathlib import Path

import pytest

from oarag.meili import (
    LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
    LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE,
    lecture_segment_settings_hash,
)
from oarag.project_index import index_project_segments, segment_artifact_path
from oarag.schemas import (
    LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
    LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
    ensure_lecture_segment_semantic_contract,
)


class FakeMeiliClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def wait_task(self, task):
        self.calls.append(("wait_task", task))
        return {"status": "succeeded"}

    def delete_index(self, uid: str):
        self.calls.append(("delete_index", uid))
        return {"taskUid": 1}

    def create_index(self, uid: str, primary_key: str):
        self.calls.append(("create_index", uid, primary_key))
        return {"taskUid": 2}

    def update_settings(self, index_uid: str, settings: dict):
        self.calls.append(("update_settings", index_uid, settings))
        return {"taskUid": 3}

    def add_documents(self, index_uid: str, documents: list[dict]):
        self.calls.append(("add_documents", index_uid, documents))
        return {"taskUid": 4}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_segment_artifact_prefers_aligned_file(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    segments_dir = project_dir / "segments"
    aligned = segments_dir / "lecture_segments_aligned.jsonl"
    fallback = segments_dir / "lecture_segments.jsonl"
    write_jsonl(fallback, [{"segment_id": "base"}])
    write_jsonl(aligned, [{"segment_id": "aligned"}])

    selected = segment_artifact_path(project_dir)

    assert selected == aligned


def test_segment_artifact_falls_back_to_default_file(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    fallback = project_dir / "segments" / "lecture_segments.jsonl"
    write_jsonl(fallback, [{"segment_id": "base"}])

    selected = segment_artifact_path(project_dir)

    assert selected == fallback


def test_segment_artifact_raises_when_no_default_or_aligned(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        segment_artifact_path(project_dir)


def test_index_project_segments_batches_documents(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    segments_path = project_dir / "segments" / "lecture_segments.jsonl"
    rows = [
        {"segment_id": "s1", "transcript_text": "alpha"},
        {"segment_id": "s2", "transcript_text": "beta"},
        {"segment_id": "s3", "transcript_text": "gamma"},
    ]
    write_jsonl(segments_path, rows)
    client = FakeMeiliClient()

    summary = index_project_segments(
        client,
        index_uid="local_segments",
        project_dir=project_dir,
        batch_size=2,
        reset=True,
    )

    add_calls = [call for call in client.calls if call[0] == "add_documents"]
    settings_call = next(call for call in client.calls if call[0] == "update_settings")
    expected_rows = [ensure_lecture_segment_semantic_contract(row) for row in rows]
    assert len(add_calls) == 2
    assert add_calls[0][1] == "local_segments"
    assert add_calls[0][2] == expected_rows[:2]
    assert add_calls[1][2] == expected_rows[2:]
    assert settings_call[1] == "local_segments"
    assert ("delete_index", "local_segments") in client.calls
    assert summary["index"] == "local_segments"
    assert summary["indexed_documents"] == 3
    assert summary["indexed_batches"] == 2
    assert summary["segments_path"] == str(segments_path)
    assert summary["settings_profile"] == LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE
    assert summary["settings_hash"] == lecture_segment_settings_hash(settings_call[2])
    assert summary["settings_snapshot"] == {
        "profile": LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
        "hash": summary["settings_hash"],
        "settings": settings_call[2],
    }


def test_index_project_segments_adds_semantic_contract_to_segment_only_documents(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    segments_path = project_dir / "segments" / "lecture_segments.jsonl"
    rows = [
        {"segment_id": "s1", "transcript_text": "alpha transcript"},
        {
            "segment_id": "s2",
            "transcript_text": "beta transcript",
            "semantic_text": "custom beta retrieval text",
            "semantic_source_fields": ["transcript_text", "visual_entities.text"],
        },
    ]
    write_jsonl(segments_path, rows)
    client = FakeMeiliClient()

    index_project_segments(
        client,
        index_uid="local_segments",
        project_dir=project_dir,
        batch_size=10,
    )

    add_call = next(call for call in client.calls if call[0] == "add_documents")
    documents = add_call[2]
    assert documents[0][LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD] == "alpha transcript"
    assert documents[0][LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD] == [
        "transcript_text"
    ]
    assert documents[1][LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD] == "custom beta retrieval text"
    assert documents[1][LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD] == [
        "transcript_text",
        "visual_entities.text",
    ]


def test_index_project_segments_can_use_legacy_settings_profile(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    segments_path = project_dir / "segments" / "lecture_segments.jsonl"
    write_jsonl(segments_path, [{"segment_id": "s1", "transcript_text": "alpha"}])
    client = FakeMeiliClient()

    summary = index_project_segments(
        client,
        index_uid="local_segments",
        project_dir=project_dir,
        settings_profile=LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE,
    )

    settings_call = next(call for call in client.calls if call[0] == "update_settings")
    assert settings_call[2]["displayedAttributes"] == ["*"]
    assert summary["settings_profile"] == LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE
    assert summary["settings_hash"] == lecture_segment_settings_hash(
        settings_call[2],
        profile=LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE,
    )
