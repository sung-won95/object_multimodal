from __future__ import annotations

import json
from pathlib import Path

import pytest

from oarag.meili import (
    HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE,
    LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
    LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE,
    LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE,
    MeiliTaskError,
    VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
    hybrid_embedder_settings_hash,
    lecture_segment_settings_hash,
    lecture_window_settings_hash,
    visual_entity_settings_hash,
)
from oarag.project_index import (
    build_project_windows,
    build_window_index_documents,
    index_project_segments,
    index_project_visual_entities,
    index_project_windows,
    segment_artifact_path,
    visual_entity_artifact_path,
    window_artifact_path,
)
from oarag.schemas import (
    LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
    LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
    ensure_lecture_segment_semantic_contract,
)


class FakeMeiliClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.settings_by_index: dict[str, dict] = {}

    def wait_task(self, task, **kwargs):
        self.calls.append(("wait_task", task, kwargs))
        return {"status": "succeeded"}

    def delete_index(self, uid: str):
        self.calls.append(("delete_index", uid))
        return {"taskUid": 1}

    def create_index(self, uid: str, primary_key: str):
        self.calls.append(("create_index", uid, primary_key))
        return {"taskUid": 2}

    def update_settings(self, index_uid: str, settings: dict):
        self.calls.append(("update_settings", index_uid, settings))
        self.settings_by_index[index_uid] = settings
        return {"taskUid": 3}

    def get_settings(self, index_uid: str):
        self.calls.append(("get_settings", index_uid))
        return self.settings_by_index[index_uid]

    def add_documents(self, index_uid: str, documents: list[dict]):
        self.calls.append(("add_documents", index_uid, documents))
        return {"taskUid": 4}


class DeleteFailureMeiliClient(FakeMeiliClient):
    def __init__(self, error_code: str) -> None:
        super().__init__()
        self.error_code = error_code

    def wait_task(self, task, **kwargs):
        self.calls.append(("wait_task", task, kwargs))
        if task == {"taskUid": 1}:
            payload = {
                "status": "failed",
                "error": {
                    "code": self.error_code,
                    "message": "delete failed",
                },
            }
            ignored_error_codes = set(kwargs.get("ignored_error_codes", ()))
            if self.error_code in ignored_error_codes:
                return payload
            raise MeiliTaskError(1, "failed", payload)
        return {"status": "succeeded"}


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


def test_visual_entity_artifact_uses_default_manifest_path(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    visual_entities = project_dir / "manifests" / "visual_entities.jsonl"
    write_jsonl(visual_entities, [{"entity_id": "entity_board"}])

    selected = visual_entity_artifact_path(project_dir)

    assert selected == visual_entities


def test_window_artifact_uses_default_segment_path(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    windows = project_dir / "segments" / "lecture_windows.jsonl"
    write_jsonl(windows, [{"window_id": "window_seg_1"}])

    selected = window_artifact_path(project_dir)

    assert selected == windows


def test_build_window_index_documents_serializes_context_and_semantic_text() -> None:
    segments = [
        _segment("seg_1", 1, 0.0, 2.0, "alpha intro", ["frame_000001"]),
        _segment("seg_2", 2, 3.0, 5.0, "beta target", ["frame_000002"]),
        _segment("seg_3", 3, 7.0, 9.0, "gamma outro", ["frame_000003"]),
    ]
    frames = [
        {"frame_id": "frame_000001", "timestamp": 1.0, "frame_path": "frames/1.jpg"},
        {"frame_id": "frame_000002", "timestamp": 4.0, "frame_path": "frames/2.jpg"},
        {"frame_id": "frame_000003", "timestamp": 8.0, "frame_path": "frames/3.jpg"},
    ]
    visual_entities = [
        {
            **_visual_entity("entity_beta", "frame_000002", "matrix beta", 4.0),
            "segment_id": "seg_2",
            "visual_description": "A beta matrix is highlighted.",
        }
    ]

    documents = build_window_index_documents(
        segments,
        frames=frames,
        visual_entities=visual_entities,
        neighbor_count=1,
    )

    document = next(item for item in documents if item["target_segment_id"] == "seg_2")
    assert document["window_id"].startswith("window_seg_2_")
    assert document["segment_id"] == "seg_2"
    assert document["source_segment_ids"] == ["seg_1", "seg_2", "seg_3"]
    assert document["start_time"] == 0.0
    assert document["end_time"] == 9.0
    assert document["timestamp_center"] == 4.5
    assert document["target_start_time"] == 3.0
    assert document["transcript_window_text"] == "alpha intro beta target gamma outro"
    assert document[LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD] == (
        "alpha intro beta target gamma outro matrix beta A beta matrix is highlighted."
    )
    assert document[LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD] == [
        "transcript_window_text",
        "visual_entities.text",
        "visual_entities.visual_description",
    ]
    assert [frame["frame_path"] for frame in document["frame_refs"]] == [
        "frames/1.jpg",
        "frames/2.jpg",
        "frames/3.jpg",
    ]
    assert document["visual_entities"][0]["entity_id"] == "entity_beta"
    assert document["evidence_window"]["target_segment"]["segment_id"] == "seg_2"
    assert document["window_config"] == {
        "mode": "neighbors",
        "neighbor_count": 1,
        "previous_neighbor_count": 1,
        "next_neighbor_count": 1,
    }


def test_build_project_windows_writes_jsonl_and_updates_manifest(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    segments_path = project_dir / "segments" / "lecture_segments.jsonl"
    manifest_path = project_dir / "manifests" / "project_manifest.json"
    write_jsonl(
        segments_path,
        [
            _segment("seg_1", 1, 0.0, 2.0, "alpha", []),
            _segment("seg_2", 2, 3.0, 5.0, "beta", []),
        ],
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"artifacts": {}, "counts": {}}), encoding="utf-8")

    summary = build_project_windows(project_dir=project_dir, neighbor_count=0)

    windows_path = project_dir / "segments" / "lecture_windows.jsonl"
    rows = [
        json.loads(line)
        for line in windows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert summary["counts"]["windows_total"] == 2
    assert rows[0]["target_segment_id"] == "seg_1"
    assert rows[0]["source_segment_ids"] == ["seg_1"]
    assert manifest["artifacts"]["lecture_windows"] == str(windows_path)
    assert manifest["counts"]["lecture_windows"] == 2
    assert manifest["window_indexing"]["window_config"]["neighbor_count"] == 0


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


def test_index_project_segments_appends_visual_entities_to_semantic_text(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    segments_path = project_dir / "segments" / "lecture_segments.jsonl"
    visual_entities_path = project_dir / "manifests" / "visual_entities.jsonl"
    write_jsonl(
        segments_path,
        [
            {
                "segment_id": "s1",
                "transcript_text": "The determinant is introduced.",
                "frame_refs": ["frame_000001"],
            }
        ],
    )
    write_jsonl(
        visual_entities_path,
        [
            {
                **_visual_entity("entity_a", "frame_000001", "det A", 1.0),
                "visual_description": "A determinant equation on the slide",
                "frame_path": "frames/frame_000001.jpg",
                "source": "vlm:stub-vlm",
            }
        ],
    )
    client = FakeMeiliClient()

    summary = index_project_segments(
        client,
        index_uid="local_segments",
        project_dir=project_dir,
    )

    add_call = next(call for call in client.calls if call[0] == "add_documents")
    document = add_call[2][0]
    assert document[LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD] == (
        "The determinant is introduced. det A A determinant equation on the slide"
    )
    assert document[LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD] == [
        "transcript_text",
        "visual_entities.text",
        "visual_entities.visual_description",
    ]
    assert document["visual_entities"][0]["source"] == "vlm:stub-vlm"
    assert "frame_path" not in document["visual_entities"][0]
    assert summary["embedded_visual_entities"] == 1
    assert summary["semantic_source_field_counts"]["visual_entities.visual_description"] == 1


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


def test_index_project_segments_applies_hybrid_embedder_profile_and_live_smoke(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    segments_path = project_dir / "segments" / "lecture_segments.jsonl"
    write_jsonl(segments_path, [{"segment_id": "s1", "transcript_text": "alpha"}])
    client = FakeMeiliClient()

    summary = index_project_segments(
        client,
        index_uid="local_segments",
        project_dir=project_dir,
        hybrid_embedder_profile=HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE,
        hybrid_embedder_name="lecture_embedder",
        hybrid_embedder_dimensions=768,
        hybrid_embedder_live_smoke=True,
    )

    settings_call = next(call for call in client.calls if call[0] == "update_settings")
    assert settings_call[2]["embedders"] == {
        "lecture_embedder": {
            "source": "userProvided",
            "dimensions": 768,
        }
    }
    assert ("get_settings", "local_segments") in client.calls
    assert summary["hybrid_embedder_profile"] == HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE
    assert summary["hybrid_embedder_hash"] == hybrid_embedder_settings_hash(
        {"embedders": settings_call[2]["embedders"]}
    )
    assert summary["hybrid_embedder_live_smoke"] == {
        "enabled": True,
        "ok": True,
        "checked_embedder_names": ["lecture_embedder"],
        "settings_hash": summary["hybrid_embedder_hash"],
    }
    assert summary["settings_snapshot"]["settings"]["embedders"] == settings_call[2]["embedders"]


def test_index_project_segments_redacts_hybrid_embedder_credentials_in_summary(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    segments_path = project_dir / "segments" / "lecture_segments.jsonl"
    write_jsonl(segments_path, [{"segment_id": "s1", "transcript_text": "alpha"}])
    client = FakeMeiliClient()

    summary = index_project_segments(
        client,
        index_uid="local_segments",
        project_dir=project_dir,
        hybrid_embedder_config={
            "embedders": {
                "default": {
                    "source": "openAi",
                    "model": "text-embedding-3-small",
                    "apiKey": "sk-private-test-key",
                    "documentTemplate": "{{doc.semantic_text}}",
                }
            }
        },
        hybrid_embedder_live_smoke=True,
    )

    settings_call = next(call for call in client.calls if call[0] == "update_settings")
    summary_json = json.dumps(summary)
    assert settings_call[2]["embedders"]["default"]["apiKey"] == "sk-private-test-key"
    assert "sk-private-test-key" not in summary_json
    assert summary["hybrid_embedder_snapshot"]["settings"]["embedders"]["default"]["apiKey"] == (
        "<redacted>"
    )
    assert summary["settings_snapshot"]["settings"]["embedders"]["default"]["apiKey"] == (
        "<redacted>"
    )
    assert summary["hybrid_embedder_live_smoke"]["ok"] is True


def test_index_project_segments_live_smoke_requires_hybrid_embedder_settings(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    segments_path = project_dir / "segments" / "lecture_segments.jsonl"
    write_jsonl(segments_path, [{"segment_id": "s1", "transcript_text": "alpha"}])

    with pytest.raises(ValueError, match="Hybrid embedder options require"):
        index_project_segments(
            FakeMeiliClient(),
            index_uid="local_segments",
            project_dir=project_dir,
            hybrid_embedder_live_smoke=True,
        )


def test_index_project_segments_reset_treats_missing_index_as_noop(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    segments_path = project_dir / "segments" / "lecture_segments.jsonl"
    write_jsonl(segments_path, [{"segment_id": "s1", "transcript_text": "alpha"}])
    client = DeleteFailureMeiliClient("index_not_found")

    summary = index_project_segments(
        client,
        index_uid="fresh_segments",
        project_dir=project_dir,
        reset=True,
    )

    delete_wait = next(call for call in client.calls if call[0] == "wait_task")
    assert delete_wait[2]["ignored_error_codes"] == {"index_not_found"}
    assert ("create_index", "fresh_segments", "segment_id") in client.calls
    assert summary["indexed_documents"] == 1


def test_index_project_segments_reset_propagates_non_missing_delete_failure(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    segments_path = project_dir / "segments" / "lecture_segments.jsonl"
    write_jsonl(segments_path, [{"segment_id": "s1", "transcript_text": "alpha"}])
    client = DeleteFailureMeiliClient("internal")

    with pytest.raises(MeiliTaskError, match="Meilisearch task 1 ended with status failed"):
        index_project_segments(
            client,
            index_uid="broken_segments",
            project_dir=project_dir,
            reset=True,
        )

    assert ("create_index", "broken_segments", "segment_id") not in client.calls


def test_index_project_windows_batches_documents_with_window_settings(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    segments_path = project_dir / "segments" / "lecture_segments.jsonl"
    write_jsonl(
        segments_path,
        [
            _segment("seg_1", 1, 0.0, 2.0, "alpha", []),
            _segment("seg_2", 2, 3.0, 5.0, "beta", []),
            _segment("seg_3", 3, 6.0, 8.0, "gamma", []),
        ],
    )
    client = FakeMeiliClient()

    summary = index_project_windows(
        client,
        index_uid="local_windows",
        project_dir=project_dir,
        batch_size=2,
        reset=True,
        neighbor_count=0,
    )

    add_calls = [call for call in client.calls if call[0] == "add_documents"]
    settings_call = next(call for call in client.calls if call[0] == "update_settings")
    assert len(add_calls) == 2
    assert add_calls[0][1] == "local_windows"
    assert [document["target_segment_id"] for document in add_calls[0][2]] == ["seg_1", "seg_2"]
    assert [document["target_segment_id"] for document in add_calls[1][2]] == ["seg_3"]
    assert all(document["source_segment_ids"] == [document["target_segment_id"]] for document in add_calls[0][2])
    assert ("delete_index", "local_windows") in client.calls
    assert ("create_index", "local_windows", "window_id") in client.calls
    assert settings_call[1] == "local_windows"
    assert "transcript_window_text" in settings_call[2]["searchableAttributes"]
    assert summary["index"] == "local_windows"
    assert summary["indexed_documents"] == 3
    assert summary["indexed_batches"] == 2
    assert summary["windows_path"] is None
    assert summary["segments_path"] == str(segments_path)
    assert summary["settings_profile"] == LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE
    assert summary["settings_hash"] == lecture_window_settings_hash(settings_call[2])
    assert summary["semantic_source_field_counts"] == {"transcript_window_text": 3}


def test_index_project_windows_applies_hybrid_embedder_profile(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    segments_path = project_dir / "segments" / "lecture_segments.jsonl"
    write_jsonl(
        segments_path,
        [
            _segment("seg_1", 1, 0.0, 2.0, "alpha", []),
        ],
    )
    client = FakeMeiliClient()

    summary = index_project_windows(
        client,
        index_uid="local_windows",
        project_dir=project_dir,
        neighbor_count=0,
        hybrid_embedder_profile=HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE,
        hybrid_embedder_dimensions=512,
    )

    settings_call = next(call for call in client.calls if call[0] == "update_settings")
    assert settings_call[2]["embedders"] == {
        "default": {
            "source": "userProvided",
            "dimensions": 512,
        }
    }
    assert summary["hybrid_embedder_profile"] == HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE
    assert summary["hybrid_embedder_snapshot"]["settings"] == {
        "embedders": settings_call[2]["embedders"]
    }
    assert summary["hybrid_embedder_live_smoke"] == {"enabled": False}


def test_index_project_visual_entities_batches_documents(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    visual_entities_path = project_dir / "manifests" / "visual_entities.jsonl"
    rows = [
        _visual_entity("entity_a", "frame_000001", "matrix A", 1.0),
        _visual_entity("entity_b", "frame_000002", "range grid", 2.0),
        _visual_entity("entity_c", "frame_000003", "bet size", 3.0),
    ]
    write_jsonl(visual_entities_path, rows)
    client = FakeMeiliClient()

    summary = index_project_visual_entities(
        client,
        index_uid="local_visual_entities",
        project_dir=project_dir,
        batch_size=2,
        reset=True,
    )

    add_calls = [call for call in client.calls if call[0] == "add_documents"]
    settings_call = next(call for call in client.calls if call[0] == "update_settings")
    assert len(add_calls) == 2
    assert add_calls[0][1] == "local_visual_entities"
    assert add_calls[0][2] == rows[:2]
    assert add_calls[1][2] == rows[2:]
    assert settings_call[1] == "local_visual_entities"
    assert ("delete_index", "local_visual_entities") in client.calls
    assert ("create_index", "local_visual_entities", "entity_id") in client.calls
    assert summary["index"] == "local_visual_entities"
    assert summary["indexed_documents"] == 3
    assert summary["indexed_batches"] == 2
    assert summary["visual_entities_path"] == str(visual_entities_path)
    assert summary["semantic_source_field_counts"] == {
        "text": 3,
        "visual_description": 3,
    }
    assert summary["source_counts"] == {"test": 3}
    assert summary["settings_profile"] == VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE
    assert summary["settings_hash"] == visual_entity_settings_hash(settings_call[2])


def test_index_project_visual_entities_preserves_optional_segment_hint(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    visual_entities_path = project_dir / "manifests" / "visual_entities.jsonl"
    row = {
        **_visual_entity("entity_a", "frame_000001", "matrix A", 1.0),
        "segment_id": "seg_1",
        "video_id": "video_1",
    }
    write_jsonl(visual_entities_path, [row])
    client = FakeMeiliClient()

    index_project_visual_entities(
        client,
        index_uid="local_visual_entities",
        project_dir=project_dir,
    )

    add_call = next(call for call in client.calls if call[0] == "add_documents")
    assert add_call[2][0]["segment_id"] == "seg_1"
    assert add_call[2][0]["video_id"] == "video_1"


def _segment(
    segment_id: str,
    sample_index: int,
    start_time: float,
    end_time: float,
    transcript_text: str,
    frame_refs: list[str],
) -> dict:
    return {
        "segment_id": segment_id,
        "project_id": "project",
        "video_id": "video",
        "video_name": "Lecture video",
        "sample_id": segment_id,
        "sample_index": sample_index,
        "start_time": start_time,
        "end_time": end_time,
        "timestamp_center": (start_time + end_time) / 2,
        "transcript_text": transcript_text,
        "frame_refs": frame_refs,
        "source": "test",
    }


def _visual_entity(entity_id: str, frame_id: str, text: str, timestamp: float) -> dict:
    return {
        "entity_id": entity_id,
        "project_id": "project",
        "frame_id": frame_id,
        "timestamp": timestamp,
        "frame_path": f"/tmp/{frame_id}.jpg",
        "bbox": None,
        "text": text,
        "entity_type": "visual_observation",
        "confidence": 0.91,
        "source": "test",
        "visual_description": text,
        "position": None,
        "relations": [],
        "parser_version": "test-v1",
        "source_model": "stub-vlm",
        "semantic_source_fields": ["text", "visual_description"],
    }
