import json
from pathlib import Path

from oarag.retrieval.evidence_unit_index import query_project_evidence_units
from oarag.retrieval.evidence_units import build_project_evidence_units
from oarag.retrieval.project_index import index_project_evidence_units


def test_build_project_evidence_units_marks_timestamp_only_as_candidate_fallback(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_before", 0.0, 4.0, "We set up optimization."),
            _segment("seg_target", 10.0, 14.0, "This gradient arrow shows the descent direction."),
            _segment("seg_after", 20.0, 24.0, "Now the loss curve gets smaller."),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {"frame_id": "frame_gradient", "timestamp": 12.0, "frame_path": "frames/gradient.jpg"},
            {"frame_id": "frame_loss", "timestamp": 22.0, "frame_path": "frames/loss.jpg"},
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_gradient_arrow",
                "project_id": "sample_project",
                "frame_id": "frame_gradient",
                "timestamp": 12.0,
                "frame_path": "frames/gradient.jpg",
                "bbox": None,
                "text": "gradient arrow",
                "entity_type": "diagram_component",
                "confidence": 0.91,
                "source": "vlm",
                "visual_description": "Arrow indicating the descent direction.",
            },
            {
                "entity_id": "ent_loss_ocr",
                "project_id": "sample_project",
                "frame_id": "frame_loss",
                "timestamp": 22.0,
                "frame_path": "frames/loss.jpg",
                "bbox": None,
                "text": "loss",
                "entity_type": "ocr_text",
                "confidence": 0.8,
                "source": "ocr",
            },
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_gradient",
                "project_id": "sample_project",
                "segment_id": "seg_target",
                "entity_id": "ent_gradient_arrow",
                "frame_id": "frame_gradient",
                "link_type": "time_overlap+lexical_match",
                "score": 0.9,
                "evidence": ["time_overlap", "lexical_match"],
                "time_overlap": True,
                "lexical_match": ["gradient"],
                "mention_candidate": [],
            },
            {
                "link_id": "link_loss_timestamp",
                "project_id": "sample_project",
                "segment_id": "seg_after",
                "entity_id": "ent_loss_ocr",
                "frame_id": "frame_loss",
                "link_type": "time_overlap",
                "score": 0.2,
                "evidence": ["time_overlap", "timestamp_fallback"],
                "time_overlap": True,
                "lexical_match": [],
                "mention_candidate": [],
            },
        ],
    )

    summary = build_project_evidence_units(
        project_dir=project_dir,
        previous_neighbor_count=0,
        next_neighbor_count=1,
    )

    rows = _read_jsonl(project_dir / "segments" / "evidence_units.jsonl")
    target = next(row for row in rows if row["target_segment_id"] == "seg_target")
    assert summary["counts"]["evidence_units_total"] == 3
    assert target["source_segment_ids"] == ["seg_target", "seg_after"]
    assert "This gradient arrow" in target["transcript_window_text"]
    assert target["visual_state_ids"]
    assert target["visual_entity_ids"] == ["ent_gradient_arrow", "ent_loss_ocr"]
    assert target["candidate_entity_link_ids"] == ["link_gradient", "link_loss_timestamp"]
    assert target["verified_entity_link_ids"] == []
    assert target["candidate_entity_link_statuses"]["link_gradient"] == "candidate"
    assert target["candidate_entity_link_statuses"]["link_loss_timestamp"] == "timestamp_fallback"
    assert target["alignment_status"] == "candidate"
    assert target["source_quality"]["has_vlm_entity"] is True
    assert target["source_quality"]["has_verified_link"] is False
    assert target["source_quality"]["has_timestamp_fallback_link"] is True
    assert "gradient arrow" in target["semantic_text"]

    manifest = json.loads((project_dir / "manifests" / "project_manifest.json").read_text())
    assert manifest["artifacts"]["evidence_units"].endswith("segments/evidence_units.jsonl")
    assert manifest["counts"]["evidence_units"] == 3


def test_build_project_evidence_units_supports_transcript_only_project(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "transcript_only"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments.jsonl",
        [
            _segment("seg_only", 0.0, 5.0, "A transcript-only explanation."),
        ],
    )

    build_project_evidence_units(project_dir=project_dir)

    rows = _read_jsonl(project_dir / "segments" / "evidence_units.jsonl")
    assert rows[0]["alignment_status"] == "transcript_only"
    assert rows[0]["modality"] == ["speech"]
    assert rows[0]["visual_state_ids"] == []
    assert rows[0]["visual_entity_ids"] == []


def test_index_project_evidence_units_indexes_artifact_and_preserves_fallback_status(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    _write_jsonl(
        project_dir / "segments" / "evidence_units.jsonl",
        [
            {
                "evidence_unit_id": "evu_seg_1",
                "project_id": "sample_project",
                "video_id": "sample_video",
                "target_segment_id": "seg_1",
                "source_segment_ids": ["seg_1"],
                "start_time": 10.0,
                "end_time": 14.0,
                "transcript_window_text": "This gradient arrow shows descent.",
                "visual_state_ids": ["vstate_1"],
                "visual_entity_ids": ["ent_gradient_arrow"],
                "verified_entity_link_ids": [],
                "candidate_entity_link_ids": ["link_timestamp"],
                "candidate_entity_link_statuses": {
                    "link_timestamp": "timestamp_fallback",
                },
                "alignment_status": "candidate",
                "source_quality": {
                    "has_visual_state": True,
                    "has_visual_entity": True,
                    "has_vlm_entity": False,
                    "has_verified_link": False,
                    "has_timestamp_fallback_link": True,
                },
                "evidence_text": "Transcript: This gradient arrow shows descent.",
                "semantic_text": "gradient arrow descent",
            }
        ],
    )
    client = _FakeMeiliClient()

    summary = index_project_evidence_units(
        client,
        index_uid="sample_evidence_units",
        project_dir=project_dir,
        reset=True,
    )

    assert client.created_indexes == [("sample_evidence_units", "evidence_unit_id")]
    assert client.deleted_indexes == ["sample_evidence_units"]
    assert client.settings["searchableAttributes"][:2] == ["semantic_text", "evidence_text"]
    assert summary["indexed_documents"] == 1
    assert summary["alignment_status_counts"] == {"candidate": 1}
    indexed = client.documents[0]
    assert indexed["evidence_unit_id"] == "evu_seg_1"
    assert indexed["candidate_entity_link_statuses"] == {
        "link_timestamp": "timestamp_fallback"
    }
    assert indexed["verified_entity_link_ids"] == []


def test_query_project_evidence_units_returns_required_fields(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    _write_jsonl(
        project_dir / "segments" / "evidence_units.jsonl",
        [
            {
                "evidence_unit_id": "evu_seg_1",
                "project_id": "sample_project",
            }
        ],
    )
    hit = {
        "evidence_unit_id": "evu_seg_1",
        "project_id": "sample_project",
        "video_id": "sample_video",
        "target_segment_id": "seg_1",
        "source_segment_ids": ["seg_1"],
        "start_time": 10.0,
        "end_time": 14.0,
        "visual_state_ids": ["vstate_1"],
        "visual_entity_ids": ["ent_gradient_arrow"],
        "verified_entity_link_ids": [],
        "candidate_entity_link_ids": ["link_timestamp"],
        "candidate_entity_link_statuses": {
            "link_timestamp": "timestamp_fallback",
        },
        "alignment_status": "candidate",
        "source_quality": {
            "has_verified_link": False,
            "timestamp_fallback_link_count": 1,
        },
        "evidence_text": "Transcript: gradient arrow",
        "semantic_text": "gradient arrow",
        "_rankingScore": 0.9,
    }
    client = _FakeMeiliClient(search_hits=[hit])

    response = query_project_evidence_units(
        client=client,
        index_uid="sample_evidence_units",
        project_dir=project_dir,
        query="gradient arrow",
        limit=1,
    )

    assert client.search_calls == [
        {
            "index_uid": "sample_evidence_units",
            "query": "gradient arrow",
            "limit": 1,
            "filter": 'project_id = "sample_project"',
        }
    ]
    candidate = response["candidates"][0]
    required_fields = {
        "evidence_unit_id",
        "project_id",
        "video_id",
        "target_segment_id",
        "source_segment_ids",
        "start_time",
        "end_time",
        "visual_state_ids",
        "visual_entity_ids",
        "candidate_entity_link_ids",
        "candidate_entity_link_statuses",
        "alignment_status",
        "source_quality",
    }
    assert required_fields <= set(candidate)
    assert candidate["verified_entity_link_ids"] == []
    assert candidate["candidate_entity_link_statuses"]["link_timestamp"] == "timestamp_fallback"
    assert candidate["source_quality"]["has_verified_link"] is False


def test_explicit_verified_link_is_not_downgraded_to_timestamp_fallback(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "verified_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_verified", 10.0, 14.0, "The instructor refers to this object."),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {"frame_id": "frame_verified", "timestamp": 12.0, "frame_path": "frames/verified.jpg"},
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_verified",
                "project_id": "verified_project",
                "frame_id": "frame_verified",
                "timestamp": 12.0,
                "frame_path": "frames/verified.jpg",
                "bbox": None,
                "text": "object",
                "entity_type": "diagram_component",
                "confidence": 0.9,
                "source": "vlm",
            },
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_explicit_verified_bool",
                "project_id": "verified_project",
                "segment_id": "seg_verified",
                "entity_id": "ent_verified",
                "frame_id": "frame_verified",
                "link_type": "time_overlap",
                "score": 0.2,
                "evidence": ["time_overlap", "timestamp_fallback"],
                "time_overlap": True,
                "lexical_match": [],
                "mention_candidate": [],
                "verified": True,
            },
            {
                "link_id": "link_explicit_verified_status",
                "project_id": "verified_project",
                "segment_id": "seg_verified",
                "entity_id": "ent_verified",
                "frame_id": "frame_verified",
                "link_type": "time_overlap",
                "score": 0.2,
                "evidence": [],
                "time_overlap": True,
                "lexical_match": [],
                "mention_candidate": [],
                "verification_status": "verified",
            },
            {
                "link_id": "link_plain_timestamp",
                "project_id": "verified_project",
                "segment_id": "seg_verified",
                "entity_id": "ent_verified",
                "frame_id": "frame_verified",
                "link_type": "time_overlap",
                "score": 0.2,
                "evidence": ["time_overlap"],
                "time_overlap": True,
                "lexical_match": [],
                "mention_candidate": [],
            },
        ],
    )

    summary = build_project_evidence_units(project_dir=project_dir)

    rows = _read_jsonl(project_dir / "segments" / "evidence_units.jsonl")
    unit = rows[0]
    assert unit["verified_entity_link_ids"] == [
        "link_explicit_verified_bool",
        "link_explicit_verified_status",
    ]
    assert unit["candidate_entity_link_ids"] == ["link_plain_timestamp"]
    assert unit["candidate_entity_link_statuses"] == {
        "link_plain_timestamp": "timestamp_fallback"
    }
    assert unit["source_quality"]["has_verified_link"] is True
    assert unit["source_quality"]["verified_link_count"] == 2
    assert unit["source_quality"]["timestamp_fallback_link_count"] == 1
    assert unit["alignment_status"] == "verified"
    assert summary["counts"]["verified_links"] == 2
    assert summary["counts"]["timestamp_fallback_links"] == 1
    assert summary["alignment_status_counts"] == {"verified": 1}


def _segment(segment_id: str, start_time: float, end_time: float, text: str) -> dict:
    return {
        "segment_id": segment_id,
        "project_id": "sample_project",
        "video_id": "sample_video",
        "sample_id": segment_id,
        "sample_index": 0,
        "start_time": start_time,
        "end_time": end_time,
        "timestamp_center": (start_time + end_time) / 2.0,
        "transcript_text": text,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class _FakeMeiliClient:
    def __init__(self, search_hits: list[dict] | None = None) -> None:
        self.search_hits = search_hits or []
        self.created_indexes: list[tuple[str, str]] = []
        self.deleted_indexes: list[str] = []
        self.settings: dict = {}
        self.documents: list[dict] = []
        self.search_calls: list[dict] = []

    def delete_index(self, index_uid: str):
        self.deleted_indexes.append(index_uid)
        return {"taskUid": 1}

    def create_index(self, index_uid: str, primary_key: str):
        self.created_indexes.append((index_uid, primary_key))
        return {"taskUid": 2}

    def update_settings(self, index_uid: str, settings: dict):
        self.settings = settings
        return {"taskUid": 3}

    def add_documents(self, index_uid: str, documents: list[dict]):
        self.documents.extend(documents)
        return {"taskUid": 4}

    def search(self, index_uid: str, query: str, *, limit: int, filter: str):
        self.search_calls.append(
            {
                "index_uid": index_uid,
                "query": query,
                "limit": limit,
                "filter": filter,
            }
        )
        return {"hits": self.search_hits[:limit], "processingTimeMs": 1}

    def wait_task(self, task, ignored_error_codes=None):
        return task
