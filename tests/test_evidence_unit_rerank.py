from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oarag.retrieval.evidence_unit_index import (
    classify_evidence_unit_query_modality,
    query_project_evidence_units,
)


class FakeEvidenceUnitClient:
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self.hits = hits

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
    ) -> dict[str, Any]:
        return {"hits": self.hits[:limit], "processingTimeMs": 3}


def test_query_modality_classifier_is_conservative() -> None:
    assert classify_evidence_unit_query_modality("Which diagram shows the red arrow?") == "visual-heavy"
    assert classify_evidence_unit_query_modality("What did the lecturer say about convergence?") == "speech-heavy"
    assert classify_evidence_unit_query_modality("What does the slide say about the diagram?") == "mixed"
    assert classify_evidence_unit_query_modality("loss curve slope") == "unknown"


def test_modality_aware_rerank_promotes_verified_visual_evidence(tmp_path: Path) -> None:
    project_dir = _write_project(tmp_path)
    hits = [_ocr_fallback_hit(), _verified_visual_hit()]

    response = query_project_evidence_units(
        client=FakeEvidenceUnitClient(hits),
        index_uid="evidence_units",
        project_dir=project_dir,
        query="Which diagram shows the loss curve arrow?",
        limit=2,
        evidence_unit_rerank="modality_aware",
    )

    candidates = response["candidates"]
    assert candidates[0]["evidence_unit_id"] == "evu_verified_visual"
    assert candidates[0]["modality_aware_rerank"]["query_type"] == "visual-heavy"
    assert candidates[0]["modality_aware_rerank"]["components"]["verified_link"] > 0
    assert candidates[0]["modality_aware_rerank"]["components"]["vlm_entity"] > 0
    assert candidates[1]["evidence_unit_id"] == "evu_ocr_fallback"
    assert candidates[1]["modality_aware_rerank"]["components"]["ocr_only_penalty"] < 0
    assert candidates[1]["modality_aware_rerank"]["components"]["timestamp_fallback_penalty"] < 0
    assert candidates[1]["modality_aware_rerank"]["components"]["verified_link"] == 0
    assert response["retrieval_context"]["evidence_unit_rerank"]["top_changed"] is True
    assert "Which diagram" not in json.dumps(candidates[0]["modality_aware_rerank"])


def test_speech_heavy_query_keeps_text_overlap_candidate_first(tmp_path: Path) -> None:
    project_dir = _write_project(tmp_path)
    transcript_hit = {
        "evidence_unit_id": "evu_speech",
        "project_id": "project_private",
        "target_segment_id": "seg_speech",
        "source_segment_ids": ["seg_speech"],
        "start_time": 1.0,
        "end_time": 3.0,
        "alignment_status": "transcript_only",
        "source_quality": {
            "has_visual_state": False,
            "has_visual_entity": False,
            "has_vlm_entity": False,
            "has_verified_link": False,
        },
        "evidence_text": "The lecturer says convergence follows from the gradient condition.",
        "semantic_text": "lecturer says convergence gradient condition",
        "transcript_window_text": "lecturer says convergence gradient condition",
        "_rankingScore": 0.93,
    }
    visual_hit = _verified_visual_hit()
    visual_hit["evidence_text"] = "The diagram shows a loss curve arrow."
    visual_hit["semantic_text"] = "diagram loss curve arrow"

    response = query_project_evidence_units(
        client=FakeEvidenceUnitClient([transcript_hit, visual_hit]),
        index_uid="evidence_units",
        project_dir=project_dir,
        query="What did the lecturer say about convergence?",
        limit=2,
        modality_aware_rerank=True,
    )

    assert response["candidates"][0]["evidence_unit_id"] == "evu_speech"
    assert response["candidates"][0]["modality_aware_rerank"]["query_type"] == "speech-heavy"
    assert response["candidates"][0]["modality_aware_rerank"]["components"][
        "semantic_text_query_overlap"
    ] > 0


def _write_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "evidence_units.jsonl",
        [{"project_id": "project_private", "evidence_unit_id": "evu_manifest"}],
    )
    return project_dir


def _ocr_fallback_hit() -> dict[str, Any]:
    return {
        "evidence_unit_id": "evu_ocr_fallback",
        "project_id": "project_private",
        "target_segment_id": "seg_ocr",
        "source_segment_ids": ["seg_ocr"],
        "start_time": 10.0,
        "end_time": 12.0,
        "visual_state_ids": ["state_ocr"],
        "visual_entity_ids": ["ent_ocr"],
        "verified_entity_link_ids": [],
        "candidate_entity_link_ids": ["link_time"],
        "candidate_entity_link_statuses": {"link_time": "timestamp_fallback"},
        "alignment_status": "candidate",
        "source_quality": {
            "has_visual_state": True,
            "has_visual_entity": True,
            "has_vlm_entity": False,
            "has_verified_link": False,
            "uses_ocr_only": True,
            "has_timestamp_fallback_link": True,
            "candidate_link_count": 0,
            "timestamp_fallback_link_count": 1,
            "verified_link_count": 0,
        },
        "evidence_text": "OCR label near a loss curve.",
        "semantic_text": "OCR loss curve",
        "transcript_window_text": "unrelated transcript",
        "_rankingScore": 0.99,
    }


def _verified_visual_hit() -> dict[str, Any]:
    return {
        "evidence_unit_id": "evu_verified_visual",
        "project_id": "project_private",
        "target_segment_id": "seg_visual",
        "source_segment_ids": ["seg_visual"],
        "start_time": 20.0,
        "end_time": 22.0,
        "visual_state_ids": ["state_visual"],
        "visual_entity_ids": ["ent_visual"],
        "verified_entity_link_ids": ["link_verified"],
        "candidate_entity_link_ids": ["link_verified"],
        "candidate_entity_link_statuses": {"link_verified": "verified"},
        "alignment_status": "verified",
        "source_quality": {
            "has_visual_state": True,
            "has_visual_entity": True,
            "has_vlm_entity": True,
            "has_verified_link": True,
            "uses_ocr_only": False,
            "has_timestamp_fallback_link": False,
            "candidate_link_count": 0,
            "timestamp_fallback_link_count": 0,
            "verified_link_count": 1,
        },
        "evidence_text": "Verified diagram shows a loss curve arrow.",
        "semantic_text": "verified visual diagram loss curve arrow",
        "transcript_window_text": "short transcript",
        "_rankingScore": 0.6,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
