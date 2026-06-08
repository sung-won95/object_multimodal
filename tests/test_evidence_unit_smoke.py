from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oarag.evidence_unit_smoke import run_evidence_unit_smoke


class FakeTask:
    uid = 1


class AvailableFakeClient:
    def __init__(self) -> None:
        self.documents: dict[str, list[dict[str, Any]]] = {}

    def health(self) -> dict[str, Any]:
        return {"status": "available"}

    def delete_index(self, uid: str) -> FakeTask:
        self.documents.pop(uid, None)
        return FakeTask()

    def create_index(self, uid: str, primary_key: str) -> FakeTask:
        self.documents.setdefault(uid, [])
        return FakeTask()

    def update_settings(self, index_uid: str, settings: dict[str, Any]) -> FakeTask:
        return FakeTask()

    def add_documents(self, index_uid: str, documents: list[dict[str, Any]]) -> FakeTask:
        self.documents.setdefault(index_uid, []).extend(documents)
        return FakeTask()

    def wait_task(
        self,
        task: FakeTask | None,
        timeout_seconds: float = 60.0,
        ignored_error_codes=(),
    ) -> dict[str, Any] | None:
        return {"status": "succeeded"} if task is not None else None

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if index_uid == "segment_baseline":
            return {
                "hits": [
                    {
                        "segment_id": "seg_private_2",
                        "transcript_text": "SECRET baseline transcript",
                        "timestamp_center": 12.0,
                        "_rankingScore": 0.42,
                    }
                ],
                "processingTimeMs": 1,
            }
        hits = list(self.documents.get(index_uid, []))
        hits = list(reversed(hits))
        return {"hits": hits[:limit], "processingTimeMs": 2}


class UnavailableFakeClient:
    def health(self) -> dict[str, Any]:
        raise OSError("connection refused on localhost:7700")


class QualityRerankFakeClient(AvailableFakeClient):
    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if index_uid != "evidence_index":
            return super().search(index_uid, query, limit=limit, filter=filter, **kwargs)
        hits = [
            {
                "evidence_unit_id": "evu_top_transcript_private",
                "project_id": "project_private",
                "video_id": "video_private",
                "target_segment_id": "seg_private_2",
                "source_segment_ids": ["seg_private_2"],
                "start_time": 20.0,
                "end_time": 22.0,
                "evidence_text": "SECRET top transcript",
                "semantic_text": "SECRET top semantic",
                "visual_state_ids": [],
                "visual_entity_ids": [],
                "verified_entity_link_ids": [],
                "candidate_entity_link_ids": [],
                "alignment_status": "transcript_only",
                "source_quality": {
                    "has_visual_state": False,
                    "has_visual_entity": False,
                    "has_vlm_entity": False,
                    "has_verified_link": False,
                    "has_timestamp_fallback_link": False,
                },
                "_rankingScore": 0.95,
            },
            {
                "evidence_unit_id": "evu_target_visual_private",
                "project_id": "project_private",
                "video_id": "video_private",
                "target_segment_id": "seg_private_1",
                "source_segment_ids": ["seg_private_1"],
                "start_time": 10.0,
                "end_time": 12.0,
                "evidence_text": "SECRET target transcript",
                "semantic_text": "SECRET target semantic",
                "visual_state_ids": ["vstate_private"],
                "visual_entity_ids": ["entity_private_1", "entity_private_2", "entity_private_3"],
                "verified_entity_link_ids": [],
                "candidate_entity_link_ids": ["link_private"],
                "alignment_status": "candidate",
                "source_quality": {
                    "has_visual_state": True,
                    "has_visual_entity": True,
                    "has_vlm_entity": True,
                    "has_verified_link": False,
                    "has_timestamp_fallback_link": True,
                },
                "_rankingScore": 0.55,
            },
        ]
        return {"hits": hits[:limit], "processingTimeMs": 2}


class ModalityAwareRerankFakeClient(AvailableFakeClient):
    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if index_uid != "evidence_index":
            return super().search(index_uid, query, limit=limit, filter=filter, **kwargs)
        hits = [
            {
                "evidence_unit_id": "evu_ocr_private",
                "project_id": "project_private",
                "video_id": "video_private",
                "target_segment_id": "seg_private_2",
                "source_segment_ids": ["seg_private_2"],
                "start_time": 20.0,
                "end_time": 22.0,
                "evidence_text": "SECRET OCR fallback visual evidence",
                "semantic_text": "SECRET OCR fallback semantic",
                "visual_state_ids": ["vstate_ocr"],
                "visual_entity_ids": ["entity_ocr"],
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
                "_rankingScore": 0.99,
            },
            {
                "evidence_unit_id": "evu_verified_private",
                "project_id": "project_private",
                "video_id": "video_private",
                "target_segment_id": "seg_private_1",
                "source_segment_ids": ["seg_private_1"],
                "start_time": 10.0,
                "end_time": 12.0,
                "evidence_text": "SECRET verified diagram arrow evidence",
                "semantic_text": "SECRET verified diagram arrow semantic",
                "visual_state_ids": ["vstate_verified"],
                "visual_entity_ids": ["entity_verified"],
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
                "_rankingScore": 0.6,
            },
        ]
        return {"hits": hits[:limit], "processingTimeMs": 2}


def test_evidence_unit_smoke_available_path_writes_sanitized_outputs(tmp_path: Path) -> None:
    project_dir = _write_project(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        project_dir=project_dir,
        extra_suite={
            "segment_index": "segment_baseline",
            "previous_neighbor_count": 0,
            "next_neighbor_count": 0,
        },
    )

    run = run_evidence_unit_smoke(
        client=AvailableFakeClient(),
        manifest_path=manifest_path,
        output_dir=tmp_path / "public",
        repo_root=tmp_path,
    )

    assert run.metrics_path.exists()
    assert run.query_results_path.exists()
    assert run.summary_path.exists()
    payload = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    assert payload["meilisearch"]["available"] is True
    suite = payload["suites"][0]
    assert suite["build"]["counts"]["evidence_units_total"] == 2
    assert suite["build"]["alignment_status_counts"] == {"candidate": 2}
    assert suite["build"]["source_quality_counts"]["units_with_candidate_link"] == 0
    assert suite["build"]["link_counts"]["verified_links"] == 0
    assert suite["build"]["link_counts"]["timestamp_fallback_links"] == 1
    assert suite["build"]["link_diagnostics"]["candidate_visual_support"][
        "units_with_candidate_visual_support"
    ] == 2
    assert suite["build"]["link_diagnostics"]["candidate_visual_support"][
        "units_with_timestamp_fallback_link"
    ] == 1
    assert suite["build"]["link_diagnostics"]["candidate_link_signal_counts"][
        "timestamp_fallback"
    ] == 1
    assert suite["build"]["link_diagnostics"]["verified_object_alignment"][
        "units_with_verified_object_alignment"
    ] == 0
    assert suite["build"]["link_diagnostics"]["verified_object_alignment"][
        "verified_links"
    ] == 0
    assert suite["build"]["link_diagnostics"]["verified_object_alignment"][
        "timestamp_fallback_counted_as_verified"
    ] is False
    assert suite["build"]["visual_state_coverage"]["source"] == "sampled_frame_midpoints"
    assert suite["build"]["visual_state_coverage"]["visual_states_total"] == 1
    assert suite["build"]["visual_state_coverage"]["evidence_units_with_visual_state"] == 2
    assert suite["build"]["visual_state_coverage"]["transcript_only_units"] == 0
    assert suite["build"]["visual_state_coverage"]["coverage_gate"]["status"] == "not_configured"
    assert suite["build"]["visual_state_coverage"]["interval_duration_seconds"]["buckets"]["15-30s"] == 1
    assert suite["index"]["status"] == "indexed"
    assert suite["rag_input_inspection"]["inspectable_top_hit_count"] == 1

    rows = [
        json.loads(line)
        for line in run.query_results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["status"] == "queried"
    assert rows[0]["top_evidence_unit"]["alignment_status"] == "candidate"
    assert rows[0]["top_evidence_unit"]["source_quality"]["has_verified_link"] is False
    assert rows[0]["top_evidence_unit"]["source_quality"]["has_timestamp_fallback_link"] is False
    assert rows[0]["top_evidence_unit"]["candidate_visual_support"][
        "has_candidate_visual_support"
    ] is True
    assert rows[0]["top_evidence_unit"]["verified_object_alignment"][
        "has_verified_object_alignment"
    ] is False
    assert rows[0]["top_evidence_unit"]["verified_object_alignment"][
        "timestamp_fallback_counted_as_verified"
    ] is False
    assert rows[0]["top_evidence_unit"]["content_coverage"]["semantic_text_char_count"] > 0
    assert rows[0]["top_evidence_unit"]["query_term_coverage"]["query_term_count"] > 0
    assert rows[0]["top_evidence_unit"]["query_term_coverage"]["combined_match_bucket"] in {
        "none",
        "low",
        "medium",
        "high",
        "very_high",
    }
    assert rows[0]["top_hit_memo"]["top_expected_match"] is False
    assert rows[0]["target_diagnostics"]["target_configured"] is True
    assert rows[0]["target_diagnostics"]["target_found_in_top_k"] is True
    assert rows[0]["target_diagnostics"]["target_rank"] == 2
    assert rows[0]["target_diagnostics"]["target_rank_bucket"] == "top5"
    assert rows[0]["target_diagnostics"]["target_evidence_unit_quality"]["has_verified_link"] is False
    assert rows[0]["target_diagnostics"]["target_evidence_unit_quality"]["has_timestamp_fallback_link"] is True
    assert rows[0]["target_diagnostics"]["target_evidence_unit_quality"][
        "candidate_visual_support"
    ]["timestamp_fallback_link_count"] == 1
    assert rows[0]["target_diagnostics"]["target_evidence_unit_quality"][
        "verified_object_alignment"
    ]["verified_link_count"] == 0
    assert rows[0]["target_diagnostics"]["target_content_coverage"]["semantic_text_char_count"] > 0
    assert rows[0]["target_diagnostics"]["target_query_term_coverage"]["query_term_count"] > 0
    assert rows[0]["target_diagnostics"]["top_vs_target_quality_delta"]["status"] == "available"
    assert rows[0]["target_diagnostics"]["top_vs_target_quality_delta"]["same_evidence_unit"] is False
    assert rows[0]["target_diagnostics"]["top_vs_target_content_delta"]["status"] == "available"
    assert "combined_query_term_match_count_delta" in rows[0]["target_diagnostics"]["top_vs_target_content_delta"]
    assert rows[0]["target_diagnostics"]["top_vs_target_quality_delta"]["verified_alignment_note"].startswith(
        "has_verified_link only reflects explicit verified links"
    )
    assert rows[0]["segment_baseline"]["top_hit_memo"]["top_expected_match"] is False
    assert rows[0]["rag_input_inspectable"] is True
    assert payload["target_rank_diagnostics"]["target_configured_count"] == 1
    assert payload["target_rank_diagnostics"]["target_found_in_top_k_count"] == 1
    assert payload["target_rank_diagnostics"]["rank_bucket_counts"] == {"top5": 1}
    assert payload["target_rank_diagnostics"]["found_target_candidate_link_signal_counts"][
        "timestamp_fallback"
    ] == 1
    assert payload["target_rank_diagnostics"]["found_target_verified_link_source_counts"][
        "explicit_verified_flag"
    ] == 0
    assert "found_target_query_term_bucket_counts" in payload["target_rank_diagnostics"]
    assert suite["target_rank_diagnostics"]["rank_bucket_counts"] == {"top5": 1}

    public_text = _public_text(run)
    for sensitive in [
        "PRIVATE RAW QUERY TEXT",
        "SECRET transcript one",
        "SECRET transcript two",
        "SECRET visual token",
        "SECRET baseline transcript",
        str(project_dir),
        "seg_private_1",
        "evu_seg_private_1",
        "video_private",
        "RAW QUERY TEXT",
    ]:
        assert sensitive not in public_text
    assert "Timestamp-only overlap is not counted" in run.summary_path.read_text(encoding="utf-8")
    assert "Visual-state interval overlap" in run.summary_path.read_text(encoding="utf-8")
    assert "redacted" in public_text


def test_evidence_unit_smoke_quality_rerank_compares_base_and_reranked(tmp_path: Path) -> None:
    project_dir = _write_project(tmp_path)
    manifest_path = _write_manifest(tmp_path, project_dir=project_dir)

    run = run_evidence_unit_smoke(
        client=QualityRerankFakeClient(),
        manifest_path=manifest_path,
        output_dir=tmp_path / "public",
        repo_root=tmp_path,
        quality_rerank=True,
    )

    payload = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    row = json.loads(run.query_results_path.read_text(encoding="utf-8").splitlines()[0])

    assert row["top_hit_memo"]["top_expected_match"] is False
    assert row["target_diagnostics"]["target_rank"] == 2
    assert row["rerank_diagnostics"]["enabled"] is True
    assert row["rerank_diagnostics"]["strategy"] == "deterministic_quality_v1"
    assert row["rerank_diagnostics"]["base_top_expected_match"] is False
    assert row["rerank_diagnostics"]["reranked_top_expected_match"] is True
    assert row["rerank_diagnostics"]["base_target_rank_bucket"] == "top5"
    assert row["rerank_diagnostics"]["reranked_target_rank_bucket"] == "top1"
    assert row["rerank_diagnostics"]["reranked_top"]["original_rank"] == 2
    components = row["rerank_diagnostics"]["reranked_top"]["score_components"]
    assert components["vlm_entity_presence"] > 0
    assert components["verified_link_presence"] == 0
    assert components["timestamp_fallback_penalty"] < 0
    assert row["reranked_top_hit_memo"]["top_expected_match"] is True
    assert row["reranked_top_evidence_unit"]["source_quality"]["has_vlm_entity"] is True
    assert payload["rerank_diagnostics"]["base_top_match_count"] == 0
    assert payload["rerank_diagnostics"]["reranked_top_match_count"] == 1
    assert payload["rerank_diagnostics"]["base_target_rank_bucket_counts"] == {"top5": 1}
    assert payload["rerank_diagnostics"]["reranked_target_rank_bucket_counts"] == {"top1": 1}
    assert payload["suites"][0]["rerank_diagnostics"]["top_changed_count"] == 1

    public_text = _public_text(run)
    for sensitive in [
        "SECRET top transcript",
        "SECRET target transcript",
        "evu_target_visual_private",
        "seg_private_1",
        str(project_dir),
    ]:
        assert sensitive not in public_text


def test_evidence_unit_smoke_modality_aware_rerank_reports_public_breakdown(
    tmp_path: Path,
) -> None:
    project_dir = _write_project(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        project_dir=project_dir,
        extra_suite={
            "modality_aware_rerank": True,
            "target_rank_limit": 2,
            "queries": [
                {
                    "query_id": "q_visual",
                    "query_label": "visual concept label",
                    "query_text": "PRIVATE RAW diagram arrow query should stay private",
                    "expected_segment_id": "seg_private_1",
                }
            ],
        },
    )

    run = run_evidence_unit_smoke(
        client=ModalityAwareRerankFakeClient(),
        manifest_path=manifest_path,
        output_dir=tmp_path / "public",
        repo_root=tmp_path,
    )

    payload = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    row = json.loads(run.query_results_path.read_text(encoding="utf-8").splitlines()[0])
    rerank = row["modality_aware_rerank"]

    assert rerank["enabled"] is True
    assert rerank["strategy"] == "modality_aware"
    assert rerank["query_type"] == "visual-heavy"
    assert rerank["top_changed"] is True
    assert rerank["base_target_rank_bucket"] == "top5"
    assert rerank["reranked_target_rank_bucket"] == "top1"
    assert rerank["failure_mode"] == "ranking_improved"
    assert row["top_evidence_unit"]["source_quality"]["has_verified_link"] is True
    top_rerank = row["top_evidence_unit"]["modality_aware_rerank"]
    assert top_rerank["score_components"]["verified_link"] > 0
    assert top_rerank["score_components"]["vlm_entity"] > 0
    assert "verified_link" in rerank["component_names"]
    assert payload["modality_aware_rerank_diagnostics"]["query_type_counts"] == {
        "visual-heavy": 1
    }
    assert payload["modality_aware_rerank_diagnostics"]["failure_mode_counts"] == {
        "ranking_improved": 1
    }

    public_text = _public_text(run)
    for sensitive in [
        "PRIVATE RAW diagram arrow query",
        "SECRET OCR fallback",
        "SECRET verified diagram",
        "evu_verified_private",
        "seg_private_1",
        str(project_dir),
    ]:
        assert sensitive not in public_text


def test_evidence_unit_smoke_unavailable_path_records_skip_reason(tmp_path: Path) -> None:
    project_dir = _write_project(tmp_path)
    manifest_path = _write_manifest(tmp_path, project_dir=project_dir)

    run = run_evidence_unit_smoke(
        client=UnavailableFakeClient(),
        manifest_path=manifest_path,
        output_dir=tmp_path / "public",
        repo_root=tmp_path,
    )

    payload = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    assert payload["meilisearch"]["available"] is False
    assert "connection refused" in payload["meilisearch"]["skip_reason"]
    assert payload["suites"][0]["build"]["counts"]["evidence_units_total"] == 2
    assert payload["suites"][0]["index"]["status"] == "skipped"

    rows = [
        json.loads(line)
        for line in run.query_results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["status"] == "skipped"
    assert rows[0]["segment_baseline"]["status"] == "skipped"
    assert rows[0]["target_diagnostics"]["target_rank_bucket"] == "not_queried"
    assert "PRIVATE RAW QUERY TEXT" not in _public_text(run)


def test_evidence_unit_smoke_dry_run_does_not_contact_meilisearch(tmp_path: Path) -> None:
    project_dir = _write_project(tmp_path)
    manifest_path = _write_manifest(tmp_path, project_dir=project_dir)

    run = run_evidence_unit_smoke(
        client=UnavailableFakeClient(),
        manifest_path=manifest_path,
        output_dir=tmp_path / "public",
        repo_root=tmp_path,
        dry_run=True,
    )

    payload = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["suites"][0]["build"]["status"] == "dry_run"
    assert payload["suites"][0]["index"]["status"] == "dry_run"


def _write_manifest(tmp_path: Path, *, project_dir: Path, extra_suite: dict[str, Any] | None = None) -> Path:
    suite = {
        "suite_id": "private_suite",
        "project_dir": str(project_dir),
        "index": "evidence_index",
        "limit": 1,
        "reset": True,
        "queries": [
            {
                "query_id": "q1",
                "query_label": "public concept label",
                "query_text": "PRIVATE RAW QUERY TEXT should stay private",
                "expected_segment_id": "seg_private_1",
            }
        ],
    }
    suite.update(extra_suite or {})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"run_id": "run_private", "suites": [suite]}),
        encoding="utf-8",
    )
    return manifest_path


def _write_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_private_1",
                "project_id": "project_private",
                "video_id": "video_private",
                "start_time": 10.0,
                "end_time": 12.0,
                "timestamp_center": 11.0,
                "transcript_text": "SECRET transcript one",
            },
            {
                "segment_id": "seg_private_2",
                "project_id": "project_private",
                "video_id": "video_private",
                "start_time": 20.0,
                "end_time": 22.0,
                "timestamp_center": 21.0,
                "transcript_text": "SECRET transcript two",
            },
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_private", "video_id": "video_private", "timestamp": 11.0}],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "entity_private",
                "frame_id": "frame_private",
                "video_id": "video_private",
                "timestamp": 11.0,
                "text": "SECRET visual token",
                "source": "ocr",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_private",
                "segment_id": "seg_private_1",
                "entity_id": "entity_private",
                "frame_id": "frame_private",
                "evidence": ["time_overlap", "timestamp_fallback"],
            }
        ],
    )
    return project_dir


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _public_text(run) -> str:
    return "\n".join(
        [
            run.metrics_path.read_text(encoding="utf-8"),
            run.query_results_path.read_text(encoding="utf-8"),
            run.summary_path.read_text(encoding="utf-8"),
        ]
    )
