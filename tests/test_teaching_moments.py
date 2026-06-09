from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oarag.benchmark import run_benchmark
from oarag.evaluation.benchmark import DEFAULT_MATRIX_VARIANTS


class TeachingMomentClient:
    def __init__(self, hits_by_index: dict[str, list[dict[str, Any]]]) -> None:
        self.hits_by_index = hits_by_index
        self.searches: list[tuple[str, str, int, Any]] = []

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        self.searches.append((index_uid, query, limit, filter))
        hits = list(self.hits_by_index.get(index_uid, []))
        return {"hits": hits[:limit], "processingTimeMs": 2, "indexUid": index_uid}


def test_teaching_moment_span_prefers_explanation_context_over_concept_label(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    _write_project_segments(
        project_dir,
        [
            _segment("seg_label", 10.0, "Gradient clipping label appears on a slide."),
            _segment(
                "seg_explain",
                300.0,
                "The clipping threshold prevents exploding gradients by limiting the update norm.",
            ),
        ],
    )
    _write_jsonl(project_dir / "manifests" / "frames_manifest.jsonl", [])
    _write_jsonl(project_dir / "manifests" / "visual_entities.jsonl", [])
    _write_jsonl(project_dir / "manifests" / "entity_links.jsonl", [])
    _write_jsonl(
        project_dir / "segments" / "evidence_units.jsonl",
        [
            {
                "evidence_unit_id": "evu_seg_label",
                "project_id": "tm_project",
                "video_id": "tm_video",
                "target_segment_id": "seg_label",
                "source_segment_ids": ["seg_label"],
                "start_time": 10.0,
                "end_time": 14.0,
                "concept_labels": ["gradient clipping"],
                "concept_aliases": ["clipping threshold"],
                "concept_search_text": "gradient clipping clipping threshold",
                "evidence_text": "Label-only concept card.",
                "semantic_text": "gradient clipping",
                "source_quality": {
                    "concept_count": 1,
                    "concept_label_count": 1,
                    "concept_alias_count": 1,
                    "candidate_link_signal_counts": _zero_candidate_signal_counts(),
                    "verified_link_source_counts": _zero_verified_source_counts(),
                },
            },
            {
                "evidence_unit_id": "evu_seg_explain",
                "project_id": "tm_project",
                "video_id": "tm_video",
                "target_segment_id": "seg_explain",
                "source_segment_ids": ["seg_explain"],
                "start_time": 300.0,
                "end_time": 304.0,
                "concept_labels": ["gradient clipping"],
                "concept_aliases": ["clipping threshold"],
                "concept_search_text": "gradient clipping threshold exploding gradients update norm",
                "evidence_text": (
                    "Clipping threshold prevents exploding gradients by limiting update norm."
                ),
                "semantic_text": (
                    "gradient clipping threshold prevents exploding gradients update norm"
                ),
                "transcript_window_text": (
                    "The clipping threshold prevents exploding gradients by limiting the update norm."
                ),
                "source_quality": {
                    "concept_count": 1,
                    "concept_label_count": 1,
                    "concept_alias_count": 1,
                    "candidate_link_signal_counts": _zero_candidate_signal_counts(),
                    "verified_link_source_counts": _zero_verified_source_counts(),
                },
            },
        ],
    )
    manifest_path = _write_matrix_manifest(
        tmp_path,
        project_dir=project_dir,
        query_text="SECRET clipping threshold prevents exploding gradients update norm",
        expected_segment_id="seg_explain",
        expected_time_hint="300-304s",
        variants=["teaching_moment_span"],
    )
    client = TeachingMomentClient(
        {
            "tm_segments": [
                _hit("seg_label", 10.0, "Gradient clipping label appears on a slide.", 0.99),
                _hit(
                    "seg_explain",
                    300.0,
                    "The clipping threshold prevents exploding gradients.",
                    0.45,
                ),
            ]
        }
    )

    run = run_benchmark(
        client=client,
        manifest_path=manifest_path,
        output_dir=tmp_path / "matrix",
        repo_root=tmp_path,
    )

    suite = run.metrics["suites"][0]
    variant = suite["variant_metrics"]["teaching_moment_span"]
    assert suite["variant_count"] == 1
    assert variant["teaching_moment_span"]["enabled_query_count"] == 1
    assert variant["teaching_moment_span"]["top_changed_count"] == 1
    assert variant["target_rank_bucket_counts"] == {"top1": 1}
    assert variant["hit_at_10s"] == 1.0

    row = _only_row(run.query_results_path)
    assert row["top1_expected_segment_match"] is True
    assert row["teaching_moment_span"]["top_changed"] is True
    assert row["top_candidate"]["teaching_moment_span"]["aggregate_counts"][
        "evidence_match_count"
    ] > 0
    assert "query_text_overlap" in row["top_candidate"]["teaching_moment_span"]["reason_codes"]

    public_text = _public_output_text(run)
    assert "SECRET clipping threshold" not in public_text
    assert "prevents exploding gradients" not in public_text
    assert "seg_explain" not in public_text
    assert "evu_seg_explain" not in public_text
    assert str(project_dir) not in public_text
    assert "span:" in public_text


def test_teaching_moment_span_works_without_graph_or_concept_signal(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_project_segments(
        project_dir,
        [
            _segment("seg_background", 5.0, "A short unrelated intro."),
            _segment(
                "seg_lexical",
                120.0,
                "Dropout masks activations during training to reduce co-adaptation.",
            ),
        ],
    )
    _write_jsonl(project_dir / "manifests" / "frames_manifest.jsonl", [])
    _write_jsonl(project_dir / "manifests" / "visual_entities.jsonl", [])
    _write_jsonl(project_dir / "manifests" / "entity_links.jsonl", [])
    manifest_path = _write_matrix_manifest(
        tmp_path,
        project_dir=project_dir,
        query_text="SECRET raw dropout query",
        expected_segment_id="seg_lexical",
        expected_time_hint="120-124s",
        variants=[
            {
                "variant_id": "teaching_moment_span",
                "hybrid_retrieval": True,
                "hybrid_query_vector_dimensions": 3,
            }
        ],
    )
    client = TeachingMomentClient(
        {
            "tm_segments": [
                _hit(
                    "seg_lexical",
                    120.0,
                    "Dropout masks activations during training.",
                    0.81,
                )
            ]
        }
    )

    run = run_benchmark(
        client=client,
        manifest_path=manifest_path,
        output_dir=tmp_path / "matrix",
        repo_root=tmp_path,
    )

    row = _only_row(run.query_results_path)
    assert row["top1_expected_segment_match"] is True
    assert row["teaching_moment_span"]["candidate_count"] == 1
    assert row["top_candidate"]["teaching_moment_span"]["aggregate_counts"][
        "evidence_unit_count"
    ] == 0
    assert row["semantic_retrieval"]["semantic_channel_executed"] is True
    assert any(search[2] >= 1 for search in client.searches)


def test_teaching_moment_span_keeps_timestamp_only_signal_candidate_only(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    _write_project_segments(
        project_dir,
        [_segment("seg_time", 42.0, "The diagram is mentioned at this moment.")],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_time", "timestamp": 42.0, "frame_path": "/private/frame.jpg"}],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_time",
                "project_id": "tm_project",
                "video_id": "tm_video",
                "frame_id": "frame_time",
                "timestamp": 42.0,
                "text": "SECRET visual label",
                "source": "ocr",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_time",
                "project_id": "tm_project",
                "segment_id": "seg_time",
                "entity_id": "ent_time",
                "frame_id": "frame_time",
                "link_type": "time_overlap",
                "evidence": ["time_overlap", "timestamp_fallback"],
                "time_overlap": True,
            }
        ],
    )
    manifest_path = _write_matrix_manifest(
        tmp_path,
        project_dir=project_dir,
        query_text="SECRET raw timestamp query",
        expected_segment_id="seg_time",
        expected_time_hint="42-46s",
        variants=["teaching_moment_span"],
    )
    client = TeachingMomentClient(
        {"tm_segments": [_hit("seg_time", 42.0, "The diagram is mentioned.", 0.7)]}
    )

    run = run_benchmark(
        client=client,
        manifest_path=manifest_path,
        output_dir=tmp_path / "matrix",
        repo_root=tmp_path,
    )

    row = _only_row(run.query_results_path)
    assert row["candidate_visual_support"]["timestamp_fallback_link_count"] == 1
    assert row["candidate_visual_support"]["candidate_link_signal_counts"][
        "timestamp_fallback"
    ] == 1
    assert row["verified_object_alignment"]["has_verified_object_alignment"] is False
    assert row["verified_object_alignment"]["verified_link_count"] == 0
    assert row["verified_object_alignment"]["timestamp_fallback_counted_as_verified"] is False
    assert row["top_candidate"]["verified_object_alignment"]["verified_link_count"] == 0

    public_text = _public_output_text(run)
    assert "SECRET visual label" not in public_text
    assert "/private/frame.jpg" not in public_text


def test_required_paper_matrix_variants_remain_nine() -> None:
    manifest = json.loads(
        Path("eval/mit_deep_learning_stt/benchmark_matrix_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    required = manifest["paper_matrix"]["required_variants"]

    assert len(DEFAULT_MATRIX_VARIANTS) == 9
    assert [variant["variant_id"] for variant in DEFAULT_MATRIX_VARIANTS] == required
    assert "teaching_moment_span" not in required


def _write_matrix_manifest(
    tmp_path: Path,
    *,
    project_dir: Path,
    query_text: str,
    expected_segment_id: str,
    expected_time_hint: str | None = None,
    variants: list[Any],
) -> Path:
    manifest_path = tmp_path / f"matrix_{len(list(tmp_path.glob('matrix_*.json')))}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "teaching_moment_matrix",
                "deltas": [5, 10],
                "suites": [
                    {
                        "suite_id": "teaching_moment_public",
                        "type": "retrieval_answer_matrix",
                        "domain": "public_synthetic",
                        "project_dir": str(project_dir),
                        "queries": [
                            {
                                "query_id": "q_teaching_moment",
                                "public_label": "teaching moment",
                                "query_text": query_text,
                                "expected_segment_id": expected_segment_id,
                                "expected_time_hint": expected_time_hint,
                            }
                        ],
                        "index": "tm_segments",
                        "limit": 2,
                        "candidate_pool_limit": 6,
                        "span_window_seconds": 30,
                        "neighbor_count": 0,
                        "include_answer": True,
                        "variants": variants,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_project_segments(project_dir: Path, segments: list[dict[str, Any]]) -> None:
    _write_jsonl(project_dir / "segments" / "lecture_segments_aligned.jsonl", segments)


def _segment(segment_id: str, start_time: float, transcript_text: str) -> dict[str, Any]:
    return {
        "segment_id": segment_id,
        "project_id": "tm_project",
        "video_id": "tm_video",
        "sample_id": segment_id,
        "start_time": start_time,
        "end_time": start_time + 4.0,
        "timestamp_center": start_time + 2.0,
        "transcript_text": transcript_text,
    }


def _hit(segment_id: str, start_time: float, transcript_text: str, score: float) -> dict[str, Any]:
    return {
        "segment_id": segment_id,
        "sample_id": segment_id,
        "video_id": "tm_video",
        "start_time": start_time,
        "end_time": start_time + 4.0,
        "timestamp_center": start_time + 2.0,
        "transcript_text": transcript_text,
        "_rankingScore": score,
    }


def _only_row(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    return rows[0]


def _public_output_text(run: Any) -> str:
    return "\n".join(
        [
            run.metrics_path.read_text(encoding="utf-8"),
            run.metrics_csv_path.read_text(encoding="utf-8"),
            run.query_results_path.read_text(encoding="utf-8"),
            run.summary_path.read_text(encoding="utf-8"),
        ]
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _zero_candidate_signal_counts() -> dict[str, int]:
    return {
        "temporal_overlap": 0,
        "lexical_overlap": 0,
        "mention_deictic_hook": 0,
        "spatial_position": 0,
        "visual_text_overlap": 0,
        "vlm_object_visual_description_overlap": 0,
        "semantic_domain_hint": 0,
        "timestamp_fallback": 0,
    }


def _zero_verified_source_counts() -> dict[str, int]:
    return {
        "explicit_verified_flag": 0,
        "explicit_verified_status": 0,
        "human_gold": 0,
        "vlm_verifier": 0,
        "strict_deterministic_rule": 0,
        "unspecified_verified": 0,
    }
