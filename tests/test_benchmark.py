from __future__ import annotations

import json
from pathlib import Path

from oarag.benchmark import parse_time_hint, run_benchmark
from oarag.evaluation.quality_gate import evaluate_retrieval_quality_gate


class FakeClient:
    def __init__(self) -> None:
        self.searches: list[tuple[str, str, int, str | list[str] | None]] = []

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
    ) -> dict:
        self.searches.append((index_uid, query, limit, filter))
        if index_uid == "local_visual_index" and "bet size" in query.lower():
            hits = [
                {
                    "entity_id": "local_project__ent_bet",
                    "local_entity_id": "ent_bet",
                    "project_id": "local_project",
                    "frame_id": "frame_000012",
                    "timestamp": 12.0,
                    "frame_path": "/tmp/frame.jpg",
                    "text": "bet size",
                    "entity_type": "ocr_text",
                    "confidence": 0.9,
                    "source": "test",
                    "_rankingScore": 0.99,
                }
            ]
        elif "gravity" in query.lower():
            hits = [
                {
                    "segment_id": "seg_gravity",
                    "sample_id": "sample_gravity",
                    "video_id": "edu_video",
                    "timestamp_center": 101.0,
                    "transcript_text": "gravity example",
                    "_rankingScore": 0.9,
                }
            ]
        elif "bet size" in query.lower():
            hits = [
                {
                    "segment_id": "seg_local_0",
                    "sample_id": "seg_local_0",
                    "video_id": "local_video",
                    "start_time": 58.0,
                    "end_time": 62.0,
                    "timestamp_center": 60.0,
                    "transcript_text": "Introductory aside",
                    "_rankingScore": 0.95,
                },
                {
                    "segment_id": "seg_local_1",
                    "sample_id": "seg_local_1",
                    "video_id": "local_video",
                    "start_time": 10.0,
                    "end_time": 14.0,
                    "timestamp_center": 12.0,
                    "transcript_text": "Bet size is visible",
                    "_rankingScore": 0.4,
                }
            ]
        else:
            hits = []
        return {"hits": hits[:limit], "processingTimeMs": 3, "indexUid": index_uid}


class FakeAblationClient:
    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
    ) -> dict:
        if index_uid == "public_segments":
            hits = [
                {
                    "segment_id": "seg_wrap_public",
                    "sample_id": "seg_wrap_public",
                    "video_id": "public_demo_video",
                    "start_time": 30.0,
                    "end_time": 34.0,
                    "timestamp_center": 32.0,
                    "transcript_text": "PUBLIC SYNTHETIC wrong transcript must not leak",
                    "_rankingScore": 0.91,
                },
                {
                    "segment_id": "seg_loss_public",
                    "sample_id": "seg_loss_public",
                    "video_id": "public_demo_video",
                    "start_time": 10.0,
                    "end_time": 14.0,
                    "timestamp_center": 12.0,
                    "transcript_text": "PUBLIC SYNTHETIC correct transcript must not leak",
                    "_rankingScore": 0.8,
                },
            ]
            return {"hits": hits[:limit], "processingTimeMs": 5, "indexUid": index_uid}
        if index_uid == "public_visual_entities":
            hits = [
                {
                    "entity_id": "ent_loss_public",
                    "frame_id": "frame_loss_public",
                    "timestamp": 12.0,
                    "text": "PUBLIC VISUAL LABEL loss curve",
                    "_rankingScore": 0.97,
                }
            ]
            return {"hits": hits[:limit], "processingTimeMs": 7, "indexUid": index_uid}
        return {"hits": [], "processingTimeMs": 1, "indexUid": index_uid}


class FakeMatrixClient:
    def __init__(self) -> None:
        self.searches: list[tuple[str, str, str, list[float] | None]] = []

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        hybrid: dict | None = None,
        vector: list[float] | None = None,
        filter: str | list[str] | None = None,
    ) -> dict:
        mode = "semantic" if hybrid else "lexical"
        self.searches.append((index_uid, query, mode, vector))
        if index_uid == "public_evidence_units":
            hits = _public_evidence_unit_hits()
            return {"hits": hits[:limit], "processingTimeMs": 6, "indexUid": index_uid}
        if index_uid == "public_windows":
            hits = [
                {
                    "window_id": "window_loss_public",
                    "target_segment_id": "seg_loss_public",
                    "sample_id": "seg_loss_public",
                    "video_id": "public_demo_video",
                    "start_time": 0.0,
                    "end_time": 34.0,
                    "timestamp_center": 12.0,
                    "target_start_time": 10.0,
                    "target_end_time": 14.0,
                    "target_timestamp_center": 12.0,
                    "source_segment_ids": [
                        "seg_intro_public",
                        "seg_loss_public",
                        "seg_wrap_public",
                    ],
                    "transcript_window_text": "PUBLIC SYNTHETIC window must not leak",
                    "_rankingScore": 0.89,
                }
            ]
            return {"hits": hits[:limit], "processingTimeMs": 4, "indexUid": index_uid}
        if index_uid == "public_visual_entities":
            hits = [
                {
                    "entity_id": "ent_loss_public",
                    "frame_id": "frame_loss_public",
                    "timestamp": 12.0,
                    "text": "PUBLIC VISUAL LABEL loss curve",
                    "_rankingScore": 0.9,
                }
            ]
            return {"hits": hits[:limit], "processingTimeMs": 2, "indexUid": index_uid}
        if index_uid == "public_segments":
            hits = [
                {
                    "segment_id": "seg_wrap_public",
                    "sample_id": "seg_wrap_public",
                    "video_id": "public_demo_video",
                    "start_time": 30.0,
                    "end_time": 34.0,
                    "timestamp_center": 32.0,
                    "transcript_text": "PUBLIC SYNTHETIC wrong transcript must not leak",
                    "_rankingScore": 0.95 if mode == "lexical" else 0.7,
                },
                {
                    "segment_id": "seg_loss_public",
                    "sample_id": "seg_loss_public",
                    "video_id": "public_demo_video",
                    "start_time": 10.0,
                    "end_time": 14.0,
                    "timestamp_center": 12.0,
                    "transcript_text": "PUBLIC SYNTHETIC correct transcript must not leak",
                    "_rankingScore": 0.55 if mode == "lexical" else 0.96,
                },
            ]
            if "validation loss" in query:
                hits = list(reversed(hits))
            return {"hits": hits[:limit], "processingTimeMs": 5, "indexUid": index_uid}
        return {"hits": [], "processingTimeMs": 1, "indexUid": index_uid}


class TimestampFallbackMatrixClient:
    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        hybrid: dict | None = None,
        vector: list[float] | None = None,
        filter: str | list[str] | None = None,
    ) -> dict:
        if index_uid == "fallback_segments":
            return {
                "hits": [
                    {
                        "segment_id": "seg_fallback",
                        "sample_id": "seg_fallback",
                        "video_id": "fallback_video",
                        "start_time": 10.0,
                        "end_time": 14.0,
                        "timestamp_center": 12.0,
                        "transcript_text": "SECRET fallback transcript",
                        "_rankingScore": 0.9,
                    }
                ][:limit],
                "processingTimeMs": 3,
                "indexUid": index_uid,
            }
        if index_uid == "fallback_visual":
            return {
                "hits": [
                    {
                        "entity_id": "ent_fallback",
                        "frame_id": "frame_fallback",
                        "timestamp": 12.0,
                        "text": "SECRET visual label",
                        "_rankingScore": 0.8,
                    }
                ][:limit],
                "processingTimeMs": 2,
                "indexUid": index_uid,
            }
        return {"hits": [], "processingTimeMs": 1, "indexUid": index_uid}


def _public_evidence_unit_hits() -> list[dict]:
    candidate_signal_counts = {
        "temporal_overlap": 1,
        "lexical_overlap": 1,
        "mention_deictic_hook": 0,
        "spatial_position": 0,
        "visual_text_overlap": 1,
        "vlm_object_visual_description_overlap": 0,
        "semantic_domain_hint": 1,
        "timestamp_fallback": 0,
    }
    verified_signal_counts = {
        "temporal_overlap": 1,
        "lexical_overlap": 1,
        "mention_deictic_hook": 1,
        "spatial_position": 1,
        "visual_text_overlap": 1,
        "vlm_object_visual_description_overlap": 1,
        "semantic_domain_hint": 1,
        "timestamp_fallback": 0,
    }
    empty_verified_sources = {
        "explicit_verified_flag": 0,
        "explicit_verified_status": 0,
        "human_gold": 0,
        "vlm_verifier": 0,
        "strict_deterministic_rule": 0,
        "unspecified_verified": 0,
    }
    verified_sources = {
        "explicit_verified_flag": 1,
        "explicit_verified_status": 1,
        "human_gold": 0,
        "vlm_verifier": 1,
        "strict_deterministic_rule": 0,
        "unspecified_verified": 0,
    }
    return [
        {
            "evidence_unit_id": "evu_wrap_candidate_public",
            "project_id": "public_retrieval_ablation",
            "video_id": "public_demo_video",
            "target_segment_id": "seg_wrap_public",
            "source_segment_ids": ["seg_wrap_public"],
            "start_time": 30.0,
            "end_time": 34.0,
            "visual_state_ids": ["state_wrap_public"],
            "visual_entity_ids": ["ent_wrap_public"],
            "verified_entity_link_ids": [],
            "candidate_entity_link_ids": ["link_wrap_candidate_public"],
            "candidate_entity_link_statuses": {"link_wrap_candidate_public": "candidate"},
            "alignment_status": "candidate",
            "source_quality": {
                "has_visual_state": True,
                "has_visual_entity": True,
                "has_vlm_entity": False,
                "has_verified_link": False,
                "uses_ocr_only": True,
                "has_detected_text": True,
                "has_visual_description": False,
                "visual_state_detected_text_count": 1,
                "visual_entity_detected_text_count": 1,
                "visual_description_count": 0,
                "has_timestamp_fallback_link": False,
                "candidate_link_count": 1,
                "timestamp_fallback_link_count": 0,
                "verified_link_count": 0,
                "candidate_link_signal_counts": candidate_signal_counts,
                "verified_link_source_counts": empty_verified_sources,
                "candidate_visual_support": {
                    "has_candidate_visual_support": True,
                    "visual_state_count": 1,
                    "visual_entity_count": 1,
                    "candidate_link_count": 1,
                    "timestamp_fallback_link_count": 0,
                    "candidate_link_signal_counts": candidate_signal_counts,
                    "paper_claim_eligible": False,
                },
                "verified_object_alignment": {
                    "has_verified_object_alignment": False,
                    "verified_link_count": 0,
                    "verified_link_source_counts": empty_verified_sources,
                    "timestamp_fallback_counted_as_verified": False,
                    "paper_claim_eligible": False,
                },
            },
            "evidence_text": "PUBLIC SYNTHETIC wrap evidence mentions loss curve slope but must not leak",
            "semantic_text": "PUBLIC SYNTHETIC wrap semantic mentions loss curve slope but must not leak",
            "transcript_window_text": (
                "PUBLIC SYNTHETIC wrap evidence-unit transcript mentions loss curve slope "
                "but must not leak"
            ),
            "_rankingScore": 0.93,
        },
        {
            "evidence_unit_id": "evu_loss_verified_public",
            "project_id": "public_retrieval_ablation",
            "video_id": "public_demo_video",
            "target_segment_id": "seg_loss_public",
            "source_segment_ids": ["seg_intro_public", "seg_loss_public", "seg_wrap_public"],
            "start_time": 10.0,
            "end_time": 14.0,
            "visual_state_ids": ["state_loss_public"],
            "visual_entity_ids": ["ent_loss_public"],
            "verified_entity_link_ids": ["link_loss_verified_public"],
            "candidate_entity_link_ids": ["link_loss_verified_public"],
            "candidate_entity_link_statuses": {"link_loss_verified_public": "verified"},
            "alignment_status": "verified",
            "source_quality": {
                "has_visual_state": True,
                "has_visual_entity": True,
                "has_vlm_entity": True,
                "has_verified_link": True,
                "uses_ocr_only": False,
                "has_detected_text": True,
                "has_visual_description": True,
                "visual_state_detected_text_count": 1,
                "visual_entity_detected_text_count": 1,
                "visual_description_count": 1,
                "has_timestamp_fallback_link": False,
                "candidate_link_count": 0,
                "timestamp_fallback_link_count": 0,
                "verified_link_count": 1,
                "candidate_link_signal_counts": verified_signal_counts,
                "verified_link_source_counts": verified_sources,
                "candidate_visual_support": {
                    "has_candidate_visual_support": True,
                    "visual_state_count": 1,
                    "visual_entity_count": 1,
                    "candidate_link_count": 0,
                    "timestamp_fallback_link_count": 0,
                    "candidate_link_signal_counts": verified_signal_counts,
                    "paper_claim_eligible": False,
                },
                "verified_object_alignment": {
                    "has_verified_object_alignment": True,
                    "verified_link_count": 1,
                    "verified_link_source_counts": verified_sources,
                    "timestamp_fallback_counted_as_verified": False,
                    "paper_claim_eligible": True,
                },
            },
            "evidence_text": (
                "PUBLIC SYNTHETIC loss verified evidence mentions loss curve slope "
                "but must not leak"
            ),
            "semantic_text": (
                "PUBLIC SYNTHETIC loss verified semantic mentions loss curve slope "
                "but must not leak"
            ),
            "transcript_window_text": (
                "PUBLIC SYNTHETIC loss evidence-unit transcript mentions loss curve slope "
                "but must not leak"
            ),
            "_rankingScore": 0.91,
        },
    ]


MIT_PAPER_MATRIX_VARIANTS = [
    "segment_lexical",
    "domain_lexicon",
    "hybrid",
    "window",
    "window_hybrid",
    "rerank",
    "evidence_unit_candidate",
    "evidence_unit_verified",
    "evidence_unit_quality_rerank",
]


def test_parse_time_hint_extracts_multiple_ranges() -> None:
    assert parse_time_hint("95-102s or 126-131s") == [(95.0, 102.0), (126.0, 131.0)]


def test_mit_deep_learning_matrix_manifest_schema_smoke() -> None:
    manifest_path = Path("eval/mit_deep_learning_stt/benchmark_matrix_manifest.json")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    suites = manifest["suites"]

    assert manifest["run_id"] == "mit_deep_learning_stt_paper_matrix_v1"
    assert manifest["output_dir"] == "../../reports/mit_deep_learning_eval/paper_matrix_v1"
    assert manifest["paper_matrix"]["required_variants"] == MIT_PAPER_MATRIX_VARIANTS
    assert manifest["paper_matrix"]["baseline_variant_id"] == "segment_lexical"
    assert manifest["paper_matrix"]["domain_lexicon"] == "domain_lexicon.json"
    assert len(suites) == 24
    assert {suite["type"] for suite in suites} == {"retrieval_answer_matrix"}
    assert {suite["window_index"] for suite in suites} == {"mit_deep_learning_stt_windows"}
    assert {suite["index"] for suite in suites} == {"mit_deep_learning_stt_segments"}
    assert {suite["visual_index"] for suite in suites} == {
        "mit_deep_learning_stt_visual_entities"
    }
    assert {suite["evidence_unit_index"] for suite in suites} == {
        "mit_deep_learning_stt_evidence_units"
    }

    for suite in suites:
        assert suite["variants"] == MIT_PAPER_MATRIX_VARIANTS
        assert suite["domain_lexicon"] == "domain_lexicon.json"
        assert suite["hybrid_query_vector_dimensions"] == 384
        assert suite["hybrid_query_vector_embedder"] == "default"
        assert suite["include_answer"] is True
        assert suite["dataset_descriptor"]["privacy"] == "aggregate_only"
        assert not Path(suite["project_dir"]).is_absolute()
        assert not Path(suite["queries"]).is_absolute()
        assert suite["project_dir"].startswith(
            "../../artifacts/paper_mit_deep_learning/projects/"
        )
        assert suite["queries"].startswith("suites/")

    assert "/Users/" not in manifest_text
    assert "transcript_text" not in manifest_text
    assert "reference_answer" not in manifest_text


def test_run_benchmark_writes_cross_domain_outputs(tmp_path: Path) -> None:
    eduvidqa_path = tmp_path / "eduvidqa.jsonl"
    _write_jsonl(
        eduvidqa_path,
        [
            {
                "dataset_name": "eduvidqa",
                "subset_name": "mathsc_timestamp",
                "split_name": "test",
                "sample_index": 1,
                "sample_id": "sample_gravity",
                "video_name": "edu_video",
                "question": "What does gravity do?",
                "answer": "It pulls things down.",
                "transcript_text": "gravity example",
                "timestamp_points": [100.0],
                "has_timestamp": True,
                "qid": "sample_gravity",
                "raw_keys": [],
            }
        ],
    )

    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_local_0",
                "project_id": "local_project",
                "video_id": "local_video",
                "sample_id": "seg_local_0",
                "sample_index": 0,
                "start_time": 58.0,
                "end_time": 62.0,
                "timestamp_center": 60.0,
                "transcript_text": "Introductory aside",
                "frame_refs": [],
            },
            {
                "segment_id": "seg_local_1",
                "project_id": "local_project",
                "video_id": "local_video",
                "sample_id": "seg_local_1",
                "sample_index": 1,
                "start_time": 10.0,
                "end_time": 14.0,
                "timestamp_center": 12.0,
                "transcript_text": "Bet size is visible",
                "frame_refs": ["frame_000012"],
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000012", "timestamp": 12.0, "frame_path": "/tmp/frame.jpg"}],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_bet",
                "project_id": "local_project",
                "frame_id": "frame_000012",
                "timestamp": 12.0,
                "frame_path": "/tmp/frame.jpg",
                "bbox": None,
                "text": "bet size",
                "entity_type": "ocr_text",
                "confidence": 0.9,
                "source": "test",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_1",
                "project_id": "local_project",
                "segment_id": "seg_local_1",
                "entity_id": "ent_bet",
                "frame_id": "frame_000012",
                "link_type": "time_overlap+lexical_match",
                "score": 0.8,
                "evidence": ["time_overlap", "lexical_match"],
                "time_overlap": True,
                "lexical_match": ["bet", "size"],
                "mention_candidate": [],
            }
        ],
    )
    (project_dir / "domain_lexicon.json").write_text(
        json.dumps({"aliases": {"bet": ["wager"], "size": ["sizing"]}}),
        encoding="utf-8",
    )

    queries_path = tmp_path / "queries.csv"
    queries_path.write_text(
        "query_id,video_id,query_text,expected_topic,expected_time_hint,expected_visual_hint,notes\n"
        "q1,local_video,bet size 10-14s,bet sizing,10-14s,bet size text,safe\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "test_run",
                "output_dir": "perf_out",
                "deltas": [5, 10, 15],
                "suites": [
                    {
                        "suite_id": "edu_suite",
                        "type": "eduvidqa",
                        "domain": "public_eduvidqa",
                        "input": str(eduvidqa_path),
                        "index": "edu_index",
                        "limit": 3,
                        "diagnostic_top_k": 2,
                    },
                    {
                        "suite_id": "local_suite",
                        "type": "local_project",
                        "domain": "local_pilot",
                        "project_dir": str(project_dir),
                        "queries": str(queries_path),
                        "index": "local_index",
                        "visual_index": "local_visual_index",
                        "domain_lexicon": "domain_lexicon.json",
                        "limit": 3,
                        "rerank": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    client = FakeClient()
    run = run_benchmark(client=client, manifest_path=manifest_path, repo_root=tmp_path)

    assert run.metrics_path.exists()
    assert run.query_results_path.exists()
    assert run.summary_path.exists()
    assert run.metrics["suite_count"] == 2
    assert run.metrics["anti_overfit"]["domain_count"] == 2
    local_metrics = next(
        suite for suite in run.metrics["suites"] if suite["suite_id"] == "local_suite"
    )
    edu_metrics = next(
        suite for suite in run.metrics["suites"] if suite["suite_id"] == "edu_suite"
    )
    assert edu_metrics["top1_recall"] == 1.0
    assert edu_metrics["top5_recall"] == 1.0
    assert edu_metrics["top10_recall"] == 1.0
    assert edu_metrics["top50_recall"] == 1.0
    assert edu_metrics["median_abs_error"] == 1.0
    assert edu_metrics["p75_abs_error"] == 1.0
    assert edu_metrics["p90_abs_error"] == 1.0
    assert edu_metrics["diagnostic_top_k"] == 2
    assert local_metrics["frame_backed_ratio"] == 1.0
    assert local_metrics["linked_entity_backed_ratio"] == 1.0
    assert local_metrics["visual_index"] == "local_visual_index"
    assert local_metrics["top1_mean_abs_error"] == 0.0
    assert local_metrics["rerank"]["enabled"] is True
    assert local_metrics["rerank"]["strategy"] == "deterministic_evidence_v1"
    assert local_metrics["domain_lexicon"]["enabled"] is True
    assert local_metrics["domain_lexicon"]["source_path"] == str(
        (project_dir / "domain_lexicon.json").resolve()
    )
    query_rows = [
        json.loads(line)
        for line in run.query_results_path.read_text(encoding="utf-8").splitlines()
    ]
    local_row = next(row for row in query_rows if row["suite_id"] == "local_suite")
    edu_row = next(row for row in query_rows if row["suite_id"] == "edu_suite")
    assert edu_row["topk_recall"] == {"1": True, "5": True, "10": True, "50": True}
    assert edu_row["diagnostic_candidates"] == [
        {
            "rank": 1,
            "segment_id": "seg_gravity",
            "sample_id": "sample_gravity",
            "video_id": "edu_video",
            "timestamp_center": 101.0,
            "abs_error": 1.0,
            "same_sample": True,
            "same_video": True,
        }
    ]
    assert local_row["top_candidate"]["segment_id"] == "seg_local_1"
    assert local_row["visual_index"] == "local_visual_index"
    assert local_row["top_rerank"]["original_rank"] in {1, 2}
    assert any(call[0] == "local_visual_index" for call in client.searches)
    summary = run.summary_path.read_text(encoding="utf-8")
    assert "| local_suite | local_project | local_pilot | on (domain_lexicon.json) |" in summary
    assert "on (deterministic_evidence_v1)" in summary
    assert "Anti-Overfit View" in summary
    assert "Retrieval Diagnostics" in summary


def test_retrieval_ablation_public_fixture_outputs_are_sanitized(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures/public_retrieval_ablation_project").resolve()
    run = run_benchmark(
        client=FakeAblationClient(),
        manifest_path=fixture_dir / "benchmark_manifest.json",
        output_dir=tmp_path / "ablation",
        repo_root=Path.cwd(),
    )

    assert run.metrics_path.exists()
    assert run.query_results_path.exists()
    assert run.summary_path.exists()
    assert run.metrics["query_count"] == 4
    suite = run.metrics["suites"][0]
    assert suite["schema_version"] == "retrieval-ablation-public-v1"
    assert suite["query_count"] == 1
    assert suite["mode_count"] == 4
    mode_metrics = suite["mode_metrics"]
    assert mode_metrics["transcript-only"]["hit_at_5s"] == 1.0
    assert mode_metrics["transcript-only"]["frame_backed_ratio"] == 0.0
    assert mode_metrics["transcript-only"]["linked_entity_ratio"] == 0.0
    assert mode_metrics["visual-only"]["frame_backed_ratio"] == 1.0
    assert mode_metrics["visual-only"]["linked_entity_ratio"] == 0.0
    assert mode_metrics["time-aligned"]["frame_backed_ratio"] == 1.0
    assert mode_metrics["object-aligned"]["linked_entity_ratio"] == 1.0

    rows = [
        json.loads(line)
        for line in run.query_results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["mode"] for row in rows} == {
        "transcript-only",
        "visual-only",
        "time-aligned",
        "object-aligned",
    }
    assert all("query_text" not in row for row in rows)
    assert all(row["privacy"]["raw_query_text"] == "redacted" for row in rows)
    assert all(row["top_candidate"]["ref"] for row in rows)

    public_text = "\n".join(
        [
            run.metrics_path.read_text(encoding="utf-8"),
            run.query_results_path.read_text(encoding="utf-8"),
            run.summary_path.read_text(encoding="utf-8"),
        ]
    )
    for sensitive in [
        "PUBLIC RAW QUERY",
        "loss curve slope should stay private-safe",
        "PUBLIC SYNTHETIC",
        "PUBLIC VISUAL LABEL",
        "seg_loss_public",
        "seg_wrap_public",
        "ent_loss_public",
        "frame_loss_public",
        "frames/frame_loss_public.jpg",
        str(fixture_dir),
    ]:
        assert sensitive not in public_text
    assert "Retrieval Ablation Modes" in public_text
    assert "raw_query_text" in public_text
    assert "redacted" in public_text


def test_local_project_resolves_domain_lexicon_relative_to_manifest(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_local_1",
                "project_id": "local_project",
                "video_id": "local_video",
                "sample_id": "seg_local_1",
                "sample_index": 1,
                "start_time": 10.0,
                "end_time": 14.0,
                "timestamp_center": 12.0,
                "transcript_text": "Bet size is visible",
                "frame_refs": ["frame_000012"],
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000012", "timestamp": 12.0, "frame_path": "/tmp/frame.jpg"}],
    )
    _write_jsonl(project_dir / "manifests" / "visual_entities.jsonl", [])
    _write_jsonl(project_dir / "manifests" / "entity_links.jsonl", [])

    manifest_dir = tmp_path / "benchmarks"
    manifest_dir.mkdir()
    lexicon_path = manifest_dir / "domain_lexicon.json"
    lexicon_path.write_text(
        json.dumps({"aliases": {"bet": ["wager"], "size": ["sizing"]}}),
        encoding="utf-8",
    )
    queries_path = manifest_dir / "queries.csv"
    queries_path.write_text(
        "query_id,video_id,query_text,expected_topic,expected_time_hint,expected_visual_hint,notes\n"
        "q1,local_video,bet size 10-14s,bet sizing,10-14s,bet size text,safe\n",
        encoding="utf-8",
    )

    manifest_path = manifest_dir / "benchmark.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "manifest_relative_lexicon",
                "suites": [
                    {
                        "suite_id": "local_manifest_lexicon",
                        "type": "local_project",
                        "project_dir": str(project_dir),
                        "queries": "queries.csv",
                        "index": "local_index",
                        "domain_lexicon": "domain_lexicon.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    run = run_benchmark(client=FakeClient(), manifest_path=manifest_path, repo_root=tmp_path)

    suite = run.metrics["suites"][0]
    assert suite["domain_lexicon"]["enabled"] is True
    assert suite["domain_lexicon"]["source_path"] == str(lexicon_path.resolve())


def test_retrieval_answer_matrix_fixture_writes_aggregate_outputs(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures/public_retrieval_ablation_project").resolve()
    client = FakeMatrixClient()
    run = run_benchmark(
        client=client,
        manifest_path=fixture_dir / "benchmark_matrix_manifest.json",
        output_dir=tmp_path / "matrix",
        repo_root=Path.cwd(),
    )

    assert run.metrics_path.exists()
    assert run.metrics_csv_path.exists()
    assert run.query_results_path.exists()
    assert run.summary_path.exists()
    assert run.semantic_smoke_path.exists()
    assert run.metrics["query_count"] == 9
    suite = run.metrics["suites"][0]
    assert suite["schema_version"] == "retrieval-answer-ablation-matrix-v1"
    assert suite["query_count"] == 1
    assert suite["variant_count"] == 9
    assert set(suite["variant_metrics"]) == set(MIT_PAPER_MATRIX_VARIANTS)
    assert suite["variant_metrics"]["segment_lexical"]["config"]["domain_lexicon"] is False
    assert suite["variant_metrics"]["domain_lexicon"]["config"]["domain_lexicon"] is True
    assert suite["variant_metrics"]["hybrid"]["config"]["hybrid_retrieval"] is True
    assert suite["variant_metrics"]["window"]["config"]["index_kind"] == "window"
    assert suite["variant_metrics"]["window_hybrid"]["config"]["hybrid_retrieval"] is True
    assert suite["variant_metrics"]["rerank"]["config"]["rerank"] is True
    assert suite["variant_metrics"]["rerank"]["config"]["candidate_pool_limit"] == 30
    assert suite["variant_metrics"]["evidence_unit_candidate"]["config"][
        "evidence_unit_priority"
    ] == "candidate"
    assert suite["variant_metrics"]["evidence_unit_verified"]["config"][
        "evidence_unit_priority"
    ] == "verified"
    assert suite["variant_metrics"]["evidence_unit_quality_rerank"]["config"][
        "evidence_unit_priority"
    ] == "quality"
    assert suite["variant_metrics"]["evidence_unit_quality_rerank"]["config"][
        "evidence_unit_rerank"
    ] == "modality_aware"
    assert suite["variant_metrics"]["window"]["hit_at_10s"] == 1.0
    assert suite["variant_metrics"]["rerank"]["hit_at_10s"] == 1.0
    assert suite["variant_metrics"]["evidence_unit_candidate"]["hit_at_10s"] == 1.0
    assert suite["variant_metrics"]["evidence_unit_candidate"]["mrr_at_max_delta"] == 0.5
    assert suite["variant_metrics"]["evidence_unit_candidate"]["target_rank_bucket_counts"] == {
        "top5": 1
    }
    assert suite["variant_metrics"]["evidence_unit_verified"]["target_rank_bucket_counts"] == {
        "top1": 1
    }
    assert suite["variant_metrics"]["evidence_unit_candidate"][
        "candidate_visual_support_ratio"
    ] == 1.0
    assert suite["variant_metrics"]["evidence_unit_candidate"][
        "verified_object_alignment_ratio"
    ] == 0.0
    assert suite["variant_metrics"]["evidence_unit_verified"][
        "verified_object_alignment_ratio"
    ] == 1.0
    assert suite["variant_metrics"]["evidence_unit_verified"]["vlm_entity_coverage_ratio"] == 1.0
    assert suite["variant_metrics"]["evidence_unit_candidate"]["ocr_only_coverage_ratio"] == 1.0
    assert suite["variant_metrics"]["evidence_unit_quality_rerank"][
        "modality_aware_rerank"
    ]["enabled_query_count"] == 1
    assert suite["variant_metrics"]["evidence_unit_quality_rerank"][
        "modality_aware_rerank"
    ]["top_changed_count"] == 1
    assert suite["variant_metrics"]["evidence_unit_quality_rerank"][
        "modality_aware_rerank"
    ]["score_component_presence_counts"]["verified_link"] == 1
    assert suite["variant_metrics"]["evidence_unit_candidate"]["skipped_count"] == 0
    assert suite["variant_metrics"]["evidence_unit_candidate"][
        "candidate_link_signal_counts"
    ]["temporal_overlap"] == 1
    assert suite["variant_metrics"]["evidence_unit_candidate"][
        "candidate_link_signal_counts"
    ]["lexical_overlap"] == 1
    assert suite["variant_metrics"]["evidence_unit_candidate"][
        "candidate_link_signal_counts"
    ]["visual_text_overlap"] == 1
    assert suite["variant_metrics"]["evidence_unit_verified"][
        "candidate_link_signal_counts"
    ]["vlm_object_visual_description_overlap"] == 1
    assert suite["variant_metrics"]["evidence_unit_verified"][
        "verified_link_source_counts"
    ]["explicit_verified_flag"] == 1
    assert suite["variant_metrics"]["evidence_unit_verified"][
        "verified_link_source_counts"
    ]["vlm_verifier"] == 1
    assert suite["variant_metrics"]["rerank"]["grounded_answer_ratio"] == 1.0
    assert suite["variant_metrics"]["window"]["answer_citation_precision"] == 1.0
    assert suite["variant_metrics"]["window"]["answer_citation_recall"] == 1.0
    assert suite["variant_metrics"]["window"]["expected_citation_hit_ratio"] == 1.0
    assert suite["variant_metrics"]["window"]["mean_unsupported_claim_count"] == 0.0
    assert suite["variant_metrics"]["window"]["candidate_visual_support_ratio"] == 1.0
    assert suite["variant_metrics"]["window"]["verified_object_alignment_ratio"] == 0.0
    assert suite["variant_metrics"]["window"]["timestamp_fallback_counted_as_verified_count"] == 0
    assert "vlm_object_visual_description_overlap" in suite["variant_metrics"]["window"][
        "candidate_link_signal_counts"
    ]
    assert "explicit_verified_flag" in suite["variant_metrics"]["window"][
        "verified_link_source_counts"
    ]
    assert suite["variant_metrics"]["rerank"]["mean_unsupported_claim_count"] == 2.0
    assert suite["variant_metrics"]["window"]["answer_grounding_gap_counts"] == {
        "grounded_expected_citation": 1
    }
    assert suite["variant_metrics"]["segment_lexical"]["answer_grounding_gap_counts"] == {
        "unsupported_claims": 1
    }
    assert suite["variant_metrics"]["segment_lexical"]["answer_policy_reason_counts"] == {
        "query_terms_grounded_in_candidate": 1
    }
    assert any(search[2] == "semantic" for search in client.searches)
    semantic_smoke = json.loads(run.semantic_smoke_path.read_text(encoding="utf-8"))
    assert semantic_smoke["schema_version"] == "semantic-live-smoke-aggregate-v1"
    assert semantic_smoke["semantic_live_smoke"]["passed"] is False
    assert semantic_smoke["semantic_live_smoke"]["quality_claim"] == "none"
    assert any("provider-backed embedding metadata" in warning for warning in semantic_smoke["warnings"])
    assert semantic_smoke["privacy"]["raw_vectors"] == "excluded"

    rows = [
        json.loads(line)
        for line in run.query_results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["variant_id"] for row in rows} == set(suite["variant_metrics"])
    assert all("query_text" not in row for row in rows)
    assert all(row["privacy"]["answer_text"] == "redacted" for row in rows)
    assert all(row["top_candidate"]["ref"] for row in rows)
    assert all(row["answer"]["answer_type"] for row in rows)
    assert all(row["answer_grounding"]["expected_available"] is True for row in rows)
    assert all(row["answer_grounding"]["expected_citation_hit"] is True for row in rows)
    assert all(row["answer_grounding"]["citation_recall"] is not None for row in rows)
    assert all(row["config"]["index_ref"].startswith("index:") for row in rows)
    assert all("candidate_visual_support" in row for row in rows)
    assert all("verified_object_alignment" in row for row in rows)
    assert all(
        row["verified_object_alignment"]["timestamp_fallback_counted_as_verified"] is False
        for row in rows
    )
    assert all("candidate_visual_support" in row["top_candidate"] for row in rows)
    assert all("verified_object_alignment" in row["top_candidate"] for row in rows)
    rerank_row = next(row for row in rows if row["variant_id"] == "rerank")
    assert rerank_row["config"]["candidate_pool_limit"] == 30
    evidence_candidate_row = next(
        row for row in rows if row["variant_id"] == "evidence_unit_candidate"
    )
    evidence_verified_row = next(
        row for row in rows if row["variant_id"] == "evidence_unit_verified"
    )
    evidence_quality_row = next(
        row for row in rows if row["variant_id"] == "evidence_unit_quality_rerank"
    )
    assert evidence_candidate_row["status"] == "queried"
    assert evidence_candidate_row["top_candidate"]["source"] == "evidence_unit"
    assert evidence_candidate_row["verified_object_alignment"]["verified_link_count"] == 0
    assert evidence_candidate_row["candidate_visual_support"]["candidate_link_signal_counts"][
        "temporal_overlap"
    ] == 1
    assert evidence_candidate_row["candidate_visual_support"]["candidate_link_signal_counts"][
        "visual_text_overlap"
    ] == 1
    assert evidence_candidate_row["target_rank_bucket"] == "top5"
    assert evidence_verified_row["verified_object_alignment"]["verified_link_count"] == 1
    assert evidence_verified_row["candidate_visual_support"]["candidate_link_signal_counts"][
        "vlm_object_visual_description_overlap"
    ] == 1
    assert evidence_verified_row["verified_object_alignment"]["verified_link_source_counts"][
        "explicit_verified_flag"
    ] == 1
    assert evidence_verified_row["verified_object_alignment"]["verified_link_source_counts"][
        "vlm_verifier"
    ] == 1
    assert evidence_verified_row["target_rank_bucket"] == "top1"
    assert evidence_verified_row["object_evidence_coverage"]["has_vlm_entity"] is True
    assert evidence_quality_row["config"]["evidence_unit_rerank"] == "modality_aware"
    assert evidence_quality_row["modality_aware_rerank"]["enabled"] is True
    assert evidence_quality_row["modality_aware_rerank"]["strategy"] == "modality_aware"
    assert evidence_quality_row["modality_aware_rerank"]["top_changed"] is True
    assert evidence_quality_row["top_candidate"]["modality_aware_rerank"]["score_components"][
        "verified_link"
    ] > 0
    assert evidence_quality_row["top_candidate"]["object_evidence_coverage"][
        "has_verified_link"
    ] is True

    metrics_csv = run.metrics_csv_path.read_text(encoding="utf-8")
    assert "variant,segment_lexical" in metrics_csv
    assert "answer_citation_precision" in metrics_csv
    assert "verified_object_alignment_ratio" in metrics_csv
    summary = run.summary_path.read_text(encoding="utf-8")
    assert "Retrieval/Answer Matrix" in summary
    assert "candidate support" in summary
    assert "verified align" in summary
    assert "deterministic expected hint overlap" in summary

    gate_config = json.loads(
        (fixture_dir / "retrieval_quality_gate.json").read_text(encoding="utf-8")
    )
    gate_result = evaluate_retrieval_quality_gate(metrics=run.metrics, config=gate_config)
    assert gate_result["passed"] is True
    assert gate_result["privacy"]["query_content"] == "excluded"

    public_text = "\n".join(
        [
            run.metrics_path.read_text(encoding="utf-8"),
            run.metrics_csv_path.read_text(encoding="utf-8"),
            run.query_results_path.read_text(encoding="utf-8"),
            run.semantic_smoke_path.read_text(encoding="utf-8"),
            run.summary_path.read_text(encoding="utf-8"),
        ]
    )
    for sensitive in [
        "PUBLIC RAW QUERY",
        "loss curve slope should stay private-safe",
        "PUBLIC SYNTHETIC",
        "PUBLIC VISUAL LABEL",
        "seg_loss_public",
        "seg_wrap_public",
        "ent_loss_public",
        "frame_loss_public",
        "frames/frame_loss_public.jpg",
        str(fixture_dir),
    ]:
        assert sensitive not in public_text
    assert "raw_candidate_ids" in public_text
    assert "hashed" in public_text


def test_retrieval_answer_matrix_skips_evidence_unit_variant_without_index(
    tmp_path: Path,
) -> None:
    fixture_dir = Path("tests/fixtures/public_retrieval_ablation_project").resolve()
    manifest_path = tmp_path / "benchmark_matrix_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "missing_evidence_unit_index_matrix",
                "deltas": [5, 10],
                "suites": [
                    {
                        "suite_id": "public_matrix_skip",
                        "type": "retrieval_answer_matrix",
                        "domain": "public_synthetic",
                        "project_dir": str(fixture_dir),
                        "queries": str(fixture_dir / "queries.jsonl"),
                        "index": "public_segments",
                        "window_index": "public_windows",
                        "visual_index": "public_visual_entities",
                        "variants": ["evidence_unit_candidate"],
                        "include_answer": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    run = run_benchmark(
        client=FakeMatrixClient(),
        manifest_path=manifest_path,
        output_dir=tmp_path / "matrix",
        repo_root=Path.cwd(),
    )

    suite = run.metrics["suites"][0]
    variant = suite["variant_metrics"]["evidence_unit_candidate"]
    assert variant["query_status_counts"] == {"skipped": 1}
    assert variant["skip_reason_counts"] == {"missing_evidence_unit_index": 1}
    assert variant["skipped_count"] == 1
    assert variant["hit_at_10s"] == 0.0
    assert variant["verified_object_alignment_ratio"] == 0.0
    assert variant["candidate_visual_support_ratio"] == 0.0
    assert suite["skip_reason_counts"] == {"missing_evidence_unit_index": 1}

    row = json.loads(run.query_results_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["status"] == "skipped"
    assert row["skip_reason"] == "missing_evidence_unit_index"
    assert row["config"]["index_ref"] is None
    assert row["verified_object_alignment"]["verified_link_count"] == 0
    assert row["verified_object_alignment"]["timestamp_fallback_counted_as_verified"] is False
    assert row["object_evidence_coverage"]["has_vlm_entity"] is False

    public_text = "\n".join(
        [
            run.metrics_path.read_text(encoding="utf-8"),
            run.query_results_path.read_text(encoding="utf-8"),
            run.summary_path.read_text(encoding="utf-8"),
        ]
    )
    assert "PUBLIC RAW QUERY" not in public_text
    assert str(fixture_dir) not in public_text
    assert "missing_evidence_unit_index" in public_text


def test_retrieval_answer_matrix_reports_timestamp_fallback_as_candidate_not_verified(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "fallback_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_fallback",
                "project_id": "fallback_project",
                "video_id": "fallback_video",
                "sample_id": "seg_fallback",
                "sample_index": 0,
                "start_time": 10.0,
                "end_time": 14.0,
                "timestamp_center": 12.0,
                "transcript_text": "SECRET fallback transcript",
                "frame_refs": ["frame_fallback"],
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_fallback", "timestamp": 12.0, "frame_path": "/private/frame.jpg"}],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_fallback",
                "project_id": "fallback_project",
                "frame_id": "frame_fallback",
                "timestamp": 12.0,
                "text": "SECRET visual label",
                "source": "ocr",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_timestamp_fallback",
                "project_id": "fallback_project",
                "segment_id": "seg_fallback",
                "entity_id": "ent_fallback",
                "frame_id": "frame_fallback",
                "link_type": "time_overlap",
                "score": 0.2,
                "evidence": ["time_overlap", "timestamp_fallback"],
                "time_overlap": True,
            }
        ],
    )
    manifest_path = tmp_path / "benchmark_matrix_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "fallback_matrix",
                "deltas": [5, 10],
                "suites": [
                    {
                        "suite_id": "fallback_matrix",
                        "type": "retrieval_answer_matrix",
                        "domain": "public_synthetic",
                        "project_dir": str(project_dir),
                        "queries": [
                            {
                                "query_id": "q_fallback",
                                "public_label": "fallback concept",
                                "query_text": "SECRET raw fallback query",
                                "expected_time_hint": "10-14s",
                                "expected_segment_id": "seg_fallback",
                            }
                        ],
                        "index": "fallback_segments",
                        "visual_index": "fallback_visual",
                        "limit": 1,
                        "neighbor_count": 0,
                        "include_answer": False,
                        "variants": [
                            {
                                "variant_id": "segment_lexical",
                                "label": "Segment lexical baseline",
                                "index_kind": "segment",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    run = run_benchmark(
        client=TimestampFallbackMatrixClient(),
        manifest_path=manifest_path,
        output_dir=tmp_path / "matrix",
        repo_root=tmp_path,
    )

    suite = run.metrics["suites"][0]
    variant = suite["variant_metrics"]["segment_lexical"]
    assert variant["candidate_visual_support_ratio"] == 1.0
    assert variant["verified_object_alignment_ratio"] == 0.0
    assert variant["timestamp_fallback_link_ratio"] == 1.0
    assert variant["timestamp_fallback_counted_as_verified_count"] == 0
    assert variant["candidate_link_signal_counts"]["timestamp_fallback"] == 1
    assert variant["verified_link_source_counts"]["explicit_verified_flag"] == 0

    row = json.loads(run.query_results_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["candidate_visual_support"]["timestamp_fallback_link_count"] == 1
    assert row["candidate_visual_support"]["candidate_link_signal_counts"]["timestamp_fallback"] == 1
    assert row["verified_object_alignment"]["has_verified_object_alignment"] is False
    assert row["verified_object_alignment"]["verified_link_count"] == 0
    assert row["verified_object_alignment"]["timestamp_fallback_counted_as_verified"] is False
    assert row["top_candidate"]["verified_object_alignment"]["verified_link_count"] == 0

    public_text = "\n".join(
        [
            run.metrics_path.read_text(encoding="utf-8"),
            run.query_results_path.read_text(encoding="utf-8"),
            run.summary_path.read_text(encoding="utf-8"),
        ]
    )
    assert "SECRET fallback transcript" not in public_text
    assert "SECRET visual label" not in public_text
    assert "SECRET raw fallback query" not in public_text
    assert "/private/frame.jpg" not in public_text


def test_retrieval_answer_matrix_resolves_domain_lexicon_relative_to_manifest(
    tmp_path: Path,
) -> None:
    fixture_dir = Path("tests/fixtures/public_retrieval_ablation_project").resolve()
    manifest_dir = tmp_path / "matrix_manifest"
    manifest_dir.mkdir()
    (manifest_dir / "domain_lexicon.json").write_text(
        json.dumps({"aliases": {"validation loss": ["loss curve"]}}),
        encoding="utf-8",
    )
    manifest_path = manifest_dir / "benchmark_matrix_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "manifest_relative_matrix_lexicon",
                "deltas": [5, 10, 15],
                "suites": [
                    {
                        "suite_id": "public_matrix_manifest_lexicon",
                        "type": "retrieval_answer_matrix",
                        "domain": "public_synthetic",
                        "project_dir": str(fixture_dir),
                        "queries": str(fixture_dir / "queries.jsonl"),
                        "index": "public_segments",
                        "window_index": "public_windows",
                        "visual_index": "public_visual_entities",
                        "domain_lexicon": "domain_lexicon.json",
                        "limit": 3,
                        "neighbor_count": 0,
                        "include_answer": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    client = FakeMatrixClient()

    run = run_benchmark(
        client=client,
        manifest_path=manifest_path,
        output_dir=tmp_path / "matrix",
        repo_root=Path.cwd(),
    )

    rows = [
        json.loads(line)
        for line in run.query_results_path.read_text(encoding="utf-8").splitlines()
    ]
    domain_row = next(row for row in rows if row["variant_id"] == "domain_lexicon")
    assert domain_row["query_expansion"]["enabled"] is True
    assert domain_row["query_expansion"]["applied"] is True
    assert any(search[1] == "validation loss" for search in client.searches)


def test_retrieval_answer_matrix_generates_sanitized_local_hash_query_vectors(
    tmp_path: Path,
) -> None:
    fixture_dir = Path("tests/fixtures/public_retrieval_ablation_project").resolve()
    manifest_path = tmp_path / "benchmark_matrix_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "local_hash_matrix_vector_smoke",
                "suites": [
                    {
                        "suite_id": "public_matrix_local_hash",
                        "type": "retrieval_answer_matrix",
                        "domain": "public_synthetic",
                        "project_dir": str(fixture_dir),
                        "queries": str(fixture_dir / "queries.jsonl"),
                        "index": "public_segments",
                        "window_index": "public_windows",
                        "visual_index": "public_visual_entities",
                        "variants": ["hybrid", "window_hybrid"],
                        "limit": 3,
                        "neighbor_count": 0,
                        "include_answer": False,
                        "hybrid_query_vector_dimensions": 3,
                        "hybrid_query_vector_embedder": "default",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    client = FakeMatrixClient()

    run = run_benchmark(
        client=client,
        manifest_path=manifest_path,
        output_dir=tmp_path / "matrix",
        repo_root=Path.cwd(),
    )

    rows = [
        json.loads(line)
        for line in run.query_results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["variant_id"] for row in rows} == {"hybrid", "window_hybrid"}
    assert any(search[3] is not None and len(search[3]) == 3 for search in client.searches)
    for row in rows:
        query_vector = row["config"]["query_vector"]
        assert query_vector == {
            "configured": True,
            "source": "local_hash_v1",
            "embedder": "default",
            "dimensions": 3,
            "name_present": False,
            "manifest_ref": None,
            "provider": None,
            "source_model": None,
            "quality_claim": "none",
            "backend_contract": None,
            "purpose": "local_reproducibility_smoke_fallback",
        }
    semantic_smoke = json.loads(run.semantic_smoke_path.read_text(encoding="utf-8"))
    assert semantic_smoke["semantic_live_smoke"]["passed"] is False
    assert semantic_smoke["counts"]["local_hash_query_vector_count"] == 2
    assert any("local_hash_v1 vectors" in warning for warning in semantic_smoke["warnings"])
    public_text = "\n".join(
        [
            run.query_results_path.read_text(encoding="utf-8"),
            run.semantic_smoke_path.read_text(encoding="utf-8"),
        ]
    )
    assert '"vector": [' not in public_text
    assert "PUBLIC RAW QUERY" not in public_text


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
