from __future__ import annotations

import json
from pathlib import Path

from oarag.benchmark import parse_time_hint, run_benchmark


class FakeClient:
    def search(self, index_uid: str, query: str, limit: int = 10) -> dict:
        if "gravity" in query.lower():
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


def test_parse_time_hint_extracts_multiple_ranges() -> None:
    assert parse_time_hint("95-102s or 126-131s") == [(95.0, 102.0), (126.0, 131.0)]


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
                    },
                    {
                        "suite_id": "local_suite",
                        "type": "local_project",
                        "domain": "local_pilot",
                        "project_dir": str(project_dir),
                        "queries": str(queries_path),
                        "index": "local_index",
                        "domain_lexicon": "domain_lexicon.json",
                        "limit": 3,
                        "rerank": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    run = run_benchmark(client=FakeClient(), manifest_path=manifest_path, repo_root=tmp_path)

    assert run.metrics_path.exists()
    assert run.query_results_path.exists()
    assert run.summary_path.exists()
    assert run.metrics["suite_count"] == 2
    assert run.metrics["anti_overfit"]["domain_count"] == 2
    local_metrics = next(
        suite for suite in run.metrics["suites"] if suite["suite_id"] == "local_suite"
    )
    assert local_metrics["frame_backed_ratio"] == 1.0
    assert local_metrics["linked_entity_backed_ratio"] == 1.0
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
    assert local_row["top_candidate"]["segment_id"] == "seg_local_1"
    assert local_row["top_rerank"]["original_rank"] == 2
    summary = run.summary_path.read_text(encoding="utf-8")
    assert "| local_suite | local_project | local_pilot | on (domain_lexicon.json) |" in summary
    assert "on (deterministic_evidence_v1)" in summary
    assert "Anti-Overfit View" in summary


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
