from __future__ import annotations

import json
from pathlib import Path

import pytest

from oarag.lecture_smoke import run_lecture_smoke


class FakeClient:
    def search(self, index_uid: str, query: str, limit: int = 10) -> dict:
        if "PRIVATE RAW QUERY TEXT" in query:
            hits = [
                {
                    "segment_id": "seg_private_hit",
                    "sample_id": "seg_private_hit",
                    "video_id": "private_video_slug",
                    "start_time": 10.0,
                    "end_time": 14.0,
                    "timestamp_center": 12.0,
                    "transcript_text": "SECRET transcript excerpt should not be public",
                    "semantic_source_fields": ["transcript_text", "visual_entities.text"],
                    "_rankingScore": 0.99,
                }
            ]
        else:
            hits = [
                {
                    "segment_id": "seg_fallback",
                    "sample_id": "seg_fallback",
                    "video_id": "private_video_slug",
                    "start_time": 40.0,
                    "end_time": 44.0,
                    "timestamp_center": 42.0,
                    "transcript_text": "SECOND SECRET transcript should not be public",
                    "semantic_source_fields": ["transcript_text"],
                    "_rankingScore": 0.5,
                }
            ]
        return {"hits": hits[:limit], "processingTimeMs": 4, "indexUid": index_uid}


def test_lecture_smoke_writes_sanitized_public_outputs(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    private_frame_path = "/Users/example/private_lecture/frame_secret.jpg"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_private_hit",
                "project_id": "private_project",
                "video_id": "private_video_slug",
                "sample_id": "seg_private_hit",
                "sample_index": 1,
                "start_time": 10.0,
                "end_time": 14.0,
                "timestamp_center": 12.0,
                "transcript_text": "SECRET transcript excerpt should not be public",
                "frame_refs": ["frame_secret"],
                "semantic_text": "SECRET semantic text should not be public",
                "semantic_source_fields": ["transcript_text", "visual_entities.text"],
            },
            {
                "segment_id": "seg_fallback",
                "project_id": "private_project",
                "video_id": "private_video_slug",
                "sample_id": "seg_fallback",
                "sample_index": 2,
                "start_time": 40.0,
                "end_time": 44.0,
                "timestamp_center": 42.0,
                "transcript_text": "SECOND SECRET transcript should not be public",
                "frame_refs": [],
                "semantic_source_fields": ["transcript_text"],
            },
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_secret", "timestamp": 12.0, "frame_path": private_frame_path}],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "entity_secret",
                "project_id": "private_project",
                "frame_id": "frame_secret",
                "timestamp": 12.0,
                "frame_path": private_frame_path,
                "bbox": None,
                "text": "SECRET VISUAL LABEL",
                "entity_type": "ocr_text",
                "confidence": 0.9,
                "source": "fixture",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_secret",
                "project_id": "private_project",
                "segment_id": "seg_private_hit",
                "entity_id": "entity_secret",
                "frame_id": "frame_secret",
                "link_type": "time_overlap+lexical_match",
                "score": 0.8,
                "evidence": ["time_overlap", "lexical_match"],
                "time_overlap": True,
                "lexical_match": ["secret"],
                "mention_candidate": [],
                "score_breakdown": {"time_overlap": 0.2, "visual_text_match": 0.6},
                "reason_metadata": {"summary": "fixture_secret_match"},
            }
        ],
    )

    manifest_path = tmp_path / "lecture_smoke.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "private_safe_test",
                "suites": [
                    {
                        "suite_id": "safe_suite",
                        "type": "lecture_project",
                        "domain": "private_fixture",
                        "project_dir": str(project_dir),
                        "index": "segments",
                        "limit": 2,
                        "neighbor_count": 0,
                        "queries": [
                            {
                                "query_id": "q_private",
                                "query_text": "PRIVATE RAW QUERY TEXT should not be public",
                            },
                            {
                                "query_id": "q_fallback",
                                "query_text": "fallback query should not be public",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    run = run_lecture_smoke(
        client=FakeClient(),
        manifest_path=manifest_path,
        output_dir=tmp_path / "public",
        repo_root=tmp_path,
    )

    assert run.metrics_path.name == "metrics.json"
    assert run.query_results_path.name == "query_results.jsonl"
    assert run.summary_path.name == "summary.md"
    assert run.metrics_path.exists()
    assert run.query_results_path.exists()
    assert run.summary_path.exists()

    rows = [
        json.loads(line)
        for line in run.query_results_path.read_text(encoding="utf-8").splitlines()
    ]
    first = rows[0]
    assert first["search_hit_count"] == 1
    assert first["top_candidate_source"] == "segment"
    assert first["top_candidate_timestamp_available"] is True
    assert first["frame_backed_evidence"] is True
    assert first["linked_entity_count"] == 1
    assert first["semantic_source_fields_available"] is True
    assert first["semantic_source_fields_count"] == 2
    assert first["transcript_only_fallback"] is False
    assert rows[1]["transcript_only_fallback"] is True

    public_text = "\n".join(
        [
            run.metrics_path.read_text(encoding="utf-8"),
            run.query_results_path.read_text(encoding="utf-8"),
            run.summary_path.read_text(encoding="utf-8"),
        ]
    )
    for sensitive in [
        "PRIVATE RAW QUERY TEXT",
        "fallback query should not be public",
        "SECRET transcript excerpt",
        "SECOND SECRET transcript",
        "SECRET semantic text",
        "SECRET VISUAL LABEL",
        private_frame_path,
        str(project_dir),
        "seg_private_hit",
        "private_video_slug",
    ]:
        assert sensitive not in public_text
    assert "raw_query_text" in public_text
    assert "redacted" in public_text


def test_private_output_requires_explicit_allow_flag(tmp_path: Path) -> None:
    manifest_path = tmp_path / "lecture_smoke.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "private_blocked",
                "suites": [
                    {
                        "suite_id": "safe_suite",
                        "project_dir": str(tmp_path / "missing_project"),
                        "index": "segments",
                        "queries": [
                            {"query_id": "q1", "query_text": "private query"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Private lecture-smoke output requires"):
        run_lecture_smoke(
            client=FakeClient(),
            manifest_path=manifest_path,
            output_dir=tmp_path / "public",
            repo_root=tmp_path,
            private_output_dir=tmp_path / "private",
        )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
