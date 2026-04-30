from __future__ import annotations

import json
from pathlib import Path

from oarag.entity_links import link_entities


def test_link_entities_writes_timestamp_only_links_and_updates_manifest(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_1",
                "project_id": "sample_project",
                "video_id": "video",
                "start_time": 0.0,
                "end_time": 5.0,
                "timestamp_center": 2.5,
                "transcript_text": "introduction only",
                "mention_candidates": [],
                "frame_refs": ["frame_000001"],
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_1",
                "project_id": "sample_project",
                "frame_id": "frame_000001",
                "timestamp": 1.0,
                "frame_path": "/tmp/frame_000001.jpg",
                "bbox": None,
                "text": "42",
                "entity_type": "ocr_text",
                "confidence": 0.9,
                "source": "ocr:tesseract",
            }
        ],
    )
    project_manifest = project_dir / "manifests" / "project_manifest.json"
    project_manifest.parent.mkdir(parents=True, exist_ok=True)
    project_manifest.write_text(
        json.dumps({"project_id": "sample_project", "artifacts": {}, "counts": {}}),
        encoding="utf-8",
    )

    summary = link_entities(project_dir=project_dir)

    output_path = project_dir / "manifests" / "entity_links.jsonl"
    rows = _read_jsonl(output_path)
    assert summary["counts"]["entity_links"] == 1
    assert summary["counts"]["timestamp_only_links"] == 1
    assert summary["counts"]["evidence_type_counts"] == {
        "time_overlap": 1,
        "lexical_match": 0,
        "mention_candidate": 0,
        "timestamp_only": 1,
    }
    assert rows[0]["link_type"] == "time_overlap"
    assert rows[0]["lexical_match"] == []
    assert rows[0]["mention_candidate"] == []

    manifest = json.loads(project_manifest.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["entity_links"] == str(output_path)
    assert manifest["counts"]["entity_links"] == 1
    assert manifest["entity_linking"]["domain_lexicon"] == {
        "enabled": False,
        "source_path": None,
        "canonical_term_count": 0,
        "alias_count": 0,
        "term_count": 0,
    }


def test_link_entities_is_idempotent_for_repeated_runs(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_1",
                "project_id": "sample_project",
                "video_id": "video",
                "start_time": 10.0,
                "end_time": 12.0,
                "timestamp_center": 11.0,
                "transcript_text": "stack size on the board",
                "mention_candidates": ["stack", "board"],
                "frame_refs": ["frame_000010"],
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_1",
                "project_id": "sample_project",
                "frame_id": "frame_000010",
                "timestamp": 11.0,
                "frame_path": "/tmp/frame_000010.jpg",
                "bbox": None,
                "text": "STACK SIZE ON BOARD",
                "entity_type": "ocr_text",
                "confidence": 0.9,
                "source": "ocr:tesseract",
            }
        ],
    )

    first_summary = link_entities(project_dir=project_dir)
    first_output = (project_dir / "manifests" / "entity_links.jsonl").read_text(encoding="utf-8")
    second_summary = link_entities(project_dir=project_dir)
    second_output = (project_dir / "manifests" / "entity_links.jsonl").read_text(encoding="utf-8")

    assert first_summary["counts"] == second_summary["counts"]
    assert first_output == second_output


def test_link_entities_adds_lexical_and_mention_evidence(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_1",
                "project_id": "sample_project",
                "video_id": "video",
                "start_time": 10.0,
                "end_time": 12.0,
                "timestamp_center": 11.0,
                "transcript_text": "the stack size is visible on the board",
                "mention_candidates": ["stack", "board"],
                "frame_refs": ["frame_000010"],
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_1",
                "project_id": "sample_project",
                "frame_id": "frame_000010",
                "timestamp": 11.0,
                "frame_path": "/tmp/frame_000010.jpg",
                "bbox": None,
                "text": "STACK SIZE ON BOARD",
                "entity_type": "ocr_text",
                "confidence": 0.9,
                "source": "ocr:tesseract",
            }
        ],
    )

    summary = link_entities(project_dir=project_dir)

    rows = _read_jsonl(project_dir / "manifests" / "entity_links.jsonl")
    assert summary["counts"]["entity_links"] == 1
    assert summary["counts"]["lexical_links"] == 1
    assert summary["counts"]["mention_links"] == 1
    assert rows[0]["link_type"] == "time_overlap+lexical_match+mention_candidate"
    assert rows[0]["lexical_match"] == ["board", "size", "stack"]
    assert rows[0]["mention_candidate"] == ["board", "stack"]
    assert rows[0]["score"] == 1.2


def test_link_entities_keeps_domain_aliases_off_without_project_lexicon(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_1",
                "project_id": "sample_project",
                "video_id": "video",
                "start_time": 20.0,
                "end_time": 22.0,
                "timestamp_center": 21.0,
                "transcript_text": "여기 보이는 벳 사이즈는 보드에 따라 달라집니다",
                "mention_candidates": ["여기", "벳", "사이즈", "보드"],
                "frame_refs": ["frame_000020"],
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_1",
                "project_id": "sample_project",
                "frame_id": "frame_000020",
                "timestamp": 21.0,
                "frame_path": "/tmp/frame_000020.jpg",
                "bbox": None,
                "text": "BET SIZE BOARD",
                "entity_type": "ocr_text",
                "confidence": 0.9,
                "source": "ocr:tesseract",
            }
        ],
    )

    summary = link_entities(project_dir=project_dir)

    rows = _read_jsonl(project_dir / "manifests" / "entity_links.jsonl")
    assert summary["counts"]["entity_links"] == 1
    assert summary["domain_lexicon"]["enabled"] is False
    assert summary["counts"]["evidence_type_counts"]["lexical_match"] == 0
    assert summary["counts"]["evidence_type_counts"]["mention_candidate"] == 0
    assert rows[0]["link_type"] == "time_overlap"
    assert rows[0]["lexical_match"] == []
    assert rows[0]["mention_candidate"] == []


def test_link_entities_uses_project_domain_lexicon_for_aliases(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_1",
                "project_id": "sample_project",
                "video_id": "video",
                "start_time": 20.0,
                "end_time": 22.0,
                "timestamp_center": 21.0,
                "transcript_text": "여기 보이는 벳 사이즈는 보드에 따라 달라집니다",
                "mention_candidates": ["여기", "벳", "사이즈", "보드"],
                "frame_refs": ["frame_000020"],
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_1",
                "project_id": "sample_project",
                "frame_id": "frame_000020",
                "timestamp": 21.0,
                "frame_path": "/tmp/frame_000020.jpg",
                "bbox": None,
                "text": "BET SIZE BOARD",
                "entity_type": "ocr_text",
                "confidence": 0.9,
                "source": "ocr:tesseract",
            }
        ],
    )
    (project_dir / "domain_lexicon.json").write_text(
        json.dumps(
            {
                "aliases": {
                    "bet": ["벳"],
                    "size": ["사이즈"],
                    "board": ["보드"],
                }
            }
        ),
        encoding="utf-8",
    )

    summary = link_entities(project_dir=project_dir)

    rows = _read_jsonl(project_dir / "manifests" / "entity_links.jsonl")
    assert summary["domain_lexicon"]["enabled"] is True
    assert summary["domain_lexicon"]["canonical_term_count"] == 3
    assert summary["counts"]["entity_links"] == 1
    assert summary["counts"]["evidence_type_counts"]["lexical_match"] == 1
    assert summary["counts"]["evidence_type_counts"]["mention_candidate"] == 1
    assert rows[0]["lexical_match"] == ["bet", "board", "size"]
    assert rows[0]["mention_candidate"] == ["bet", "board", "size"]


def test_link_entities_skips_non_overlapping_entities(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_1",
                "project_id": "sample_project",
                "video_id": "video",
                "start_time": 0.0,
                "end_time": 2.0,
                "timestamp_center": 1.0,
                "transcript_text": "stack size",
                "mention_candidates": ["stack", "bet size"],
                "frame_refs": ["frame_000001"],
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_far",
                "project_id": "sample_project",
                "frame_id": "frame_000099",
                "timestamp": 99.0,
                "frame_path": "/tmp/frame_000099.jpg",
                "bbox": None,
                "text": "STACK SIZE",
                "entity_type": "ocr_text",
                "confidence": 0.9,
                "source": "ocr:tesseract",
            }
        ],
    )

    summary = link_entities(project_dir=project_dir)

    assert summary["counts"]["entity_links"] == 0
    assert (project_dir / "manifests" / "entity_links.jsonl").read_text(encoding="utf-8") == ""


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
