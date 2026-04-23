from __future__ import annotations

import json
from pathlib import Path

import pytest

from oarag.schemas import VisualEntity
from oarag.visual_entities import extract_visual_entities


def test_extract_visual_entities_stub_writes_empty_jsonl_and_updates_manifest(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    frames_manifest = project_dir / "manifests" / "frames_manifest.jsonl"
    project_manifest = project_dir / "manifests" / "project_manifest.json"
    _write_jsonl(
        frames_manifest,
        [
            {
                "frame_id": "frame_000001",
                "frame_path": str(project_dir / "frames" / "frame_000001.jpg"),
                "timestamp": 0.0,
            },
            {
                "frame_id": "frame_000002",
                "frame_path": str(project_dir / "frames" / "frame_000002.jpg"),
                "timestamp": 1.0,
            },
        ],
    )
    project_manifest.parent.mkdir(parents=True, exist_ok=True)
    project_manifest.write_text(
        json.dumps({"project_id": "sample_project", "artifacts": {}, "counts": {}}),
        encoding="utf-8",
    )

    summary = extract_visual_entities(project_dir=project_dir, backend="stub")

    output_path = project_dir / "manifests" / "visual_entities.jsonl"
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == ""
    assert summary["backend"] == "stub"
    assert summary["counts"] == {"frames_total": 2, "visual_entities": 0}

    manifest = json.loads(project_manifest.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["visual_entities"] == str(output_path)
    assert manifest["counts"]["visual_entities"] == 0
    assert manifest["visual_entity_extraction"]["backend"] == "stub"


def test_extract_visual_entities_rejects_entity_with_mismatched_frame_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "bad_project"
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {
                "frame_id": "frame_000001",
                "frame_path": str(project_dir / "frames" / "frame_000001.jpg"),
                "timestamp": 0.0,
            }
        ],
    )

    class BadExtractor:
        backend = "bad-backend"

        def extract(self, *, project_id: str, frame) -> list[VisualEntity]:
            return [
                VisualEntity(
                    entity_id="ent_bad_1",
                    project_id=project_id,
                    frame_id="frame_other",
                    timestamp=frame.timestamp,
                    frame_path=frame.frame_path,
                    bbox=None,
                    text="bad",
                    entity_type="ocr_text",
                    confidence=None,
                    source="test",
                )
            ]

    monkeypatch.setattr("oarag.visual_entities.make_extractor", lambda **_: BadExtractor())

    with pytest.raises(ValueError, match="frame_id mismatch"):
        extract_visual_entities(project_dir=project_dir, backend="stub")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
