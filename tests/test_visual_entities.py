from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from oarag.schemas import VisualEntity
from oarag.visual_entities import (
    FrameRecord,
    TesseractVisualEntityExtractor,
    extract_visual_entities,
    filter_visual_entities,
)


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
    assert summary["counts"]["frames_total"] == 2
    assert summary["counts"]["raw_visual_entities"] == 0
    assert summary["counts"]["visual_entities"] == 0
    assert summary["counts"]["dropped_visual_entities"] == 0

    manifest = json.loads(project_manifest.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["visual_entities"] == str(output_path)
    assert manifest["counts"]["visual_entities"] == 0
    assert manifest["visual_entity_extraction"]["backend"] == "stub"
    assert manifest["visual_entity_extraction"]["raw_visual_entities"] == 0


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


def test_tesseract_language_option_is_placed_before_tsv_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        captured.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
                "left\ttop\twidth\theight\tconf\ttext\n"
                "5\t1\t1\t1\t1\t1\t10\t20\t30\t40\t90.0\tSTACK\n"
            ),
        )

    monkeypatch.setattr("oarag.visual_entities.shutil.which", lambda _: "/usr/bin/tesseract")
    monkeypatch.setattr("oarag.visual_entities._run", fake_run)

    extractor = TesseractVisualEntityExtractor(language="eng")
    entities = extractor.extract(
        project_id="sample_project",
        frame=FrameRecord(frame_id="frame_000001", frame_path="/tmp/frame.jpg", timestamp=1.0),
    )

    assert captured == [["tesseract", "/tmp/frame.jpg", "stdout", "-l", "eng", "tsv"]]
    assert [entity.text for entity in entities] == ["STACK"]


def test_filter_visual_entities_drops_noise_and_duplicates() -> None:
    base = {
        "project_id": "sample_project",
        "frame_id": "frame_000001",
        "timestamp": 1.0,
        "frame_path": "/tmp/frame.jpg",
        "bbox": {"left": 10.0, "top": 20.0, "width": 30.0, "height": 40.0},
        "entity_type": "ocr_text",
        "source": "ocr:tesseract",
    }
    entities = [
        VisualEntity(entity_id="keep", text="STACK", confidence=0.91, **base),
        VisualEntity(entity_id="low_conf", text="CALL", confidence=0.32, **base),
        VisualEntity(entity_id="punct", text="---", confidence=0.99, **base),
        VisualEntity(entity_id="short", text="x", confidence=0.70, **base),
        VisualEntity(entity_id="short_high_conf", text="A", confidence=0.96, **base),
        VisualEntity(entity_id="duplicate", text="STACK", confidence=0.92, **base),
    ]

    filtered, summary = filter_visual_entities(entities)

    assert [entity.entity_id for entity in filtered] == ["keep", "short_high_conf"]
    assert summary["raw_visual_entities"] == 6
    assert summary["visual_entities"] == 2
    assert summary["dropped_visual_entities"] == 4
    assert summary["filter_reasons"] == {
        "empty_text": 0,
        "low_confidence": 1,
        "no_alnum": 1,
        "short_low_confidence": 1,
        "duplicate": 1,
    }


def test_extract_visual_entities_records_filter_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "filtered_project"
    frame_path = project_dir / "frames" / "frame_000001.jpg"
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000001", "frame_path": str(frame_path), "timestamp": 0.0}],
    )

    class NoisyExtractor:
        backend = "noisy-test"

        def extract(self, *, project_id: str, frame) -> list[VisualEntity]:
            return [
                VisualEntity(
                    entity_id="ent_good",
                    project_id=project_id,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    frame_path=frame.frame_path,
                    bbox=None,
                    text="Matrix",
                    entity_type="ocr_text",
                    confidence=0.88,
                    source="test",
                ),
                VisualEntity(
                    entity_id="ent_noise",
                    project_id=project_id,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    frame_path=frame.frame_path,
                    bbox=None,
                    text=".",
                    entity_type="ocr_text",
                    confidence=0.90,
                    source="test",
                ),
            ]

    monkeypatch.setattr("oarag.visual_entities.make_extractor", lambda **_: NoisyExtractor())

    summary = extract_visual_entities(project_dir=project_dir, backend="stub")

    assert summary["counts"]["raw_visual_entities"] == 2
    assert summary["counts"]["visual_entities"] == 1
    assert summary["counts"]["dropped_visual_entities"] == 1
    assert summary["counts"]["filter_reasons"]["no_alnum"] == 1

    manifest = json.loads(
        (project_dir / "manifests" / "project_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["visual_entity_extraction"]["raw_visual_entities"] == 2
    assert manifest["visual_entity_extraction"]["dropped_visual_entities"] == 1


def test_extract_visual_entities_vlm_jsonl_loads_structured_parser_output(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "vlm_project"
    frame_path = project_dir / "frames" / "frame_000001.jpg"
    frames_manifest = project_dir / "manifests" / "frames_manifest.jsonl"
    vlm_jsonl = project_dir / "manifests" / "vlm_parser_output.jsonl"
    _write_jsonl(
        frames_manifest,
        [{"frame_id": "frame_000001", "frame_path": str(frame_path), "timestamp": 3.5}],
    )
    _write_jsonl(
        vlm_jsonl,
        [
            {
                "frame_id": "frame_000001",
                "parser_version": "vlm-jsonl-v1",
                "source_model": "stub-vlm",
                "entities": [
                    {
                        "visual_description": "A blue matrix diagram with highlighted row",
                        "entity_type": "diagram",
                        "confidence": 0.93,
                        "position": {"region": "center", "x": 0.5, "y": 0.45},
                        "relations": [{"type": "points_to", "target": "row_label"}],
                    }
                ],
            }
        ],
    )

    summary = extract_visual_entities(
        project_dir=project_dir,
        backend="vlm-jsonl",
        vlm_jsonl_path=vlm_jsonl,
    )

    output_path = project_dir / "manifests" / "visual_entities.jsonl"
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["backend"] == "vlm-jsonl"
    assert summary["counts"]["visual_entities"] == 1
    assert rows == [
        {
            "entity_id": "ent_frame_000001_0001",
            "project_id": "vlm_project",
            "frame_id": "frame_000001",
            "timestamp": 3.5,
            "frame_path": str(frame_path),
            "bbox": None,
            "text": "A blue matrix diagram with highlighted row",
            "entity_type": "diagram",
            "confidence": 0.93,
            "source": "vlm:stub-vlm",
            "visual_description": "A blue matrix diagram with highlighted row",
            "position": {"region": "center", "x": 0.5, "y": 0.45},
            "relations": [{"type": "points_to", "target": "row_label"}],
            "parser_version": "vlm-jsonl-v1",
            "source_model": "stub-vlm",
        }
    ]

    manifest = json.loads(
        (project_dir / "manifests" / "project_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["visual_entity_extraction"]["backend"] == "vlm-jsonl"


def test_vlm_jsonl_backend_requires_valid_frame_scoped_entities(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "bad_vlm_project"
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
    _write_jsonl(
        project_dir / "manifests" / "vlm_parser_output.jsonl",
        [
            {
                "frame_id": "frame_000001",
                "entities": [{"frame_id": "frame_000002", "visual_description": "bad"}],
            }
        ],
    )

    with pytest.raises(ValueError, match="Entity frame_id mismatch"):
        extract_visual_entities(
            project_dir=project_dir,
            backend="vlm-jsonl",
            vlm_jsonl_path=project_dir / "manifests" / "vlm_parser_output.jsonl",
        )


def test_vlm_jsonl_backend_rejects_unknown_frame_id(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "unknown_vlm_project"
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
    _write_jsonl(
        project_dir / "manifests" / "vlm_parser_output.jsonl",
        [
            {
                "frame_id": "frame_000999",
                "entities": [{"visual_description": "unknown frame"}],
            }
        ],
    )

    with pytest.raises(ValueError, match="not present in frames manifest"):
        extract_visual_entities(
            project_dir=project_dir,
            backend="vlm-jsonl",
            vlm_jsonl_path=project_dir / "manifests" / "vlm_parser_output.jsonl",
        )


def test_filter_visual_entities_keeps_vlm_description_without_text() -> None:
    entity = VisualEntity(
        entity_id="ent_vlm",
        project_id="sample_project",
        frame_id="frame_000001",
        timestamp=1.0,
        frame_path="/tmp/frame.jpg",
        bbox=None,
        text="",
        entity_type="diagram",
        confidence=0.91,
        source="vlm:stub-vlm",
        visual_description="A labeled matrix diagram",
        position={"region": "center"},
        relations=[],
        parser_version="vlm-jsonl-v1",
        source_model="stub-vlm",
    )

    filtered, summary = filter_visual_entities([entity])

    assert filtered == [entity]
    assert summary["visual_entities"] == 1
    assert summary["dropped_visual_entities"] == 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
