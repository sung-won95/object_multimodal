from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from oarag.schemas import VisualEntity
from oarag.visual_entities import (
    FrameRecord,
    TesseractVisualEntityExtractor,
    extract_visual_entities,
    filter_visual_entities,
)
from oarag.vlm import run_vlm


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


def test_extract_visual_entities_vlm_first_prefers_observations_over_ocr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "vlm_first_project"
    frame_path = project_dir / "frames" / "frame_000001.jpg"
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000001", "frame_path": str(frame_path), "timestamp": 1.0}],
    )
    _write_jsonl(
        project_dir / "manifests" / "vlm_visual_observations.jsonl",
        [
            {
                "observation_id": "obs_vlm_first",
                "project_id": "vlm_first_project",
                "video_id": "lecture_01",
                "frame_id": "frame_000001",
                "timestamp": 1.0,
                "segment_id": None,
                "backend": "jsonl",
                "source_model": "fixture-vlm",
                "model_version": None,
                "confidence": 0.9,
                "status": "success",
                "observation_type": "diagram",
                "visual_description": "A VLM-first diagram",
                "metadata": {"parser_version": "fixture-parser-v1"},
            }
        ],
    )
    monkeypatch.setattr("oarag.visual_entities.shutil.which", lambda _: "/usr/bin/tesseract")

    summary = extract_visual_entities(project_dir=project_dir)
    rows = _read_jsonl(project_dir / "manifests" / "visual_entities.jsonl")

    assert summary["requested_backend"] == "vlm-first"
    assert summary["backend"] == "vlm-observations"
    assert summary["backend_role"] == "vlm_parser"
    assert summary["fallback_policy"]["ocr_role"] == "baseline_or_fallback_only"
    assert rows[0]["parser_version"] == "fixture-parser-v1"
    assert rows[0]["source"] == "vlm:fixture-vlm"


def test_extract_visual_entities_vlm_first_falls_back_to_ocr_when_no_vlm_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "ocr_fallback_project"
    frame_path = project_dir / "frames" / "frame_000001.jpg"
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000001", "frame_path": str(frame_path), "timestamp": 2.0}],
    )

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
                "left\ttop\twidth\theight\tconf\ttext\n"
                "5\t1\t1\t1\t1\t1\t10\t20\t30\t40\t91.0\tOCR\n"
            ),
        )

    monkeypatch.setattr("oarag.visual_entities.shutil.which", lambda _: "/usr/bin/tesseract")
    monkeypatch.setattr("oarag.visual_entities._run", fake_run)

    summary = extract_visual_entities(project_dir=project_dir)
    rows = _read_jsonl(project_dir / "manifests" / "visual_entities.jsonl")

    assert summary["requested_backend"] == "vlm-first"
    assert summary["backend"] == "local-ocr"
    assert summary["backend_role"] == "ocr_baseline_fallback"
    assert rows[0]["text"] == "OCR"
    assert rows[0]["source"] == "ocr:tesseract"


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


def test_extract_visual_entities_vlm_observations_maps_success_observations(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "vlm_observation_project"
    frame_path = project_dir / "frames" / "frame_000001.jpg"
    observations_path = project_dir / "manifests" / "vlm_visual_observations.jsonl"
    project_manifest = project_dir / "manifests" / "project_manifest.json"
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000001", "frame_path": str(frame_path), "timestamp": 7.25}],
    )
    project_manifest.parent.mkdir(parents=True, exist_ok=True)
    project_manifest.write_text(
        json.dumps({"project_id": "vlm_observation_project", "artifacts": {}, "counts": {}}),
        encoding="utf-8",
    )
    _write_jsonl(
        observations_path,
        [
            {
                "observation_id": "obs_diagram_1",
                "project_id": "vlm_observation_project",
                "video_id": "lecture_01",
                "frame_id": "frame_000001",
                "timestamp": 7.25,
                "segment_id": "seg_0001",
                "backend": "deterministic",
                "source_model": "offline-vlm",
                "model_version": "test-v1",
                "confidence": 0.92,
                "status": "success",
                "observation_type": "diagram",
                "visual_description": "A labeled covariance matrix diagram",
                "detected_text": "Covariance Matrix",
                "bbox": {"left": 12, "top": 34, "width": 56, "height": 78},
                "position": {"region": "center", "x": 0.5, "y": 0.4},
                "relations": [{"type": "contains", "target": "matrix_label"}],
                "metadata": {"parser_version": "observation-parser-v2"},
            },
            {
                "observation_id": "obs_failure",
                "project_id": "vlm_observation_project",
                "video_id": "lecture_01",
                "frame_id": "frame_000001",
                "timestamp": 7.25,
                "segment_id": "seg_0001",
                "backend": "deterministic",
                "source_model": "offline-vlm",
                "model_version": "test-v1",
                "confidence": 1.0,
                "status": "backend_failure",
                "observation_type": "frame_error",
                "visual_description": "This failure should not become an entity",
            },
        ],
    )

    summary = extract_visual_entities(
        project_dir=project_dir,
        backend="vlm-observations",
    )

    output_path = project_dir / "manifests" / "visual_entities.jsonl"
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["backend"] == "vlm-observations"
    assert summary["paths"]["vlm_visual_observations"] == str(observations_path)
    assert summary["counts"]["raw_visual_entities"] == 1
    assert summary["counts"]["visual_entities"] == 1
    assert rows == [
        {
            "entity_id": "ent_obs_diagram_1",
            "project_id": "vlm_observation_project",
            "frame_id": "frame_000001",
            "timestamp": 7.25,
            "frame_path": str(frame_path),
            "bbox": {"left": 12.0, "top": 34.0, "width": 56.0, "height": 78.0},
            "text": "Covariance Matrix",
            "entity_type": "diagram",
            "confidence": 0.92,
            "source": "vlm:offline-vlm",
            "visual_description": "A labeled covariance matrix diagram",
            "detected_text": "Covariance Matrix",
            "position": {"region": "center", "x": 0.5, "y": 0.4},
            "relations": [{"type": "contains", "target": "matrix_label"}],
            "parser_version": "observation-parser-v2",
            "source_model": "offline-vlm",
        }
    ]

    manifest = json.loads(project_manifest.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["visual_entities"] == str(output_path)
    assert manifest["counts"]["visual_entities"] == 1
    assert manifest["visual_entity_extraction"]["backend"] == "vlm-observations"


def test_run_vlm_to_extract_visual_entities_vlm_observations_smoke(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "vlm_smoke_project"
    frame_path = project_dir / "frames" / "frame_000001.jpg"
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000001", "frame_path": str(frame_path), "timestamp": 2.0}],
    )

    command_code = (
        "import json,sys;"
        "req=json.load(sys.stdin);"
        "f=req['frame'];"
        "print(json.dumps({'frame_id':f['frame_id'],"
        "'observation_type':'equation','visual_description':'A displayed equation',"
        "'detected_text':'E = mc^2','confidence':0.88,"
        "'bbox':{'left':1,'top':2,'width':3,'height':4}}))"
    )

    vlm_summary = run_vlm(
        project_dir=project_dir,
        backend="command",
        model="fixture-vlm",
        options={
            "command": [sys.executable, "-c", command_code],
            "input_mode": "json-stdin",
            "prompt_template_version": "smoke-template-v1",
        },
    )
    entity_summary = extract_visual_entities(
        project_dir=project_dir,
        backend="vlm-observations",
    )

    rows = [
        json.loads(line)
        for line in (project_dir / "manifests" / "visual_entities.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert vlm_summary["counts"]["vlm_visual_observations"] == 1
    assert entity_summary["counts"]["visual_entities"] == 1
    assert rows[0]["source"] == "vlm:fixture-vlm"
    assert rows[0]["text"] == "E = mc^2"
    assert rows[0]["visual_description"] == "A displayed equation"
    assert rows[0]["detected_text"] == "E = mc^2"


def test_vlm_observations_backend_rejects_unknown_frame_id(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "unknown_observation_project"
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
        project_dir / "manifests" / "vlm_visual_observations.jsonl",
        [
            {
                "observation_id": "obs_unknown",
                "project_id": "unknown_observation_project",
                "video_id": "lecture_01",
                "frame_id": "frame_000999",
                "timestamp": 0.0,
                "segment_id": None,
                "backend": "deterministic",
                "source_model": "offline-vlm",
                "model_version": None,
                "confidence": 0.9,
                "status": "success",
                "observation_type": "diagram",
                "visual_description": "unknown frame",
            }
        ],
    )

    with pytest.raises(ValueError, match="VLM observations contain frame_id"):
        extract_visual_entities(
            project_dir=project_dir,
            backend="vlm-observations",
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


def test_filter_visual_entities_applies_low_confidence_policy_to_ocr_only() -> None:
    base = {
        "project_id": "sample_project",
        "frame_id": "frame_000001",
        "timestamp": 1.0,
        "frame_path": "/tmp/frame.jpg",
        "bbox": None,
        "confidence": 0.32,
    }
    ocr_entity = VisualEntity(
        entity_id="ent_ocr",
        text="CALL",
        entity_type="ocr_text",
        source="ocr:tesseract",
        **base,
    )
    vlm_entity = VisualEntity(
        entity_id="ent_vlm",
        text="",
        entity_type="diagram",
        source="vlm:stub-vlm",
        visual_description="A low-confidence but useful visual description",
        **base,
    )

    filtered, summary = filter_visual_entities([ocr_entity, vlm_entity])

    assert [entity.entity_id for entity in filtered] == ["ent_vlm"]
    assert summary["filter_reasons"]["low_confidence"] == 1
    assert summary["policy"]["ocr_only_confidence_filters"] is True


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
