import json
import sys
from pathlib import Path

import pytest

from oarag.vlm import make_vlm_backend, run_vlm


def test_run_vlm_deterministic_backend_writes_observations_and_manifest(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    frames_manifest = project_dir / "manifests" / "frames_manifest.jsonl"
    project_manifest = project_dir / "manifests" / "project_manifest.json"
    output = project_dir / "manifests" / "vlm_visual_observations.jsonl"
    _write_jsonl(
        frames_manifest,
        [
            {
                "frame_id": "frame_000001",
                "frame_path": "frames/frame_000001.jpg",
                "timestamp": 1.25,
            },
            {
                "frame_id": "frame_000002",
                "frame_path": "frames/frame_000002.jpg",
                "timestamp": 2.5,
            },
        ],
    )
    _write_json(
        project_manifest,
        {
            "project_id": "sample_project",
            "video_id": "lecture_01",
            "artifacts": {"frames_manifest": str(frames_manifest)},
            "counts": {"frames": 2},
        },
    )

    summary = run_vlm(
        project_dir=project_dir,
        backend="deterministic",
        model="stub-vlm",
        device="cpu",
        options={
            "description_prefix": "Fixture observation",
            "detected_text": "Matrix A",
            "confidence": "0.82",
            "model_version": "test-v1",
        },
    )

    rows = _read_jsonl(output)
    manifest = json.loads(project_manifest.read_text(encoding="utf-8"))

    assert summary["backend"] == "deterministic"
    assert summary["source_model"] == "stub-vlm"
    assert summary["model_version"] == "test-v1"
    assert summary["counts"]["vlm_visual_observations"] == 2
    assert [row["frame_id"] for row in rows] == ["frame_000001", "frame_000002"]
    assert rows[0]["visual_description"] == "Fixture observation: frame_000001 at 1.250s"
    assert rows[0]["detected_text"] == "Matrix A"
    assert rows[0]["confidence"] == 0.82
    assert rows[0]["source_model"] == "stub-vlm"
    assert manifest["artifacts"]["frames_manifest"] == str(frames_manifest)
    assert manifest["artifacts"]["vlm_visual_observations"] == str(output)
    assert manifest["counts"]["frames"] == 2
    assert manifest["counts"]["vlm_visual_observations"] == 2
    assert manifest["vlm_consistency"]["backend"] == "deterministic"
    assert manifest["vlm_consistency"]["source_model"] == "stub-vlm"
    assert manifest["vlm_consistency"]["settings"]["device"] == "cpu"
    assert manifest["vlm_consistency"]["settings"]["run_id"].startswith("vlm_")


def test_run_vlm_mock_backend_alias_is_dependency_free(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "mock_project"
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {
                "frame_id": "frame_000001",
                "frame_path": "frames/frame_000001.jpg",
                "timestamp": 1.0,
            }
        ],
    )

    summary = run_vlm(project_dir=project_dir, backend="mock", model="stub-vlm")
    rows = _read_jsonl(project_dir / "manifests" / "vlm_visual_observations.jsonl")

    assert summary["backend"] == "mock"
    assert summary["counts"]["vlm_visual_observations"] == 1
    assert rows[0]["backend"] == "mock"
    assert rows[0]["source_model"] == "stub-vlm"


def test_run_vlm_jsonl_backend_replays_fixture_observations_and_redacts_path(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "jsonl_project"
    frames_manifest = project_dir / "manifests" / "frames_manifest.jsonl"
    fixture = project_dir / "manifests" / "vlm_fixture_observations.jsonl"
    project_manifest = project_dir / "manifests" / "project_manifest.json"
    _write_jsonl(
        frames_manifest,
        [
            {
                "frame_id": "frame_000001",
                "frame_path": "frames/frame_000001.jpg",
                "timestamp": 1.25,
            }
        ],
    )
    _write_jsonl(
        fixture,
        [
            {
                "parser_version": "jsonl-parser-v1",
                "source_model": "fixture-vlm",
                "observations": [
                    {
                        "frame_id": "frame_000001",
                        "observation_type": "diagram",
                        "visual_description": "JSONL fixture diagram",
                        "detected_text": "fixture label",
                        "confidence": 0.77,
                        "position": {"region": "right"},
                        "relations": [{"type": "points_to", "target": "axis"}],
                    }
                ],
            }
        ],
    )

    summary = run_vlm(
        project_dir=project_dir,
        backend="jsonl",
        model="fallback-vlm",
        options={"jsonl_path": "manifests/vlm_fixture_observations.jsonl"},
    )

    rows = _read_jsonl(project_dir / "manifests" / "vlm_visual_observations.jsonl")
    manifest = json.loads(project_manifest.read_text(encoding="utf-8"))

    assert summary["backend"] == "jsonl"
    assert summary["counts"]["vlm_visual_observations"] == 1
    assert rows[0]["backend"] == "jsonl"
    assert rows[0]["source_model"] == "fixture-vlm"
    assert rows[0]["visual_description"] == "JSONL fixture diagram"
    assert rows[0]["metadata"]["parser_version"] == "jsonl-parser-v1"
    assert rows[0]["metadata"]["backend_options"]["jsonl_path"] == "<configured>"
    assert manifest["vlm_consistency"]["settings"]["options"]["jsonl_path"] == "<configured>"


def test_run_vlm_uses_default_frame_candidates_when_present(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "candidate_project"
    frames_manifest = project_dir / "manifests" / "frames_manifest.jsonl"
    frame_candidates = project_dir / "manifests" / "vlm_frame_candidates.jsonl"
    _write_jsonl(
        frames_manifest,
        [
            {
                "frame_id": "frame_000001",
                "frame_path": "frames/frame_000001.jpg",
                "timestamp": 1.0,
            },
            {
                "frame_id": "frame_000002",
                "frame_path": "frames/frame_000002.jpg",
                "timestamp": 2.0,
            },
        ],
    )
    _write_jsonl(
        frame_candidates,
        [
            {
                "project_id": "candidate_project",
                "video_id": "lecture_01",
                "frame_id": "frame_000002",
                "timestamp": 2.0,
                "segment_id": "seg_0001",
                "backend": "vlm-frame-candidate-selector",
                "source_model": None,
                "model_version": None,
                "confidence": None,
                "status": "selected",
                "frame_path": "frames/frame_000002.jpg",
                "selection_reason": "segment_coverage",
                "rank": 1,
            }
        ],
    )

    summary = run_vlm(
        project_dir=project_dir,
        backend="deterministic",
        model="stub-vlm",
    )

    rows = _read_jsonl(project_dir / "manifests" / "vlm_visual_observations.jsonl")

    assert summary["counts"]["frames_total"] == 1
    assert rows[0]["frame_id"] == "frame_000002"
    assert rows[0]["segment_id"] == "seg_0001"
    assert rows[0]["attributes"]["rank"] == 1


def test_run_vlm_resume_skips_existing_frame_observations(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "resume_project"
    frames_manifest = project_dir / "manifests" / "frames_manifest.jsonl"
    output = project_dir / "manifests" / "vlm_visual_observations.jsonl"
    _write_jsonl(
        frames_manifest,
        [
            {
                "frame_id": "frame_000001",
                "frame_path": "frames/frame_000001.jpg",
                "timestamp": 1.0,
            },
            {
                "frame_id": "frame_000002",
                "frame_path": "frames/frame_000002.jpg",
                "timestamp": 2.0,
            },
        ],
    )
    _write_jsonl(
        output,
        [
            {
                "observation_id": "obs_existing",
                "project_id": "resume_project",
                "video_id": "resume_project",
                "frame_id": "frame_000001",
                "timestamp": 1.0,
                "segment_id": None,
                "backend": "deterministic",
                "source_model": "stub-vlm",
                "model_version": None,
                "confidence": 1.0,
                "status": "success",
                "observation_type": "frame_summary",
                "visual_description": "Existing observation",
            }
        ],
    )

    summary = run_vlm(
        project_dir=project_dir,
        backend="deterministic",
        model="stub-vlm",
        resume=True,
    )

    rows = _read_jsonl(output)
    manifest = json.loads(
        (project_dir / "manifests" / "project_manifest.json").read_text(encoding="utf-8")
    )

    assert summary["counts"]["frames_total"] == 2
    assert summary["counts"]["frames_processed"] == 1
    assert summary["counts"]["frames_skipped_resumed"] == 1
    assert summary["frame_status_counts"] == {"skipped_resumed": 1, "success": 1}
    assert [row["frame_id"] for row in rows] == ["frame_000001", "frame_000002"]
    assert rows[0]["visual_description"] == "Existing observation"
    assert rows[1]["status"] == "success"
    assert manifest["vlm_consistency"]["skips"] == {
        "count": 1,
        "reasons": {"resume": 1},
    }
    assert manifest["vlm_consistency"]["frame_status_counts"]["skipped_resumed"] == 1


def test_run_vlm_records_frame_level_backend_and_parse_failures(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "failure_project"
    frames_manifest = project_dir / "manifests" / "frames_manifest.jsonl"
    _write_jsonl(
        frames_manifest,
        [
            {
                "frame_id": "frame_000001",
                "frame_path": "frames/frame_000001.jpg",
                "timestamp": 1.0,
            },
            {
                "frame_id": "frame_000002",
                "frame_path": "frames/frame_000002.jpg",
                "timestamp": 2.0,
            },
            {
                "frame_id": "frame_000003",
                "frame_path": "frames/frame_000003.jpg",
                "timestamp": 3.0,
            },
        ],
    )

    summary = run_vlm(
        project_dir=project_dir,
        backend="deterministic",
        model="stub-vlm",
        options={
            "fail_frame_ids": "frame_000002",
            "parse_fail_frame_ids": ["frame_000003"],
        },
    )

    rows = _read_jsonl(project_dir / "manifests" / "vlm_visual_observations.jsonl")
    manifest = json.loads(
        (project_dir / "manifests" / "project_manifest.json").read_text(encoding="utf-8")
    )

    assert summary["status"] == "completed_with_errors"
    assert summary["counts"]["frames_failed"] == 2
    assert [row["status"] for row in rows] == [
        "success",
        "backend_failure",
        "parse_failure",
    ]
    assert rows[1]["metadata"]["failure_reason"].startswith(
        "Deterministic backend failure requested"
    )
    assert rows[2]["metadata"]["failure_reason"].startswith(
        "Deterministic parse failure requested"
    )
    assert manifest["vlm_consistency"]["status"] == "completed_with_errors"
    assert manifest["vlm_consistency"]["failures"] == {
        "count": 2,
        "reasons": {"backend_failure": 1, "parse_failure": 1},
    }
    assert manifest["vlm_consistency"]["frame_status_counts"] == {
        "success": 1,
        "backend_failure": 1,
        "parse_failure": 1,
    }
    assert isinstance(manifest["vlm_consistency"]["elapsed_seconds"], float)


def test_run_vlm_command_backend_accepts_json_stdin_and_records_public_contract(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "command_project"
    frames_manifest = project_dir / "manifests" / "frames_manifest.jsonl"
    project_manifest = project_dir / "manifests" / "project_manifest.json"
    output = project_dir / "manifests" / "vlm_visual_observations.jsonl"
    _write_jsonl(
        frames_manifest,
        [
            {
                "frame_id": "frame_000001",
                "frame_path": "frames/frame_000001.jpg",
                "timestamp": 4.25,
            }
        ],
    )

    command_code = (
        "import json,sys;"
        "req=json.load(sys.stdin);"
        "f=req['frame'];"
        "print(json.dumps({'observations':[{'frame_id':f['frame_id'],"
        "'observation_type':'diagram','visual_description':'A VLM parsed chart',"
        "'parser_version':'command-parser-v1',"
        "'detected_text':'Chart A','confidence':0.91,"
        "'position':{'region':'center'},"
        "'relations':[{'type':'contains','target':'label'}]}]}))"
    )

    summary = run_vlm(
        project_dir=project_dir,
        backend="command",
        model="fixture-vlm",
        options={
            "command": [sys.executable, "-c", command_code],
            "input_mode": "json-stdin",
            "prompt_template_version": "fixture-template-v2",
            "temperature": 0,
            "api_key": "do-not-write",
        },
    )

    rows = _read_jsonl(output)
    manifest = json.loads(project_manifest.read_text(encoding="utf-8"))

    assert summary["status"] == "completed"
    assert rows[0]["backend"] == "command"
    assert rows[0]["source_model"] == "fixture-vlm"
    assert rows[0]["visual_description"] == "A VLM parsed chart"
    assert rows[0]["metadata"]["parser_version"] == "command-parser-v1"
    assert rows[0]["metadata"]["prompt_template_version"] == "fixture-template-v2"
    assert rows[0]["metadata"]["backend_options"]["command"] == "<configured>"
    assert rows[0]["metadata"]["backend_options"]["api_key"] == "<redacted>"
    assert manifest["vlm_consistency"]["settings"]["prompt_template_version"] == (
        "fixture-template-v2"
    )
    assert manifest["vlm_consistency"]["settings"]["options"]["command"] == "<configured>"
    assert manifest["vlm_consistency"]["settings"]["options"]["api_key"] == "<redacted>"


def test_run_vlm_command_backend_schema_failures_become_parse_failure_rows(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "bad_command_project"
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {
                "frame_id": "frame_000001",
                "frame_path": "frames/frame_000001.jpg",
                "timestamp": 1.0,
            }
        ],
    )

    command_code = "import json; print(json.dumps({'observations': []}))"

    summary = run_vlm(
        project_dir=project_dir,
        backend="command",
        model="fixture-vlm",
        options={"command": [sys.executable, "-c", command_code]},
    )

    rows = _read_jsonl(project_dir / "manifests" / "vlm_visual_observations.jsonl")

    assert summary["status"] == "completed_with_errors"
    assert summary["counts"]["frames_failed"] == 1
    assert rows[0]["status"] == "parse_failure"
    assert rows[0]["metadata"]["failure_reason"].startswith(
        "VLM backend returned no observations"
    )


def test_run_vlm_command_backend_failure_reason_redacts_command_secrets(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "secret_command_project"
    secret = "SECRET_SHOULD_NOT_APPEAR"
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {
                "frame_id": "frame_000001",
                "frame_path": "frames/frame_000001.jpg",
                "timestamp": 1.0,
            }
        ],
    )

    summary = run_vlm(
        project_dir=project_dir,
        backend="command",
        model="fixture-vlm",
        options={
            "command": [
                sys.executable,
                "-c",
                "import sys; sys.exit(7)",
                "--api-key",
                secret,
            ],
            "api_key": secret,
        },
    )

    observations_text = (
        project_dir / "manifests" / "vlm_visual_observations.jsonl"
    ).read_text(encoding="utf-8")
    manifest_text = (project_dir / "manifests" / "project_manifest.json").read_text(
        encoding="utf-8"
    )
    rows = _read_jsonl(project_dir / "manifests" / "vlm_visual_observations.jsonl")

    assert summary["status"] == "completed_with_errors"
    assert rows[0]["status"] == "backend_failure"
    assert rows[0]["metadata"]["failure_reason"].endswith("exited with 7")
    assert "--api-key" not in rows[0]["metadata"]["failure_reason"]
    assert secret not in observations_text
    assert secret not in manifest_text


def test_run_vlm_command_backend_rejects_unknown_placeholder(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "placeholder_project"
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {
                "frame_id": "frame_000001",
                "frame_path": "frames/frame_000001.jpg",
                "timestamp": 1.0,
            }
        ],
    )

    summary = run_vlm(
        project_dir=project_dir,
        backend="command",
        model="fixture-vlm",
        options={"command": ["echo", "{unknown}"]},
    )

    rows = _read_jsonl(project_dir / "manifests" / "vlm_visual_observations.jsonl")

    assert summary["status"] == "completed_with_errors"
    assert rows[0]["status"] == "backend_failure"
    assert rows[0]["metadata"]["failure_reason"] == (
        "Unknown VLM command placeholder: unknown"
    )


def test_vlm_backend_errors_are_clear() -> None:
    with pytest.raises(ValueError, match="Unsupported VLM backend"):
        make_vlm_backend("missing-backend")


def test_run_vlm_requires_model(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    project_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="--vlm-model is required"):
        run_vlm(project_dir=project_dir, model=None)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
