import json
from pathlib import Path

import pytest

from oarag.cli import (
    build_parser,
    build_vlm_alignment_parser,
    cmd_build_project_evidence_units,
    cmd_build_project_windows,
    cmd_index_project_evidence_units,
    cmd_index_project,
    cmd_index_project_visual_entities,
    cmd_index_project_windows,
    cmd_query_project_evidence_units,
    parse_vlm_options,
)
from oarag.meili import (
    EVIDENCE_UNIT_DEFAULT_SETTINGS_PROFILE,
    HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE,
    LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
    LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE,
    LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE,
    VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
)


def test_stt_language_defaults_to_env(monkeypatch) -> None:
    monkeypatch.setenv("OARAG_STT_LANGUAGE", "en")

    args = build_parser().parse_args(
        ["ingest-video", "--video", "sample.mp4", "--project-id", "sample"]
    )

    assert args.stt_language == "en"


def test_stt_language_cli_argument_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("OARAG_STT_LANGUAGE", "ko")

    args = build_parser().parse_args(
        [
            "ingest-video",
            "--video",
            "sample.mp4",
            "--project-id",
            "sample",
            "--stt-language",
            "en",
        ]
    )

    assert args.stt_language == "en"


def test_blank_stt_language_env_is_ignored(monkeypatch) -> None:
    monkeypatch.setenv("OARAG_STT_LANGUAGE", "  ")

    args = build_parser().parse_args(
        ["ingest-video", "--video", "sample.mp4", "--project-id", "sample"]
    )

    assert args.stt_language is None


def test_health_neo4j_cli_calls_helper(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "oarag.cli.check_neo4j_health",
        lambda: {"ok": True, "uri": "bolt://127.0.0.1:7687"},
    )

    args = build_parser().parse_args(["health-neo4j"])
    args.func(args)

    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "uri": "bolt://127.0.0.1:7687",
    }


def test_align_frames_cli_defaults() -> None:
    args = build_parser().parse_args(["align-frames", "--project-id", "sample"])

    assert args.project_id == "sample"
    assert args.output_root.as_posix() == "artifacts/projects"
    assert args.margin_seconds == 0.0


def test_index_project_accepts_project_id() -> None:
    args = build_parser().parse_args(
        ["index-project", "--index", "local_segments", "--project-id", "sample", "--batch-size", "100"]
    )

    assert args.index == "local_segments"
    assert args.project_id == "sample"
    assert args.project_dir is None
    assert args.batch_size == 100
    assert args.settings_profile == LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE
    assert args.hybrid_embedder_profile is None
    assert args.hybrid_embedder_config is None
    assert args.hybrid_embedder_name == "default"
    assert args.hybrid_embedder_dimensions is None
    assert args.hybrid_embedder_live_smoke is False
    assert args.vector_manifest is None


def test_index_project_accepts_project_dir() -> None:
    args = build_parser().parse_args(
        [
            "index-project",
            "--index",
            "local_segments",
            "--project-dir",
            "artifacts/projects/sample",
            "--settings-profile",
            LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE,
        ]
    )

    assert args.index == "local_segments"
    assert str(args.project_dir) == "artifacts/projects/sample"
    assert args.project_id is None
    assert args.settings_profile == LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE


def test_cmd_index_project_forwards_settings_profile(monkeypatch, capsys) -> None:
    calls = {}
    fake_client = object()
    project_dir = Path("/tmp/project")

    def fake_project_dir_from_args(*, project_id, project_dir):
        calls["location"] = (project_id, project_dir)
        return Path("/tmp/project")

    def fake_index_project_segments(client, **kwargs):
        calls["client"] = client
        calls["index_kwargs"] = kwargs
        return {"settings_profile": kwargs["settings_profile"]}

    monkeypatch.setattr("oarag.cli.client_from_args", lambda args: fake_client)
    monkeypatch.setattr("oarag.cli.project_dir_from_args", fake_project_dir_from_args)
    monkeypatch.setattr("oarag.cli.index_project_segments", fake_index_project_segments)

    args = build_parser().parse_args(
        [
            "index-project",
            "--index",
            "local_segments",
            "--project-dir",
            str(project_dir),
            "--settings-profile",
            LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE,
        ]
    )

    cmd_index_project(args)

    assert calls["location"] == (None, project_dir)
    assert calls["client"] is fake_client
    assert calls["index_kwargs"]["settings_profile"] == LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE
    assert calls["index_kwargs"]["hybrid_embedder_profile"] is None
    assert calls["index_kwargs"]["hybrid_embedder_config"] is None
    assert calls["index_kwargs"]["vector_manifest"] is None
    assert json.loads(capsys.readouterr().out)["settings_profile"] == (
        LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE
    )


def test_cmd_index_project_forwards_hybrid_embedder_options(monkeypatch, capsys) -> None:
    calls = {}
    fake_client = object()
    project_dir = Path("/tmp/project")

    def fake_project_dir_from_args(*, project_id, project_dir):
        calls["location"] = (project_id, project_dir)
        return Path("/tmp/project")

    def fake_index_project_segments(client, **kwargs):
        calls["client"] = client
        calls["index_kwargs"] = kwargs
        return {
            "hybrid_embedder_profile": kwargs["hybrid_embedder_profile"],
            "hybrid_embedder_live_smoke": kwargs["hybrid_embedder_live_smoke"],
        }

    monkeypatch.setattr("oarag.cli.client_from_args", lambda args: fake_client)
    monkeypatch.setattr("oarag.cli.project_dir_from_args", fake_project_dir_from_args)
    monkeypatch.setattr("oarag.cli.index_project_segments", fake_index_project_segments)

    args = build_parser().parse_args(
        [
            "index-project",
            "--index",
            "local_segments",
            "--project-dir",
            str(project_dir),
            "--hybrid-embedder-profile",
            HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE,
            "--hybrid-embedder-name",
            "lecture_embedder",
            "--hybrid-embedder-dimensions",
            "768",
            "--hybrid-embedder-live-smoke",
            "--vector-manifest",
            "manifests/segment_vectors.json",
        ]
    )

    cmd_index_project(args)

    assert calls["client"] is fake_client
    assert calls["index_kwargs"]["hybrid_embedder_profile"] == (
        HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE
    )
    assert calls["index_kwargs"]["hybrid_embedder_config"] is None
    assert calls["index_kwargs"]["hybrid_embedder_name"] == "lecture_embedder"
    assert calls["index_kwargs"]["hybrid_embedder_dimensions"] == 768
    assert calls["index_kwargs"]["hybrid_embedder_live_smoke"] is True
    assert calls["index_kwargs"]["vector_manifest"] == Path("manifests/segment_vectors.json")
    assert json.loads(capsys.readouterr().out)["hybrid_embedder_live_smoke"] is True


def test_build_project_windows_cli_defaults() -> None:
    args = build_parser().parse_args(
        ["build-project-windows", "--project-id", "sample_project"]
    )

    assert args.project_id == "sample_project"
    assert args.project_dir is None
    assert args.segments is None
    assert args.frames_manifest is None
    assert args.visual_entities is None
    assert args.output is None
    assert args.manifest is None
    assert args.neighbor_count == 1
    assert args.window_seconds is None
    assert args.previous_neighbor_count is None
    assert args.next_neighbor_count is None


def test_cmd_build_project_windows_forwards_window_options(monkeypatch, capsys) -> None:
    calls = {}
    project_dir = Path("/tmp/project")

    def fake_project_dir_from_args(*, project_id, project_dir):
        calls["location"] = (project_id, project_dir)
        return Path("/tmp/project")

    def fake_build_project_windows(**kwargs):
        calls["build_kwargs"] = kwargs
        return {"counts": {"windows_total": 1}, "window_config": {"mode": "neighbors"}}

    monkeypatch.setattr("oarag.cli.project_dir_from_args", fake_project_dir_from_args)
    monkeypatch.setattr("oarag.cli.build_project_windows", fake_build_project_windows)

    args = build_parser().parse_args(
        [
            "build-project-windows",
            "--project-dir",
            str(project_dir),
            "--segments",
            "segments/custom.jsonl",
            "--output",
            "segments/lecture_windows.jsonl",
            "--previous-neighbor-count",
            "2",
            "--next-neighbor-count",
            "0",
        ]
    )

    cmd_build_project_windows(args)

    assert calls["location"] == (None, project_dir)
    assert calls["build_kwargs"]["project_dir"] == Path("/tmp/project")
    assert calls["build_kwargs"]["segments"].as_posix() == "segments/custom.jsonl"
    assert calls["build_kwargs"]["output_path"].as_posix() == "segments/lecture_windows.jsonl"
    assert calls["build_kwargs"]["previous_neighbor_count"] == 2
    assert calls["build_kwargs"]["next_neighbor_count"] == 0
    assert json.loads(capsys.readouterr().out)["counts"]["windows_total"] == 1


def test_build_project_evidence_units_cli_defaults() -> None:
    args = build_parser().parse_args(
        ["build-project-evidence-units", "--project-id", "sample_project"]
    )

    assert args.project_id == "sample_project"
    assert args.project_dir is None
    assert args.segments is None
    assert args.frames_manifest is None
    assert args.visual_entities is None
    assert args.entity_links is None
    assert args.output is None
    assert args.manifest is None
    assert args.state_padding_seconds == 15.0
    assert args.neighbor_count == 1
    assert args.window_seconds is None
    assert args.previous_neighbor_count is None
    assert args.next_neighbor_count is None


def test_cmd_build_project_evidence_units_forwards_inputs(monkeypatch, capsys) -> None:
    calls = {}
    project_dir = Path("/tmp/project")

    def fake_project_dir_from_args(*, project_id, project_dir):
        calls["location"] = (project_id, project_dir)
        return Path("/tmp/project")

    def fake_build_project_evidence_units(**kwargs):
        calls["build_kwargs"] = kwargs
        return {"counts": {"evidence_units_total": 1}, "alignment_status_counts": {"candidate": 1}}

    monkeypatch.setattr("oarag.cli.project_dir_from_args", fake_project_dir_from_args)
    monkeypatch.setattr(
        "oarag.cli.build_project_evidence_units",
        fake_build_project_evidence_units,
    )

    args = build_parser().parse_args(
        [
            "build-project-evidence-units",
            "--project-dir",
            str(project_dir),
            "--segments",
            "segments/custom.jsonl",
            "--entity-links",
            "manifests/entity_links.jsonl",
            "--output",
            "segments/evidence_units.jsonl",
            "--state-padding-seconds",
            "20",
            "--previous-neighbor-count",
            "2",
            "--next-neighbor-count",
            "0",
        ]
    )

    cmd_build_project_evidence_units(args)

    assert calls["location"] == (None, project_dir)
    assert calls["build_kwargs"]["project_dir"] == Path("/tmp/project")
    assert calls["build_kwargs"]["segments"].as_posix() == "segments/custom.jsonl"
    assert calls["build_kwargs"]["entity_links"].as_posix() == "manifests/entity_links.jsonl"
    assert calls["build_kwargs"]["output_path"].as_posix() == "segments/evidence_units.jsonl"
    assert calls["build_kwargs"]["state_padding_seconds"] == 20.0
    assert calls["build_kwargs"]["previous_neighbor_count"] == 2
    assert calls["build_kwargs"]["next_neighbor_count"] == 0
    assert json.loads(capsys.readouterr().out)["counts"]["evidence_units_total"] == 1


def test_index_project_evidence_units_cli_defaults() -> None:
    args = build_parser().parse_args(
        [
            "index-project-evidence-units",
            "--index",
            "local_evidence_units",
            "--project-id",
            "sample_project",
        ]
    )

    assert args.index == "local_evidence_units"
    assert args.project_id == "sample_project"
    assert args.project_dir is None
    assert args.evidence_units is None
    assert args.batch_size == 500
    assert args.reset is False
    assert args.settings_profile == EVIDENCE_UNIT_DEFAULT_SETTINGS_PROFILE


def test_cmd_index_project_evidence_units_forwards_inputs(monkeypatch, capsys) -> None:
    calls = {}
    fake_client = object()
    project_dir = Path("/tmp/project")

    def fake_project_dir_from_args(*, project_id, project_dir):
        calls["location"] = (project_id, project_dir)
        return Path("/tmp/project")

    def fake_index_project_evidence_units(client, **kwargs):
        calls["client"] = client
        calls["index_kwargs"] = kwargs
        return {
            "settings_profile": kwargs["settings_profile"],
            "indexed_documents": 2,
        }

    monkeypatch.setattr("oarag.cli.client_from_args", lambda args: fake_client)
    monkeypatch.setattr("oarag.cli.project_dir_from_args", fake_project_dir_from_args)
    monkeypatch.setattr(
        "oarag.cli.index_project_evidence_units",
        fake_index_project_evidence_units,
    )

    args = build_parser().parse_args(
        [
            "index-project-evidence-units",
            "--index",
            "local_evidence_units",
            "--project-dir",
            str(project_dir),
            "--evidence-units",
            "segments/evidence_units.jsonl",
            "--batch-size",
            "100",
            "--reset",
        ]
    )

    cmd_index_project_evidence_units(args)

    assert calls["location"] == (None, project_dir)
    assert calls["client"] is fake_client
    assert calls["index_kwargs"]["index_uid"] == "local_evidence_units"
    assert calls["index_kwargs"]["project_dir"] == Path("/tmp/project")
    assert calls["index_kwargs"]["evidence_units"].as_posix() == "segments/evidence_units.jsonl"
    assert calls["index_kwargs"]["batch_size"] == 100
    assert calls["index_kwargs"]["reset"] is True
    assert json.loads(capsys.readouterr().out)["indexed_documents"] == 2


def test_index_project_windows_cli_defaults() -> None:
    args = build_parser().parse_args(
        [
            "index-project-windows",
            "--index",
            "local_windows",
            "--project-id",
            "sample_project",
        ]
    )

    assert args.index == "local_windows"
    assert args.project_id == "sample_project"
    assert args.project_dir is None
    assert args.windows is None
    assert args.segments is None
    assert args.frames_manifest is None
    assert args.visual_entities is None
    assert args.batch_size == 500
    assert args.reset is False
    assert args.settings_profile == LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE
    assert args.hybrid_embedder_profile is None
    assert args.hybrid_embedder_config is None
    assert args.hybrid_embedder_name == "default"
    assert args.hybrid_embedder_dimensions is None
    assert args.hybrid_embedder_live_smoke is False
    assert args.vector_manifest is None
    assert args.neighbor_count == 1


def test_cmd_index_project_windows_forwards_settings_profile(monkeypatch, capsys) -> None:
    calls = {}
    fake_client = object()
    project_dir = Path("/tmp/project")

    def fake_project_dir_from_args(*, project_id, project_dir):
        calls["location"] = (project_id, project_dir)
        return Path("/tmp/project")

    def fake_index_project_windows(client, **kwargs):
        calls["client"] = client
        calls["index_kwargs"] = kwargs
        return {"settings_profile": kwargs["settings_profile"]}

    monkeypatch.setattr("oarag.cli.client_from_args", lambda args: fake_client)
    monkeypatch.setattr("oarag.cli.project_dir_from_args", fake_project_dir_from_args)
    monkeypatch.setattr("oarag.cli.index_project_windows", fake_index_project_windows)

    args = build_parser().parse_args(
        [
            "index-project-windows",
            "--index",
            "local_windows",
            "--project-dir",
            str(project_dir),
            "--windows",
            "segments/lecture_windows.jsonl",
            "--settings-profile",
            LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE,
            "--hybrid-embedder-profile",
            HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE,
            "--hybrid-embedder-dimensions",
            "512",
            "--window-before-seconds",
            "3",
            "--window-after-seconds",
            "5",
        ]
    )

    cmd_index_project_windows(args)

    assert calls["location"] == (None, project_dir)
    assert calls["client"] is fake_client
    assert calls["index_kwargs"]["settings_profile"] == LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE
    assert calls["index_kwargs"]["hybrid_embedder_profile"] == (
        HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE
    )
    assert calls["index_kwargs"]["hybrid_embedder_dimensions"] == 512
    assert calls["index_kwargs"]["windows"].as_posix() == "segments/lecture_windows.jsonl"
    assert calls["index_kwargs"]["vector_manifest"] is None
    assert calls["index_kwargs"]["window_before_seconds"] == 3.0
    assert calls["index_kwargs"]["window_after_seconds"] == 5.0
    assert json.loads(capsys.readouterr().out)["settings_profile"] == (
        LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE
    )


def test_index_project_visual_entities_cli_defaults() -> None:
    args = build_parser().parse_args(
        [
            "index-project-visual-entities",
            "--index",
            "local_visual_entities",
            "--project-id",
            "sample",
        ]
    )

    assert args.index == "local_visual_entities"
    assert args.project_id == "sample"
    assert args.project_dir is None
    assert args.visual_entities is None
    assert args.batch_size == 500
    assert args.reset is False
    assert args.settings_profile == VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE
    assert args.vector_manifest is None


def test_cmd_index_project_visual_entities_forwards_settings_profile(monkeypatch, capsys) -> None:
    calls = {}
    fake_client = object()
    project_dir = Path("/tmp/project")

    def fake_project_dir_from_args(*, project_id, project_dir):
        calls["location"] = (project_id, project_dir)
        return Path("/tmp/project")

    def fake_index_project_visual_entities(client, **kwargs):
        calls["client"] = client
        calls["index_kwargs"] = kwargs
        return {"settings_profile": kwargs["settings_profile"]}

    monkeypatch.setattr("oarag.cli.client_from_args", lambda args: fake_client)
    monkeypatch.setattr("oarag.cli.project_dir_from_args", fake_project_dir_from_args)
    monkeypatch.setattr(
        "oarag.cli.index_project_visual_entities",
        fake_index_project_visual_entities,
    )

    args = build_parser().parse_args(
        [
            "index-project-visual-entities",
            "--index",
            "local_visual_entities",
            "--project-dir",
            str(project_dir),
            "--visual-entities",
            "manifests/visual_entities.jsonl",
            "--settings-profile",
            VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
        ]
    )

    cmd_index_project_visual_entities(args)

    assert calls["location"] == (None, project_dir)
    assert calls["client"] is fake_client
    assert calls["index_kwargs"]["settings_profile"] == VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE
    assert calls["index_kwargs"]["visual_entities"].as_posix() == "manifests/visual_entities.jsonl"
    assert calls["index_kwargs"]["vector_manifest"] is None
    assert json.loads(capsys.readouterr().out)["settings_profile"] == (
        VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE
    )


def test_evidence_window_cli_accepts_project_and_segment() -> None:
    args = build_parser().parse_args(
        [
            "evidence-window",
            "--project-id",
            "sample",
            "--segment-id",
            "seg_1",
            "--segment-id",
            "seg_2",
            "--query",
            "bet size",
            "--neighbor-count",
            "2",
            "--previous-neighbor-count",
            "1",
            "--next-neighbor-count",
            "3",
            "--window-before-seconds",
            "4.5",
            "--window-after-seconds",
            "6.5",
        ]
    )

    assert args.project_id == "sample"
    assert args.project_dir is None
    assert args.segment_id == ["seg_1", "seg_2"]
    assert args.query == "bet size"
    assert args.neighbor_count == 2
    assert args.previous_neighbor_count == 1
    assert args.next_neighbor_count == 3
    assert args.window_before_seconds == 4.5
    assert args.window_after_seconds == 6.5


def test_extract_visual_entities_cli_defaults() -> None:
    args = build_parser().parse_args(
        ["extract-visual-entities", "--project-id", "sample_project"]
    )

    assert args.project_id == "sample_project"
    assert args.project_dir is None
    assert args.backend == "vlm-first"
    assert args.vlm_jsonl is None
    assert args.vlm_observations is None
    assert args.frames_manifest is None
    assert args.output is None
    assert args.manifest is None
    assert args.ocr_language is None


def test_extract_visual_entities_cli_accepts_vlm_jsonl_backend() -> None:
    args = build_parser().parse_args(
        [
            "extract-visual-entities",
            "--project-id",
            "sample_project",
            "--backend",
            "vlm-jsonl",
            "--vlm-jsonl",
            "manifests/vlm_parser_output.jsonl",
        ]
    )

    assert args.backend == "vlm-jsonl"
    assert args.vlm_jsonl.as_posix() == "manifests/vlm_parser_output.jsonl"


def test_extract_visual_entities_cli_accepts_vlm_observations_backend() -> None:
    args = build_parser().parse_args(
        [
            "extract-visual-entities",
            "--project-id",
            "sample_project",
            "--backend",
            "vlm-observations",
            "--vlm-observations",
            "manifests/vlm_visual_observations.jsonl",
        ]
    )

    assert args.backend == "vlm-observations"
    assert args.vlm_observations.as_posix() == "manifests/vlm_visual_observations.jsonl"


def test_extract_visual_entities_cli_help_mentions_vlm_observations(capsys) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["extract-visual-entities", "--help"])

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "vlm-observations" in captured.out
    assert "--vlm-observations" in captured.out
    assert "baseline/fallback" in captured.out


def test_run_vlm_cli_accepts_backend_model_and_options() -> None:
    args = build_parser().parse_args(
        [
            "run-vlm",
            "--project-id",
            "sample_project",
            "--vlm-backend",
            "deterministic",
            "--vlm-model",
            "stub-vlm",
            "--vlm-device",
            "cpu",
            "--vlm-options",
            "confidence=0.8,detected_text=Matrix A",
            "--vlm-frame-candidates",
            "manifests/vlm_frame_candidates.jsonl",
            "--resume",
        ]
    )

    assert args.project_id == "sample_project"
    assert args.vlm_backend == "deterministic"
    assert args.vlm_model == "stub-vlm"
    assert args.vlm_device == "cpu"
    assert args.vlm_options == "confidence=0.8,detected_text=Matrix A"
    assert args.vlm_frame_candidates.as_posix() == "manifests/vlm_frame_candidates.jsonl"
    assert args.resume is True


def test_run_vlm_cli_accepts_jsonl_and_mock_backend_choices() -> None:
    jsonl_args = build_parser().parse_args(
        [
            "run-vlm",
            "--project-id",
            "sample_project",
            "--vlm-backend",
            "jsonl",
            "--vlm-model",
            "fixture-vlm",
            "--vlm-options",
            "jsonl_path=manifests/vlm_backend_fixture.jsonl",
        ]
    )
    mock_args = build_parser().parse_args(
        [
            "run-vlm",
            "--project-id",
            "sample_project",
            "--vlm-backend",
            "mock",
            "--vlm-model",
            "fixture-vlm",
        ]
    )

    assert jsonl_args.vlm_backend == "jsonl"
    assert mock_args.vlm_backend == "mock"


def test_run_vlm_alignment_cli_accepts_pipeline_options() -> None:
    args = build_parser().parse_args(
        [
            "run-vlm-alignment",
            "--project-id",
            "sample_project",
            "--vlm-backend",
            "deterministic",
            "--vlm-model",
            "stub-vlm",
            "--vlm-options",
            "confidence=0.8,detected_text=Matrix A",
            "--max-vlm-frames",
            "12",
            "--candidate-max-per-segment",
            "3",
            "--candidate-window-seconds",
            "45",
            "--candidate-max-per-window",
            "5",
            "--candidate-min-time-gap-seconds",
            "1.5",
            "--consistency-window-margin-seconds",
            "0.25",
            "--resume",
            "--dry-run",
        ]
    )

    assert args.project_id == "sample_project"
    assert args.vlm_backend == "deterministic"
    assert args.vlm_model == "stub-vlm"
    assert args.vlm_options == "confidence=0.8,detected_text=Matrix A"
    assert args.max_vlm_frames == 12
    assert args.candidate_max_per_segment == 3
    assert args.candidate_window_seconds == 45
    assert args.candidate_max_per_window == 5
    assert args.candidate_min_time_gap_seconds == 1.5
    assert args.consistency_window_margin_seconds == 0.25
    assert args.resume is True
    assert args.dry_run is True


def test_run_vlm_alignment_script_parser_accepts_options_without_subcommand() -> None:
    args = build_vlm_alignment_parser().parse_args(
        [
            "--project-dir",
            "artifacts/projects/sample_project",
            "--vlm-model",
            "stub-vlm",
            "--max-vlm-frames",
            "2",
        ]
    )

    assert args.project_id is None
    assert args.project_dir.as_posix() == "artifacts/projects/sample_project"
    assert args.vlm_model == "stub-vlm"
    assert args.max_vlm_frames == 2


def test_parse_vlm_options_accepts_json_and_key_value_pairs() -> None:
    assert parse_vlm_options('{"confidence": 0.7, "detected_text": "Matrix A"}') == {
        "confidence": 0.7,
        "detected_text": "Matrix A",
    }
    assert parse_vlm_options("confidence=0.8,detected_text=Matrix A") == {
        "confidence": 0.8,
        "detected_text": "Matrix A",
    }


def test_link_entities_cli_defaults() -> None:
    args = build_parser().parse_args(
        ["link-entities", "--project-id", "sample_project"]
    )

    assert args.project_id == "sample_project"
    assert args.project_dir is None
    assert args.segments is None
    assert args.visual_entities is None
    assert args.output is None
    assert args.manifest is None
    assert args.domain_lexicon is None


def test_query_project_cli_defaults() -> None:
    args = build_parser().parse_args(
        ["query-project", "--index", "sample_segments", "--project-id", "sample_project", "--query", "bet size"]
    )

    assert args.index == "sample_segments"
    assert args.index_kind == "segment"
    assert args.visual_index is None
    assert args.project_id == "sample_project"
    assert args.project_dir is None
    assert args.query == "bet size"
    assert args.limit == 5
    assert args.segments is None
    assert args.frames_manifest is None
    assert args.visual_entities is None
    assert args.entity_links is None
    assert args.domain_lexicon is None
    assert args.window_seconds is None
    assert args.neighbor_count == 1
    assert args.previous_neighbor_count is None
    assert args.next_neighbor_count is None
    assert args.window_before_seconds is None
    assert args.window_after_seconds is None
    assert args.rerank is False
    assert args.rerank_time_hint is None
    assert args.rerank_backend == "deterministic"
    assert args.hybrid_retrieval is False
    assert args.hybrid_embedder == "default"
    assert args.hybrid_semantic_ratio == 1.0
    assert args.output is None


def test_query_project_cli_accepts_rerank_options() -> None:
    args = build_parser().parse_args(
        [
            "query-project",
            "--index",
            "sample_segments",
            "--project-id",
            "sample_project",
            "--query",
            "bet size",
            "--rerank",
            "--rerank-time-hint",
            "10-14s",
            "--rerank-backend",
            "stub",
        ]
    )

    assert args.rerank is True
    assert args.rerank_time_hint == "10-14s"
    assert args.rerank_backend == "stub"


def test_query_project_cli_accepts_visual_index() -> None:
    args = build_parser().parse_args(
        [
            "query-project",
            "--index",
            "sample_segments",
            "--visual-index",
            "sample_visual_entities",
            "--project-id",
            "sample_project",
            "--query",
            "range grid",
        ]
    )

    assert args.index == "sample_segments"
    assert args.visual_index == "sample_visual_entities"


def test_query_project_cli_accepts_hybrid_options() -> None:
    args = build_parser().parse_args(
        [
            "query-project",
            "--index",
            "sample_segments",
            "--project-id",
            "sample_project",
            "--query",
            "range grid",
            "--hybrid-retrieval",
            "--hybrid-embedder",
            "lecture_embedder",
            "--hybrid-semantic-ratio",
            "0.75",
        ]
    )

    assert args.hybrid_retrieval is True
    assert args.hybrid_embedder == "lecture_embedder"
    assert args.hybrid_semantic_ratio == 0.75


def test_query_project_cli_accepts_window_index_kind() -> None:
    args = build_parser().parse_args(
        [
            "query-project",
            "--index",
            "sample_windows",
            "--index-kind",
            "window",
            "--project-id",
            "sample_project",
            "--query",
            "range grid",
        ]
    )

    assert args.index == "sample_windows"
    assert args.index_kind == "window"


def test_query_project_evidence_units_cli_defaults() -> None:
    args = build_parser().parse_args(
        [
            "query-project-evidence-units",
            "--index",
            "sample_evidence_units",
            "--project-id",
            "sample_project",
            "--query",
            "gradient arrow",
        ]
    )

    assert args.index == "sample_evidence_units"
    assert args.project_id == "sample_project"
    assert args.project_dir is None
    assert args.query == "gradient arrow"
    assert args.limit == 5
    assert args.evidence_units is None
    assert args.output is None


def test_cmd_query_project_evidence_units_forwards_inputs(monkeypatch, capsys) -> None:
    calls = {}
    fake_client = object()
    project_dir = Path("/tmp/project")

    def fake_project_dir_from_args(*, project_id, project_dir):
        calls["location"] = (project_id, project_dir)
        return Path("/tmp/project")

    def fake_query_project_evidence_units(**kwargs):
        calls["query_kwargs"] = kwargs
        return {"query": kwargs["query"], "candidates": []}

    monkeypatch.setattr("oarag.cli.client_from_args", lambda args: fake_client)
    monkeypatch.setattr("oarag.cli.project_dir_from_args", fake_project_dir_from_args)
    monkeypatch.setattr(
        "oarag.cli.query_project_evidence_units",
        fake_query_project_evidence_units,
    )

    args = build_parser().parse_args(
        [
            "query-project-evidence-units",
            "--index",
            "sample_evidence_units",
            "--project-dir",
            str(project_dir),
            "--query",
            "gradient arrow",
            "--limit",
            "3",
            "--evidence-units",
            "segments/evidence_units.jsonl",
        ]
    )

    cmd_query_project_evidence_units(args)

    assert calls["location"] == (None, project_dir)
    assert calls["query_kwargs"]["client"] is fake_client
    assert calls["query_kwargs"]["index_uid"] == "sample_evidence_units"
    assert calls["query_kwargs"]["project_dir"] == Path("/tmp/project")
    assert calls["query_kwargs"]["query"] == "gradient arrow"
    assert calls["query_kwargs"]["limit"] == 3
    assert calls["query_kwargs"]["evidence_units"].as_posix() == "segments/evidence_units.jsonl"
    assert json.loads(capsys.readouterr().out)["query"] == "gradient arrow"


def test_ask_project_cli_defaults() -> None:
    args = build_parser().parse_args(
        [
            "ask-project",
            "--index",
            "sample_segments",
            "--project-id",
            "sample_project",
            "--query",
            "bet size",
            "--format",
            "json",
        ]
    )

    assert args.index == "sample_segments"
    assert args.index_kind == "segment"
    assert args.visual_index is None
    assert args.project_id == "sample_project"
    assert args.project_dir is None
    assert args.query == "bet size"
    assert args.limit == 5
    assert args.rerank_backend == "deterministic"
    assert args.hybrid_retrieval is False
    assert args.hybrid_embedder == "default"
    assert args.hybrid_semantic_ratio == 1.0
    assert args.output_format == "json"
    assert args.output is None


def test_ask_project_cli_accepts_window_hybrid_and_rerank_options() -> None:
    args = build_parser().parse_args(
        [
            "ask-project",
            "--index",
            "sample_windows",
            "--index-kind",
            "window",
            "--project-id",
            "sample_project",
            "--query",
            "range grid",
            "--rerank",
            "--rerank-time-hint",
            "10-14s",
            "--rerank-backend",
            "stub",
            "--hybrid-retrieval",
            "--hybrid-embedder",
            "lecture_embedder",
            "--hybrid-semantic-ratio",
            "0.75",
        ]
    )

    assert args.index == "sample_windows"
    assert args.index_kind == "window"
    assert args.rerank is True
    assert args.rerank_time_hint == "10-14s"
    assert args.rerank_backend == "stub"
    assert args.hybrid_retrieval is True
    assert args.hybrid_embedder == "lecture_embedder"
    assert args.hybrid_semantic_ratio == 0.75


def test_graph_query_cli_defaults() -> None:
    args = build_parser().parse_args(
        ["graph-query", "--index", "sample_segments", "--project-id", "sample_project", "--query", "아까 개념"]
    )

    assert args.index == "sample_segments"
    assert args.project_id == "sample_project"
    assert args.project_dir is None
    assert args.query == "아까 개념"
    assert args.limit == 5
    assert args.segments is None
    assert args.frames_manifest is None
    assert args.visual_entities is None
    assert args.entity_links is None
    assert args.domain_lexicon is None
    assert args.graph_lookback_segments == 3
    assert args.graph_limit == 12
    assert args.output is None


def test_batch_ingest_cli_defaults() -> None:
    args = build_parser().parse_args(["batch-ingest", "--root", "lectures"])

    assert str(args.root) == "lectures"
    assert args.output_root.as_posix() == "artifacts/projects"
    assert args.frame_sampling == "uniform"
    assert args.frame_selection == "none"
    assert args.transcript_source == "auto"
    assert args.force is False
    assert args.strict is False
    assert args.dry_run is False


def test_ingest_video_cli_accepts_prefix_frame_sampling() -> None:
    args = build_parser().parse_args(
        [
            "ingest-video",
            "--video",
            "sample.mp4",
            "--project-id",
            "sample",
            "--frame-sampling",
            "prefix",
        ]
    )

    assert args.frame_sampling == "prefix"


def test_ingest_video_cli_accepts_representative_frame_selection() -> None:
    args = build_parser().parse_args(
        [
            "ingest-video",
            "--video",
            "sample.mp4",
            "--project-id",
            "sample",
            "--frame-selection",
            "representative",
        ]
    )

    assert args.frame_selection == "representative"


def test_ingest_folder_alias_maps_to_batch_ingest() -> None:
    args = build_parser().parse_args(["ingest-folder", "--root", "lectures", "--force"])

    assert str(args.root) == "lectures"
    assert args.force is True


def test_benchmark_retrieval_cli_accepts_manifest_and_output_dir() -> None:
    args = build_parser().parse_args(
        [
            "benchmark-retrieval",
            "--manifest",
            "benchmarks/retrieval.json",
            "--output-dir",
            "reports/perf_runs/dev",
            "--diagnostic-top-k",
            "10",
        ]
    )

    assert str(args.manifest) == "benchmarks/retrieval.json"
    assert str(args.output_dir) == "reports/perf_runs/dev"
    assert args.diagnostic_top_k == 10


def test_run_paper_experiment_cli_accepts_manifest_gate_output_and_run_id() -> None:
    args = build_parser().parse_args(
        [
            "run-paper-experiment",
            "--manifest",
            "benchmarks/retrieval.json",
            "--output-dir",
            "reports/paper/dev",
            "--run-id",
            "paper_dev",
            "--gate-config",
            "benchmarks/gate.json",
            "--diagnostic-top-k",
            "5",
            "--no-fail-on-gate",
        ]
    )

    assert str(args.manifest) == "benchmarks/retrieval.json"
    assert str(args.output_dir) == "reports/paper/dev"
    assert args.run_id == "paper_dev"
    assert str(args.gate_config) == "benchmarks/gate.json"
    assert args.diagnostic_top_k == 5
    assert args.no_fail_on_gate is True


def test_audit_paper_readiness_cli_accepts_bundle_paths() -> None:
    args = build_parser().parse_args(
        [
            "audit-paper-readiness",
            "--experiment-manifest",
            "reports/paper/dev/experiment_manifest.json",
            "--output-dir",
            "reports/paper/dev/readiness",
            "--metrics",
            "reports/paper/dev/metrics.json",
            "--query-results",
            "reports/paper/dev/query_results.jsonl",
            "--quality-gate-result",
            "reports/paper/dev/quality_gate_result.json",
            "--reproducibility",
            "reports/paper/dev/paper_report/reproducibility.json",
            "--semantic-smoke",
            "reports/paper/dev/semantic_smoke.json",
            "--fail-on-gap",
        ]
    )

    assert str(args.experiment_manifest) == "reports/paper/dev/experiment_manifest.json"
    assert str(args.output_dir) == "reports/paper/dev/readiness"
    assert str(args.metrics) == "reports/paper/dev/metrics.json"
    assert str(args.query_results) == "reports/paper/dev/query_results.jsonl"
    assert str(args.quality_gate_result) == "reports/paper/dev/quality_gate_result.json"
    assert str(args.reproducibility) == "reports/paper/dev/paper_report/reproducibility.json"
    assert str(args.semantic_smoke) == "reports/paper/dev/semantic_smoke.json"
    assert args.fail_on_gap is True


def test_eval_eduvidqa_cli_accepts_diagnostic_options() -> None:
    args = build_parser().parse_args(
        [
            "eval-eduvidqa",
            "--input",
            "eduvidqa.jsonl",
            "--index",
            "edu_segments",
            "--diagnostic-top-k",
            "50",
            "--allow-private-output",
        ]
    )

    assert str(args.input) == "eduvidqa.jsonl"
    assert args.index == "edu_segments"
    assert args.diagnostic_top_k == 50
    assert args.allow_private_output is True


def test_lecture_smoke_cli_accepts_manifest_and_output_dir() -> None:
    args = build_parser().parse_args(
        [
            "lecture-smoke",
            "--manifest",
            "reports/lecture_smoke/manifest.json",
            "--output-dir",
            "reports/lecture_smoke/dev",
            "--private-output-dir",
            "/tmp/private_lecture_smoke",
            "--allow-private-output",
        ]
    )

    assert str(args.manifest) == "reports/lecture_smoke/manifest.json"
    assert str(args.output_dir) == "reports/lecture_smoke/dev"
    assert str(args.private_output_dir) == "/tmp/private_lecture_smoke"
    assert args.allow_private_output is True


def test_evidence_unit_smoke_cli_accepts_manifest_output_and_dry_run() -> None:
    args = build_parser().parse_args(
        [
            "evidence-unit-smoke",
            "--manifest",
            "reports/evidence_unit_smoke/manifest.json",
            "--output-dir",
            "reports/evidence_unit_smoke/dev",
            "--dry-run",
        ]
    )

    assert str(args.manifest) == "reports/evidence_unit_smoke/manifest.json"
    assert str(args.output_dir) == "reports/evidence_unit_smoke/dev"
    assert args.dry_run is True


def test_cmd_evidence_unit_smoke_forwards_inputs(monkeypatch, capsys) -> None:
    calls = {}
    fake_client = object()

    class FakeRun:
        run_id = "run1"
        output_dir = Path("reports/run1")
        metrics_path = Path("reports/run1/metrics.json")
        query_results_path = Path("reports/run1/query_results.jsonl")
        summary_path = Path("reports/run1/summary.md")

    def fake_run_evidence_unit_smoke(**kwargs):
        calls["run_kwargs"] = kwargs
        return FakeRun()

    monkeypatch.setattr("oarag.cli.client_from_args", lambda args: fake_client)
    monkeypatch.setattr("oarag.cli.run_evidence_unit_smoke", fake_run_evidence_unit_smoke)

    args = build_parser().parse_args(
        [
            "evidence-unit-smoke",
            "--manifest",
            "manifest.json",
            "--output-dir",
            "reports/public",
            "--dry-run",
        ]
    )

    args.func(args)

    assert calls["run_kwargs"]["client"] is fake_client
    assert calls["run_kwargs"]["manifest_path"].as_posix() == "manifest.json"
    assert calls["run_kwargs"]["output_dir"].as_posix() == "reports/public"
    assert calls["run_kwargs"]["dry_run"] is True
    assert json.loads(capsys.readouterr().out)["summary"] == "reports/run1/summary.md"
