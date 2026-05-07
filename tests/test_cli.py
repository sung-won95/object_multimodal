import json
from pathlib import Path

from oarag.cli import (
    build_parser,
    build_vlm_alignment_parser,
    cmd_index_project,
    parse_vlm_options,
)
from oarag.meili import (
    LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
    LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE,
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
    assert json.loads(capsys.readouterr().out)["settings_profile"] == (
        LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE
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
    assert args.backend == "auto"
    assert args.vlm_jsonl is None
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
        ]
    )

    assert args.rerank is True
    assert args.rerank_time_hint == "10-14s"


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
        ]
    )

    assert str(args.manifest) == "benchmarks/retrieval.json"
    assert str(args.output_dir) == "reports/perf_runs/dev"
