from oarag.cli import build_parser


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


def test_align_frames_cli_defaults() -> None:
    args = build_parser().parse_args(["align-frames", "--project-id", "sample"])

    assert args.project_id == "sample"
    assert args.output_root.as_posix() == "artifacts/projects"
    assert args.margin_seconds == 0.0
