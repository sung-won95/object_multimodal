from pathlib import Path

from oarag.ingest import choose_transcript_source
from oarag.stt import stt_segments_from_mlx_result


def test_choose_transcript_source_prefers_srt_in_auto() -> None:
    assert choose_transcript_source("auto", srt_path=Path("sample.srt")) == "srt"
    assert choose_transcript_source("auto", srt_path=None) == "stt"
    assert choose_transcript_source("none", srt_path=Path("sample.srt")) == "none"


def test_stt_segments_from_mlx_result_normalizes_segments() -> None:
    result = {
        "segments": [
            {"start": 1, "end": 2.5, "text": " 안녕하세요\n여러분 "},
            {"start": 5, "end": 4, "text": "끝"},
            {"start": 9, "end": 10, "text": "   "},
        ]
    }

    segments = stt_segments_from_mlx_result(result)

    assert len(segments) == 2
    assert segments[0].seq_no == 1
    assert segments[0].start_time == 1.0
    assert segments[0].end_time == 2.5
    assert segments[0].text == "안녕하세요 여러분"
    assert segments[1].seq_no == 2
    assert segments[1].start_time == 5.0
    assert segments[1].end_time == 5.0


def test_stt_segments_from_mlx_result_falls_back_to_text() -> None:
    segments = stt_segments_from_mlx_result({"text": "전체 transcript"})

    assert len(segments) == 1
    assert segments[0].text == "전체 transcript"
