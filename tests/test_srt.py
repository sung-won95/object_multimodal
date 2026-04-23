from pathlib import Path

from oarag.srt import iter_srt_cues, parse_timestamp


def test_parse_timestamp() -> None:
    assert parse_timestamp("00:01:02,500") == 62.5
    assert parse_timestamp("01:00:00.000") == 3600.0


def test_iter_srt_cues(tmp_path: Path) -> None:
    srt = tmp_path / "sample.srt"
    srt.write_text(
        """1
00:00:01,000 --> 00:00:02,500
첫 번째 줄
두 번째 줄

2
00:00:03,000 --> 00:00:04,000
다음 자막
""",
        encoding="utf-8",
    )

    cues = list(iter_srt_cues(srt))

    assert len(cues) == 2
    assert cues[0].start_time == 1.0
    assert cues[0].end_time == 2.5
    assert cues[0].text == "첫 번째 줄 두 번째 줄"

