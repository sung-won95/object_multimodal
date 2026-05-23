from oarag.frame_selection import (
    FrameSelectionConfig,
    analyze_pixels,
    select_representative_frames,
    summarize_frame_temporal_coverage,
)


def test_representative_frame_selection_drops_blank_and_duplicate_frames() -> None:
    frames = [
        _frame("blank"),
        _frame("slide_a"),
        _frame("slide_a_duplicate"),
        _frame("slide_b"),
    ]
    checker = bytes([0, 255, 0, 255] * 4)
    pixels_by_id = {
        "blank": bytes([8] * 16),
        "slide_a": checker,
        "slide_a_duplicate": checker,
        "slide_b": bytes([255, 0, 255, 0] * 4),
    }

    result = select_representative_frames(
        frames,
        config=FrameSelectionConfig(strategy="representative"),
        analyzer=_pixel_analyzer(pixels_by_id),
    )

    assert [frame["frame_id"] for frame in result["frames"]] == ["slide_a", "slide_b"]
    assert result["dropped_frame_paths"] == ["blank.jpg", "slide_a_duplicate.jpg"]
    assert result["summary"]["candidate_frame_count"] == 4
    assert result["summary"]["selected_frame_count"] == 2
    assert result["summary"]["drop_reasons"] == {
        "near_duplicate": 1,
        "low_information": 1,
    }
    assert result["summary"]["tradeoff"]["frame_reduction_ratio"] == 0.5
    assert result["frames"][0]["selection"]["strategy"] == "representative"
    assert result["frames"][0]["selection"]["detail_score"] > 0


def test_representative_frame_selection_keeps_best_fallback_when_all_frames_drop() -> None:
    frames = [_frame("black"), _frame("gray")]
    pixels_by_id = {
        "black": bytes([0] * 16),
        "gray": bytes([24] * 16),
    }

    result = select_representative_frames(
        frames,
        config=FrameSelectionConfig(strategy="representative"),
        analyzer=_pixel_analyzer(pixels_by_id),
    )

    assert [frame["frame_id"] for frame in result["frames"]] == ["black"]
    assert result["frames"][0]["selection"]["fallback_kept"] is True
    assert result["summary"]["tradeoff"]["fallback_kept"] is True
    assert result["summary"]["tradeoff"]["recall_risk"] == "high"


def test_none_frame_selection_leaves_frames_unchanged_without_analysis() -> None:
    frames = [_frame("frame_000001")]

    def fail_if_called(_frame):
        raise AssertionError("analyzer should not be called")

    result = select_representative_frames(
        frames,
        config=FrameSelectionConfig(strategy="none"),
        analyzer=fail_if_called,
    )

    assert result["frames"] == frames
    assert result["summary"]["enabled"] is False
    assert result["summary"]["selected_frame_count"] == 1


def test_analyze_pixels_reports_contrast_and_detail_signal() -> None:
    signal = analyze_pixels(
        frame=_frame("checker"),
        pixels=bytes([0, 255, 0, 255] * 4),
        width=4,
        height=4,
    )

    assert signal.brightness == 0.5
    assert signal.contrast > 0.49
    assert signal.detail_score > 0.49


def test_frame_temporal_coverage_warns_when_cap_leaves_late_segments_empty() -> None:
    summary = summarize_frame_temporal_coverage(
        frames=[{"frame_id": "early", "timestamp": 2.0}],
        duration_sec=100.0,
        frame_rate=1.0,
        segments=[
            {"segment_id": "seg_early", "start_time": 0.0, "end_time": 5.0},
            {"segment_id": "seg_late", "start_time": 90.0, "end_time": 95.0},
        ],
    )

    assert summary["temporal_coverage_ratio"] == 0.03
    assert summary["frame_free_segment_ratio"] == 0.5
    assert [warning["code"] for warning in summary["warnings"]] == [
        "low_temporal_coverage",
        "high_frame_free_segment_ratio",
    ]


def _frame(frame_id: str) -> dict:
    return {
        "frame_id": frame_id,
        "frame_path": f"{frame_id}.jpg",
        "timestamp": 0.0,
    }


def _pixel_analyzer(pixels_by_id: dict[str, bytes]):
    def analyzer(frame: dict):
        return analyze_pixels(
            frame=frame,
            pixels=pixels_by_id[frame["frame_id"]],
            width=4,
            height=4,
        )

    return analyzer
