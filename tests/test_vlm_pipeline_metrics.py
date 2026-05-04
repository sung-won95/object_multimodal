from oarag.vlm_pipeline_metrics import extract_vlm_smoke_metrics


def test_extract_vlm_smoke_metrics_returns_compact_pipeline_metrics() -> None:
    summary = {
        "status": "completed_with_errors",
        "elapsed_seconds": 12.345,
        "stages": {
            "frame_candidates": {
                "summary": {
                    "counts": {
                        "input_frame_count": 10,
                        "selected_candidate_count": 4,
                    },
                    "tradeoff": {"estimated_vlm_cost_reduction_ratio": 0.6},
                }
            },
            "visual_observations": {
                "elapsed_seconds": 2.5,
                "average_frame_latency_seconds": 0.25,
                "counts": {
                    "frames_total": 4,
                    "frames_processed": 3,
                    "frames_failed": 1,
                    "frames_skipped_resumed": 1,
                },
                "frame_status_counts": {"success": 3, "backend_failure": 1},
            },
            "audio_visual_consistency": {
                "counts": {"audio_visual_consistency": 4},
                "status_counts": {"success": 3, "source_failure": 1},
                "consistency_counts": {"aligned": 2, "uncertain": "1", "mismatch": 1},
            },
        },
    }

    metrics = extract_vlm_smoke_metrics(summary)

    assert metrics == {
        "pipeline_status": "completed_with_errors",
        "pipeline_elapsed_seconds": 12.345,
        "candidate_frame_reduction_ratio": 0.6,
        "input_frame_count": 10,
        "selected_candidate_count": 4,
        "frames_total": 4,
        "frames_processed": 3,
        "frames_failed": 1,
        "frames_skipped": 1,
        "average_frame_latency_seconds": 0.25,
        "visual_observation_elapsed_seconds": 2.5,
        "frame_status_counts": {"success": 3, "backend_failure": 1},
        "audio_visual_consistency_records": 4,
        "audio_visual_status_counts": {"success": 3, "source_failure": 1},
        "audio_visual_consistency_counts": {
            "aligned": 2,
            "uncertain": 1,
            "mismatch": 1,
        },
    }


def test_extract_vlm_smoke_metrics_falls_back_to_counts_for_reduction_ratio() -> None:
    summary = {
        "stages": {
            "frame_candidates": {
                "counts": {
                    "input_frame_count": "5",
                    "vlm_frame_candidates": "2",
                }
            }
        }
    }

    metrics = extract_vlm_smoke_metrics(summary)

    assert metrics["pipeline_status"] == ""
    assert metrics["pipeline_elapsed_seconds"] is None
    assert metrics["candidate_frame_reduction_ratio"] == 0.6
    assert metrics["input_frame_count"] == 5
    assert metrics["selected_candidate_count"] == 2
