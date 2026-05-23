from oarag.eval import candidate_diagnostics, evaluate_query, summarize
from oarag.schemas import EduVidQARecord, SearchCandidate


def test_evaluate_query_hits_any_gold_timestamp() -> None:
    record = EduVidQARecord(
        dataset_name="eduvidqa",
        subset_name="mathsc_timestamp",
        split_name="test",
        sample_index=0,
        sample_id="sample",
        video_name="video",
        question="question",
        answer="answer",
        transcript_text="text",
        timestamp_points=[100.0, 150.0],
        has_timestamp=True,
        qid=None,
        raw_keys=[],
    )
    candidate = SearchCandidate(
        rank=1,
        segment_id="seg",
        sample_id="other",
        video_id="video",
        start_time=90.0,
        end_time=110.0,
        timestamp_center=107.0,
        transcript_excerpt="text",
        score=None,
    )

    result = evaluate_query(record, [candidate], deltas=[5, 10])

    assert result.hit_by_delta[5] is False
    assert result.hit_by_delta[10] is True
    assert result.best_abs_error == 7.0
    assert result.recall_by_k[1] is False


def test_candidate_diagnostics_are_sanitized_and_mark_source_match() -> None:
    record = _record(sample_id="sample", video_name="video", timestamp_points=[100.0])
    candidates = [
        _candidate(
            rank=1,
            segment_id="wrong",
            sample_id="other",
            video_id="video",
            timestamp_center=90.0,
            transcript_excerpt="SECRET transcript",
        ),
        _candidate(
            rank=2,
            segment_id="gold",
            sample_id="sample",
            video_id="video",
            timestamp_center=103.0,
            transcript_excerpt="SECRET gold transcript",
        ),
    ]

    rows = candidate_diagnostics(record, candidates, top_k=2)

    assert rows == [
        {
            "rank": 1,
            "segment_id": "wrong",
            "sample_id": "other",
            "video_id": "video",
            "timestamp_center": 90.0,
            "abs_error": 10.0,
            "same_sample": False,
            "same_video": True,
        },
        {
            "rank": 2,
            "segment_id": "gold",
            "sample_id": "sample",
            "video_id": "video",
            "timestamp_center": 103.0,
            "abs_error": 3.0,
            "same_sample": True,
            "same_video": True,
        },
    ]
    assert "transcript_excerpt" not in rows[0]


def test_summarize_includes_topk_recall_and_error_percentiles() -> None:
    records = [
        _record(sample_id="s1", video_name="v1", timestamp_points=[10.0]),
        _record(sample_id="s2", video_name="v2", timestamp_points=[20.0]),
    ]
    evals = [
        evaluate_query(
            records[0],
            [_candidate(rank=1, sample_id="s1", video_id="v1", timestamp_center=11.0)],
            deltas=[5],
        ),
        evaluate_query(
            records[1],
            [
                _candidate(rank=1, sample_id="other", video_id="v2", timestamp_center=40.0),
                _candidate(rank=2, sample_id="s2", video_id="v2", timestamp_center=23.0),
            ],
            deltas=[5],
        ),
    ]

    summary = summarize(evals, deltas=[5])

    assert summary["top1_recall"] == 0.5
    assert summary["top5_recall"] == 1.0
    assert summary["top10_recall"] == 1.0
    assert summary["top50_recall"] == 1.0
    assert summary["median_abs_error"] == 1.0
    assert summary["p75_abs_error"] == 3.0
    assert summary["p90_abs_error"] == 3.0


def _record(
    *,
    sample_id: str,
    video_name: str,
    timestamp_points: list[float],
) -> EduVidQARecord:
    return EduVidQARecord(
        dataset_name="eduvidqa",
        subset_name="mathsc_timestamp",
        split_name="test",
        sample_index=0,
        sample_id=sample_id,
        video_name=video_name,
        question="question",
        answer="answer",
        transcript_text="text",
        timestamp_points=timestamp_points,
        has_timestamp=True,
        qid=None,
        raw_keys=[],
    )


def _candidate(
    *,
    rank: int,
    segment_id: str = "seg",
    sample_id: str = "sample",
    video_id: str = "video",
    timestamp_center: float | None = 100.0,
    transcript_excerpt: str = "text",
) -> SearchCandidate:
    return SearchCandidate(
        rank=rank,
        segment_id=segment_id,
        sample_id=sample_id,
        video_id=video_id,
        start_time=None,
        end_time=None,
        timestamp_center=timestamp_center,
        transcript_excerpt=transcript_excerpt,
        score=None,
    )
