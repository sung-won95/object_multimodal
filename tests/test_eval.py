from oarag.eval import evaluate_query
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

