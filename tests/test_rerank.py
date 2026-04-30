from __future__ import annotations

from oarag.rerank import parse_timestamp_hints, rerank_bundles


def test_parse_timestamp_hints_extracts_ranges_and_points() -> None:
    assert parse_timestamp_hints("compare the board around 10-14s") == [(10.0, 14.0)]
    assert parse_timestamp_hints("show the slide at 22초") == [(22.0, 22.0)]


def test_rerank_bundles_uses_general_evidence_signals() -> None:
    bundles = [
        _bundle(
            rank=1,
            segment_id="seg_ranked_first",
            text="introductory aside",
            start_time=58.0,
            end_time=62.0,
            score=0.95,
            frame_refs=[],
            linked_score=None,
        ),
        _bundle(
            rank=2,
            segment_id="seg_evidence_rich",
            text="bet size appears on the board",
            start_time=10.0,
            end_time=14.0,
            score=0.4,
            frame_refs=["frame_000012"],
            linked_score=1.2,
        ),
    ]

    reranked, metadata = rerank_bundles(
        bundles=bundles,
        query="bet size board 10-14s",
        timestamp_hint=None,
    )

    assert metadata["enabled"] is True
    assert metadata["strategy"] == "deterministic_evidence_v1"
    assert [bundle["candidate"]["segment_id"] for bundle in reranked] == [
        "seg_evidence_rich",
        "seg_ranked_first",
    ]
    top = reranked[0]["rerank"]
    assert top["rank"] == 1
    assert top["original_rank"] == 2
    assert top["breakdown"]["signals"]["query_overlap"]["matched_terms"] == [
        "bet",
        "board",
        "size",
    ]
    assert top["breakdown"]["signals"]["timestamp_proximity"]["distance_seconds"] == 0.0
    assert top["breakdown"]["signals"]["frame_backed"]["score"] == 1.0
    assert top["breakdown"]["signals"]["entity_link"]["max_link_score"] == 1.2
    assert "query_overlap" in top["explanation"]


def _bundle(
    *,
    rank: int,
    segment_id: str,
    text: str,
    start_time: float,
    end_time: float,
    score: float,
    frame_refs: list[str],
    linked_score: float | None,
) -> dict:
    linked_entities = []
    if linked_score is not None:
        linked_entities.append(
            {
                "score": linked_score,
                "entity": {"entity_id": f"ent_{segment_id}", "text": "bet size board"},
            }
        )
    frame_objects = [
        {"frame_id": frame_id, "timestamp": start_time, "frame_path": f"/tmp/{frame_id}.jpg"}
        for frame_id in frame_refs
    ]
    return {
        "rank": rank,
        "candidate": {
            "rank": rank,
            "segment_id": segment_id,
            "sample_id": segment_id,
            "video_id": "video",
            "start_time": start_time,
            "end_time": end_time,
            "timestamp_center": (start_time + end_time) / 2,
            "transcript_excerpt": text,
            "score": score,
        },
        "evidence_window": {
            "target_segment": {
                "segment_id": segment_id,
                "transcript_text": text,
                "frame_refs": frame_refs,
            },
            "transcript_segments": [{"segment_id": segment_id, "transcript_text": text}],
            "frame_refs": frame_objects,
        },
        "visual_entities": [],
        "linked_entities": linked_entities,
        "summary": {"text": f"#{rank} video: {text}"},
    }
