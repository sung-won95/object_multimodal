import json
from pathlib import Path

from oarag.cli import build_parser
from oarag.project_query import query_project


PUBLIC_USER_PROVIDED_VECTOR_PROJECT = (
    Path(__file__).parent / "fixtures" / "public_userprovided_vector_project"
)


class FakeClient:
    def __init__(self, hits: list[dict], processing_time_ms: int = 7) -> None:
        self.hits = hits
        self.processing_time_ms = processing_time_ms
        self.queries: list[str] = []

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
    ) -> dict:
        self.queries.append(query)
        return {
            "hits": self.hits[:limit],
            "processingTimeMs": self.processing_time_ms,
            "indexUid": index_uid,
            "query": query,
        }


class MultiIndexFakeClient:
    def __init__(self, hits_by_index: dict[str, list[dict]]) -> None:
        self.hits_by_index = hits_by_index
        self.searches: list[tuple[str, str, int, str | list[str] | None]] = []

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
    ) -> dict:
        self.searches.append((index_uid, query, limit, filter))
        return {
            "hits": self.hits_by_index.get(index_uid, [])[:limit],
            "processingTimeMs": 3,
            "indexUid": index_uid,
            "query": query,
        }


class QueryAwareFakeClient:
    def __init__(self, hits_by_query: dict[str, list[dict]]) -> None:
        self.hits_by_query = hits_by_query
        self.queries: list[str] = []

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
    ) -> dict:
        self.queries.append(query)
        return {
            "hits": self.hits_by_query.get(query, [])[:limit],
            "processingTimeMs": 2,
            "indexUid": index_uid,
            "query": query,
        }


class HybridFakeClient:
    def __init__(self, hits_by_call: dict[tuple[str, str, str], list[dict]]) -> None:
        self.hits_by_call = hits_by_call
        self.searches: list[tuple[str, str, int, str, dict | None]] = []

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        hybrid: dict | None = None,
        filter: str | list[str] | None = None,
    ) -> dict:
        mode = "semantic" if hybrid else "lexical"
        self.searches.append((index_uid, query, limit, mode, hybrid))
        return {
            "hits": self.hits_by_call.get((index_uid, query, mode), [])[:limit],
            "processingTimeMs": 2,
            "indexUid": index_uid,
            "query": query,
        }


class HybridVectorFakeClient:
    def __init__(self, hits_by_call: dict[tuple[str, str, str], list[dict]]) -> None:
        self.hits_by_call = hits_by_call
        self.searches: list[dict] = []

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        hybrid: dict | None = None,
        vector: list[float] | None = None,
        filter: str | list[str] | None = None,
    ) -> dict:
        mode = "semantic" if hybrid else "lexical"
        self.searches.append(
            {
                "index": index_uid,
                "query": query,
                "limit": limit,
                "mode": mode,
                "hybrid": hybrid,
                "vector": vector,
            }
        )
        return {
            "hits": self.hits_by_call.get((index_uid, query, mode), [])[:limit],
            "processingTimeMs": 2,
            "indexUid": index_uid,
            "query": query,
        }


class LegacyMultiIndexNoFilterClient:
    def __init__(self, hits_by_index: dict[str, list[dict]]) -> None:
        self.hits_by_index = hits_by_index
        self.searches: list[tuple[str, str, int]] = []

    def search(self, index_uid: str, query: str, limit: int = 10) -> dict:
        self.searches.append((index_uid, query, limit))
        return {
            "hits": self.hits_by_index.get(index_uid, [])[:limit],
            "processingTimeMs": 2,
            "indexUid": index_uid,
            "query": query,
        }


class LegacyHybridVectorNoFilterClient:
    def __init__(self, hits_by_call: dict[tuple[str, str, str], list[dict]]) -> None:
        self.hits_by_call = hits_by_call
        self.searches: list[dict] = []

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        hybrid: dict | None = None,
        vector: list[float] | None = None,
    ) -> dict:
        mode = "semantic" if hybrid else "lexical"
        self.searches.append(
            {
                "index": index_uid,
                "query": query,
                "limit": limit,
                "mode": mode,
                "hybrid": hybrid,
                "vector": vector,
            }
        )
        return {
            "hits": self.hits_by_call.get((index_uid, query, mode), [])[:limit],
            "processingTimeMs": 2,
            "indexUid": index_uid,
            "query": query,
        }


def test_query_project_returns_multimodal_bundle(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_1", 1, 0.0, 4.0, "Intro", ["frame_000001"]),
            _segment("seg_2", 2, 5.0, 9.0, "Bet size is on this board", ["frame_000006", "frame_000008"]),
            _segment("seg_3", 3, 10.0, 14.0, "Next street", ["frame_000012"]),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {"frame_id": "frame_000001", "timestamp": 0.0, "frame_path": "/tmp/f1.jpg"},
            {"frame_id": "frame_000006", "timestamp": 5.0, "frame_path": "/tmp/f6.jpg"},
            {"frame_id": "frame_000008", "timestamp": 7.0, "frame_path": "/tmp/f8.jpg"},
            {"frame_id": "frame_000012", "timestamp": 11.0, "frame_path": "/tmp/f12.jpg"},
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "entity_board",
                "project_id": "project",
                "frame_id": "frame_000006",
                "timestamp": 5.0,
                "frame_path": "/tmp/f6.jpg",
                "bbox": None,
                "text": "bet size board",
                "entity_type": "ocr_text",
                "confidence": 0.91,
                "source": "local-ocr",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_seg_2_entity_board",
                "project_id": "project",
                "segment_id": "seg_2",
                "entity_id": "entity_board",
                "frame_id": "frame_000006",
                "link_type": "time_overlap+lexical_match+mention_candidate",
                "score": 1.35,
                "evidence": ["time_overlap", "lexical_match", "mention_candidate"],
                "time_overlap": True,
                "lexical_match": ["bet", "board", "size"],
                "mention_candidate": ["board"],
                "score_breakdown": {
                    "time_overlap": 0.2,
                    "visual_text_match": 0.6,
                    "mention_candidate": 0.45,
                    "domain_lexicon_match": 0.1,
                },
                "reason_metadata": {
                    "domain_lexicon_matches_by_field": {"text": ["bet"]},
                    "reference_cues_by_field": {"position": ["this", "center"]},
                },
            }
        ],
    )

    response = query_project(
        client=FakeClient(
            hits=[
                {
                    "segment_id": "seg_2",
                    "sample_id": "seg_2",
                    "video_id": "video",
                    "start_time": 5.0,
                    "end_time": 9.0,
                    "timestamp_center": 7.0,
                    "transcript_text": "Bet size is on this board",
                    "_rankingScore": 0.87,
                }
            ]
        ),
        index_uid="local_segments",
        project_dir=project_dir,
        query="bet size",
        neighbor_count=1,
    )

    assert response["artifact_availability"] == {
        "frames_manifest": True,
        "visual_entities": True,
        "entity_links": True,
        "domain_lexicon": False,
    }
    assert response["counts"]["search_hits"] == 1
    assert response["counts"]["bundles"] == 1
    assert response["retrieval_context"]["window_config"] == {
        "mode": "neighbors",
        "neighbor_count": 1,
        "previous_neighbor_count": 1,
        "next_neighbor_count": 1,
    }
    bundle = response["bundles"][0]
    assert bundle["candidate"]["segment_id"] == "seg_2"
    assert bundle["evidence_window"]["target_segment"]["segment_id"] == "seg_2"
    assert [segment["segment_id"] for segment in bundle["evidence_window"]["neighbor_segments"]] == [
        "seg_1",
        "seg_3",
    ]
    assert [segment["segment_id"] for segment in bundle["evidence_window"]["transcript_segments"]] == [
        "seg_1",
        "seg_2",
        "seg_3",
    ]
    assert [frame["frame_path"] for frame in bundle["evidence_window"]["frame_refs"]] == [
        "/tmp/f1.jpg",
        "/tmp/f6.jpg",
        "/tmp/f8.jpg",
        "/tmp/f12.jpg",
    ]
    assert bundle["visual_entities"][0]["entity_id"] == "entity_board"
    assert bundle["linked_entities"][0]["entity"]["text"] == "bet size board"
    assert "shared terms: bet, board, size" in bundle["linked_entities"][0]["explanation"]
    assert "mention/entity hint: board" in bundle["linked_entities"][0]["explanation"]
    assert "domain alias: bet" in bundle["linked_entities"][0]["explanation"]
    assert "reference/position cue: center, this" in bundle["linked_entities"][0]["explanation"]
    assert "score components:" in bundle["linked_entities"][0]["explanation"]
    assert bundle["linked_entities"][0]["score_breakdown"]["domain_lexicon_match"] == 0.1
    assert bundle["linked_entities"][0]["reason_metadata"]["reference_cues_by_field"] == {
        "position": ["this", "center"]
    }
    assert bundle["summary"]["frame_paths"] == ["/tmp/f1.jpg", "/tmp/f6.jpg", "/tmp/f8.jpg", "/tmp/f12.jpg"]
    assert "bet size board" in response["summary_lines"][0]


def test_query_project_without_visual_artifacts_falls_back_to_transcript_and_frames(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_1", 1, 0.0, 2.0, "Only transcript evidence", ["frame_000001"])],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000001", "timestamp": 0.0, "frame_path": "/tmp/f1.jpg"}],
    )

    response = query_project(
        client=FakeClient(
            hits=[
                {
                    "segment_id": "seg_1",
                    "sample_id": "seg_1",
                    "video_id": "video",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "timestamp_center": 1.0,
                    "transcript_text": "Only transcript evidence",
                }
            ]
        ),
        index_uid="local_segments",
        project_dir=project_dir,
        query="transcript evidence",
        neighbor_count=0,
    )

    assert response["artifact_availability"] == {
        "frames_manifest": True,
        "visual_entities": False,
        "entity_links": False,
        "domain_lexicon": False,
    }
    assert response["warnings"] == []
    bundle = response["bundles"][0]
    assert bundle["visual_entities"] == []
    assert bundle["linked_entities"] == []
    assert bundle["evidence_window"]["frame_refs"] == [
        {"frame_id": "frame_000001", "timestamp": 0.0, "frame_path": "/tmp/f1.jpg"}
    ]
    assert "linked_entities=none" in bundle["summary"]["text"]


def test_query_project_maps_window_hit_to_target_segment_bundle(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_1", 1, 0.0, 2.0, "Intro context", ["frame_000001"]),
            _segment("seg_2", 2, 3.0, 5.0, "Target range grid", ["frame_000002"]),
            _segment("seg_3", 3, 6.0, 8.0, "Next context", ["frame_000003"]),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {"frame_id": "frame_000001", "timestamp": 1.0, "frame_path": "/tmp/f1.jpg"},
            {"frame_id": "frame_000002", "timestamp": 4.0, "frame_path": "/tmp/f2.jpg"},
            {"frame_id": "frame_000003", "timestamp": 7.0, "frame_path": "/tmp/f3.jpg"},
        ],
    )

    response = query_project(
        client=FakeClient(
            hits=[
                {
                    "window_id": "window_seg_2_abc123",
                    "target_segment_id": "seg_2",
                    "sample_id": "seg_2",
                    "video_id": "video",
                    "start_time": 0.0,
                    "end_time": 8.0,
                    "timestamp_center": 4.0,
                    "target_start_time": 3.0,
                    "target_end_time": 5.0,
                    "target_timestamp_center": 4.0,
                    "source_segment_ids": ["seg_1", "seg_2", "seg_3"],
                    "transcript_window_text": "Intro context Target range grid Next context",
                    "semantic_source_fields": ["transcript_window_text"],
                    "_rankingScore": 0.91,
                }
            ]
        ),
        index_uid="local_windows",
        retrieval_index_kind="window",
        project_dir=project_dir,
        query="range grid",
        neighbor_count=0,
    )

    assert response["index_kind"] == "window"
    assert response["counts"]["window_search_hits"] == 1
    assert response["counts"]["segment_search_hits"] == 0
    assert response["retrieval_context"]["indexes"]["window"] == "local_windows"
    assert response["retrieval_context"]["searches"]["window"]["hit_count"] == 1
    bundle = response["bundles"][0]
    assert bundle["candidate"]["source"] == "window"
    assert bundle["candidate"]["segment_id"] == "seg_2"
    assert bundle["candidate"]["window_id"] == "window_seg_2_abc123"
    assert bundle["retrieval_sources"][0]["source"] == "window"
    assert bundle["retrieval_sources"][0]["source_segment_ids"] == ["seg_1", "seg_2", "seg_3"]
    assert bundle["evidence_window"]["target_segment"]["segment_id"] == "seg_2"


def test_query_project_merges_visual_entity_hits_with_segment_targets(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_1", 1, 0.0, 4.0, "Intro", ["frame_000001"]),
            _segment("seg_2", 2, 5.0, 9.0, "The range grid is on screen", ["frame_000006"]),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {"frame_id": "frame_000001", "timestamp": 0.0, "frame_path": "/tmp/f1.jpg"},
            {"frame_id": "frame_000006", "timestamp": 6.0, "frame_path": "/tmp/f6.jpg"},
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            _visual_entity(
                "entity_grid",
                "frame_000006",
                "range grid",
                6.0,
                frame_path="/tmp/f6.jpg",
            )
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_seg_2_entity_grid",
                "project_id": "project",
                "segment_id": "seg_2",
                "entity_id": "entity_grid",
                "frame_id": "frame_000006",
                "link_type": "time_overlap+lexical_match",
                "score": 1.25,
                "evidence": ["time_overlap", "lexical_match"],
                "time_overlap": True,
                "lexical_match": ["range", "grid"],
                "mention_candidate": [],
            }
        ],
    )
    client = MultiIndexFakeClient(
        {
            "local_segments": [
                {
                    "segment_id": "seg_2",
                    "sample_id": "seg_2",
                    "video_id": "video",
                    "start_time": 5.0,
                    "end_time": 9.0,
                    "timestamp_center": 7.0,
                    "transcript_text": "The range grid is on screen",
                    "_rankingScore": 0.77,
                }
            ],
            "local_visual_entities": [
                {
                    **_visual_entity(
                        "entity_grid",
                        "frame_000006",
                        "range grid",
                        6.0,
                        frame_path="/tmp/f6.jpg",
                    ),
                    "entity_id": "project__entity_grid",
                    "local_entity_id": "entity_grid",
                    "_rankingScore": 0.93,
                }
            ],
        }
    )

    response = query_project(
        client=client,
        index_uid="local_segments",
        visual_index_uid="local_visual_entities",
        project_dir=project_dir,
        query="range grid",
        neighbor_count=0,
    )

    assert client.searches == [
        ("local_segments", "range grid", 5, 'project_id = "project"'),
        ("local_visual_entities", "range grid", 5, 'project_id = "project"'),
    ]
    assert [candidate["source"] for candidate in response["candidates"]] == [
        "segment",
        "visual_entity",
    ]
    assert response["counts"]["search_hits"] == 2
    assert response["counts"]["segment_search_hits"] == 1
    assert response["counts"]["visual_entity_search_hits"] == 1
    assert response["counts"]["bundles"] == 1
    bundle = response["bundles"][0]
    assert bundle["candidate"]["source"] == "segment"
    assert bundle["merge"] == {
        "target_key": "segment:seg_2",
        "target_segment_id": "seg_2",
        "source_count": 2,
        "selected_source": "segment",
        "selected_rank": 1,
        "deduplicated": True,
    }
    assert [source["source"] for source in bundle["retrieval_sources"]] == [
        "segment",
        "visual_entity",
    ]
    assert bundle["retrieval_sources"][1]["target_resolution"]["method"] == "entity_link"
    assert bundle["visual_entities"][0]["entity_id"] == "entity_grid"


def test_query_project_builds_bundle_from_visual_entity_only_hit(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment(
                "seg_1",
                1,
                5.0,
                9.0,
                "Look at this grid",
                [{"frame_id": "frame_000006"}],
            )
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000006", "timestamp": 6.0, "frame_path": "/tmp/f6.jpg"}],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            _visual_entity(
                "entity_grid",
                "frame_000006",
                "range grid",
                6.0,
                frame_path="/tmp/f6.jpg",
            )
        ],
    )
    client = MultiIndexFakeClient(
        {
            "local_segments": [],
            "local_visual_entities": [
                {
                    **_visual_entity(
                        "entity_grid",
                        "frame_000006",
                        "range grid",
                        6.0,
                        frame_path="/tmp/f6.jpg",
                    ),
                    "entity_id": "project__entity_grid",
                    "local_entity_id": "entity_grid",
                    "_rankingScore": 0.93,
                }
            ],
        }
    )

    response = query_project(
        client=client,
        index_uid="local_segments",
        visual_index_uid="local_visual_entities",
        project_dir=project_dir,
        query="range grid",
        neighbor_count=0,
    )

    assert response["counts"]["bundles"] == 1
    bundle = response["bundles"][0]
    assert bundle["candidate"]["source"] == "visual_entity"
    assert bundle["candidate"]["entity_id"] == "entity_grid"
    assert bundle["candidate"]["segment_id"] == "seg_1"
    assert bundle["evidence_window"]["target_segment"]["segment_id"] == "seg_1"
    assert bundle["retrieval_sources"][0]["target_resolution"]["method"] == "frame_ref"
    assert bundle["evidence_window"]["frame_refs"] == [
        {"frame_id": "frame_000006", "timestamp": 6.0, "frame_path": "/tmp/f6.jpg"}
    ]
    assert bundle["visual_entities"][0]["text"] == "range grid"
    assert "visual_entity=range grid" in response["summary_lines"][0]


def test_query_project_supports_legacy_search_clients_without_filter_kwarg(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_1", 1, 5.0, 9.0, "Look at this grid", ["frame_000006"])],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000006", "timestamp": 6.0, "frame_path": "/tmp/f6.jpg"}],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            _visual_entity(
                "entity_grid",
                "frame_000006",
                "range grid",
                6.0,
                frame_path="/tmp/f6.jpg",
            )
        ],
    )
    client = LegacyMultiIndexNoFilterClient(
        {
            "local_segments": [
                {
                    "segment_id": "seg_1",
                    "sample_id": "seg_1",
                    "video_id": "video",
                    "start_time": 5.0,
                    "end_time": 9.0,
                    "timestamp_center": 7.0,
                    "transcript_text": "Look at this grid",
                    "_rankingScore": 0.82,
                }
            ],
            "local_visual_entities": [
                {
                    **_visual_entity(
                        "entity_grid",
                        "frame_000006",
                        "range grid",
                        6.0,
                        frame_path="/tmp/f6.jpg",
                    ),
                    "entity_id": "project__entity_grid",
                    "local_entity_id": "entity_grid",
                    "_rankingScore": 0.93,
                }
            ],
        }
    )

    response = query_project(
        client=client,
        index_uid="local_segments",
        visual_index_uid="local_visual_entities",
        project_dir=project_dir,
        query="range grid",
        neighbor_count=0,
    )

    assert client.searches == [
        ("local_segments", "range grid", 5),
        ("local_visual_entities", "range grid", 5),
    ]
    assert response["counts"]["search_hits"] == 2
    assert response["counts"]["bundles"] == 1
    assert [candidate["source"] for candidate in response["candidates"]] == [
        "segment",
        "visual_entity",
    ]


def test_query_project_expands_query_when_domain_lexicon_exists(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_1", 1, 0.0, 2.0, "Wager sizing", ["frame_000001"])],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000001", "timestamp": 0.0, "frame_path": "/tmp/f1.jpg"}],
    )
    (project_dir / "domain_lexicon.json").write_text(
        json.dumps({"aliases": {"bet": ["wager"], "size": ["sizing"]}}),
        encoding="utf-8",
    )
    client = MultiIndexFakeClient(
        {
            "local_segments": [
                {
                    "segment_id": "seg_1",
                    "sample_id": "seg_1",
                    "video_id": "video",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "timestamp_center": 1.0,
                    "transcript_text": "Wager sizing",
                }
            ],
            "local_visual_entities": [],
        }
    )

    response = query_project(
        client=client,
        index_uid="local_segments",
        visual_index_uid="local_visual_entities",
        project_dir=project_dir,
        query="wager sizing",
        neighbor_count=0,
    )

    assert client.searches == [
        ("local_segments", "wager sizing", 5, 'project_id = "project"'),
        ("local_segments", "bet", 5, 'project_id = "project"'),
        ("local_segments", "size", 5, 'project_id = "project"'),
        ("local_visual_entities", "wager sizing", 5, 'project_id = "project"'),
        ("local_visual_entities", "bet", 5, 'project_id = "project"'),
        ("local_visual_entities", "size", 5, 'project_id = "project"'),
    ]
    assert response["query"] == "wager sizing"
    assert response["domain_lexicon"]["enabled"] is True
    assert response["domain_lexicon"]["source_path"] == str(
        (project_dir / "domain_lexicon.json").resolve()
    )
    assert response["query_expansion"] == {
        "enabled": True,
        "applied": True,
        "added_term_count": 2,
        "expanded_query": "wager sizing bet size",
        "source_path": str((project_dir / "domain_lexicon.json").resolve()),
        "search_queries": ["wager sizing", "bet", "size"],
        "terms": [
            {
                "term": "bet",
                "source": "domain_lexicon",
                "canonical": "bet",
                "matched_terms": ["wager"],
            },
            {
                "term": "size",
                "source": "domain_lexicon",
                "canonical": "size",
                "matched_terms": ["sizing"],
            },
        ],
    }


def test_query_project_expansion_hits_can_enter_top_limit(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_original", 1, 0.0, 2.0, "Generic introduction", []),
            _segment("seg_alias", 2, 3.0, 5.0, "Bet size explanation", []),
        ],
    )
    (project_dir / "domain_lexicon.json").write_text(
        json.dumps({"aliases": {"bet": ["wager"]}}, ensure_ascii=False),
        encoding="utf-8",
    )
    client = QueryAwareFakeClient(
        {
            "wager": [
                {
                    "segment_id": "seg_original",
                    "sample_id": "seg_original",
                    "video_id": "video",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "timestamp_center": 1.0,
                    "transcript_text": "Generic introduction",
                    "_rankingScore": 0.1,
                }
            ],
            "bet": [
                {
                    "segment_id": "seg_alias",
                    "sample_id": "seg_alias",
                    "video_id": "video",
                    "start_time": 3.0,
                    "end_time": 5.0,
                    "timestamp_center": 4.0,
                    "transcript_text": "Bet size explanation",
                    "_rankingScore": 0.9,
                }
            ],
        }
    )

    response = query_project(
        client=client,
        index_uid="local_segments",
        project_dir=project_dir,
        query="wager",
        limit=1,
        neighbor_count=0,
    )

    assert client.queries == ["wager", "bet"]
    assert response["query_expansion"]["search_queries"] == ["wager", "bet"]
    assert response["retrieval_context"]["searches"]["segment"]["hit_count"] == 1
    assert response["bundles"][0]["candidate"]["segment_id"] == "seg_alias"


def test_query_project_hybrid_retrieval_calls_each_expanded_query_per_channel_and_merges(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_lexical", 1, 0.0, 2.0, "Wager introduction", []),
            _segment("seg_semantic", 2, 3.0, 5.0, "Bet sizing concept", []),
            _segment("seg_alias", 3, 6.0, 8.0, "Alias-only followup", []),
        ],
    )
    (project_dir / "domain_lexicon.json").write_text(
        json.dumps({"aliases": {"bet": ["wager"]}}, ensure_ascii=False),
        encoding="utf-8",
    )
    client = HybridFakeClient(
        {
            ("local_segments", "wager", "lexical"): [
                {
                    "segment_id": "seg_lexical",
                    "sample_id": "seg_lexical",
                    "video_id": "video",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "timestamp_center": 1.0,
                    "transcript_text": "Wager introduction",
                    "_rankingScore": 0.5,
                }
            ],
            ("local_segments", "wager", "semantic"): [
                {
                    "segment_id": "seg_semantic",
                    "sample_id": "seg_semantic",
                    "video_id": "video",
                    "start_time": 3.0,
                    "end_time": 5.0,
                    "timestamp_center": 4.0,
                    "transcript_text": "Bet sizing concept",
                    "_rankingScore": 0.8,
                }
            ],
            ("local_segments", "bet", "lexical"): [
                {
                    "segment_id": "seg_semantic",
                    "sample_id": "seg_semantic",
                    "video_id": "video",
                    "start_time": 3.0,
                    "end_time": 5.0,
                    "timestamp_center": 4.0,
                    "transcript_text": "Bet sizing concept",
                    "_rankingScore": 0.6,
                }
            ],
            ("local_segments", "bet", "semantic"): [
                {
                    "segment_id": "seg_alias",
                    "sample_id": "seg_alias",
                    "video_id": "video",
                    "start_time": 6.0,
                    "end_time": 8.0,
                    "timestamp_center": 7.0,
                    "transcript_text": "Alias-only followup",
                    "_rankingScore": 0.7,
                }
            ],
        }
    )

    response = query_project(
        client=client,
        index_uid="local_segments",
        project_dir=project_dir,
        query="wager",
        limit=2,
        neighbor_count=0,
        hybrid_retrieval=True,
        hybrid_embedder="default",
        hybrid_semantic_ratio=1.0,
    )

    assert client.searches == [
        ("local_segments", "wager", 2, "lexical", None),
        (
            "local_segments",
            "wager",
            2,
            "semantic",
            {"embedder": "default", "semanticRatio": 1.0},
        ),
        ("local_segments", "bet", 2, "lexical", None),
        (
            "local_segments",
            "bet",
            2,
            "semantic",
            {"embedder": "default", "semanticRatio": 1.0},
        ),
    ]
    assert response["query_expansion"]["search_queries"] == ["wager", "bet"]
    assert response["retrieval_context"]["hybrid_retrieval"] == {
        "enabled": True,
        "modes": ["lexical", "semantic"],
        "embedder": "default",
        "semantic_ratio": 1.0,
        "query_vector": {"used": False},
        "fusion": {
            "method": "reciprocal_rank_fusion",
            "rank_constant": 60,
        },
    }
    assert response["retrieval_context"]["searches"]["segment"]["retrieval_modes"] == [
        "lexical",
        "semantic",
    ]
    assert response["retrieval_context"]["searches"]["segment"]["raw_hit_count_before_dedupe"] == 4
    assert [bundle["candidate"]["segment_id"] for bundle in response["bundles"]] == [
        "seg_semantic",
        "seg_alias",
    ]
    top = response["bundles"][0]
    assert top["candidate"]["source"] == "segment"
    assert top["candidate"]["modality"] == "transcript"
    assert top["candidate"]["retrieval_mode"] == "semantic"
    assert top["candidate"]["original_rank"] == 1
    assert [match["retrieval_mode"] for match in top["candidate"]["retrieval_matches"]] == [
        "semantic",
        "lexical",
    ]
    assert [match["query"] for match in top["candidate"]["retrieval_matches"]] == [
        "wager",
        "bet",
    ]
    assert top["merge"]["retrieval_match_count"] == 2
    assert top["merge"]["selected_retrieval_mode"] == "semantic"
    assert top["merge"]["deduplicated"] is True
    assert top["retrieval_sources"][0]["deduplicated"] is True
    assert [match["rank"] for match in top["retrieval_sources"][0]["matches"]] == [1, 1]


def test_query_project_userprovided_vector_hits_semantic_channel_without_leaking_raw_vector(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_vector", 1, 0.0, 2.0, "Public vector target", []),
            _segment("seg_other", 2, 3.0, 5.0, "Unrelated public text", []),
        ],
    )
    query_vector = [0.111111, 0.222222, 0.333333]
    client = HybridVectorFakeClient(
        {
            ("local_segments", "opaque miss", "semantic"): [
                {
                    "segment_id": "seg_vector",
                    "sample_id": "seg_vector",
                    "video_id": "video",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "timestamp_center": 1.0,
                    "transcript_text": "Public vector target",
                    "_rankingScore": 0.91,
                }
            ],
        }
    )

    response = query_project(
        client=client,
        index_uid="local_segments",
        project_dir=project_dir,
        query="opaque miss",
        limit=1,
        neighbor_count=0,
        hybrid_retrieval=True,
        hybrid_query_vector=query_vector,
        hybrid_query_vector_embedder="default",
        hybrid_query_vector_dimensions=3,
    )

    assert client.searches == [
        {
            "index": "local_segments",
            "query": "opaque miss",
            "limit": 1,
            "mode": "lexical",
            "hybrid": None,
            "vector": None,
        },
        {
            "index": "local_segments",
            "query": "opaque miss",
            "limit": 1,
            "mode": "semantic",
            "hybrid": {"embedder": "default", "semanticRatio": 1.0},
            "vector": query_vector,
        },
    ]
    assert response["bundles"][0]["candidate"]["segment_id"] == "seg_vector"
    assert response["bundles"][0]["candidate"]["retrieval_mode"] == "semantic"
    assert response["retrieval_context"]["hybrid_retrieval"]["query_vector"] == {
        "used": True,
        "embedder": "default",
        "dimensions": 3,
        "source": "argument",
    }
    serialized = json.dumps(response, sort_keys=True)
    assert "0.111111" not in serialized
    assert "0.222222" not in serialized
    assert "0.333333" not in serialized


def test_query_project_hybrid_vector_supports_legacy_clients_without_filter_kwarg(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_vector", 1, 0.0, 2.0, "Public vector target", [])],
    )
    query_vector = [0.111111, 0.222222, 0.333333]
    client = LegacyHybridVectorNoFilterClient(
        {
            ("local_segments", "opaque miss", "semantic"): [
                {
                    "segment_id": "seg_vector",
                    "sample_id": "seg_vector",
                    "video_id": "video",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "timestamp_center": 1.0,
                    "transcript_text": "Public vector target",
                    "_rankingScore": 0.91,
                }
            ],
        }
    )

    response = query_project(
        client=client,
        index_uid="local_segments",
        project_dir=project_dir,
        query="opaque miss",
        limit=1,
        neighbor_count=0,
        hybrid_retrieval=True,
        hybrid_query_vector=query_vector,
        hybrid_query_vector_embedder="default",
        hybrid_query_vector_dimensions=3,
    )

    assert client.searches == [
        {
            "index": "local_segments",
            "query": "opaque miss",
            "limit": 1,
            "mode": "lexical",
            "hybrid": None,
            "vector": None,
        },
        {
            "index": "local_segments",
            "query": "opaque miss",
            "limit": 1,
            "mode": "semantic",
            "hybrid": {"embedder": "default", "semanticRatio": 1.0},
            "vector": query_vector,
        },
    ]
    assert response["bundles"][0]["candidate"]["segment_id"] == "seg_vector"
    assert response["bundles"][0]["candidate"]["retrieval_mode"] == "semantic"


def test_query_project_loads_userprovided_vector_manifest_for_window_search() -> None:
    client = HybridVectorFakeClient(
        {
            ("local_windows", "opaque window miss", "semantic"): [
                {
                    "window_id": "vector_semantic_window_001",
                    "target_segment_id": "vector_semantic_seg_001",
                    "sample_id": "vector_semantic_seg_001",
                    "video_id": "synthetic_vector_lesson",
                    "start_time": 0.0,
                    "end_time": 6.0,
                    "timestamp_center": 3.0,
                    "target_start_time": 0.0,
                    "target_end_time": 6.0,
                    "target_timestamp_center": 3.0,
                    "source_segment_ids": ["vector_semantic_seg_001"],
                    "transcript_window_text": "The lesson compares public synthetic objects.",
                    "_rankingScore": 0.89,
                }
            ],
        }
    )

    response = query_project(
        client=client,
        index_uid="local_windows",
        retrieval_index_kind="window",
        project_dir=PUBLIC_USER_PROVIDED_VECTOR_PROJECT,
        query="opaque window miss",
        limit=1,
        neighbor_count=0,
        hybrid_retrieval=True,
        hybrid_query_vector_name="gradient_direction",
    )

    assert client.searches[1]["vector"] == [0.98, 0.02, 0.0]
    assert response["bundles"][0]["candidate"]["segment_id"] == "vector_semantic_seg_001"
    assert response["retrieval_context"]["hybrid_retrieval"]["query_vector"] == {
        "used": True,
        "embedder": "default",
        "dimensions": 3,
        "source": "manifest",
        "name": "gradient_direction",
    }
    semantic_call = response["retrieval_context"]["searches"]["window"]["calls"][1]
    assert semantic_call["query_vector"]["dimensions"] == 3
    assert "vector" not in semantic_call["query_vector"]


def test_query_project_userprovided_vector_errors_are_specific(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_1", 1, 0.0, 2.0, "text", [])],
    )

    try:
        query_project(
            client=HybridVectorFakeClient({}),
            index_uid="local_segments",
            project_dir=project_dir,
            query="text",
            hybrid_retrieval=True,
            hybrid_query_vector_embedder="default",
        )
    except ValueError as exc:
        assert "query vector is required for userProvided embedder 'default'" in str(exc)
    else:
        raise AssertionError("expected missing query vector ValueError")

    try:
        query_project(
            client=HybridVectorFakeClient({}),
            index_uid="local_segments",
            project_dir=project_dir,
            query="text",
            hybrid_retrieval=True,
            hybrid_query_vector=[1.0, 0.0],
            hybrid_query_vector_dimensions=3,
        )
    except ValueError as exc:
        assert "dimension mismatch: expected 3, got 2" in str(exc)
    else:
        raise AssertionError("expected dimension mismatch ValueError")

    try:
        query_project(
            client=HybridVectorFakeClient({}),
            index_uid="local_segments",
            project_dir=project_dir,
            query="text",
            hybrid_retrieval=True,
            hybrid_embedder="default",
            hybrid_query_vector=[1.0, 0.0, 0.0],
            hybrid_query_vector_embedder="manual",
        )
    except ValueError as exc:
        assert "query vector embedder mismatch" in str(exc)
    else:
        raise AssertionError("expected embedder mismatch ValueError")

    manifest_path = project_dir / "manifests" / "query_vectors.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "embedders": {"default": {"source": "openAi", "dimensions": 3}},
                "queries": {
                    "bad_settings": {
                        "embedder": "default",
                        "dimensions": 3,
                        "vector": [1.0, 0.0, 0.0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    try:
        query_project(
            client=HybridVectorFakeClient({}),
            index_uid="local_segments",
            project_dir=project_dir,
            query="text",
            hybrid_retrieval=True,
            hybrid_query_vector_name="bad_settings",
        )
    except ValueError as exc:
        assert "query vector manifest embedder settings problem" in str(exc)
        assert "source 'userProvided'" in str(exc)
    else:
        raise AssertionError("expected embedder settings ValueError")


def test_query_project_hybrid_retrieval_rejects_invalid_semantic_ratio(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_1", 1, 0.0, 2.0, "text", [])],
    )

    try:
        query_project(
            client=FakeClient([]),
            index_uid="local_segments",
            project_dir=project_dir,
            query="text",
            hybrid_retrieval=True,
            hybrid_semantic_ratio=0.0,
        )
    except ValueError as exc:
        assert "hybrid_semantic_ratio" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_query_project_rerank_reorders_bundles_and_records_breakdown(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_1", 1, 58.0, 62.0, "Introductory aside", []),
            _segment("seg_2", 2, 10.0, 14.0, "Bet size appears on the board", ["frame_000012"]),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000012", "timestamp": 12.0, "frame_path": "/tmp/f12.jpg"}],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "entity_board",
                "project_id": "project",
                "frame_id": "frame_000012",
                "timestamp": 12.0,
                "frame_path": "/tmp/f12.jpg",
                "bbox": None,
                "text": "bet size board",
                "entity_type": "ocr_text",
                "confidence": 0.91,
                "source": "test",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_seg_2_entity_board",
                "project_id": "project",
                "segment_id": "seg_2",
                "entity_id": "entity_board",
                "frame_id": "frame_000012",
                "link_type": "time_overlap+lexical_match",
                "score": 1.2,
                "evidence": ["time_overlap", "lexical_match"],
                "time_overlap": True,
                "lexical_match": ["bet", "size", "board"],
                "mention_candidate": [],
            }
        ],
    )

    response = query_project(
        client=FakeClient(
            hits=[
                {
                    "segment_id": "seg_1",
                    "sample_id": "seg_1",
                    "video_id": "video",
                    "start_time": 58.0,
                    "end_time": 62.0,
                    "timestamp_center": 60.0,
                    "transcript_text": "Introductory aside",
                    "_rankingScore": 0.95,
                },
                {
                    "segment_id": "seg_2",
                    "sample_id": "seg_2",
                    "video_id": "video",
                    "start_time": 10.0,
                    "end_time": 14.0,
                    "timestamp_center": 12.0,
                    "transcript_text": "Bet size appears on the board",
                    "_rankingScore": 0.4,
                },
            ]
        ),
        index_uid="local_segments",
        project_dir=project_dir,
        query="bet size board 10-14s",
        neighbor_count=0,
        rerank=True,
    )

    assert response["retrieval_context"]["rerank"]["enabled"] is True
    assert [bundle["candidate"]["segment_id"] for bundle in response["bundles"]] == [
        "seg_2",
        "seg_1",
    ]
    top = response["bundles"][0]
    assert top["rank"] == 1
    assert top["candidate"]["rank"] == 2
    assert top["rerank"]["original_rank"] == 2
    assert top["rerank"]["breakdown"]["signals"]["query_overlap"]["matched_terms"] == [
        "bet",
        "board",
        "size",
    ]
    assert "#1 (orig #2) video @10.0-14.0s" in response["summary_lines"][0]
    assert "rerank_score=" in response["summary_lines"][0]


def test_query_project_rerank_backend_option_records_private_safe_breakdown(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_1", 1, 1.0, 3.0, "Introductory aside", []),
            _segment("seg_2", 2, 5.0, 9.0, "Plain transcript context", ["frame_000006"]),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000006", "timestamp": 6.0, "frame_path": "/tmp/f6.jpg"}],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "entity_grid",
                "project_id": "project",
                "frame_id": "frame_000006",
                "timestamp": 6.0,
                "frame_path": "/tmp/f6.jpg",
                "bbox": None,
                "text": "range grid diagram",
                "entity_type": "ocr_text",
                "confidence": 0.9,
                "source": "test",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_seg_2_entity_grid",
                "project_id": "project",
                "segment_id": "seg_2",
                "entity_id": "entity_grid",
                "frame_id": "frame_000006",
                "link_type": "time_overlap+lexical_match",
                "score": 1.0,
                "evidence": ["time_overlap", "lexical_match"],
                "time_overlap": True,
                "lexical_match": ["range", "grid", "diagram"],
                "mention_candidate": [],
            }
        ],
    )

    response = query_project(
        client=FakeClient(
            hits=[
                {
                    "segment_id": "seg_1",
                    "sample_id": "seg_1",
                    "video_id": "video",
                    "start_time": 1.0,
                    "end_time": 3.0,
                    "timestamp_center": 2.0,
                    "transcript_text": "Introductory aside",
                    "_rankingScore": 0.95,
                },
                {
                    "segment_id": "seg_2",
                    "sample_id": "seg_2",
                    "video_id": "video",
                    "start_time": 5.0,
                    "end_time": 9.0,
                    "timestamp_center": 7.0,
                    "transcript_text": "Plain transcript context",
                    "_rankingScore": 0.2,
                },
            ]
        ),
        index_uid="local_segments",
        project_dir=project_dir,
        query="range grid diagram",
        neighbor_count=0,
        rerank=True,
        rerank_backend="stub",
    )

    assert response["retrieval_context"]["rerank"]["backend"] == "stub"
    assert response["retrieval_context"]["rerank"]["external_transport"] == "none"
    assert response["bundles"][0]["candidate"]["segment_id"] == "seg_2"
    top = response["bundles"][0]["rerank"]
    assert top["backend"] == "stub"
    assert top["breakdown"]["score"] == top["score"]
    assert top["breakdown"]["original_rank"] == top["original_rank"]
    assert "linked_entities.entity.text" in top["used_fields"]
    assert "visual_entities.text" in top["used_fields"]
    assert "rerank_score=" in response["summary_lines"][0]


def test_query_project_cli_prints_summary_and_writes_output(tmp_path: Path, monkeypatch, capsys) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_1", 1, 1.0, 3.0, "Shown on slide", ["frame_000001"])],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000001", "timestamp": 1.0, "frame_path": "/tmp/f1.jpg"}],
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "query-project",
            "--index",
            "local_segments",
            "--project-dir",
            str(project_dir),
            "--query",
            "shown",
            "--rerank",
            "--rerank-time-hint",
            "1-3s",
            "--output",
            "query-result.json",
        ]
    )
    monkeypatch.setattr(
        "oarag.cli.client_from_args",
        lambda _: FakeClient(
            hits=[
                {
                    "segment_id": "seg_1",
                    "sample_id": "seg_1",
                    "video_id": "video",
                    "start_time": 1.0,
                    "end_time": 3.0,
                    "timestamp_center": 2.0,
                    "transcript_text": "Shown on slide",
                }
            ]
        ),
    )

    args.func(args)
    captured = capsys.readouterr()
    assert "#1 video @1.0-3.0s: Shown on slide" in captured.err

    payload = json.loads(captured.out)
    assert payload["summary_lines"][0].startswith("#1 video @1.0-3.0s")
    output_path = project_dir / "query-result.json"
    assert output_path.exists()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["retrieval_context"]["rerank"]["enabled"] is True
    assert written["bundles"][0]["rerank"]["breakdown"]["signals"]["timestamp_proximity"][
        "distance_seconds"
    ] == 0.0
    assert written["bundles"][0]["candidate"]["segment_id"] == "seg_1"


def test_ask_project_cli_format_json_returns_answer_schema(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_1", 1, 1.0, 3.0, "Shown on slide", ["frame_000001"])],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000001", "timestamp": 1.0, "frame_path": "/tmp/f1.jpg"}],
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "ask-project",
            "--index",
            "local_segments",
            "--project-dir",
            str(project_dir),
            "--query",
            "shown slide",
            "--format",
            "json",
            "--output",
            "ask-result.json",
        ]
    )
    monkeypatch.setattr(
        "oarag.cli.client_from_args",
        lambda _: FakeClient(
            hits=[
                {
                    "segment_id": "seg_1",
                    "sample_id": "seg_1",
                    "video_id": "video",
                    "start_time": 1.0,
                    "end_time": 3.0,
                    "timestamp_center": 2.0,
                    "transcript_text": "Shown on slide",
                    "_rankingScore": 0.8,
                }
            ]
        ),
    )

    args.func(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["answer_type"] == "grounded_answer"
    assert payload["answer"]["answer_type"] == "grounded_answer"
    assert payload["answer"]["claims"][0]["citation_ids"] == ["citation_1"]
    assert payload["answer"]["citations"][0]["frame_refs"] == [
        {"frame_id": "frame_000001", "timestamp": 1.0, "frame_path": "/tmp/f1.jpg"}
    ]
    written = json.loads((project_dir / "ask-result.json").read_text(encoding="utf-8"))
    assert written["answer"]["schema_version"] == "grounded-answer-v1"


def test_ask_project_cli_window_index_kind_maps_window_hit(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_1", 1, 0.0, 2.0, "Intro context", []),
            _segment("seg_2", 2, 3.0, 5.0, "Target range grid appears", []),
        ],
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "ask-project",
            "--index",
            "local_windows",
            "--index-kind",
            "window",
            "--project-dir",
            str(project_dir),
            "--query",
            "target range grid",
            "--format",
            "json",
        ]
    )
    monkeypatch.setattr(
        "oarag.cli.client_from_args",
        lambda _: FakeClient(
            hits=[
                {
                    "window_id": "window_seg_2_abc123",
                    "target_segment_id": "seg_2",
                    "sample_id": "seg_2",
                    "video_id": "video",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "timestamp_center": 2.5,
                    "target_start_time": 3.0,
                    "target_end_time": 5.0,
                    "target_timestamp_center": 4.0,
                    "source_segment_ids": ["seg_1", "seg_2"],
                    "transcript_window_text": "Intro context Target range grid appears",
                    "_rankingScore": 0.8,
                }
            ]
        ),
    )

    args.func(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["index_kind"] == "window"
    assert payload["counts"]["window_search_hits"] == 1
    citation = payload["answer"]["citations"][0]
    assert citation["retrieval_index_kind"] == "window"
    assert citation["retrieval_sources"][0]["source"] == "window"
    assert citation["retrieval_sources"][0]["window_id"] == "window_seg_2_abc123"


def _segment(
    segment_id: str,
    sample_index: int,
    start_time: float,
    end_time: float,
    transcript_text: str,
    frame_refs: list[str],
) -> dict:
    return {
        "segment_id": segment_id,
        "project_id": "project",
        "video_id": "video",
        "sample_index": sample_index,
        "start_time": start_time,
        "end_time": end_time,
        "timestamp_center": (start_time + end_time) / 2,
        "transcript_text": transcript_text,
        "frame_refs": frame_refs,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _visual_entity(
    entity_id: str,
    frame_id: str,
    text: str,
    timestamp: float,
    *,
    frame_path: str,
) -> dict:
    return {
        "entity_id": entity_id,
        "project_id": "project",
        "frame_id": frame_id,
        "timestamp": timestamp,
        "frame_path": frame_path,
        "bbox": None,
        "text": text,
        "entity_type": "visual_observation",
        "confidence": 0.91,
        "source": "test",
        "visual_description": text,
        "position": None,
        "relations": [],
        "parser_version": "test-v1",
        "source_model": "stub-vlm",
    }
