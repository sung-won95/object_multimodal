from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oarag.graph.graph_ingest import GraphIngestUnavailableError
from oarag.retrieval.dual_candidates import (
    GRAPH_TRAVERSAL_SOURCE,
    MEILI_EXPANDED_SOURCE,
    MEILI_RAW_SOURCE,
    analyze_dual_candidate_query,
    graph_candidate_cypher,
    query_project_dual_candidates,
    serialize_graph_candidate_record,
)


class QueryAwareEvidenceClient:
    def __init__(self, hits_by_query: dict[str, list[dict[str, Any]]]) -> None:
        self.hits_by_query = hits_by_query
        self.searches: list[dict[str, Any]] = []

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
    ) -> dict[str, Any]:
        self.searches.append(
            {"index_uid": index_uid, "query": query, "limit": limit, "filter": filter}
        )
        return {
            "hits": self.hits_by_query.get(query, [])[:limit],
            "processingTimeMs": 2,
            "indexUid": index_uid,
            "query": query,
        }


class FakeGraphSession:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def run(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "parameters": parameters or {}})
        return self.rows


class UnavailableGraphSession:
    def run(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        raise GraphIngestUnavailableError("Neo4j is unavailable in fixture")


def test_analyzer_extracts_concepts_and_intents_without_llm() -> None:
    analysis = analyze_dual_candidate_query("Which diagram explains gradient descent step size?")

    assert analysis.query_type == "visual_relation"
    assert "visual" in analysis.intent_candidates
    assert "relation" in analysis.intent_candidates
    assert analysis.concept_candidates[0].text == "diagram explains gradient descent step size"
    assert "gradient descent" in [candidate.text for candidate in analysis.concept_candidates]


def test_dual_candidate_union_dedupes_and_preserves_candidate_sources(tmp_path: Path) -> None:
    project_dir = _write_project(tmp_path)
    client = QueryAwareEvidenceClient(
        {
            "Which diagram explains gradient descent step size?": [
                _hit("evu_raw", "seg_raw", score=0.92),
            ],
            "diagram explains gradient descent step size": [
                _hit("evu_raw", "seg_raw", score=0.88),
                _hit("evu_expanded", "seg_expanded", score=0.8),
            ],
            "diagram explains gradient": [
                _hit("evu_expanded", "seg_expanded", score=0.79),
            ],
        }
    )

    response = query_project_dual_candidates(
        client=client,
        index_uid="sample_evidence_units",
        project_dir=project_dir,
        query="Which diagram explains gradient descent step size?",
        limit=3,
        candidate_pool_limit=3,
        enable_graph=False,
    )

    candidates = response["candidates"]
    by_id = {candidate["evidence_unit_id"]: candidate for candidate in candidates}
    assert set(by_id) == {"evu_raw", "evu_expanded"}
    assert by_id["evu_raw"]["candidate_source_types"] == [MEILI_RAW_SOURCE, MEILI_EXPANDED_SOURCE]
    assert by_id["evu_raw"]["candidate_sources"][0]["rank"] == 1
    assert by_id["evu_expanded"]["candidate_source_types"] == [MEILI_EXPANDED_SOURCE]
    assert response["retrieval_context"]["graph"]["skip_reason"] == "graph_disabled"
    assert response["diagnostics"]["source_counts"][MEILI_RAW_SOURCE]["unique_candidates"] == 1
    assert response["diagnostics"]["source_counts"][MEILI_EXPANDED_SOURCE]["unique_candidates"] == 2


def test_graph_only_target_recall_fixture_records_source_recall(tmp_path: Path) -> None:
    project_dir = _write_project(tmp_path)
    graph_session = FakeGraphSession(
        [
            {
                "evidence_unit_id": "evu_graph_target",
                "project_id": "public_project",
                "matched_concept": {
                    "concept_id": "concept_step_size",
                    "label": "Step size",
                    "lecture_id": "lecture_public",
                },
                "related_concept": {
                    "concept_id": "concept_gradient_descent",
                    "label": "Gradient descent",
                    "lecture_id": "lecture_public",
                },
                "graph_path": [
                    "concept:lecture_public:concept_step_size",
                    "concept:lecture_public:concept_gradient_descent",
                    "evidence_unit:lecture_public:evu_graph_target",
                ],
                "relationships": ["USES", "CONCEPT_SOURCE_EVIDENCE"],
                "graph_match_type": "related_concept_evidence",
                "score": 0.74,
            }
        ]
    )

    response = query_project_dual_candidates(
        client=QueryAwareEvidenceClient(
            {
                "Where is the step size diagram?": [_hit("evu_meili_only", "seg_meili", score=0.9)]
            }
        ),
        index_uid="sample_evidence_units",
        project_dir=project_dir,
        query="Where is the step size diagram?",
        limit=2,
        candidate_pool_limit=2,
        graph_session=graph_session,
        target_evidence_unit_ids=["evu_graph_target"],
    )

    candidate_ids = [candidate["evidence_unit_id"] for candidate in response["candidates"]]
    assert candidate_ids == ["evu_meili_only", "evu_graph_target"]
    graph_candidate = response["candidates"][1]
    assert graph_candidate["candidate_source_types"] == [GRAPH_TRAVERSAL_SOURCE]
    assert graph_candidate["candidate_sources"][0]["graph_path"][-1] == (
        "evidence_unit:lecture_public:evu_graph_target"
    )
    assert graph_candidate["candidate_sources"][0]["matched_concept"]["concept_id"] == (
        "concept_step_size"
    )
    recall = response["diagnostics"]["source_recall"]["by_source"]
    assert MEILI_RAW_SOURCE not in recall
    assert recall[GRAPH_TRAVERSAL_SOURCE]["recalled_targets"] == [
        "evidence_unit:evu_graph_target"
    ]
    assert response["retrieval_context"]["graph"]["status"] == "hit"
    assert graph_session.calls[0]["parameters"]["concept_terms"][0] == "step size diagram"


def test_graph_aware_rerank_fixture_lifts_graph_relation_over_generic_hit(
    tmp_path: Path,
) -> None:
    project_dir = _write_project(tmp_path)
    graph_session = FakeGraphSession(
        [
            {
                "evidence_unit_id": "evu_graph_target",
                "project_id": "public_project",
                "matched_concept": {
                    "concept_id": "concept_step_size",
                    "label": "Step size",
                    "lecture_id": "lecture_public",
                },
                "related_concept": {
                    "concept_id": "concept_gradient_descent",
                    "label": "Gradient descent",
                    "lecture_id": "lecture_public",
                },
                "graph_path": [
                    "concept:lecture_public:concept_step_size",
                    "concept:lecture_public:concept_gradient_descent",
                    "evidence_unit:lecture_public:evu_graph_target",
                ],
                "relationships": ["USES", "CONCEPT_SOURCE_EVIDENCE"],
                "graph_match_type": "related_concept_evidence",
                "score": 0.7,
            }
        ]
    )

    response = query_project_dual_candidates(
        client=QueryAwareEvidenceClient(
            {
                "Which diagram explains gradient descent step size?": [
                    {
                        **_hit("evu_meili_only", "seg_meili", score=0.98),
                        "evidence_text": "A generic public hit mentions gradient descent.",
                    }
                ]
            }
        ),
        index_uid="sample_evidence_units",
        project_dir=project_dir,
        query="Which diagram explains gradient descent step size?",
        limit=2,
        candidate_pool_limit=2,
        graph_session=graph_session,
        target_evidence_unit_ids=["evu_graph_target"],
        graph_aware_rerank=True,
    )

    candidate_ids = [candidate["evidence_unit_id"] for candidate in response["candidates"]]
    assert candidate_ids == ["evu_graph_target", "evu_meili_only"]
    rerank = response["candidates"][0]["graph_aware_rerank"]
    assert rerank["original_rank"] == 2
    assert rerank["components"]["related_concept_graph_path"] > 0
    assert rerank["components"]["relation_type_match"] > 0
    diagnostics = response["diagnostics"]["graph_aware_rerank"]
    assert diagnostics["public_safe"] is True
    assert diagnostics["top_changed"] is True
    assert diagnostics["returned_breakdowns"][0]["source_ids"] == [
        "evidence_unit:evu_graph_target",
        "evu_graph_target",
        "seg_graph",
    ]
    rendered_diagnostics = json.dumps(diagnostics, ensure_ascii=False)
    assert "Which diagram explains gradient descent step size?" not in rendered_diagnostics
    assert "A generic public hit" not in rendered_diagnostics


def test_graph_candidate_cypher_supports_global_concept_merge_paths() -> None:
    cypher = graph_candidate_cypher()

    assert "INSTANCE_OF_GLOBAL_CONCEPT" in cypher
    assert "GlobalConcept" in cypher
    assert "global_concept_related_evidence" in cypher
    assert "RELATED_SOURCE_EVIDENCE" in cypher


def test_graph_unavailable_falls_back_to_meili_only_with_skip_reason(tmp_path: Path) -> None:
    project_dir = _write_project(tmp_path)

    response = query_project_dual_candidates(
        client=QueryAwareEvidenceClient(
            {"Where is the step size diagram?": [_hit("evu_meili_only", "seg_meili", score=0.9)]}
        ),
        index_uid="sample_evidence_units",
        project_dir=project_dir,
        query="Where is the step size diagram?",
        limit=2,
        graph_session=UnavailableGraphSession(),
    )

    assert [candidate["evidence_unit_id"] for candidate in response["candidates"]] == [
        "evu_meili_only"
    ]
    assert response["retrieval_context"]["graph"]["available"] is False
    assert response["retrieval_context"]["graph"]["status"] == "skipped_unavailable"
    assert response["retrieval_context"]["graph"]["skip_reason"] == "graph_db_unavailable"
    assert response["candidates"][0]["candidate_source_types"] == [MEILI_RAW_SOURCE]


def test_graph_only_can_skip_meili_candidate_generation(tmp_path: Path) -> None:
    project_dir = _write_project(tmp_path)
    client = QueryAwareEvidenceClient(
        {"Where is the step size diagram?": [_hit("evu_meili_only", "seg_meili", score=0.9)]}
    )
    graph_session = FakeGraphSession(
        [
            {
                "evidence_unit_id": "evu_graph_target",
                "project_id": "public_project",
                "matched_concept": {"concept_id": "concept_step_size", "label": "Step size"},
                "graph_path": ["concept:step_size", "evidence_unit:evu_graph_target"],
                "relationships": ["MENTIONS"],
                "graph_match_type": "direct_mention",
                "score": 0.7,
            }
        ]
    )

    response = query_project_dual_candidates(
        client=client,
        index_uid="sample_evidence_units",
        project_dir=project_dir,
        query="Where is the step size diagram?",
        limit=2,
        candidate_pool_limit=2,
        enable_meili=False,
        graph_session=graph_session,
        target_evidence_unit_ids=["evu_graph_target"],
    )

    assert client.searches == []
    assert [candidate["evidence_unit_id"] for candidate in response["candidates"]] == [
        "evu_graph_target"
    ]
    assert response["retrieval_context"]["candidate_generation"]["sources"] == [
        GRAPH_TRAVERSAL_SOURCE
    ]
    assert response["retrieval_context"]["meilisearch"]["raw"]["skip_reason"] == "meili_disabled"
    assert response["diagnostics"]["source_recall"]["by_source"][GRAPH_TRAVERSAL_SOURCE][
        "recalled_count"
    ] == 1


def test_serialize_graph_candidate_record_keeps_public_safe_graph_metadata() -> None:
    serialized = serialize_graph_candidate_record(
        {
            "evidence_unit_id": "evu_public",
            "matched_concept": {
                "concept_id": "concept_gradient",
                "label": "Gradient",
                "description": "raw private description omitted",
                "metadata": {"raw": "private"},
            },
            "graph_path": ["concept:gradient", "evidence_unit:evu_public"],
            "relationships": ["MENTIONS"],
            "score": "0.8",
        },
        rank=1,
    )

    assert serialized["candidate"]["evidence_unit_id"] == "evu_public"
    assert serialized["source"]["matched_concept"] == {
        "concept_id": "concept_gradient",
        "label": "Gradient",
    }
    rendered = json.dumps(serialized, ensure_ascii=False)
    assert "raw private description" not in rendered
    assert "metadata" not in rendered


def test_serialize_graph_candidate_record_preserves_zero_timestamps() -> None:
    serialized = serialize_graph_candidate_record(
        {
            "evidence_unit_id": "evu_zero",
            "start_time": 0.0,
            "end_time": 0.0,
            "score": 0.5,
        },
        rank=1,
    )

    assert serialized["candidate"]["start_time"] == 0.0
    assert serialized["candidate"]["end_time"] == 0.0


def _write_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "evidence_units.jsonl",
        [
            {
                "project_id": "public_project",
                "evidence_unit_id": "evu_manifest",
            },
            _hit("evu_graph_target", "seg_graph", score=0.0),
        ],
    )
    return project_dir


def _hit(evidence_unit_id: str, target_segment_id: str, *, score: float) -> dict[str, Any]:
    return {
        "evidence_unit_id": evidence_unit_id,
        "project_id": "public_project",
        "video_id": "video_public",
        "target_segment_id": target_segment_id,
        "source_segment_ids": [target_segment_id],
        "start_time": 1.0,
        "end_time": 3.0,
        "concept_ids": ["concept_gradient_descent"],
        "concept_labels": ["Gradient descent"],
        "concept_aliases": ["steepest descent"],
        "concept_relation_text": "Gradient descent uses step size",
        "alignment_status": "candidate",
        "source_quality": {"has_visual_state": True, "concept_count": 1},
        "_rankingScore": score,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
