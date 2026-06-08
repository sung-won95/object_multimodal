from __future__ import annotations

import json
from typing import Any

from oarag.retrieval.graph_rerank import (
    graph_aware_rerank_breakdown,
    graph_aware_rerank_candidates,
)


def test_graph_aware_rerank_scores_graph_relation_and_object_evidence() -> None:
    breakdown = graph_aware_rerank_breakdown(
        _graph_relation_candidate(),
        query="Which diagram explains gradient descent step size?",
        query_analysis=_analysis(),
        original_rank=2,
    )

    components = breakdown["components"]
    assert components["query_concept_direct_match"] > 0
    assert components["related_concept_graph_path"] > 0.6
    assert components["relation_type_match"] > 0
    assert components["visual_description"] > 0
    assert components["visual_object_support"] > 0
    assert components["verified_alignment"] > 0
    assert breakdown["flags"]["has_related_graph_path"] is True
    assert breakdown["flags"]["has_verified_alignment"] is True


def test_timestamp_only_fallback_gets_penalty_without_verified_alignment_boost() -> None:
    timestamp_candidate = _graph_relation_candidate()
    timestamp_candidate["candidate_entity_link_statuses"] = {"link_public_ts": "timestamp_fallback"}
    timestamp_candidate["verified_entity_link_ids"] = []
    timestamp_candidate["source_quality"] = {
        "has_visual_state": True,
        "has_visual_entity": True,
        "has_vlm_entity": False,
        "has_verified_link": True,
        "has_timestamp_fallback_link": True,
        "timestamp_fallback_link_count": 1,
        "verified_link_count": 0,
        "uses_ocr_only": True,
        "verified_object_alignment": {
            "has_verified_object_alignment": True,
            "timestamp_fallback_counted_as_verified": True,
        },
    }

    breakdown = graph_aware_rerank_breakdown(
        timestamp_candidate,
        query="Where is the gradient descent diagram?",
        query_analysis=_analysis(),
        original_rank=1,
    )

    assert breakdown["components"]["verified_alignment"] == 0.0
    assert breakdown["components"]["timestamp_fallback_penalty"] < 0
    assert breakdown["components"]["ocr_only_penalty"] < 0
    assert breakdown["flags"]["timestamp_only_fallback"] is True
    assert breakdown["flags"]["has_verified_alignment"] is False


def test_graph_aware_rerank_lifts_relation_match_over_generic_lexical_hit() -> None:
    generic = {
        "evidence_unit_id": "evu_generic",
        "rank": 1,
        "score": 0.98,
        "candidate_key": "evidence_unit:evu_generic",
        "candidate_source_types": ["meili_raw"],
        "candidate_sources": [{"source_type": "meili_raw", "rank": 1, "score": 0.98}],
        "concept_labels": ["Gradient descent"],
        "source_quality": {"has_visual_state": True},
        "evidence_text": "A generic slide mentions gradient descent and step size.",
    }
    graph_match = _graph_relation_candidate()
    graph_match["rank"] = 2
    graph_match["score"] = 0.62

    reranked = graph_aware_rerank_candidates(
        [generic, graph_match],
        query="Which diagram explains gradient descent step size?",
        query_analysis=_analysis(),
    )

    assert [candidate["evidence_unit_id"] for candidate in reranked] == [
        "evu_graph_relation",
        "evu_generic",
    ]
    assert reranked[0]["graph_aware_rerank"]["original_rank"] == 2
    rendered_rerank = json.dumps(reranked[0]["graph_aware_rerank"])
    assert "Which diagram explains gradient descent step size?" not in rendered_rerank


def _analysis() -> dict[str, Any]:
    return {
        "query_type": "visual_relation",
        "intent_candidates": ["visual", "relation"],
        "concept_candidates": [
            {"text": "diagram explains gradient descent step size", "rank": 1},
            {"text": "gradient descent", "rank": 2},
            {"text": "step size", "rank": 3},
        ],
    }


def _graph_relation_candidate() -> dict[str, Any]:
    return {
        "evidence_unit_id": "evu_graph_relation",
        "rank": 2,
        "score": 0.74,
        "candidate_key": "evidence_unit:evu_graph_relation",
        "candidate_source_types": ["graph_traversal"],
        "candidate_sources": [
            {
                "source_type": "graph_traversal",
                "rank": 1,
                "score": 0.74,
                "graph_match_type": "related_concept_evidence",
                "graph_path": [
                    "concept:lecture_public:concept_gradient_descent",
                    "concept:lecture_public:concept_step_size",
                    "evidence_unit:lecture_public:evu_graph_relation",
                ],
                "relationships": ["USES", "CONCEPT_SOURCE_EVIDENCE"],
                "matched_concept": {
                    "concept_id": "concept_gradient_descent",
                    "label": "Gradient descent",
                },
                "related_concept": {
                    "concept_id": "concept_step_size",
                    "label": "Step size",
                },
            }
        ],
        "concept_ids": ["concept_step_size"],
        "concept_labels": ["Step size"],
        "concept_aliases": ["Learning rate"],
        "concept_relation_text": "Gradient descent uses step size",
        "visual_state_ids": ["state_public_step"],
        "visual_entity_ids": ["entity_public_arrow"],
        "verified_entity_link_ids": ["link_public_verified"],
        "candidate_entity_link_ids": ["link_public_verified"],
        "candidate_entity_link_statuses": {"link_public_verified": "verified"},
        "source_quality": {
            "has_visual_state": True,
            "has_visual_entity": True,
            "has_vlm_entity": True,
            "has_visual_description": True,
            "visual_description_count": 1,
            "has_verified_link": True,
            "verified_link_count": 1,
            "candidate_visual_support": {
                "has_candidate_visual_support": True,
                "visual_state_count": 1,
                "visual_entity_count": 1,
            },
            "verified_object_alignment": {
                "has_verified_object_alignment": True,
                "timestamp_fallback_counted_as_verified": False,
            },
        },
        "semantic_text": "A public diagram shows a step size controlling gradient updates.",
    }
