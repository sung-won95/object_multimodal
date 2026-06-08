from __future__ import annotations

from pathlib import Path

import pytest

from oarag.graph.concept_graph_schema import (
    CONCEPT_GRAPH_ARTIFACT_CONTRACT,
    CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH,
    CONCEPT_GRAPH_DB_EDGE_CONTRACT,
    CONCEPT_GRAPH_DB_NODE_CONTRACT,
    CONCEPT_GRAPH_SCHEMA_VERSION,
    CONCEPT_SOURCE_SIGNAL_CONTRACT,
    ConceptGraphConceptNode,
    ConceptGraphEvidenceSource,
    ConceptGraphRelationEdge,
    load_concept_graph_artifact,
    validate_concept_graph_records,
    write_concept_graph_artifact,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "public_concept_graph_project"
    / "manifests"
    / "concept_graph.jsonl"
)


def test_public_concept_graph_fixture_validates_and_roundtrips(tmp_path: Path) -> None:
    records = load_concept_graph_artifact(FIXTURE_PATH)

    summary = validate_concept_graph_records(records)

    assert summary == {
        "schema_version": CONCEPT_GRAPH_SCHEMA_VERSION,
        "concepts": 3,
        "relations": 2,
        "candidate_relations": 1,
        "verified_relations": 1,
        "timestamp_only_candidate_relations": 1,
        "timestamp_fallback_counted_as_verified": False,
    }

    output_path = tmp_path / CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH
    count = write_concept_graph_artifact(output_path, records)
    reloaded = load_concept_graph_artifact(output_path)

    assert count == 5
    assert [record.to_dict() for record in reloaded] == [
        record.to_dict() for record in records
    ]


def test_concept_node_and_relation_expose_graph_db_contract_properties() -> None:
    records = load_concept_graph_artifact(FIXTURE_PATH)
    concept = next(record for record in records if isinstance(record, ConceptGraphConceptNode))
    relation = next(
        record
        for record in records
        if isinstance(record, ConceptGraphRelationEdge)
        and record.edge_id == "edge_gradient_descent_uses_learning_rate"
    )
    timestamp_relation = next(
        record
        for record in records
        if isinstance(record, ConceptGraphRelationEdge)
        and record.edge_id == "edge_gradient_descent_related_loss_timestamp_candidate"
    )

    assert concept.graph_db_labels() == ("GraphNode", "Concept")
    assert concept.graph_db_key() == "concept:public_lecture_graph_001:concept_gradient_descent"
    assert concept.graph_db_properties()["source_evidence_unit_ids"] == [
        "evu_public_001_0001"
    ]

    assert relation.graph_db_relationship_type() == "USES"
    assert relation.graph_db_properties()["relation_status"] == "verified"
    assert relation.graph_db_properties()["timestamp_fallback_counted_as_verified"] is False
    assert timestamp_relation.graph_db_relationship_type() == "RELATED_TO"
    assert timestamp_relation.graph_db_properties()["timestamp_only_candidate"] is True
    assert timestamp_relation.graph_db_properties()["relation_status"] == "candidate"


def test_graph_db_label_and_property_contract_is_machine_readable() -> None:
    assert CONCEPT_GRAPH_ARTIFACT_CONTRACT["path"] == "manifests/concept_graph.jsonl"
    assert CONCEPT_GRAPH_DB_NODE_CONTRACT["Concept"]["labels"] == ("GraphNode", "Concept")
    assert "confidence" in CONCEPT_GRAPH_DB_NODE_CONTRACT["Concept"]["required_properties"]
    assert CONCEPT_GRAPH_DB_EDGE_CONTRACT["USES"]["from"] == "Concept"
    assert CONCEPT_GRAPH_DB_EDGE_CONTRACT["USES"]["to"] == "Concept"
    assert "source_signals" in CONCEPT_GRAPH_DB_EDGE_CONTRACT["USES"]["optional_properties"]
    assert CONCEPT_SOURCE_SIGNAL_CONTRACT["timestamp_overlap"]["status"] == "candidate_only"
    assert "candidate signals only" in CONCEPT_GRAPH_ARTIFACT_CONTRACT[
        "timestamp_only_relation_contract"
    ]


def test_timestamp_only_relation_cannot_be_verified() -> None:
    with pytest.raises(ValueError, match="timestamp-only relation edges must remain candidate"):
        ConceptGraphRelationEdge.from_dict(
            {
                "record_type": "relation_edge",
                "edge_id": "edge_bad_timestamp_verified",
                "lecture_id": "public_lecture_graph_001",
                "source_concept_id": "concept_a",
                "relation_type": "related_to",
                "target_concept_id": "concept_b",
                "evidence_unit_ids": ["evu_public_001_0001"],
                "source_signals": ["timestamp_overlap"],
                "confidence": 0.4,
                "relation_status": "verified",
            }
        )


def test_timestamp_only_evidence_source_cannot_claim_verified_object_alignment() -> None:
    with pytest.raises(
        ValueError,
        match="timestamp-only source signals cannot be verified object alignment",
    ):
        ConceptGraphEvidenceSource.from_dict(
            {
                "evidence_unit_id": "evu_public_001_0001",
                "source_type": "visual_state_interval",
                "source_signal": "timestamp_overlap",
                "is_verified_object_alignment": True,
            }
        )


def test_validation_rejects_relation_edges_with_missing_concepts() -> None:
    concept = ConceptGraphConceptNode.from_dict(
        {
            "record_type": "concept_node",
            "concept_id": "concept_a",
            "lecture_id": "public_lecture_graph_001",
            "label": "Concept A",
            "concept_type": "topic",
            "source_evidence_unit_ids": ["evu_public_001_0001"],
            "confidence": 0.8,
        }
    )
    relation = ConceptGraphRelationEdge.from_dict(
        {
            "record_type": "relation_edge",
            "edge_id": "edge_missing_target",
            "lecture_id": "public_lecture_graph_001",
            "source_concept_id": "concept_a",
            "relation_type": "related_to",
            "target_concept_id": "concept_missing",
            "evidence_unit_ids": ["evu_public_001_0001"],
            "source_signals": ["transcript_statement"],
            "confidence": 0.7,
            "relation_status": "candidate",
        }
    )

    with pytest.raises(ValueError, match="references missing concepts"):
        validate_concept_graph_records([concept, relation])
