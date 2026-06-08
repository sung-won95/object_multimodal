from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from oarag.graph.concept_graph_ingest import (
    build_concept_graph_document,
    build_concept_graph_ingest_plan,
    ingest_concept_graph,
)
from oarag.graph.concept_graph_schema import (
    ConceptGraphConceptNode,
    load_concept_graph_artifact,
)
from oarag.graph.graph_ingest import GraphIngestUnavailableError, run_graph_ingest_plan


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "public_concept_graph_project"
    / "manifests"
    / "concept_graph.jsonl"
)


def test_concept_graph_plan_merges_fixture_idempotently() -> None:
    records = load_concept_graph_artifact(FIXTURE_PATH)
    plan = build_concept_graph_ingest_plan(records)
    session = FakeMergeSession()

    run_graph_ingest_plan(session, plan)
    first_counts = (len(session.nodes), len(session.relationships))
    run_graph_ingest_plan(session, plan)

    assert first_counts == (6, 7)
    assert (len(session.nodes), len(session.relationships)) == first_counts
    assert session.nodes["lecture:public_lecture_graph_001"]["lecture_id"] == (
        "public_lecture_graph_001"
    )
    assert "MERGE (start)-[r:USES {key: row.key}]->(end)" in _relationship_statement(
        plan,
        "USES",
    ).cypher
    assert session.relationships[
        (
            "USES",
            "rel:USES:public_lecture_graph_001:edge_gradient_descent_uses_learning_rate",
            "concept:public_lecture_graph_001:concept_gradient_descent",
            "concept:public_lecture_graph_001:concept_learning_rate",
        )
    ]["relation_status"] == "verified"
    assert session.relationships[
        (
            "RELATED_TO",
            "rel:RELATED_TO:public_lecture_graph_001:"
            "edge_gradient_descent_related_loss_timestamp_candidate",
            "concept:public_lecture_graph_001:concept_gradient_descent",
            "concept:public_lecture_graph_001:concept_loss_function",
        )
    ]["timestamp_only_candidate"] is True
    assert session.relationships[
        (
            "RELATED_TO",
            "rel:RELATED_TO:public_lecture_graph_001:"
            "edge_gradient_descent_related_loss_timestamp_candidate",
            "concept:public_lecture_graph_001:concept_gradient_descent",
            "concept:public_lecture_graph_001:concept_loss_function",
        )
    ]["relation_status"] == "candidate"


def test_concept_graph_document_derives_visual_object_nodes_and_edges() -> None:
    records = [
        ConceptGraphConceptNode.from_dict(
            {
                "record_type": "concept_node",
                "concept_id": "concept_visual_gradient",
                "lecture_id": "public_lecture_visual_001",
                "label": "Gradient arrow",
                "concept_type": "visual_example",
                "source_evidence_unit_ids": ["evu_public_visual_0001"],
                "evidence_sources": [
                    {
                        "evidence_unit_id": "evu_public_visual_0001",
                        "source_type": "visual_object",
                        "source_signal": "visual_object_relation",
                        "source_id": "vobj_public_arrow_001",
                        "confidence": 0.91,
                        "is_verified_object_alignment": True,
                        "metadata": {"label": "arrow"},
                    }
                ],
                "confidence": 0.9,
            }
        )
    ]

    document = build_concept_graph_document(records)

    assert document["counts"]["nodes_by_label"] == {
        "Concept": 1,
        "EvidenceUnit": 1,
        "Lecture": 1,
        "VisualObject": 1,
    }
    visual_object = next(
        node for node in document["nodes"] if node["key"].startswith("visual_object:")
    )
    assert visual_object["properties"]["visual_object_id"] == "vobj_public_arrow_001"
    assert visual_object["properties"]["label"] == "arrow"
    assert document["counts"]["relationships_by_type"]["SHOWS"] == 1
    assert document["counts"]["relationships_by_type"]["DEPICTS"] == 1
    depicts = next(
        relationship
        for relationship in document["relationships"]
        if relationship["type"] == "DEPICTS"
    )
    assert depicts["start_node_key"] == (
        "visual_object:public_lecture_visual_001:vobj_public_arrow_001"
    )
    assert depicts["properties"]["relation_status"] == "verified"


def test_concept_graph_dry_run_summary_is_public_safe_counts_only() -> None:
    summary = ingest_concept_graph(concept_graph_path=FIXTURE_PATH, dry_run=True)
    rendered = json.dumps(summary, ensure_ascii=False, sort_keys=True)

    assert summary["runtime"] == {
        "backend": "neo4j",
        "attempted": False,
        "available": False,
        "status": "dry_run",
        "skip_reason": "dry_run",
    }
    assert summary["records"]["relations"] == 2
    assert summary["nodes"]["by_label"] == {"Concept": 3, "EvidenceUnit": 2, "Lecture": 1}
    assert str(FIXTURE_PATH) not in rendered
    assert "Gradient descent" not in rendered
    assert "seg_public_001_0001" not in rendered
    assert "dev-password" not in rendered


def test_concept_graph_ingest_skips_when_neo4j_runtime_unavailable() -> None:
    def unavailable_driver_factory(_config: Any) -> Any:
        raise GraphIngestUnavailableError("Neo4j is unavailable at a private endpoint")

    summary = ingest_concept_graph(
        concept_graph_path=FIXTURE_PATH,
        dry_run=False,
        driver_factory=unavailable_driver_factory,
    )

    assert summary["runtime"] == {
        "backend": "neo4j",
        "attempted": True,
        "available": False,
        "status": "skipped_unavailable",
        "skip_reason": "neo4j_runtime_unavailable",
    }
    assert summary["statements_executed"] == 0


def _relationship_statement(plan: Any, relationship_type: str) -> Any:
    for statement in plan.relationship_statements:
        if f"[r:{relationship_type} " in statement.cypher:
            return statement
    raise AssertionError(f"missing relationship statement for {relationship_type}")


class FakeMergeSession:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.relationships: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def run(self, query: str, parameters: dict[str, Any] | None = None) -> None:
        params = parameters or {}
        if "UNWIND $nodes AS row" in query:
            for row in params["nodes"]:
                self.nodes[row["key"]] = dict(row["properties"])
            return
        if "UNWIND $relationships AS row" in query:
            match = re.search(r"MERGE \(start\)-\[r:([A-Z_]+) \{key: row\.key\}\]->", query)
            assert match is not None
            relationship_type = match.group(1)
            for row in params["relationships"]:
                self.relationships[
                    (
                        relationship_type,
                        row["key"],
                        row["start_node_key"],
                        row["end_node_key"],
                    )
                ] = dict(row["properties"])
