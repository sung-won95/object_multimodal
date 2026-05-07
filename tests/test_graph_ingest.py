from __future__ import annotations

import json
from pathlib import Path

import pytest

from oarag.graph_ingest import (
    build_graph_ingest_plan,
    ingest_project_graph,
    run_graph_ingest_plan,
)


def test_build_graph_ingest_plan_groups_idempotent_node_and_relationship_merges() -> None:
    document = _graph_document()

    plan = build_graph_ingest_plan(document)

    assert plan.schema_statements[0].cypher.startswith("CREATE CONSTRAINT graph_node_key")
    assert "IF NOT EXISTS" in plan.schema_statements[0].cypher
    assert any("FOR ()-[r:HAS_SEGMENT]-() ON (r.key)" in item.cypher for item in plan.schema_statements)
    assert len(plan.node_statements) == 2
    assert len(plan.relationship_statements) == 1
    assert plan.skipped_relationships == ()
    project_statement = plan.node_statements[0]
    segment_statement = plan.node_statements[1]
    assert "MERGE (n:GraphNode {key: row.key})" in project_statement.cypher
    assert "SET n:Project" in project_statement.cypher
    assert project_statement.parameters == {
        "nodes": [{"key": "project:sample", "properties": {"project_id": "sample"}}]
    }
    assert "SET n:Segment" in segment_statement.cypher
    assert segment_statement.parameters["nodes"][0]["properties"]["bbox"] == '{"left": 1, "top": 2}'
    relationship_statement = plan.relationship_statements[0]
    assert "MATCH (start:GraphNode {key: row.start_node_key})" in relationship_statement.cypher
    assert "MERGE (start)-[r:HAS_SEGMENT {key: row.key}]->(end)" in relationship_statement.cypher
    assert relationship_statement.parameters["relationships"] == [
        {
            "key": "rel:HAS_SEGMENT:project:sample:segment:s1",
            "type": "HAS_SEGMENT",
            "start_node_key": "project:sample",
            "end_node_key": "segment:s1",
            "properties": {"source": "test"},
        }
    ]


def test_build_graph_ingest_plan_skips_relationships_with_missing_endpoints() -> None:
    document = _graph_document()
    document["relationships"].append(
        {
            "key": "rel:HAS_SEGMENT:project:sample:segment:missing",
            "type": "HAS_SEGMENT",
            "start_node_key": "project:sample",
            "end_node_key": "segment:missing",
            "properties": {},
        }
    )

    plan = build_graph_ingest_plan(document)

    assert len(plan.relationship_statements[0].parameters["relationships"]) == 1
    assert plan.skipped_relationships == (
        {
            "key": "rel:HAS_SEGMENT:project:sample:segment:missing",
            "type": "HAS_SEGMENT",
            "missing_node_keys": ["segment:missing"],
        },
    )


def test_run_graph_ingest_plan_executes_schema_before_merges() -> None:
    plan = build_graph_ingest_plan(_graph_document())
    session = FakeSession()

    count = run_graph_ingest_plan(session, plan)

    assert count == len(plan.statements)
    assert session.calls[0][0].startswith("CREATE CONSTRAINT graph_node_key")
    assert "MERGE (n:GraphNode" in session.calls[len(plan.schema_statements)][0]


def test_ingest_project_graph_dry_run_loads_document_and_reports_summary(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph_document.json"
    graph_path.write_text(json.dumps(_graph_document()), encoding="utf-8")

    summary = ingest_project_graph(graph_document_path=graph_path, dry_run=True)

    assert summary["project_id"] == "sample"
    assert summary["dry_run"] is True
    assert summary["nodes"]["by_label"] == {"Project": 1, "Segment": 1}
    assert summary["relationships"]["by_type"] == {"HAS_SEGMENT": 1}
    assert summary["missing_artifacts"] == {"count": 1, "names": ["frames_manifest"]}
    assert summary["statements_executed"] == 0


def test_build_graph_ingest_plan_rejects_unsafe_labels_and_types() -> None:
    document = _graph_document()
    document["nodes"][0]["labels"] = ["Project`) DETACH DELETE n //"]

    with pytest.raises(ValueError, match="Invalid Neo4j node label"):
        build_graph_ingest_plan(document)

    document = _graph_document()
    document["relationships"][0]["type"] = "HAS SEGMENT"

    with pytest.raises(ValueError, match="Invalid Neo4j relationship type"):
        build_graph_ingest_plan(document)


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, parameters: dict | None = None) -> None:
        self.calls.append((query, parameters or {}))


def _graph_document() -> dict:
    return {
        "schema_version": "graph-document-v1",
        "project_id": "sample",
        "nodes": [
            {
                "key": "project:sample",
                "labels": ["Project"],
                "properties": {"project_id": "sample"},
            },
            {
                "key": "segment:s1",
                "labels": ["Segment"],
                "properties": {
                    "segment_id": "s1",
                    "project_id": "sample",
                    "bbox": {"left": 1, "top": 2},
                },
            },
        ],
        "relationships": [
            {
                "key": "rel:HAS_SEGMENT:project:sample:segment:s1",
                "type": "HAS_SEGMENT",
                "start_node_key": "project:sample",
                "end_node_key": "segment:s1",
                "properties": {"source": "test"},
            }
        ],
        "availability": {
            "artifacts": {
                "segments": {"available": True},
                "frames_manifest": {"available": False},
            }
        },
        "counts": {
            "nodes": 2,
            "relationships": 1,
            "nodes_by_label": {"Project": 1, "Segment": 1},
            "relationships_by_type": {"HAS_SEGMENT": 1},
        },
    }
