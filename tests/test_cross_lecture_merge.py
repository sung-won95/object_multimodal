from __future__ import annotations

import json
from pathlib import Path

from oarag.graph.cross_lecture_merge import (
    GLOBAL_CONCEPT_MERGE_SCHEMA_VERSION,
    build_global_concept_graph_document,
    merge_cross_lecture_concepts,
)
from oarag.graph.graph_ingest import build_graph_ingest_plan
from oarag.graph.concept_graph_schema import load_concept_graph_artifact


FIXTURE_DIR = (
    Path(__file__).parent
    / "fixtures"
    / "public_cross_lecture_concept_project"
    / "manifests"
)
FIXTURE_PATHS = (
    FIXTURE_DIR / "lecture_a_concept_graph.jsonl",
    FIXTURE_DIR / "lecture_b_concept_graph.jsonl",
    FIXTURE_DIR / "lecture_c_concept_graph.jsonl",
    FIXTURE_DIR / "lecture_d_concept_graph.jsonl",
)


def test_cross_lecture_merge_builds_global_concepts_and_relations() -> None:
    records = [load_concept_graph_artifact(path) for path in FIXTURE_PATHS]

    document = build_global_concept_graph_document(records, project_id="public_cross_merge")

    assert document["schema_version"] == GLOBAL_CONCEPT_MERGE_SCHEMA_VERSION
    assert document["counts"]["lecture_graphs"] == 4
    assert document["counts"]["local_concepts"] == 8
    assert document["counts"]["global_concepts"] == 6
    assert document["counts"]["merged_global_concepts"] == 2
    assert document["counts"]["merge_decisions"] == 2
    assert document["counts"]["conflicts"] == 1
    assert document["counts"]["global_relations"] == 3
    assert document["counts"]["mapping_relations"] == 8

    merged_ids = {
        decision["global_concept_id"]
        for decision in document["decisions"]
        if len(decision["local_concept_refs"]) == 2
    }
    assert {"global_gradient_descent", "global_learning_rate"} <= merged_ids
    assert all(decision["source_evidence_refs"] for decision in document["decisions"])

    global_uses = [
        relationship
        for relationship in document["relationships"]
        if relationship["type"] == "GLOBAL_USES"
    ]
    optimizer_relation = next(
        relationship
        for relationship in global_uses
        if relationship["start_node_key"] == "global_concept:global_gradient_descent"
    )
    assert optimizer_relation["end_node_key"] == "global_concept:global_learning_rate"
    assert optimizer_relation["properties"]["local_relation_count"] == 2
    assert optimizer_relation["properties"]["source_evidence_refs"]

    plan = build_graph_ingest_plan(document)

    assert plan.skipped_relationships == ()
    assert any("MERGE (start)-[r:GLOBAL_USES" in stmt.cypher for stmt in plan.statements)


def test_same_alias_with_different_context_is_recorded_as_conflict() -> None:
    records = [load_concept_graph_artifact(path) for path in FIXTURE_PATHS]

    document = build_global_concept_graph_document(records)

    conflict = document["conflicts"][0]
    assert conflict["reasons"] == [
        "same_label_or_alias_without_merge",
        "missing_relation_context_overlap",
        "concept_type_mismatch",
    ]
    assert conflict["score"]["guards"]["has_label_or_alias_match"] is True
    assert conflict["score"]["guards"]["has_relation_context_evidence"] is False

    mappings = [
        relationship
        for relationship in document["relationships"]
        if relationship["type"] == "INSTANCE_OF_GLOBAL_CONCEPT"
    ]
    kernel_targets = {
        relationship["end_node_key"]
        for relationship in mappings
        if relationship["properties"]["concept_id"] in {"concept_kernel_visual", "concept_kernel_os"}
    }
    assert len(kernel_targets) == 2


def test_project_id_is_propagated_to_all_graph_properties() -> None:
    records = [load_concept_graph_artifact(path) for path in FIXTURE_PATHS]

    document = build_global_concept_graph_document(records, project_id="public_cross_merge")

    assert document["project_id"] == "public_cross_merge"
    assert {tuple(node["labels"]) for node in document["nodes"]} >= {
        ("GraphNode", "GlobalConcept"),
        ("GraphNode", "Concept"),
    }
    assert {
        relationship["type"]
        for relationship in document["relationships"]
        if relationship["type"].startswith("GLOBAL_")
    } == {"GLOBAL_USES"}
    assert any(
        relationship["type"] == "INSTANCE_OF_GLOBAL_CONCEPT"
        for relationship in document["relationships"]
    )
    assert all(
        node["properties"].get("project_id") == "public_cross_merge"
        for node in document["nodes"]
    )
    assert all(
        relationship["properties"].get("project_id") == "public_cross_merge"
        for relationship in document["relationships"]
    )


def test_merge_cross_lecture_concepts_writes_counts_only_summary(tmp_path: Path) -> None:
    output_path = tmp_path / "global_concept_graph.json"
    summary_path = tmp_path / "global_concept_merge_summary.json"

    summary = merge_cross_lecture_concepts(
        concept_graph_paths=FIXTURE_PATHS,
        output_path=output_path,
        summary_path=summary_path,
        project_id="public_cross_merge",
    )

    assert output_path.exists()
    assert summary_path.exists()
    assert summary["counts"]["merged_global_concepts"] == 2
    assert summary["counts"]["conflicts"] == 1

    rendered_summary = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    persisted_summary = summary_path.read_text(encoding="utf-8")
    assert "Gradient descent" not in rendered_summary
    assert "Kernel" not in rendered_summary
    assert "seg_public" not in rendered_summary
    assert str(FIXTURE_DIR) not in rendered_summary
    assert "Gradient descent" not in persisted_summary
    assert "Kernel" not in persisted_summary

    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["public_summary"]["level"] == "counts_only"
    assert document["public_summary"]["counts"]["conflicts"] == 1
