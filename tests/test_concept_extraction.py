import json
import shutil
from pathlib import Path

from oarag.cli import main
from oarag.graph.concept_extraction import (
    SOURCE_SIGNAL_TRANSCRIPT_STATEMENT,
    SOURCE_SIGNAL_DETECTED_TEXT,
    SOURCE_SIGNAL_SLIDE_TEXT,
    SOURCE_SIGNAL_TRANSCRIPT_MENTION,
    SOURCE_SIGNAL_VLM_VISUAL_DESCRIPTION,
    extract_project_concept_candidates,
)
from oarag.graph.concept_graph_schema import (
    CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH,
    CONCEPT_SOURCE_SIGNAL_CONTRACT,
    ConceptGraphConceptNode,
    ConceptGraphRelationEdge,
    load_concept_graph_artifact,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "public_concept_candidate_project"


def test_extract_project_concept_candidates_writes_public_safe_nodes(tmp_path: Path) -> None:
    project_dir = _copy_fixture(tmp_path)

    summary = extract_project_concept_candidates(project_dir=project_dir)

    artifact_path = project_dir / CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH
    records = load_concept_graph_artifact(artifact_path)
    concepts = {
        record.label: record for record in records if isinstance(record, ConceptGraphConceptNode)
    }
    concepts_by_id = {
        record.concept_id: record
        for record in records
        if isinstance(record, ConceptGraphConceptNode)
    }
    relations = [record for record in records if isinstance(record, ConceptGraphRelationEdge)]

    assert summary["counts"]["evidence_units_total"] == 3
    assert summary["counts"]["concept_candidates_total"] >= 6
    assert summary["counts"]["relation_edges_total"] >= 6
    assert summary["coverage"]["unit_coverage_ratio"] == 1.0
    assert "Gradient descent" in concepts
    assert "Learning rate" in concepts
    assert "Loss function" in concepts
    assert "Cross entropy loss" in concepts
    assert "Convolutional neural network" in concepts
    assert "Slide" not in concepts
    assert "Diagram" not in concepts
    assert "Model parameters" not in concepts

    assert "CNN" in concepts["Convolutional neural network"].aliases
    assert _aliases_include(concepts.values(), {"Soft max", "Softmax"})
    assert _aliases_include(concepts.values(), {"Drop out", "Dropout"})

    gradient_signals = _source_signals(concepts["Gradient descent"])
    assert SOURCE_SIGNAL_TRANSCRIPT_MENTION in gradient_signals
    assert SOURCE_SIGNAL_SLIDE_TEXT in gradient_signals
    assert SOURCE_SIGNAL_VLM_VISUAL_DESCRIPTION in gradient_signals

    learning_rate_signals = _source_signals(concepts["Learning rate"])
    assert SOURCE_SIGNAL_TRANSCRIPT_MENTION in learning_rate_signals
    assert SOURCE_SIGNAL_DETECTED_TEXT in learning_rate_signals

    for concept in concepts.values():
        assert concept.source_evidence_unit_ids
        assert concept.evidence_sources
        assert all(source.source_signal for source in concept.evidence_sources)
        assert all(not source.is_verified_object_alignment for source in concept.evidence_sources)

    assert _relation(concepts_by_id, relations, "Gradient descent", "uses", "Learning rate")
    assert _relation(
        concepts_by_id,
        relations,
        "Loss function",
        "defines",
        "Objective function",
    )
    assert _relation(
        concepts_by_id,
        relations,
        "Back propagation",
        "prerequisite_of",
        "Gradient descent",
    )
    assert _relation(concepts_by_id, relations, "Soft max", "contrasts_with", "Logits")
    assert _relation(
        concepts_by_id,
        relations,
        "Convolutional neural network",
        "visually_depicts",
        "Feature map",
    )

    for relation in relations:
        assert relation.evidence_unit_ids
        assert relation.source_signals
        assert relation.evidence_sources
        assert all(source.evidence_unit_id for source in relation.evidence_sources)
        assert all(source.source_signal for source in relation.evidence_sources)
        assert all(source.confidence is not None for source in relation.evidence_sources)
        assert all(not source.is_verified_object_alignment for source in relation.evidence_sources)

    uses_relation = _relation(
        concepts_by_id, relations, "Gradient descent", "uses", "Learning rate"
    )
    assert SOURCE_SIGNAL_TRANSCRIPT_STATEMENT in uses_relation.source_signals

    timestamp_relation = _relation(
        concepts_by_id,
        relations,
        "Gradient descent",
        "related_to",
        "Loss function",
    )
    assert timestamp_relation.relation_status == "candidate"
    assert timestamp_relation.source_signals == ("timestamp_overlap",)
    assert timestamp_relation.confidence <= 0.35
    assert summary["counts"]["timestamp_only_candidate_relation_edges_total"] >= 1
    assert summary["relation_extraction"]["timestamp_only_verified_demotions_total"] == 1

    summary_text = json.dumps(summary, ensure_ascii=False)
    artifact_text = artifact_path.read_text(encoding="utf-8")
    assert "updates model parameters" not in summary_text
    assert "updates model parameters" not in artifact_text
    assert "is defined as objective function" not in artifact_text
    assert str(project_dir) not in summary_text
    assert "/private" not in summary_text
    assert "/private" not in artifact_text

    manifest = json.loads((project_dir / "manifests" / "project_manifest.json").read_text())
    assert manifest["artifacts"]["concept_graph"] == "manifests/concept_graph.jsonl"
    assert manifest["counts"]["concept_candidates"] == summary["counts"]["concept_candidates_total"]
    assert manifest["counts"]["concept_relations"] == summary["counts"]["relation_edges_total"]


def test_concept_candidate_source_signals_are_in_schema_contract() -> None:
    assert CONCEPT_SOURCE_SIGNAL_CONTRACT[SOURCE_SIGNAL_TRANSCRIPT_MENTION]["status"] == (
        "candidate_or_verified"
    )
    assert CONCEPT_SOURCE_SIGNAL_CONTRACT[SOURCE_SIGNAL_SLIDE_TEXT]["status"] == (
        "candidate_or_verified"
    )
    assert CONCEPT_SOURCE_SIGNAL_CONTRACT[SOURCE_SIGNAL_DETECTED_TEXT]["status"] == (
        "candidate_or_verified"
    )
    assert CONCEPT_SOURCE_SIGNAL_CONTRACT[SOURCE_SIGNAL_VLM_VISUAL_DESCRIPTION]["status"] == (
        "candidate_or_verified"
    )
    assert CONCEPT_SOURCE_SIGNAL_CONTRACT[SOURCE_SIGNAL_TRANSCRIPT_STATEMENT]["status"] == (
        "candidate_or_verified"
    )


def test_extract_concept_candidates_cli_smoke_on_public_fixture(
    tmp_path: Path,
    capsys,
) -> None:
    project_dir = _copy_fixture(tmp_path)

    main(["extract-concept-candidates", "--project-dir", str(project_dir)])

    summary = json.loads(capsys.readouterr().out)
    artifact_path = project_dir / CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH
    records = load_concept_graph_artifact(artifact_path)
    labels = {record.label for record in records if isinstance(record, ConceptGraphConceptNode)}

    assert summary["lecture_id"] == "public_concept_candidate_lecture"
    assert summary["paths"]["concept_graph"] == "manifests/concept_graph.jsonl"
    assert "Gradient descent" in labels
    assert "raw_text_redacted" in json.dumps(summary)


def _copy_fixture(tmp_path: Path) -> Path:
    project_dir = tmp_path / "public_concept_candidate_project"
    shutil.copytree(FIXTURE_DIR, project_dir)
    return project_dir


def _source_signals(concept: ConceptGraphConceptNode) -> set[str]:
    return {source.source_signal for source in concept.evidence_sources}


def _aliases_include(
    concepts: object,
    expected_labels: set[str],
) -> bool:
    for concept in concepts:
        labels = {concept.label, *concept.aliases}
        if expected_labels.issubset(labels):
            return True
    return False


def _relation(
    concepts_by_id: dict[str, ConceptGraphConceptNode],
    relations: list[ConceptGraphRelationEdge],
    source_label: str,
    relation_type: str,
    target_label: str,
) -> ConceptGraphRelationEdge:
    for relation in relations:
        source = concepts_by_id[relation.source_concept_id]
        target = concepts_by_id[relation.target_concept_id]
        if (
            source.label == source_label
            and relation.relation_type == relation_type
            and target.label == target_label
        ):
            return relation
    raise AssertionError(f"missing relation: {source_label} {relation_type} {target_label}")
