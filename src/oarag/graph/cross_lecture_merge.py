from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from oarag.core.io import write_json
from oarag.core.schemas import slugify
from oarag.graph.concept_graph_schema import (
    CONCEPT_GRAPH_SCHEMA_VERSION,
    TIMESTAMP_ONLY_SOURCE_SIGNALS,
    ConceptGraphConceptNode,
    ConceptGraphEvidenceSource,
    ConceptGraphRecord,
    ConceptGraphRelationEdge,
    load_concept_graph_artifact,
)


GLOBAL_CONCEPT_MERGE_SCHEMA_VERSION = "oarag-global-concept-merge-v1"
GLOBAL_CONCEPT_GRAPH_ARTIFACT_NAME = "global_concept_graph"
GLOBAL_CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH = Path("manifests") / "global_concept_graph.json"
GLOBAL_CONCEPT_MERGE_SUMMARY_RELATIVE_PATH = (
    Path("manifests") / "global_concept_merge_summary.json"
)

DEFAULT_MERGE_THRESHOLD = 0.74
DEFAULT_CONFLICT_THRESHOLD = 0.42

GLOBAL_RELATION_TYPE_PREFIX = "GLOBAL_"
INSTANCE_OF_GLOBAL_CONCEPT_RELATION = "INSTANCE_OF_GLOBAL_CONCEPT"

_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+")


@dataclass(frozen=True)
class CrossLectureMergeConfig:
    merge_threshold: float = DEFAULT_MERGE_THRESHOLD
    conflict_threshold: float = DEFAULT_CONFLICT_THRESHOLD
    require_relation_context: bool = True


@dataclass(frozen=True)
class LocalConceptRef:
    lecture_id: str
    concept_id: str

    @property
    def key(self) -> str:
        return f"{self.lecture_id}:{self.concept_id}"

    def to_dict(self) -> dict[str, str]:
        return {"lecture_id": self.lecture_id, "concept_id": self.concept_id}


@dataclass(frozen=True)
class MergeScore:
    score: float
    label_alias_score: float
    source_evidence_score: float
    relation_context_score: float
    concept_type_score: float
    has_label_or_alias_match: bool
    has_relation_context_evidence: bool
    reasons: tuple[str, ...]
    relation_context_overlap_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "score_breakdown": {
                "label_alias": self.label_alias_score,
                "source_evidence": self.source_evidence_score,
                "relation_context": self.relation_context_score,
                "concept_type": self.concept_type_score,
            },
            "guards": {
                "has_label_or_alias_match": self.has_label_or_alias_match,
                "has_relation_context_evidence": self.has_relation_context_evidence,
            },
            "relation_context_overlap_count": self.relation_context_overlap_count,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class MergeDecision:
    decision_id: str
    global_concept_id: str
    local_concept_refs: tuple[LocalConceptRef, ...]
    score: float
    score_breakdown: dict[str, float]
    relation_context_overlap_count: int
    source_evidence_refs: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "decision_type": "merge",
            "global_concept_id": self.global_concept_id,
            "local_concept_refs": [ref.to_dict() for ref in self.local_concept_refs],
            "score": self.score,
            "score_breakdown": self.score_breakdown,
            "relation_context_overlap_count": self.relation_context_overlap_count,
            "source_evidence_refs": list(self.source_evidence_refs),
            "public_safe": True,
        }


@dataclass(frozen=True)
class MergeConflict:
    conflict_id: str
    local_concept_refs: tuple[LocalConceptRef, LocalConceptRef]
    score: MergeScore
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "local_concept_refs": [ref.to_dict() for ref in self.local_concept_refs],
            "score": self.score.to_dict(),
            "reasons": list(self.reasons),
            "public_safe": True,
        }


@dataclass
class _LocalConcept:
    concept: ConceptGraphConceptNode
    path_index: int
    labels: frozenset[str]
    source_signals: frozenset[str]
    source_types: frozenset[str]
    relation_context: frozenset[str] = field(default_factory=frozenset)

    @property
    def ref(self) -> LocalConceptRef:
        return LocalConceptRef(
            lecture_id=self.concept.lecture_id,
            concept_id=self.concept.concept_id,
        )

    @property
    def key(self) -> str:
        return self.ref.key


class _UnionFind:
    def __init__(self, keys: Iterable[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def merge_cross_lecture_concepts(
    *,
    concept_graph_paths: Sequence[Path],
    output_path: Path | None = None,
    summary_path: Path | None = None,
    project_id: str | None = None,
    config: CrossLectureMergeConfig | None = None,
) -> dict[str, Any]:
    """Merge lecture-local concept graphs into a global graph document artifact."""
    if not concept_graph_paths:
        raise ValueError("At least one concept graph path is required")

    records_by_path = [
        (path.expanduser(), load_concept_graph_artifact(path))
        for path in concept_graph_paths
    ]
    document = build_global_concept_graph_document(
        [records for _, records in records_by_path],
        project_id=project_id,
        config=config,
    )

    resolved_output = output_path.expanduser() if output_path else None
    if resolved_output is not None:
        write_json(resolved_output, document)

    summary = public_merge_summary(
        document,
        output_path=resolved_output,
        summary_path=summary_path.expanduser() if summary_path else None,
    )
    if summary_path is not None:
        write_json(summary_path.expanduser(), summary)
    return summary


def build_global_concept_graph_document(
    concept_graphs: Sequence[Sequence[ConceptGraphRecord]],
    *,
    project_id: str | None = None,
    config: CrossLectureMergeConfig | None = None,
) -> dict[str, Any]:
    active_config = config or CrossLectureMergeConfig()
    local_concepts, relations = _collect_local_graphs(concept_graphs)
    scored_pairs = _score_candidate_pairs(local_concepts, active_config)
    union_find = _UnionFind(local_concepts)
    conflicts: list[MergeConflict] = []

    for left_key, right_key, score in scored_pairs:
        if _should_merge(score, active_config):
            union_find.union(left_key, right_key)
        elif _is_conflict_candidate(score, active_config):
            conflicts.append(
                _merge_conflict(
                    local_concepts[left_key],
                    local_concepts[right_key],
                    score,
                    active_config,
                )
            )

    clusters = _clusters(local_concepts, union_find)
    global_id_by_local_key, global_nodes, decisions = _build_global_nodes_and_decisions(
        clusters,
        local_concepts,
        scored_pairs,
    )
    local_nodes = [_local_concept_node(item) for item in local_concepts.values()]
    mapping_relationships = _mapping_relationships(
        local_concepts=local_concepts,
        global_id_by_local_key=global_id_by_local_key,
        scored_pairs=scored_pairs,
    )
    global_relationships = _global_relation_relationships(
        relations=relations,
        global_id_by_local_key=global_id_by_local_key,
        local_concepts=local_concepts,
    )
    nodes = sorted([*global_nodes, *local_nodes], key=lambda row: row["key"])
    relationships = sorted(
        [*mapping_relationships, *global_relationships],
        key=lambda row: row["key"],
    )
    counts = _counts(
        nodes=nodes,
        relationships=relationships,
        local_concepts=local_concepts,
        clusters=clusters,
        decisions=decisions,
        conflicts=conflicts,
        graph_count=len(concept_graphs),
    )
    return {
        "schema_version": GLOBAL_CONCEPT_MERGE_SCHEMA_VERSION,
        "concept_graph_schema_version": CONCEPT_GRAPH_SCHEMA_VERSION,
        "artifact_name": GLOBAL_CONCEPT_GRAPH_ARTIFACT_NAME,
        "project_id": project_id,
        "nodes": nodes,
        "relationships": relationships,
        "decisions": [decision.to_dict() for decision in decisions],
        "conflicts": [conflict.to_dict() for conflict in conflicts],
        "counts": counts,
        "public_summary": _counts_only_public_summary(counts),
        "public_safety": {
            "raw_text_redacted": True,
            "raw_frame_paths_redacted": True,
            "local_absolute_paths_redacted": True,
            "summary_level": "counts_only",
        },
    }


def public_merge_summary(
    document: dict[str, Any],
    *,
    output_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    counts = document.get("counts") if isinstance(document.get("counts"), dict) else {}
    paths: dict[str, str] = {}
    if output_path is not None:
        paths["global_concept_graph"] = output_path.name
    if summary_path is not None:
        paths["summary"] = summary_path.name
    return {
        "schema_version": GLOBAL_CONCEPT_MERGE_SCHEMA_VERSION,
        "artifact": GLOBAL_CONCEPT_GRAPH_ARTIFACT_NAME,
        "paths": paths,
        "counts": _counts_only(counts),
        "public_safety": {
            "level": "counts_only",
            "omits": ["raw_transcript", "local_paths", "frame_paths", "secrets"],
        },
    }


def load_global_concept_graph_document(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected global concept graph JSON object in {path}")
    if payload.get("schema_version") != GLOBAL_CONCEPT_MERGE_SCHEMA_VERSION:
        raise ValueError(
            "Expected schema_version "
            f"{GLOBAL_CONCEPT_MERGE_SCHEMA_VERSION!r}, got {payload.get('schema_version')!r}"
        )
    return payload


def _collect_local_graphs(
    concept_graphs: Sequence[Sequence[ConceptGraphRecord]],
) -> tuple[dict[str, _LocalConcept], list[ConceptGraphRelationEdge]]:
    local_concepts: dict[str, _LocalConcept] = {}
    relations: list[ConceptGraphRelationEdge] = []
    relations_by_lecture: dict[str, list[ConceptGraphRelationEdge]] = {}
    concept_by_ref: dict[tuple[str, str], ConceptGraphConceptNode] = {}

    for path_index, records in enumerate(concept_graphs):
        for record in records:
            if isinstance(record, ConceptGraphConceptNode):
                key = (record.lecture_id, record.concept_id)
                if key in concept_by_ref:
                    raise ValueError(f"Duplicate lecture concept: {record.lecture_id}:{record.concept_id}")
                concept_by_ref[key] = record
                labels = frozenset(_normalized_labels(record))
                local_concepts[f"{record.lecture_id}:{record.concept_id}"] = _LocalConcept(
                    concept=record,
                    path_index=path_index,
                    labels=labels,
                    source_signals=frozenset(_concept_source_signals(record.evidence_sources)),
                    source_types=frozenset(_source_type_set(record.evidence_sources)),
                )
            elif isinstance(record, ConceptGraphRelationEdge):
                relations.append(record)
                relations_by_lecture.setdefault(record.lecture_id, []).append(record)

    for key, local in list(local_concepts.items()):
        relation_context = _relation_context_for_concept(
            local.concept,
            relations_by_lecture.get(local.concept.lecture_id, []),
            concept_by_ref,
        )
        local_concepts[key] = _LocalConcept(
            concept=local.concept,
            path_index=local.path_index,
            labels=local.labels,
            source_signals=local.source_signals,
            source_types=local.source_types,
            relation_context=frozenset(relation_context),
        )

    return local_concepts, relations


def _score_candidate_pairs(
    local_concepts: dict[str, _LocalConcept],
    config: CrossLectureMergeConfig,
) -> list[tuple[str, str, MergeScore]]:
    del config
    pairs: list[tuple[str, str, MergeScore]] = []
    items = sorted(local_concepts.values(), key=lambda item: item.key)
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if left.concept.lecture_id == right.concept.lecture_id:
                continue
            if not left.labels.intersection(right.labels):
                continue
            score = _merge_score(left, right)
            pairs.append((left.key, right.key, score))
    return pairs


def _merge_score(left: _LocalConcept, right: _LocalConcept) -> MergeScore:
    label_overlap = left.labels.intersection(right.labels)
    label_alias_score = 0.35 if _normalized_label(left.concept.label) in label_overlap else 0.3
    if _normalized_label(left.concept.label) == _normalized_label(right.concept.label):
        label_alias_score = 0.4

    source_signal_overlap = left.source_signals.intersection(right.source_signals)
    source_type_overlap = left.source_types.intersection(right.source_types)
    source_evidence_score = 0.0
    if source_signal_overlap:
        source_evidence_score += 0.08
    if source_type_overlap:
        source_evidence_score += 0.06
    if len(source_signal_overlap) >= 2:
        source_evidence_score += 0.04
    source_evidence_score = min(0.16, source_evidence_score)

    relation_context_overlap = left.relation_context.intersection(right.relation_context)
    relation_context_score = min(0.34, 0.18 * len(relation_context_overlap))
    concept_type_score = 0.1 if left.concept.concept_type == right.concept.concept_type else 0.0

    score = round(
        min(1.0, label_alias_score + source_evidence_score + relation_context_score + concept_type_score),
        3,
    )
    reasons: list[str] = []
    if label_overlap:
        reasons.append("label_or_alias_match")
    if source_signal_overlap or source_type_overlap:
        reasons.append("source_evidence_overlap")
    if relation_context_overlap:
        reasons.append("relation_context_overlap")
    if concept_type_score:
        reasons.append("concept_type_match")
    return MergeScore(
        score=score,
        label_alias_score=round(label_alias_score, 3),
        source_evidence_score=round(source_evidence_score, 3),
        relation_context_score=round(relation_context_score, 3),
        concept_type_score=concept_type_score,
        has_label_or_alias_match=bool(label_overlap),
        has_relation_context_evidence=bool(relation_context_overlap),
        reasons=tuple(reasons),
        relation_context_overlap_count=len(relation_context_overlap),
    )


def _should_merge(score: MergeScore, config: CrossLectureMergeConfig) -> bool:
    if score.score < config.merge_threshold:
        return False
    if not score.has_label_or_alias_match:
        return False
    if config.require_relation_context and not score.has_relation_context_evidence:
        return False
    return True


def _is_conflict_candidate(score: MergeScore, config: CrossLectureMergeConfig) -> bool:
    return (
        score.has_label_or_alias_match
        and score.score >= config.conflict_threshold
        and not _should_merge(score, config)
    )


def _merge_conflict(
    left: _LocalConcept,
    right: _LocalConcept,
    score: MergeScore,
    config: CrossLectureMergeConfig,
) -> MergeConflict:
    reasons = ["same_label_or_alias_without_merge"]
    if config.require_relation_context and not score.has_relation_context_evidence:
        reasons.append("missing_relation_context_overlap")
    if left.concept.concept_type != right.concept.concept_type:
        reasons.append("concept_type_mismatch")
    return MergeConflict(
        conflict_id=_stable_id("conflict", left.key, right.key),
        local_concept_refs=(left.ref, right.ref),
        score=score,
        reasons=tuple(reasons),
    )


def _clusters(
    local_concepts: dict[str, _LocalConcept],
    union_find: _UnionFind,
) -> list[list[str]]:
    by_root: dict[str, list[str]] = {}
    for key in sorted(local_concepts):
        by_root.setdefault(union_find.find(key), []).append(key)
    return sorted(by_root.values(), key=lambda group: (len(group) == 1, group[0]))


def _build_global_nodes_and_decisions(
    clusters: list[list[str]],
    local_concepts: dict[str, _LocalConcept],
    scored_pairs: list[tuple[str, str, MergeScore]],
) -> tuple[dict[str, str], list[dict[str, Any]], list[MergeDecision]]:
    pair_scores = {
        frozenset((left, right)): score
        for left, right, score in scored_pairs
    }
    global_id_by_local_key: dict[str, str] = {}
    used_global_ids: set[str] = set()
    nodes: list[dict[str, Any]] = []
    decisions: list[MergeDecision] = []

    for cluster in clusters:
        concepts = [local_concepts[key].concept for key in cluster]
        global_concept_id = _unique_global_concept_id(
            _global_concept_id(concepts, cluster),
            cluster,
            used_global_ids,
        )
        used_global_ids.add(global_concept_id)
        for key in cluster:
            global_id_by_local_key[key] = global_concept_id
        confidence = round(sum(concept.confidence for concept in concepts) / len(concepts), 3)
        canonical_label = _canonical_label(concepts)
        source_evidence_refs = _source_evidence_refs_for_concepts(concepts)
        nodes.append(
            _node(
                key=_global_concept_key(global_concept_id),
                labels=("GraphNode", "GlobalConcept"),
                properties=_compact(
                    {
                        "global_concept_id": global_concept_id,
                        "canonical_label": canonical_label,
                        "aliases": _global_aliases(concepts),
                        "normalized_labels": _unique_sorted(
                            label for concept in concepts for label in _normalized_labels(concept)
                        ),
                        "concept_types": _unique_sorted(concept.concept_type for concept in concepts),
                        "source_lecture_ids": _unique_sorted(concept.lecture_id for concept in concepts),
                        "local_concept_refs": [
                            {"lecture_id": concept.lecture_id, "concept_id": concept.concept_id}
                            for concept in concepts
                        ],
                        "local_concept_count": len(concepts),
                        "source_evidence_refs": source_evidence_refs,
                        "confidence": confidence,
                        "merge_status": "merged" if len(concepts) > 1 else "singleton",
                        "source": GLOBAL_CONCEPT_GRAPH_ARTIFACT_NAME,
                    }
                ),
            )
        )
        if len(cluster) <= 1:
            continue
        score_values = [
            pair_scores[frozenset((left, right))]
            for index, left in enumerate(cluster)
            for right in cluster[index + 1 :]
            if frozenset((left, right)) in pair_scores
        ]
        decisions.append(
            MergeDecision(
                decision_id=_stable_id("merge", *cluster),
                global_concept_id=global_concept_id,
                local_concept_refs=tuple(local_concepts[key].ref for key in cluster),
                score=round(sum(score.score for score in score_values) / len(score_values), 3),
                score_breakdown=_average_score_breakdown(score_values),
                relation_context_overlap_count=sum(
                    score.relation_context_overlap_count for score in score_values
                ),
                source_evidence_refs=tuple(source_evidence_refs),
            )
        )

    return global_id_by_local_key, nodes, decisions


def _local_concept_node(local: _LocalConcept) -> dict[str, Any]:
    concept = local.concept
    return _node(
        key=_local_concept_key(concept.lecture_id, concept.concept_id),
        labels=("GraphNode", "Concept"),
        properties=_compact(
            {
                "concept_id": concept.concept_id,
                "lecture_id": concept.lecture_id,
                "label": concept.label,
                "aliases": list(concept.aliases),
                "concept_type": concept.concept_type,
                "confidence": concept.confidence,
                "source_evidence_unit_ids": list(concept.source_evidence_unit_ids),
                "source": GLOBAL_CONCEPT_GRAPH_ARTIFACT_NAME,
            }
        ),
    )


def _mapping_relationships(
    *,
    local_concepts: dict[str, _LocalConcept],
    global_id_by_local_key: dict[str, str],
    scored_pairs: list[tuple[str, str, MergeScore]],
) -> list[dict[str, Any]]:
    score_by_key = _best_score_by_local_key(scored_pairs)
    relationships: list[dict[str, Any]] = []
    for key, local in sorted(local_concepts.items()):
        global_concept_id = global_id_by_local_key[key]
        best_score = score_by_key.get(key)
        relationships.append(
            _relationship(
                key=(
                    f"rel:{INSTANCE_OF_GLOBAL_CONCEPT_RELATION}:"
                    f"{local.concept.lecture_id}:{local.concept.concept_id}:{global_concept_id}"
                ),
                relationship_type=INSTANCE_OF_GLOBAL_CONCEPT_RELATION,
                start_node_key=_local_concept_key(local.concept.lecture_id, local.concept.concept_id),
                end_node_key=_global_concept_key(global_concept_id),
                properties=_compact(
                    {
                        "lecture_id": local.concept.lecture_id,
                        "concept_id": local.concept.concept_id,
                        "global_concept_id": global_concept_id,
                        "merge_score": best_score.score if best_score else None,
                        "score_breakdown": (
                            best_score.to_dict()["score_breakdown"] if best_score else None
                        ),
                        "source_evidence_refs": _source_evidence_refs(
                            local.concept.lecture_id,
                            local.concept.evidence_sources,
                        ),
                        "source": GLOBAL_CONCEPT_GRAPH_ARTIFACT_NAME,
                    }
                ),
            )
        )
    return relationships


def _global_relation_relationships(
    *,
    relations: list[ConceptGraphRelationEdge],
    global_id_by_local_key: dict[str, str],
    local_concepts: dict[str, _LocalConcept],
) -> list[dict[str, Any]]:
    aggregate: dict[tuple[str, str, str], dict[str, Any]] = {}
    for relation in relations:
        source_key = f"{relation.lecture_id}:{relation.source_concept_id}"
        target_key = f"{relation.lecture_id}:{relation.target_concept_id}"
        if source_key not in local_concepts or target_key not in local_concepts:
            continue
        source_global = global_id_by_local_key[source_key]
        target_global = global_id_by_local_key[target_key]
        if source_global == target_global:
            continue
        relationship_type = f"{GLOBAL_RELATION_TYPE_PREFIX}{_relationship_label(relation.relation_type)}"
        key = (source_global, relationship_type, target_global)
        item = aggregate.setdefault(
            key,
            {
                "relation_type": relation.relation_type,
                "relationship_type": relationship_type,
                "source_global": source_global,
                "target_global": target_global,
                "local_relation_refs": [],
                "source_evidence_refs": [],
                "source_signals": set(),
                "confidence_values": [],
                "relation_statuses": set(),
            },
        )
        item["local_relation_refs"].append(
            {
                "lecture_id": relation.lecture_id,
                "edge_id": relation.edge_id,
                "source_concept_id": relation.source_concept_id,
                "target_concept_id": relation.target_concept_id,
            }
        )
        item["source_evidence_refs"].extend(
            _source_evidence_refs(relation.lecture_id, relation.evidence_sources)
        )
        item["source_signals"].update(relation.source_signals)
        item["confidence_values"].append(relation.confidence)
        item["relation_statuses"].add(relation.relation_status)

    relationships: list[dict[str, Any]] = []
    for (_, relationship_type, _), item in sorted(aggregate.items()):
        source_global = item["source_global"]
        target_global = item["target_global"]
        confidence_values = item["confidence_values"]
        relation_statuses = item["relation_statuses"]
        relationships.append(
            _relationship(
                key=f"rel:{relationship_type}:{source_global}:{target_global}",
                relationship_type=relationship_type,
                start_node_key=_global_concept_key(source_global),
                end_node_key=_global_concept_key(target_global),
                properties={
                    "relation_type": item["relation_type"],
                    "global_source_concept_id": source_global,
                    "global_target_concept_id": target_global,
                    "local_relation_refs": item["local_relation_refs"],
                    "source_evidence_refs": _unique_dicts(item["source_evidence_refs"]),
                    "source_signals": sorted(item["source_signals"]),
                    "confidence": round(max(confidence_values), 3) if confidence_values else None,
                    "relation_status": (
                        "verified" if "verified" in relation_statuses else "candidate"
                    ),
                    "local_relation_count": len(item["local_relation_refs"]),
                    "source": GLOBAL_CONCEPT_GRAPH_ARTIFACT_NAME,
                },
            )
        )
    return relationships


def _relation_context_for_concept(
    concept: ConceptGraphConceptNode,
    relations: list[ConceptGraphRelationEdge],
    concept_by_ref: dict[tuple[str, str], ConceptGraphConceptNode],
) -> set[str]:
    context: set[str] = set()
    for relation in relations:
        if _timestamp_only_context(relation):
            continue
        if relation.source_concept_id == concept.concept_id:
            neighbor = concept_by_ref.get((concept.lecture_id, relation.target_concept_id))
            if neighbor is not None:
                for label in _normalized_labels(neighbor):
                    context.add(f"out:{relation.relation_type}:{label}")
        if relation.target_concept_id == concept.concept_id:
            neighbor = concept_by_ref.get((concept.lecture_id, relation.source_concept_id))
            if neighbor is not None:
                for label in _normalized_labels(neighbor):
                    context.add(f"in:{relation.relation_type}:{label}")
    return context


def _timestamp_only_context(relation: ConceptGraphRelationEdge) -> bool:
    return bool(relation.source_signals) and all(
        signal in TIMESTAMP_ONLY_SOURCE_SIGNALS for signal in relation.source_signals
    )


def _normalized_labels(concept: ConceptGraphConceptNode) -> list[str]:
    return _unique_sorted(
        label
        for label in (_normalized_label(concept.label), *(_normalized_label(alias) for alias in concept.aliases))
        if label
    )


def _normalized_label(label: str) -> str:
    tokens = _TOKEN_RE.findall(label.casefold())
    return " ".join(tokens)


def _concept_source_signals(
    sources: Sequence[ConceptGraphEvidenceSource],
) -> list[str]:
    return sorted(
        {
            source.source_signal
            for source in sources
            if source.source_signal not in TIMESTAMP_ONLY_SOURCE_SIGNALS
        }
    )


def _source_type_set(sources: Sequence[ConceptGraphEvidenceSource]) -> list[str]:
    return sorted({source.source_type for source in sources})


def _source_evidence_refs_for_concepts(
    concepts: Sequence[ConceptGraphConceptNode],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for concept in concepts:
        refs.extend(_source_evidence_refs(concept.lecture_id, concept.evidence_sources))
    return _unique_dicts(refs)


def _source_evidence_refs(
    lecture_id: str,
    sources: Sequence[ConceptGraphEvidenceSource],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for source in sources:
        refs.append(
            _compact(
                {
                    "lecture_id": lecture_id,
                    "evidence_unit_id": source.evidence_unit_id,
                    "source_type": source.source_type,
                    "source_signal": source.source_signal,
                    "source_id": source.source_id,
                    "confidence": source.confidence,
                    "is_verified_object_alignment": (
                        True if source.is_verified_object_alignment else None
                    ),
                }
            )
        )
    return _unique_dicts(refs)


def _global_concept_id(
    concepts: Sequence[ConceptGraphConceptNode],
    cluster_keys: Sequence[str],
) -> str:
    canonical = _normalized_label(_canonical_label(concepts)).replace(" ", "_")
    base = slugify(canonical).casefold()
    if not base or base == "empty":
        base = hashlib.sha1("|".join(cluster_keys).encode("utf-8")).hexdigest()[:12]
    if len(concepts) == 1:
        concept = concepts[0]
        suffix = slugify(f"{concept.lecture_id}_{concept.concept_id}").casefold()
        return f"global_{base}_{suffix}"
    return f"global_{base}"


def _unique_global_concept_id(
    global_concept_id: str,
    cluster_keys: Sequence[str],
    used_global_ids: set[str],
) -> str:
    if global_concept_id not in used_global_ids:
        return global_concept_id
    digest = hashlib.sha1("|".join(cluster_keys).encode("utf-8")).hexdigest()[:8]
    return f"{global_concept_id}_{digest}"


def _canonical_label(concepts: Sequence[ConceptGraphConceptNode]) -> str:
    counts: Counter[str] = Counter()
    first_label: dict[str, str] = {}
    confidence_by_label: Counter[str] = Counter()
    for concept in concepts:
        for label in (concept.label, *concept.aliases):
            normalized = _normalized_label(label)
            if not normalized:
                continue
            counts[normalized] += 1
            confidence_by_label[normalized] += concept.confidence
            first_label.setdefault(normalized, label)
    if not counts:
        return concepts[0].label
    best = sorted(
        counts,
        key=lambda key: (-counts[key], -confidence_by_label[key], first_label[key].casefold()),
    )[0]
    return first_label[best]


def _global_aliases(concepts: Sequence[ConceptGraphConceptNode]) -> list[str]:
    canonical_key = _normalized_label(_canonical_label(concepts))
    aliases: list[str] = []
    seen: set[str] = {canonical_key}
    for concept in concepts:
        for label in (concept.label, *concept.aliases):
            key = _normalized_label(label)
            if not key or key in seen:
                continue
            seen.add(key)
            aliases.append(label)
    return aliases


def _best_score_by_local_key(
    scored_pairs: list[tuple[str, str, MergeScore]],
) -> dict[str, MergeScore]:
    best: dict[str, MergeScore] = {}
    for left, right, score in scored_pairs:
        for key in (left, right):
            if key not in best or score.score > best[key].score:
                best[key] = score
    return best


def _average_score_breakdown(scores: list[MergeScore]) -> dict[str, float]:
    if not scores:
        return {}
    return {
        "label_alias": round(sum(score.label_alias_score for score in scores) / len(scores), 3),
        "source_evidence": round(
            sum(score.source_evidence_score for score in scores) / len(scores),
            3,
        ),
        "relation_context": round(
            sum(score.relation_context_score for score in scores) / len(scores),
            3,
        ),
        "concept_type": round(sum(score.concept_type_score for score in scores) / len(scores), 3),
    }


def _relationship_label(relation_type: str) -> str:
    return re.sub(r"[^A-Z0-9_]+", "_", relation_type.upper()).strip("_") or "RELATED_TO"


def _node(*, key: str, labels: tuple[str, ...], properties: dict[str, Any]) -> dict[str, Any]:
    return {"key": key, "labels": list(labels), "properties": properties}


def _relationship(
    *,
    key: str,
    relationship_type: str,
    start_node_key: str,
    end_node_key: str,
    properties: dict[str, Any],
) -> dict[str, Any]:
    return {
        "key": key,
        "type": relationship_type,
        "start_node_key": start_node_key,
        "end_node_key": end_node_key,
        "properties": _compact(properties),
    }


def _local_concept_key(lecture_id: str, concept_id: str) -> str:
    return f"concept:{lecture_id}:{concept_id}"


def _global_concept_key(global_concept_id: str) -> str:
    return f"global_concept:{global_concept_id}"


def _counts(
    *,
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    local_concepts: dict[str, _LocalConcept],
    clusters: list[list[str]],
    decisions: list[MergeDecision],
    conflicts: list[MergeConflict],
    graph_count: int,
) -> dict[str, Any]:
    node_counts: Counter[str] = Counter()
    for node in nodes:
        for label in node["labels"]:
            if label != "GraphNode":
                node_counts[label] += 1
    relationship_counts = Counter(relationship["type"] for relationship in relationships)
    merged_clusters = [cluster for cluster in clusters if len(cluster) > 1]
    return {
        "lecture_graphs": graph_count,
        "local_concepts": len(local_concepts),
        "global_concepts": len(clusters),
        "merged_global_concepts": len(merged_clusters),
        "singleton_global_concepts": len(clusters) - len(merged_clusters),
        "merge_decisions": len(decisions),
        "conflicts": len(conflicts),
        "nodes": len(nodes),
        "relationships": len(relationships),
        "nodes_by_label": dict(sorted(node_counts.items())),
        "relationships_by_type": dict(sorted(relationship_counts.items())),
        "global_relations": sum(
            count
            for relationship_type, count in relationship_counts.items()
            if relationship_type.startswith(GLOBAL_RELATION_TYPE_PREFIX)
        ),
        "mapping_relations": relationship_counts.get(INSTANCE_OF_GLOBAL_CONCEPT_RELATION, 0),
        "source_evidence_refs": _source_evidence_ref_count(nodes, relationships),
    }


def _source_evidence_ref_count(
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> int:
    total = 0
    for item in [*nodes, *relationships]:
        refs = item.get("properties", {}).get("source_evidence_refs")
        if isinstance(refs, list):
            total += len(refs)
    return total


def _counts_only_public_summary(counts: dict[str, Any]) -> dict[str, Any]:
    return {
        "level": "counts_only",
        "counts": _counts_only(counts),
    }


def _counts_only(counts: dict[str, Any]) -> dict[str, int]:
    keys = (
        "lecture_graphs",
        "local_concepts",
        "global_concepts",
        "merged_global_concepts",
        "singleton_global_concepts",
        "merge_decisions",
        "conflicts",
        "global_relations",
        "mapping_relations",
        "source_evidence_refs",
    )
    return {key: int(counts.get(key) or 0) for key in keys}


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value is not None and value != [] and value != {} and value != ()
    }


def _unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})


def _unique_dicts(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(value)
    return unique


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"
