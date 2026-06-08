from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from oarag.core.io import write_json
from oarag.core.schemas import slugify
from oarag.graph.concept_graph_schema import (
    CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH,
    CONCEPT_GRAPH_SCHEMA_VERSION,
    TIMESTAMP_ONLY_SOURCE_SIGNALS,
    ConceptGraphConceptNode,
    ConceptGraphEvidenceSource,
    ConceptGraphRelationEdge,
    write_concept_graph_artifact,
)
from oarag.retrieval.project_index import evidence_unit_artifact_path, iter_jsonl_documents


CONCEPT_CANDIDATE_EXTRACTION_SCHEMA_VERSION = "oarag-concept-candidates-v1"
CONCEPT_RELATION_EXTRACTION_SCHEMA_VERSION = "oarag-concept-relations-v1"
CONCEPT_CANDIDATE_PUBLIC_NOTE = (
    "Concept candidates are lecture-local nodes extracted from evidence unit source "
    "signals. Public summaries report counts and coverage only; raw transcripts, raw "
    "frame paths, and local absolute paths are not copied into the summary."
)

SOURCE_SIGNAL_TRANSCRIPT_STATEMENT = "transcript_statement"
SOURCE_SIGNAL_TRANSCRIPT_MENTION = "transcript_mention"
SOURCE_SIGNAL_SLIDE_TEXT = "slide_text"
SOURCE_SIGNAL_DETECTED_TEXT = "detected_text"
SOURCE_SIGNAL_VLM_VISUAL_DESCRIPTION = "vlm_visual_observation"

DEFAULT_SOURCE_CONFIDENCE = {
    SOURCE_SIGNAL_TRANSCRIPT_STATEMENT: 0.72,
    SOURCE_SIGNAL_TRANSCRIPT_MENTION: 0.64,
    SOURCE_SIGNAL_SLIDE_TEXT: 0.62,
    SOURCE_SIGNAL_DETECTED_TEXT: 0.58,
    SOURCE_SIGNAL_VLM_VISUAL_DESCRIPTION: 0.56,
}

RELATION_TYPE_DEFINES = "defines"
RELATION_TYPE_USES = "uses"
RELATION_TYPE_PREREQUISITE_OF = "prerequisite_of"
RELATION_TYPE_CONTRASTS_WITH = "contrasts_with"
RELATION_TYPE_VISUALLY_DEPICTS = "visually_depicts"
RELATION_TYPE_RELATED_TO = "related_to"

_RELATION_TYPE_ALIASES = {
    "contrast": RELATION_TYPE_CONTRASTS_WITH,
    "contrasts": RELATION_TYPE_CONTRASTS_WITH,
    "depicts": RELATION_TYPE_VISUALLY_DEPICTS,
    "prerequisite": RELATION_TYPE_PREREQUISITE_OF,
    "prerequisite_for": RELATION_TYPE_PREREQUISITE_OF,
    "requires": RELATION_TYPE_PREREQUISITE_OF,
    "visual_depicts": RELATION_TYPE_VISUALLY_DEPICTS,
}

_RELATION_RULE_BASE_CONFIDENCE = {
    RELATION_TYPE_DEFINES: 0.74,
    RELATION_TYPE_USES: 0.7,
    RELATION_TYPE_PREREQUISITE_OF: 0.72,
    RELATION_TYPE_CONTRASTS_WITH: 0.68,
    RELATION_TYPE_VISUALLY_DEPICTS: 0.66,
    RELATION_TYPE_RELATED_TO: 0.5,
}

_TOKEN_RE = re.compile(r"[A-Za-z가-힣][A-Za-z0-9가-힣]*|[0-9]+[A-Za-z가-힣]+|[0-9]+")
_MAX_CANDIDATE_TOKENS = 4

_STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "that",
    "these",
    "those",
    "to",
    "we",
    "with",
}

_ACTION_TERMS = {
    "adjust",
    "adjusts",
    "compare",
    "compared",
    "compares",
    "comparing",
    "contrast",
    "contrasted",
    "contrasts",
    "define",
    "defined",
    "defines",
    "describe",
    "describes",
    "depict",
    "depicted",
    "depicting",
    "depicts",
    "enter",
    "enters",
    "explain",
    "explains",
    "highlight",
    "highlights",
    "illustrate",
    "illustrated",
    "illustrates",
    "indicate",
    "indicates",
    "mean",
    "meaning",
    "means",
    "measure",
    "measures",
    "move",
    "moves",
    "moving",
    "need",
    "needs",
    "prerequisite",
    "require",
    "required",
    "requires",
    "show",
    "showing",
    "shown",
    "shows",
    "transform",
    "transforms",
    "update",
    "updates",
    "use",
    "uses",
    "using",
}

_GENERIC_TERMS = {
    "annotation",
    "candidate",
    "caption",
    "chart",
    "concept",
    "course",
    "data",
    "description",
    "diagram",
    "example",
    "figure",
    "frame",
    "graph",
    "image",
    "lecture",
    "model",
    "object",
    "overview",
    "parameter",
    "parameters",
    "picture",
    "rule",
    "sample",
    "slide",
    "source",
    "summary",
    "term",
    "text",
    "thing",
    "title",
    "topic",
    "video",
    "visual",
    "visually",
    "vs",
    "versus",
}

_BOUNDARY_TERMS = (
    _STOP_TERMS
    | _ACTION_TERMS
    | {
        "arrow",
        "board",
        "box",
        "caption",
        "diagram",
        "example",
        "figure",
        "image",
        "lower",
        "rule",
        "slide",
        "toward",
        "visual",
    }
)

_UPPERCASE_TERMS = {
    "api",
    "cnn",
    "l1",
    "l2",
    "ocr",
    "relu",
    "rnn",
    "vlm",
}


@dataclass(frozen=True)
class ConceptCandidateExtractionResult:
    lecture_id: str
    concepts: tuple[ConceptGraphConceptNode, ...]
    relations: tuple[ConceptGraphRelationEdge, ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _SourceText:
    evidence_unit_id: str
    source_type: str
    source_signal: str
    text: str
    source_id: str | None = None
    confidence: float | None = None
    start_time: float | None = None
    end_time: float | None = None


@dataclass(frozen=True)
class _ConceptMention:
    concept_id: str
    start: int
    end: int


@dataclass(frozen=True)
class _RelationMatch:
    source_concept_id: str
    relation_type: str
    target_concept_id: str
    rule: str
    confidence: float
    relation_status: str = "candidate"


@dataclass
class _ConceptAggregate:
    label: str
    first_seen: int
    occurrence_count: int = 0
    evidence_sources: dict[tuple[str, str, str | None], ConceptGraphEvidenceSource] = field(
        default_factory=dict
    )
    source_unit_ids: list[str] = field(default_factory=list)
    source_signal_counts: Counter[str] = field(default_factory=Counter)
    source_type_counts: Counter[str] = field(default_factory=Counter)

    def add(self, source: _SourceText) -> None:
        self.occurrence_count += 1
        self.source_signal_counts[source.source_signal] += 1
        self.source_type_counts[source.source_type] += 1
        if source.evidence_unit_id not in self.source_unit_ids:
            self.source_unit_ids.append(source.evidence_unit_id)
        key = (source.evidence_unit_id, source.source_signal, source.source_id)
        existing = self.evidence_sources.get(key)
        evidence_source = ConceptGraphEvidenceSource(
            evidence_unit_id=source.evidence_unit_id,
            source_type=source.source_type,
            source_signal=source.source_signal,
            source_id=source.source_id,
            confidence=source.confidence,
            start_time=source.start_time,
            end_time=source.end_time,
            is_verified_object_alignment=False,
            metadata={"raw_text_redacted": True},
        )
        if existing is None:
            self.evidence_sources[key] = evidence_source
            return
        if (evidence_source.confidence or 0.0) > (existing.confidence or 0.0):
            self.evidence_sources[key] = evidence_source


@dataclass
class _RelationAggregate:
    source_concept_id: str
    relation_type: str
    target_concept_id: str
    evidence_sources: dict[tuple[str, str, str | None], ConceptGraphEvidenceSource] = field(
        default_factory=dict
    )
    rule_counts: Counter[str] = field(default_factory=Counter)
    relation_status: str = "candidate"

    def add(
        self,
        *,
        source: _SourceText,
        source_signal: str,
        confidence: float,
        rule: str,
        relation_status: str = "candidate",
    ) -> None:
        normalized_status = _relation_status_for_signals(
            relation_status,
            source_signals=(source_signal,),
        )
        self.rule_counts[rule] += 1
        if self.relation_status != "verified" and normalized_status == "verified":
            self.relation_status = "verified"

        evidence_source = ConceptGraphEvidenceSource(
            evidence_unit_id=source.evidence_unit_id,
            source_type=source.source_type,
            source_signal=source_signal,
            source_id=source.source_id,
            confidence=confidence,
            start_time=source.start_time,
            end_time=source.end_time,
            is_verified_object_alignment=False,
            metadata={
                "relation_rule": rule,
                "raw_text_redacted": True,
            },
        )
        key = (source.evidence_unit_id, source_signal, source.source_id)
        existing = self.evidence_sources.get(key)
        if existing is None or (evidence_source.confidence or 0.0) > (existing.confidence or 0.0):
            self.evidence_sources[key] = evidence_source


def extract_project_concept_candidates(
    *,
    project_dir: Path,
    evidence_units: Path | None = None,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    lecture_id: str | None = None,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    if not resolved_project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {resolved_project_dir}")

    evidence_units_path = evidence_unit_artifact_path(resolved_project_dir, evidence_units)
    resolved_output_path = _resolve_project_output_path(
        project_dir=resolved_project_dir,
        path=output_path,
        default=resolved_project_dir / CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH,
    )
    resolved_manifest_path = _resolve_project_output_path(
        project_dir=resolved_project_dir,
        path=manifest_path,
        default=resolved_project_dir / "manifests" / "project_manifest.json",
    )
    rows = list(iter_jsonl_documents(evidence_units_path))
    manifest = _load_json_object(resolved_manifest_path)
    resolved_lecture_id = _lecture_id(
        explicit=lecture_id,
        manifest=manifest,
        evidence_units=rows,
        project_dir=resolved_project_dir,
    )
    confidence_threshold = _confidence_threshold(min_confidence)
    result = extract_concept_candidates(
        rows,
        lecture_id=resolved_lecture_id,
        min_confidence=confidence_threshold,
    )
    records = [*result.concepts, *result.relations]
    write_concept_graph_artifact(resolved_output_path, records)
    summary = _summary(
        project_dir=resolved_project_dir,
        evidence_units_path=evidence_units_path,
        output_path=resolved_output_path,
        manifest_path=resolved_manifest_path,
        result=result,
    )
    _update_project_manifest(
        manifest_path=resolved_manifest_path,
        project_dir=resolved_project_dir,
        concept_graph_path=resolved_output_path,
        summary=summary,
    )
    return summary


def extract_concept_candidates(
    evidence_units: Iterable[dict[str, Any]],
    *,
    lecture_id: str,
    min_confidence: float = 0.0,
) -> ConceptCandidateExtractionResult:
    aggregates: dict[str, _ConceptAggregate] = {}
    source_texts: list[_SourceText] = []
    evidence_unit_rows: list[dict[str, Any]] = []
    source_texts_total = 0
    extracted_mentions_total = 0
    pruned_mentions_total = 0
    evidence_units_total = 0
    evidence_units_with_candidates: set[str] = set()
    source_text_counts: Counter[str] = Counter()

    for evidence_unit in evidence_units:
        evidence_unit_rows.append(dict(evidence_unit))
        evidence_units_total += 1
        unit_candidate_keys: set[str] = set()
        for source_text in _source_texts_for_unit(evidence_unit):
            source_texts.append(source_text)
            source_texts_total += 1
            source_text_counts[source_text.source_signal] += 1
            labels, pruned_count = _candidate_labels_from_text(source_text.text)
            pruned_mentions_total += pruned_count
            for label in labels:
                key = _candidate_key(label)
                aggregate = aggregates.get(key)
                if aggregate is None:
                    aggregate = _ConceptAggregate(
                        label=label,
                        first_seen=len(aggregates) + 1,
                    )
                    aggregates[key] = aggregate
                aggregate.add(source_text)
                unit_candidate_keys.add(key)
                extracted_mentions_total += 1
        if unit_candidate_keys:
            evidence_unit_id = _text(evidence_unit.get("evidence_unit_id"))
            if evidence_unit_id:
                evidence_units_with_candidates.add(evidence_unit_id)

    canonical_aggregates, canonicalization = _canonicalize_aggregates(aggregates.values())
    concepts = [
        _concept_node(
            aggregate,
            lecture_id=lecture_id,
            aliases=tuple(canonicalization["aliases_by_label"].get(aggregate.label, ())),
            alias_rules=tuple(canonicalization["rules_by_label"].get(aggregate.label, ())),
        )
        for aggregate in canonical_aggregates
    ]
    concepts = [
        concept
        for concept in concepts
        if concept.confidence >= _confidence_threshold(min_confidence)
    ]
    concepts.sort(
        key=lambda concept: (
            -concept.confidence,
            int(concept.metadata.get("first_seen") or 0),
            concept.label.casefold(),
        )
    )
    concepts = [_with_rank(concept, rank=index + 1) for index, concept in enumerate(concepts)]
    relations, relation_diagnostics = _extract_relation_edges(
        concepts=concepts,
        source_texts=source_texts,
        evidence_units=evidence_unit_rows,
        lecture_id=lecture_id,
    )
    source_signal_counts = Counter()
    for concept in concepts:
        source_signal_counts.update(concept.metadata.get("source_signal_counts", {}))
    diagnostics = {
        "schema_version": CONCEPT_CANDIDATE_EXTRACTION_SCHEMA_VERSION,
        "relation_extraction_schema_version": CONCEPT_RELATION_EXTRACTION_SCHEMA_VERSION,
        "evidence_units_total": evidence_units_total,
        "evidence_units_with_candidates": len(evidence_units_with_candidates),
        "source_texts_total": source_texts_total,
        "source_text_counts": dict(sorted(source_text_counts.items())),
        "extracted_mentions_total": extracted_mentions_total,
        "pruned_mentions_total": pruned_mentions_total,
        "raw_concept_candidates_total": len(aggregates),
        "concept_candidates_total": len(concepts),
        "source_evidence_total": sum(len(concept.evidence_sources) for concept in concepts),
        "source_signal_counts": dict(sorted(source_signal_counts.items())),
        "canonicalization": canonicalization["diagnostics"],
        "relations": relation_diagnostics,
        "raw_text_redacted": True,
    }
    return ConceptCandidateExtractionResult(
        lecture_id=lecture_id,
        concepts=tuple(concepts),
        relations=tuple(relations),
        diagnostics=diagnostics,
    )


def _concept_node(
    aggregate: _ConceptAggregate,
    *,
    lecture_id: str,
    aliases: tuple[str, ...] = (),
    alias_rules: tuple[str, ...] = (),
) -> ConceptGraphConceptNode:
    source_signals = dict(sorted(aggregate.source_signal_counts.items()))
    confidence = _aggregate_confidence(aggregate)
    return ConceptGraphConceptNode(
        concept_id=_concept_id(aggregate.label),
        lecture_id=lecture_id,
        label=aggregate.label,
        aliases=aliases,
        concept_type="canonical_concept",
        source_evidence_unit_ids=tuple(aggregate.source_unit_ids),
        evidence_sources=tuple(aggregate.evidence_sources.values()),
        confidence=confidence,
        metadata={
            "candidate_extractor_schema_version": CONCEPT_CANDIDATE_EXTRACTION_SCHEMA_VERSION,
            "canonicalization_schema_version": "oarag-concept-canonicalization-v1",
            "first_seen": aggregate.first_seen,
            "occurrence_count": aggregate.occurrence_count,
            "source_unit_count": len(aggregate.source_unit_ids),
            "source_signal_counts": source_signals,
            "source_type_counts": dict(sorted(aggregate.source_type_counts.items())),
            "generic_pruning": "minimal_v1",
            "alias_count": len(aliases),
            "alias_rules": list(alias_rules),
            "canonicalization_method": "deterministic_lexical_alias_rules_v1",
            "raw_text_redacted": True,
        },
    )


def _with_rank(
    concept: ConceptGraphConceptNode,
    *,
    rank: int,
) -> ConceptGraphConceptNode:
    metadata = dict(concept.metadata)
    metadata["candidate_rank"] = rank
    return ConceptGraphConceptNode(
        concept_id=concept.concept_id,
        lecture_id=concept.lecture_id,
        label=concept.label,
        concept_type=concept.concept_type,
        source_evidence_unit_ids=concept.source_evidence_unit_ids,
        evidence_sources=concept.evidence_sources,
        confidence=concept.confidence,
        aliases=concept.aliases,
        description=concept.description,
        metadata=metadata,
    )


def _canonicalize_aggregates(
    aggregates: Iterable[_ConceptAggregate],
) -> tuple[list[_ConceptAggregate], dict[str, Any]]:
    items = list(aggregates)
    if not items:
        return [], {
            "aliases_by_label": {},
            "rules_by_label": {},
            "diagnostics": {
                "method": "deterministic_lexical_alias_rules_v1",
                "input_candidates_total": 0,
                "canonical_concepts_total": 0,
                "alias_labels_total": 0,
                "alias_groups_total": 0,
                "alias_rule_counts": {},
            },
        }

    key_by_label = {_candidate_key(item.label): item for item in items}
    parent = {key: key for key in key_by_label}
    rule_counts: Counter[str] = Counter()

    def find(key: str) -> str:
        root = parent[key]
        if root != key:
            parent[key] = find(root)
        return parent[key]

    def union(left: str, right: str, rule: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        parent[right_root] = left_root
        rule_counts[rule] += 1

    compact_index: dict[str, list[str]] = {}
    acronym_full_index: dict[str, list[str]] = {}
    acronym_labels: dict[str, list[str]] = {}
    for key, aggregate in key_by_label.items():
        tokens = _tokens(aggregate.label)
        compact = _compact_alias_key(aggregate.label)
        if compact:
            compact_index.setdefault(compact, []).append(key)
        acronym = _acronym_for_label(aggregate.label)
        if acronym:
            acronym_full_index.setdefault(acronym, []).append(key)
        if _is_acronym_label(aggregate.label):
            acronym_labels.setdefault(_compact_alias_key(aggregate.label), []).append(key)

        normalized_key = " ".join(tokens)
        if normalized_key and normalized_key != key and normalized_key in key_by_label:
            union(key, normalized_key, "normalized_string")

    for keys in compact_index.values():
        if len(keys) < 2:
            continue
        anchor = keys[0]
        for key in keys[1:]:
            union(anchor, key, "compact_spacing_hyphen")

    for acronym, short_keys in acronym_labels.items():
        for short_key in short_keys:
            for full_key in acronym_full_index.get(acronym, []):
                if short_key == full_key:
                    continue
                if _share_evidence_unit(key_by_label[short_key], key_by_label[full_key]):
                    union(full_key, short_key, "acronym_cooccurrence")

    groups: dict[str, list[_ConceptAggregate]] = {}
    for key, aggregate in key_by_label.items():
        groups.setdefault(find(key), []).append(aggregate)

    merged: list[_ConceptAggregate] = []
    aliases_by_label: dict[str, tuple[str, ...]] = {}
    rules_by_label: dict[str, tuple[str, ...]] = {}
    for group in groups.values():
        group.sort(key=lambda item: item.first_seen)
        canonical = _merge_canonical_group(group)
        aliases = _aliases_for_group(group, canonical_label=canonical.label)
        rules = _alias_rules_for_group(group)
        merged.append(canonical)
        aliases_by_label[canonical.label] = aliases
        rules_by_label[canonical.label] = rules

    merged.sort(key=lambda item: item.first_seen)
    alias_groups_total = sum(1 for aliases in aliases_by_label.values() if aliases)
    alias_labels_total = sum(len(aliases) for aliases in aliases_by_label.values())
    return merged, {
        "aliases_by_label": aliases_by_label,
        "rules_by_label": rules_by_label,
        "diagnostics": {
            "method": "deterministic_lexical_alias_rules_v1",
            "input_candidates_total": len(items),
            "canonical_concepts_total": len(merged),
            "alias_labels_total": alias_labels_total,
            "alias_groups_total": alias_groups_total,
            "alias_rule_counts": dict(sorted(rule_counts.items())),
            "embedding_or_llm_alias_hook_used": False,
        },
    }


def _merge_canonical_group(group: list[_ConceptAggregate]) -> _ConceptAggregate:
    canonical_source = max(group, key=_canonical_label_score)
    merged = _ConceptAggregate(
        label=canonical_source.label,
        first_seen=min(item.first_seen for item in group),
    )
    for aggregate in group:
        merged.occurrence_count += aggregate.occurrence_count
        merged.source_signal_counts.update(aggregate.source_signal_counts)
        merged.source_type_counts.update(aggregate.source_type_counts)
        for evidence_unit_id in aggregate.source_unit_ids:
            if evidence_unit_id not in merged.source_unit_ids:
                merged.source_unit_ids.append(evidence_unit_id)
        for key, source in aggregate.evidence_sources.items():
            existing = merged.evidence_sources.get(key)
            if existing is None or (source.confidence or 0.0) > (existing.confidence or 0.0):
                merged.evidence_sources[key] = source
    return merged


def _canonical_label_score(aggregate: _ConceptAggregate) -> tuple[int, int, int, int, int]:
    tokens = _tokens(aggregate.label)
    acronym_penalty = 0 if _is_acronym_label(aggregate.label) else 1
    return (
        aggregate.occurrence_count,
        len(aggregate.source_unit_ids),
        acronym_penalty,
        len(tokens),
        -aggregate.first_seen,
    )


def _aliases_for_group(
    group: list[_ConceptAggregate],
    *,
    canonical_label: str,
) -> tuple[str, ...]:
    aliases: list[str] = []
    canonical_key = _candidate_key(canonical_label)
    for aggregate in sorted(group, key=lambda item: (item.first_seen, item.label.casefold())):
        if _candidate_key(aggregate.label) == canonical_key:
            continue
        if aggregate.label not in aliases:
            aliases.append(aggregate.label)
    return tuple(aliases)


def _alias_rules_for_group(group: list[_ConceptAggregate]) -> tuple[str, ...]:
    labels = [item.label for item in group]
    rules: set[str] = set()
    compact_counts = Counter(_compact_alias_key(label) for label in labels)
    if any(key and count > 1 for key, count in compact_counts.items()):
        rules.add("compact_spacing_hyphen")
    acronyms = {_compact_alias_key(label) for label in labels if _is_acronym_label(label)}
    full_acronyms = {_acronym_for_label(label) for label in labels}
    if any(acronym and acronym in full_acronyms for acronym in acronyms):
        rules.add("acronym_cooccurrence")
    return tuple(sorted(rules))


def _compact_alias_key(label: str) -> str:
    return "".join(_tokens(label)).casefold()


def _acronym_for_label(label: str) -> str:
    tokens = [token for token in _tokens(label) if _has_letter(token)]
    if len(tokens) < 2:
        return ""
    acronym = "".join(token[0] for token in tokens if token)
    return acronym.casefold() if 2 <= len(acronym) <= 8 else ""


def _is_acronym_label(label: str) -> bool:
    tokens = _tokens(label)
    if len(tokens) != 1:
        return False
    compact = "".join(character for character in label if character.isalnum())
    return (
        2 <= len(compact) <= 8
        and any(character.isalpha() for character in compact)
        and compact.upper() == compact
    )


def _share_evidence_unit(left: _ConceptAggregate, right: _ConceptAggregate) -> bool:
    return bool(set(left.source_unit_ids) & set(right.source_unit_ids))


def _source_texts_for_unit(unit: dict[str, Any]) -> list[_SourceText]:
    evidence_unit_id = _text(unit.get("evidence_unit_id"))
    if not evidence_unit_id:
        return []
    start_time = _optional_float(unit.get("start_time"))
    end_time = _optional_float(unit.get("end_time"))
    sources: list[_SourceText] = []

    transcript = _text(unit.get("transcript_window_text"))
    if transcript:
        sources.append(
            _SourceText(
                evidence_unit_id=evidence_unit_id,
                source_type="transcript",
                source_signal=SOURCE_SIGNAL_TRANSCRIPT_MENTION,
                text=transcript,
                source_id=_transcript_source_id(unit),
                confidence=DEFAULT_SOURCE_CONFIDENCE[SOURCE_SIGNAL_TRANSCRIPT_MENTION],
                start_time=start_time,
                end_time=end_time,
            )
        )

    for state in _dict_items(unit.get("visual_states")):
        state_id = _text(state.get("visual_state_id")) or None
        for text in _text_values(
            [
                state.get("slide_text"),
                state.get("slide_texts"),
                state.get("title"),
                state.get("heading"),
                state.get("state_summary"),
            ]
        ):
            sources.append(
                _SourceText(
                    evidence_unit_id=evidence_unit_id,
                    source_type="slide_text",
                    source_signal=SOURCE_SIGNAL_SLIDE_TEXT,
                    text=text,
                    source_id=state_id,
                    confidence=DEFAULT_SOURCE_CONFIDENCE[SOURCE_SIGNAL_SLIDE_TEXT],
                    start_time=start_time,
                    end_time=end_time,
                )
            )
        for text in _text_values([state.get("detected_text")]):
            sources.append(
                _SourceText(
                    evidence_unit_id=evidence_unit_id,
                    source_type="detected_text",
                    source_signal=SOURCE_SIGNAL_DETECTED_TEXT,
                    text=text,
                    source_id=state_id,
                    confidence=DEFAULT_SOURCE_CONFIDENCE[SOURCE_SIGNAL_DETECTED_TEXT],
                    start_time=start_time,
                    end_time=end_time,
                )
            )

    for entity in _dict_items(unit.get("visual_entities")):
        entity_id = _text(entity.get("entity_id")) or None
        if _is_ocr_like_entity(entity):
            for text in _text_values([entity.get("text")]):
                sources.append(
                    _SourceText(
                        evidence_unit_id=evidence_unit_id,
                        source_type="detected_text",
                        source_signal=SOURCE_SIGNAL_DETECTED_TEXT,
                        text=text,
                        source_id=entity_id,
                        confidence=_entity_confidence(entity, SOURCE_SIGNAL_DETECTED_TEXT),
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
        for text in _text_values([entity.get("detected_text")]):
            sources.append(
                _SourceText(
                    evidence_unit_id=evidence_unit_id,
                    source_type="detected_text",
                    source_signal=SOURCE_SIGNAL_DETECTED_TEXT,
                    text=text,
                    source_id=entity_id,
                    confidence=_entity_confidence(entity, SOURCE_SIGNAL_DETECTED_TEXT),
                    start_time=start_time,
                    end_time=end_time,
                )
            )
        for text in _text_values([entity.get("visual_description"), entity.get("description")]):
            sources.append(
                _SourceText(
                    evidence_unit_id=evidence_unit_id,
                    source_type="vlm_visual_description",
                    source_signal=SOURCE_SIGNAL_VLM_VISUAL_DESCRIPTION,
                    text=text,
                    source_id=entity_id,
                    confidence=_entity_confidence(entity, SOURCE_SIGNAL_VLM_VISUAL_DESCRIPTION),
                    start_time=start_time,
                    end_time=end_time,
                )
            )
    return sources


def _extract_relation_edges(
    *,
    concepts: list[ConceptGraphConceptNode],
    source_texts: list[_SourceText],
    evidence_units: list[dict[str, Any]],
    lecture_id: str,
) -> tuple[list[ConceptGraphRelationEdge], dict[str, Any]]:
    if len(concepts) < 2:
        return [], _relation_diagnostics([])

    alias_index = _concept_alias_index(concepts)
    aggregates: dict[tuple[str, str, str], _RelationAggregate] = {}
    pruned_explicit_relations = 0
    timestamp_demotions = 0

    for source_text in source_texts:
        for tokens in _token_groups(source_text.text):
            mentions = _concept_mentions(tokens, alias_index)
            if len({mention.concept_id for mention in mentions}) < 2:
                continue
            for relation_match in _relation_matches_for_source(source_text, tokens, mentions):
                _add_relation_match(aggregates, source_text, relation_match)

    for evidence_unit in evidence_units:
        explicit_edges, pruned_count, demoted_count = _explicit_relation_matches(
            evidence_unit=evidence_unit,
            concepts=concepts,
        )
        pruned_explicit_relations += pruned_count
        timestamp_demotions += demoted_count
        for source_text, relation_match in explicit_edges:
            _add_relation_match(aggregates, source_text, relation_match)

    relations = [
        _relation_edge(aggregate, lecture_id=lecture_id)
        for aggregate in aggregates.values()
        if aggregate.evidence_sources
    ]
    relations.sort(
        key=lambda relation: (
            relation.source_concept_id,
            relation.relation_type,
            relation.target_concept_id,
        )
    )
    diagnostics = _relation_diagnostics(
        relations,
        pruned_explicit_relations=pruned_explicit_relations,
        timestamp_demotions=timestamp_demotions,
    )
    return relations, diagnostics


def _concept_alias_index(
    concepts: list[ConceptGraphConceptNode],
) -> list[tuple[tuple[str, ...], str]]:
    rows: list[tuple[tuple[str, ...], str]] = []
    seen: set[tuple[tuple[str, ...], str]] = set()
    for concept in concepts:
        for label in (concept.label, *concept.aliases):
            tokens = tuple(_tokens(label))
            if not tokens:
                continue
            key = (tokens, concept.concept_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
    rows.sort(key=lambda row: (-len(row[0]), row[0], row[1]))
    return rows


def _concept_mentions(
    tokens: list[str],
    alias_index: list[tuple[tuple[str, ...], str]],
) -> list[_ConceptMention]:
    mentions: list[_ConceptMention] = []
    occupied: set[int] = set()
    for start in range(len(tokens)):
        if start in occupied:
            continue
        for alias_tokens, concept_id in alias_index:
            end = start + len(alias_tokens)
            if end > len(tokens):
                continue
            if any(index in occupied for index in range(start, end)):
                continue
            if tuple(tokens[start:end]) != alias_tokens:
                continue
            mentions.append(_ConceptMention(concept_id=concept_id, start=start, end=end))
            occupied.update(range(start, end))
            break
    mentions.sort(key=lambda mention: (mention.start, mention.end, mention.concept_id))
    return mentions


def _relation_matches_for_source(
    source_text: _SourceText,
    tokens: list[str],
    mentions: list[_ConceptMention],
) -> list[_RelationMatch]:
    matches: list[_RelationMatch] = []
    for index, left in enumerate(mentions):
        for right in mentions[index + 1 :]:
            if left.concept_id == right.concept_id:
                continue
            if right.start < left.end:
                continue
            between = tokens[left.end : right.start]
            if len(between) > 12:
                continue
            relation_match = _relation_match_from_context(
                source_text=source_text,
                tokens=tokens,
                left=left,
                right=right,
                between=between,
            )
            if relation_match is not None:
                matches.append(relation_match)
    return matches


def _relation_match_from_context(
    *,
    source_text: _SourceText,
    tokens: list[str],
    left: _ConceptMention,
    right: _ConceptMention,
    between: list[str],
) -> _RelationMatch | None:
    source_signal = _relation_source_signal(source_text.source_signal)

    if (
        _has_sequence(between, ("contrast", "with"))
        or _has_sequence(
            between,
            ("contrasts", "with"),
        )
        or _has_sequence(between, ("compared", "to"))
        or _has_sequence(
            between,
            ("compared", "with"),
        )
        or _has_any_token(between, {"versus", "vs"})
    ):
        return _RelationMatch(
            source_concept_id=left.concept_id,
            relation_type=RELATION_TYPE_CONTRASTS_WITH,
            target_concept_id=right.concept_id,
            rule="contrast_marker_between_mentions",
            confidence=_relation_confidence(
                source=source_text,
                relation_type=RELATION_TYPE_CONTRASTS_WITH,
                source_signal=source_signal,
            ),
        )

    if (
        _has_sequence(between, ("is", "prerequisite", "for"))
        or _has_sequence(
            between,
            ("is", "required", "before"),
        )
        or _has_any_token(between, {"before"})
    ):
        return _RelationMatch(
            source_concept_id=left.concept_id,
            relation_type=RELATION_TYPE_PREREQUISITE_OF,
            target_concept_id=right.concept_id,
            rule="prerequisite_marker_between_mentions",
            confidence=_relation_confidence(
                source=source_text,
                relation_type=RELATION_TYPE_PREREQUISITE_OF,
                source_signal=source_signal,
            ),
        )

    if _has_any_token(between, {"requires", "require", "needs", "need"}):
        return _RelationMatch(
            source_concept_id=right.concept_id,
            relation_type=RELATION_TYPE_PREREQUISITE_OF,
            target_concept_id=left.concept_id,
            rule="requires_marker_reversed",
            confidence=_relation_confidence(
                source=source_text,
                relation_type=RELATION_TYPE_PREREQUISITE_OF,
                source_signal=source_signal,
            ),
        )

    if (
        _has_sequence(between, ("is", "defined", "as"))
        or _has_any_token(
            between,
            {"means", "defines"},
        )
        or (between == ["is"] and source_text.source_signal == SOURCE_SIGNAL_TRANSCRIPT_MENTION)
    ):
        return _RelationMatch(
            source_concept_id=left.concept_id,
            relation_type=RELATION_TYPE_DEFINES,
            target_concept_id=right.concept_id,
            rule="definition_marker_between_mentions",
            confidence=_relation_confidence(
                source=source_text,
                relation_type=RELATION_TYPE_DEFINES,
                source_signal=source_signal,
            ),
        )

    if _has_any_token(
        between,
        {
            "uses",
            "use",
            "using",
            "used",
            "applies",
            "apply",
            "combines",
            "combine",
        },
    ) or (
        "with" in between
        and _has_any_token(
            between,
            {"updates", "update", "adjusts", "adjust", "measures", "measure"},
        )
    ):
        return _RelationMatch(
            source_concept_id=left.concept_id,
            relation_type=RELATION_TYPE_USES,
            target_concept_id=right.concept_id,
            rule="uses_marker_between_mentions",
            confidence=_relation_confidence(
                source=source_text,
                relation_type=RELATION_TYPE_USES,
                source_signal=source_signal,
            ),
        )

    if source_text.source_signal == SOURCE_SIGNAL_VLM_VISUAL_DESCRIPTION and _has_any_token(
        tokens,
        {
            "depicts",
            "depicting",
            "shows",
            "showing",
            "illustrates",
            "illustrated",
            "visualizes",
            "visualized",
        },
    ):
        return _RelationMatch(
            source_concept_id=left.concept_id,
            relation_type=RELATION_TYPE_VISUALLY_DEPICTS,
            target_concept_id=right.concept_id,
            rule="vlm_visual_depiction_marker",
            confidence=_relation_confidence(
                source=source_text,
                relation_type=RELATION_TYPE_VISUALLY_DEPICTS,
                source_signal=source_signal,
            ),
        )

    return None


def _explicit_relation_matches(
    *,
    evidence_unit: dict[str, Any],
    concepts: list[ConceptGraphConceptNode],
) -> tuple[list[tuple[_SourceText, _RelationMatch]], int, int]:
    evidence_unit_id = _text(evidence_unit.get("evidence_unit_id"))
    if not evidence_unit_id:
        return [], 0, 0
    concept_by_id = {concept.concept_id: concept for concept in concepts}
    concept_by_label_key = {
        _candidate_key(label): concept.concept_id
        for concept in concepts
        for label in (concept.label, *concept.aliases)
        if _candidate_key(label)
    }
    rows = [
        row
        for row in _dict_items(evidence_unit.get("concept_relations"))
        + _dict_items(evidence_unit.get("concept_relation_candidates"))
        if isinstance(row, dict)
    ]
    matches: list[tuple[_SourceText, _RelationMatch]] = []
    pruned = 0
    demoted = 0
    for row in rows:
        source_concept_id = _relation_endpoint_id(
            row, "source", concept_by_id, concept_by_label_key
        )
        target_concept_id = _relation_endpoint_id(
            row, "target", concept_by_id, concept_by_label_key
        )
        relation_type = _normalize_relation_type(
            _text(row.get("relation_type") or row.get("type") or row.get("relation"))
        )
        if (
            not source_concept_id
            or not target_concept_id
            or source_concept_id == target_concept_id
            or not relation_type
        ):
            pruned += 1
            continue
        source_signals = _explicit_source_signals(row)
        if not source_signals:
            pruned += 1
            continue
        relation_status = (
            _text(row.get("relation_status") or row.get("status") or row.get("evidence_status"))
            or "candidate"
        )
        normalized_status = _relation_status_for_signals(
            relation_status,
            source_signals=tuple(source_signals),
        )
        if normalized_status != relation_status and _timestamp_only_signals(tuple(source_signals)):
            demoted += 1
        for source_signal in source_signals:
            source = _SourceText(
                evidence_unit_id=evidence_unit_id,
                source_type=_text(row.get("source_type")) or "explicit_relation_hint",
                source_signal=source_signal,
                text="",
                source_id=_text(row.get("source_id") or row.get("edge_id")) or None,
                confidence=_optional_float(row.get("confidence")),
                start_time=_optional_float(
                    row.get("start_time") or evidence_unit.get("start_time")
                ),
                end_time=_optional_float(row.get("end_time") or evidence_unit.get("end_time")),
            )
            confidence = _explicit_relation_confidence(row, source_signal=source_signal)
            matches.append(
                (
                    source,
                    _RelationMatch(
                        source_concept_id=source_concept_id,
                        relation_type=relation_type,
                        target_concept_id=target_concept_id,
                        rule="explicit_relation_hint",
                        confidence=confidence,
                        relation_status=normalized_status,
                    ),
                ),
            )
    return matches, pruned, demoted


def _relation_endpoint_id(
    row: dict[str, Any],
    role: str,
    concept_by_id: dict[str, ConceptGraphConceptNode],
    concept_by_label_key: dict[str, str],
) -> str | None:
    raw = (
        row.get(f"{role}_concept_id")
        or row.get(f"{role}_id")
        or row.get(f"{role}_label")
        or row.get(role)
    )
    text = _text(raw)
    if not text:
        return None
    if text in concept_by_id:
        return text
    return concept_by_label_key.get(_candidate_key(text))


def _explicit_source_signals(row: dict[str, Any]) -> list[str]:
    raw = row.get("source_signals")
    if raw is None:
        raw = row.get("source_signal")
    if raw is None:
        raw = row.get("evidence")
    return [_snake_case(value) for value in _string_list(raw)]


def _normalize_relation_type(value: str) -> str | None:
    if not value:
        return None
    relation_type = _snake_case(value)
    relation_type = _RELATION_TYPE_ALIASES.get(relation_type, relation_type)
    if relation_type == "prerequisite":
        return RELATION_TYPE_PREREQUISITE_OF
    supported = {
        RELATION_TYPE_DEFINES,
        RELATION_TYPE_USES,
        RELATION_TYPE_PREREQUISITE_OF,
        RELATION_TYPE_CONTRASTS_WITH,
        RELATION_TYPE_VISUALLY_DEPICTS,
        RELATION_TYPE_RELATED_TO,
        "mentions",
        "explains",
        "shows",
        "alias_of",
    }
    return relation_type if relation_type in supported else None


def _add_relation_match(
    aggregates: dict[tuple[str, str, str], _RelationAggregate],
    source_text: _SourceText,
    relation_match: _RelationMatch,
) -> None:
    if relation_match.source_concept_id == relation_match.target_concept_id:
        return
    source_signal = (
        source_text.source_signal
        if source_text.text == ""
        else _relation_source_signal(source_text.source_signal)
    )
    key = (
        relation_match.source_concept_id,
        relation_match.relation_type,
        relation_match.target_concept_id,
    )
    aggregate = aggregates.get(key)
    if aggregate is None:
        aggregate = _RelationAggregate(
            source_concept_id=relation_match.source_concept_id,
            relation_type=relation_match.relation_type,
            target_concept_id=relation_match.target_concept_id,
        )
        aggregates[key] = aggregate
    aggregate.add(
        source=source_text,
        source_signal=source_signal,
        confidence=relation_match.confidence,
        rule=relation_match.rule,
        relation_status=relation_match.relation_status,
    )


def _relation_edge(
    aggregate: _RelationAggregate,
    *,
    lecture_id: str,
) -> ConceptGraphRelationEdge:
    evidence_sources = tuple(aggregate.evidence_sources.values())
    evidence_unit_ids = _unique_tuple(source.evidence_unit_id for source in evidence_sources)
    source_signals = _unique_tuple(source.source_signal for source in evidence_sources)
    relation_status = _relation_status_for_signals(
        aggregate.relation_status,
        source_signals=source_signals,
    )
    confidence = _relation_aggregate_confidence(evidence_sources, source_signals)
    return ConceptGraphRelationEdge(
        edge_id=_relation_edge_id(
            aggregate.source_concept_id,
            aggregate.relation_type,
            aggregate.target_concept_id,
        ),
        lecture_id=lecture_id,
        source_concept_id=aggregate.source_concept_id,
        relation_type=aggregate.relation_type,
        target_concept_id=aggregate.target_concept_id,
        evidence_unit_ids=evidence_unit_ids,
        source_signals=source_signals,
        evidence_sources=evidence_sources,
        confidence=confidence,
        relation_status=relation_status,
        description=None,
        metadata={
            "relation_extractor_schema_version": CONCEPT_RELATION_EXTRACTION_SCHEMA_VERSION,
            "rule_counts": dict(sorted(aggregate.rule_counts.items())),
            "timestamp_only_candidate": _timestamp_only_signals(source_signals),
            "embedding_or_llm_relation_hook_used": False,
            "raw_text_redacted": True,
        },
    )


def _relation_edge_id(source_concept_id: str, relation_type: str, target_concept_id: str) -> str:
    raw = f"{source_concept_id}_{relation_type}_{target_concept_id}"
    base = slugify(raw).casefold()
    if len(base) <= 120:
        return f"edge_{base}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"edge_{base[:96]}_{digest}"


def _relation_source_signal(source_signal: str) -> str:
    if source_signal == SOURCE_SIGNAL_TRANSCRIPT_MENTION:
        return SOURCE_SIGNAL_TRANSCRIPT_STATEMENT
    return source_signal


def _relation_confidence(
    *,
    source: _SourceText,
    relation_type: str,
    source_signal: str,
) -> float:
    base = _RELATION_RULE_BASE_CONFIDENCE.get(relation_type, 0.6)
    signal_confidence = source.confidence
    if signal_confidence is None:
        signal_confidence = DEFAULT_SOURCE_CONFIDENCE.get(source_signal, 0.55)
    return round(min(0.9, (base * 0.65) + (float(signal_confidence) * 0.35)), 3)


def _explicit_relation_confidence(row: dict[str, Any], *, source_signal: str) -> float:
    parsed = _optional_float(row.get("confidence"))
    if parsed is None:
        parsed = DEFAULT_SOURCE_CONFIDENCE.get(source_signal, 0.5)
    parsed = max(0.0, min(1.0, parsed))
    if source_signal in TIMESTAMP_ONLY_SOURCE_SIGNALS:
        parsed = min(parsed, 0.35)
    return round(parsed, 3)


def _relation_aggregate_confidence(
    evidence_sources: tuple[ConceptGraphEvidenceSource, ...],
    source_signals: tuple[str, ...],
) -> float:
    if not evidence_sources:
        return 0.0
    base = max(source.confidence or 0.0 for source in evidence_sources)
    evidence_bonus = min(0.08, 0.02 * max(len(evidence_sources) - 1, 0))
    confidence = min(0.95, base + evidence_bonus)
    if _timestamp_only_signals(source_signals):
        confidence = min(confidence, 0.35)
    return round(confidence, 3)


def _relation_status_for_signals(
    relation_status: str,
    *,
    source_signals: tuple[str, ...],
) -> str:
    normalized = _snake_case(relation_status or "candidate")
    if normalized not in {"candidate", "verified"}:
        normalized = "candidate"
    if _timestamp_only_signals(source_signals):
        return "candidate"
    return normalized


def _timestamp_only_signals(source_signals: tuple[str, ...]) -> bool:
    return bool(source_signals) and set(source_signals).issubset(TIMESTAMP_ONLY_SOURCE_SIGNALS)


def _relation_diagnostics(
    relations: list[ConceptGraphRelationEdge],
    *,
    pruned_explicit_relations: int = 0,
    timestamp_demotions: int = 0,
) -> dict[str, Any]:
    relation_type_counts = Counter(relation.relation_type for relation in relations)
    source_signal_counts = Counter(
        signal for relation in relations for signal in relation.source_signals
    )
    candidate_relations = sum(
        1 for relation in relations if relation.relation_status == "candidate"
    )
    verified_relations = sum(1 for relation in relations if relation.relation_status == "verified")
    timestamp_only_relations = sum(
        1 for relation in relations if _timestamp_only_signals(relation.source_signals)
    )
    return {
        "relation_edges_total": len(relations),
        "candidate_relation_edges_total": candidate_relations,
        "verified_relation_edges_total": verified_relations,
        "timestamp_only_candidate_relation_edges_total": timestamp_only_relations,
        "timestamp_only_verified_demotions_total": timestamp_demotions,
        "pruned_explicit_relations_total": pruned_explicit_relations,
        "relation_source_evidence_total": sum(
            len(relation.evidence_sources) for relation in relations
        ),
        "relation_type_counts": dict(sorted(relation_type_counts.items())),
        "source_signal_counts": dict(sorted(source_signal_counts.items())),
        "timestamp_fallback_counted_as_verified": False,
        "raw_text_redacted": True,
    }


def _has_sequence(tokens: list[str], sequence: tuple[str, ...]) -> bool:
    if not sequence or len(sequence) > len(tokens):
        return False
    width = len(sequence)
    return any(
        tuple(tokens[index : index + width]) == sequence for index in range(len(tokens) - width + 1)
    )


def _has_any_token(tokens: list[str], values: set[str]) -> bool:
    return bool(set(tokens) & values)


def _candidate_labels_from_text(text: str) -> tuple[list[str], int]:
    chunks = _content_chunks(text)
    labels: list[str] = []
    pruned = 0
    for chunk in chunks:
        spans = _candidate_spans(chunk)
        if not spans:
            pruned += 1
            continue
        for span in spans:
            if not _valid_candidate_tokens(span):
                pruned += 1
                continue
            labels.append(_label_from_tokens(span))
    return _unique_labels(labels), pruned


def _content_chunks(text: str) -> list[list[str]]:
    chunks: list[list[str]] = []
    for tokens in _token_groups(text):
        current: list[str] = []
        for token in tokens:
            if token in _BOUNDARY_TERMS:
                if current:
                    chunks.append(current)
                    current = []
                continue
            current.append(token)
        if current:
            chunks.append(current)
    return chunks


def _candidate_spans(tokens: list[str]) -> list[list[str]]:
    trimmed = _trim_generic_edges(tokens)
    if not trimmed:
        return []
    if len(trimmed) <= _MAX_CANDIDATE_TOKENS:
        spans = [trimmed]
        if len(trimmed) > 2:
            spans.extend(trimmed[index : index + 2] for index in range(len(trimmed) - 1))
        return spans

    spans = []
    for width in (4, 3, 2):
        for index in range(0, len(trimmed) - width + 1):
            spans.append(trimmed[index : index + width])
    return spans


def _trim_generic_edges(tokens: list[str]) -> list[str]:
    start = 0
    end = len(tokens)
    while start < end and tokens[start] in _GENERIC_TERMS:
        start += 1
    while end > start and tokens[end - 1] in _GENERIC_TERMS:
        end -= 1
    return tokens[start:end]


def _valid_candidate_tokens(tokens: list[str]) -> bool:
    if not tokens or len(tokens) > _MAX_CANDIDATE_TOKENS:
        return False
    if not any(_has_letter(token) for token in tokens):
        return False
    specific = [
        token
        for token in tokens
        if token not in _STOP_TERMS and token not in _GENERIC_TERMS and _has_letter(token)
    ]
    if not specific:
        return False
    if len(tokens) == 1:
        token = tokens[0]
        return len(token) >= 3 and token not in _GENERIC_TERMS
    return tokens[0] not in _STOP_TERMS and tokens[-1] not in _STOP_TERMS


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]


def _token_groups(text: str) -> list[list[str]]:
    groups: list[list[str]] = []
    for sentence in re.split(r"[\n\r.!?;:]+", text):
        tokens = _tokens(sentence)
        if tokens:
            groups.append(tokens)
    return groups


def _label_from_tokens(tokens: list[str]) -> str:
    formatted = [_format_token(token) for token in tokens]
    label = " ".join(formatted).strip()
    return label[:1].upper() + label[1:] if label else ""


def _format_token(token: str) -> str:
    if token in _UPPERCASE_TERMS:
        return token.upper()
    return token


def _unique_labels(labels: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for label in labels:
        key = _candidate_key(label)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(label)
    return unique


def _unique_tuple(values: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return tuple(unique)


def _candidate_key(label: str) -> str:
    return " ".join(_tokens(label)).casefold()


def _concept_id(label: str) -> str:
    base = slugify(_candidate_key(label).replace(" ", "_")).casefold()
    if not base or base == "empty":
        digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:12]
        base = f"non_ascii_{digest}"
    return f"concept_{base}"


def _aggregate_confidence(aggregate: _ConceptAggregate) -> float:
    base = max((source.confidence or 0.5) for source in aggregate.evidence_sources.values())
    signal_bonus = min(0.16, 0.04 * max(len(aggregate.source_signal_counts) - 1, 0))
    unit_bonus = min(0.12, 0.04 * max(len(aggregate.source_unit_ids) - 1, 0))
    occurrence_bonus = min(0.08, 0.01 * max(aggregate.occurrence_count - 1, 0))
    return round(min(0.95, base + signal_bonus + unit_bonus + occurrence_bonus), 3)


def _summary(
    *,
    project_dir: Path,
    evidence_units_path: Path,
    output_path: Path,
    manifest_path: Path,
    result: ConceptCandidateExtractionResult,
) -> dict[str, Any]:
    diagnostics = dict(result.diagnostics)
    evidence_units_total = int(diagnostics.get("evidence_units_total") or 0)
    units_with_candidates = int(diagnostics.get("evidence_units_with_candidates") or 0)
    coverage_ratio = (
        round(units_with_candidates / evidence_units_total, 6) if evidence_units_total else None
    )
    return {
        "schema_version": CONCEPT_CANDIDATE_EXTRACTION_SCHEMA_VERSION,
        "concept_graph_schema_version": CONCEPT_GRAPH_SCHEMA_VERSION,
        "relation_extraction_schema_version": CONCEPT_RELATION_EXTRACTION_SCHEMA_VERSION,
        "lecture_id": result.lecture_id,
        "paths": {
            "evidence_units": _public_project_path(project_dir, evidence_units_path),
            "concept_graph": _public_project_path(project_dir, output_path),
            "project_manifest": _public_project_path(project_dir, manifest_path),
        },
        "counts": {
            "evidence_units_total": evidence_units_total,
            "evidence_units_with_candidates": units_with_candidates,
            "concept_candidates_total": diagnostics["concept_candidates_total"],
            "source_evidence_total": diagnostics["source_evidence_total"],
            "source_texts_total": diagnostics["source_texts_total"],
            "extracted_mentions_total": diagnostics["extracted_mentions_total"],
            "pruned_mentions_total": diagnostics["pruned_mentions_total"],
            "raw_concept_candidates_total": diagnostics["raw_concept_candidates_total"],
            "relation_edges_total": diagnostics["relations"]["relation_edges_total"],
            "candidate_relation_edges_total": diagnostics["relations"][
                "candidate_relation_edges_total"
            ],
            "verified_relation_edges_total": diagnostics["relations"][
                "verified_relation_edges_total"
            ],
            "timestamp_only_candidate_relation_edges_total": diagnostics["relations"][
                "timestamp_only_candidate_relation_edges_total"
            ],
        },
        "coverage": {
            "unit_coverage_ratio": coverage_ratio,
            "source_signal_counts": diagnostics["source_signal_counts"],
            "source_text_counts": diagnostics["source_text_counts"],
            "relation_type_counts": diagnostics["relations"]["relation_type_counts"],
            "relation_source_signal_counts": diagnostics["relations"]["source_signal_counts"],
        },
        "canonicalization": diagnostics["canonicalization"],
        "relation_extraction": {
            "schema_version": diagnostics["relation_extraction_schema_version"],
            "timestamp_fallback_counted_as_verified": False,
            "timestamp_only_verified_demotions_total": diagnostics["relations"][
                "timestamp_only_verified_demotions_total"
            ],
            "pruned_explicit_relations_total": diagnostics["relations"][
                "pruned_explicit_relations_total"
            ],
            "raw_text_redacted": True,
        },
        "public_safety": {
            "raw_text_redacted": True,
            "raw_frame_paths_redacted": True,
            "local_absolute_paths_redacted": True,
        },
        "public_note": CONCEPT_CANDIDATE_PUBLIC_NOTE,
    }


def _update_project_manifest(
    *,
    manifest_path: Path,
    project_dir: Path,
    concept_graph_path: Path,
    summary: dict[str, Any],
) -> None:
    payload = _load_json_object(manifest_path)
    artifacts = payload.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        payload["artifacts"] = artifacts
    artifacts["concept_graph"] = _public_project_path(project_dir, concept_graph_path)

    counts = payload.setdefault("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        payload["counts"] = counts
    counts["concept_candidates"] = summary["counts"]["concept_candidates_total"]
    counts["concept_relations"] = summary["counts"]["relation_edges_total"]

    payload["concept_candidate_extraction"] = {
        "schema_version": summary["schema_version"],
        "lecture_id": summary["lecture_id"],
        "counts": summary["counts"],
        "coverage": summary["coverage"],
        "canonicalization": summary["canonicalization"],
        "relation_extraction": summary["relation_extraction"],
        "public_safety": summary["public_safety"],
        "public_note": summary["public_note"],
    }
    write_json(manifest_path, payload)


def _lecture_id(
    *,
    explicit: str | None,
    manifest: dict[str, Any],
    evidence_units: list[dict[str, Any]],
    project_dir: Path,
) -> str:
    if _text(explicit):
        return _text(explicit)
    for key in ("lecture_id", "video_id", "project_id"):
        value = _text(manifest.get(key))
        if value:
            return value
    for row in evidence_units:
        for key in ("lecture_id", "video_id", "project_id"):
            value = _text(row.get(key))
            if value:
                return value
    return project_dir.name


def _transcript_source_id(unit: dict[str, Any]) -> str | None:
    source_segment_ids = _string_list(unit.get("source_segment_ids"))
    if source_segment_ids:
        return source_segment_ids[0]
    return _text(unit.get("target_segment_id")) or None


def _entity_confidence(entity: dict[str, Any], source_signal: str) -> float:
    confidence = _optional_float(entity.get("confidence"))
    if confidence is None:
        return DEFAULT_SOURCE_CONFIDENCE[source_signal]
    return max(0.0, min(1.0, confidence))


def _is_ocr_like_entity(entity: dict[str, Any]) -> bool:
    source = _text(entity.get("source")).casefold()
    entity_type = _text(entity.get("entity_type")).casefold()
    return source.startswith("ocr") or entity_type == "ocr_text"


def _resolve_project_output_path(*, project_dir: Path, path: Path | None, default: Path) -> Path:
    if path is None:
        return default
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_dir / candidate).resolve()


def _public_project_path(project_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _confidence_threshold(value: float) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    return parsed


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _text_values(values: Iterable[Any]) -> list[str]:
    texts: list[str] = []
    for value in values:
        if isinstance(value, dict):
            texts.extend(_text_values(value.values()))
        elif isinstance(value, (list, tuple, set)):
            texts.extend(_text_values(value))
        else:
            text = _text(value)
            if text:
                texts.append(text)
    return texts


def _string_list(value: Any) -> list[str]:
    return _text_values([value])


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_letter(token: str) -> bool:
    return any(character.isalpha() for character in token)


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _snake_case(value: str) -> str:
    text = value.strip().replace("-", "_").replace(" ", "_")
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()
    return re.sub(r"_+", "_", text).strip("_")
