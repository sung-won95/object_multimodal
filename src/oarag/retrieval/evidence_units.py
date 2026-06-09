from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from oarag.core.domain_lexicon import DomainLexicon, load_domain_lexicon
from oarag.core.io import write_json, write_jsonl
from oarag.core.schemas import slugify
from oarag.graph.concept_graph_schema import (
    CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH,
    TIMESTAMP_ONLY_SOURCE_SIGNALS,
    ConceptGraphConceptNode,
    ConceptGraphRecord,
    ConceptGraphRelationEdge,
    load_concept_graph_artifact,
)
from oarag.retrieval.evidence import (
    resolve_window_config,
    segment_sort_key,
    select_window_segments,
)
from oarag.retrieval.project_index import iter_jsonl_documents, segment_artifact_path
from oarag.vision.vlm_evidence_validator import is_paper_quality_vlm_entity


EVIDENCE_UNITS_ARTIFACT_RELATIVE_PATH = Path("segments") / "evidence_units.jsonl"
VISUAL_STATES_ARTIFACT_RELATIVE_PATH = Path("manifests") / "visual_states.jsonl"
VISUAL_STATES_SCHEMA_VERSION = "oarag-visual-states-jsonl-v1"
VISUAL_STATE_COVERAGE_SCHEMA_VERSION = "oarag-visual-state-coverage-v1"
LINK_DIAGNOSTICS_SCHEMA_VERSION = "oarag-object-link-diagnostics-public-v1"
CONCEPT_FIELD_COVERAGE_SCHEMA_VERSION = "oarag-evidence-unit-concept-field-coverage-v1"
SEARCH_FIELD_COVERAGE_SCHEMA_VERSION = "oarag-evidence-unit-search-field-coverage-v1"
DEFAULT_STATE_PADDING_SECONDS = 15.0
VISUAL_STATE_PUBLIC_NOTE = (
    "Visual states are candidate interval support derived from sampled-frame midpoints "
    "or an explicit visual_states artifact. First/last interval padding is a coverage "
    "fallback, not verified object persistence, and timestamp-only overlap is not "
    "counted as verified object alignment."
)
LINK_DIAGNOSTICS_PUBLIC_NOTE = (
    "Candidate visual support is reported separately from verified object alignment. "
    "Timestamp fallback links are candidate/fallback evidence only and are never "
    "counted as verified object alignment unless an explicit verified field is present."
)
CONCEPT_FIELD_PUBLIC_NOTE = (
    "Concept field coverage reports public-safe counts only. Concept graph labels, "
    "aliases, and relation text may enrich local evidence-unit search, but timestamp-only "
    "concept relation signals remain candidate-only and are not verified object alignment."
)
SEARCH_FIELD_PUBLIC_NOTE = (
    "Search-field coverage reports public-safe availability counts only. Local evidence "
    "unit artifacts may contain transcript/evidence text for retrieval, but public reports "
    "must expose counts, buckets, hashes, and flags instead of raw text."
)
CANDIDATE_LINK_SIGNAL_KEYS = (
    "temporal_overlap",
    "lexical_overlap",
    "mention_deictic_hook",
    "spatial_position",
    "visual_text_overlap",
    "vlm_object_visual_description_overlap",
    "semantic_domain_hint",
    "timestamp_fallback",
)
VERIFIED_LINK_SOURCE_KEYS = (
    "explicit_verified_flag",
    "explicit_verified_status",
    "human_gold",
    "vlm_verifier",
    "strict_deterministic_rule",
    "unspecified_verified",
)
TRANSCRIPT_KEYWORD_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "before",
    "between",
    "from",
    "here",
    "into",
    "just",
    "like",
    "more",
    "next",
    "only",
    "over",
    "that",
    "then",
    "there",
    "these",
    "this",
    "those",
    "through",
    "with",
    "what",
    "when",
    "where",
    "which",
    "while",
    "will",
    "would",
}
CONCEPT_PHRASE_STOPWORDS = TRANSCRIPT_KEYWORD_STOPWORDS | {
    "and",
    "are",
    "can",
    "does",
    "for",
    "has",
    "have",
    "how",
    "its",
    "our",
    "see",
    "the",
    "their",
    "they",
    "use",
    "using",
    "was",
    "were",
    "you",
    "your",
}
CONCEPT_PHRASE_SIGNAL_TERMS = {
    "algorithm",
    "attention",
    "bias",
    "classifier",
    "curve",
    "descent",
    "distribution",
    "embedding",
    "estimate",
    "estimates",
    "feature",
    "function",
    "gradient",
    "layer",
    "learning",
    "loss",
    "matrix",
    "model",
    "normalization",
    "objective",
    "optimizer",
    "probability",
    "rate",
    "reduction",
    "regression",
    "representation",
    "token",
    "training",
    "variance",
    "vector",
    "weight",
}


def build_project_visual_states(
    *,
    project_dir: Path,
    output_path: Path | None = None,
    frames_manifest: Path | None = None,
    manifest_path: Path | None = None,
    state_padding_seconds: float = DEFAULT_STATE_PADDING_SECONDS,
    min_visual_states: int | None = None,
    fail_on_visual_state_gate: bool = False,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    frames_path = _optional_existing_project_path(
        project_dir=resolved_project_dir,
        path=frames_manifest,
        default=resolved_project_dir / "manifests" / "frames_manifest.jsonl",
        artifact_name="frames manifest",
    )
    resolved_output_path = _resolve_project_output_path(
        project_dir=resolved_project_dir,
        path=output_path,
        default=resolved_project_dir / VISUAL_STATES_ARTIFACT_RELATIVE_PATH,
    )
    resolved_manifest_path = _resolve_project_output_path(
        project_dir=resolved_project_dir,
        path=manifest_path,
        default=resolved_project_dir / "manifests" / "project_manifest.json",
    )
    defaults = _project_defaults_from_manifest(resolved_manifest_path)
    frame_rows = list(iter_jsonl_documents(frames_path)) if frames_path else []
    visual_states = build_visual_states(
        frame_rows,
        default_project_id=defaults.get("project_id"),
        default_video_id=defaults.get("video_id"),
        state_padding_seconds=state_padding_seconds,
    )
    write_visual_states_artifact(resolved_output_path, visual_states)
    interval_summary = _interval_duration_summary(visual_states)
    gate = _visual_state_gate_summary(
        visual_states_total=len(visual_states),
        evidence_units_total=None,
        evidence_units_with_visual_state=None,
        min_visual_states=min_visual_states,
        min_unit_coverage_ratio=None,
    )
    summary = {
        "schema_version": VISUAL_STATES_SCHEMA_VERSION,
        "project_dir": str(resolved_project_dir),
        "paths": {
            "frames_manifest": str(frames_path) if frames_path else None,
            "visual_states": str(resolved_output_path),
            "project_manifest": str(resolved_manifest_path),
        },
        "source": "sampled_frame_midpoints",
        "state_padding_seconds": _nonnegative_float(
            state_padding_seconds,
            default=DEFAULT_STATE_PADDING_SECONDS,
        ),
        "counts": {
            "frames_total": len(frame_rows),
            "visual_states_total": len(visual_states),
        },
        "interval_duration_seconds": interval_summary,
        "coverage_gate": gate,
        "public_note": VISUAL_STATE_PUBLIC_NOTE,
    }
    _update_visual_state_manifest(
        manifest_path=resolved_manifest_path,
        visual_states_path=resolved_output_path,
        summary=summary,
    )
    _raise_for_failed_visual_state_gate(
        gate=gate,
        fail_on_visual_state_gate=fail_on_visual_state_gate,
    )
    return summary


def build_project_evidence_units(
    *,
    project_dir: Path,
    output_path: Path | None = None,
    segments: Path | None = None,
    frames_manifest: Path | None = None,
    visual_states: Path | None = None,
    visual_states_output: Path | None = None,
    visual_entities: Path | None = None,
    entity_links: Path | None = None,
    concept_graph: Path | None = None,
    domain_lexicon: Path | None = None,
    manifest_path: Path | None = None,
    window_seconds: float | None = None,
    neighbor_count: int = 1,
    previous_neighbor_count: int | None = None,
    next_neighbor_count: int | None = None,
    window_before_seconds: float | None = None,
    window_after_seconds: float | None = None,
    state_padding_seconds: float = DEFAULT_STATE_PADDING_SECONDS,
    visual_state_min_coverage_ratio: float | None = None,
    visual_state_min_total: int | None = None,
    fail_on_visual_state_gate: bool = False,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    segments_path = segment_artifact_path(resolved_project_dir, segments=segments)
    frames_path = _optional_existing_project_path(
        project_dir=resolved_project_dir,
        path=frames_manifest,
        default=resolved_project_dir / "manifests" / "frames_manifest.jsonl",
        artifact_name="frames manifest",
    )
    visual_states_path = (
        _optional_existing_project_path(
            project_dir=resolved_project_dir,
            path=visual_states,
            default=resolved_project_dir / VISUAL_STATES_ARTIFACT_RELATIVE_PATH,
            artifact_name="visual states",
        )
        if visual_states is not None
        else None
    )
    resolved_visual_states_output_path = (
        _resolve_project_output_path(
            project_dir=resolved_project_dir,
            path=visual_states_output,
            default=resolved_project_dir / VISUAL_STATES_ARTIFACT_RELATIVE_PATH,
        )
        if visual_states_output is not None
        else None
    )
    visual_entities_path = _optional_existing_project_path(
        project_dir=resolved_project_dir,
        path=visual_entities,
        default=resolved_project_dir / "manifests" / "visual_entities.jsonl",
        artifact_name="visual entities",
    )
    entity_links_path = _optional_existing_project_path(
        project_dir=resolved_project_dir,
        path=entity_links,
        default=resolved_project_dir / "manifests" / "entity_links.jsonl",
        artifact_name="entity links",
    )
    concept_graph_path = _optional_existing_project_path(
        project_dir=resolved_project_dir,
        path=concept_graph,
        default=resolved_project_dir / CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH,
        artifact_name="concept graph",
    )
    resolved_output_path = _resolve_project_output_path(
        project_dir=resolved_project_dir,
        path=output_path,
        default=resolved_project_dir / EVIDENCE_UNITS_ARTIFACT_RELATIVE_PATH,
    )
    resolved_manifest_path = _resolve_project_output_path(
        project_dir=resolved_project_dir,
        path=manifest_path,
        default=resolved_project_dir / "manifests" / "project_manifest.json",
    )

    segment_rows = list(iter_jsonl_documents(segments_path))
    frame_rows = list(iter_jsonl_documents(frames_path)) if frames_path else []
    visual_entity_rows = list(iter_jsonl_documents(visual_entities_path)) if visual_entities_path else []
    entity_link_rows = list(iter_jsonl_documents(entity_links_path)) if entity_links_path else []
    concept_graph_records = (
        load_concept_graph_artifact(concept_graph_path)
        if concept_graph_path is not None
        else []
    )
    domain_lexicon_source = load_domain_lexicon(
        project_dir=resolved_project_dir,
        domain_lexicon_path=domain_lexicon,
    )
    if visual_states_path is not None:
        visual_state_rows = load_visual_states_artifact(visual_states_path)
        visual_state_source = "external_visual_states_artifact"
    else:
        visual_state_rows = build_visual_states(
            frame_rows,
            default_project_id=_first_text(segment_rows, "project_id"),
            default_video_id=_first_text(segment_rows, "video_id"),
            state_padding_seconds=state_padding_seconds,
        )
        visual_state_source = "sampled_frame_midpoints"
    if resolved_visual_states_output_path is not None:
        write_visual_states_artifact(resolved_visual_states_output_path, visual_state_rows)
    documents = build_evidence_unit_documents(
        segment_rows,
        frames=frame_rows,
        visual_states=visual_state_rows,
        visual_entities=visual_entity_rows,
        entity_links=entity_link_rows,
        concept_graph_records=concept_graph_records,
        domain_lexicon=domain_lexicon_source,
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
        state_padding_seconds=state_padding_seconds,
    )
    write_jsonl(resolved_output_path, documents)

    counts = {
        "segments_total": len(segment_rows),
        "frames_total": len(frame_rows),
        "visual_entities_total": len(visual_entity_rows),
        "entity_links_total": len(entity_link_rows),
        "concept_graph_records_total": len(concept_graph_records),
        "concept_nodes_total": sum(
            1 for record in concept_graph_records if isinstance(record, ConceptGraphConceptNode)
        ),
        "concept_relation_edges_total": sum(
            1 for record in concept_graph_records if isinstance(record, ConceptGraphRelationEdge)
        ),
        "domain_lexicon_canonical_terms_total": domain_lexicon_source.metadata()[
            "canonical_term_count"
        ],
        "evidence_units_total": len(documents),
        "units_with_visual_state": sum(1 for document in documents if document["source_quality"]["has_visual_state"]),
        "units_with_visual_entity": sum(1 for document in documents if document["source_quality"]["has_visual_entity"]),
        "units_with_vlm_entity": sum(1 for document in documents if document["source_quality"]["has_vlm_entity"]),
        "units_with_concept": sum(1 for document in documents if document["source_quality"]["has_concept"]),
        "units_with_concept_relation": sum(
            1 for document in documents if document["source_quality"]["has_concept_relation"]
        ),
        "units_with_concept_search_text": sum(1 for document in documents if _has_concept_search_text(document)),
        "units_with_evidence_text": sum(1 for document in documents if _text(document.get("evidence_text"))),
        "units_with_semantic_text": sum(1 for document in documents if _text(document.get("semantic_text"))),
        "units_with_transcript_keywords": sum(1 for document in documents if _string_list(document.get("transcript_keywords"))),
        "units_with_visual_state_search_text": sum(1 for document in documents if _text(document.get("visual_state_text"))),
        "units_with_visual_entity_search_text": sum(1 for document in documents if _text(document.get("visual_entity_text"))),
        "units_with_link_signal_search_text": sum(
            1
            for document in documents
            if _text(document.get("candidate_link_signal_summary"))
            or _text(document.get("verified_link_signal_summary"))
        ),
        "units_with_candidate_link": sum(
            1
            for document in documents
            if document["source_quality"]["candidate_link_count"] > 0
        ),
        "units_with_verified_link": sum(1 for document in documents if document["source_quality"]["has_verified_link"]),
        "units_with_timestamp_fallback_link": sum(
            1
            for document in documents
            if document["source_quality"]["timestamp_fallback_link_count"] > 0
        ),
        "units_with_candidate_visual_support": sum(
            1
            for document in documents
            if document["source_quality"]["candidate_visual_support"]["has_candidate_visual_support"]
        ),
        "units_with_verified_object_alignment": sum(
            1
            for document in documents
            if document["source_quality"]["verified_object_alignment"]["has_verified_object_alignment"]
        ),
        "units_with_detected_text": sum(1 for document in documents if document["source_quality"]["has_detected_text"]),
        "units_with_visual_description": sum(
            1 for document in documents if document["source_quality"]["has_visual_description"]
        ),
        "timestamp_fallback_links": sum(
            document["source_quality"]["timestamp_fallback_link_count"] for document in documents
        ),
        "candidate_links": sum(document["source_quality"]["candidate_link_count"] for document in documents),
        "verified_links": sum(document["source_quality"]["verified_link_count"] for document in documents),
        "concept_mentions": sum(document["source_quality"]["concept_count"] for document in documents),
        "concept_relation_mentions": sum(
            document["source_quality"]["concept_relation_count"] for document in documents
        ),
    }
    visual_state_coverage = _visual_state_coverage_summary(
        visual_states=visual_state_rows,
        documents=documents,
        source=visual_state_source,
        min_unit_coverage_ratio=visual_state_min_coverage_ratio,
        min_visual_states=visual_state_min_total,
    )
    link_diagnostics = _evidence_unit_link_diagnostics(documents)
    concept_field_coverage = _concept_field_coverage_summary(
        documents=documents,
        concept_graph_records=concept_graph_records,
        source="concept_graph_artifact" if concept_graph_path is not None else "not_available",
    )
    search_field_coverage = _search_field_coverage_summary(documents)
    summary = {
        "project_dir": str(resolved_project_dir),
        "paths": {
            "segments": str(segments_path),
            "frames_manifest": str(frames_path) if frames_path else None,
            "visual_states": (
                str(resolved_visual_states_output_path or visual_states_path)
                if (resolved_visual_states_output_path or visual_states_path)
                else None
            ),
            "visual_entities": str(visual_entities_path) if visual_entities_path else None,
            "entity_links": str(entity_links_path) if entity_links_path else None,
            "concept_graph": str(concept_graph_path) if concept_graph_path else None,
            "evidence_units": str(resolved_output_path),
            "project_manifest": str(resolved_manifest_path),
        },
        "window_config": _window_config(
            window_seconds=window_seconds,
            neighbor_count=neighbor_count,
            previous_neighbor_count=previous_neighbor_count,
            next_neighbor_count=next_neighbor_count,
            window_before_seconds=window_before_seconds,
            window_after_seconds=window_after_seconds,
        ),
        "visual_state_config": {
            "mode": visual_state_source,
            "schema_version": VISUAL_STATES_SCHEMA_VERSION,
            "state_padding_seconds": _nonnegative_float(
                state_padding_seconds,
                default=DEFAULT_STATE_PADDING_SECONDS,
            ),
            "artifact_loaded": visual_states_path is not None,
            "artifact_written": resolved_visual_states_output_path is not None,
            "public_note": VISUAL_STATE_PUBLIC_NOTE,
        },
        "visual_state_coverage": visual_state_coverage,
        "counts": counts,
        "alignment_status_counts": _status_counts(documents),
        "link_diagnostics": link_diagnostics,
        "concept_field_coverage": concept_field_coverage,
        "search_field_coverage": search_field_coverage,
    }
    _update_project_manifest(
        manifest_path=resolved_manifest_path,
        evidence_units_path=resolved_output_path,
        summary=summary,
    )
    _raise_for_failed_visual_state_gate(
        gate=visual_state_coverage["coverage_gate"],
        fail_on_visual_state_gate=fail_on_visual_state_gate,
    )
    return summary


def build_evidence_unit_documents(
    segments: list[dict[str, Any]],
    *,
    frames: list[dict[str, Any]] | None = None,
    visual_states: list[dict[str, Any]] | None = None,
    visual_entities: list[dict[str, Any]] | None = None,
    entity_links: list[dict[str, Any]] | None = None,
    concept_graph_records: list[ConceptGraphRecord] | None = None,
    domain_lexicon: DomainLexicon | None = None,
    window_seconds: float | None = None,
    neighbor_count: int = 1,
    previous_neighbor_count: int | None = None,
    next_neighbor_count: int | None = None,
    window_before_seconds: float | None = None,
    window_after_seconds: float | None = None,
    state_padding_seconds: float = DEFAULT_STATE_PADDING_SECONDS,
) -> list[dict[str, Any]]:
    sorted_segments = [
        segment for _, segment in sorted(enumerate(segments), key=lambda item: segment_sort_key(item[1], item[0]))
    ]
    visual_state_rows = (
        [_normalize_visual_state_record(state, row_number=index + 1) for index, state in enumerate(visual_states)]
        if visual_states is not None
        else build_visual_states(
            frames or [],
            default_project_id=_first_text(sorted_segments, "project_id"),
            default_video_id=_first_text(sorted_segments, "video_id"),
            state_padding_seconds=state_padding_seconds,
        )
    )
    states_by_id = {str(state["visual_state_id"]): state for state in visual_state_rows}
    entities_by_frame_id = _group_by_text(visual_entities or [], "frame_id")
    entities_by_id = {str(entity.get("entity_id")): entity for entity in visual_entities or []}
    links_by_segment_id = _group_by_text(entity_links or [], "segment_id")
    concept_context_by_evidence_unit = _concept_context_by_evidence_unit(
        concept_graph_records or []
    )
    domain_lexicon = domain_lexicon or DomainLexicon()
    window_config = _window_config(
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
    )

    documents: list[dict[str, Any]] = []
    for target in sorted_segments:
        target_segment_id = str(target.get("segment_id") or "")
        if not target_segment_id:
            continue
        window_segments = select_window_segments(
            sorted_segments,
            target_segment_id=target_segment_id,
            window_seconds=window_seconds,
            neighbor_count=neighbor_count,
            previous_neighbor_count=previous_neighbor_count,
            next_neighbor_count=next_neighbor_count,
            window_before_seconds=window_before_seconds,
            window_after_seconds=window_after_seconds,
        )
        documents.append(
            _evidence_unit_document(
                target=target,
                window_segments=window_segments,
                states_by_id=states_by_id,
                visual_states=visual_state_rows,
                entities_by_frame_id=entities_by_frame_id,
                entities_by_id=entities_by_id,
                links_by_segment_id=links_by_segment_id,
                concept_context_by_evidence_unit=concept_context_by_evidence_unit,
                domain_lexicon=domain_lexicon,
                window_config=window_config,
            )
        )
    return documents


def build_visual_states(
    frames: list[dict[str, Any]],
    *,
    default_project_id: str | None = None,
    default_video_id: str | None = None,
    state_padding_seconds: float = DEFAULT_STATE_PADDING_SECONDS,
) -> list[dict[str, Any]]:
    padding = _nonnegative_float(state_padding_seconds, default=DEFAULT_STATE_PADDING_SECONDS)
    grouped_frames: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        frame_id = _text(frame.get("frame_id"))
        timestamp = _optional_float(frame.get("timestamp"))
        if not frame_id or timestamp is None:
            continue
        video_id = _text(frame.get("video_id")) or default_video_id or "__default__"
        grouped_frames.setdefault(video_id, []).append(frame)

    states: list[dict[str, Any]] = []
    for video_id, video_frames in sorted(grouped_frames.items()):
        sorted_frames = sorted(
            video_frames,
            key=lambda frame: (
                _optional_float(frame.get("timestamp")) or 0.0,
                _text(frame.get("frame_id")),
            ),
        )
        timestamps = [_optional_float(frame.get("timestamp")) or 0.0 for frame in sorted_frames]
        for index, frame in enumerate(sorted_frames):
            frame_id = _text(frame.get("frame_id"))
            timestamp = timestamps[index]
            start_time = (
                (timestamps[index - 1] + timestamp) / 2.0
                if index > 0
                else max(0.0, timestamp - padding)
            )
            end_time = (
                (timestamp + timestamps[index + 1]) / 2.0
                if index < len(timestamps) - 1
                else timestamp + padding
            )
            state_summary = _compact_text(
                " ".join(
                    _text_values(
                        [
                            frame.get("state_summary"),
                            frame.get("visual_description"),
                            frame.get("detected_text"),
                        ]
                    )
                )
            )
            states.append(
                {
                    "schema_version": VISUAL_STATES_SCHEMA_VERSION,
                    "visual_state_id": f"vstate_{slugify(video_id)}_{slugify(frame_id)}",
                    "project_id": _text(frame.get("project_id")) or default_project_id,
                    "video_id": video_id if video_id != "__default__" else default_video_id,
                    "representative_frame_id": frame_id,
                    "frame_ids": [frame_id],
                    "valid_start_time": round(start_time, 3),
                    "valid_end_time": round(end_time, 3),
                    "interval_source": "sampled_frame_midpoint",
                    "state_summary": state_summary,
                    "detected_text": _text_values([frame.get("detected_text")]),
                    "source": "sampled_frame_interval",
                    "confidence": _optional_float(frame.get("confidence")),
                }
            )
    return states


def write_visual_states_artifact(path: Path, visual_states: list[dict[str, Any]]) -> int:
    return write_jsonl(
        path,
        [_visual_state_artifact_record(state) for state in visual_states],
    )


def load_visual_states_artifact(path: Path) -> list[dict[str, Any]]:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"visual_states artifact not found: {resolved_path}")
    return [
        _normalize_visual_state_record(row, row_number=index + 1)
        for index, row in enumerate(iter_jsonl_documents(resolved_path))
    ]


def _visual_state_artifact_record(state: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_visual_state_record(state)
    return {
        key: value
        for key, value in {
            "schema_version": VISUAL_STATES_SCHEMA_VERSION,
            "visual_state_id": normalized["visual_state_id"],
            "project_id": normalized.get("project_id"),
            "video_id": normalized.get("video_id"),
            "representative_frame_id": normalized.get("representative_frame_id"),
            "frame_ids": normalized.get("frame_ids", []),
            "valid_start_time": normalized["valid_start_time"],
            "valid_end_time": normalized["valid_end_time"],
            "interval_source": normalized.get("interval_source"),
            "state_summary": normalized.get("state_summary", ""),
            "detected_text": normalized.get("detected_text", []),
            "source": normalized.get("source"),
            "confidence": normalized.get("confidence"),
        }.items()
        if value not in (None, "", [])
    }


def _normalize_visual_state_record(
    state: dict[str, Any],
    *,
    row_number: int | None = None,
) -> dict[str, Any]:
    prefix = f"visual_states row {row_number}" if row_number is not None else "visual state"
    visual_state_id = _text(state.get("visual_state_id"))
    if not visual_state_id:
        raise ValueError(f"{prefix} is missing visual_state_id")
    start_time = _optional_float(state.get("valid_start_time"))
    end_time = _optional_float(state.get("valid_end_time"))
    if start_time is None or end_time is None:
        raise ValueError(f"{prefix} is missing valid_start_time/valid_end_time")
    if end_time < start_time:
        start_time, end_time = end_time, start_time
    return {
        "schema_version": VISUAL_STATES_SCHEMA_VERSION,
        "visual_state_id": visual_state_id,
        "project_id": _text(state.get("project_id")) or None,
        "video_id": _text(state.get("video_id")) or None,
        "representative_frame_id": _text(state.get("representative_frame_id")) or None,
        "frame_ids": _string_list(state.get("frame_ids")),
        "valid_start_time": round(start_time, 3),
        "valid_end_time": round(end_time, 3),
        "interval_source": _text(state.get("interval_source")) or _text(state.get("source")) or None,
        "state_summary": _compact_text(_text(state.get("state_summary"))),
        "detected_text": _text_values([state.get("detected_text")]),
        "source": _text(state.get("source")) or None,
        "confidence": _optional_float(state.get("confidence")),
    }


def _visual_state_coverage_summary(
    *,
    visual_states: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    source: str,
    min_unit_coverage_ratio: float | None,
    min_visual_states: int | None,
) -> dict[str, Any]:
    units_with_visual_state = sum(
        1 for document in documents if _mapping(document.get("source_quality")).get("has_visual_state") is True
    )
    evidence_units_total = len(documents)
    transcript_only_units = sum(1 for document in documents if document.get("alignment_status") == "transcript_only")
    coverage_ratio = (
        round(units_with_visual_state / evidence_units_total, 6)
        if evidence_units_total
        else None
    )
    return {
        "schema_version": VISUAL_STATE_COVERAGE_SCHEMA_VERSION,
        "source": source,
        "visual_states_total": len(visual_states),
        "evidence_units_total": evidence_units_total,
        "evidence_units_with_visual_state": units_with_visual_state,
        "transcript_only_units": transcript_only_units,
        "unit_coverage_ratio": coverage_ratio,
        "interval_duration_seconds": _interval_duration_summary(visual_states),
        "coverage_gate": _visual_state_gate_summary(
            visual_states_total=len(visual_states),
            evidence_units_total=evidence_units_total,
            evidence_units_with_visual_state=units_with_visual_state,
            min_visual_states=min_visual_states,
            min_unit_coverage_ratio=min_unit_coverage_ratio,
        ),
        "public_note": VISUAL_STATE_PUBLIC_NOTE,
    }


def _interval_duration_summary(visual_states: list[dict[str, Any]]) -> dict[str, Any]:
    durations = []
    for state in visual_states:
        start_time = _optional_float(state.get("valid_start_time"))
        end_time = _optional_float(state.get("valid_end_time"))
        if start_time is None or end_time is None:
            continue
        durations.append(abs(end_time - start_time))
    buckets = {
        "0-5s": 0,
        "5-15s": 0,
        "15-30s": 0,
        "30-60s": 0,
        "60s+": 0,
    }
    for duration in durations:
        if duration < 5.0:
            buckets["0-5s"] += 1
        elif duration < 15.0:
            buckets["5-15s"] += 1
        elif duration < 30.0:
            buckets["15-30s"] += 1
        elif duration < 60.0:
            buckets["30-60s"] += 1
        else:
            buckets["60s+"] += 1
    if not durations:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "buckets": buckets,
        }
    return {
        "count": len(durations),
        "min": round(min(durations), 3),
        "max": round(max(durations), 3),
        "mean": round(sum(durations) / len(durations), 3),
        "buckets": buckets,
    }


def _visual_state_gate_summary(
    *,
    visual_states_total: int,
    evidence_units_total: int | None,
    evidence_units_with_visual_state: int | None,
    min_visual_states: int | None,
    min_unit_coverage_ratio: float | None,
) -> dict[str, Any]:
    thresholds: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    if min_visual_states is not None:
        threshold = max(0, int(min_visual_states))
        thresholds["min_visual_states"] = threshold
        if visual_states_total < threshold:
            failures.append(
                {
                    "metric": "visual_states_total",
                    "actual": visual_states_total,
                    "minimum": threshold,
                }
            )
    if min_unit_coverage_ratio is not None:
        threshold = max(0.0, min(1.0, float(min_unit_coverage_ratio)))
        actual = (
            evidence_units_with_visual_state / evidence_units_total
            if evidence_units_total
            else 0.0
        )
        thresholds["min_unit_coverage_ratio"] = threshold
        if actual < threshold:
            failures.append(
                {
                    "metric": "unit_coverage_ratio",
                    "actual": round(actual, 6),
                    "minimum": threshold,
                }
            )
    checked = bool(thresholds)
    return {
        "status": "passed" if checked and not failures else "failed" if failures else "not_configured",
        "checked": checked,
        "thresholds": thresholds,
        "failure_count": len(failures),
        "failures": failures,
    }


def _raise_for_failed_visual_state_gate(
    *,
    gate: dict[str, Any],
    fail_on_visual_state_gate: bool,
) -> None:
    if fail_on_visual_state_gate and gate.get("status") == "failed":
        raise ValueError("visual state coverage gate failed")


def _evidence_unit_document(
    *,
    target: dict[str, Any],
    window_segments: list[dict[str, Any]],
    states_by_id: dict[str, dict[str, Any]],
    visual_states: list[dict[str, Any]],
    entities_by_frame_id: dict[str, list[dict[str, Any]]],
    entities_by_id: dict[str, dict[str, Any]],
    links_by_segment_id: dict[str, list[dict[str, Any]]],
    concept_context_by_evidence_unit: dict[str, dict[str, Any]],
    domain_lexicon: DomainLexicon,
    window_config: dict[str, Any],
) -> dict[str, Any]:
    source_segment_ids = [_text(segment.get("segment_id")) for segment in window_segments]
    source_segment_ids = [segment_id for segment_id in source_segment_ids if segment_id]
    start_time, end_time = _window_bounds(window_segments)
    target_segment_id = _text(target.get("segment_id"))
    evidence_unit_id = f"evu_{slugify(target_segment_id)}"
    video_id = _text(target.get("video_id"))
    window_state_ids = _state_ids_for_window(
        visual_states,
        video_id=video_id,
        start_time=start_time,
        end_time=end_time,
    )
    window_states = [states_by_id[state_id] for state_id in window_state_ids if state_id in states_by_id]
    visual_entity_rows = _entities_for_states(window_states, entities_by_frame_id)
    window_links = [
        link
        for segment_id in source_segment_ids
        for link in links_by_segment_id.get(segment_id, [])
        if _text(link.get("entity_id")) in entities_by_id
    ]
    visual_entity_rows = _merge_entities(visual_entity_rows, window_links, entities_by_id)
    visual_entity_ids = [_text(entity.get("entity_id")) for entity in visual_entity_rows if _text(entity.get("entity_id"))]

    link_statuses = {
        _text(link.get("link_id")): _link_alignment_status(link)
        for link in window_links
        if _text(link.get("link_id"))
    }
    verified_link_ids = [
        link_id for link_id, status in link_statuses.items() if status == "verified"
    ]
    candidate_link_ids = [
        link_id for link_id, status in link_statuses.items() if status != "verified"
    ]
    transcript_window_text = _transcript_window_text(window_segments)
    transcript_keywords = _transcript_keywords(transcript_window_text)
    visual_state_text = _visual_state_summary_text(window_states)
    visual_entity_text = _visual_entity_summary_text(visual_entity_rows)
    visual_text = _compact_text(
        " ".join(_unique_text_values([visual_state_text, visual_entity_text]))
    )
    concept_context = _public_concept_search_context(
        base_context=concept_context_by_evidence_unit.get(evidence_unit_id),
        transcript_text=transcript_window_text,
        visual_state_text=visual_state_text,
        visual_entity_text=visual_entity_text,
        domain_lexicon=domain_lexicon,
    )
    concept_text = _concept_summary_text(concept_context)
    source_quality = _source_quality(
        visual_states=window_states,
        visual_entities=visual_entity_rows,
        links=window_links,
        link_statuses=link_statuses,
        concept_context=concept_context,
    )
    candidate_link_signal_summary = _link_signal_summary_text(
        source_quality.get("candidate_link_signal_counts"),
        prefix="candidate",
    )
    verified_link_signal_summary = _link_signal_summary_text(
        source_quality.get("verified_link_source_counts"),
        prefix="verified",
    )
    link_signal_text = _compact_text(
        " ".join(
            _unique_text_values(
                [candidate_link_signal_summary, verified_link_signal_summary]
            )
        )
    )
    evidence_text = _compact_text(
        " ".join(
            part
            for part in (
                f"Transcript: {transcript_window_text}" if transcript_window_text else "",
                f"Transcript keywords: {' '.join(transcript_keywords)}" if transcript_keywords else "",
                f"Visual: {visual_text}" if visual_text else "",
                f"Concepts: {concept_text}" if concept_text else "",
                f"Link signals: {link_signal_text}" if link_signal_text else "",
            )
            if part
        )
    )
    semantic_text = _compact_text(
        f"{transcript_window_text} {' '.join(transcript_keywords)} "
        f"{visual_text} {concept_text} {link_signal_text}"
    )
    source_quality = dict(source_quality)
    source_quality["search_field_coverage"] = _search_field_coverage(
        {
            "evidence_text": evidence_text,
            "semantic_text": semantic_text,
            "transcript_keywords": transcript_keywords,
            "concept_search_text": concept_text,
            "visual_state_text": visual_state_text,
            "visual_entity_text": visual_entity_text,
            "candidate_link_signal_summary": candidate_link_signal_summary,
            "verified_link_signal_summary": verified_link_signal_summary,
        }
    )
    return {
        "evidence_unit_id": evidence_unit_id,
        "project_id": _text(target.get("project_id")),
        "video_id": video_id,
        "target_segment_id": target_segment_id,
        "source_segment_ids": source_segment_ids,
        "start_time": start_time,
        "end_time": end_time,
        "transcript_window_text": transcript_window_text,
        "transcript_keywords": transcript_keywords,
        "visual_state_ids": window_state_ids,
        "visual_states": [_visual_state_context(state) for state in window_states],
        "visual_state_text": visual_state_text,
        "visual_entity_ids": visual_entity_ids,
        "visual_entities": [_visual_entity_context(entity) for entity in visual_entity_rows],
        "visual_entity_text": visual_entity_text,
        "verified_entity_link_ids": verified_link_ids,
        "candidate_entity_link_ids": candidate_link_ids,
        "candidate_entity_link_statuses": {
            link_id: status for link_id, status in link_statuses.items() if status != "verified"
        },
        "concept_ids": concept_context["concept_ids"],
        "concept_labels": concept_context["concept_labels"],
        "concept_aliases": concept_context["concept_aliases"],
        "concept_relation_text": concept_context["concept_relation_text"],
        "concept_search_text": concept_text,
        "concepts": concept_context["concepts"],
        "concept_relations": concept_context["concept_relations"],
        "candidate_link_signal_summary": candidate_link_signal_summary,
        "verified_link_signal_summary": verified_link_signal_summary,
        "modality": ["speech", "visual"] if source_quality["has_visual_state"] or source_quality["has_visual_entity"] else ["speech"],
        "evidence_text": evidence_text,
        "semantic_text": semantic_text,
        "alignment_score": _alignment_score(window_links, link_statuses),
        "alignment_status": _alignment_status(source_quality),
        "source_quality": source_quality,
        "window_config": window_config,
    }


def _state_ids_for_window(
    visual_states: list[dict[str, Any]],
    *,
    video_id: str,
    start_time: float | None,
    end_time: float | None,
) -> list[str]:
    if start_time is None or end_time is None:
        return []
    state_ids: list[str] = []
    for state in visual_states:
        state_video_id = _text(state.get("video_id"))
        if state_video_id and video_id and state_video_id != video_id:
            continue
        if _overlaps(
            _optional_float(state.get("valid_start_time")),
            _optional_float(state.get("valid_end_time")),
            start_time,
            end_time,
        ):
            state_ids.append(_text(state.get("visual_state_id")))
    return [state_id for state_id in state_ids if state_id]


def _entities_for_states(
    states: list[dict[str, Any]],
    entities_by_frame_id: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for state in states:
        for frame_id in _string_list(state.get("frame_ids")):
            for entity in entities_by_frame_id.get(frame_id, []):
                entity_id = _text(entity.get("entity_id"))
                if not entity_id or entity_id in seen:
                    continue
                seen.add(entity_id)
                entities.append(entity)
    return entities


def _merge_entities(
    entities: list[dict[str, Any]],
    links: list[dict[str, Any]],
    entities_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(entities)
    seen = {_text(entity.get("entity_id")) for entity in merged}
    for link in links:
        entity_id = _text(link.get("entity_id"))
        if not entity_id or entity_id in seen:
            continue
        entity = entities_by_id.get(entity_id)
        if entity is None:
            continue
        seen.add(entity_id)
        merged.append(entity)
    return merged


def _link_alignment_status(link: dict[str, Any]) -> str:
    explicit_status = _text(
        link.get("alignment_status")
        or link.get("verification_status")
        or link.get("status")
    ).casefold()
    if explicit_status == "verified" or link.get("verified") is True:
        return "verified"
    if _is_timestamp_fallback_link(link):
        return "timestamp_fallback"
    return "candidate"


def _is_timestamp_fallback_link(link: dict[str, Any]) -> bool:
    evidence = {str(item) for item in link.get("evidence", [])}
    link_type = _text(link.get("link_type")).casefold()
    reason_summary = _text(
        _mapping(link.get("reason_metadata")).get("summary")
    ).casefold()
    has_strong_signal = bool(
        evidence
        - {
            "time_overlap",
            "timestamp_fallback",
        }
    )
    return (
        not has_strong_signal
        or link_type == "time_overlap"
        or reason_summary == "timestamp_fallback_only"
    )


def _source_quality(
    *,
    visual_states: list[dict[str, Any]],
    visual_entities: list[dict[str, Any]],
    links: list[dict[str, Any]],
    link_statuses: dict[str, str],
    concept_context: dict[str, Any],
) -> dict[str, Any]:
    has_vlm_entity = any(_is_vlm_entity(entity) for entity in visual_entities)
    visual_state_detected_text_count = sum(
        1 for state in visual_states if _text_values([state.get("detected_text")])
    )
    visual_entity_detected_text_count = sum(
        1 for entity in visual_entities if _text_values([entity.get("detected_text")])
    )
    visual_description_count = sum(
        1 for entity in visual_entities if _text(entity.get("visual_description"))
    )
    candidate_link_count = sum(1 for status in link_statuses.values() if status == "candidate")
    timestamp_fallback_link_count = sum(
        1 for status in link_statuses.values() if status == "timestamp_fallback"
    )
    verified_link_count = sum(1 for status in link_statuses.values() if status == "verified")
    candidate_signal_counts = _candidate_link_signal_counts(
        links=links,
        link_statuses=link_statuses,
    )
    verified_source_counts = _verified_link_source_counts(
        links=links,
        link_statuses=link_statuses,
    )
    has_candidate_visual_support = bool(
        visual_states
        or visual_entities
        or candidate_link_count
        or timestamp_fallback_link_count
    )
    has_verified_object_alignment = verified_link_count > 0
    concept_count = max(
        len(_string_list(concept_context.get("concept_ids"))),
        len(_string_list(concept_context.get("concept_labels"))),
    )
    concept_relation_count = len(_list_of_dicts(concept_context.get("concept_relations")))
    timestamp_only_concept_relation_count = sum(
        1
        for relation in _list_of_dicts(concept_context.get("concept_relations"))
        if relation.get("timestamp_only_candidate") is True
    )
    return {
        "has_visual_state": bool(visual_states),
        "has_visual_entity": bool(visual_entities),
        "has_vlm_entity": has_vlm_entity,
        "has_concept": concept_count > 0,
        "has_concept_relation": concept_relation_count > 0,
        "has_verified_link": any(status == "verified" for status in link_statuses.values()),
        "uses_ocr_only": bool(visual_entities) and not has_vlm_entity,
        "has_detected_text": bool(visual_state_detected_text_count or visual_entity_detected_text_count),
        "has_visual_description": bool(visual_description_count),
        "has_concept_search_text": _has_concept_search_text(concept_context),
        "visual_state_detected_text_count": visual_state_detected_text_count,
        "visual_entity_detected_text_count": visual_entity_detected_text_count,
        "visual_description_count": visual_description_count,
        "concept_count": concept_count,
        "concept_label_count": len(_string_list(concept_context.get("concept_labels"))),
        "concept_alias_count": len(_string_list(concept_context.get("concept_aliases"))),
        "concept_relation_count": concept_relation_count,
        "timestamp_only_concept_relation_count": timestamp_only_concept_relation_count,
        "has_timestamp_fallback_link": any(
            status == "timestamp_fallback" for status in link_statuses.values()
        ),
        "candidate_link_count": candidate_link_count,
        "timestamp_fallback_link_count": timestamp_fallback_link_count,
        "verified_link_count": verified_link_count,
        "candidate_link_signal_counts": candidate_signal_counts,
        "verified_link_source_counts": verified_source_counts,
        "candidate_visual_support": {
            "has_candidate_visual_support": has_candidate_visual_support,
            "visual_state_count": len(visual_states),
            "visual_entity_count": len(visual_entities),
            "candidate_link_count": candidate_link_count,
            "timestamp_fallback_link_count": timestamp_fallback_link_count,
            "candidate_link_signal_counts": candidate_signal_counts,
            "paper_claim_eligible": False,
        },
        "verified_object_alignment": {
            "has_verified_object_alignment": has_verified_object_alignment,
            "verified_link_count": verified_link_count,
            "verified_link_source_counts": verified_source_counts,
            "timestamp_fallback_counted_as_verified": False,
            "paper_claim_eligible": has_verified_object_alignment,
        },
        "concept_field_coverage": {
            "has_concept_search_text": _has_concept_search_text(concept_context),
            "concept_count": concept_count,
            "concept_label_count": len(_string_list(concept_context.get("concept_labels"))),
            "concept_alias_count": len(_string_list(concept_context.get("concept_aliases"))),
            "concept_relation_count": concept_relation_count,
            "timestamp_only_concept_relation_count": timestamp_only_concept_relation_count,
            "timestamp_only_counted_as_verified_object_alignment": False,
        },
    }


def _alignment_status(source_quality: dict[str, Any]) -> str:
    if source_quality["has_verified_link"]:
        return "verified"
    if (
        source_quality["candidate_link_count"]
        or source_quality["timestamp_fallback_link_count"]
        or source_quality["has_visual_entity"]
        or source_quality["has_visual_state"]
    ):
        return "candidate"
    return "transcript_only"


def _alignment_score(links: list[dict[str, Any]], statuses: dict[str, str]) -> float:
    if not links:
        return 0.0
    weighted_scores: list[float] = []
    for link in links:
        link_id = _text(link.get("link_id"))
        score = _optional_float(link.get("score")) or 0.0
        status = statuses.get(link_id, "candidate")
        if status == "verified":
            weighted_scores.append(max(score, 1.0))
        elif status == "timestamp_fallback":
            weighted_scores.append(min(score, 0.2))
        else:
            weighted_scores.append(min(score, 0.8))
    return round(max(weighted_scores), 3)


def _candidate_link_signal_counts(
    *,
    links: list[dict[str, Any]],
    link_statuses: dict[str, str],
) -> dict[str, int]:
    counts = _zero_count_map(CANDIDATE_LINK_SIGNAL_KEYS)
    for link in links:
        link_id = _text(link.get("link_id"))
        status = link_statuses.get(link_id)
        if status == "verified":
            continue
        evidence = _link_evidence(link)
        if link.get("time_overlap") is True or "time_overlap" in evidence:
            counts["temporal_overlap"] += 1
        if _string_list(link.get("lexical_match")) or evidence & {
            "lexical_match",
            "visual_text_match",
        }:
            counts["lexical_overlap"] += 1
        if _string_list(link.get("mention_candidate")) or evidence & {
            "mention_candidate",
            "reference_cue",
        }:
            counts["mention_deictic_hook"] += 1
        if evidence & {"position_match", "relations_match"}:
            counts["spatial_position"] += 1
        if "visual_text_match" in evidence:
            counts["visual_text_overlap"] += 1
        if evidence & {"visual_description_match", "entity_type_match"}:
            counts["vlm_object_visual_description_overlap"] += 1
        if evidence & {"semantic_hint", "domain_lexicon_match"}:
            counts["semantic_domain_hint"] += 1
        if status == "timestamp_fallback" or "timestamp_fallback" in evidence:
            counts["timestamp_fallback"] += 1
    return counts


def _verified_link_source_counts(
    *,
    links: list[dict[str, Any]],
    link_statuses: dict[str, str],
) -> dict[str, int]:
    counts = _zero_count_map(VERIFIED_LINK_SOURCE_KEYS)
    for link in links:
        link_id = _text(link.get("link_id"))
        if link_statuses.get(link_id) != "verified":
            continue
        matched = False
        explicit_status = _text(
            link.get("alignment_status")
            or link.get("verification_status")
            or link.get("status")
        ).casefold()
        if link.get("verified") is True:
            counts["explicit_verified_flag"] += 1
            matched = True
        if explicit_status == "verified":
            counts["explicit_verified_status"] += 1
            matched = True
        source_text = _verified_source_text(link)
        if any(token in source_text for token in ("human", "gold", "annotator", "annotation")):
            counts["human_gold"] += 1
            matched = True
        if any(token in source_text for token in ("vlm", "vision", "verifier", "validator", "model")):
            counts["vlm_verifier"] += 1
            matched = True
        if any(token in source_text for token in ("strict", "deterministic", "rule")):
            counts["strict_deterministic_rule"] += 1
            matched = True
        if not matched:
            counts["unspecified_verified"] += 1
    return counts


def _verified_source_text(link: dict[str, Any]) -> str:
    metadata = _mapping(link.get("reason_metadata"))
    values = [
        link.get("verification_source"),
        link.get("verified_source"),
        link.get("verified_by"),
        link.get("verifier"),
        link.get("source"),
        link.get("source_model"),
        metadata.get("verification_source"),
        metadata.get("verified_by"),
        metadata.get("verifier"),
        metadata.get("source"),
        metadata.get("source_model"),
        metadata.get("summary"),
    ]
    return " ".join(_text_values(values)).casefold()


def _link_evidence(link: dict[str, Any]) -> set[str]:
    return {_text(item).casefold() for item in link.get("evidence") or [] if _text(item)}


def _zero_count_map(keys: Iterable[str]) -> dict[str, int]:
    return {key: 0 for key in keys}


def _evidence_unit_link_diagnostics(documents: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_signal_counts = _zero_count_map(CANDIDATE_LINK_SIGNAL_KEYS)
    verified_source_counts = _zero_count_map(VERIFIED_LINK_SOURCE_KEYS)
    units_with_candidate_visual_support = 0
    units_with_candidate_link = 0
    units_with_timestamp_fallback_link = 0
    units_with_verified_object_alignment = 0
    candidate_links = 0
    timestamp_fallback_links = 0
    verified_links = 0
    for document in documents:
        source_quality = _mapping(document.get("source_quality"))
        candidate_links += int(source_quality.get("candidate_link_count") or 0)
        timestamp_fallback_links += int(source_quality.get("timestamp_fallback_link_count") or 0)
        verified_links += int(source_quality.get("verified_link_count") or 0)
        if int(source_quality.get("candidate_link_count") or 0) > 0:
            units_with_candidate_link += 1
        if int(source_quality.get("timestamp_fallback_link_count") or 0) > 0:
            units_with_timestamp_fallback_link += 1

        support = _mapping(source_quality.get("candidate_visual_support"))
        verified = _mapping(source_quality.get("verified_object_alignment"))
        if support.get("has_candidate_visual_support") is True:
            units_with_candidate_visual_support += 1
        if verified.get("has_verified_object_alignment") is True:
            units_with_verified_object_alignment += 1
        _add_counts(
            candidate_signal_counts,
            _mapping(source_quality.get("candidate_link_signal_counts")),
        )
        _add_counts(
            verified_source_counts,
            _mapping(source_quality.get("verified_link_source_counts")),
        )
    return {
        "schema_version": LINK_DIAGNOSTICS_SCHEMA_VERSION,
        "evidence_units_total": len(documents),
        "candidate_visual_support": {
            "units_with_candidate_visual_support": units_with_candidate_visual_support,
            "units_with_candidate_link": units_with_candidate_link,
            "units_with_timestamp_fallback_link": units_with_timestamp_fallback_link,
            "candidate_links": candidate_links,
            "timestamp_fallback_links": timestamp_fallback_links,
            "candidate_link_signal_counts": candidate_signal_counts,
            "paper_claim_eligible": False,
        },
        "verified_object_alignment": {
            "units_with_verified_object_alignment": units_with_verified_object_alignment,
            "verified_links": verified_links,
            "verified_link_source_counts": verified_source_counts,
            "timestamp_fallback_counted_as_verified": False,
            "paper_claim_eligible_units": units_with_verified_object_alignment,
        },
        "candidate_link_signal_counts": candidate_signal_counts,
        "verified_link_source_counts": verified_source_counts,
        "public_note": LINK_DIAGNOSTICS_PUBLIC_NOTE,
    }


def _concept_field_coverage_summary(
    *,
    documents: list[dict[str, Any]],
    concept_graph_records: list[ConceptGraphRecord],
    source: str,
) -> dict[str, Any]:
    evidence_units_total = len(documents)
    units_with_concepts = 0
    units_with_relations = 0
    units_with_search_text = 0
    units_with_labels = 0
    units_with_aliases = 0
    concept_mentions = 0
    concept_label_mentions = 0
    concept_alias_mentions = 0
    relation_mentions = 0
    timestamp_only_relation_mentions = 0
    for document in documents:
        source_quality = _mapping(document.get("source_quality"))
        labels = _string_list(document.get("concept_labels"))
        aliases = _string_list(document.get("concept_aliases"))
        concept_count = int(
            source_quality.get("concept_count")
            or max(len(_string_list(document.get("concept_ids"))), len(labels))
        )
        relation_count = int(
            source_quality.get("concept_relation_count")
            or len(_list_of_dicts(document.get("concept_relations")))
        )
        timestamp_only_relation_count = int(
            source_quality.get("timestamp_only_concept_relation_count") or 0
        )
        concept_mentions += concept_count
        concept_label_mentions += len(labels)
        concept_alias_mentions += len(aliases)
        relation_mentions += relation_count
        timestamp_only_relation_mentions += timestamp_only_relation_count
        if concept_count:
            units_with_concepts += 1
        if labels:
            units_with_labels += 1
        if aliases:
            units_with_aliases += 1
        if relation_count:
            units_with_relations += 1
        if _has_concept_search_text(document):
            units_with_search_text += 1
    concept_nodes_total = sum(
        1 for record in concept_graph_records if isinstance(record, ConceptGraphConceptNode)
    )
    relation_edges_total = sum(
        1 for record in concept_graph_records if isinstance(record, ConceptGraphRelationEdge)
    )
    return {
        "schema_version": CONCEPT_FIELD_COVERAGE_SCHEMA_VERSION,
        "source": source,
        "concept_graph_loaded": bool(concept_graph_records),
        "concept_graph_records_total": len(concept_graph_records),
        "concept_nodes_total": concept_nodes_total,
        "concept_relation_edges_total": relation_edges_total,
        "evidence_units_total": evidence_units_total,
        "evidence_units_with_concepts": units_with_concepts,
        "evidence_units_with_concept_labels": units_with_labels,
        "evidence_units_with_concept_aliases": units_with_aliases,
        "evidence_units_with_concept_relations": units_with_relations,
        "evidence_units_with_concept_search_text": units_with_search_text,
        "unit_concept_coverage_ratio": _ratio_or_none(units_with_concepts, evidence_units_total),
        "unit_concept_relation_coverage_ratio": _ratio_or_none(
            units_with_relations,
            evidence_units_total,
        ),
        "concept_mentions": concept_mentions,
        "concept_label_mentions": concept_label_mentions,
        "concept_alias_mentions": concept_alias_mentions,
        "concept_relation_mentions": relation_mentions,
        "timestamp_only_concept_relation_mentions": timestamp_only_relation_mentions,
        "timestamp_only_counted_as_verified_object_alignment": False,
        "public_note": CONCEPT_FIELD_PUBLIC_NOTE,
    }


def _search_field_coverage_summary(documents: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(documents)
    keys = (
        "evidence_text",
        "semantic_text",
        "transcript_keywords",
        "concept_search_text",
        "visual_state_text",
        "visual_entity_text",
        "candidate_link_signal_summary",
        "verified_link_signal_summary",
    )
    field_counts = {key: 0 for key in keys}
    units_with_link_signal_search_text = 0
    for document in documents:
        coverage = _search_field_coverage(document)
        for key in keys:
            if coverage.get(key):
                field_counts[key] += 1
        if coverage.get("link_signal_search_text"):
            units_with_link_signal_search_text += 1
    return {
        "schema_version": SEARCH_FIELD_COVERAGE_SCHEMA_VERSION,
        "evidence_units_total": total,
        "field_unit_counts": field_counts,
        "field_unit_ratios": {
            key: _ratio_or_none(count, total) for key, count in field_counts.items()
        },
        "units_with_link_signal_search_text": units_with_link_signal_search_text,
        "unit_link_signal_search_text_ratio": _ratio_or_none(
            units_with_link_signal_search_text,
            total,
        ),
        "public_note": SEARCH_FIELD_PUBLIC_NOTE,
    }


def _search_field_coverage(document: dict[str, Any]) -> dict[str, bool]:
    candidate_link_text = _text(document.get("candidate_link_signal_summary"))
    verified_link_text = _text(document.get("verified_link_signal_summary"))
    return {
        "evidence_text": bool(_text(document.get("evidence_text"))),
        "semantic_text": bool(_text(document.get("semantic_text"))),
        "transcript_keywords": bool(_string_list(document.get("transcript_keywords"))),
        "concept_search_text": bool(
            _text(document.get("concept_search_text")) or _has_concept_search_text(document)
        ),
        "visual_state_text": bool(_text(document.get("visual_state_text"))),
        "visual_entity_text": bool(_text(document.get("visual_entity_text"))),
        "candidate_link_signal_summary": bool(candidate_link_text),
        "verified_link_signal_summary": bool(verified_link_text),
        "link_signal_search_text": bool(candidate_link_text or verified_link_text),
    }


def _add_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    for key in target:
        target[key] += int(source.get(key) or 0)


def _is_vlm_entity(entity: dict[str, Any]) -> bool:
    return is_paper_quality_vlm_entity(entity)


def _visual_state_context(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "visual_state_id": state.get("visual_state_id"),
            "representative_frame_id": state.get("representative_frame_id"),
            "frame_ids": state.get("frame_ids", []),
            "valid_start_time": state.get("valid_start_time"),
            "valid_end_time": state.get("valid_end_time"),
            "interval_source": state.get("interval_source"),
            "state_summary": state.get("state_summary", ""),
            "detected_text": state.get("detected_text", []),
        }.items()
        if value not in (None, "", [])
    }


def _visual_entity_context(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "entity_id": entity.get("entity_id"),
            "frame_id": entity.get("frame_id"),
            "timestamp": entity.get("timestamp"),
            "entity_type": entity.get("entity_type"),
            "text": entity.get("text"),
            "visual_description": entity.get("visual_description"),
            "position": entity.get("position"),
            "relations": entity.get("relations"),
            "detected_text": entity.get("detected_text"),
            "confidence": entity.get("confidence"),
            "source": entity.get("source"),
            "source_model": entity.get("source_model"),
        }.items()
        if value not in (None, "", [])
    }


def _visual_summary_text(
    *,
    window_states: list[dict[str, Any]],
    visual_entities: list[dict[str, Any]],
) -> str:
    return _compact_text(
        " ".join(
            _unique_text_values(
                [
                    _visual_state_summary_text(window_states),
                    _visual_entity_summary_text(visual_entities),
                ]
            )
        )
    )


def _visual_state_summary_text(window_states: list[dict[str, Any]]) -> str:
    values: list[Any] = []
    for state in window_states:
        values.extend([state.get("state_summary"), state.get("detected_text")])
    return _compact_text(" ".join(_unique_text_values(values)))


def _visual_entity_summary_text(visual_entities: list[dict[str, Any]]) -> str:
    values: list[Any] = []
    for entity in visual_entities:
        values.extend(
            [
                entity.get("text"),
                entity.get("visual_description"),
                entity.get("entity_type"),
                entity.get("detected_text"),
                entity.get("position"),
                entity.get("relations"),
            ]
        )
    return _compact_text(" ".join(_unique_text_values(values)))


def _link_signal_summary_text(value: Any, *, prefix: str) -> str:
    counts = _mapping(value)
    labels = [
        f"{prefix} {_humanize_snake(key)}"
        for key, count in sorted(counts.items())
        if int(count or 0) > 0
    ]
    return _compact_text(" ".join(labels))


def _concept_context_by_evidence_unit(
    records: list[ConceptGraphRecord],
) -> dict[str, dict[str, Any]]:
    if not records:
        return {}

    concepts_by_id = {
        record.concept_id: record
        for record in records
        if isinstance(record, ConceptGraphConceptNode)
    }
    context_by_unit: dict[str, dict[str, Any]] = {}
    seen_concepts: dict[str, set[str]] = {}
    seen_relations: dict[str, set[str]] = {}

    def context_for(evidence_unit_id: str) -> dict[str, Any]:
        context = context_by_unit.setdefault(evidence_unit_id, _empty_concept_context())
        seen_concepts.setdefault(evidence_unit_id, set())
        seen_relations.setdefault(evidence_unit_id, set())
        return context

    def add_concept(evidence_unit_id: str, concept: ConceptGraphConceptNode) -> None:
        if not evidence_unit_id:
            return
        context = context_for(evidence_unit_id)
        seen = seen_concepts[evidence_unit_id]
        if concept.concept_id not in seen:
            seen.add(concept.concept_id)
            context["concepts"].append(_concept_context(concept))
        context["concept_ids"] = _unique_text_values([*context["concept_ids"], concept.concept_id])
        context["concept_labels"] = _unique_text_values([*context["concept_labels"], concept.label])
        context["concept_aliases"] = _unique_text_values([*context["concept_aliases"], concept.aliases])

    for record in records:
        if not isinstance(record, ConceptGraphConceptNode):
            continue
        for evidence_unit_id in record.source_evidence_unit_ids:
            add_concept(evidence_unit_id, record)

    for record in records:
        if not isinstance(record, ConceptGraphRelationEdge):
            continue
        relation_context = _concept_relation_context(record, concepts_by_id)
        for evidence_unit_id in record.evidence_unit_ids:
            if not evidence_unit_id:
                continue
            context = context_for(evidence_unit_id)
            seen = seen_relations[evidence_unit_id]
            if record.edge_id not in seen:
                seen.add(record.edge_id)
                context["concept_relations"].append(relation_context)
            for concept_id in (record.source_concept_id, record.target_concept_id):
                concept = concepts_by_id.get(concept_id)
                if concept is None:
                    continue
                context["concept_ids"] = _unique_text_values([*context["concept_ids"], concept.concept_id])
                context["concept_labels"] = _unique_text_values([*context["concept_labels"], concept.label])
                context["concept_aliases"] = _unique_text_values([*context["concept_aliases"], concept.aliases])
            context["concept_relation_text"] = _compact_text(
                " ".join(
                    _unique_text_values(
                        [context.get("concept_relation_text"), relation_context.get("relation_text")]
                    )
                )
            )

    return context_by_unit


def _empty_concept_context() -> dict[str, Any]:
    return {
        "concept_ids": [],
        "concept_labels": [],
        "concept_aliases": [],
        "concept_relation_text": "",
        "concepts": [],
        "concept_relations": [],
    }


def _public_concept_search_context(
    *,
    base_context: dict[str, Any] | None,
    transcript_text: str,
    visual_state_text: str,
    visual_entity_text: str,
    domain_lexicon: DomainLexicon,
) -> dict[str, Any]:
    context = _clone_concept_context(base_context)
    search_basis = _compact_text(
        " ".join(
            _unique_text_values([transcript_text, visual_state_text, visual_entity_text])
        )
    )
    if not search_basis:
        return context

    lexicon_matches = domain_lexicon.matched_canonical_terms(search_basis)
    if lexicon_matches:
        lexicon_labels = list(lexicon_matches)
        lexicon_aliases: list[str] = []
        for canonical in lexicon_labels:
            lexicon_aliases.extend(domain_lexicon.aliases_by_canonical.get(canonical, ()))
            lexicon_aliases.extend(lexicon_matches.get(canonical, ()))
        context["concept_labels"] = _unique_text_values(
            [*context["concept_labels"], lexicon_labels]
        )
        context["concept_aliases"] = _unique_text_values(
            [*context["concept_aliases"], lexicon_aliases]
        )

    phrase_labels = (
        []
        if _string_list(context.get("concept_labels"))
        else _descriptive_concept_phrases(search_basis)
    )
    if phrase_labels:
        context["concept_labels"] = _unique_text_values(
            [*context["concept_labels"], phrase_labels]
        )
    return context


def _clone_concept_context(value: dict[str, Any] | None) -> dict[str, Any]:
    source = value if isinstance(value, dict) else _empty_concept_context()
    return {
        "concept_ids": _string_list(source.get("concept_ids")),
        "concept_labels": _string_list(source.get("concept_labels")),
        "concept_aliases": _string_list(source.get("concept_aliases")),
        "concept_relation_text": _text(source.get("concept_relation_text")),
        "concepts": _list_of_dicts(source.get("concepts")),
        "concept_relations": _list_of_dicts(source.get("concept_relations")),
    }


def _concept_context(concept: ConceptGraphConceptNode) -> dict[str, Any]:
    source_signals = _unique_text_values(
        source.source_signal for source in concept.evidence_sources
    )
    source_types = _unique_text_values(
        source.source_type for source in concept.evidence_sources
    )
    return {
        key: value
        for key, value in {
            "concept_id": concept.concept_id,
            "label": concept.label,
            "canonical_label": _concept_canonical_label(concept),
            "aliases": list(concept.aliases),
            "concept_type": concept.concept_type,
            "confidence": concept.confidence,
            "description": concept.description,
            "definition": _metadata_text(concept.metadata, ("definition", "defines", "meaning")),
            "example": _metadata_text(concept.metadata, ("example", "examples")),
            "formula": _metadata_text(concept.metadata, ("formula", "equation", "math")),
            "source_signals": source_signals,
            "source_types": source_types,
        }.items()
        if value not in (None, "", [])
    }


def _concept_relation_context(
    relation: ConceptGraphRelationEdge,
    concepts_by_id: dict[str, ConceptGraphConceptNode],
) -> dict[str, Any]:
    source = concepts_by_id.get(relation.source_concept_id)
    target = concepts_by_id.get(relation.target_concept_id)
    timestamp_only = _timestamp_only_concept_relation(relation)
    return {
        key: value
        for key, value in {
            "edge_id": relation.edge_id,
            "source_concept_id": relation.source_concept_id,
            "source_label": source.label if source is not None else relation.source_concept_id,
            "relation_type": relation.relation_type,
            "target_concept_id": relation.target_concept_id,
            "target_label": target.label if target is not None else relation.target_concept_id,
            "relation_status": relation.relation_status,
            "source_signals": list(relation.source_signals),
            "evidence_source_types": _unique_text_values(
                source.source_type for source in relation.evidence_sources
            ),
            "confidence": relation.confidence,
            "description": relation.description,
            "relation_text": _concept_relation_text(relation, concepts_by_id),
            "timestamp_only_candidate": timestamp_only,
            "timestamp_only_counted_as_verified_object_alignment": False,
        }.items()
        if value not in (None, "", [])
    }


def _concept_relation_text(
    relation: ConceptGraphRelationEdge,
    concepts_by_id: dict[str, ConceptGraphConceptNode],
) -> str:
    source = concepts_by_id.get(relation.source_concept_id)
    target = concepts_by_id.get(relation.target_concept_id)
    source_label = source.label if source is not None else relation.source_concept_id
    target_label = target.label if target is not None else relation.target_concept_id
    relation_label = _humanize_snake(relation.relation_type)
    return _compact_text(
        " ".join(
            _unique_text_values(
                [
                    f"{source_label} {relation_label} {target_label}",
                    relation.description,
                ]
            )
        )
    )


def _concept_summary_text(concept_context: dict[str, Any]) -> str:
    concepts = _list_of_dicts(concept_context.get("concepts"))
    relations = _list_of_dicts(concept_context.get("concept_relations"))
    return _compact_text(
        " ".join(
            _unique_text_values(
                [
                    concept_context.get("concept_labels"),
                    concept_context.get("concept_aliases"),
                    concept_context.get("concept_relation_text"),
                    [concept.get("canonical_label") for concept in concepts],
                    [concept.get("description") for concept in concepts],
                    [concept.get("definition") for concept in concepts],
                    [concept.get("example") for concept in concepts],
                    [concept.get("formula") for concept in concepts],
                    [concept.get("source_signals") for concept in concepts],
                    [relation.get("source_signals") for relation in relations],
                    [relation.get("evidence_source_types") for relation in relations],
                ]
            )
        )
    )


def _descriptive_concept_phrases(text: str, *, limit: int = 12) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]*", text.casefold())
        if len(token) >= 2 and token not in CONCEPT_PHRASE_STOPWORDS
    ]
    phrases: list[str] = []
    for width in (4, 3, 2):
        for start in range(0, max(0, len(tokens) - width + 1)):
            phrase_tokens = tokens[start : start + width]
            if not _concept_phrase_tokens(phrase_tokens):
                continue
            phrases.append(" ".join(phrase_tokens))
    ranked = sorted(
        _unique_text_values(phrases),
        key=lambda phrase: (-len(phrase.split()), phrase),
    )
    return ranked[: max(0, int(limit))]


def _concept_phrase_tokens(tokens: list[str]) -> bool:
    if len(tokens) < 2:
        return False
    if any(token in CONCEPT_PHRASE_STOPWORDS for token in tokens):
        return False
    return any(token in CONCEPT_PHRASE_SIGNAL_TERMS for token in tokens)


def _concept_canonical_label(concept: ConceptGraphConceptNode) -> str:
    metadata = concept.metadata
    return (
        _metadata_text(metadata, ("canonical_label", "canonical", "canonical_name"))
        or concept.label
    )


def _metadata_text(metadata: dict[str, Any], keys: tuple[str, ...]) -> str:
    if not metadata:
        return ""
    values: list[Any] = []
    for key in keys:
        if key in metadata:
            values.append(metadata.get(key))
    return _compact_text(" ".join(_unique_text_values(values)))


def _transcript_keywords(text: str, *, limit: int = 32) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", text.casefold())
        if token not in TRANSCRIPT_KEYWORD_STOPWORDS
    ]
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for index, token in enumerate(tokens):
        counts[token] = counts.get(token, 0) + 1
        first_seen.setdefault(token, index)
    ranked = sorted(counts, key=lambda token: (-counts[token], first_seen[token], token))
    return ranked[: max(0, int(limit))]


def _timestamp_only_concept_relation(relation: ConceptGraphRelationEdge) -> bool:
    source_signals = {_text(signal).casefold() for signal in relation.source_signals if _text(signal)}
    return bool(source_signals) and source_signals <= TIMESTAMP_ONLY_SOURCE_SIGNALS


def _humanize_snake(value: str) -> str:
    return _compact_text(value.replace("_", " "))


def _transcript_window_text(segments: list[dict[str, Any]]) -> str:
    return _compact_text(" ".join(_unique_text_values(segment.get("transcript_text") for segment in segments)))


def _window_bounds(segments: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    starts = [_optional_float(segment.get("start_time")) for segment in segments]
    ends = [_optional_float(segment.get("end_time")) for segment in segments]
    bounds = [value for value in [*starts, *ends] if value is not None]
    if not bounds:
        centers = [_optional_float(segment.get("timestamp_center")) for segment in segments]
        bounds = [value for value in centers if value is not None]
    if not bounds:
        return None, None
    return min(bounds), max(bounds)


def _overlaps(
    left_start: float | None,
    left_end: float | None,
    right_start: float | None,
    right_end: float | None,
) -> bool:
    if left_start is None or left_end is None or right_start is None or right_end is None:
        return False
    return min(left_start, left_end) <= max(right_start, right_end) and min(right_start, right_end) <= max(left_start, left_end)


def _window_config(
    *,
    window_seconds: float | None,
    neighbor_count: int,
    previous_neighbor_count: int | None,
    next_neighbor_count: int | None,
    window_before_seconds: float | None,
    window_after_seconds: float | None,
) -> dict[str, Any]:
    return resolve_window_config(
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
    )


def _status_counts(documents: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for document in documents:
        status = _text(document.get("alignment_status")) or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


def _optional_existing_project_path(
    *,
    project_dir: Path,
    path: Path | None,
    default: Path,
    artifact_name: str,
) -> Path | None:
    candidate = default if path is None else _resolve_project_output_path(project_dir=project_dir, path=path, default=default)
    if candidate.exists():
        return candidate
    if path is not None:
        raise FileNotFoundError(f"{artifact_name} artifact not found: {candidate}")
    return None


def _resolve_project_output_path(*, project_dir: Path, path: Path | None, default: Path) -> Path:
    if path is None:
        return default
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_dir / candidate).resolve()


def _update_project_manifest(
    *,
    manifest_path: Path,
    evidence_units_path: Path,
    summary: dict[str, Any],
) -> None:
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {}

    artifacts = payload.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        payload["artifacts"] = artifacts
    artifacts["evidence_units"] = str(evidence_units_path)
    visual_states_path = _text(_mapping(summary.get("paths")).get("visual_states"))
    if visual_states_path:
        artifacts["visual_states"] = visual_states_path

    counts = payload.setdefault("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        payload["counts"] = counts
    counts["evidence_units"] = int(summary["counts"]["evidence_units_total"])
    visual_states_total = _mapping(summary.get("visual_state_coverage")).get("visual_states_total")
    if visual_states_total is not None:
        counts["visual_states"] = int(visual_states_total)

    payload["evidence_unit_storage"] = {
        "window_config": summary["window_config"],
        "visual_state_config": summary["visual_state_config"],
        "visual_state_coverage": summary["visual_state_coverage"],
        "counts": summary["counts"],
        "alignment_status_counts": summary["alignment_status_counts"],
        "link_diagnostics": summary["link_diagnostics"],
        "concept_field_coverage": summary["concept_field_coverage"],
        "search_field_coverage": summary["search_field_coverage"],
    }
    write_json(manifest_path, payload)


def _update_visual_state_manifest(
    *,
    manifest_path: Path,
    visual_states_path: Path,
    summary: dict[str, Any],
) -> None:
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {}

    artifacts = payload.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        payload["artifacts"] = artifacts
    artifacts["visual_states"] = str(visual_states_path)

    counts = payload.setdefault("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        payload["counts"] = counts
    counts["visual_states"] = int(_mapping(summary.get("counts")).get("visual_states_total") or 0)

    payload["visual_state_storage"] = {
        "schema_version": VISUAL_STATES_SCHEMA_VERSION,
        "source": summary.get("source"),
        "state_padding_seconds": summary.get("state_padding_seconds"),
        "counts": summary.get("counts", {}),
        "interval_duration_seconds": summary.get("interval_duration_seconds", {}),
        "coverage_gate": summary.get("coverage_gate", {}),
        "public_note": VISUAL_STATE_PUBLIC_NOTE,
    }
    write_json(manifest_path, payload)


def _project_defaults_from_manifest(manifest_path: Path) -> dict[str, str | None]:
    if not manifest_path.exists():
        return {"project_id": None, "video_id": None}
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = loaded if isinstance(loaded, dict) else {}
    return {
        "project_id": _text(payload.get("project_id")) or None,
        "video_id": _text(payload.get("video_id")) or None,
    }


def _group_by_text(rows: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = _text(row.get(key))
        if not value:
            continue
        grouped.setdefault(value, []).append(row)
    return grouped


def _first_text(rows: Iterable[dict[str, Any]], key: str) -> str | None:
    for row in rows:
        value = _text(row.get(key))
        if value:
            return value
    return None


def _unique_text_values(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        for text in _text_values([value]):
            compacted = _compact_text(text)
            if compacted and compacted not in seen:
                seen.add(compacted)
                unique.append(compacted)
    return unique


def _text_values(values: Iterable[Any]) -> list[str]:
    text_values: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            text_values.extend(_text_values(value.values()))
        elif isinstance(value, (list, tuple)):
            text_values.extend(_text_values(value))
        else:
            text = _text(value)
            if text:
                text_values.append(text)
    return text_values


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [_text(item) for item in value if _text(item)]


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _compact_text(value: str) -> str:
    return " ".join(value.split())


def _has_concept_search_text(document: dict[str, Any]) -> bool:
    return bool(
        _text(document.get("concept_search_text"))
        or
        _string_list(document.get("concept_labels"))
        or _string_list(document.get("concept_aliases"))
        or _text(document.get("concept_relation_text"))
    )


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nonnegative_float(value: Any, *, default: float) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        return default
    return max(0.0, parsed)
