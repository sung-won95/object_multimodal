from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from oarag.core.config import Neo4jConfig, load_neo4j_config
from oarag.graph.graph_ingest import classify_neo4j_error, create_neo4j_driver
from oarag.integrations.meili import MeiliClient
from oarag.retrieval.evidence_unit_index import (
    _evidence_unit_candidate,
    _escape_meili_filter_string,
    _project_id_from_evidence_units,
)
from oarag.retrieval.graph_rerank import (
    graph_aware_rerank_candidates,
    graph_aware_rerank_context,
    graph_aware_rerank_diagnostics,
)
from oarag.retrieval.project_index import evidence_unit_artifact_path, iter_jsonl_documents
from oarag.retrieval.project_query import _search_hit_records, _search_with_expanded_queries


MEILI_RAW_SOURCE = "meili_raw"
MEILI_EXPANDED_SOURCE = "meili_expanded"
GRAPH_TRAVERSAL_SOURCE = "graph_traversal"
DUAL_CANDIDATE_SCHEMA_VERSION = "oarag-dual-candidates-v1"

_SOURCE_PRIORITY = {
    MEILI_RAW_SOURCE: 0,
    GRAPH_TRAVERSAL_SOURCE: 1,
    MEILI_EXPANDED_SOURCE: 2,
}
_STOPWORDS = {
    "about",
    "after",
    "before",
    "does",
    "during",
    "explain",
    "find",
    "from",
    "give",
    "how",
    "is",
    "into",
    "lecture",
    "lecturer",
    "part",
    "show",
    "shows",
    "slide",
    "tell",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}
_VISUAL_INTENT_TERMS = {
    "arrow",
    "chart",
    "diagram",
    "figure",
    "graph",
    "matrix",
    "plot",
    "slide",
    "visual",
}
_RELATION_INTENT_TERMS = {
    "alias",
    "compare",
    "contrast",
    "define",
    "depends",
    "explains",
    "prerequisite",
    "related",
    "uses",
}


class GraphCandidateSession(Protocol):
    def run(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        ...


@dataclass(frozen=True)
class QueryConceptCandidate:
    text: str
    source: str
    rank: int
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "rank": self.rank,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class QueryAnalysis:
    query_type: str
    concept_candidates: tuple[QueryConceptCandidate, ...]
    intent_candidates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "concept_candidates": [candidate.to_dict() for candidate in self.concept_candidates],
            "intent_candidates": list(self.intent_candidates),
        }


@dataclass(frozen=True)
class DualCandidateConfig:
    candidate_pool_limit: int
    graph_limit: int
    max_expanded_queries: int = 6
    enable_graph: bool = True


@dataclass
class _CandidateEntry:
    key: str
    candidate: dict[str, Any]
    sources: list[dict[str, Any]] = field(default_factory=list)


def query_project_dual_candidates(
    *,
    client: MeiliClient,
    index_uid: str,
    project_dir: Path,
    query: str,
    limit: int = 5,
    candidate_pool_limit: int | None = None,
    graph_limit: int | None = None,
    evidence_units: Path | None = None,
    target_evidence_unit_ids: list[str] | None = None,
    target_segment_ids: list[str] | None = None,
    enable_graph: bool = True,
    graph_aware_rerank: bool = False,
    graph_session: GraphCandidateSession | None = None,
    neo4j_config: Neo4jConfig | None = None,
) -> dict[str, Any]:
    """Generate a public-safe union of Meili raw/expanded and graph candidates."""

    result_limit = _positive_int(limit, field_name="limit")
    pool_limit = _positive_int(
        candidate_pool_limit if candidate_pool_limit is not None else max(result_limit, 10),
        field_name="candidate_pool_limit",
    )
    resolved_graph_limit = _positive_int(
        graph_limit if graph_limit is not None else pool_limit,
        field_name="graph_limit",
    )
    resolved_project_dir = project_dir.expanduser().resolve()
    project_id = _project_id_from_evidence_units(
        project_dir=resolved_project_dir,
        evidence_units=evidence_units,
    )
    filter_expr = f'project_id = "{_escape_meili_filter_string(project_id)}"'
    analysis = analyze_dual_candidate_query(query)
    config = DualCandidateConfig(
        candidate_pool_limit=pool_limit,
        graph_limit=resolved_graph_limit,
        enable_graph=enable_graph,
    )
    evidence_lookup = _load_evidence_unit_lookup(
        project_dir=resolved_project_dir,
        evidence_units=evidence_units,
    )

    started_at = time.monotonic()
    raw_entries, raw_context = _meili_candidate_entries(
        client=client,
        index_uid=index_uid,
        queries=[query],
        query_role="raw",
        source_type=MEILI_RAW_SOURCE,
        limit=pool_limit,
        filter=filter_expr,
    )
    expanded_queries = _expanded_queries(query=query, analysis=analysis, max_queries=config.max_expanded_queries)
    expanded_entries, expanded_context = _meili_candidate_entries(
        client=client,
        index_uid=index_uid,
        queries=expanded_queries,
        query_role="concept_expanded",
        source_type=MEILI_EXPANDED_SOURCE,
        limit=pool_limit,
        filter=filter_expr,
    )
    graph_entries, graph_status = _graph_candidate_entries(
        query=query,
        project_id=project_id,
        analysis=analysis,
        config=config,
        evidence_lookup=evidence_lookup,
        graph_session=graph_session,
        neo4j_config=neo4j_config,
    )

    merged_entries = _merge_candidate_entries([*raw_entries, *expanded_entries, *graph_entries])
    ranked_entries = sorted(merged_entries.values(), key=_candidate_entry_sort_key)
    base_candidates = [
        _serialize_candidate_entry(entry, rank) for rank, entry in enumerate(ranked_entries, 1)
    ]
    rerank_context = {"enabled": False, "strategy": None}
    if graph_aware_rerank:
        all_candidates = graph_aware_rerank_candidates(
            base_candidates,
            query=query,
            query_analysis=analysis.to_dict(),
        )
        rerank_context = graph_aware_rerank_context(
            base_candidates=base_candidates,
            reranked_candidates=all_candidates,
            query_analysis=analysis.to_dict(),
        )
    else:
        all_candidates = base_candidates
    candidates = all_candidates[:result_limit]
    diagnostics = dual_candidate_diagnostics(
        candidates=all_candidates,
        returned_candidates=candidates,
        target_evidence_unit_ids=target_evidence_unit_ids,
        target_segment_ids=target_segment_ids,
    )
    if graph_aware_rerank:
        diagnostics["graph_aware_rerank"] = graph_aware_rerank_diagnostics(
            base_candidates=base_candidates,
            reranked_candidates=all_candidates,
            returned_candidates=candidates,
            query_analysis=analysis.to_dict(),
        )
    elapsed_ms = round((time.monotonic() - started_at) * 1000, 2)

    return {
        "schema_version": DUAL_CANDIDATE_SCHEMA_VERSION,
        "query": query,
        "index": index_uid,
        "project_id": project_id,
        "limit": result_limit,
        "candidate_pool_limit": pool_limit,
        "processing_time_ms": elapsed_ms,
        "query_analysis": analysis.to_dict(),
        "candidates": candidates,
        "retrieval_context": {
            "candidate_generation": {
                "mode": "meili_graph_dual",
                "sources": [MEILI_RAW_SOURCE, MEILI_EXPANDED_SOURCE, GRAPH_TRAVERSAL_SOURCE],
                "candidate_pool_limit": pool_limit,
                "graph_limit": resolved_graph_limit,
                "expanded_queries": expanded_queries,
            },
            "meilisearch": {
                "raw": raw_context,
                "expanded": expanded_context,
            },
            "graph": graph_status,
            "graph_aware_rerank": rerank_context,
        },
        "diagnostics": diagnostics,
        "counts": {
            "meili_raw_candidates": len(raw_entries),
            "meili_expanded_candidates": len(expanded_entries),
            "graph_candidates": len(graph_entries),
            "union_candidates": len(ranked_entries),
            "returned_candidates": len(candidates),
        },
    }


def analyze_dual_candidate_query(query: str) -> QueryAnalysis:
    normalized_tokens = _query_tokens(query)
    intent_candidates: list[str] = []
    if any(token in _VISUAL_INTENT_TERMS for token in normalized_tokens):
        intent_candidates.append("visual")
    if any(token in _RELATION_INTENT_TERMS for token in normalized_tokens):
        intent_candidates.append("relation")
    if any(token in {"define", "definition", "explain"} for token in normalized_tokens):
        intent_candidates.append("definition")
    if not intent_candidates:
        intent_candidates.append("concept")

    concept_texts: list[tuple[str, str, int]] = []
    compact_tokens = [token for token in normalized_tokens if token not in _STOPWORDS]
    if 1 < len(compact_tokens) <= 6:
        concept_texts.append((" ".join(compact_tokens), "query_phrase", len(compact_tokens)))
    for ngram_size in (3, 2, 1):
        for index in range(0, max(0, len(compact_tokens) - ngram_size + 1)):
            tokens = compact_tokens[index : index + ngram_size]
            if not tokens:
                continue
            text = " ".join(tokens)
            if len(text) < 3:
                continue
            concept_texts.append((text, f"{ngram_size}gram", ngram_size))

    seen: set[str] = set()
    candidates: list[QueryConceptCandidate] = []
    for text, source, token_count in concept_texts:
        normalized = _normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(
            QueryConceptCandidate(
                text=text,
                source=source,
                rank=len(candidates) + 1,
                token_count=token_count,
            )
        )
        if len(candidates) >= 12:
            break

    query_type = "unknown"
    if "visual" in intent_candidates and "relation" in intent_candidates:
        query_type = "visual_relation"
    elif "visual" in intent_candidates:
        query_type = "visual"
    elif "relation" in intent_candidates:
        query_type = "relation"
    elif candidates:
        query_type = "concept"

    return QueryAnalysis(
        query_type=query_type,
        concept_candidates=tuple(candidates),
        intent_candidates=tuple(intent_candidates),
    )


def graph_candidate_cypher() -> str:
    return """
MATCH (matched:GraphNode:Concept)
WHERE ($project_id = '' OR matched.project_id = $project_id OR matched.project_id IS NULL)
  AND any(term IN $concept_terms WHERE
    toLower(coalesce(matched.label, '')) CONTAINS term
    OR toLower(coalesce(matched.concept_id, '')) CONTAINS term
    OR any(alias IN coalesce(matched.aliases, []) WHERE toLower(toString(alias)) CONTAINS term)
  )
CALL {
  WITH matched
  MATCH (evidence:GraphNode:EvidenceUnit)-[mention:MENTIONS]->(matched)
  RETURN evidence.evidence_unit_id AS evidence_unit_id,
         evidence.project_id AS evidence_project_id,
         evidence.video_id AS video_id,
         evidence.target_segment_id AS target_segment_id,
         evidence.source_segment_ids AS source_segment_ids,
         evidence.start_time AS start_time,
         evidence.end_time AS end_time,
         matched AS matched_concept,
         null AS related_concept,
         [evidence.key, matched.key] AS graph_path,
         ['MENTIONS'] AS relationships,
         coalesce(mention.confidence, matched.confidence, 0.9) AS score,
         'direct_mention' AS graph_match_type
  UNION
  WITH matched
  UNWIND coalesce(matched.source_evidence_unit_ids, []) AS source_evidence_unit_id
  RETURN source_evidence_unit_id AS evidence_unit_id,
         matched.project_id AS evidence_project_id,
         null AS video_id,
         null AS target_segment_id,
         [] AS source_segment_ids,
         null AS start_time,
         null AS end_time,
         matched AS matched_concept,
         null AS related_concept,
         [matched.key] AS graph_path,
         ['CONCEPT_SOURCE_EVIDENCE'] AS relationships,
         coalesce(matched.confidence, 0.82) AS score,
         'concept_source_evidence' AS graph_match_type
  UNION
  WITH matched
  MATCH path=(matched)-[relation]-(related:GraphNode:Concept)
  WHERE ($project_id = '' OR related.project_id = $project_id OR related.project_id IS NULL)
  UNWIND coalesce(related.source_evidence_unit_ids, []) AS related_evidence_unit_id
  RETURN related_evidence_unit_id AS evidence_unit_id,
         related.project_id AS evidence_project_id,
         null AS video_id,
         null AS target_segment_id,
         [] AS source_segment_ids,
         null AS start_time,
         null AS end_time,
         matched AS matched_concept,
         related AS related_concept,
         [node IN nodes(path) | node.key] AS graph_path,
         [type(relation), 'RELATED_SOURCE_EVIDENCE'] AS relationships,
         coalesce(relation.confidence, related.confidence, 0.68) AS score,
         'related_concept_evidence' AS graph_match_type
  UNION
  WITH matched
  MATCH path=(matched)-[:INSTANCE_OF_GLOBAL_CONCEPT]->(global:GraphNode:GlobalConcept)
    -[relation]-(related_global:GraphNode:GlobalConcept)
    <-[:INSTANCE_OF_GLOBAL_CONCEPT]-(related:GraphNode:Concept)
  WHERE ($project_id = '' OR related.project_id = $project_id OR related.project_id IS NULL)
  UNWIND coalesce(related.source_evidence_unit_ids, []) AS related_evidence_unit_id
  RETURN related_evidence_unit_id AS evidence_unit_id,
         related.project_id AS evidence_project_id,
         null AS video_id,
         null AS target_segment_id,
         [] AS source_segment_ids,
         null AS start_time,
         null AS end_time,
         matched AS matched_concept,
         related AS related_concept,
         [node IN nodes(path) | node.key] AS graph_path,
         [
           'INSTANCE_OF_GLOBAL_CONCEPT',
           type(relation),
           'INSTANCE_OF_GLOBAL_CONCEPT',
           'RELATED_SOURCE_EVIDENCE'
         ] AS relationships,
         coalesce(relation.confidence, related.confidence, 0.62) AS score,
         'global_concept_related_evidence' AS graph_match_type
}
WITH evidence_unit_id, evidence_project_id, video_id, target_segment_id, source_segment_ids,
     start_time, end_time, matched_concept, related_concept, graph_path, relationships,
     score, graph_match_type
WHERE evidence_unit_id IS NOT NULL AND evidence_unit_id <> ''
RETURN evidence_unit_id,
       evidence_project_id AS project_id,
       video_id,
       target_segment_id,
       source_segment_ids,
       start_time,
       end_time,
       properties(matched_concept) AS matched_concept,
       CASE WHEN related_concept IS NULL THEN null ELSE properties(related_concept) END
         AS related_concept,
       graph_path,
       relationships,
       graph_match_type,
       score
ORDER BY score DESC, evidence_unit_id ASC
LIMIT $limit
"""


def serialize_graph_candidate_record(record: Any, *, rank: int) -> dict[str, Any]:
    row = _record_data(record)
    evidence_unit = _node_payload(row.get("evidence_unit")) or {}
    matched_concept = _public_concept_payload(row.get("matched_concept") or row.get("concept"))
    related_concept = _public_concept_payload(row.get("related_concept"))
    source_evidence_ref = _public_source_evidence_ref(row.get("source_evidence_ref"))
    evidence_unit_id = str(
        row.get("evidence_unit_id")
        or evidence_unit.get("evidence_unit_id")
        or source_evidence_ref.get("evidence_unit_id")
        or ""
    ).strip()
    graph_path = _string_list(row.get("graph_path") or row.get("path"))
    relationships = _string_list(row.get("relationships") or row.get("relationship_types"))
    score = _optional_float(row.get("score"))
    candidate = _compact(
        {
            "evidence_unit_id": evidence_unit_id,
            "project_id": row.get("project_id") or evidence_unit.get("project_id"),
            "video_id": row.get("video_id") or evidence_unit.get("video_id"),
            "target_segment_id": row.get("target_segment_id") or evidence_unit.get("target_segment_id"),
            "source_segment_ids": _string_list(
                row.get("source_segment_ids") or evidence_unit.get("source_segment_ids")
            ),
            "start_time": _optional_float(
                _first_present(row.get("start_time"), evidence_unit.get("start_time"))
            ),
            "end_time": _optional_float(
                _first_present(row.get("end_time"), evidence_unit.get("end_time"))
            ),
            "score": score,
            "rank": rank,
        }
    )
    source = _compact(
        {
            "source_type": GRAPH_TRAVERSAL_SOURCE,
            "rank": rank,
            "score": score,
            "graph_match_type": row.get("graph_match_type") or row.get("type") or "graph_path",
            "graph_path": graph_path,
            "relationships": relationships,
            "path_length": len(graph_path),
            "matched_concept": matched_concept,
            "related_concept": related_concept,
            "source_evidence_ref": source_evidence_ref,
            "evidence_unit_id": evidence_unit_id,
            "lecture_id": row.get("lecture_id")
            or matched_concept.get("lecture_id")
            or related_concept.get("lecture_id"),
        }
    )
    return {"candidate": candidate, "source": source}


def dual_candidate_diagnostics(
    *,
    candidates: list[dict[str, Any]],
    returned_candidates: list[dict[str, Any]],
    target_evidence_unit_ids: list[str] | None = None,
    target_segment_ids: list[str] | None = None,
) -> dict[str, Any]:
    source_counts: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        for source in _candidate_sources(candidate):
            source_type = str(source.get("source_type") or "")
            if not source_type:
                continue
            counts = source_counts.setdefault(
                source_type,
                {"source_hits": 0, "unique_candidates": 0, "returned_candidates": 0},
            )
            counts["source_hits"] += 1
    for source_type, candidate_keys in _candidate_keys_by_source(candidates).items():
        source_counts.setdefault(
            source_type,
            {"source_hits": 0, "unique_candidates": 0, "returned_candidates": 0},
        )["unique_candidates"] = len(candidate_keys)
    for source_type, candidate_keys in _candidate_keys_by_source(returned_candidates).items():
        source_counts.setdefault(
            source_type,
            {"source_hits": 0, "unique_candidates": 0, "returned_candidates": 0},
        )["returned_candidates"] = len(candidate_keys)

    targets = {
        "evidence_unit_ids": sorted({str(item) for item in target_evidence_unit_ids or [] if item}),
        "target_segment_ids": sorted({str(item) for item in target_segment_ids or [] if item}),
    }
    return {
        "public_safe": True,
        "schema_version": DUAL_CANDIDATE_SCHEMA_VERSION,
        "source_counts": dict(sorted(source_counts.items())),
        "source_recall": _source_recall(candidates=candidates, targets=targets),
        "returned_source_recall": _source_recall(candidates=returned_candidates, targets=targets),
        "candidate_pool": {
            "generated": len(candidates),
            "returned": len(returned_candidates),
        },
        "omits": ["raw_transcript", "local_paths", "frame_paths", "secrets"],
    }


def _meili_candidate_entries(
    *,
    client: MeiliClient,
    index_uid: str,
    queries: list[str],
    query_role: str,
    source_type: str,
    limit: int,
    filter: str,
) -> tuple[list[_CandidateEntry], dict[str, Any]]:
    if not queries:
        return [], {"attempted": False, "hit_count": 0, "queries": []}
    response = _search_with_expanded_queries(
        client=client,
        index_uid=index_uid,
        queries=queries,
        limit=limit,
        filter=filter,
    )
    entries: list[_CandidateEntry] = []
    for aggregate_rank, record in enumerate(_search_hit_records(response), start=1):
        candidate = _evidence_unit_candidate(rank=aggregate_rank, hit=record["hit"])
        key = _candidate_key(candidate)
        if not key:
            continue
        source = _meili_source_summary(
            source_type=source_type,
            index_uid=index_uid,
            query_role=query_role,
            aggregate_rank=aggregate_rank,
            record=record,
        )
        entries.append(_CandidateEntry(key=key, candidate=candidate, sources=[source]))
    return entries, _meili_context(response=response, queries=queries, source_type=source_type)


def _graph_candidate_entries(
    *,
    query: str,
    project_id: str,
    analysis: QueryAnalysis,
    config: DualCandidateConfig,
    evidence_lookup: dict[str, dict[str, Any]],
    graph_session: GraphCandidateSession | None,
    neo4j_config: Neo4jConfig | None,
) -> tuple[list[_CandidateEntry], dict[str, Any]]:
    if not config.enable_graph:
        return [], _graph_status(
            attempted=False,
            available=False,
            status="skipped",
            skip_reason="graph_disabled",
        )
    concept_terms = _graph_concept_terms(query=query, analysis=analysis)
    if not concept_terms:
        return [], _graph_status(
            attempted=False,
            available=False,
            status="skipped",
            skip_reason="no_concept_candidates",
        )

    started_at = time.monotonic()
    try:
        if graph_session is not None:
            rows = graph_session.run(
                graph_candidate_cypher(),
                {"project_id": project_id, "concept_terms": concept_terms, "limit": config.graph_limit},
            )
        else:
            active_config = neo4j_config or load_neo4j_config()
            driver = create_neo4j_driver(active_config)
            try:
                driver.verify_connectivity()
                with driver.session(database=active_config.database) as session:
                    rows = list(
                        session.run(
                            graph_candidate_cypher(),
                            {
                                "project_id": project_id,
                                "concept_terms": concept_terms,
                                "limit": config.graph_limit,
                            },
                        )
                    )
            finally:
                driver.close()
        entries = _entries_from_graph_rows(rows, evidence_lookup=evidence_lookup)
        elapsed_ms = round((time.monotonic() - started_at) * 1000, 2)
        return entries, _graph_status(
            attempted=True,
            available=True,
            status="hit" if entries else "miss",
            hit_count=len(entries),
            concept_terms=concept_terms,
            elapsed_ms=elapsed_ms,
        )
    except Exception as exc:
        classified = classify_neo4j_error(exc)
        elapsed_ms = round((time.monotonic() - started_at) * 1000, 2)
        return [], _graph_status(
            attempted=True,
            available=False,
            status="skipped_unavailable",
            skip_reason="graph_db_unavailable",
            message=str(classified),
            concept_terms=concept_terms,
            elapsed_ms=elapsed_ms,
        )


def _entries_from_graph_rows(
    rows: Any,
    *,
    evidence_lookup: dict[str, dict[str, Any]],
) -> list[_CandidateEntry]:
    entries: list[_CandidateEntry] = []
    seen_sources: set[tuple[str, str, tuple[str, ...]]] = set()
    for rank, record in enumerate(rows or [], start=1):
        serialized = serialize_graph_candidate_record(record, rank=rank)
        candidate = serialized["candidate"]
        source = serialized["source"]
        evidence_unit_id = str(candidate.get("evidence_unit_id") or "")
        if not evidence_unit_id:
            continue
        if evidence_unit_id in evidence_lookup:
            enriched = _evidence_unit_candidate(rank=rank, hit=evidence_lookup[evidence_unit_id])
            enriched["score"] = candidate.get("score")
            candidate = _merge_missing_values(enriched, candidate)
        key = _candidate_key(candidate)
        if not key:
            continue
        identity = (
            key,
            str(source.get("graph_match_type") or ""),
            tuple(_string_list(source.get("graph_path"))),
        )
        if identity in seen_sources:
            continue
        seen_sources.add(identity)
        entries.append(_CandidateEntry(key=key, candidate=candidate, sources=[source]))
    return entries


def _merge_candidate_entries(entries: list[_CandidateEntry]) -> dict[str, _CandidateEntry]:
    merged: dict[str, _CandidateEntry] = {}
    for entry in entries:
        existing = merged.get(entry.key)
        if existing is None:
            merged[entry.key] = _CandidateEntry(
                key=entry.key,
                candidate=dict(entry.candidate),
                sources=[dict(source) for source in entry.sources],
            )
            continue
        existing.candidate = _merge_missing_values(existing.candidate, entry.candidate)
        for source in entry.sources:
            if _source_identity(source) not in {_source_identity(item) for item in existing.sources}:
                existing.sources.append(dict(source))
    return merged


def _serialize_candidate_entry(entry: _CandidateEntry, rank: int) -> dict[str, Any]:
    candidate = dict(entry.candidate)
    candidate["rank"] = rank
    candidate["candidate_key"] = entry.key
    candidate["candidate_sources"] = sorted(entry.sources, key=_candidate_source_sort_key)
    candidate["candidate_source_types"] = [
        str(source.get("source_type") or "") for source in candidate["candidate_sources"]
    ]
    candidate["candidate_source_count"] = len(candidate["candidate_sources"])
    candidate["selected_source"] = candidate["candidate_source_types"][0]
    return candidate


def _expanded_queries(*, query: str, analysis: QueryAnalysis, max_queries: int) -> list[str]:
    query_norm = _normalize_text(query)
    queries: list[str] = []
    seen: set[str] = {query_norm}
    for candidate in analysis.concept_candidates:
        normalized = _normalize_text(candidate.text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        queries.append(candidate.text)
        if len(queries) >= max_queries:
            break
    return queries


def _graph_concept_terms(*, query: str, analysis: QueryAnalysis) -> list[str]:
    terms: list[str] = []
    for candidate in analysis.concept_candidates:
        normalized = _normalize_text(candidate.text)
        if normalized:
            terms.append(normalized)
    query_normalized = _normalize_text(query)
    if query_normalized and len(query_normalized) <= 80:
        terms.append(query_normalized)
    return _unique_strings(terms)[:12]


def _meili_source_summary(
    *,
    source_type: str,
    index_uid: str,
    query_role: str,
    aggregate_rank: int,
    record: dict[str, Any],
) -> dict[str, Any]:
    return _compact(
        {
            "source_type": source_type,
            "rank": aggregate_rank,
            "score": _optional_float(record.get("score")),
            "index": index_uid,
            "query_role": query_role,
            "query": record.get("query"),
            "query_index": record.get("query_index"),
            "retrieval_mode": record.get("retrieval_mode"),
            "original_rank": record.get("rank"),
            "original_score": _optional_float(record.get("score")),
            "fusion_score": _optional_float(record.get("fusion_score")),
            "match_count": len(record.get("matches") or []),
        }
    )


def _meili_context(
    *,
    response: dict[str, Any],
    queries: list[str],
    source_type: str,
) -> dict[str, Any]:
    return {
        "attempted": True,
        "source_type": source_type,
        "queries": queries,
        "hit_count": len(response.get("hits") or []),
        "hit_count_before_limit": response.get("hitCountBeforeLimit"),
        "raw_hit_count_before_dedupe": response.get("rawHitCountBeforeDedupe"),
        "processing_time_ms": response.get("processingTimeMs"),
    }


def _graph_status(
    *,
    attempted: bool,
    available: bool,
    status: str,
    skip_reason: str | None = None,
    message: str | None = None,
    hit_count: int = 0,
    concept_terms: list[str] | None = None,
    elapsed_ms: float | None = None,
) -> dict[str, Any]:
    return _compact(
        {
            "backend": "neo4j",
            "attempted": attempted,
            "available": available,
            "status": status,
            "skip_reason": skip_reason,
            "message": message,
            "hit_count": hit_count,
            "concept_terms": concept_terms or [],
            "elapsed_ms": elapsed_ms,
        }
    )


def _load_evidence_unit_lookup(
    *,
    project_dir: Path,
    evidence_units: Path | None,
) -> dict[str, dict[str, Any]]:
    try:
        artifact_path = evidence_unit_artifact_path(project_dir, evidence_units=evidence_units)
    except FileNotFoundError:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for document in iter_jsonl_documents(artifact_path):
        evidence_unit_id = str(document.get("evidence_unit_id") or "").strip()
        if evidence_unit_id:
            lookup[evidence_unit_id] = document
    return lookup


def _candidate_entry_sort_key(entry: _CandidateEntry) -> tuple[int, int, float, str]:
    best_source = sorted(entry.sources, key=_candidate_source_sort_key)[0] if entry.sources else {}
    return (
        _SOURCE_PRIORITY.get(str(best_source.get("source_type") or ""), 99),
        _optional_int(best_source.get("rank")) or 10**9,
        -float(_optional_float(best_source.get("score")) or 0.0),
        entry.key,
    )


def _candidate_source_sort_key(source: dict[str, Any]) -> tuple[int, int, float, str]:
    return (
        _SOURCE_PRIORITY.get(str(source.get("source_type") or ""), 99),
        _optional_int(source.get("rank")) or 10**9,
        -float(_optional_float(source.get("score")) or 0.0),
        str(source.get("query") or source.get("graph_match_type") or ""),
    )


def _candidate_key(candidate: dict[str, Any]) -> str:
    evidence_unit_id = str(candidate.get("evidence_unit_id") or "").strip()
    if evidence_unit_id:
        return f"evidence_unit:{evidence_unit_id}"
    target_segment_id = str(candidate.get("target_segment_id") or "").strip()
    if target_segment_id:
        return f"target_segment:{target_segment_id}"
    segment_id = str(candidate.get("segment_id") or "").strip()
    if segment_id:
        return f"segment:{segment_id}"
    return ""


def _source_identity(source: dict[str, Any]) -> tuple[Any, ...]:
    return (
        source.get("source_type"),
        source.get("rank"),
        source.get("query"),
        tuple(_string_list(source.get("graph_path"))),
        source.get("graph_match_type"),
    )


def _candidate_sources(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    value = candidate.get("candidate_sources")
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _candidate_keys_by_source(candidates: list[dict[str, Any]]) -> dict[str, set[str]]:
    by_source: dict[str, set[str]] = {}
    for candidate in candidates:
        key = _candidate_key(candidate) or str(candidate.get("candidate_key") or "")
        for source in _candidate_sources(candidate):
            source_type = str(source.get("source_type") or "")
            if source_type and key:
                by_source.setdefault(source_type, set()).add(key)
    return by_source


def _source_recall(
    *,
    candidates: list[dict[str, Any]],
    targets: dict[str, list[str]],
) -> dict[str, Any]:
    evidence_targets = set(targets.get("evidence_unit_ids") or [])
    segment_targets = set(targets.get("target_segment_ids") or [])
    target_count = len(evidence_targets) + len(segment_targets)
    by_source: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        matched_targets = _matched_targets(candidate, evidence_targets, segment_targets)
        if not matched_targets:
            continue
        for source in _candidate_sources(candidate):
            source_type = str(source.get("source_type") or "")
            if not source_type:
                continue
            summary = by_source.setdefault(
                source_type,
                {"recalled_count": 0, "recalled_targets": [], "ranks": []},
            )
            for target in matched_targets:
                if target not in summary["recalled_targets"]:
                    summary["recalled_targets"].append(target)
                    summary["recalled_count"] += 1
                summary["ranks"].append(
                    {
                        "target": target,
                        "source_rank": source.get("rank"),
                        "candidate_rank": candidate.get("rank"),
                    }
                )
    for summary in by_source.values():
        summary["recalled_targets"] = sorted(summary["recalled_targets"])
    return {
        "target_count": target_count,
        "targets": targets,
        "by_source": dict(sorted(by_source.items())),
    }


def _matched_targets(
    candidate: dict[str, Any],
    evidence_targets: set[str],
    segment_targets: set[str],
) -> list[str]:
    matched: list[str] = []
    evidence_unit_id = str(candidate.get("evidence_unit_id") or "")
    target_segment_id = str(candidate.get("target_segment_id") or candidate.get("segment_id") or "")
    if evidence_unit_id in evidence_targets:
        matched.append(f"evidence_unit:{evidence_unit_id}")
    if target_segment_id in segment_targets:
        matched.append(f"target_segment:{target_segment_id}")
    return matched


def _query_tokens(query: str) -> list[str]:
    token = ""
    tokens: list[str] = []
    for character in query.casefold():
        if character.isalnum() or "가" <= character <= "힣":
            token += character
            continue
        if token:
            tokens.append(token)
            token = ""
    if token:
        tokens.append(token)
    return [item for item in tokens if len(item) >= 2]


def _normalize_text(value: str) -> str:
    return " ".join(_query_tokens(value))


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _record_data(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return record
    if hasattr(record, "data"):
        data = record.data()
        if isinstance(data, dict):
            return data
    return dict(record)


def _node_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    return None


def _public_concept_payload(value: Any) -> dict[str, Any]:
    payload = _node_payload(value) or {}
    allowed_keys = {
        "canonical_label",
        "concept_id",
        "concept_type",
        "confidence",
        "global_concept_id",
        "label",
        "lecture_id",
        "project_id",
    }
    return _compact({key: payload.get(key) for key in allowed_keys})


def _public_source_evidence_ref(value: Any) -> dict[str, Any]:
    payload = _node_payload(value) or {}
    allowed_keys = {
        "concept_id",
        "confidence",
        "evidence_unit_id",
        "lecture_id",
        "source_signal",
        "source_type",
    }
    return _compact({key: payload.get(key) for key in allowed_keys})


def _merge_missing_values(base: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in fallback.items():
        if key not in merged or merged[key] in (None, "", [], {}):
            merged[key] = value
    return merged


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value is not None and value != "" and value != [] and value != {}
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value
