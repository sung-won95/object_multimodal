from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import Neo4jConfig, load_neo4j_config
from .graph_ingest import classify_neo4j_error, create_neo4j_driver
from .meili import MeiliClient
from .project_query import query_project


REFERENCE_HINT_TERMS = (
    "아까",
    "앞에서",
    "앞서",
    "이전",
    "전에",
    "방금",
    "말했던",
    "설명한",
    "설명했던",
    "내용",
    "이 부분",
    "그 부분",
    "여기",
    "그거",
    "그것",
    "저것",
    "해당",
    "previous",
    "earlier",
    "before",
    "above",
    "this part",
    "that part",
    "mentioned",
    "explained",
    "that",
)


class GraphQuerySession(Protocol):
    def run(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        ...


@dataclass(frozen=True)
class GraphTraversalConfig:
    lookback_segments: int = 3
    per_candidate_limit: int = 12


def graph_query(
    *,
    client: MeiliClient,
    index_uid: str,
    project_dir: Path,
    query: str,
    limit: int = 5,
    segments_path: Path | None = None,
    frames_manifest_path: Path | None = None,
    visual_entities_path: Path | None = None,
    entity_links_path: Path | None = None,
    domain_lexicon_path: Path | None = None,
    project_id: str | None = None,
    config: Neo4jConfig | None = None,
    traversal_config: GraphTraversalConfig | None = None,
    session: GraphQuerySession | None = None,
) -> dict[str, Any]:
    base_response = query_project(
        client=client,
        index_uid=index_uid,
        project_dir=project_dir,
        query=query,
        limit=limit,
        segments_path=segments_path,
        frames_manifest_path=frames_manifest_path,
        visual_entities_path=visual_entities_path,
        entity_links_path=entity_links_path,
        domain_lexicon_path=domain_lexicon_path,
    )
    active_project_id = project_id or str(base_response.get("project_id") or "")
    active_traversal_config = traversal_config or GraphTraversalConfig()
    candidates = [
        candidate
        for candidate in base_response.get("candidates", [])
        if isinstance(candidate, dict) and candidate.get("segment_id")
    ]
    hints = detect_graph_query_hints(query)

    graph_evidence: list[dict[str, Any]] = []
    graph_status = {
        "meilisearch": {
            "available": True,
            "hit_count": len(candidates),
            "status": "hit" if candidates else "miss",
        },
        "neo4j": {
            "available": False,
            "status": "not_attempted",
            "message": None,
        },
        "graph_evidence": {
            "available": False,
            "hit_count": 0,
            "status": "not_attempted",
        },
    }

    started_at = time.monotonic()
    if not candidates:
        graph_status["graph_evidence"]["status"] = "skipped_no_meilisearch_hits"
    elif not hints["has_graph_hint"]:
        graph_status["graph_evidence"]["status"] = "skipped_no_temporal_or_reference_hint"
    else:
        try:
            if session is not None:
                graph_evidence = run_graph_traversal(
                    session=session,
                    project_id=active_project_id,
                    candidates=candidates,
                    config=active_traversal_config,
                )
            else:
                active_config = config or load_neo4j_config()
                driver = create_neo4j_driver(active_config)
                try:
                    driver.verify_connectivity()
                    with driver.session(database=active_config.database) as active_session:
                        graph_evidence = run_graph_traversal(
                            session=active_session,
                            project_id=active_project_id,
                            candidates=candidates,
                            config=active_traversal_config,
                        )
                finally:
                    driver.close()
            graph_status["neo4j"] = {"available": True, "status": "queried", "message": None}
            graph_status["graph_evidence"] = {
                "available": bool(graph_evidence),
                "hit_count": len(graph_evidence),
                "status": "hit" if graph_evidence else "miss",
            }
        except Exception as exc:
            classified = classify_neo4j_error(exc)
            graph_status["neo4j"] = {
                "available": False,
                "status": "unavailable",
                "message": str(classified),
            }
            graph_status["graph_evidence"]["status"] = "skipped_neo4j_unavailable"

    summary = summarize_graph_traversal(
        hints=hints,
        candidates=candidates,
        evidence=graph_evidence,
        elapsed_ms=round((time.monotonic() - started_at) * 1000, 2),
        status=graph_status,
        config=active_traversal_config,
    )
    extended = dict(base_response)
    extended["retrieval_context"] = dict(base_response.get("retrieval_context") or {})
    extended["retrieval_context"]["graph_query"] = {
        "enabled": True,
        "hint_detection": hints,
        "traversal_config": {
            "lookback_segments": active_traversal_config.lookback_segments,
            "per_candidate_limit": active_traversal_config.per_candidate_limit,
        },
    }
    extended["graph_availability"] = graph_status
    extended["graph_evidence"] = graph_evidence
    extended["traversal_summary"] = summary
    extended["counts"] = dict(base_response.get("counts") or {})
    extended["counts"]["graph_evidence"] = len(graph_evidence)
    return extended


def detect_graph_query_hints(query: str) -> dict[str, Any]:
    normalized = " ".join(query.lower().split())
    matched_terms = [term for term in REFERENCE_HINT_TERMS if term.lower() in normalized]
    temporal_terms = [
        term
        for term in matched_terms
        if term in {"아까", "앞에서", "앞서", "이전", "전에", "방금", "previous", "earlier", "before", "above"}
    ]
    reference_terms = [term for term in matched_terms if term not in set(temporal_terms)]
    return {
        "has_graph_hint": bool(matched_terms),
        "has_temporal_hint": bool(temporal_terms),
        "has_reference_hint": bool(reference_terms),
        "matched_terms": matched_terms,
    }


def run_graph_traversal(
    *,
    session: GraphQuerySession,
    project_id: str,
    candidates: list[dict[str, Any]],
    config: GraphTraversalConfig,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        segment_id = str(candidate.get("segment_id") or "")
        if not segment_id:
            continue
        rows = session.run(
            graph_traversal_cypher(lookback_segments=config.lookback_segments),
            {
                "project_id": project_id,
                "segment_id": segment_id,
                "candidate_rank": candidate.get("rank"),
                "candidate_score": candidate.get("score"),
                "limit": config.per_candidate_limit,
            },
        )
        for row in rows:
            item = serialize_traversal_record(row)
            key = (
                item.get("candidate", {}).get("segment_id"),
                item.get("type"),
                tuple(item.get("path", [])),
                _node_id(item.get("resolved_concept")),
                _node_id(item.get("source_segment")),
                _node_id(item.get("visual_entity")),
                _node_id(item.get("frame")),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            evidence.append(item)
    evidence.sort(key=lambda item: (-float(item.get("score") or 0.0), item.get("type") or ""))
    return evidence


def serialize_traversal_record(record: Any) -> dict[str, Any]:
    row = _record_data(record)
    return {
        "type": str(row.get("type") or "graph_path"),
        "candidate": _compact(
            {
                "segment_id": row.get("candidate_segment_id"),
                "rank": row.get("candidate_rank"),
                "score": row.get("candidate_score"),
            }
        ),
        "graph_path": [str(item) for item in row.get("graph_path") or []],
        "path": [str(item) for item in row.get("path") or row.get("graph_path") or []],
        "resolved_concept": _node_payload(row.get("resolved_concept")),
        "source_segment": _node_payload(row.get("source_segment")),
        "visual_entity": _node_payload(row.get("visual_entity")),
        "frame": _node_payload(row.get("frame")),
        "reason": str(row.get("reason") or ""),
        "evidence": [str(item) for item in row.get("evidence") or []],
        "score": _optional_float(row.get("score")),
    }


def summarize_graph_traversal(
    *,
    hints: dict[str, Any],
    candidates: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    elapsed_ms: float,
    status: dict[str, Any],
    config: GraphTraversalConfig,
) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for item in evidence:
        item_type = str(item.get("type") or "graph_path")
        by_type[item_type] = by_type.get(item_type, 0) + 1
    return {
        "attempted": (
            bool(candidates)
            and bool(hints.get("has_graph_hint"))
            and status.get("neo4j", {}).get("status") == "queried"
        ),
        "hint_detection": hints,
        "candidate_segments": [str(candidate.get("segment_id")) for candidate in candidates],
        "evidence_count": len(evidence),
        "evidence_by_type": by_type,
        "lookback_segments": config.lookback_segments,
        "elapsed_ms": elapsed_ms,
        "status": {
            "meilisearch": status.get("meilisearch", {}).get("status"),
            "neo4j": status.get("neo4j", {}).get("status"),
            "graph_evidence": status.get("graph_evidence", {}).get("status"),
        },
    }


def graph_traversal_cypher(*, lookback_segments: int) -> str:
    depth = max(1, min(int(lookback_segments), 10))
    return f"""
MATCH (start:GraphNode:Segment {{project_id: $project_id, segment_id: $segment_id}})
CALL {{
  WITH start
  MATCH path=(source:GraphNode:Segment)-[:NEXT_SEGMENT*1..{depth}]->(start)
  OPTIONAL MATCH (source)-[:MENTIONS]->(concept:GraphNode:Concept)
  OPTIONAL MATCH (entity:GraphNode:VisualEntity)-[:REPRESENTS]->(concept)
  OPTIONAL MATCH (entity_frame:GraphNode:Frame)-[:CONTAINS]->(entity)
  OPTIONAL MATCH (source)-[:ALIGNED_WITH]->(aligned_frame:GraphNode:Frame)
  RETURN 'previous_segment' AS type, start, source, concept, entity AS visual_entity,
         coalesce(entity_frame, aligned_frame) AS frame,
         [node IN nodes(path) | node.key]
           + CASE WHEN concept IS NULL THEN [] ELSE [concept.key] END
           + CASE WHEN entity IS NULL THEN [] ELSE [entity.key] END AS graph_path,
         'Previous transcript segment reached through NEXT_SEGMENT lookback' AS reason,
         ['NEXT_SEGMENT', 'MENTIONS', 'REPRESENTS', 'ALIGNED_WITH'] AS evidence,
         0.72 AS score
  UNION
  WITH start
  MATCH (start)-[ref:REFERS_TO]->(reference:GraphNode:ReferenceMention)-[res:RESOLVES_TO]->(target:GraphNode)
  OPTIONAL MATCH (target)-[:MENTIONS]->(target_concept:GraphNode:Concept)
  OPTIONAL MATCH (target)-[:ALIGNED_WITH]->(segment_frame:GraphNode:Frame)
  OPTIONAL MATCH (target)-[:CONTAINS]->(contained_entity:GraphNode:VisualEntity)
  OPTIONAL MATCH (entity_frame:GraphNode:Frame)-[:CONTAINS]->(target)
  RETURN 'reference_resolution' AS type, start,
         CASE WHEN target:Segment THEN target ELSE null END AS source,
         CASE WHEN target:Concept THEN target ELSE target_concept END AS concept,
         CASE WHEN target:VisualEntity THEN target ELSE contained_entity END AS visual_entity,
         coalesce(segment_frame, entity_frame) AS frame,
         [start.key, reference.key, target.key] AS graph_path,
         coalesce(res.reason, ref.reason, reference.reason, 'Reference mention resolved in graph') AS reason,
         coalesce(res.evidence, ref.evidence, reference.evidence, ['REFERS_TO', 'RESOLVES_TO']) AS evidence,
         coalesce(res.score, ref.score, reference.score, 0.82) AS score
  UNION
  WITH start
  MATCH (start)-[:MENTIONS]->(concept:GraphNode:Concept)
  OPTIONAL MATCH (entity:GraphNode:VisualEntity)-[:REPRESENTS]->(concept)
  OPTIONAL MATCH (frame:GraphNode:Frame)-[:CONTAINS]->(entity)
  RETURN 'concept_visual_expansion' AS type, start, null AS source, concept, entity AS visual_entity, frame,
         [start.key, concept.key] + CASE WHEN entity IS NULL THEN [] ELSE [entity.key] END AS graph_path,
         'Candidate segment concept expanded to linked visual evidence' AS reason,
         ['MENTIONS', 'REPRESENTS'] AS evidence,
         0.64 AS score
}}
RETURN type,
       $segment_id AS candidate_segment_id,
       $candidate_rank AS candidate_rank,
       $candidate_score AS candidate_score,
       graph_path,
       CASE WHEN concept IS NULL THEN null ELSE properties(concept) END AS resolved_concept,
       CASE WHEN source IS NULL THEN null ELSE properties(source) END AS source_segment,
       CASE WHEN visual_entity IS NULL THEN null ELSE properties(visual_entity) END AS visual_entity,
       CASE WHEN frame IS NULL THEN null ELSE properties(frame) END AS frame,
       reason,
       evidence,
       score
ORDER BY score DESC
LIMIT $limit
"""


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
        payload = dict(value)
    elif hasattr(value, "items"):
        payload = dict(value.items())
    else:
        return {"value": str(value)}
    return _compact(payload)


def _node_id(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    for key in ("key", "segment_id", "concept_id", "canonical", "entity_id", "frame_id", "reference_id"):
        if value.get(key):
            return value.get(key)
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}
