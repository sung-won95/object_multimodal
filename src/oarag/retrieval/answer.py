from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from oarag.retrieval.project_query import (
    DEFAULT_HYBRID_EMBEDDER,
    DEFAULT_HYBRID_SEMANTIC_RATIO,
    SEGMENT_HIT_SOURCE,
    VISUAL_ENTITY_HIT_SOURCE,
    query_project,
)
from oarag.retrieval.rerank import DEFAULT_RERANK_BACKEND


ANSWER_SCHEMA_VERSION = "grounded-answer-v1"
NO_ANSWER_POLICY_VERSION = "deterministic-no-answer-v1"
GROUNDED_ANSWER = "grounded_answer"
CANDIDATE_EVIDENCE_ONLY = "candidate_evidence_only"
VALID_ANSWER_TYPES = {GROUNDED_ANSWER, CANDIDATE_EVIDENCE_ONLY}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_가-힣]+")
_STOPWORDS = {
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
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class AnswerBackend(Protocol):
    def compose_answer(
        self,
        *,
        query: str,
        retrieval_response: dict[str, Any],
        deterministic_answer: dict[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class NoAnswerPolicy:
    min_query_overlap: float = 0.25
    min_search_score: float = 0.2
    candidate_evidence_limit: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ask_project(
    *,
    client: Any,
    index_uid: str,
    project_dir: Path,
    query: str,
    retrieval_index_kind: str = SEGMENT_HIT_SOURCE,
    visual_index_uid: str | None = None,
    limit: int = 5,
    candidate_pool_limit: int | None = None,
    segments_path: Path | None = None,
    frames_manifest_path: Path | None = None,
    visual_entities_path: Path | None = None,
    entity_links_path: Path | None = None,
    domain_lexicon_path: Path | None = None,
    window_seconds: float | None = None,
    neighbor_count: int = 1,
    previous_neighbor_count: int | None = None,
    next_neighbor_count: int | None = None,
    window_before_seconds: float | None = None,
    window_after_seconds: float | None = None,
    rerank: bool = False,
    rerank_time_hint: str | None = None,
    rerank_backend: str = DEFAULT_RERANK_BACKEND,
    hybrid_retrieval: bool = False,
    hybrid_embedder: str | None = DEFAULT_HYBRID_EMBEDDER,
    hybrid_semantic_ratio: float = DEFAULT_HYBRID_SEMANTIC_RATIO,
    answer_backend: AnswerBackend | None = None,
    no_answer_policy: NoAnswerPolicy | None = None,
) -> dict[str, Any]:
    retrieval_response = query_project(
        client=client,
        index_uid=index_uid,
        retrieval_index_kind=retrieval_index_kind,
        visual_index_uid=visual_index_uid,
        project_dir=project_dir,
        query=query,
        limit=limit,
        candidate_pool_limit=candidate_pool_limit,
        segments_path=segments_path,
        frames_manifest_path=frames_manifest_path,
        visual_entities_path=visual_entities_path,
        entity_links_path=entity_links_path,
        domain_lexicon_path=domain_lexicon_path,
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
        rerank=rerank,
        rerank_time_hint=rerank_time_hint,
        rerank_backend=rerank_backend,
        hybrid_retrieval=hybrid_retrieval,
        hybrid_embedder=hybrid_embedder,
        hybrid_semantic_ratio=hybrid_semantic_ratio,
    )
    answer = compose_answer(
        retrieval_response,
        answer_backend=answer_backend,
        no_answer_policy=no_answer_policy,
    )
    response = dict(retrieval_response)
    response["answer_type"] = answer["answer_type"]
    response["answer"] = answer
    return response


def compose_answer(
    retrieval_response: dict[str, Any],
    *,
    answer_backend: AnswerBackend | None = None,
    no_answer_policy: NoAnswerPolicy | None = None,
) -> dict[str, Any]:
    policy = no_answer_policy or NoAnswerPolicy()
    deterministic_answer = _compose_deterministic_answer(retrieval_response, policy=policy)
    if answer_backend is None:
        return deterministic_answer

    backend_answer = answer_backend.compose_answer(
        query=str(retrieval_response.get("query") or ""),
        retrieval_response=retrieval_response,
        deterministic_answer=deterministic_answer,
    )
    _validate_answer_schema(backend_answer)
    return backend_answer


def format_answer_text(answer: dict[str, Any]) -> str:
    answer_type = str(answer.get("answer_type") or CANDIDATE_EVIDENCE_ONLY)
    lines = [str(answer.get("answer") or "")]
    claims = _list_of_dicts(answer.get("claims"))
    if claims:
        lines.append("")
        lines.append("Claims:")
        for claim in claims:
            citation_ids = ", ".join(str(item) for item in claim.get("citation_ids") or [])
            suffix = f" [{citation_ids}]" if citation_ids else ""
            lines.append(f"- {claim.get('text')}{suffix}")

    evidence = _list_of_dicts(answer.get("candidate_evidence"))
    if answer_type == CANDIDATE_EVIDENCE_ONLY and evidence:
        lines.append("")
        lines.append("Candidate evidence:")
        for item in evidence:
            time_label = _time_label(
                _optional_float(item.get("start_time")),
                _optional_float(item.get("end_time")),
            )
            text = str(item.get("text") or "").strip()
            lines.append(f"- #{item.get('rank')} {item.get('segment_id')} {time_label}: {text}")

    citations = _list_of_dicts(answer.get("citations"))
    if citations:
        lines.append("")
        lines.append("Citations:")
        for citation in citations:
            time_label = _time_label(
                _optional_float(citation.get("start_time")),
                _optional_float(citation.get("end_time")),
            )
            lines.append(
                f"- {citation.get('citation_id')}: segment={citation.get('segment_id')} "
                f"{time_label}, modalities={','.join(citation.get('modalities') or [])}"
            )
    return "\n".join(lines).strip() + "\n"


def _compose_deterministic_answer(
    retrieval_response: dict[str, Any],
    *,
    policy: NoAnswerPolicy,
) -> dict[str, Any]:
    query = str(retrieval_response.get("query") or "")
    support_queries = _support_queries(retrieval_response, fallback_query=query)
    bundles = _list_of_dicts(retrieval_response.get("bundles"))
    citations = [
        _citation_for_bundle(
            bundle=bundle,
            citation_id=f"citation_{index}",
            rank=index,
            retrieval_response=retrieval_response,
        )
        for index, bundle in enumerate(bundles[: policy.candidate_evidence_limit], start=1)
    ]
    candidate_evidence = _candidate_evidence_items(
        bundles=bundles,
        citations=citations,
        support_queries=support_queries,
        retrieval_response=retrieval_response,
        limit=policy.candidate_evidence_limit,
    )
    top_bundle = bundles[0] if bundles else None
    top_signals = (
        _support_signals(top_bundle, support_queries=support_queries)
        if top_bundle
        else _empty_signals(support_queries)
    )
    answer_type, reason = _answer_type_from_signals(
        has_bundle=top_bundle is not None,
        signals=top_signals,
        policy=policy,
    )

    claims: list[dict[str, Any]] = []
    if answer_type == GROUNDED_ANSWER and top_bundle is not None and citations:
        claims = _claims_for_bundle(top_bundle, citation_id=citations[0]["citation_id"])

    answer_text = _answer_text(
        answer_type=answer_type,
        claims=claims,
        candidate_evidence=candidate_evidence,
    )
    return {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "answer_type": answer_type,
        "answer": answer_text,
        "claims": claims,
        "citations": citations,
        "candidate_evidence": candidate_evidence,
        "no_answer_policy": {
            "schema_version": NO_ANSWER_POLICY_VERSION,
            "decision": answer_type,
            "reason": reason,
            "thresholds": policy.to_dict(),
            "signals": top_signals,
        },
        "llm": {
            "enabled": False,
            "backend": None,
        },
    }


def _answer_type_from_signals(
    *,
    has_bundle: bool,
    signals: dict[str, Any],
    policy: NoAnswerPolicy,
) -> tuple[str, str]:
    if not has_bundle:
        return CANDIDATE_EVIDENCE_ONLY, "no_candidate_bundles"
    if signals["query_term_count"] == 0:
        return CANDIDATE_EVIDENCE_ONLY, "no_informative_query_terms"
    score = _optional_float(signals.get("search_score"))
    if score is not None and score < policy.min_search_score:
        return CANDIDATE_EVIDENCE_ONLY, "search_score_below_threshold"
    if not signals["has_textual_claim_source"] and not signals["has_visual_claim_source"]:
        return CANDIDATE_EVIDENCE_ONLY, "no_claim_source_in_candidate"
    if signals["query_overlap_ratio"] >= policy.min_query_overlap:
        return GROUNDED_ANSWER, "query_terms_grounded_in_candidate"
    if signals["linked_entity_count"] > 0 and signals["matched_query_terms"]:
        return GROUNDED_ANSWER, "query_terms_grounded_by_linked_visual_entity"
    return CANDIDATE_EVIDENCE_ONLY, "insufficient_query_overlap"


def _claims_for_bundle(bundle: dict[str, Any], *, citation_id: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    transcript_claim = _first_sentence(_transcript_text(bundle))
    if transcript_claim:
        claims.append(
            {
                "claim_id": f"claim_{len(claims) + 1}",
                "text": transcript_claim,
                "modalities": ["transcript"],
                "citation_ids": [citation_id],
            }
        )

    visual_labels = _visual_labels(bundle)
    if visual_labels:
        claims.append(
            {
                "claim_id": f"claim_{len(claims) + 1}",
                "text": "Visual evidence includes " + ", ".join(visual_labels[:3]) + ".",
                "modalities": ["visual"],
                "citation_ids": [citation_id],
            }
        )
    return claims


def _answer_text(
    *,
    answer_type: str,
    claims: list[dict[str, Any]],
    candidate_evidence: list[dict[str, Any]],
) -> str:
    if answer_type == GROUNDED_ANSWER:
        claim_texts = [str(claim.get("text") or "").strip() for claim in claims]
        claim_texts = [text for text in claim_texts if text]
        return " ".join(claim_texts)

    if candidate_evidence:
        return (
            "The retrieved evidence is not strong enough to answer confidently. "
            "Returning candidate evidence only."
        )
    return "No answer could be grounded because no candidate evidence was retrieved."


def _candidate_evidence_items(
    *,
    bundles: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    support_queries: list[str],
    retrieval_response: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, bundle in enumerate(bundles[:limit], start=1):
        citation = citations[index - 1]
        candidate = _candidate(bundle)
        target = _target_segment(bundle)
        text = _first_sentence(_transcript_text(bundle)) or ", ".join(_visual_labels(bundle)[:3])
        items.append(
            {
                "rank": _optional_int(bundle.get("rank")) or index,
                "segment_id": _segment_id(bundle),
                "video_id": _first_nonempty(candidate.get("video_id"), target.get("video_id")),
                "start_time": _first_float(candidate.get("start_time"), target.get("start_time")),
                "end_time": _first_float(candidate.get("end_time"), target.get("end_time")),
                "timestamp": _first_float(
                    candidate.get("timestamp_center"),
                    target.get("timestamp_center"),
                ),
                "text": text,
                "modalities": _modalities_for_bundle(bundle),
                "citation_ids": [citation["citation_id"]],
                "support": _support_signals(bundle, support_queries=support_queries),
                **_answer_bundle_retrieval_metadata(
                    bundle=bundle,
                    retrieval_response=retrieval_response,
                ),
            }
        )
    return items


def _citation_for_bundle(
    *,
    bundle: dict[str, Any],
    citation_id: str,
    rank: int,
    retrieval_response: dict[str, Any],
) -> dict[str, Any]:
    candidate = _candidate(bundle)
    target = _target_segment(bundle)
    evidence_window = _evidence_window(bundle)
    return {
        "citation_id": citation_id,
        "bundle_rank": _optional_int(bundle.get("rank")) or rank,
        "candidate_rank": _optional_int(candidate.get("rank")),
        "segment_id": _segment_id(bundle),
        "video_id": _first_nonempty(candidate.get("video_id"), target.get("video_id")),
        "start_time": _first_float(candidate.get("start_time"), target.get("start_time")),
        "end_time": _first_float(candidate.get("end_time"), target.get("end_time")),
        "timestamp": _first_float(
            candidate.get("timestamp_center"),
            target.get("timestamp_center"),
        ),
        "frame_refs": _frame_refs(evidence_window),
        "visual_entity_ids": _visual_entity_ids(bundle),
        "entity_link_ids": _entity_link_ids(bundle),
        "modalities": _modalities_for_bundle(bundle),
        **_answer_bundle_retrieval_metadata(
            bundle=bundle,
            retrieval_response=retrieval_response,
        ),
    }


def _answer_bundle_retrieval_metadata(
    *,
    bundle: dict[str, Any],
    retrieval_response: dict[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "retrieval_index_kind": str(retrieval_response.get("index_kind") or ""),
        "retrieval_sources": _retrieval_sources(bundle),
    }
    context = retrieval_response.get("retrieval_context")
    context = context if isinstance(context, dict) else {}
    hybrid = context.get("hybrid_retrieval")
    if isinstance(hybrid, dict):
        metadata["hybrid_retrieval"] = hybrid

    bundle_rerank = bundle.get("rerank")
    if isinstance(bundle_rerank, dict):
        metadata["rerank"] = bundle_rerank
    else:
        rerank_context = context.get("rerank")
        if isinstance(rerank_context, dict):
            metadata["rerank"] = rerank_context
    return metadata


def _support_signals(
    bundle: dict[str, Any] | None,
    *,
    support_queries: list[str],
) -> dict[str, Any]:
    query_terms = _support_query_terms(support_queries)
    if bundle is None:
        return _empty_signals(support_queries)
    candidate = _candidate(bundle)
    text_parts = _bundle_text_parts(bundle)
    evidence_terms = _tokenize(" ".join(text_parts))
    matched_terms = sorted(query_terms & evidence_terms)
    search_score = _optional_float(candidate.get("score"))
    frame_count = len(_frame_refs(_evidence_window(bundle)))
    linked_entity_count = len(_list_of_dicts(bundle.get("linked_entities")))
    visual_entity_count = len(_list_of_dicts(bundle.get("visual_entities")))
    visual_source_hit = candidate.get("source") == VISUAL_ENTITY_HIT_SOURCE or any(
        source.get("source") == VISUAL_ENTITY_HIT_SOURCE
        for source in _list_of_dicts(bundle.get("retrieval_sources"))
    )
    query_overlap_ratio = len(matched_terms) / len(query_terms) if query_terms else 0.0
    return {
        "query_term_count": len(query_terms),
        "support_query_count": len(support_queries),
        "matched_query_terms": matched_terms,
        "query_overlap_ratio": round(query_overlap_ratio, 4),
        "search_score": search_score,
        "frame_ref_count": frame_count,
        "visual_entity_count": visual_entity_count,
        "linked_entity_count": linked_entity_count,
        "visual_source_hit": visual_source_hit,
        "has_textual_claim_source": bool(_transcript_text(bundle).strip()),
        "has_visual_claim_source": bool(
            visual_entity_count or linked_entity_count or visual_source_hit
        ),
    }


def _empty_signals(support_queries: list[str]) -> dict[str, Any]:
    return {
        "query_term_count": len(_support_query_terms(support_queries)),
        "support_query_count": len(support_queries),
        "matched_query_terms": [],
        "query_overlap_ratio": 0.0,
        "search_score": None,
        "frame_ref_count": 0,
        "visual_entity_count": 0,
        "linked_entity_count": 0,
        "visual_source_hit": False,
        "has_textual_claim_source": False,
        "has_visual_claim_source": False,
    }


def _bundle_text_parts(bundle: dict[str, Any]) -> list[str]:
    parts = [_transcript_text(bundle)]
    parts.extend(_visual_text_parts(bundle))
    return [part for part in parts if part.strip()]


def _transcript_text(bundle: dict[str, Any]) -> str:
    target = _target_segment(bundle)
    candidate = _candidate(bundle)
    return str(
        target.get("transcript_text")
        or candidate.get("transcript_excerpt")
        or ""
    ).strip()


def _visual_text_parts(bundle: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for entity in _list_of_dicts(bundle.get("visual_entities")):
        parts.extend(
            str(entity.get(key) or "").strip()
            for key in ("text", "visual_description", "detected_text")
        )
    for link in _list_of_dicts(bundle.get("linked_entities")):
        entity = link.get("entity") if isinstance(link.get("entity"), dict) else {}
        parts.extend(
            str(entity.get(key) or "").strip()
            for key in ("text", "visual_description", "detected_text")
        )
        parts.extend(str(item) for item in link.get("lexical_match") or [])
        parts.extend(str(item) for item in link.get("mention_candidate") or [])
    candidate = _candidate(bundle)
    parts.extend(
        str(candidate.get(key) or "").strip()
        for key in ("visual_entity_text", "visual_description")
    )
    return [part for part in parts if part]


def _visual_labels(bundle: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for entity in _list_of_dicts(bundle.get("visual_entities")):
        for key in ("text", "visual_description", "detected_text"):
            label = _compact(str(entity.get(key) or ""))
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)
    for link in _list_of_dicts(bundle.get("linked_entities")):
        entity = link.get("entity") if isinstance(link.get("entity"), dict) else {}
        for key in ("text", "visual_description", "detected_text"):
            label = _compact(str(entity.get(key) or ""))
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)
    candidate = _candidate(bundle)
    for key in ("visual_entity_text", "visual_description"):
        label = _compact(str(candidate.get(key) or ""))
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _retrieval_sources(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for source in _list_of_dicts(bundle.get("retrieval_sources")):
        sources.append(
            {
                key: source.get(key)
                for key in (
                    "source",
                    "modality",
                    "index",
                    "retrieval_mode",
                    "rank",
                    "score",
                    "original_rank",
                    "original_score",
                    "search_query",
                    "query_index",
                    "fusion_score",
                    "target_segment_id",
                    "deduplicated",
                    "matches",
                    "window_id",
                    "source_segment_ids",
                    "entity_id",
                    "frame_id",
                    "timestamp",
                    "visual_entity_text",
                    "target_resolution",
                )
                if key in source
            }
        )
    return sources


def _modalities_for_bundle(bundle: dict[str, Any]) -> list[str]:
    modalities: list[str] = []
    if _transcript_text(bundle).strip():
        modalities.append("transcript")
    if (
        _frame_refs(_evidence_window(bundle))
        or _list_of_dicts(bundle.get("visual_entities"))
        or _list_of_dicts(bundle.get("linked_entities"))
        or _candidate(bundle).get("source") == VISUAL_ENTITY_HIT_SOURCE
    ):
        modalities.append("visual")
    return modalities


def _frame_refs(evidence_window: dict[str, Any]) -> list[dict[str, Any]]:
    frame_refs: list[dict[str, Any]] = []
    for frame in _list_of_dicts(evidence_window.get("frame_refs")):
        frame_refs.append(
            {
                key: frame.get(key)
                for key in ("frame_id", "timestamp", "frame_path")
                if frame.get(key) is not None
            }
        )
    return frame_refs


def _visual_entity_ids(bundle: dict[str, Any]) -> list[str]:
    entity_ids = {
        str(entity.get("entity_id"))
        for entity in _list_of_dicts(bundle.get("visual_entities"))
        if entity.get("entity_id")
    }
    for link in _list_of_dicts(bundle.get("linked_entities")):
        if link.get("entity_id"):
            entity_ids.add(str(link["entity_id"]))
    candidate_entity_id = _candidate(bundle).get("entity_id")
    if candidate_entity_id:
        entity_ids.add(str(candidate_entity_id))
    return sorted(entity_ids)


def _entity_link_ids(bundle: dict[str, Any]) -> list[str]:
    return sorted(
        str(link["link_id"])
        for link in _list_of_dicts(bundle.get("linked_entities"))
        if link.get("link_id")
    )


def _candidate(bundle: dict[str, Any]) -> dict[str, Any]:
    candidate = bundle.get("candidate")
    return candidate if isinstance(candidate, dict) else {}


def _evidence_window(bundle: dict[str, Any]) -> dict[str, Any]:
    evidence_window = bundle.get("evidence_window")
    return evidence_window if isinstance(evidence_window, dict) else {}


def _target_segment(bundle: dict[str, Any]) -> dict[str, Any]:
    target = _evidence_window(bundle).get("target_segment")
    return target if isinstance(target, dict) else {}


def _segment_id(bundle: dict[str, Any]) -> str:
    candidate = _candidate(bundle)
    target = _target_segment(bundle)
    return str(
        _first_nonempty(
            target.get("segment_id"),
            candidate.get("segment_id"),
            _evidence_window(bundle).get("target_segment_id"),
        )
        or ""
    )


def _validate_answer_schema(answer: dict[str, Any]) -> None:
    answer_type = answer.get("answer_type")
    if answer_type not in VALID_ANSWER_TYPES:
        raise ValueError(
            "Answer backend returned invalid answer_type: "
            f"{answer_type!r}; expected one of {sorted(VALID_ANSWER_TYPES)}"
        )
    if "schema_version" not in answer:
        raise ValueError("Answer backend response must include schema_version")


def _support_queries(retrieval_response: dict[str, Any], *, fallback_query: str) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for candidate in [fallback_query, *_expanded_search_queries(retrieval_response)]:
        text = str(candidate or "").strip()
        if not text:
            continue
        normalized = text.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        queries.append(text)
    return queries


def _expanded_search_queries(retrieval_response: dict[str, Any]) -> list[str]:
    expansion = retrieval_response.get("query_expansion")
    if not isinstance(expansion, dict):
        return []
    queries = expansion.get("search_queries")
    if not isinstance(queries, list):
        return []
    return [str(query) for query in queries if str(query or "").strip()]


def _support_query_terms(support_queries: list[str]) -> set[str]:
    terms: set[str] = set()
    for query in support_queries:
        terms.update(_tokenize(query))
    return terms


def _tokenize(value: str) -> set[str]:
    terms: set[str] = set()
    for match in _TOKEN_RE.finditer(value.lower()):
        term = match.group(0).strip("_")
        if not term or term in _STOPWORDS:
            continue
        if len(term) == 1 and term.isascii() and term.isalpha():
            continue
        terms.add(term)
    return terms


def _first_sentence(value: str, *, limit: int = 360) -> str:
    compacted = _compact(value)
    if not compacted:
        return ""
    match = re.search(r"(?<=[.!?])\s+", compacted)
    sentence = compacted[: match.start()].strip() if match else compacted
    if len(sentence) > limit:
        sentence = sentence[: limit - 3].rstrip() + "..."
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _time_label(start_time: float | None, end_time: float | None) -> str:
    if start_time is None and end_time is None:
        return "@unknown"
    if start_time is None:
        return f"@{end_time:.1f}s"
    if end_time is None:
        return f"@{start_time:.1f}s"
    return f"@{start_time:.1f}-{end_time:.1f}s"
