from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from oarag.cli import build_parser
from oarag.cross_lecture_retrieval_smoke import (
    GRAPH_UNAVAILABLE_SKIP,
    MEILI_UNAVAILABLE_SKIP,
    _rerank_rank_delta,
    _score_component_presence,
    build_candidate_source_metrics,
    load_cross_lecture_smoke_manifest,
    run_cross_lecture_retrieval_smoke,
)
from oarag.graph.graph_ingest import GraphIngestUnavailableError


class AvailableFakeClient:
    def __init__(self, hits_by_query: dict[str, list[dict[str, Any]]]) -> None:
        self.hits_by_query = hits_by_query
        self.searches: list[dict[str, Any]] = []

    def health(self) -> dict[str, Any]:
        return {"status": "available"}

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.searches.append(
            {"index_uid": index_uid, "query": query, "limit": limit, "filter": filter}
        )
        return {
            "hits": self.hits_by_query.get(query, [])[:limit],
            "processingTimeMs": 1,
            "indexUid": index_uid,
            "query": query,
        }


class UnavailableFakeClient:
    def health(self) -> dict[str, Any]:
        raise OSError("connection refused")

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise OSError("connection refused")


class CrossLectureGraphSession:
    def run(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        project_id = str((parameters or {}).get("project_id") or "")
        if project_id.endswith("_b"):
            evidence_unit_id = "evu_secret_graph_b"
            segment_id = "seg_secret_graph_b"
            lecture_id = "lecture_public_b"
        else:
            evidence_unit_id = "evu_secret_graph_a"
            segment_id = "seg_secret_graph_a"
            lecture_id = "lecture_public_a"
        return [
            {
                "evidence_unit_id": evidence_unit_id,
                "project_id": project_id,
                "target_segment_id": segment_id,
                "matched_concept": {
                    "concept_id": "concept_step_size",
                    "canonical_label": "Step size",
                    "aliases": ["learning rate"],
                    "label": "Step size",
                    "lecture_id": lecture_id,
                },
                "related_concept": {
                    "concept_id": "concept_gradient_descent",
                    "label": "Gradient descent",
                    "lecture_id": lecture_id,
                },
                "graph_path": [
                    f"concept:{lecture_id}:step_size",
                    f"concept:{lecture_id}:gradient_descent",
                    f"evidence_unit:{lecture_id}:{evidence_unit_id}",
                ],
                "relationships": ["USES", "CONCEPT_SOURCE_EVIDENCE"],
                "graph_match_type": "related_concept_evidence",
                "source_evidence_ref": {
                    "source_signal": "transcript_statement",
                    "source_type": "concept_graph_artifact",
                    "evidence_unit_id": evidence_unit_id,
                },
                "score": 0.78,
            },
            {
                "evidence_unit_id": evidence_unit_id,
                "project_id": project_id,
                "target_segment_id": segment_id,
                "matched_concept": {
                    "concept_id": "concept_step_size",
                    "canonical_label": "Step size",
                    "aliases": ["learning rate"],
                    "label": "Step size",
                    "lecture_id": lecture_id,
                },
                "graph_path": [
                    f"concept:{lecture_id}:step_size",
                    f"evidence_unit:{lecture_id}:{evidence_unit_id}",
                    f"private_marker:{lecture_id}",
                ],
                "relationships": ["USES"],
                "graph_match_type": "direct_mention",
                "source_evidence_ref": {
                    "source_signal": "PRIVATE RAW TRANSCRIPT says secret formula",
                    "source_type": "/private/tmp/raw/path",
                    "evidence_unit_id": evidence_unit_id,
                },
                "score": 0.71,
            },
        ]


class UnavailableGraphSession:
    def run(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        raise GraphIngestUnavailableError("Neo4j fixture unavailable")


def test_cross_lecture_manifest_parsing_resolves_projects_and_queries(tmp_path: Path) -> None:
    project_a, project_b = _write_projects(tmp_path)
    manifest_path = _write_manifest(tmp_path, project_a=project_a, project_b=project_b)

    manifest = load_cross_lecture_smoke_manifest(
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
    )

    suite = manifest.suites[0]
    assert suite.suite_id == "public_cross_lecture_smoke"
    assert set(suite.projects) == {"lecture_a", "lecture_b"}
    assert (
        suite.projects["lecture_a"].evidence_units
        == project_a / "segments" / "evidence_units.jsonl"
    )
    assert suite.queries[0].query_id == "q_private_a"
    assert suite.queries[0].project_ref == "lecture_a"
    assert suite.queries[1].target_evidence_unit_ids == ("evu_secret_graph_b",)
    assert suite.variants == (
        "meili_only",
        "graph_only",
        "meili_graph",
        "graph_aware_rerank",
    )


def test_candidate_source_metrics_records_recall_and_contribution() -> None:
    metrics = build_candidate_source_metrics(
        {
            "limit": 2,
            "diagnostics": {
                "candidate_pool": {"generated": 3, "returned": 2},
                "source_counts": {
                    "meili_raw": {
                        "source_hits": 2,
                        "unique_candidates": 2,
                        "returned_candidates": 1,
                    },
                    "graph_traversal": {
                        "source_hits": 1,
                        "unique_candidates": 1,
                        "returned_candidates": 1,
                    },
                },
                "source_recall": {
                    "target_count": 1,
                    "by_source": {
                        "graph_traversal": {
                            "recalled_count": 1,
                            "recalled_targets": ["evidence_unit:evu_secret_graph_a"],
                            "ranks": [
                                {
                                    "target": "evidence_unit:evu_secret_graph_a",
                                    "source_rank": 1,
                                    "candidate_rank": 2,
                                }
                            ],
                        }
                    },
                },
                "returned_source_recall": {
                    "target_count": 1,
                    "by_source": {
                        "graph_traversal": {
                            "recalled_count": 1,
                            "recalled_targets": ["evidence_unit:evu_secret_graph_a"],
                            "ranks": [
                                {
                                    "target": "evidence_unit:evu_secret_graph_a",
                                    "source_rank": 1,
                                    "candidate_rank": 2,
                                }
                            ],
                        }
                    },
                },
            },
        },
        top_k=2,
    )

    assert metrics["target_rank"] == 2
    assert metrics["target_rank_bucket"] == "top5"
    assert metrics["top_k_recall"] is True
    assert metrics["target_found_bucket_by_source"] == {
        "generated": "graph_only",
        "top_k": "graph_only",
    }
    assert metrics["graph_recovered_meili_not_found_target"] is True
    assert metrics["source_counts"]["meili_raw"]["unique_candidates"] == 2
    assert metrics["source_mix_counts"] == {"meili": 0, "graph": 0, "both": 0, "unknown": 0}
    graph_metrics = metrics["by_source"]["graph_traversal"]
    assert graph_metrics["top_k_recalled_count"] == 1
    assert graph_metrics["rank_bucket_counts"] == {"top5": 1}
    rendered = json.dumps(metrics, ensure_ascii=False)
    assert "evu_secret_graph_a" not in rendered
    assert "evidence_unit:" not in rendered


def test_cross_lecture_smoke_writes_public_safe_variant_report(tmp_path: Path) -> None:
    project_a, project_b = _write_projects(tmp_path)
    manifest_path = _write_manifest(tmp_path, project_a=project_a, project_b=project_b)
    client = AvailableFakeClient(
        {
            "PRIVATE RAW QUERY TEXT gradient step size diagram A": [
                _hit("evu_secret_meili_a", "seg_secret_meili_a", project_id="public_project_a")
            ],
            "PRIVATE RAW QUERY TEXT gradient step size diagram B": [
                _hit("evu_secret_meili_b", "seg_secret_meili_b", project_id="public_project_b")
            ],
        }
    )

    run = run_cross_lecture_retrieval_smoke(
        client=client,
        manifest_path=manifest_path,
        output_dir=tmp_path / "public_report",
        repo_root=tmp_path,
        graph_session=CrossLectureGraphSession(),
    )

    rows = [
        json.loads(line) for line in run.query_results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 8
    assert {row["variant"] for row in rows} == {
        "meili_only",
        "graph_only",
        "meili_graph",
        "graph_aware_rerank",
    }
    graph_only = next(row for row in rows if row["variant"] == "graph_only")
    assert graph_only["top_k_recall"] is True
    assert graph_only["target_rank"] == 1
    assert graph_only["candidate_source_mix"]["graph"] == 1
    assert graph_only["target_found_bucket_by_source"]["generated"] == "graph_only"
    assert graph_only["graph_recovered_meili_not_found_target"] is False
    assert (
        graph_only["candidate_source_metrics"]["by_source"]["graph_traversal"][
            "top_k_recalled_count"
        ]
        == 1
    )
    assert graph_only["graph_source_buckets"]["relation_type_counts"]["uses"] == 2
    assert graph_only["graph_source_buckets"]["source_signal_counts"]["transcript_statement"] == 1
    assert graph_only["graph_source_buckets"]["source_signal_counts"]["free_text_redacted"] == 1
    assert graph_only["graph_source_buckets"]["source_type_counts"]["concept_graph_artifact"] == 1
    assert graph_only["graph_source_buckets"]["source_type_counts"]["path_like_redacted"] == 1
    assert graph_only["graph_source_buckets"]["concept_match_bucket_counts"]["canonical_match"] == 2
    meili_graph = next(row for row in rows if row["variant"] == "meili_graph")
    assert meili_graph["target_found_bucket_by_source"]["generated"] == "graph_only"
    assert meili_graph["graph_recovered_meili_not_found_target"] is True
    assert meili_graph["candidate_source_mix"] == {
        "meili": 1,
        "graph": 1,
        "both": 0,
        "unknown": 0,
    }
    graph_rerank = next(row for row in rows if row["variant"] == "graph_aware_rerank")
    assert graph_rerank["graph_aware_rerank"]["enabled"] is True
    assert graph_rerank["graph_aware_rerank"]["top_changed"] is True
    assert graph_rerank["rank_delta_vs_meili_graph"]["bucket"] == "top5_to_top1"
    assert graph_rerank["rank_delta_vs_meili_graph"]["top1_recall_delta"] == 1
    assert graph_rerank["rank_delta_vs_meili_graph"]["top5_recall_delta"] == 0
    assert graph_rerank["rank_delta_vs_meili_graph"]["top10_recall_delta"] == 0
    component_presence = graph_rerank["graph_aware_rerank"]["score_component_presence"]
    assert component_presence["query_relevance"]["candidate_count"] >= 1
    assert component_presence["graph_relation_match"]["candidate_count"] >= 1
    assert component_presence["concept_alias_canonical_match"]["candidate_count"] >= 1
    assert component_presence["vlm_object_evidence"]["candidate_count"] >= 1
    assert component_presence["verified_link"]["candidate_count"] >= 1
    payload = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    assert (
        payload["source_recall_by_variant"]["graph_only"]["graph_traversal"]["top_k_recalled_count"]
        == 2
    )
    ablation = payload["graph_candidate_ablation"]
    assert ablation["meili_graph"]["graph_recovered_meili_not_found_target_count"] == 2
    assert ablation["graph_only"]["graph_recovered_meili_not_found_target_count"] == 0
    assert ablation["graph_only"]["candidate_source_mix"]["graph"] == 2
    assert ablation["graph_only"]["graph_source_buckets"]["relation_type_counts"]["uses"] == 4
    assert (
        ablation["graph_only"]["graph_source_buckets"]["source_signal_counts"]["free_text_redacted"]
        == 2
    )
    assert (
        ablation["graph_only"]["graph_source_buckets"]["source_type_counts"]["path_like_redacted"]
        == 2
    )
    assert (
        ablation["graph_only"]["graph_source_buckets"]["concept_match_bucket_counts"][
            "canonical_match"
        ]
        == 4
    )
    assert payload["rerank_diagnostics"]["top_changed_count"] == 2
    assert payload["rerank_diagnostics"]["rank_delta_bucket_counts"] == {"top5_to_top1": 2}
    assert payload["rerank_diagnostics"]["candidate_recall_failure_count"] == 0
    assert payload["rerank_diagnostics"]["target_recall_delta"]["top1"]["net_delta"] == 2
    assert payload["rerank_diagnostics"]["target_recall_delta"]["top5"]["net_delta"] == 0
    assert payload["rerank_diagnostics"]["target_recall_delta"]["top10"]["net_delta"] == 0
    assert (
        payload["rerank_diagnostics"]["score_component_presence"]["graph_relation_match"][
            "candidate_count"
        ]
        >= 2
    )

    public_text = _public_text(run)
    for sensitive in [
        "PRIVATE RAW QUERY TEXT",
        "SECRET transcript",
        "SECRET meili evidence",
        "evu_secret_graph_a",
        "seg_secret_graph_a",
        "PRIVATE RAW TRANSCRIPT",
        "private_raw_transcript_says_secret_formula",
        "/private/tmp",
        "private_tmp_raw_path",
        str(project_a),
        str(project_b),
    ]:
        assert sensitive not in public_text
    for row in rows:
        bucket_keys = json.dumps(row.get("graph_source_buckets", {}), ensure_ascii=False)
        assert "private_raw_transcript_says_secret_formula" not in bucket_keys
        assert "private_tmp_raw_path" not in bucket_keys
        assert "/private/tmp" not in bucket_keys


def test_rerank_rank_delta_buckets_and_candidate_recall_failure_are_public_safe() -> None:
    baseline = {
        "suite_id": "suite_public",
        "project_ref": "lecture_public",
        "query_id": "q_public",
        "variant": "meili_graph",
        "status": "queried",
        "target_configured": True,
        "target_rank": 78,
        "target_rank_bucket": "top100",
        "candidate_source_metrics": {"generated_recalled_count": 1},
    }
    reranked = {
        "suite_id": "suite_public",
        "project_ref": "lecture_public",
        "query_id": "q_public",
        "variant": "graph_aware_rerank",
        "status": "queried",
        "target_configured": True,
        "target_rank": 4,
        "target_rank_bucket": "top5",
        "candidate_source_metrics": {"generated_recalled_count": 1},
    }

    delta = _rerank_rank_delta(baseline=baseline, reranked=reranked)

    assert delta["bucket"] == "top100_to_top5"
    assert delta["rank_delta"] == 74
    assert delta["top10_recall_delta"] == 1
    assert delta["top5_recall_delta"] == 1
    assert delta["top1_recall_delta"] == 0
    assert delta["public_safe"] is True
    rendered = json.dumps(delta, ensure_ascii=False)
    assert "q_public" not in rendered
    assert "lecture_public" not in rendered

    miss = _rerank_rank_delta(
        baseline={
            **baseline,
            "target_rank": None,
            "target_rank_bucket": "not_found",
            "candidate_source_metrics": {"generated_recalled_count": 0},
        },
        reranked={
            **reranked,
            "target_rank": None,
            "target_rank_bucket": "not_found",
            "candidate_source_metrics": {"generated_recalled_count": 0},
        },
    )
    assert miss["bucket"] == "candidate_recall_failure"
    assert miss["candidate_recall_failure"] is True


def test_score_component_presence_uses_public_component_names_only() -> None:
    presence = _score_component_presence(
        [
            {
                "candidate_ref": "evu:public_hash",
                "components": {
                    "evidence_text_query_overlap": 0.2,
                    "relation_type_match": 0.42,
                    "query_concept_direct_match": 0.58,
                    "vlm_visual_entity": 0.26,
                    "verified_alignment": 0.46,
                    "timestamp_fallback_penalty": -0.32,
                    "ocr_only_penalty": -0.38,
                },
            }
        ]
    )

    assert presence["query_relevance"]["candidate_count"] == 1
    assert presence["graph_relation_match"]["candidate_count"] == 1
    assert presence["concept_alias_canonical_match"]["candidate_count"] == 1
    assert presence["vlm_object_evidence"]["candidate_count"] == 1
    assert presence["verified_link"]["candidate_count"] == 1
    assert presence["timestamp_fallback_penalty"]["negative_count"] == 1
    assert presence["ocr_only_penalty"]["negative_count"] == 1
    rendered = json.dumps(presence, ensure_ascii=False)
    assert "evu:public_hash" not in rendered
    assert "evidence text" not in rendered.casefold()


def test_cross_lecture_smoke_separates_meili_and_graph_unavailable_skips(
    tmp_path: Path,
) -> None:
    project_a, project_b = _write_projects(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        project_a=project_a,
        project_b=project_b,
        query_count=1,
    )

    run = run_cross_lecture_retrieval_smoke(
        client=UnavailableFakeClient(),
        manifest_path=manifest_path,
        output_dir=tmp_path / "skip_report",
        repo_root=tmp_path,
        graph_session=UnavailableGraphSession(),
    )

    rows = [
        json.loads(line) for line in run.query_results_path.read_text(encoding="utf-8").splitlines()
    ]
    by_variant = {row["variant"]: row for row in rows}
    assert by_variant["meili_only"]["skip_reason"] == MEILI_UNAVAILABLE_SKIP
    assert by_variant["graph_only"]["skip_reason"] == GRAPH_UNAVAILABLE_SKIP
    assert by_variant["meili_graph"]["skip_reason"] == MEILI_UNAVAILABLE_SKIP
    assert by_variant["graph_only"]["graph"]["reproduce_command"] == (
        "PYTHONPATH=src python -m oarag cross-lecture-retrieval-smoke "
        "--manifest <manifest.json> --output-dir <public-output-dir>"
    )
    payload = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    assert payload["meili_skip_reasons"][MEILI_UNAVAILABLE_SKIP] == 3
    assert payload["graph_skip_reasons"][GRAPH_UNAVAILABLE_SKIP] == 1
    assert payload["graph_candidate_ablation"]["graph_only"][
        "graph_unavailable_reproduce_command"
    ] == (
        "PYTHONPATH=src python -m oarag cross-lecture-retrieval-smoke "
        "--manifest <manifest.json> --output-dir <public-output-dir>"
    )


def test_cross_lecture_retrieval_smoke_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["cross-lecture-retrieval-smoke", "--help"])

    assert exc_info.value.code == 0
    assert "cross-lecture-retrieval-smoke" in capsys.readouterr().out


def _write_manifest(
    tmp_path: Path,
    *,
    project_a: Path,
    project_b: Path,
    query_count: int = 2,
) -> Path:
    queries = [
        {
            "query_id": "q_private_a",
            "project_ref": "lecture_a",
            "query": "PRIVATE RAW QUERY TEXT gradient step size diagram A",
            "target_evidence_unit_ids": ["evu_secret_graph_a"],
        },
        {
            "query_id": "q_private_b",
            "project_ref": "lecture_b",
            "query": "PRIVATE RAW QUERY TEXT gradient step size diagram B",
            "target_evidence_unit_ids": ["evu_secret_graph_b"],
        },
    ][:query_count]
    manifest = {
        "schema_version": "cross-lecture-retrieval-smoke-manifest-v1",
        "run_id": "public_cross_lecture_smoke_run",
        "suites": [
            {
                "suite_id": "public_cross_lecture_smoke",
                "top_k": 2,
                "candidate_pool_limit": 2,
                "graph_limit": 2,
                "projects": [
                    {
                        "project_ref": "lecture_a",
                        "project_dir": str(project_a),
                        "index": "public_cross_lecture_index",
                        "evidence_units": "segments/evidence_units.jsonl",
                    },
                    {
                        "project_ref": "lecture_b",
                        "project_dir": str(project_b),
                        "index": "public_cross_lecture_index",
                        "evidence_units": "segments/evidence_units.jsonl",
                    },
                ],
                "queries": queries,
            }
        ],
    }
    path = tmp_path / "cross_lecture_smoke_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_projects(tmp_path: Path) -> tuple[Path, Path]:
    project_a = tmp_path / "lecture_a_project"
    project_b = tmp_path / "lecture_b_project"
    _write_jsonl(
        project_a / "segments" / "evidence_units.jsonl",
        [
            _hit("evu_secret_meili_a", "seg_secret_meili_a", project_id="public_project_a"),
            _hit("evu_secret_graph_a", "seg_secret_graph_a", project_id="public_project_a"),
        ],
    )
    _write_jsonl(
        project_b / "segments" / "evidence_units.jsonl",
        [
            _hit("evu_secret_meili_b", "seg_secret_meili_b", project_id="public_project_b"),
            _hit("evu_secret_graph_b", "seg_secret_graph_b", project_id="public_project_b"),
        ],
    )
    return project_a, project_b


def _hit(evidence_unit_id: str, target_segment_id: str, *, project_id: str) -> dict[str, Any]:
    return {
        "evidence_unit_id": evidence_unit_id,
        "project_id": project_id,
        "video_id": "video_secret_public_fixture",
        "target_segment_id": target_segment_id,
        "source_segment_ids": [target_segment_id],
        "start_time": 1.0,
        "end_time": 3.0,
        "evidence_text": "SECRET meili evidence text must not appear in public output.",
        "semantic_text": "SECRET semantic evidence text must not appear either.",
        "transcript_window_text": "SECRET transcript text must stay private.",
        "concept_ids": ["concept_gradient_descent", "concept_step_size"],
        "concept_labels": ["Gradient descent", "Step size"],
        "concept_aliases": ["learning rate"],
        "concept_relation_text": "Gradient descent uses step size",
        "visual_entity_ids": ["entity_public_fixture"],
        "source_quality": {
            "has_visual_state": True,
            "has_visual_entity": True,
            "has_vlm_entity": True,
            "has_verified_link": True,
            "candidate_link_count": 1,
            "verified_link_count": 1,
        },
        "_rankingScore": 0.95,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _public_text(run: Any) -> str:
    return "\n".join(
        [
            run.metrics_path.read_text(encoding="utf-8"),
            run.query_results_path.read_text(encoding="utf-8"),
            run.summary_path.read_text(encoding="utf-8"),
        ]
    )
