from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from oarag.cli import build_parser
from oarag.cross_lecture_retrieval_smoke import (
    GRAPH_UNAVAILABLE_SKIP,
    MEILI_UNAVAILABLE_SKIP,
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
                "score": 0.78,
            }
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
    assert suite.projects["lecture_a"].evidence_units == project_a / "segments" / "evidence_units.jsonl"
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
    assert metrics["source_counts"]["meili_raw"]["unique_candidates"] == 2
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
        json.loads(line)
        for line in run.query_results_path.read_text(encoding="utf-8").splitlines()
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
    assert graph_only["candidate_source_metrics"]["by_source"]["graph_traversal"][
        "top_k_recalled_count"
    ] == 1
    graph_rerank = next(row for row in rows if row["variant"] == "graph_aware_rerank")
    assert graph_rerank["graph_aware_rerank"]["enabled"] is True
    assert graph_rerank["graph_aware_rerank"]["top_changed"] is True
    payload = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    assert payload["source_recall_by_variant"]["graph_only"]["graph_traversal"][
        "top_k_recalled_count"
    ] == 2
    assert payload["rerank_diagnostics"]["top_changed_count"] == 2

    public_text = _public_text(run)
    for sensitive in [
        "PRIVATE RAW QUERY TEXT",
        "SECRET transcript",
        "SECRET meili evidence",
        "evu_secret_graph_a",
        "seg_secret_graph_a",
        str(project_a),
        str(project_b),
    ]:
        assert sensitive not in public_text


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
        json.loads(line)
        for line in run.query_results_path.read_text(encoding="utf-8").splitlines()
    ]
    by_variant = {row["variant"]: row for row in rows}
    assert by_variant["meili_only"]["skip_reason"] == MEILI_UNAVAILABLE_SKIP
    assert by_variant["graph_only"]["skip_reason"] == GRAPH_UNAVAILABLE_SKIP
    assert by_variant["meili_graph"]["skip_reason"] == MEILI_UNAVAILABLE_SKIP
    payload = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    assert payload["meili_skip_reasons"][MEILI_UNAVAILABLE_SKIP] == 3
    assert payload["graph_skip_reasons"][GRAPH_UNAVAILABLE_SKIP] == 1


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
