from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_validate_run_rejects_fixture_provider_for_paper_ready(tmp_path: Path) -> None:
    module = _load_script_module()
    metrics_path, query_results_path, semantic_smoke_path = _write_validation_artifacts(tmp_path)

    validation = module.validate_run(
        indexed_projects=[_indexed_project(provider_id="deterministic_fixture")],
        metrics_path=metrics_path,
        query_results_path=query_results_path,
        semantic_smoke_path=semantic_smoke_path,
        require_real_provider=True,
    )

    assert validation["ok"] is False
    assert "deterministic_fixture" in " ".join(validation["failures"])
    assert validation["semantic_smoke_ok"] is True


def test_validate_run_rejects_local_hash_query_vector_source(tmp_path: Path) -> None:
    module = _load_script_module()
    metrics_path, query_results_path, semantic_smoke_path = _write_validation_artifacts(
        tmp_path,
        query_vector_source="local_hash_v1",
    )

    validation = module.validate_run(
        indexed_projects=[_indexed_project(provider_id="openai_compatible")],
        metrics_path=metrics_path,
        query_results_path=query_results_path,
        semantic_smoke_path=semantic_smoke_path,
        require_real_provider=True,
    )

    assert validation["ok"] is False
    assert "query vector source" in " ".join(validation["failures"])


def test_validate_run_accepts_real_provider_manifest_path(tmp_path: Path) -> None:
    module = _load_script_module()
    metrics_path, query_results_path, semantic_smoke_path = _write_validation_artifacts(tmp_path)

    validation = module.validate_run(
        indexed_projects=[_indexed_project(provider_id="openai_compatible")],
        metrics_path=metrics_path,
        query_results_path=query_results_path,
        semantic_smoke_path=semantic_smoke_path,
        require_real_provider=True,
    )

    assert validation["ok"] is True
    assert validation["provider_ids"] == ["openai_compatible"]
    assert validation["query_vector_sources"] == ["manifest"]


def test_main_fixture_mode_orchestrates_without_public_private_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_script_module()
    manifest_path = tmp_path / "benchmark_matrix_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suites": [
                    {
                        "suite_id": "mitdl_public_01",
                        "type": "retrieval_answer_matrix",
                        "project_dir": "project_a",
                        "queries": "queries.csv",
                        "index": "unused_segments",
                        "window_index": "unused_windows",
                        "visual_index": "unused_visual",
                        "variants": ["segment_lexical", "hybrid", "window_hybrid"],
                        "neighbor_count": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(module, "MeiliClient", lambda **kwargs: object())
    monkeypatch.setattr(
        module,
        "load_input_records",
        lambda path, input_format="auto": [{"segment_id": "seg_1", "window_id": "win_1", "entity_id": "ent_1", "query_id": "q1", "query_text": "What is gradient descent?", "semantic_text": "gradient"}],
    )
    monkeypatch.setattr(
        module,
        "build_project_windows",
        lambda **kwargs: {"counts": {"segments_total": 1, "windows_total": 1, "frames_total": 1, "visual_entities_total": 1}},
    )

    def fake_build_vector_manifest(**kwargs):
        calls.append(("vector", kwargs))
        return {
            "schema_version": "oarag-vector-manifest-v1",
            "kind": kwargs["kind"],
            "embedder": kwargs["embedder"],
            "provider": {
                "provider": "deterministic_fixture",
                "model": "deterministic_fixture_v1",
                "dimensions": 4,
            },
            "dimensions": 4,
            "record_count": 1,
            "vector_count": 1,
            "output_file": kwargs["output_path"].name,
        }

    monkeypatch.setattr(module, "build_vector_manifest", fake_build_vector_manifest)
    monkeypatch.setattr(module, "index_project_segments", lambda *args, **kwargs: _index_summary("segment", kwargs))
    monkeypatch.setattr(module, "index_project_windows", lambda *args, **kwargs: _index_summary("window", kwargs))
    monkeypatch.setattr(module, "index_project_visual_entities", lambda *args, **kwargs: _index_summary("visual", kwargs))

    def fake_run_benchmark(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path, query_results_path, semantic_smoke_path = _write_validation_artifacts(output_dir)
        summary_path = output_dir / "summary.md"
        summary_path.write_text("summary\n", encoding="utf-8")
        metrics_csv_path = output_dir / "metrics_summary.csv"
        metrics_csv_path.write_text("metric,value\n", encoding="utf-8")
        return SimpleNamespace(
            metrics_path=metrics_path,
            metrics_csv_path=metrics_csv_path,
            query_results_path=query_results_path,
            semantic_smoke_path=semantic_smoke_path,
            summary_path=summary_path,
        )

    monkeypatch.setattr(module, "run_benchmark", fake_run_benchmark)

    summary = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--suite-id",
            "mitdl_public_01",
            "--output-root",
            str(tmp_path / "smoke"),
            "--provider",
            "deterministic-fixture",
            "--dimensions",
            "4",
            "--allow-fixture-provider",
        ]
    )
    output = capsys.readouterr().out

    assert summary["paper_ready"] is False
    assert summary["plumbing_ready"] is True
    assert [call[0] for call in calls] == ["vector", "vector", "vector", "vector"]
    assert "deterministic_fixture" in output
    assert str(tmp_path) not in output
    assert "api_key" not in output.lower()


def _write_validation_artifacts(
    directory: Path,
    *,
    query_vector_source: str = "manifest",
) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory / "metrics.json"
    query_results_path = directory / "query_results.jsonl"
    semantic_smoke_path = directory / "semantic_smoke.json"
    metrics_path.write_text(json.dumps({"query_count": 1}), encoding="utf-8")
    query_results_path.write_text(
        json.dumps(
            {
                "config": {
                    "hybrid_retrieval": True,
                    "query_vector": {
                        "configured": True,
                        "source": query_vector_source,
                    },
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    semantic_smoke_path.write_text(
        json.dumps(
            {
                "semantic_live_smoke": {"ok": True},
                "counts": {"query_vector_configured_count": 1},
            }
        ),
        encoding="utf-8",
    )
    return metrics_path, query_results_path, semantic_smoke_path


def _indexed_project(*, provider_id: str) -> dict:
    return {
        "vector_manifests": {
            "segment": {"provider": {"provider": provider_id}},
            "window": {"provider": {"provider": provider_id}},
            "visual": {"provider": {"provider": provider_id}},
            "query": {"provider": {"provider": provider_id}},
        },
        "indexes": {
            "segment": _index_summary("segment", {}),
            "window": _index_summary("window", {}),
            "visual": _index_summary("visual", {}),
        },
    }


def _index_summary(name: str, kwargs: dict) -> dict:
    return {
        "index": kwargs.get("index_uid", f"{name}_index"),
        "indexed_documents": 1,
        "indexed_batches": 1,
        "document_vectors": {
            "generator": None,
            "purpose": "real_embedding_manifest",
            "generated_vector_count": 0,
            "manifest_vector_count": 1,
            "expected_vector_count": 1,
            "missing_vector_count": 0,
        },
    }


def _load_script_module() -> ModuleType:
    script_path = Path("scripts/run_mit_real_embedding_smoke.py").resolve()
    spec = importlib.util.spec_from_file_location("mit_real_embedding_smoke", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
