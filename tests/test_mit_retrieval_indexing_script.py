from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def test_parse_stages_accepts_aliases() -> None:
    module = _load_script_module()

    assert module.parse_stages("segments,windows,visual_entities") == (
        "segment",
        "window",
        "visual",
    )
    assert module.parse_stages("all") == ("segment", "window", "visual")


def test_mit_retrieval_indexing_script_uses_shared_indexes_and_public_summary(
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
                        "video_id": "video_public_01",
                        "neighbor_count": 0,
                    },
                    {
                        "suite_id": "mitdl_public_02",
                        "type": "retrieval_answer_matrix",
                        "project_dir": "project_b",
                        "video_id": "video_public_02",
                        "neighbor_count": 2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    client_inits: list[tuple[str, str]] = []
    calls: list[tuple[str, dict]] = []
    build_calls: list[dict] = []

    class FakeMeiliClient:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            client_inits.append((base_url, api_key))

    def fake_build_project_windows(**kwargs):
        build_calls.append(kwargs)
        return {
            "counts": {
                "segments_total": 2,
                "windows_total": 2,
                "frames_total": 1,
                "visual_entities_total": 1,
                "window_frame_refs": 1,
                "window_visual_entities": 1,
            },
            "window_config": {"mode": "neighbors", "neighbor_count": kwargs["neighbor_count"]},
        }

    def fake_index_project_segments(client, **kwargs):
        calls.append(("segment", kwargs))
        return _index_summary(kwargs=kwargs, documents=3)

    def fake_index_project_windows(client, **kwargs):
        calls.append(("window", kwargs))
        return _index_summary(kwargs=kwargs, documents=2)

    def fake_index_project_visual_entities(client, **kwargs):
        calls.append(("visual", kwargs))
        return _index_summary(kwargs=kwargs, documents=1)

    monkeypatch.setattr(module, "MeiliClient", FakeMeiliClient)
    monkeypatch.setattr(module, "build_project_windows", fake_build_project_windows)
    monkeypatch.setattr(module, "index_project_segments", fake_index_project_segments)
    monkeypatch.setattr(module, "index_project_windows", fake_index_project_windows)
    monkeypatch.setattr(
        module,
        "index_project_visual_entities",
        fake_index_project_visual_entities,
    )

    summary = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--url",
            "http://meili.example",
            "--api-key",
            "private-test-key",
            "--batch-size",
            "2",
            "--reset",
            "--stages",
            "segment,window,visual",
            "--hybrid-embedder-profile",
            "manual_user_provided_v1",
            "--hybrid-embedder-dimensions",
            "384",
            "--hybrid-embedder-live-smoke",
        ]
    )
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert payload == summary
    assert client_inits == [("http://meili.example", "private-test-key")]
    assert [call[0] for call in calls] == [
        "segment",
        "window",
        "visual",
        "segment",
        "window",
        "visual",
    ]
    for stage in ("segment", "window", "visual"):
        stage_calls = [kwargs for name, kwargs in calls if name == stage]
        assert stage_calls[0]["configure_index"] is True
        assert stage_calls[0]["reset"] is True
        assert stage_calls[0]["hybrid_embedder_live_smoke"] is True
        assert stage_calls[1]["configure_index"] is False
        assert stage_calls[1]["reset"] is False
        assert stage_calls[1]["hybrid_embedder_live_smoke"] is False
    assert [call["neighbor_count"] for call in build_calls] == [0, 2]
    assert payload["totals"] == {
        "segment": {"indexed_documents": 6, "indexed_batches": 4},
        "window": {"built_documents": 4, "indexed_documents": 4, "indexed_batches": 2},
        "visual": {"indexed_documents": 2, "indexed_batches": 2},
    }
    assert str(tmp_path) not in output
    assert "private-test-key" not in output
    assert "settings_snapshot" not in output
    assert "hybrid_embedder_snapshot" not in output


def test_project_vector_manifest_resolves_explicit_and_project_relative_paths(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    project = {"project_dir": tmp_path / "project_a"}

    explicit = module._project_vector_manifest(
        project=project,
        explicit_path=tmp_path / "vectors" / "segments.json",
        default_relative=Path("manifests/segment_vectors.json"),
    )
    relative = module._project_vector_manifest(
        project=project,
        explicit_path=None,
        default_relative=Path("manifests/segment_vectors.json"),
    )
    missing = module._project_vector_manifest(
        project=project,
        explicit_path=None,
        default_relative=None,
    )

    assert explicit == (tmp_path / "vectors" / "segments.json").resolve()
    assert relative == (tmp_path / "project_a" / "manifests" / "segment_vectors.json").resolve()
    assert missing is None


def _index_summary(*, kwargs: dict, documents: int) -> dict:
    return {
        "index": kwargs["index_uid"],
        "indexed_documents": documents,
        "indexed_batches": 1 if documents <= kwargs["batch_size"] else 2,
        "configure_index": kwargs["configure_index"],
        "reset": kwargs["reset"],
        "settings_profile": "test_profile",
        "settings_hash": f"{kwargs['index_uid']}_settings_hash",
        "semantic_source_field_counts": {"semantic_text": documents},
        "document_vectors": {
            "enabled": True,
            "source": "userProvided",
            "generator": "local_hash_v1",
            "purpose": "local_reproducibility_smoke_fallback",
            "quality_claim": "none",
            "embedder_names": ["default"],
            "dimensions_by_embedder": {"default": 384},
            "documents_seen": documents,
            "documents_with_vectors": documents,
            "generated_vector_count": documents,
            "existing_vector_count": 0,
        },
        "hybrid_embedder_profile": "manual_user_provided_v1",
        "hybrid_embedder_hash": "hybrid_settings_hash",
        "hybrid_embedder_live_smoke": {"enabled": kwargs["hybrid_embedder_live_smoke"]},
    }


def _load_script_module() -> ModuleType:
    script_path = Path("scripts/index_mit_deep_learning_retrieval_indexes.py").resolve()
    spec = importlib.util.spec_from_file_location("mit_retrieval_indexes", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
