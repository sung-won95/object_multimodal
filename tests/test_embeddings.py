from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from oarag.embeddings import (
    EmbeddingDimensionError,
    EmbeddingProviderError,
    EmptyEmbeddingTextError,
    OpenAICompatibleEmbeddingProvider,
    build_embedding_vectors,
)


class FakeEmbeddingProvider:
    def __init__(
        self,
        *,
        dimensions: int = 3,
        fail: bool = False,
        returned_dimensions: int | None = None,
    ) -> None:
        self._dimensions = dimensions
        self.fail = fail
        self.returned_dimensions = returned_dimensions or dimensions
        self.calls: list[list[str]] = []

    @property
    def provider_name(self) -> str:
        return "fake_provider"

    @property
    def model(self) -> str:
        return "fake-model"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def public_metadata(self) -> dict[str, Any]:
        return {
            "name": self.provider_name,
            "model": self.model,
            "dimensions": self.dimensions,
        }

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.calls.append(batch)
        if self.fail:
            raise RuntimeError(f"provider failed while embedding {batch[0]}")
        return [
            [float((len(text) + offset) % 17) for offset in range(self.returned_dimensions)]
            for text in batch
        ]


def test_build_embedding_vectors_batches_writes_manifest_and_uses_cache(tmp_path: Path) -> None:
    input_path = tmp_path / "segments.jsonl"
    output_path = tmp_path / "vectors.jsonl"
    manifest_path = tmp_path / "vector_manifest.json"
    cache_dir = tmp_path / "cache"
    write_jsonl(
        input_path,
        [
            {"segment_id": "seg_1", "semantic_text": "alpha semantic"},
            {
                "segment_id": "seg_2",
                "transcript_text": "beta transcript",
                "visual_entities": [{"text": "diagram beta"}],
            },
            {"segment_id": "seg_3", "semantic_text": "gamma semantic"},
        ],
    )
    provider = FakeEmbeddingProvider(dimensions=3)

    summary = build_embedding_vectors(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        provider=provider,
        embedder="lecture_embedder",
        source_fields=["semantic_text", "transcript_text", "visual_entities.text"],
        cache_dir=cache_dir,
        batch_size=2,
    )

    assert provider.calls == [
        ["alpha semantic", "beta transcript diagram beta"],
        ["gamma semantic"],
    ]
    assert summary["provider"] == "fake_provider"
    assert summary["model"] == "fake-model"
    assert summary["dimensions"] == 3
    assert summary["embedder"] == "lecture_embedder"
    assert summary["cache"] == {"enabled": True, "hits": 0, "misses": 3, "writes": 3}
    assert summary["counts"]["provider_batches"] == 2
    assert summary["counts"]["provider_records"] == 3
    assert str(tmp_path) not in json.dumps(summary)

    vector_rows = read_jsonl(output_path)
    assert [row["record_id"] for row in vector_rows] == ["seg_1", "seg_2", "seg_3"]
    assert all(row["provider"] == "fake_provider" for row in vector_rows)
    assert all(len(row["_vectors"]["lecture_embedder"]) == 3 for row in vector_rows)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_json = json.dumps(manifest)
    assert manifest["schema_version"] == "oarag-vector-manifest-v1"
    assert manifest["provider"] == "fake_provider"
    assert manifest["model"] == "fake-model"
    assert manifest["dimensions"] == 3
    assert manifest["embedder"] == "lecture_embedder"
    assert manifest["records"][1]["source_fields"] == [
        "transcript_text",
        "visual_entities.text",
    ]
    assert all(record["text_hash"].startswith("sha256:") for record in manifest["records"])
    assert "alpha semantic" not in manifest_json
    assert "beta transcript" not in manifest_json
    assert str(tmp_path) not in manifest_json

    second_provider = FakeEmbeddingProvider(dimensions=3)
    second_manifest_path = tmp_path / "vector_manifest_second.json"
    second_output_path = tmp_path / "vectors_second.jsonl"
    second_summary = build_embedding_vectors(
        input_path=input_path,
        output_path=second_output_path,
        manifest_path=second_manifest_path,
        provider=second_provider,
        embedder="lecture_embedder",
        source_fields=["semantic_text", "transcript_text", "visual_entities.text"],
        cache_dir=cache_dir,
        batch_size=2,
    )

    assert second_provider.calls == []
    assert second_summary["cache"] == {"enabled": True, "hits": 3, "misses": 0, "writes": 0}
    second_manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
    assert {record["cache"] for record in second_manifest["records"]} == {"hit"}


def test_build_embedding_vectors_rejects_empty_text(tmp_path: Path) -> None:
    input_path = tmp_path / "segments.jsonl"
    write_jsonl(input_path, [{"segment_id": "seg_empty", "semantic_text": "   "}])

    with pytest.raises(EmptyEmbeddingTextError, match="seg_empty"):
        build_embedding_vectors(
            input_path=input_path,
            output_path=tmp_path / "vectors.jsonl",
            manifest_path=tmp_path / "manifest.json",
            provider=FakeEmbeddingProvider(),
            source_fields=["semantic_text"],
        )


def test_build_embedding_vectors_rejects_provider_dimension_mismatch(tmp_path: Path) -> None:
    input_path = tmp_path / "segments.jsonl"
    write_jsonl(input_path, [{"segment_id": "seg_1", "semantic_text": "alpha"}])

    with pytest.raises(EmbeddingDimensionError, match="record 'seg_1'"):
        build_embedding_vectors(
            input_path=input_path,
            output_path=tmp_path / "vectors.jsonl",
            manifest_path=tmp_path / "manifest.json",
            provider=FakeEmbeddingProvider(dimensions=3, returned_dimensions=4),
            source_fields=["semantic_text"],
        )


def test_build_embedding_vectors_wraps_provider_failure_without_raw_text(tmp_path: Path) -> None:
    input_path = tmp_path / "segments.jsonl"
    write_jsonl(input_path, [{"segment_id": "seg_1", "semantic_text": "alpha private text"}])

    with pytest.raises(EmbeddingProviderError) as exc:
        build_embedding_vectors(
            input_path=input_path,
            output_path=tmp_path / "vectors.jsonl",
            manifest_path=tmp_path / "manifest.json",
            provider=FakeEmbeddingProvider(fail=True),
            source_fields=["semantic_text"],
        )

    message = str(exc.value)
    assert "fake_provider" in message
    assert "alpha private text" not in message


def test_openai_compatible_provider_batches_http_request(monkeypatch) -> None:
    requests: list[tuple[str, float, dict[str, Any], dict[str, str]]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "data": [
                        {"index": 1, "embedding": [0.2, 0.8]},
                        {"index": 0, "embedding": [0.9, 0.1]},
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert isinstance(request, urllib.request.Request)
        requests.append(
            (
                request.full_url,
                timeout,
                json.loads(request.data.decode("utf-8")),
                dict(request.header_items()),
            )
        )
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleEmbeddingProvider(
        model="text-embedding-test",
        dimensions=2,
        base_url="https://provider.example/v1",
        api_key="private-api-key",
        timeout_seconds=3.5,
    )

    vectors = provider.embed_batch(["alpha", "beta"])

    assert vectors == [[0.9, 0.1], [0.2, 0.8]]
    assert requests == [
        (
            "https://provider.example/v1/embeddings",
            3.5,
            {
                "model": "text-embedding-test",
                "input": ["alpha", "beta"],
                "dimensions": 2,
            },
            {
                "Content-type": "application/json",
                "User-agent": "oarag-embedding-pipeline/1",
                "Authorization": "Bearer private-api-key",
            },
        )
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
