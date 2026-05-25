import json
import urllib.error
from pathlib import Path

import pytest

from oarag.cli import build_parser
from oarag.embeddings.cache import JsonlEmbeddingCache
from oarag.embeddings.manifest import (
    QUERY_VECTOR_MANIFEST_SCHEMA_VERSION,
    VECTOR_MANIFEST_SCHEMA_VERSION,
    build_vector_manifest,
    load_vector_manifest,
    load_input_records,
)
from oarag.embeddings.providers import (
    DeterministicFixtureEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    SentenceTransformersEmbeddingProvider,
)


def test_build_document_vector_manifest_uses_cache_without_raw_vectors_in_summary(
    tmp_path: Path,
) -> None:
    rows = [
        {"segment_id": "seg_1", "semantic_text": "gradient descent lowers loss"},
        {"segment_id": "seg_2", "semantic_text": "attention maps tokens"},
    ]
    output_path = tmp_path / "vectors.json"
    cache_path = tmp_path / "cache.jsonl"
    provider = DeterministicFixtureEmbeddingProvider(dimensions=6)

    first_summary = build_vector_manifest(
        records=rows,
        provider=provider,
        output_path=output_path,
        id_field="segment_id",
        text_fields=["semantic_text"],
        embedder="lecture_embedder",
        cache=JsonlEmbeddingCache(cache_path),
        batch_size=1,
        source_label="public_fixture",
    )
    second_summary = build_vector_manifest(
        records=rows,
        provider=provider,
        output_path=tmp_path / "vectors_second.json",
        id_field="segment_id",
        text_fields=["semantic_text"],
        embedder="lecture_embedder",
        cache=JsonlEmbeddingCache(cache_path),
        batch_size=2,
    )

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == VECTOR_MANIFEST_SCHEMA_VERSION
    assert manifest["embedder"] == "lecture_embedder"
    assert manifest["provider"]["provider"] == "deterministic_fixture"
    assert manifest["embedders"]["lecture_embedder"]["source"] == "userProvided"
    assert len(manifest["records"]) == 2
    assert len(manifest["records"][0]["vector"]) == 6
    assert first_summary["cache"]["misses"] == 2
    assert second_summary["cache"]["hits"] == 2
    assert '"vector":' not in json.dumps(first_summary, ensure_ascii=False)
    assert str(tmp_path) not in json.dumps(first_summary, ensure_ascii=False)


def test_build_query_vector_manifest_matches_existing_query_loader_shape(tmp_path: Path) -> None:
    output_path = tmp_path / "query_vectors.json"

    summary = build_vector_manifest(
        records=[{"query_id": "q_gradient", "question": "What is a gradient?"}],
        provider=DeterministicFixtureEmbeddingProvider(dimensions=4),
        output_path=output_path,
        id_field="query_id",
        text_fields=["question"],
        kind="query",
        embedder="default",
    )

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == QUERY_VECTOR_MANIFEST_SCHEMA_VERSION
    assert manifest["embedders"]["default"]["source"] == "userProvided"
    assert manifest["queries"]["q_gradient"]["embedder"] == "default"
    assert manifest["queries"]["q_gradient"]["dimensions"] == 4
    assert "vector" in manifest["queries"]["q_gradient"]
    assert summary["kind"] == "query"


def test_build_vector_manifest_fails_on_empty_text(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="has no text"):
        build_vector_manifest(
            records=[{"segment_id": "seg_empty", "semantic_text": ""}],
            provider=DeterministicFixtureEmbeddingProvider(dimensions=3),
            output_path=tmp_path / "vectors.json",
            id_field="segment_id",
            text_fields=["semantic_text"],
        )


def test_build_vector_manifest_fails_on_dimension_mismatch(tmp_path: Path) -> None:
    class BadProvider:
        provider_id = "bad"
        model = "bad-model"
        dimensions = 3

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        def public_config(self) -> dict[str, object]:
            return {"provider": self.provider_id, "model": self.model, "dimensions": self.dimensions}

    with pytest.raises(ValueError, match="dimension mismatch"):
        build_vector_manifest(
            records=[{"segment_id": "seg_1", "semantic_text": "hello"}],
            provider=BadProvider(),
            output_path=tmp_path / "vectors.json",
            id_field="segment_id",
            text_fields=["semantic_text"],
        )


def test_load_input_records_supports_csv(tmp_path: Path) -> None:
    path = tmp_path / "queries.csv"
    path.write_text("query_id,question\nq1,What is loss?\n", encoding="utf-8")

    assert load_input_records(path) == [{"query_id": "q1", "question": "What is loss?"}]


def test_load_vector_manifest_indexes_records_by_embedder(tmp_path: Path) -> None:
    output_path = tmp_path / "vectors.json"
    build_vector_manifest(
        records=[{"segment_id": "seg_1", "semantic_text": "gradient"}],
        provider=DeterministicFixtureEmbeddingProvider(dimensions=4),
        output_path=output_path,
        id_field="segment_id",
        text_fields=["semantic_text"],
        embedder="lecture_embedder",
    )

    loaded = load_vector_manifest(output_path)

    record = loaded.get(embedder="lecture_embedder", record_ids=["seg_1"])
    assert loaded.schema_version == VECTOR_MANIFEST_SCHEMA_VERSION
    assert loaded.dimensions_by_embedder == {"lecture_embedder": 4}
    assert loaded.record_count == 1
    assert record is not None
    assert record.record_id == "seg_1"
    assert len(record.vector) == 4
    assert loaded.public_summary()["provider"]["provider"] == "deterministic_fixture"


def test_load_vector_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "vectors.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": VECTOR_MANIFEST_SCHEMA_VERSION,
                "kind": "document",
                "embedder": "default",
                "dimensions": 2,
                "records": [
                    {"id": "seg_1", "vector": [1.0, 0.0]},
                    {"id": "seg_1", "vector": [0.0, 1.0]},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_vector_manifest(path)


def test_openai_compatible_provider_posts_embedding_request(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self) -> bytes:
            return json.dumps({"data": [{"embedding": [0.25, 0.75]}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.headers["Authorization"]
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleEmbeddingProvider(
        model="text-embedding-test",
        api_base="https://embedding.example.test/v1/",
        api_key="sk-test",
        dimensions=2,
    )

    assert provider.embed_texts(["hello"]) == [[0.25, 0.75]]
    assert captured["url"] == "https://embedding.example.test/v1/embeddings"
    assert captured["body"] == {
        "model": "text-embedding-test",
        "input": ["hello"],
        "dimensions": 2,
    }
    assert captured["authorization"] == "Bearer sk-test"


def test_openai_compatible_provider_redacts_http_error(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "unauthorized",
            hdrs=None,
            fp=_BytesBody(b'{"error":"sk-private-test-key"}'),
        )

    monkeypatch.setenv("OARAG_EMBEDDING_API_KEY", "sk-private-test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleEmbeddingProvider(
        model="text-embedding-test",
        api_key="sk-private-test-key",
        dimensions=2,
    )

    with pytest.raises(RuntimeError) as exc_info:
        provider.embed_texts(["hello"])
    assert "sk-private-test-key" not in str(exc_info.value)
    assert "<redacted>" in str(exc_info.value)


def test_sentence_transformers_provider_loads_cached_model(monkeypatch) -> None:
    captured = {}

    class FakeEncoded:
        def tolist(self):
            return [[0.1, 0.2, 0.3]]

    class FakeModel:
        def __init__(self, model: str, *, local_files_only: bool) -> None:
            captured["model"] = model
            captured["local_files_only"] = local_files_only

        def encode(self, texts, *, normalize_embeddings, convert_to_numpy, show_progress_bar):
            captured["texts"] = texts
            captured["normalize_embeddings"] = normalize_embeddings
            captured["convert_to_numpy"] = convert_to_numpy
            captured["show_progress_bar"] = show_progress_bar
            return FakeEncoded()

    sentence_transformers = pytest.importorskip("sentence_transformers")
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    provider = SentenceTransformersEmbeddingProvider(
        model="sentence-transformers/test-model",
        dimensions=3,
        local_files_only=True,
    )

    assert provider.embed_texts(["gradient descent"]) == [[0.1, 0.2, 0.3]]
    assert captured == {
        "model": "sentence-transformers/test-model",
        "local_files_only": True,
        "texts": ["gradient descent"],
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    assert provider.public_config()["provider"] == "sentence_transformers"
    assert provider.public_config()["quality_claim"] == "provider_embedding"


def test_cli_build_embedding_vectors_fixture(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "segments.jsonl"
    input_path.write_text(
        json.dumps({"segment_id": "seg_1", "semantic_text": "matrix multiplication"})
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "vectors.json"

    args = build_parser().parse_args(
        [
            "build-embedding-vectors",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--id-field",
            "segment_id",
            "--text-field",
            "semantic_text",
            "--provider",
            "deterministic-fixture",
            "--dimensions",
            "5",
        ]
    )
    args.func(args)

    summary = json.loads(capsys.readouterr().out)
    assert summary["provider"]["provider"] == "deterministic_fixture"
    assert summary["record_count"] == 1
    assert summary["output_file"] == "vectors.json"
    assert json.loads(output_path.read_text(encoding="utf-8"))["records"][0]["dimensions"] == 5


def test_cli_build_embedding_vectors_sentence_transformers(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    input_path = tmp_path / "segments.jsonl"
    input_path.write_text(
        json.dumps({"segment_id": "seg_1", "semantic_text": "matrix multiplication"})
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "vectors.json"

    monkeypatch.setattr(
        SentenceTransformersEmbeddingProvider,
        "embed_texts",
        lambda self, texts: [[0.1, 0.2, 0.3] for _ in texts],
    )

    args = build_parser().parse_args(
        [
            "build-embedding-vectors",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--id-field",
            "segment_id",
            "--text-field",
            "semantic_text",
            "--provider",
            "sentence-transformers",
            "--model",
            "sentence-transformers/test-model",
            "--dimensions",
            "3",
            "--local-files-only",
        ]
    )
    args.func(args)

    summary = json.loads(capsys.readouterr().out)
    assert summary["provider"] == {
        "provider": "sentence_transformers",
        "model": "sentence-transformers/test-model",
        "dimensions": 3,
        "local_files_only": True,
        "quality_claim": "provider_embedding",
    }
    assert summary["record_count"] == 1
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["provider"]["provider"] == "sentence_transformers"
    assert manifest["records"][0]["dimensions"] == 3


class _BytesBody:
    def __init__(self, value: bytes) -> None:
        self._value = value

    def read(self) -> bytes:
        return self._value

    def close(self) -> None:
        return None
