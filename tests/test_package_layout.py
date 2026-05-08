import importlib

import pytest


@pytest.mark.parametrize(
    ("compat_name", "target_name"),
    [
        ("oarag.alignment", "oarag.ingestion.alignment"),
        ("oarag.audio_visual_consistency", "oarag.vision.audio_visual_consistency"),
        ("oarag.benchmark", "oarag.evaluation.benchmark"),
        ("oarag.config", "oarag.core.config"),
        ("oarag.domain_lexicon", "oarag.core.domain_lexicon"),
        ("oarag.eduvidqa", "oarag.ingestion.eduvidqa"),
        ("oarag.entity_links", "oarag.vision.entity_links"),
        ("oarag.eval", "oarag.evaluation.eval"),
        ("oarag.evidence", "oarag.retrieval.evidence"),
        ("oarag.frame_selection", "oarag.ingestion.frame_selection"),
        ("oarag.graph_document", "oarag.graph.graph_document"),
        ("oarag.graph_ingest", "oarag.graph.graph_ingest"),
        ("oarag.graph_query", "oarag.graph.graph_query"),
        ("oarag.ingest", "oarag.ingestion.ingest"),
        ("oarag.io", "oarag.core.io"),
        ("oarag.lecture_smoke", "oarag.evaluation.lecture_smoke"),
        ("oarag.meili", "oarag.integrations.meili"),
        ("oarag.neo4j", "oarag.integrations.neo4j"),
        ("oarag.project_index", "oarag.retrieval.project_index"),
        ("oarag.project_query", "oarag.retrieval.project_query"),
        ("oarag.reference_resolution", "oarag.vision.reference_resolution"),
        ("oarag.rerank", "oarag.retrieval.rerank"),
        ("oarag.schemas", "oarag.core.schemas"),
        ("oarag.srt", "oarag.core.srt"),
        ("oarag.stt", "oarag.ingestion.stt"),
        ("oarag.visual_entities", "oarag.vision.visual_entities"),
        ("oarag.vlm", "oarag.vision.vlm"),
        ("oarag.vlm_alignment_pipeline", "oarag.vision.vlm_alignment_pipeline"),
        ("oarag.vlm_frame_candidates", "oarag.vision.vlm_frame_candidates"),
        ("oarag.vlm_pipeline_metrics", "oarag.vision.vlm_pipeline_metrics"),
    ],
)
def test_compat_modules_alias_moved_modules(compat_name: str, target_name: str) -> None:
    assert importlib.import_module(compat_name) is importlib.import_module(target_name)
