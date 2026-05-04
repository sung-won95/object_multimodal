from oarag.schemas import (
    AUDIO_VISUAL_CONSISTENCY_ARTIFACT,
    VLM_ARTIFACT_PATHS,
    VLM_COMMON_RECORD_FIELDS,
    VLM_FRAME_CANDIDATES_ARTIFACT,
    VLM_JSONL_ARTIFACT_CONTRACT,
    VLM_PROJECT_MANIFEST_SECTION,
    VLM_VISUAL_OBSERVATIONS_ARTIFACT,
    AudioVisualConsistencyRecord,
    EntityLink,
    EvidenceWindow,
    LectureSegment,
    VLMFrameCandidate,
    VLMVisualObservation,
    VisualEntity,
    build_vlm_project_manifest_fields,
    mention_candidates,
)


def test_mention_candidates_do_not_match_korean_character_substrings() -> None:
    assert "이것" not in mention_candidates("배트 사이즈는 복잡합니다.")
    assert mention_candidates("이 부분을 보세요.") == ["이 부분"]


def test_visual_entity_from_dict_to_dict_roundtrip() -> None:
    payload = {
        "entity_id": "ent_frame_1_0001",
        "project_id": "sample_project",
        "frame_id": "frame_000001",
        "timestamp": "12.34",
        "frame_path": "/tmp/frame_000001.jpg",
        "bbox": {"left": "10", "top": 20, "width": "50.5", "height": "30"},
        "text": "Matrix A",
        "entity_type": "ocr_text",
        "confidence": "0.91",
        "source": "ocr:tesseract",
        "visual_description": "Matrix A in the upper-left slide area",
        "position": {"region": "upper-left", "x": 0.1},
        "relations": [{"type": "inside", "target": "slide"}],
        "parser_version": "vlm-jsonl-v1",
        "source_model": "stub-vlm",
    }

    entity = VisualEntity.from_dict(payload)
    encoded = entity.to_dict()

    assert encoded == {
        "entity_id": "ent_frame_1_0001",
        "project_id": "sample_project",
        "frame_id": "frame_000001",
        "timestamp": 12.34,
        "frame_path": "/tmp/frame_000001.jpg",
        "bbox": {"left": 10.0, "top": 20.0, "width": 50.5, "height": 30.0},
        "text": "Matrix A",
        "entity_type": "ocr_text",
        "confidence": 0.91,
        "source": "ocr:tesseract",
        "visual_description": "Matrix A in the upper-left slide area",
        "position": {"region": "upper-left", "x": 0.1},
        "relations": [{"type": "inside", "target": "slide"}],
        "parser_version": "vlm-jsonl-v1",
        "source_model": "stub-vlm",
    }


def test_visual_entity_from_dict_defaults_vlm_metadata() -> None:
    entity = VisualEntity.from_dict(
        {
            "entity_id": "ent_frame_1_0001",
            "project_id": "sample_project",
            "frame_id": "frame_000001",
            "timestamp": 12.34,
            "frame_path": "/tmp/frame_000001.jpg",
            "text": "Matrix A",
            "entity_type": "ocr_text",
            "source": "ocr:tesseract",
        }
    )

    assert entity.visual_description is None
    assert entity.position is None
    assert entity.relations == []
    assert entity.parser_version is None
    assert entity.source_model is None


def test_vlm_frame_candidate_from_dict_to_dict_roundtrip() -> None:
    payload = {
        "schema_version": "vlm-consistency-v1",
        "project_id": "sample_project",
        "video_id": "lecture_01",
        "frame_id": "frame_000010",
        "timestamp": "12.5",
        "segment_id": "seg_lecture_01_000003",
        "backend": "vlm-jsonl",
        "source_model": "offline-vlm",
        "model_version": "2026-05-04",
        "confidence": "0.77",
        "status": "selected",
        "frame_path": "frames/frame_000010.jpg",
        "selection_reason": "segment_anchor",
        "rank": "3",
        "metadata": {"window_seconds": 2.0},
    }

    record = VLMFrameCandidate.from_dict(payload)

    assert record.to_dict() == {
        "project_id": "sample_project",
        "video_id": "lecture_01",
        "frame_id": "frame_000010",
        "timestamp": 12.5,
        "segment_id": "seg_lecture_01_000003",
        "backend": "vlm-jsonl",
        "source_model": "offline-vlm",
        "model_version": "2026-05-04",
        "confidence": 0.77,
        "status": "selected",
        "frame_path": "frames/frame_000010.jpg",
        "selection_reason": "segment_anchor",
        "rank": 3,
        "metadata": {"window_seconds": 2.0},
        "schema_version": "vlm-consistency-v1",
    }


def test_vlm_visual_observation_from_dict_to_dict_roundtrip() -> None:
    payload = {
        "project_id": "sample_project",
        "video_id": "lecture_01",
        "frame_id": "frame_000010",
        "timestamp": 12.5,
        "segment_id": "seg_lecture_01_000003",
        "backend": "vlm-jsonl",
        "source_model": "offline-vlm",
        "model_version": None,
        "confidence": "0.88",
        "status": "observed",
        "observation_id": "obs_frame_000010_0001",
        "observation_type": "diagram",
        "visual_description": "A labeled matrix is visible near the center.",
        "detected_text": "Matrix A",
        "bbox": {"left": "10", "top": 20, "width": "50.5", "height": "30"},
        "position": {"region": "center"},
        "attributes": {"color": "blue"},
        "relations": [{"type": "near", "target": "label_a"}, "ignored"],
        "metadata": {"prompt_template": "vlm-observation-v1"},
    }

    record = VLMVisualObservation.from_dict(payload)

    assert record.to_dict() == {
        "observation_id": "obs_frame_000010_0001",
        "project_id": "sample_project",
        "video_id": "lecture_01",
        "frame_id": "frame_000010",
        "timestamp": 12.5,
        "segment_id": "seg_lecture_01_000003",
        "backend": "vlm-jsonl",
        "source_model": "offline-vlm",
        "model_version": None,
        "confidence": 0.88,
        "status": "observed",
        "observation_type": "diagram",
        "visual_description": "A labeled matrix is visible near the center.",
        "detected_text": "Matrix A",
        "bbox": {"left": 10.0, "top": 20.0, "width": 50.5, "height": 30.0},
        "position": {"region": "center"},
        "attributes": {"color": "blue"},
        "relations": [{"type": "near", "target": "label_a"}],
        "metadata": {"prompt_template": "vlm-observation-v1"},
        "schema_version": "vlm-consistency-v1",
    }


def test_audio_visual_consistency_from_dict_to_dict_roundtrip() -> None:
    payload = {
        "project_id": "sample_project",
        "video_id": "lecture_01",
        "frame_id": "frame_000010",
        "timestamp": "12.5",
        "segment_id": "seg_lecture_01_000003",
        "backend": "vlm-jsonl",
        "source_model": "offline-vlm",
        "model_version": "2026-05-04",
        "confidence": "0.91",
        "status": "completed",
        "consistency_id": "avc_seg_000003_frame_000010",
        "consistency": "consistent",
        "visual_observation_ids": ["obs_frame_000010_0001"],
        "evidence_refs": ["seg_lecture_01_000003"],
        "failure_reason": None,
        "skip_reason": None,
        "metadata": {"rule": "semantic_match"},
    }

    record = AudioVisualConsistencyRecord.from_dict(payload)

    assert record.to_dict() == {
        "consistency_id": "avc_seg_000003_frame_000010",
        "project_id": "sample_project",
        "video_id": "lecture_01",
        "frame_id": "frame_000010",
        "timestamp": 12.5,
        "segment_id": "seg_lecture_01_000003",
        "backend": "vlm-jsonl",
        "source_model": "offline-vlm",
        "model_version": "2026-05-04",
        "confidence": 0.91,
        "status": "completed",
        "consistency": "consistent",
        "visual_observation_ids": ["obs_frame_000010_0001"],
        "evidence_refs": ["seg_lecture_01_000003"],
        "failure_reason": None,
        "skip_reason": None,
        "metadata": {"rule": "semantic_match"},
        "schema_version": "vlm-consistency-v1",
    }


def test_vlm_artifact_contracts_define_paths_and_common_fields() -> None:
    assert VLM_ARTIFACT_PATHS == {
        VLM_FRAME_CANDIDATES_ARTIFACT: "manifests/vlm_frame_candidates.jsonl",
        VLM_VISUAL_OBSERVATIONS_ARTIFACT: "manifests/vlm_visual_observations.jsonl",
        AUDIO_VISUAL_CONSISTENCY_ARTIFACT: "manifests/audio_visual_consistency.jsonl",
    }

    for artifact_name, contract in VLM_JSONL_ARTIFACT_CONTRACT.items():
        assert contract["path"] == VLM_ARTIFACT_PATHS[artifact_name]
        assert set(VLM_COMMON_RECORD_FIELDS).issubset(set(contract["fields"]))


def test_build_vlm_project_manifest_fields() -> None:
    fields = build_vlm_project_manifest_fields(
        backend="vlm-jsonl",
        source_model="offline-vlm",
        model_version="2026-05-04",
        status="completed",
        settings={"temperature": 0, "max_frames_per_segment": 3},
        counts={
            VLM_FRAME_CANDIDATES_ARTIFACT: 2,
            VLM_VISUAL_OBSERVATIONS_ARTIFACT: 1,
            AUDIO_VISUAL_CONSISTENCY_ARTIFACT: 1,
        },
        failures={"count": 1, "reasons": {"backend_error": 1}},
        skips={"count": "2", "reasons": {"missing_segment": 2}},
    )

    section = fields[VLM_PROJECT_MANIFEST_SECTION]

    assert fields["artifacts"] == VLM_ARTIFACT_PATHS
    assert fields["counts"][VLM_FRAME_CANDIDATES_ARTIFACT] == 2
    assert section["schema_version"] == "vlm-consistency-v1"
    assert section["backend"] == "vlm-jsonl"
    assert section["source_model"] == "offline-vlm"
    assert section["settings"] == {"temperature": 0, "max_frames_per_segment": 3}
    assert section["artifacts"] == VLM_ARTIFACT_PATHS
    assert section["counts"][AUDIO_VISUAL_CONSISTENCY_ARTIFACT] == 1
    assert section["failures"] == {"count": 1, "reasons": {"backend_error": 1}}
    assert section["skips"] == {"count": 2, "reasons": {"missing_segment": 2}}


def test_entity_link_from_dict_to_dict_roundtrip() -> None:
    payload = {
        "link_id": "link_seg_1_ent_1",
        "project_id": "sample_project",
        "segment_id": "seg_1",
        "entity_id": "ent_1",
        "frame_id": "frame_000001",
        "link_type": "time_overlap+lexical_match",
        "score": 1.25,
        "evidence": ["time_overlap", "lexical_match"],
        "time_overlap": True,
        "lexical_match": ["stack", "size"],
        "mention_candidate": ["stack"],
    }

    link = EntityLink.from_dict(payload)
    encoded = link.to_dict()

    assert encoded == payload


def test_local_transcript_segment_id_is_meili_safe() -> None:
    segment = LectureSegment.from_local_transcript(
        project_id="sample_project",
        video_id="10. Deviating From The Charts",
        seq_no=1,
        start_time=0.0,
        end_time=1.0,
        text="Sample text",
    )

    assert segment.segment_id == "seg_10_Deviating_From_The_Charts_000001"


def test_evidence_window_keeps_additive_fields_optional() -> None:
    window = EvidenceWindow(
        target_segment_id="seg_1",
        project_id="project",
        video_id="video",
        start_time=0.0,
        end_time=1.0,
        transcript_segments=[],
        frame_refs=[],
    )

    assert window.to_dict()["target_segment"] == {}
    assert window.to_dict()["neighbor_segments"] == []
    assert window.to_dict()["window_config"] == {}
