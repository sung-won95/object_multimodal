from oarag.schemas import EntityLink, EvidenceWindow, LectureSegment, VisualEntity, mention_candidates


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
    }


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
