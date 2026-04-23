from oarag.schemas import VisualEntity, mention_candidates


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
