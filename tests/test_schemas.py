from oarag.schemas import mention_candidates


def test_mention_candidates_do_not_match_korean_character_substrings() -> None:
    assert "이것" not in mention_candidates("배트 사이즈는 복잡합니다.")
    assert mention_candidates("이 부분을 보세요.") == ["이 부분"]
