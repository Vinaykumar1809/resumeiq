from src.similarity import calculate_match_score


def test_similarity_score_is_high_for_related_text():
    score = calculate_match_score(
        "Python SQL machine learning classification",
        "Python SQL machine learning classification",
    )

    assert score > 90


def test_similarity_score_handles_empty_text():
    assert calculate_match_score("", "Python developer") == 0.0
