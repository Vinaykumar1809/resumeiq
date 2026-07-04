from src.preprocessing import clean_text, tokenize_text


def test_clean_text_preserves_technical_terms():
    text = "Python, C++, C#, Node.js, React.js, Scikit-learn and REST API!"
    cleaned = clean_text(text)

    assert "python" in cleaned
    assert "c++" in cleaned
    assert "c#" in cleaned
    assert "node.js" in cleaned
    assert "react.js" in cleaned
    assert "scikit-learn" in cleaned
    assert "rest api" in cleaned


def test_tokenize_text_removes_common_stopwords():
    tokens = tokenize_text("Python is used for machine learning and SQL analysis.")

    assert "python" in tokens
    assert "machine" in tokens
    assert "learning" in tokens
    assert "is" not in tokens
