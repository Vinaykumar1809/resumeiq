"""Text cleaning utilities for ResumeIQ."""

from __future__ import annotations

import re
from typing import Iterable

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except Exception:  # pragma: no cover - import failures are handled at runtime.
    stopwords = None
    word_tokenize = None


TECH_TERM_PATTERNS = {
    "c++": "cplusplus",
    "c#": "csharp",
    "node.js": "nodejs",
    "react.js": "reactjs",
    "scikit-learn": "scikitlearn",
    "rest api": "restapi",
}

TECH_TERM_RESTORE = {value: key for key, value in TECH_TERM_PATTERNS.items()}

FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def _get_stopwords() -> set[str]:
    if stopwords is None:
        return FALLBACK_STOPWORDS
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return FALLBACK_STOPWORDS


def _protect_terms(text: str) -> str:
    protected = text
    for term, placeholder in TECH_TERM_PATTERNS.items():
        protected = re.sub(re.escape(term), placeholder, protected, flags=re.IGNORECASE)
    return protected


def _restore_terms(text: str) -> str:
    restored = text
    for placeholder, term in TECH_TERM_RESTORE.items():
        restored = re.sub(rf"\b{re.escape(placeholder)}\b", term, restored)
    return restored


def clean_text(text: str) -> str:
    """Lowercase, normalize, and remove noisy characters while preserving key terms."""
    if not text:
        return ""

    normalized = text.lower()
    normalized = _protect_terms(normalized)
    normalized = re.sub(r"[^a-z0-9+#.\-\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return _restore_terms(normalized)


def tokenize_text(text: str, remove_stopwords: bool = True) -> list[str]:
    """Return reusable tokens for ML features and skill matching."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

    if word_tokenize is not None:
        try:
            tokens = word_tokenize(cleaned)
        except LookupError:
            tokens = cleaned.split()
    else:
        tokens = cleaned.split()

    tokens = [token.strip(".,;:()[]{}") for token in tokens]
    tokens = [token for token in tokens if token]

    if remove_stopwords:
        excluded = _get_stopwords()
        tokens = [token for token in tokens if token not in excluded]

    return tokens


def normalize_skill_name(skill: str) -> str:
    """Normalize a skill label for case-insensitive comparison."""
    return clean_text(skill).strip()


def preprocess_for_model(text: str) -> str:
    """Create a compact text representation suitable for TF-IDF based models."""
    return " ".join(tokenize_text(text))


def batch_preprocess(texts: Iterable[str]) -> list[str]:
    return [preprocess_for_model(text) for text in texts]
