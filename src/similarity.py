"""TF-IDF and cosine similarity matching."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import preprocess_for_model


def calculate_match_score(resume_text: str, job_description: str) -> float:
    resume_clean = preprocess_for_model(resume_text)
    jd_clean = preprocess_for_model(job_description)

    if not resume_clean or not jd_clean:
        return 0.0

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform([resume_clean, jd_clean])
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100
    return round(float(score), 2)
