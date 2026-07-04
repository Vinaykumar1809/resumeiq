"""Candidate scoring rules."""

from __future__ import annotations


def calculate_skill_match_score(matched_skills: list[str], required_skills: list[str]) -> float:
    if not required_skills:
        return 0.0
    score = (len(set(matched_skills)) / len(set(required_skills))) * 100
    return round(score, 2)


def calculate_final_score(text_similarity_score: float, skill_match_score: float) -> float:
    score = (0.70 * text_similarity_score) + (0.30 * skill_match_score)
    return round(max(0.0, min(100.0, score)), 2)


def classify_candidate_strength(final_score: float) -> str:
    if final_score >= 80:
        return "Strong Match"
    if final_score >= 60:
        return "Moderate Match"
    if final_score >= 40:
        return "Weak Match"
    return "Very Weak Match"
