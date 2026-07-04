"""Skill-gap recommendations."""

from __future__ import annotations


def generate_skill_recommendations(missing_skills: list[str]) -> list[str]:
    if not missing_skills:
        return ["The resume covers the required skills identified in the job description."]

    recommendations = []
    for skill in missing_skills:
        recommendations.append(
            f"Add evidence of {skill} through a project, certification, work example, or measurable achievement."
        )
    recommendations.append(
        "Prioritize the missing skills that appear most central to the job description before applying."
    )
    return recommendations
