"""Dictionary-based skill extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .preprocessing import clean_text, normalize_skill_name


def load_skill_dictionary(file_path: str | Path) -> dict[str, list[str]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Skill dictionary not found at {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Skill dictionary must be a JSON object grouped by category.")

    normalized: dict[str, list[str]] = {}
    for category, skills in data.items():
        if not isinstance(skills, list):
            raise ValueError(f"Skill category '{category}' must contain a list of skills.")
        normalized[str(category)] = [str(skill) for skill in skills]
    return normalized


def _flatten_skills(skill_dict: dict[str, list[str]]) -> set[str]:
    return {normalize_skill_name(skill) for skills in skill_dict.values() for skill in skills}


def _skill_pattern(skill: str) -> re.Pattern[str]:
    escaped = re.escape(skill).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9+#]){escaped}(?![a-z0-9+#])", re.IGNORECASE)


def extract_skills(text: str, skill_dict: dict[str, list[str]]) -> set[str]:
    cleaned_text = clean_text(text)
    if not cleaned_text or not skill_dict:
        return set()

    detected: set[str] = set()
    for skill in _flatten_skills(skill_dict):
        if skill and _skill_pattern(skill).search(cleaned_text):
            detected.add(skill)
    return detected


def compare_skills(resume_skills: set[str], jd_skills: set[str]) -> dict[str, Any]:
    normalized_resume = {normalize_skill_name(skill) for skill in resume_skills}
    normalized_jd = {normalize_skill_name(skill) for skill in jd_skills}

    matched = sorted(normalized_resume.intersection(normalized_jd))
    missing = sorted(normalized_jd.difference(normalized_resume))
    extra = sorted(normalized_resume.difference(normalized_jd))

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "additional_resume_skills": extra,
        "required_skills": sorted(normalized_jd),
    }
