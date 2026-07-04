from src.skill_extractor import compare_skills, extract_skills


SKILL_DICT = {
    "programming": ["Python", "SQL", "Java"],
    "ml": ["machine learning", "support vector machine", "svm"],
}


def test_extract_skills_finds_dictionary_terms():
    skills = extract_skills("Built Python and SQL models using SVM.", SKILL_DICT)

    assert skills == {"python", "sql", "svm"}


def test_compare_skills_returns_matched_and_missing():
    comparison = compare_skills({"python", "sql"}, {"python", "svm"})

    assert comparison["matched_skills"] == ["python"]
    assert comparison["missing_skills"] == ["svm"]
    assert comparison["additional_resume_skills"] == ["sql"]
