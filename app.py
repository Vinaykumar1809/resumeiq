"""Streamlit web app for ResumeIQ."""

from __future__ import annotations

from html import escape
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.prediction import ModelUnavailableError, predict_resume_category
from src.recommendation import generate_skill_recommendations
from src.resume_parser import extract_text_from_pdf
from src.scoring import (
    calculate_final_score,
    calculate_skill_match_score,
    classify_candidate_strength,
)
from src.similarity import calculate_match_score
from src.skill_extractor import compare_skills, extract_skills, load_skill_dictionary


PROJECT_ROOT = Path(__file__).resolve().parent
SKILL_DICTIONARY_PATH = PROJECT_ROOT / "data" / "skill_dictionary.json"
MIN_JD_WORDS = 20


st.set_page_config(
    page_title="ResumeIQ",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_styles() -> None:
    st.markdown(
        """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: rgba(255, 255, 255, 0.88);}
    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        border: 1px solid #dbe4f0;
        background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 54%, #f7fafc 100%);
        border-radius: 14px;
        padding: 28px 30px;
        margin-bottom: 24px;
    }
    .hero h1 {
        color: #102033;
        font-size: 2.35rem;
        line-height: 1.1;
        margin: 0 0 10px 0;
        letter-spacing: 0;
    }
    .hero p {
        color: #44576d;
        font-size: 1.02rem;
        max-width: 860px;
        margin: 0;
    }
    .section-card {
        border: 1px solid #dbe4f0;
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }
    .metric-card {
        border: 1px solid #dbe4f0;
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 16px 14px 16px;
        min-height: 116px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }
    .metric-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.55rem;
        font-weight: 760;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }
    .strength-strong { border-top: 4px solid #16a34a; }
    .strength-moderate { border-top: 4px solid #2563eb; }
    .strength-weak { border-top: 4px solid #d97706; }
    .strength-very-weak { border-top: 4px solid #dc2626; }
    .chip-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
    }
    .chip {
        border: 1px solid #dbe4f0;
        border-radius: 999px;
        padding: 6px 10px;
        background: #f8fafc;
        color: #223044;
        font-size: 0.88rem;
        line-height: 1.1;
    }
    .chip.good {
        border-color: #bbf7d0;
        background: #f0fdf4;
        color: #14532d;
    }
    .chip.missing {
        border-color: #fecaca;
        background: #fef2f2;
        color: #7f1d1d;
    }
    .recommendation {
        border-left: 4px solid #2563eb;
        background: #f8fbff;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 8px 0;
        color: #243447;
    }
    div[data-testid="stFileUploader"] section {
        border-color: #cbd5e1;
        background: #f8fafc;
    }
    div.stButton > button {
        border-radius: 10px;
        min-height: 46px;
        font-weight: 700;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def _render_sidebar() -> None:
    st.sidebar.title("ResumeIQ")
    st.sidebar.caption("Resume shortlisting and skill gap analysis.")
    st.sidebar.markdown("### ML concepts used")
    st.sidebar.markdown(
        """
- Well-defined supervised learning problem
- TF-IDF feature representation
- Cosine similarity for resume-job matching
- Naive Bayes, Decision Tree, k-NN, and SVM for role prediction
- Weighted F1-score model selection
- Accuracy, precision, recall, F1-score, and confusion matrix
- Inductive bias and generalization through model comparison
"""
    )
    st.sidebar.info(
        "ResumeIQ is a decision-support tool. Scores should support human review, not replace it."
    )


def _strength_class(strength: str) -> str:
    return {
        "Strong Match": "strength-strong",
        "Moderate Match": "strength-moderate",
        "Weak Match": "strength-weak",
        "Very Weak Match": "strength-very-weak",
    }.get(strength, "")


def _metric_card(label: str, value: str, class_name: str = "") -> None:
    st.markdown(
        f"""
<div class="metric-card {class_name}">
    <div class="metric-label">{escape(label)}</div>
    <div class="metric-value">{escape(value)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _score_gauge(label: str, value: float) -> go.Figure:
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%"},
            title={"text": label},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 60], "color": "#fef3c7"},
                    {"range": [60, 80], "color": "#dbeafe"},
                    {"range": [80, 100], "color": "#dcfce7"},
                ],
            },
        )
    ).update_layout(height=240, margin=dict(l=20, r=20, t=40, b=10))


def _skills_chart(matched_count: int, missing_count: int) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Bar(
                x=["Matched", "Missing"],
                y=[matched_count, missing_count],
                marker_color=["#16a34a", "#dc2626"],
            )
        ]
    )
    figure.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Skills")
    return figure


def _render_skill_list(title: str, skills: list[str], empty_message: str) -> None:
    st.subheader(title)
    if not skills:
        st.caption(empty_message)
        return

    chip_class = "good" if "Matched" in title else "missing"
    chips = "".join(
        f'<span class="chip {chip_class}">{escape(skill)}</span>' for skill in skills
    )
    st.markdown(f'<div class="chip-wrap">{chips}</div>', unsafe_allow_html=True)


def _validate_inputs(uploaded_file, job_description: str) -> list[str]:
    errors = []
    if uploaded_file is None:
        errors.append("Please upload a resume PDF before analyzing.")
    elif uploaded_file.type != "application/pdf" and not uploaded_file.name.lower().endswith(".pdf"):
        errors.append("Only PDF resumes are supported.")
    elif uploaded_file.size == 0:
        errors.append("Please upload a non-empty resume PDF.")

    if not job_description.strip():
        errors.append("Please enter a valid job description.")
    elif len(job_description.split()) < MIN_JD_WORDS:
        errors.append("The job description is too short for meaningful analysis.")
    return errors


def _analyze_resume(uploaded_file, job_description: str) -> None:
    skill_dict = load_skill_dictionary(SKILL_DICTIONARY_PATH)

    parse_result = extract_text_from_pdf(uploaded_file)
    if parse_result.error or not parse_result.text.strip():
        st.error("Could not extract text from the uploaded resume. Please upload a readable text-based PDF.")
        if parse_result.error:
            st.caption(parse_result.error)
        return

    resume_text = parse_result.text
    resume_skills = extract_skills(resume_text, skill_dict)
    jd_skills = extract_skills(job_description, skill_dict)
    skill_comparison = compare_skills(resume_skills, jd_skills)

    text_score = calculate_match_score(resume_text, job_description)
    skill_score = calculate_skill_match_score(
        skill_comparison["matched_skills"],
        skill_comparison["required_skills"],
    )
    final_score = calculate_final_score(text_score, skill_score)
    strength = classify_candidate_strength(final_score)
    recommendations = generate_skill_recommendations(skill_comparison["missing_skills"])

    try:
        predicted_role = predict_resume_category(resume_text)
        prediction_warning = None
    except ModelUnavailableError:
        predicted_role = "Unavailable"
        prediction_warning = (
            "Prediction model is unavailable. Resume matching can continue, but role prediction cannot be generated."
        )
    except Exception as exc:
        predicted_role = "Unavailable"
        prediction_warning = f"Role prediction could not be generated: {exc}"

    if prediction_warning:
        st.warning(prediction_warning)
    if not jd_skills:
        st.warning("No required skills were detected from the job description using the current dictionary.")
    if not resume_skills:
        st.warning("No known skills were detected in the resume using the current dictionary.")

    st.divider()
    st.header("Analysis Dashboard")

    metric_cols = st.columns(5)
    with metric_cols[0]:
        _metric_card("Predicted Role", predicted_role)
    with metric_cols[1]:
        _metric_card("Final Score", f"{final_score:.2f}%")
    with metric_cols[2]:
        _metric_card("Resume-Job Match", f"{text_score:.2f}%")
    with metric_cols[3]:
        _metric_card("Skill Match", f"{skill_score:.2f}%")
    with metric_cols[4]:
        _metric_card("Strength", strength, _strength_class(strength))

    st.progress(final_score / 100, text=f"Final candidate score: {final_score:.2f}%")

    chart_cols = st.columns(2)
    chart_cols[0].plotly_chart(_score_gauge("Final Candidate Score", final_score), use_container_width=True)
    chart_cols[1].plotly_chart(
        _skills_chart(
            len(skill_comparison["matched_skills"]),
            len(skill_comparison["missing_skills"]),
        ),
        use_container_width=True,
    )

    skill_cols = st.columns(2)
    with skill_cols[0]:
        _render_skill_list(
            "Matched Skills",
            skill_comparison["matched_skills"],
            "No matched skills found.",
        )
    with skill_cols[1]:
        _render_skill_list(
            "Missing Skills",
            skill_comparison["missing_skills"],
            "No missing required skills found.",
        )

    st.subheader("Skill Recommendations")
    for item in recommendations:
        st.markdown(
            f'<div class="recommendation">{escape(item)}</div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    _inject_styles()
    _render_sidebar()
    st.markdown(
        """
<div class="hero">
    <h1>ResumeIQ</h1>
    <p>AI-Powered Resume Shortlisting and Skill Gap Analyzer. Upload a PDF resume and paste a job description to calculate fit, identify skill gaps, and predict the most suitable role using traditional machine learning.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    input_cols = st.columns([0.9, 1.1], gap="large")
    with input_cols[0]:
        with st.container(border=True):
            st.subheader("Resume")
            uploaded_file = st.file_uploader("Upload Candidate Resume", type=["pdf"])
            st.caption("Accepted format: PDF")

    with input_cols[1]:
        with st.container(border=True):
            st.subheader("Job Description")
            job_description = st.text_area("Paste Job Description", height=220)

    analyze = st.button("Analyze Resume", type="primary", use_container_width=True)

    if analyze:
        errors = _validate_inputs(uploaded_file, job_description)
        if errors:
            for error in errors:
                st.error(error)
            return

        try:
            with st.spinner("Analyzing resume..."):
                _analyze_resume(uploaded_file, job_description)
        except FileNotFoundError:
            st.error("Skill dictionary could not be loaded. Please check the configuration.")
        except Exception as exc:
            st.error("An unexpected error occurred while analyzing the resume.")
            st.caption(str(exc))


if __name__ == "__main__":
    main()
