"""Resume category prediction using saved scikit-learn artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from .preprocessing import preprocess_for_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "resume_classifier.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"


class ModelUnavailableError(RuntimeError):
    """Raised when trained model artifacts are unavailable."""


def load_model_files(
    model_path: Path = MODEL_PATH,
    vectorizer_path: Path = VECTORIZER_PATH,
    label_encoder_path: Path = LABEL_ENCODER_PATH,
) -> tuple[Any, Any, Any]:
    missing = [
        str(path)
        for path in (model_path, vectorizer_path, label_encoder_path)
        if not path.exists()
    ]
    if missing:
        raise ModelUnavailableError("Missing model files: " + ", ".join(missing))

    return joblib.load(model_path), joblib.load(vectorizer_path), joblib.load(label_encoder_path)


def predict_resume_category(resume_text: str) -> str:
    model, vectorizer, label_encoder = load_model_files()
    cleaned = preprocess_for_model(resume_text)
    if not cleaned:
        raise ValueError("Resume text is empty after preprocessing.")

    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)
    return str(label_encoder.inverse_transform(prediction)[0])
