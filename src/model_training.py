"""Train and compare syllabus-aligned resume classifiers."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.preprocessing import preprocess_for_model
else:
    from .preprocessing import preprocess_for_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "resume_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "Dataset not found. Place resume_dataset.csv in data/raw/ with columns "
            "'Resume' and 'Category'. Example: "
            'Resume,Category\\n"Experienced Python developer with SQL and Django experience",'
            '"Python Developer"'
        )

    dataset = pd.read_csv(path)
    required_columns = {"Resume", "Category"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError("Dataset must contain columns: Resume, Category")

    dataset = dataset.dropna(subset=["Resume", "Category"]).copy()
    if dataset.empty:
        raise ValueError("Dataset contains no usable rows after dropping empty values.")
    return dataset


def _build_models() -> dict[str, object]:
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Support Vector Machine": LinearSVC(random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    }


def train_and_compare_models(dataset_path: Path = DATASET_PATH) -> dict[str, object]:
    print("ResumeIQ model training")
    print("Comparing learning algorithms from the Machine Learning syllabus.")

    dataset = _load_dataset(dataset_path)
    dataset["clean_resume"] = dataset["Resume"].apply(preprocess_for_model)
    dataset = dataset[dataset["clean_resume"].str.len() > 0]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(dataset["Category"])

    if len(label_encoder.classes_) < 2:
        raise ValueError("At least two resume categories are required for classification.")

    stratify = y if min(pd.Series(y).value_counts()) >= 2 else None
    test_size = max(0.2, len(label_encoder.classes_) / len(dataset))
    x_train, x_test, y_train, y_test = train_test_split(
        dataset["clean_resume"],
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    results = []
    fitted_models = {}

    for name, model in _build_models().items():
        model.fit(x_train_vec, y_train)
        predictions = model.predict(x_test_vec)
        metrics = {
            "model_name": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
        }
        results.append(metrics)
        fitted_models[name] = model
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1_score']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    best_name = str(results_df.iloc[0]["model_name"])
    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(x_test_vec)

    matrix = confusion_matrix(
        y_test,
        best_predictions,
        labels=list(range(len(label_encoder.classes_))),
    )
    matrix_df = pd.DataFrame(matrix, index=label_encoder.classes_, columns=label_encoder.classes_)

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "resume_classifier.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    matrix_df.to_csv(REPORTS_DIR / "confusion_matrix.csv")

    print(f"Best selected model by weighted F1-score: {best_name}")
    print("Confusion matrix for the best model:")
    print(matrix_df)

    return {
        "best_model_name": best_name,
        "comparison": results_df,
        "confusion_matrix": matrix_df,
    }


if __name__ == "__main__":
    try:
        train_and_compare_models()
    except Exception as exc:
        print(f"Training could not be completed: {exc}")
        sys.exit(1)
