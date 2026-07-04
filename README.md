# ResumeIQ

ResumeIQ is a Streamlit web app for resume shortlisting and skill gap analysis. It follows: upload a PDF resume, paste a job description, extract skills, compute match scores, predict a suitable role, and show recommendations in a clean dashboard.

## Features

- PDF resume upload and text extraction with `pdfplumber`, plus `PyPDF2` fallback.
- Job description validation and preprocessing.
- Configurable skill extraction from `data/skill_dictionary.json`.
- TF-IDF plus cosine similarity resume-job match score.
- Skill match score and final candidate score: `0.70 * text match + 0.30 * skill match`.
- Candidate strength mapping: Strong, Moderate, Weak, Very Weak.
- Optional role prediction from saved scikit-learn model files.
- Graceful fallback when model files are missing.
- Tests for preprocessing, skill extraction, and similarity scoring.

##  ML Concepts

The core implementation uses traditional machine learning concepts :

- Well-defined learning problem: supervised multi-class resume category prediction.
- Designing a learning system: parsing, preprocessing, features, learning, evaluation, prediction.
- Decision Tree Learning: `DecisionTreeClassifier`.
- Bayesian Learning and Naive Bayes: `MultinomialNB`.
- Instance-Based Learning and k-NN: `KNeighborsClassifier`.
- Support Vector Machines: `LinearSVC`.
- Inductive bias and generalization: model comparison on a train-test split.
- Evaluating hypotheses and estimating accuracy: accuracy, precision, recall, F1-score, and confusion matrix.

No BERT, Sentence-BERT, transformer model, OpenAI API, LLM API, or deep learning model is used in the main implementation.

## Tech Stack

Runtime: Python, Streamlit, Pandas, NumPy, Scikit-learn, NLTK, pdfplumber, PyPDF2, Joblib, Plotly, and Matplotlib.

Development and testing: Pytest via `requirements-dev.txt`.

## Folder Structure

```text
resumeiq/
├── app.py
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── runtime.txt
├── .streamlit/config.toml
├── data/
│   ├── raw/resume_dataset.csv
│   ├── processed/
│   └── skill_dictionary.json
├── models/
├── reports/
├── notebooks/model_experimentation.ipynb
├── src/
└── tests/
```

## Setup

```bash
cd resumeiq
pip install -r requirements.txt
```

Use Python 3.10 or newer. The deployment runtime is declared in `runtime.txt`.

## Run The App

```bash
streamlit run app.py
```

For hosted or containerized deployment, use:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

The app still runs if model files are missing. In that case, resume parsing, skill extraction, matching, scoring, and recommendations continue, while role prediction displays:

```text
Prediction model is unavailable. Resume matching can continue, but role prediction cannot be generated.
```

## Train The Model

```bash
python src/model_training.py
```

## Run Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The training script compares:

- Multinomial Naive Bayes
- Decision Tree Classifier
- k-Nearest Neighbors
- Support Vector Machine
- Random Forest Classifier as an optional tree-based comparison model

It saves the best model by weighted F1-score to `models/resume_classifier.pkl`, the vectorizer to `models/tfidf_vectorizer.pkl`, and the label encoder to `models/label_encoder.pkl`.

## Dataset Format

Place a CSV at `data/raw/resume_dataset.csv` with these columns:

```csv
Resume,Category
"Experienced Python developer with SQL and Django experience","Python Developer"
"Data analyst skilled in Excel, SQL, Tableau, and statistics","Data Analyst"
```

A tiny sample dataset is included only so the training pipeline can run end to end. Replace it with a larger dataset for meaningful model quality.

## Reports

Training writes:

- `reports/model_comparison.csv`: model name, accuracy, precision, recall, weighted F1-score.
- `reports/confusion_matrix.csv`: confusion matrix for the best selected model.

## Evaluation Metrics

Accuracy measures overall correctness. Precision measures how reliable positive predictions are per class. Recall measures how many true examples are found. Weighted F1-score balances precision and recall while accounting for class support. The confusion matrix shows where the selected model confuses categories.

## Example Output

- Predicted Role: Machine Learning Engineer
- Final Candidate Score: 76.20%
- Resume-Job Match Score: 79.50%
- Skill Match Score: 68.50%
- Candidate Strength: Moderate Match
- Matched Skills: Python, SQL, Machine Learning, Pandas
- Missing Skills: Scikit-learn, Streamlit

## Limitations

- Scanned image-only PDFs may not produce readable text.
- Dictionary-based skill extraction may miss unusual phrasing.
- TF-IDF matching is explainable but not semantic deep understanding.
- Prediction quality depends on dataset size and label quality.
- Scores are decision-support signals, not final hiring decisions.
