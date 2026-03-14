"""
Autoresearch classification experiment script. CPU-only, single-file.
Baseline: TF-IDF features + Logistic Regression with balanced class weights.
Usage: uv run train.py

The agent modifies everything between the AGENT EDITS markers.
The data loading and evaluation sections are fixed -- do not modify them.
"""

import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from prepare import (
    CACHE_DIR,
    DATASET_FILENAME,
    RANDOM_SEED,
    TEST_SIZE,
    TIME_BUDGET,
    VAL_SIZE,
    assemble_text,
    evaluate_on_val,
    load_data,
)

t_start = time.time()

# ---------------------------------------------------------------------------
# Data loading (fixed -- do not modify)
# ---------------------------------------------------------------------------

# X_train, X_val: pandas Series of assembled job posting text
# y_train, y_val: numpy arrays, 0=real posting, 1=fake posting
X_train, y_train, X_val, y_val = load_data()

print(f"Train: {len(X_train):,} samples  ({y_train.sum():,} fake)")
print(f"Val:   {len(X_val):,} samples  ({y_val.sum():,} fake)")
print(f"Time budget: {TIME_BUDGET}s")

# ---------------------------------------------------------------------------
# AGENT EDITS BELOW THIS LINE
# ---------------------------------------------------------------------------
# Allowed model families (use what's in pyproject.toml dependencies):
#   sklearn: LogisticRegression, RandomForestClassifier, SVC, MLPClassifier,
#            GradientBoostingClassifier, VotingClassifier, StackingClassifier,
#            CalibratedClassifierCV, TfidfVectorizer, CountVectorizer,
#            HashingVectorizer, Pipeline, FeatureUnion
#   xgboost: XGBClassifier
#   lightgbm: LGBMClassifier
#
# You can change:
#   - Vectorizer type and parameters (TF-IDF, count, char n-grams, etc.)
#   - Classifier type and hyperparameters
#   - Preprocessing (text cleaning, stopwords, stemming, etc.)
#   - Feature engineering (text length, metadata features, etc.)
#   - Class weighting / cost-sensitive learning
#   - Threshold selection (default is 0.5, but you can tune it)
#   - Calibration, ensembling, stacking
#
# DO NOT:
#   - Import packages not listed in pyproject.toml
#   - Fit any transformer on val data (data leakage)
#   - Access y_val before computing y_pred (data leakage)
#   - Use the test set (evaluate_on_test) for any decisions


def build_field_features(df_split: pd.DataFrame) -> np.ndarray:
    """Field-level presence and length features for fake job detection."""
    text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
    feats = []
    presence = {}
    for col in text_cols:
        text = df_split[col].fillna("")
        p = (text.str.len() > 0).astype(float)
        presence[col] = p
        feats.append(p)                                        # field presence
        feats.append(np.log1p(text.str.len()).values)          # log length
    # Binary metadata fields
    logo = df_split["has_company_logo"].fillna(0).astype(float)
    questions = df_split["has_questions"].fillna(0).astype(float)
    telecommute = df_split["telecommuting"].fillna(0).astype(float)
    feats.extend([logo.values, questions.values, telecommute.values])
    # Key interaction features: absence of both company_profile and requirements = strong fake signal
    feats.append((presence["company_profile"] * presence["requirements"]).values)
    feats.append((logo * presence["company_profile"]).values)
    feats.append(((1 - presence["company_profile"]) * (1 - logo)).values)
    return np.column_stack(feats).astype(np.float32)


def build_metadata_tokens(df: pd.DataFrame) -> pd.Series:
    """Encode structured job metadata as sparse textual tokens."""
    columns = [
        "telecommuting",
        "has_company_logo",
        "has_questions",
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function",
        "location",
        "department",
        "salary_range",
    ]
    token_parts = []
    for column in columns:
        values = df[column].fillna("missing").astype(str).str.lower().str.strip()
        values = values.str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
        values = values.where(values.ne(""), "missing")
        token_parts.append("__" + column + "_" + values)
    return token_parts[0].str.cat(token_parts[1:], sep=" ")


data_path = os.path.join(CACHE_DIR, "data", DATASET_FILENAME)
df = pd.read_csv(data_path)
X_full = assemble_text(df)
y_full = df["fraudulent"].values.astype(int)

X_trainval_text, _, y_trainval, _, df_trainval, _ = train_test_split(
    X_full,
    y_full,
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_full,
)
val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
X_train_text, X_val_text, _, _, df_train_split, df_val_split = train_test_split(
    X_trainval_text,
    y_trainval,
    df_trainval,
    test_size=val_fraction,
    random_state=RANDOM_SEED,
    stratify=y_trainval,
)

X_train = X_train_text.reset_index(drop=True).str.cat(
    build_metadata_tokens(df_train_split).reset_index(drop=True),
    sep=" ",
)
X_val = X_val_text.reset_index(drop=True).str.cat(
    build_metadata_tokens(df_val_split).reset_index(drop=True),
    sep=" ",
)

# HashingVectorizer with large hash space: unlimited vocabulary, sublinear TF via TfidfTransformer
hash_word = Pipeline([
    ("hv", HashingVectorizer(
        ngram_range=(1, 2),
        n_features=2**20,  # ~1M buckets, minimal collision
        alternate_sign=False,
        norm="l2",
    )),
    ("tfidf", TfidfTransformer(sublinear_tf=True, use_idf=True)),
])

# Standard char TF-IDF (bounded vocabulary for chars)
char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    sublinear_tf=True,
    min_df=1,
    max_features=80_000,
)

X_train_word = hash_word.fit_transform(X_train)
X_val_word = hash_word.transform(X_val)
X_train_char = char_vec.fit_transform(X_train)
X_val_char = char_vec.transform(X_val)

X_train_tfidf = sp.hstack([1.25 * X_train_word, 1.0 * X_train_char], format="csr")
X_val_tfidf = sp.hstack([1.25 * X_val_word, 1.0 * X_val_char], format="csr")

# Explicit field features: presence + log-length per text field + binary metadata
field_train = build_field_features(df_train_split)
field_val = build_field_features(df_val_split)
scaler = MaxAbsScaler()
field_train_scaled = scaler.fit_transform(field_train)
field_val_scaled = scaler.transform(field_val)

X_train_combined = sp.hstack([X_train_tfidf, sp.csr_matrix(field_train_scaled * 5.0)])
X_val_combined = sp.hstack([X_val_tfidf, sp.csr_matrix(field_val_scaled * 5.0)])

# Classifier -- linear margin model for sparse high-dimensional text features
classifier = LinearSVC(
    class_weight={0: 1.0, 1: 20.0},
    max_iter=5000,
    C=1.0,
    dual="auto",
)

classifier.fit(X_train_combined, y_train)

y_prob = classifier.decision_function(X_val_combined)
y_pred = (y_prob >= 0.0).astype(int)

# Tune the decision threshold for macro F1 on the imbalanced validation set.
best_threshold = 0.0
best_score = f1_score(y_val, y_pred, average="macro", zero_division=0)
candidate_thresholds = np.unique(np.round(y_prob, 6))
for threshold in candidate_thresholds:
    candidate_pred = (y_prob >= threshold).astype(int)
    candidate_score = f1_score(y_val, candidate_pred, average="macro", zero_division=0)
    if candidate_score > best_score:
        best_score = candidate_score
        best_threshold = threshold
        y_pred = candidate_pred

# ---------------------------------------------------------------------------
# AGENT EDITS ABOVE THIS LINE
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Evaluation and output (fixed -- do not modify)
# ---------------------------------------------------------------------------

t_train = time.time() - t_start

# Fail fast if training exceeded budget
if t_train > TIME_BUDGET * 1.5:
    print(f"FAIL: training took {t_train:.1f}s, exceeds budget of {TIME_BUDGET}s")
    raise SystemExit(1)

results = evaluate_on_val(y_pred, y_prob)

# Print in parseable format (agent greps these lines from run.log)
print("---")
print(f"val_f1_macro:     {results['f1_macro']:.6f}")
print(f"val_f1_fake:      {results['f1_fake']:.6f}")
print(f"val_precision:    {results['precision_fake']:.6f}")
print(f"val_recall:       {results['recall_fake']:.6f}")
print(f"val_accuracy:     {results['accuracy']:.6f}")
if "roc_auc" in results:
    print(f"val_roc_auc:      {results['roc_auc']:.6f}")
    print(f"val_pr_auc:       {results['pr_auc']:.6f}")
print(f"training_seconds: {t_train:.1f}")
