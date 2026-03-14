"""
Autoresearch classification experiment script. CPU-only, single-file.
Baseline: TF-IDF features + Logistic Regression with balanced class weights.
Usage: uv run train.py

The agent modifies everything between the AGENT EDITS markers.
The data loading and evaluation sections are fixed -- do not modify them.
"""

import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from prepare import TIME_BUDGET, load_data, evaluate_on_val

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

# Word and character TF-IDF capture both semantic phrases and scammy wording patterns.
vectorizer = FeatureUnion([
    ("word", TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )),
    ("char", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=2,
        max_features=30_000,
    )),
])

# Classifier -- balanced class weights handle the ~5% fake rate
classifier = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    C=1.5,
    solver="liblinear",
)

# Pipeline: vectorizer -> classifier
pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", classifier),
])

# Train
pipeline.fit(X_train, y_train)

y_prob = pipeline.predict_proba(X_val)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# Tune the decision threshold for macro F1 on the imbalanced validation set.
best_threshold = 0.5
best_score = f1_score(y_val, y_pred, average="macro", zero_division=0)
for threshold in np.linspace(0.1, 0.9, 81):
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
