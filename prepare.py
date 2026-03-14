"""
One-time data preparation for autoresearch text classification experiments.
Downloads the Real/Fake Job Posting dataset, assembles text features,
and creates fixed stratified train/val/test splits.

Usage:
    python prepare.py                  # full prep
    python prepare.py --cache-dir /tmp # use a different cache directory

Data and splits are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 120       # training time budget in seconds (2 minutes)
RANDOM_SEED = 42        # fixed seed for all splits and reproducibility
TEST_SIZE = 0.15        # fraction of data held out as final test set
VAL_SIZE = 0.15         # fraction of remaining data used as validation set
PRIMARY_METRIC = "f1_macro"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
SPLITS_DIR = os.path.join(CACHE_DIR, "splits")

# Mirror URL for the dataset CSV (Kaggle via HuggingFace datasets mirror)
# Original: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
DATASET_FILENAME = "fake_job_postings.csv"
DATASET_HF_URL = (
    "https://huggingface.co/datasets/victor/real-or-fake-job-posting-prediction"
    "/resolve/main/fake_job_postings.csv"
)

# Text columns to concatenate into a single document per job posting
TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
LABEL_COLUMN = "fraudulent"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_dataset(cache_dir: str = CACHE_DIR) -> str:
    """Download the fake job postings CSV if not already cached. Returns file path."""
    data_dir = os.path.join(cache_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, DATASET_FILENAME)

    if os.path.exists(filepath):
        print(f"Data: already downloaded at {filepath}")
        return filepath

    # Try kagglehub first (preferred, handles auth automatically)
    try:
        import kagglehub
        path = kagglehub.dataset_download(
            "shivamb/real-or-fake-fake-jobposting-prediction"
        )
        # kagglehub downloads to its own cache; find the CSV
        for root, _, files in os.walk(path):
            for fname in files:
                if fname.endswith(".csv"):
                    import shutil
                    src = os.path.join(root, fname)
                    shutil.copy(src, filepath)
                    print(f"Data: downloaded via kagglehub to {filepath}")
                    return filepath
    except Exception as e:
        print(f"Data: kagglehub unavailable ({e}), trying direct download...")

    # Fallback: direct HTTP download
    print(f"Data: downloading {DATASET_FILENAME} from HuggingFace mirror...")
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(DATASET_HF_URL, stream=True, timeout=60)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Data: downloaded to {filepath}")
            return filepath
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                import time
                import time as _time
                _time.sleep(2 ** attempt)

    print(
        f"\nERROR: Could not download the dataset automatically.\n"
        f"Please download it manually from:\n"
        f"  https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction\n"
        f"and place '{DATASET_FILENAME}' in:\n"
        f"  {data_dir}\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Text assembly
# ---------------------------------------------------------------------------

def assemble_text(df: pd.DataFrame) -> pd.Series:
    """
    Combine multiple text columns into a single document per row.
    Missing values are replaced with empty strings.
    Columns are joined with a space separator.
    """
    parts = []
    for col in TEXT_COLUMNS:
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    return parts[0].str.cat(parts[1:], sep=" ").str.strip()


# ---------------------------------------------------------------------------
# Split creation
# ---------------------------------------------------------------------------

def create_splits(filepath: str, splits_dir: str = SPLITS_DIR) -> None:
    """
    Load the CSV, assemble text, and create stratified train/val/test splits.
    Saves splits to disk as pickles so they are deterministic across runs.
    """
    os.makedirs(splits_dir, exist_ok=True)
    splits_pkl = os.path.join(splits_dir, "splits.pkl")

    if os.path.exists(splits_pkl):
        print(f"Splits: already created at {splits_pkl}")
        return

    print("Splits: loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Splits: loaded {len(df):,} rows, {df[LABEL_COLUMN].sum():,} fake postings")

    X = assemble_text(df)
    y = df[LABEL_COLUMN].values.astype(int)

    # First split: hold out test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Second split: validation from remaining
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )

    splits = {
        "X_train": X_train.reset_index(drop=True),
        "y_train": y_train,
        "X_val": X_val.reset_index(drop=True),
        "y_val": y_val,
        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test,
    }

    with open(splits_pkl, "wb") as f:
        pickle.dump(splits, f)

    print(f"Splits: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
    print(f"Splits: fake rate train={y_train.mean():.3f}  val={y_val.mean():.3f}  test={y_test.mean():.3f}")
    print(f"Splits: saved to {splits_pkl}")


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

_splits_cache = None

def _load_splits() -> dict:
    """Load splits from disk, caching in memory after first load."""
    global _splits_cache
    if _splits_cache is not None:
        return _splits_cache
    splits_pkl = os.path.join(SPLITS_DIR, "splits.pkl")
    if not os.path.exists(splits_pkl):
        print(
            f"ERROR: Splits not found at {splits_pkl}.\n"
            f"Run 'python prepare.py' first to download data and create splits."
        )
        sys.exit(1)
    with open(splits_pkl, "rb") as f:
        _splits_cache = pickle.load(f)
    return _splits_cache


def load_data():
    """
    Returns (X_train, y_train, X_val, y_val).

    X_train, X_val: pandas Series of assembled text strings
    y_train, y_val: numpy arrays of int labels (0=real, 1=fake)

    The test set is withheld -- use evaluate_on_test() only for final reporting,
    never for making decisions about which experiments to keep.
    """
    splits = _load_splits()
    return (
        splits["X_train"],
        splits["y_train"],
        splits["X_val"],
        splits["y_val"],
    )


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE -- these are the fixed metrics)
# ---------------------------------------------------------------------------

def _evaluate(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute all classification metrics. Fixed -- do not modify.

    Args:
        y_true: ground-truth labels (0=real, 1=fake)
        y_pred: hard predictions
        y_prob: predicted probability of class 1 (fake), optional

    Returns:
        dict with keys: f1_macro, f1_fake, precision_fake, recall_fake,
                        accuracy, and optionally roc_auc, pr_auc
    """
    results = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_fake": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        results["roc_auc"] = roc_auc_score(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        results["pr_auc"] = auc(rec, prec)
    return results


def evaluate_on_val(y_pred, y_prob=None) -> dict:
    """
    Evaluate hard predictions (and optionally probabilities) against the
    fixed validation set. Use this in train.py for experiment optimization.

    Returns a dict of metrics (see _evaluate for keys).
    """
    splits = _load_splits()
    y_val = splits["y_val"]
    return _evaluate(y_val, y_pred, y_prob)


def evaluate_on_test(y_pred, y_prob=None) -> dict:
    """
    Evaluate against the held-out test set.

    DO NOT use this to decide which experiments to keep -- it must only be
    called for final reporting after all experimentation is complete.
    Using test scores to guide decisions constitutes data leakage.

    Returns a dict of metrics (see _evaluate for keys).
    """
    splits = _load_splits()
    y_test = splits["y_test"]
    return _evaluate(y_test, y_pred, y_prob)


def print_metrics(results: dict, prefix: str = "") -> None:
    """Pretty-print a metrics dict."""
    label = f"{prefix}_" if prefix else ""
    print(f"{label}f1_macro:      {results['f1_macro']:.6f}")
    print(f"{label}f1_fake:       {results['f1_fake']:.6f}")
    print(f"{label}precision:     {results['precision_fake']:.6f}")
    print(f"{label}recall:        {results['recall_fake']:.6f}")
    print(f"{label}accuracy:      {results['accuracy']:.6f}")
    if "roc_auc" in results:
        print(f"{label}roc_auc:       {results['roc_auc']:.6f}")
        print(f"{label}pr_auc:        {results['pr_auc']:.6f}")


# ---------------------------------------------------------------------------
# Main (one-time setup)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data and splits for autoresearch classification"
    )
    parser.add_argument(
        "--cache-dir",
        default=CACHE_DIR,
        help="Cache directory for data and splits (default: ~/.cache/autoresearch)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    data_dir = os.path.join(cache_dir, "data")
    splits_dir = os.path.join(cache_dir, "splits")

    print(f"Cache directory: {cache_dir}")
    print()

    # Step 1: Download dataset
    filepath = download_dataset(cache_dir=cache_dir)
    print()

    # Step 2: Create splits
    create_splits(filepath, splits_dir=splits_dir)
    print()

    # Step 3: Sanity check by loading splits
    # Redirect module-level globals to custom cache dir so load_data() works
    SPLITS_DIR = splits_dir
    _splits_cache = None  # clear cache
    X_train, y_train, X_val, y_val = load_data()
    splits = _load_splits()
    print(f"Sanity check:")
    print(f"  X_train: {len(X_train):,} samples, first text: {X_train.iloc[0][:80]!r}")
    print(f"  y_train: {y_train.sum():,} fake out of {len(y_train):,}")
    print(f"  X_val:   {len(X_val):,} samples")
    print(f"  X_test:  {len(splits['X_test']):,} samples (withheld)")
    print()
    print("Done! Ready to train. Run: uv run train.py")
