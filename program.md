# autoresearch — text classification

This is an experiment to have the LLM do its own research on a text classification task:
**Real vs. Fake Job Posting Detection** (binary classification, imbalanced ~5% positive rate).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar13`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data loading, splits, evaluation. Do not modify.
   - `train.py` — the file you modify. Feature engineering, classifier, hyperparameters.
4. **Verify data exists**: Check that `~/.cache/autoresearch/splits/splits.pkl` exists. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row:
   ```
   commit	val_f1_macro	val_pr_auc	status	description
   ```
   The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Task description

The dataset is the [Real/Fake Job Posting Prediction dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) (~18,000 job descriptions, ~800 labeled fake). The label is `fraudulent` (0=real, 1=fake).

Text is assembled from: `title`, `company_profile`, `description`, `requirements`, `benefits` — joined into a single string.

The class imbalance (~5% fake) is the core challenge. A naive classifier predicting "all real" gets ~95% accuracy but is useless. **Optimize for `val_f1_macro`**, which weighs both classes equally and is a much better proxy for real detection quality.

## Experimentation

Each experiment runs on CPU. The training script runs for a **fixed time budget of 2 minutes** (wall clock). You launch it simply as: `uv run train.py`.

**What you CAN do:**

- Modify `train.py` — this is the only file you edit. Everything between the AGENT EDITS markers is fair game: feature engineering, classifier choice, hyperparameters, preprocessing, threshold tuning, ensembling, etc.

**What you CANNOT do:**

- Modify `prepare.py`. It is read-only. It contains the fixed splits, evaluation function, and time budget constant.
- Install new packages or add dependencies. Use only what's in `pyproject.toml`:
  - `scikit-learn`: LogisticRegression, RandomForestClassifier, SVC, MLPClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, CalibratedClassifierCV, TfidfVectorizer, CountVectorizer, HashingVectorizer, Pipeline, FeatureUnion
  - `xgboost`: XGBClassifier
  - `lightgbm`: LGBMClassifier
- Modify the evaluation harness. `evaluate_on_val` and `evaluate_on_test` in `prepare.py` are the ground truth metrics.
- Introduce **data leakage**: never fit transformers on val data, never use val labels before making predictions, never access the test set.
- Use `evaluate_on_test()` to guide decisions. It exists for final reporting only. If you call it during the experiment loop, that's data leakage.

**The goal: get the highest `val_f1_macro`.** Secondary metric: `val_pr_auc` (precision-recall AUC, also important for imbalanced classification).

**Simplicity criterion**: All else being equal, simpler is better. A 0.001 f1_macro improvement that adds 40 lines of complex code? Probably not worth it. A 0.001 improvement from changing one hyperparameter? Definitely keep. An improvement of ~0 but simpler code? Keep.

**Minimum improvement threshold**: Only keep changes that either:
- Improve `val_f1_macro` by at least **0.002** (meaningful signal above noise), or
- Simplify the code with no degradation, or
- Substantially improve a secondary metric (pr_auc, recall at fixed precision) without hurting f1_macro

**The first run**: Your very first run should always be to establish the baseline, so run the training script as-is before making any changes.

## Search space

Promising directions (not exhaustive):

**Feature engineering:**
- TF-IDF: `max_features`, `ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `analyzer='char_wb'`
- Character n-grams alongside word n-grams (FeatureUnion)
- Text length features: character count, word count, sentence count (as numeric columns)
- Metadata features: presence of company logo, presence of questions field, `has_company_logo`, `required_experience`, `required_education`, `employment_type`, `industry`, `function` columns — note these are NOT currently in X_train (only text is), so you would need to modify the data loading section of train.py to include them. You are allowed to load the splits and merge in metadata as long as you do not use y_val before predicting.

**Classifiers:**
- LogisticRegression: `C`, `solver`, `class_weight`, `penalty`
- LinearSVC: fast and strong for high-dimensional TF-IDF
- XGBClassifier: `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight` for imbalance
- LGBMClassifier: `is_unbalance=True`, `num_leaves`, `learning_rate`
- RandomForestClassifier: `n_estimators`, `class_weight='balanced_subsample'`
- MLPClassifier: `hidden_layer_sizes`, `alpha`
- Ensembling: VotingClassifier (soft voting), StackingClassifier

**Class imbalance:**
- `class_weight='balanced'` (already in baseline)
- SMOTE via imbalanced-learn (NOT available — not in pyproject.toml)
- `scale_pos_weight` for XGBoost
- Threshold tuning: find the decision threshold that maximizes f1_macro on val

**Calibration:**
- `CalibratedClassifierCV(base_estimator, cv='prefit')` after fitting

## Output format

Once the script finishes it prints:

```
---
val_f1_macro:     0.823456
val_f1_fake:      0.712345
val_precision:    0.801234
val_recall:       0.641234
val_accuracy:     0.978901
val_roc_auc:      0.956789
val_pr_auc:       0.812345
training_seconds: 12.3
```

Extract the key metrics:

```bash
grep "^val_f1_macro:\|^val_pr_auc:" run.log
```

If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to diagnose.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_f1_macro	val_pr_auc	status	description
```

1. git commit hash (short, 7 chars)
2. val_f1_macro achieved (e.g. 0.823456) — use 0.000000 for crashes
3. val_pr_auc achieved (e.g. 0.812345) — use 0.000000 for crashes or if not printed
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_f1_macro	val_pr_auc	status	description
a1b2c3d	0.823456	0.812345	keep	baseline TF-IDF LR balanced
b2c3d4e	0.831200	0.819876	keep	increase max_features to 50000
c3d4e5f	0.821000	0.808000	discard	switch to CountVectorizer (worse)
d4e5f6g	0.000000	0.000000	crash	XGBClassifier OOM on full features
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar13`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit
2. Tune `train.py` with an experimental idea by directly editing the code
3. `git commit`
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything)
5. Read out the results: `grep "^val_f1_macro:\|^val_pr_auc:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the traceback. Attempt a fix. If you can't fix it after a few attempts, give up.
7. Record the results in `results.tsv` (NOTE: do not commit `results.tsv`, leave it untracked by git)
8. If `val_f1_macro` improved by ≥ 0.002 (or simplified code), you "advance" the branch, keeping the git commit
9. If `val_f1_macro` is worse or improved by < 0.002 with added complexity, `git reset` back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. You are advancing the branch so you can iterate from the current best state.

**Timeout**: Each experiment should take ≤ 2 minutes (the TIME_BUDGET). Classifiers train fast. If a run exceeds 5 minutes wall clock, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (a bug, import error, etc.), use your judgment. If it's something easy to fix (typo, missing parameter), fix it and re-run. If the idea is fundamentally broken (e.g., memory error at large scale), log "crash" and move on.

**Overfitting to val**: If you notice val_f1_macro improving by tiny amounts across many experiments, be skeptical. Prefer ideas that improve it by ≥ 0.002 cleanly, or that improve pr_auc substantially. Avoid micro-tuning a single hyperparameter 20 times.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — combine previous near-misses, try more radical changes, try a completely different model family, try ensembling the best few approaches. The loop runs until the human interrupts you, period.

Since each experiment takes ~30 seconds to a few minutes, you can run many experiments per hour. A user might leave you running while they sleep and wake up to a full experiment log.
