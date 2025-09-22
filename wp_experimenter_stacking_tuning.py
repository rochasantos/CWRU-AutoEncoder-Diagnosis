# train_stack_wp.py (refactored with tuned hyperparameters)
# ---------------------------------------------------------
# Train baseline RandomForest and a Stacking Ensemble
# on Wavelet Package (WP) features using tuned hyperparameters.
# Train on:  wp_features/setup_{n}/train.csv
# Test on:   wp_features/setup_{n}/test.csv
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ---------------------------------------------------------------------
# CatBoost optional import
# ---------------------------------------------------------------------
_HAS_CATBOOST = True
try:
    from catboost import CatBoostClassifier
except Exception:
    _HAS_CATBOOST = False
    warnings.warn("[WARN] CatBoost is not installed. Stacking will run without CatBoost.")

# ---------------------------------------------------------------------
# Choose which tuned CatBoost profile to use: "mvs" or "bayesian"
# - "mvs": uses subsample (CPU-safe with bootstrap_type='MVS')
# - "bayesian": NO subsample (Bayesian does not support subsample)
# ---------------------------------------------------------------------
CB_PROFILE = "mvs"       # change to "bayesian" if you want that tuned set


def load_xy(csv_path):
    """Load features (columns starting with 'f') and label/label_str from a CSV file."""
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c.startswith('f')]
    if not feat_cols:
        raise ValueError(f"No feature columns starting with 'f' in {csv_path}")

    if 'label' in df.columns:
        y = df['label'].values
    elif 'label_str' in df.columns:
        y = df['label_str'].values
    else:
        raise ValueError(f"No 'label' or 'label_str' column in {csv_path}")

    X = df[feat_cols].to_numpy(dtype=np.float32)
    return X, y


def make_catboost_from_profile(random_state=42):
    """Build CatBoost with tuned hyperparameters (CPU-safe)."""
    if not _HAS_CATBOOST:
        return None

    if CB_PROFILE.lower() == "mvs":
        # Tuned profile: MVS + subsample
        return CatBoostClassifier(
            depth=10,
            learning_rate=0.20686655612668728,
            l2_leaf_reg=0.41145698487608456,
            n_estimators=1128,
            rsm=0.9788372253755283,
            bootstrap_type="MVS",
            subsample=0.8228543517951666,
            task_type="CPU",
            loss_function="MultiClass",
            random_state=random_state,
            verbose=0
        )
    elif CB_PROFILE.lower() == "bayesian":
        # Tuned profile: Bayesian (NO subsample allowed)
        return CatBoostClassifier(
            depth=7,
            learning_rate=0.04272534565229648,
            l2_leaf_reg=0.006444445726898053,
            n_estimators=1076,
            rsm=0.7425182515387682,
            bootstrap_type="Bayesian",
            task_type="CPU",
            loss_function="MultiClass",
            random_state=random_state,
            verbose=0
        )
    else:
        raise ValueError("CB_PROFILE must be 'mvs' or 'bayesian'.")


def make_base_models(random_state=42):
    """
    Build base learners with tuned hyperparameters:
      - Decision Tree (DT)  [tuned]
      - Logistic Regression (LR) with scaling  [kept simple]
      - SVM (RBF) with scaling and probability=True  [tuned]
      - CatBoost (optional, tuned) if installed
    Return a list of (name, estimator) tuples for StackingClassifier.
    """
    models = []

    # Decision Tree — tuned
    dt = DecisionTreeClassifier(
        criterion="log_loss",
        max_depth=23,
        min_samples_split=17,
        min_samples_leaf=7,
        splitter="random",
        random_state=random_state
    )
    models.append(("dt", dt))

    # Logistic Regression (with scaling) — simple, meta learns how to weight
    lr = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(max_iter=3000, random_state=random_state))
    ])
    models.append(("lr", lr))

    # SVM (with scaling) — tuned (probability=True to enable predict_proba for stacking)
    svm = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svm", SVC(
            kernel="rbf",
            probability=True,
            C=112.19823876932993,
            gamma=0.5802884038732378,
            random_state=random_state
        ))
    ])
    models.append(("svm", svm))

    # CatBoost (tuned). If not available, we skip it.
    if _HAS_CATBOOST:
        cb = make_catboost_from_profile(random_state=random_state)
        models.append(("catboost", cb))

    return models


def evaluate_model(name, clf, X_test, y_test):
    """Predict on test and print metrics."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== RESULTS: {name} =====")
    print(f"Accuracy   : {acc:.4f}")
    print(f"F1-macro   : {f1m:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    return acc, f1m, cm


def main(n_stp):
    root_dir = f"wp_features/setup_{n_stp}"
    train_csv = f"{root_dir}/train.csv"
    test_csv  = f"{root_dir}/test.csv"

    print("[INFO] Loading datasets…")
    X_train, y_train = load_xy(train_csv)
    X_test,  y_test  = load_xy(test_csv)

    print(f"[INFO] Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"[INFO] Test  shape: X={X_test.shape},  y={y_test.shape}")

    # ---------------------------
    # Baseline: Random Forest (tuned)
    # ---------------------------
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=23,
        min_samples_split=14,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
    print("\n[INFO] Training RandomForest (tuned baseline)…")
    rf.fit(X_train, y_train)
    evaluate_model("RandomForest (tuned)", rf, X_test, y_test)

    # ---------------------------
    # Individual base models (tuned where applicable)
    # ---------------------------
    base_models = make_base_models(random_state=42)

    trained_bases = []
    for name, est in base_models:
        print(f"\n[INFO] Training base model: {name} …")
        est.fit(X_train, y_train)
        trained_bases.append((name, est))
        evaluate_model(f"Base::{name}", est, X_test, y_test)

    # ---------------------------
    # Stacking Ensemble (meta-learner = Logistic Regression)
    # ---------------------------
    # Fresh instances for StackingClassifier
    stacking_estimators = make_base_models(random_state=42)

    meta_lr = LogisticRegression(max_iter=4000, random_state=42)
    stack_clf = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=meta_lr,
        stack_method="predict_proba",   # use probabilities from base learners
        passthrough=False,              # only base outputs to meta
        n_jobs=-1
    )

    print("\n[INFO] Training Stacking Ensemble (DT + LR + SVM + CatBoost → LR meta)…")
    stack_clf.fit(X_train, y_train)
    evaluate_model("STACKING (DT+LR+SVM+CatBoost -> LR meta)", stack_clf, X_test, y_test)

    # (Optional) save model(s)
    # import joblib
    # joblib.dump(stack_clf, f"stack_wp_model_setup_{n_stp}.joblib")
    # print(f"[INFO] Saved model to stack_wp_model_setup_{n_stp}.joblib")


if __name__ == "__main__":
    # Exemplo: rode para 1 setup só
    for n in range(10, 10 + 1):
        print(f"\n========== Setup {n} ==========")
        main(n)
