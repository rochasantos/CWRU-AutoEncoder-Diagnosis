# train_rf_wp_gridsearch.py
# ---------------------------------------------------------
# Random Forest + GridSearchCV (cv=5) em WaveletPackage features
# Train CSV : wp_features/setup_1/train.csv
# Test  CSV : wp_features/setup_1/test.csv
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV


def load_xy(csv_path):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c.startswith('f')]
    if not feat_cols:
        raise ValueError(f"No feature columns starting with 'f' in {csv_path}")

    if 'label' in df.columns:
        y = df['label'].values
    elif 'label_str' in df.columns:
        mapping = {s: i for i, s in enumerate(sorted(df['label_str'].astype(str).unique()))}
        y = df['label_str'].map(mapping).values
    else:
        raise ValueError(f"No 'label' or 'label_str' column in {csv_path}")

    X = df[feat_cols].to_numpy(dtype=np.float32)
    return X, y


def main():
    root_dir = Path("wp_features/setup_8")
    train_csv = root_dir / "train.csv"
    test_csv  = root_dir / "test.csv"

    print("[INFO] Loading datasets…")
    X_train, y_train = load_xy(train_csv)
    X_test,  y_test  = load_xy(test_csv)

    print(f"[INFO] Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"[INFO] Test  shape: X={X_test.shape},  y={y_test.shape}")

    # --- Definição do modelo base ---
    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # --- Espaço de busca (ajuste conforme necessário) ---
    param_grid = {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    # --- Cross-validation estratificada ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- GridSearchCV (usando 'accuracy' pois as classes estão balanceadas) ---
    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        scoring="accuracy",      # ou 'f1_macro' se preferir
        cv=cv,
        n_jobs=-1,
        verbose=3,
        refit=True               # re-treina no full train com os melhores params
    )

    print("[INFO] Running GridSearchCV…")
    grid.fit(X_train, y_train)

    print("\n===== GRID BEST (CV) =====")
    print("Best params:", grid.best_params_)
    print(f"Best CV score (accuracy): {grid.best_score_:.4f}")

    # --- Avaliação no conjunto de teste ---
    best_model = grid.best_estimator_
    print("\n[INFO] Evaluating best model on test…")
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    cm  = confusion_matrix(y_test, y_pred)

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy   : {acc:.4f}")
    print(f"F1-macro   : {f1m:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # (Opcional) salvar o melhor modelo
    # import joblib
    # joblib.dump(best_model, "rf_wp_best_grid.joblib")
    # print("[INFO] Saved model to rf_wp_best_grid.joblib")


if __name__ == "__main__":
    main()

