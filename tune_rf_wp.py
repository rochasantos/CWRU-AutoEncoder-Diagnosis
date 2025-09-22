# train_rf_wp_optuna_sgkfold_all_setups.py
# ---------------------------------------------------------
# Random Forest + Optuna with StratifiedGroupKFold (4 folds)
# GLOBAL tuning across setups 1..10:
#   - Objective = mean CV accuracy across ALL setups (maximize)
# After tuning, train per-setup on full train and evaluate on test.
# ---------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import optuna

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------------- helpers ----------------------
def _extract_acquisition_from_path(p):
    """
    Extract acquisition token {acq} from filename: {label}_{acq}_{mode}_{idx}.npy
    Example: data/.../B_15_1_95.npy -> 15
    """
    fname = Path(str(p)).name
    parts = fname.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected file name format for grouping: {fname}")
    acq = parts[1]
    try:
        return int(acq)
    except ValueError:
        return acq  # fallback to string if not numeric


def load_xy_groups(csv_path):
    """
    Load X, y, groups from CSV:
      - features: columns starting with 'f'
      - labels: 'label' (int) or 'label_str' (mapped to int)
      - groups: acquisition parsed from 'path'
    """
    df = pd.read_csv(csv_path)

    if "path" not in df.columns:
        raise ValueError(f"CSV must contain 'path' column: {csv_path}")

    feat_cols = [c for c in df.columns if c.startswith("f")]
    if not feat_cols:
        raise ValueError(f"No feature columns starting with 'f' in {csv_path}")

    if "label" in df.columns:
        y = df["label"].values
    elif "label_str" in df.columns:
        mapping = {s: i for i, s in enumerate(sorted(df["label_str"].astype(str).unique()))}
        y = df["label_str"].map(mapping).values
    else:
        raise ValueError(f"No 'label' or 'label_str' column in {csv_path}")

    groups = df["path"].apply(_extract_acquisition_from_path).values
    X = df[feat_cols].to_numpy(dtype=np.float32)
    return X, y, groups


# ---------------------- objective factory ----------------------
def make_objective(all_train_data, n_splits=4, seed=42):
    """
    Create an Optuna objective:
      - Build an RF pipeline (hyperparams via set_params with model__*)
      - Compute CV accuracy per setup using StratifiedGroupKFold(groups)
      - Return the MEAN accuracy across setups (maximize)
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial):
        params = dict(
            model__n_estimators=trial.suggest_int("model__n_estimators", 100, 800, step=50),
            model__max_depth=trial.suggest_int("model__max_depth", 5, 40),
            model__min_samples_split=trial.suggest_int("model__min_samples_split", 2, 20),
            model__min_samples_leaf=trial.suggest_int("model__min_samples_leaf", 1, 10),
            model__max_features=trial.suggest_categorical("model__max_features", ["sqrt", "log2", None]),
            model__bootstrap=trial.suggest_categorical("model__bootstrap", [True, False]),
        )

        per_setup_scores = []
        for pack in all_train_data:
            X_tr, y_tr, g_tr = pack["X"], pack["y"], pack["groups"]

            pipe = Pipeline([("model", RandomForestClassifier(random_state=42, n_jobs=-1))])
            pipe.set_params(**params)

            acc_mean = cross_val_score(
                pipe, X_tr, y_tr,
                scoring="accuracy",
                cv=cv,
                n_jobs=-1,
                groups=g_tr
            ).mean()

            per_setup_scores.append(acc_mean)

        # Mean CV accuracy across all setups
        return float(np.mean(per_setup_scores))

    return objective


# ---------------------- main ----------------------
def main():
    base_dir = Path("wp_features")
    setup_ids = list(range(1, 10 + 1))  # setup_1 .. setup_10

    # ---------- preload TRAIN/TEST for all setups ----------
    all_train_data = []   # list of {"setup": i, "X":..., "y":..., "groups":...}
    all_test_data  = []   # list of {"setup": i, "X":..., "y":...}
    for i in setup_ids:
        root = base_dir / f"setup_{i}"
        train_csv = root / "train.csv"
        test_csv  = root / "test.csv"

        X_tr, y_tr, g_tr = load_xy_groups(train_csv)
        X_te, y_te, _    = load_xy_groups(test_csv)  # groups not needed for test

        all_train_data.append({"setup": i, "X": X_tr, "y": y_tr, "groups": g_tr})
        all_test_data.append({"setup": i, "X": X_te, "y": y_te})

    print(f"[INFO] Loaded {len(all_train_data)} setups for global tuning.")

    # ---------- Optuna global study ----------
    objective = make_objective(all_train_data, n_splits=4, seed=42)
    study = optuna.create_study(direction="maximize", study_name="rf_wp_optuna_sgkfold_all_setups")
    study.optimize(objective, n_trials=60)  # adjust as needed

    print("\n===== OPTUNA BEST (GLOBAL CV MEAN) =====")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
    print(f"Global CV mean accuracy: {study.best_value:.4f}")

    # ---------- Train best model per setup and evaluate ----------
    bp = study.best_params
    per_setup_acc = []

    print("\n===== TEST RESULTS PER SETUP =====")
    for pack_train, pack_test in zip(all_train_data, all_test_data):
        sid = pack_train["setup"]
        X_tr, y_tr = pack_train["X"], pack_train["y"]
        X_te, y_te = pack_test["X"], pack_test["y"]

        model = Pipeline([("model", RandomForestClassifier(random_state=42, n_jobs=-1))])
        model.set_params(**bp)

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        cm  = confusion_matrix(y_te, y_pred)
        report = classification_report(y_te, y_pred, digits=4)

        per_setup_acc.append(acc)
        print(f"\n--- setup_{sid} ---")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix (rows=true, cols=pred):")
        print(cm)
        print("Classification Report:")
        print(report)

    print("\n===== GLOBAL TEST SUMMARY =====")
    print(f"Mean accuracy over setups: {np.mean(per_setup_acc):.4f}")
    print(f"Std  accuracy over setups: {np.std(per_setup_acc):.4f}")

    # (Optional) save artifacts
    # import json
    # with open("rf_wp_optuna_all_setups_best_params.json", "w") as f:
    #     json.dump(study.best_params, f, indent=2)
    # print("[INFO] Saved best params JSON.")


if __name__ == "__main__":
    main()
