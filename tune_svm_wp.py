# tune_svm_wp_sgkfold_all_setups.py
# ---------------------------------------------------------
# SVM (RBF) + Optuna com StratifiedGroupKFold (4 folds)
# Tuning GLOBAL nos setups 1..10:
#   - Objetivo = média das acurácias de CV (por-setup) ao longo dos 10 setups
# Após o tuning, treina por-setup com os melhores hiperparâmetros globais
# e avalia no test de cada setup.
# ---------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import optuna

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -------------- helpers --------------
def _extract_acquisition_from_path(p: str):
    """
    Extrai o token de aquisição {acq} de {label}_{acq}_{mode}_{idx}.npy
    Ex.: ".../B_15_1_95.npy" -> 15
    """
    fname = Path(str(p)).name
    parts = fname.split("_")
    if len(parts) < 4:
        raise ValueError(f"Formato inesperado de filename para agrupar: {fname}")
    acq = parts[1]
    try:
        return int(acq)
    except ValueError:
        return acq  # fallback string


def load_xy_groups(csv_path: Path):
    """
    Carrega X, y, groups de um CSV:
      - features: colunas começando com 'f'
      - rótulos: 'label' (int) OU 'label_str' (mapeada para int)
      - grupos: aquisição extraída de 'path'
    """
    df = pd.read_csv(csv_path)

    if "path" not in df.columns:
        raise ValueError(f"CSV precisa conter a coluna 'path': {csv_path}")

    feat_cols = [c for c in df.columns if c.startswith("f")]
    if not feat_cols:
        raise ValueError(f"Nenhuma coluna de features iniciando em 'f' em {csv_path}")

    if "label" in df.columns:
        y = df["label"].values
    elif "label_str" in df.columns:
        mapping = {s: i for i, s in enumerate(sorted(df["label_str"].astype(str).unique()))}
        y = df["label_str"].map(mapping).values
    else:
        raise ValueError(f"CSV precisa ter 'label' ou 'label_str': {csv_path}")

    groups = df["path"].apply(_extract_acquisition_from_path).values
    X = df[feat_cols].to_numpy(dtype=np.float32)
    return X, y, groups


# -------------- objective factory --------------
def make_objective(all_train_data, n_splits=4, seed=42):
    """
    Cria o objetivo do Optuna:
      - Para cada trial, avalia SVM(RBF) com StratifiedGroupKFold em CADA setup
      - Retorna a média das acurácias de CV entre os setups
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1e1, log=True)

        per_setup_acc = []
        for pack in all_train_data:
            X_tr, y_tr, g_tr = pack["X"], pack["y"], pack["groups"]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=seed))
            ])

            acc_mean = cross_val_score(
                pipe, X_tr, y_tr,
                scoring="accuracy",
                cv=cv,
                n_jobs=-1,
                groups=g_tr
            ).mean()
            per_setup_acc.append(acc_mean)

        # média global entre setups
        return float(np.mean(per_setup_acc))

    return objective


# -------------- main --------------
def main():
    base_dir = Path("wp_features")
    setup_ids = list(range(1, 10 + 1))  # setup_1 .. setup_10

    # Pré-carrega dados de train/test de todos os setups
    all_train_data = []  # [{"setup": i, "X":..., "y":..., "groups":...}, ...]
    all_test_data  = []  # [{"setup": i, "X":..., "y":...}, ...]
    for i in setup_ids:
        root = base_dir / f"setup_{i}"
        train_csv = root / "train.csv"
        test_csv  = root / "test.csv"

        X_tr, y_tr, g_tr = load_xy_groups(train_csv)
        X_te, y_te, _    = load_xy_groups(test_csv)  # groups não são usados no teste

        all_train_data.append({"setup": i, "X": X_tr, "y": y_tr, "groups": g_tr})
        all_test_data.append({"setup": i, "X": X_te, "y": y_te})

    print(f"[INFO] Carregados {len(all_train_data)} setups para tuning global.")

    # Optuna: objetivo = média das acurácias de CV entre os 10 setups
    objective = make_objective(all_train_data, n_splits=4, seed=42)
    study = optuna.create_study(direction="maximize", study_name="svm_wp_sgkfold_all_setups")
    study.optimize(objective, n_trials=80)  # ajuste n_trials conforme recursos/tempo

    print("\n===== OPTUNA BEST (GLOBAL CV MEAN) =====")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
    print(f"Global CV mean accuracy: {study.best_value:.4f}")

    # Treina por-setup no train completo com os melhores hiperparâmetros globais
    bp = study.best_params
    per_setup_acc = []

    print("\n===== TEST RESULTS PER SETUP =====")
    for pack_train, pack_test in zip(all_train_data, all_test_data):
        sid = pack_train["setup"]
        X_tr, y_tr = pack_train["X"], pack_train["y"]
        X_te, y_te = pack_test["X"], pack_test["y"]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=bp["C"],
                gamma=bp["gamma"],
                probability=True,
                random_state=42
            ))
        ])

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        cm  = confusion_matrix(y_te, y_pred)
        rep = classification_report(y_te, y_pred, digits=4)

        per_setup_acc.append(acc)
        print(f"\n--- setup_{sid} ---")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix (rows=true, cols=pred):")
        print(cm)
        print("Classification Report:")
        print(rep)

    print("\n===== GLOBAL TEST SUMMARY =====")
    print(f"Mean accuracy over setups: {np.mean(per_setup_acc):.4f}")
    print(f"Std  accuracy over setups: {np.std(per_setup_acc):.4f}")


if __name__ == "__main__":
    main()
