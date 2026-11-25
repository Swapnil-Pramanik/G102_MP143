"""PSI per numeric column, MAE per station_type, retrain v2, promotion rule."""
import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

from ev_energy.cleaning import clean, load_raw
from ev_energy.config import load_params
from ev_energy.train import data_sha256, split_xy


# ----------------------------------------------------------------------------
# 1. PSI (Population Stability Index)
# ----------------------------------------------------------------------------
def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(train: pd.DataFrame, later: pd.DataFrame, numeric_cols: list[str], bins: int) -> dict:
    return {col: psi(train[col], later[col], bins) for col in numeric_cols}


# ----------------------------------------------------------------------------
# 2. MAE per group
# ----------------------------------------------------------------------------
def mae_by_group(model, df: pd.DataFrame, params: dict) -> dict:
    group_col = params["data"]["group_column"]
    result = {}
    for g in sorted(df[group_col].unique()):
        part = df[df[group_col] == g]
        X, y = split_xy(part, params)
        result[g] = float(mean_absolute_error(y, model.predict(X)))
    return result


# ----------------------------------------------------------------------------
# 3. Split later.csv by date
# ----------------------------------------------------------------------------
def split_later(later: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_col = params["data"]["date_column"]
    later = later.sort_values(date_col).reset_index(drop=True)
    cut = int(len(later) * params["monitoring"]["later_train_fraction"])
    return later.iloc[:cut], later.iloc[cut:]


# ----------------------------------------------------------------------------
# 4. Promotion rule
# ----------------------------------------------------------------------------
def decide(mae_v1: float, mae_v2: float, min_improvement: float) -> str:
    return "promote" if mae_v2 <= mae_v1 * (1 - min_improvement) else "keep"


def main() -> None:
    params = load_params()
    d, m = params["data"], params["mlflow"]

    # 1. MLflow setup
    mlflow.set_tracking_uri(m["tracking_uri"])
    mlflow.set_experiment(m["experiment_name"])
    name = m["registered_model_name"]

    # 2. Data
    train = clean(load_raw(d["train"]), params)
    later = clean(load_raw(d["later"]), params)

    # 3. PSI
    psi_scores = psi_table(train, later, params["features"]["numeric"],
                           params["monitoring"]["psi_bins"])
    most_drifted = max(psi_scores, key=psi_scores.get)

    # 4. v1 = current champion, MAE per group on all of later.csv
    #    Loaded from models/model.joblib (the v1 champion saved by train.py):
    #    the registry's model files live only on the laptop that ran train.py.
    v1 = joblib.load(params["paths"]["model"])
    groups = mae_by_group(v1, later, params)
    worst = max(groups, key=groups.get)

    # 5. Retrain v2 on train + first 70% of later
    first70, last30 = split_later(later, params)
    combined = pd.concat([train, first70], ignore_index=True)
    X_fit, y_fit = split_xy(combined, params)
    X_check, y_check = split_xy(last30, params)

    with mlflow.start_run(run_name="v2"):
        v2 = clone(v1)                     # same pipeline + settings, untrained
        v2.fit(X_fit, y_fit)

        # 6. Compare both on the last 30%
        mae_v1 = float(mean_absolute_error(y_check, v1.predict(X_check)))
        mae_v2 = float(mean_absolute_error(y_check, v2.predict(X_check)))
        decision = decide(mae_v1, mae_v2,
                          params["monitoring"]["promotion_min_improvement"])

        mlflow.log_param("data_sha256", data_sha256(d["train"]))
        mlflow.log_metric("mae_check", mae_v2)
        mlflow.log_metric("mae_v1_check", mae_v1)

        # 7. Register v2 in both cases
        info = mlflow.sklearn.log_model(v2, name="model",
                                        registered_model_name=name,
                                        serialization_format="cloudpickle")

    if decision == "promote":
        MlflowClient().set_registered_model_alias(
            name, "champion", info.registered_model_version)
        joblib.dump(v2, params["paths"]["model"])

    # 8. Numbers for findings.yaml
    print("psi:", {k: round(v, 4) for k, v in psi_scores.items()})
    print("most_drifted_feature:", most_drifted)
    print("mae_by_group:", {k: round(v, 4) for k, v in groups.items()})
    print("worst_served_group:", worst)
    print("mae_v1_on_check:", round(mae_v1, 4))
    print("mae_v2_on_check:", round(mae_v2, 4))
    print("decision:", decision)

if __name__ == "__main__":
    main()
