"""Train v1 with MLflow tracking, register the best model, set @champion.

Run from the repo root (after Aditya Singh's cleaning.py works):
    python -m ev_energy.train

What must exist afterwards (the marking tool checks these):
  - mlflow.db in the repo root with 3 or more runs
  - every run has a PARAMETER  data_sha256 = sha256 of the ORIGINAL train.csv
  - every run has a METRIC whose name contains "mae"
  - a registered model (name from params.yaml), version 1, alias "champion"
  - models/model.joblib  (the champion, loaded by the API)

THE CONTRACT (Swapnil's monitor.py and the API use these, keep the names):
    data_sha256(path)                     -> str
    build_pipeline(params, model_kind, model_params) -> unfitted sklearn Pipeline
    The fitted Pipeline takes a DataFrame with the 8 feature columns
    (params["features"]["numeric"] + params["features"]["categorical"])
    and returns predictions of energy_kwh.
"""
import hashlib
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ev_energy.cleaning import clean, load_raw
from ev_energy.config import load_params

# The model types you can try. Add more if you like (any sklearn regressor).
MODELS = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def data_sha256(path: str) -> str:
    """SHA-256 of the ORIGINAL file (handbook section 8)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_pipeline(params: dict, model_kind: str, model_params: dict) -> Pipeline:
    """Preprocessing + model in ONE Pipeline, so the API can call .predict(df)."""
    num_cols = params["features"]["numeric"]
    cat_cols = params["features"]["categorical"]

    num_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    model_cls = MODELS[model_kind]
    model = model_cls(**model_params)

    return Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])


def split_xy(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X with only the 8 feature columns, y = the target column)."""
    feature_cols = params["features"]["numeric"] + params["features"]["categorical"]
    target_col = params["data"]["target"]
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def train_one(params: dict, X_train, y_train, X_val, y_val,
              model_kind: str, model_params: dict, sha: str) -> tuple[str, float, Pipeline]:
    """Train ONE setting inside ONE MLflow run. Return (run_id, mae, fitted_pipeline)."""
    with mlflow.start_run(run_name=model_kind) as run:
        mlflow.log_param("data_sha256", sha)
        mlflow.log_param("model_kind", model_kind)
        mlflow.log_params(model_params)

        pipe = build_pipeline(params, model_kind, model_params)
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_val)
        mae = float(mean_absolute_error(y_val, preds))
        mlflow.log_metric("mae_val", mae)

        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            serialization_format="cloudpickle",
        )
        return run.info.run_id, mae, pipe


def register_champion(params: dict, run_id: str) -> int:
    """Register the best run's model and give it the alias "champion"."""
    reg_name = params["mlflow"]["registered_model_name"]
    result = mlflow.register_model(f"runs:/{run_id}/model", reg_name)
    MlflowClient().set_registered_model_alias(reg_name, "champion", str(result.version))
    return int(result.version)


def main() -> None:
    params = load_params()

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    train_path = params["data"]["train"]
    sha = data_sha256(train_path)
    raw = load_raw(train_path)
    cleaned = clean(raw, params)

    X, y = split_xy(cleaned, params)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"],
    )

    candidates = params["train"]["candidates"]
    best_run_id = None
    best_mae = float("inf")
    best_pipe = None

    for cand in candidates:
        model_kind = cand["kind"]
        model_params = cand.get("params", {})
        run_id, mae, pipe = train_one(
            params, X_train, y_train, X_val, y_val, model_kind, model_params, sha
        )
        print(f"Run {run_id} ({model_kind}): MAE = {mae:.4f}")
        if mae < best_mae:
            best_mae = mae
            best_run_id = run_id
            best_pipe = pipe

    print(f"\nBest run: {best_run_id} with MAE: {best_mae:.4f}")
    version = register_champion(params, best_run_id)
    print(f"Registered model {params['mlflow']['registered_model_name']} version {version} as @champion")

    model_dir = Path(params["paths"]["model"]).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, params["paths"]["model"])
    print(f"Saved best model to {params['paths']['model']}")


if __name__ == "__main__":
    main()
