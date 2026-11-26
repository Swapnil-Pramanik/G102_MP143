"""Predict holdout_features.csv with the @champion model (PROBLEM.md section 8).

Run LAST, after Swapnil's monitor.py has decided whether v2 is the champion:
    python -m ev_energy.predict

Output: predictions/holdout_predictions.csv with EXACTLY two columns
    record_id,prediction
and one row for every row of holdout_features.csv (250 rows).
"""
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from ev_energy.config import load_params


def load_champion(params: dict):
    """Load the model that has the alias "champion" from the MLflow registry.

    TODO:
      1. mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
      2. uri = f"models:/{params['mlflow']['registered_model_name']}@champion"
      3. return mlflow.sklearn.load_model(uri)
    """
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    uri = f"models:/{params['mlflow']['registered_model_name']}@champion"
    try:
        return mlflow.sklearn.load_model(uri)
    except Exception:
        # Fallback to the committed champion model file if local mlruns directory
        # was generated on a teammate's machine (as documented in monitor.py)
        return joblib.load(params["paths"]["model"])


def predict_holdout(params: dict) -> pd.DataFrame:
    """Return a DataFrame with columns record_id, prediction.

    TODO:
      1. read params["data"]["holdout"]
      2. should you run clean() on it? (Think: does holdout have the same
         spelling / unit problems? Look at it. Explain your choice in the viva.)
      3. model.predict(<the 8 feature columns>)
      4. pd.DataFrame({"record_id": ..., "prediction": ...})
    """
    holdout_path = params["data"]["holdout"]
    df = pd.read_csv(holdout_path)

    model = load_champion(params)
    feature_cols = params["features"]["numeric"] + params["features"]["categorical"]
    X = df[feature_cols]

    preds = model.predict(X)

    return pd.DataFrame({
        "record_id": df[params["data"]["id_column"]],
        "prediction": preds,
    })


def main() -> None:
    params = load_params()
    out = predict_holdout(params)
    assert len(out) == 250, f"Expected 250 rows, got {len(out)}"

    pred_path = Path(params["paths"]["predictions"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(pred_path, index=False)
    print(f"Saved {len(out)} predictions to {pred_path}")


if __name__ == "__main__":
    main()
