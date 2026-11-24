"""Tests for train.py. No CSVs, no MLflow server: tiny hand-made data only.

Run:  pytest tests/test_train.py -v
"""
import pandas as pd

from ev_energy.config import load_params
from ev_energy.train import build_pipeline


def tiny_xy():
    """20 made-up rows with the 8 feature columns + a target."""
    n = 20
    X = pd.DataFrame(
        {
            "battery_start_pct": [10 + i * 4 for i in range(n)],
            "session_minutes": [30 + i * 5 for i in range(n)],
            "charger_kw": [20.0 + i for i in range(n)],
            "temp_c": [15.0 + i * 0.5 for i in range(n)],
            "vehicle_age_yr": [1.0 + (i % 5) for i in range(n)],
            "queue_before": [i % 4 for i in range(n)],
            "vehicle": ["car", "two_wheeler", "three_wheeler", "car"] * 5,
            "station_type": ["Mall", "Highway", "Residential", "Mall"] * 5,
        }
    )
    y = pd.Series([5.0 + i for i in range(n)])
    return X, y


def test_pipeline_fits_and_predicts():
    X, y = tiny_xy()
    pipe = build_pipeline(load_params(), "ridge", {"alpha": 1.0})
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(X)


def test_pipeline_handles_missing_numbers():
    X, y = tiny_xy()
    X.loc[0, "temp_c"] = float("nan")
    pipe = build_pipeline(load_params(), "ridge", {"alpha": 1.0})
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(X)
    assert not pd.isna(preds).any()


def test_pipeline_handles_unknown_category():
    X, y = tiny_xy()
    pipe = build_pipeline(load_params(), "ridge", {"alpha": 1.0})
    pipe.fit(X, y)
    X_unknown = X.iloc[[0]].copy()
    X_unknown.loc[0, "vehicle"] = "bus"
    preds = pipe.predict(X_unknown)
    assert len(preds) == 1
