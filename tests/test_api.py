import pytest
from fastapi.testclient import TestClient

import app.main as main


class FakeModel:
    """Pretends to be the sklearn pipeline: always predicts 42.0."""

    def predict(self, df):
        return [42.0] * len(df)


@pytest.fixture
def client(monkeypatch):
    # Swap the real model for the fake one, so no models/model.joblib is needed.
    monkeypatch.setattr(main, "MODEL", FakeModel())
    return TestClient(main.app)

def good_body() -> dict:
    """A valid request body with all 8 fields.
    """
    return {
        "battery_start_pct": 44,
        "session_minutes": 91,
        "charger_kw": 57.8,
        "temp_c": 12.2,
        "vehicle_age_yr": 5.0,
        "queue_before": 9,
        "vehicle": "car",
        "station_type": "Mall",
    }
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=good_body())
    assert r.status_code == 200
    assert r.json() == {"energy_kwh": 42.0}


def test_rejects_charger_kw_above_100(client):
    body = good_body()
    body["charger_kw"] = 150
    r = client.post("/predict", json=body)
    assert r.status_code == 422


def test_rejects_session_minutes_below_5(client):
    body = good_body()
    body["session_minutes"] = 2
    r = client.post("/predict", json=body)
    assert r.status_code == 422


def test_rejects_missing_field(client):
    body = good_body()
    del body["vehicle"]
    r = client.post("/predict", json=body)
    assert r.status_code == 422