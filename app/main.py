"""FastAPI prediction service: POST /predict and GET /health.
"""
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from ev_energy.config import ROOT, load_params
from ev_energy.logging_config import get_logger

PARAMS = load_params()
logger = get_logger("mp143-api")
app = FastAPI(title="MP143 EV charging energy API")

MODEL = None
def get_model():
    global MODEL
    if MODEL is None:
        MODEL = joblib.load(ROOT / PARAMS["paths"]["model"])
    return MODEL


class SessionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    battery_start_pct: float = Field(..., ge=0, le=100, description="Battery % at plug-in")
    session_minutes: float = Field(..., ge=5, description="Session length in minutes (client rule: >= 5)")
    charger_kw: float = Field(..., gt=0, le=100, description="Charger power in kW (client rule: <= 100)")
    temp_c: float = Field(..., ge=-50, le=60, description="Ambient temperature in degrees C")
    vehicle_age_yr: float = Field(..., ge=0, le=50, description="Vehicle age in years")
    queue_before: int = Field(..., ge=0, description="Vehicles queued before this one")
    vehicle: str = Field(..., min_length=1, description="e.g. car, two_wheeler, three_wheeler")
    station_type: str = Field(..., min_length=1, description="e.g. Highway, Mall, Residential")


class PredictionOut(BaseModel):
    energy_kwh: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_path": PARAMS["paths"]["model"]}


@app.post("/predict", response_model=PredictionOut)
def predict(session: SessionIn) -> PredictionOut:
    start = time.perf_counter()
    df = pd.DataFrame([session.model_dump()])
    value = float(get_model().predict(df)[0])
    logger.info("prediction", extra={
        "request_id": str(uuid.uuid4()),
        "prediction": value,
        "latency_ms": round((time.perf_counter() - start) * 1000, 3),
        "vehicle": session.vehicle,
        "station_type": session.station_type,
    })
    return PredictionOut(energy_kwh=value)

