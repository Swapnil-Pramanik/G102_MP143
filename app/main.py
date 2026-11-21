"""FastAPI prediction service: POST /predict and GET /health.

TODO (PROBLEM.md section 5):
- pydantic request schema with exactly the 8 feature fields
- charger_kw: reject > 100  (Field(..., le=100))
- session_minutes: reject < 5  (Field(..., ge=5))
- load models/model.joblib
- one JSON log line per prediction to stdout (no personal details)
"""
