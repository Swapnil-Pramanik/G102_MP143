# G102 · MP143 — EV Charging Session Energy (Patna)

SCSE3040 MLOps mini project. Client: **Patna Green Charge Network** (made-up).
Goal: predict `energy_kwh` (energy delivered in a charging session) at plug-in time.
Regression, measured with **MAE**.

| | |
|---|---|
| Group id | **G102** |
| Problem id | **MP143** |
| Faculty | Gaurav Tripathi (Batch 84) |

## Members

| Enrolment no. | Name | Suggested role |
|---|---|---|
| S24CSEU2214 | Swapnil Pramanik | _TBD_ |
| S24CSEU2183 | Mayank Singh | _TBD_ |
| S24CSEU2200 | Aditya Singh | _TBD_ |
| S24CSEU2201 | Aditya Jaiswal Kumar | _TBD_ |

Roles (handbook §4): Data lead · Model lead · Service lead · Ops lead.
Every member commits from their **own** GitHub account.

## Status

Scaffold only. Folder layout, DVC, config, Docker, CI and Kubernetes files are in
place; data checks, cleaning, training, API, tests, monitoring and predictions are
still TODO.

| Stage (handbook §6) | Status |
|---|---|
| 1. Repo, README, DVC, find the 7 data problems | DVC done; data checks TODO |
| 2. Cleaning, MLflow training (3+ runs), registry `@champion`, tests | TODO |
| 3. FastAPI + rules, Dockerfile, green CI, k8s, JSON logs | Dockerfile / CI / k8s written; app TODO |
| 4. PSI, MAE per group, v2 retrain, promotion rule, holdout predictions | TODO |

## Architecture

```
data/raw/*.csv (DVC) ──► src/ev_energy/data_checks.py ──► reports/findings.yaml
                     └─► src/ev_energy/cleaning.py ──► src/ev_energy/train.py ──► mlflow.db (runs + registry @champion)
                                                                            └─► models/model.joblib
models/model.joblib ──► app/main.py (FastAPI) ──► Dockerfile (mp143-api:1.0) ──► k8s/ (mp143-api, mp143-svc)
later.csv ──► src/ev_energy/monitor.py (PSI, MAE per station_type, v2, promote?) ──► mlflow.db
@champion ──► src/ev_energy/predict.py ──► predictions/holdout_predictions.csv
All code reads settings from params.yaml via src/ev_energy/config.py.
```

## Project structure

```
README.md                      this file
requirements.txt               runtime + test libs, all pinned (used by CI and Docker)
requirements-dev.txt           requirements.txt + dvc (local only)
params.yaml                    paths, feature lists, MLflow names, monitoring settings
pytest.ini                     puts src/ and repo root on the import path for tests
Dockerfile                     python:3.12-slim, deps first, EXPOSE 8000, uvicorn
.dockerignore                  keeps data, venv, mlflow, tests out of the image
.gitignore                     venv, caches, mlruns/ mlartifacts/ (mlflow.db is committed)
.dvc/config, .dvcignore        from `dvc init`
.github/workflows/ci.yml       on push / PR: install requirements.txt, run pytest
docs/
  STUDENT_HANDBOOK.md          course rules, marks, layout
  PROBLEM.md                   our problem: columns, API rules, deployment names
  GROUP.md                     group members
data/raw/
  train.csv  (DVC, not in git) Jan–Jun 2026, ~1058 rows, messy
  train.csv.dvc                DVC pointer (committed)
  later.csv  (DVC, not in git) Jul–Aug 2026, 300 rows, clean
  later.csv.dvc                DVC pointer (committed)
  holdout_features.csv         Sep 2026, 250 rows, no target (committed)
  .gitignore                   written by `dvc add`
src/ev_energy/
  __init__.py
  config.py                    load_params(): yaml.safe_load(params.yaml)
  data_checks.py               TODO: detect the 7 data problems
  cleaning.py                  TODO: clean train.csv
  train.py                     TODO: MLflow runs, register, @champion
  monitor.py                   TODO: PSI, MAE by station_type, v2, promotion
  predict.py                   TODO: holdout predictions
app/
  __init__.py
  main.py                      TODO: FastAPI POST /predict, GET /health
tests/
  __init__.py                  TODO: 3+ test_*.py, one with TestClient
models/                        model.joblib (champion) goes here
predictions/                   holdout_predictions.csv goes here
reports/findings.yaml          the marking form (fill every value)
k8s/deployment.yaml            Deployment mp143-api, 4 replicas, probes on /health
k8s/service.yaml               Service mp143-svc, port 80 → 8000
mlflow.db                      (created on first training run — commit it)
```

## Key data model

Features: `battery_start_pct`, `session_minutes`, `charger_kw`, `temp_c`,
`vehicle_age_yr`, `queue_before` (numeric); `vehicle`, `station_type` (categorical).
Target: `energy_kwh`. Ids/dates: `record_id`, `date`. Group column: `station_type`
(Mall, Highway, Residential).

## API contract (PROBLEM.md §5)

- `POST /predict` — body has exactly the 8 features. Rules in the pydantic schema:
  `charger_kw` ≤ 100, `session_minutes` ≥ 5 → otherwise 422.
- `GET /health` → `{"status": "ok"}`.
- One JSON log line per prediction to stdout.

## Deployment (PROBLEM.md §6)

Image `mp143-api:1.0` · Deployment `mp143-api` · Service `mp143-svc` · replicas 4 ·
container port 8000 · service port 80 · readiness + liveness on `/health`
(liveness waits longer).

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
dvc status                      # data/raw/train.csv & later.csv are DVC-tracked
pytest                          # once tests exist
uvicorn app.main:app --reload   # once app/main.py is written
docker build -t mp143-api:1.0 .
```

`train.sha256` (log as `data_sha256` in every MLflow run):
`0085bcbda357f434781faa89664467ab3634c70db87261aa6eceaf3934388383`

**Never open `train.csv` in Excel and save it** — the hash will stop matching.

## Dependencies

pandas 3.1.1 · numpy 2.5.5 · scikit-learn 1.9.1 · mlflow 3.18.0 · fastapi 0.165.0 ·
uvicorn 0.54.0 · pydantic 2.14.4 · pyyaml 6.0.4 · joblib 1.6.0 ·
python-json-logger 4.2.0 · httpx 0.28.1 · pytest 9.2.0 · (dev) dvc 3.67.1.

## Configuration

All settings live in `params.yaml`; no secrets or env vars are needed. MLflow tracks
to `sqlite:///mlflow.db` in the repo root.
