"""Load params.yaml from the repository root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path: str | Path = ROOT / "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
