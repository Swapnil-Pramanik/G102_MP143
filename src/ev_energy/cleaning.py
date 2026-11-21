import pandas as pd

from ev_energy.config import load_params

ISSUES = {
    "leaking_column": None,
    "unit_mixup": None,          # e.g. {"column": ..., "group_value": ..., "factor": 60}
    "sentinel_value": {"column": "vehicle_age_yr", "value": -999},
    "duplicate_rows": True,
    "stuck_sensor": {
        "column": "battery_start_pct",
        "start_date": "2026-02-27",
        "end_date": "2026-03-15",
    },
    # Typos sit above a big gap: normal values stop at ~65 kWh, typos start at
    # ~214 kWh. 150 is inside the gap and above everything in later.csv (max ~110).
    "target_typos": {"threshold": 150, "factor": 10},
    "spelling_variants": None,   # e.g. {"column": "station_type"}
}


def _cols(params: dict) -> dict:
    """Column names from params.yaml (with safe defaults)."""
    data = params.get("data", {}) or {}
    feats = params.get("features", {}) or {}
    return {
        "target": data.get("target", "energy_kwh"),
        "id": data.get("id_column", "record_id"),
        "date": data.get("date_column", "date"),
        "group": data.get("group_column", "station_type"),
        "categorical": feats.get("categorical", []),
    }


def load_raw(path: str) -> pd.DataFrame:
    """Read a CSV exactly as given."""
    return pd.read_csv(path)


# Each fix_* function fixes ONE problem found in data_checks.py.
# If a problem was NOT present, the function returns df unchanged.
# Always work on a copy: df = df.copy()


def fix_duplicates(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Drop the extra copies, keeping the first one.

    record_id is left out of the comparison (same choice as data_checks.py):
    a copied session with a new id is still the same session.
    """
    if not ISSUES["duplicate_rows"]:
        return df
    id_col = _cols(params or {})["id"]
    subset = [c for c in df.columns if c != id_col]
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def fix_spelling(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Make every spelling of a category the same ('mall ', 'MALL.' -> 'Mall')."""
    issue = ISSUES["spelling_variants"]
    if not issue or issue["column"] not in df.columns:
        return df
    df = df.copy()
    col = issue["column"]
    key = df[col].str.strip().str.lower().str.replace(".", "", regex=False)
    # map each cleaned key to its most common real spelling (the one later.csv uses)
    real = df[col].groupby(key).agg(lambda s: s.mode().iloc[0])
    df[col] = key.map(real)
    return df


def fix_sentinel(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Replace the sentinel number with NaN (a real blank).

    We blank the value instead of deleting the row: the other columns of that
    row are fine, and the model pipeline's imputer fills the blank later.
    """
    issue = ISSUES["sentinel_value"]
    if not issue or issue["column"] not in df.columns:
        return df
    df = df.copy()
    df[issue["column"]] = df[issue["column"]].replace(issue["value"], float("nan"))
    return df


def fix_unit_mixup(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Divide the wrong group's column by the factor (wrong = right x factor).

    Runs AFTER fix_spelling, so the group value is spelled correctly.
    """
    issue = ISSUES["unit_mixup"]
    if not issue or issue["column"] not in df.columns:
        return df
    df = df.copy()
    mask = df[_cols(params)["group"]] == issue["group_value"]
    df.loc[mask, issue["column"]] = df.loc[mask, issue["column"]] / issue["factor"]
    return df


def fix_stuck_sensor(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Set the stuck readings (between start_date and end_date) to NaN."""
    issue = ISSUES["stuck_sensor"]
    if not issue or issue["column"] not in df.columns:
        return df
    df = df.copy()
    dates = pd.to_datetime(df[_cols(params)["date"]])
    mask = dates.between(issue["start_date"], issue["end_date"])  # both ends included
    if not mask.any():   # e.g. later.csv / holdout: other months, nothing to fix
        return df
    df[issue["column"]] = df[issue["column"]].astype("float64")   # int column can't hold NaN
    df.loc[mask, issue["column"]] = float("nan")
    return df


def fix_target_typos(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Divide the targets that were typed 10x too big by 10.

    Option (a) "divide by 10" is used instead of (b) "drop the rows": the
    check showed that /10 puts every one of them back in the normal range,
    so the true value is known and the rows are kept for training.
    Only runs if the target column exists (holdout has no target).
    """
    issue = ISSUES["target_typos"]
    target = _cols(params)["target"]
    if not issue or target not in df.columns:
        return df
    df = df.copy()
    mask = df[target] > issue["threshold"]
    df.loc[mask, target] = df.loc[mask, target] / issue["factor"]
    return df


def drop_leaking_column(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Remove the leaking column if it exists."""
    col = ISSUES["leaking_column"]
    if not col:
        return df
    return df.drop(columns=[col], errors="ignore")


def clean(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply every fix in a sensible order and return the clean table.

    Order:
      1. drop_leaking_column  - so it can never sneak into anything below
      2. fix_duplicates       - on the raw values, before any fix changes them
      3. fix_spelling         - before the unit fix, which filters on station_type
      4. fix_unit_mixup
      5. fix_sentinel         - -999 -> NaN
      6. fix_stuck_sensor     - stuck readings -> NaN
      7. fix_target_typos     - only if the target column is there
    """
    df = df.copy()
    df = drop_leaking_column(df, params)
    df = fix_duplicates(df, params)
    df = fix_spelling(df, params)
    df = fix_unit_mixup(df, params)
    df = fix_sentinel(df, params)
    df = fix_stuck_sensor(df, params)
    df = fix_target_typos(df, params)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    p = load_params()
    raw = load_raw(p["data"]["train"])
    cleaned = clean(raw, p)
    print("rows before:", len(raw), " rows after:", len(cleaned))
    print(cleaned.describe(include="all").T)