"""Tests for cleaning.py.

RULE: tests must NOT read data/raw/*.csv (those files are not on GitHub, so
CI would fail). Build a tiny DataFrame by hand inside each test instead.

Run:  pytest tests/test_cleaning.py -v
"""
import pandas as pd

from ev_energy import cleaning
from ev_energy.cleaning import clean
from ev_energy.config import load_params

PARAMS = load_params()          # reads params.yaml only, never the CSVs


def tiny_df() -> pd.DataFrame:
    """A few hand-made CLEAN rows with the same columns as train.csv.

    Each test copies this and puts ONE mistake into it.
    """
    return pd.DataFrame(
        {
            "record_id": ["R1", "R2", "R3"],
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "battery_start_pct": [40, 50, 60],
            "session_minutes": [60, 90, 120],
            "charger_kw": [30.0, 50.0, 22.0],
            "temp_c": [20.0, 25.0, 30.0],
            "vehicle_age_yr": [2.0, 3.0, 4.0],
            "queue_before": [1, 2, 3],
            "vehicle": ["car", "two_wheeler", "three_wheeler"],
            "station_type": ["Mall", "Highway", "Residential"],
            "energy_kwh": [20.0, 30.0, 40.0],
        }
    )


def test_clean_removes_duplicates():
    df = tiny_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)   # R1 copied
    out = clean(df, PARAMS)
    assert len(df) == 4
    assert len(out) == 3


def test_clean_removes_duplicate_with_new_id():
    # same session copied but given a new record_id -> still a duplicate
    df = tiny_df()
    copy = df.iloc[[0]].copy()
    copy["record_id"] = "R99"
    out = clean(pd.concat([df, copy], ignore_index=True), PARAMS)
    assert len(out) == 3


def test_clean_replaces_sentinel_with_nan():
    df = tiny_df()
    df.loc[1, "vehicle_age_yr"] = -999
    out = clean(df, PARAMS)
    assert (out["vehicle_age_yr"] == -999).sum() == 0
    assert pd.isna(out.loc[1, "vehicle_age_yr"])
    assert len(out) == 3                      # row kept, only the value blanked


def test_clean_blanks_stuck_sensor():
    df = tiny_df()
    df.loc[0, "date"] = "2026-03-01"          # inside 2026-02-27 .. 2026-03-15
    df.loc[0, "battery_start_pct"] = 11
    out = clean(df, PARAMS)
    assert pd.isna(out.loc[0, "battery_start_pct"])
    assert out.loc[1, "battery_start_pct"] == 50   # rows outside the window untouched


def test_clean_fixes_target_typos():
    df = tiny_df()
    df.loc[2, "energy_kwh"] = 400.0           # 40.0 typed with an extra zero
    out = clean(df, PARAMS)
    assert out.loc[2, "energy_kwh"] == 40.0
    assert out["energy_kwh"].tolist() == [20.0, 30.0, 40.0]


def test_clean_works_without_target():
    # holdout has no energy_kwh column: clean must not crash or add it
    df = tiny_df().drop(columns=["energy_kwh"])
    out = clean(df, PARAMS)
    assert "energy_kwh" not in out.columns
    assert len(out) == 3


def test_clean_fixes_spelling(monkeypatch):
    # spelling variants were NOT in our train.csv, so the fix is switched off
    # in ISSUES. Switch it on here to prove fix_spelling works if needed.
    monkeypatch.setitem(
        cleaning.ISSUES,
        "spelling_variants",
        {"column": "station_type", "levels": ["Mall", "Highway", "Residential"]},
    )
    df = pd.concat([tiny_df()] * 3, ignore_index=True)
    df["record_id"] = [f"R{i}" for i in range(len(df))]
    df["energy_kwh"] = range(10, 10 + len(df))     # make rows different (no duplicates)
    df.loc[3, "station_type"] = "mall "
    df.loc[4, "station_type"] = "HIGHWAY."
    out = clean(df, PARAMS)
    assert set(out["station_type"]) == {"Mall", "Highway", "Residential"}


def test_clean_keeps_good_rows():
    df = tiny_df()
    out = clean(df, PARAMS)
    # nothing was wrongly removed or changed in clean rows
    assert len(out) == 3
    assert list(out.columns) == list(df.columns)
    pd.testing.assert_frame_equal(out, df)