"""Tests for monitor.py: PSI and the promotion rule. Tiny made-up data only.

Run:  pytest tests/test_monitor.py -v
"""
import numpy as np
import pandas as pd

from ev_energy.monitor import decide, psi, psi_table, split_later


# ----------------------------------------------------------------------------
# PSI
# ----------------------------------------------------------------------------
def test_psi_same_data_is_near_zero():
    s = pd.Series(np.arange(1000, dtype=float))
    assert psi(s, s) < 0.01


def test_psi_shifted_data_is_big():
    a = pd.Series(np.arange(1000, dtype=float))
    b = a + 500
    assert psi(a, b) > 0.25


def test_psi_small_shift_is_smaller_than_big_shift():
    a = pd.Series(np.arange(1000, dtype=float))
    assert psi(a, a + 50) < psi(a, a + 500)


def test_psi_ignores_blanks():
    # cleaning turns -999 and stuck readings into NaN; PSI must not crash or
    # count them as drift
    a = pd.Series(np.arange(1000, dtype=float))
    b = a.copy()
    b[:100] = np.nan
    with_blanks = psi(a, b)
    without_blanks = psi(a, b.dropna().reset_index(drop=True))
    assert np.isfinite(with_blanks)
    assert with_blanks == without_blanks


def test_psi_table_has_one_score_per_column():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    later = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(3, 1, 500)})
    scores = psi_table(train, later, ["x", "y"], bins=10)
    assert set(scores) == {"x", "y"}
    assert scores["y"] > scores["x"]          # y moved, x did not


# ----------------------------------------------------------------------------
# Promotion rule
# ----------------------------------------------------------------------------
def test_promote_when_10_percent_better():
    assert decide(10.0, 8.9, 0.10) == "promote"


def test_keep_when_not_10_percent_better():
    assert decide(10.0, 9.5, 0.10) == "keep"


def test_promote_at_exactly_10_percent():
    # the rule is "at least 10% better", so exactly 10% counts
    assert decide(10.0, 9.0, 0.10) == "promote"


def test_keep_when_v2_is_worse():
    assert decide(10.0, 12.0, 0.10) == "keep"


# ----------------------------------------------------------------------------
# Splitting later.csv by date
# ----------------------------------------------------------------------------
def test_split_later_is_70_30_by_date():
    params = {"data": {"date_column": "date"},
              "monitoring": {"later_train_fraction": 0.7}}
    dates = pd.date_range("2026-07-01", periods=10).strftime("%Y-%m-%d")
    later = pd.DataFrame({"date": dates[::-1], "v": range(10)})   # reverse date order
    first, last = split_later(later, params)
    assert len(first) == 7 and len(last) == 3
    assert first["date"].max() < last["date"].min()   # no peeking into the future
