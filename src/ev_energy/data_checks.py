import numpy as np
import pandas as pd

from ev_energy.config import load_params

# Column names used when params.yaml does not name them.
DEFAULT_TARGET = "energy_kwh"
DEFAULT_ID_COL = "record_id"
DEFAULT_DATE_COL = "date"
DEFAULT_GROUP_COL = "station_type"

# Exact unit conversions a mix-up could come from (h->min = 60, cm->m = 100 ...).
UNIT_FACTORS = [10, 12, 60, 100, 1000]
# Numbers people type instead of leaving a cell blank.
SENTINELS = [-1, -9, -99, -999, -9999, 99, 999, 9999, 99999]


def _pick(d: dict, *keys, default=None):
    """Return the first key of `d` that exists and is not None."""
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def _py(x):
    """numpy number -> plain Python number, so the printout is clean YAML."""
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return int(x) if float(x).is_integer() else float(x)
    return x


def _norm_text(s: pd.Series) -> pd.Series:
    """'MALL.' / 'mall ' / 'Mall' -> 'mall'."""
    return s.astype(str).str.strip().str.lower().str.replace(".", "", regex=False)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_data(params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, later, holdout) as DataFrames, read exactly as given."""
    data = params["data"]
    train = pd.read_csv(_pick(data, "train", "train_path", "train_csv"))
    later = pd.read_csv(_pick(data, "later", "later_path", "later_csv"))
    holdout = pd.read_csv(
        _pick(data, "holdout", "holdout_features", "holdout_path", "holdout_csv")
    )
    return train, later, holdout


# ----------------------------------------------------------------------------
# 1. Leaking column
# ----------------------------------------------------------------------------
def check_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame, target: str) -> dict:
    """
    Keys to return: present, column
    """
    extra = sorted(set(train.columns) - set(holdout.columns) - {target})
    print(f"[leak] in train but not in holdout: {extra}")

    numeric = train.select_dtypes("number").drop(columns=[target], errors="ignore")
    corrs = numeric.corrwith(train[target]).abs().sort_values(ascending=False)
    print("[leak] |corr| with target:\n" + corrs.round(3).to_string())

    if extra:
        return {"present": True, "column": extra[0]}
    strong = corrs[corrs >= 0.99]
    if len(strong):
        return {"present": True, "column": strong.index[0]}
    return {"present": False, "column": None}


# ----------------------------------------------------------------------------
# 2. Unit mix-up
# ----------------------------------------------------------------------------
def check_unit_mixup(train: pd.DataFrame, numeric_cols: list[str], group_col: str) -> dict:
    """
    Keys to return: present, column, group_value, factor

    Groups are compared on their MEDIAN (not mean) so a few sentinels or
    target typos cannot fake a unit problem. Spelling variants are merged
    first ('mall ' and 'Mall' count as one group).
    """
    groups = _norm_text(train[group_col])
    best = None  # (ratio, column, group_value, factor)

    for col in numeric_cols:
        if col not in train.columns:
            continue
        stats = train.groupby(groups)[col].agg(["min", "median", "mean", "max"])
        print(f"[unit] {col}\n{stats.round(2).to_string()}")
        medians = stats["median"]
        if len(medians) < 2:
            continue
        for g, m in medians.items():
            others = medians.drop(g).median()
            if pd.isna(m) or pd.isna(others) or m == 0 or others == 0:
                continue
            ratio = abs(m / others)
            ratio = max(ratio, 1 / ratio)  # same number whether bigger or smaller
            if ratio < 5:  # less than 5x apart: normal variation, not units
                continue
            if best is None or ratio > best[0]:
                factor = min(UNIT_FACTORS, key=lambda f: abs(np.log(ratio / f)))
                spelling = train.loc[groups == g, group_col].mode().iloc[0]
                best = (ratio, col, spelling, factor)

    if best is None:
        return {"present": False, "column": None, "group_value": None, "factor": None}
    ratio, col, group_value, factor = best
    print(f"[unit] measured ratio {ratio:.2f} -> exact factor {factor}")
    return {"present": True, "column": col, "group_value": group_value, "factor": factor}


# ----------------------------------------------------------------------------
# 3. Sentinel value
# ----------------------------------------------------------------------------
def check_sentinel_value(train: pd.DataFrame, numeric_cols: list[str]) -> dict:
    """
    Keys to return: present, column, value, count

    A value counts as a sentinel if it is one of the usual "fake" numbers,
    it repeats (3+ times) and it lies outside the range of every other value
    in that column (so a real -1 degree reading does not count).
    """
    best = None  # (count, column, value)
    for col in numeric_cols:
        if col not in train.columns:
            continue
        s = train[col]
        print(f"[sentinel] {col}: min={s.min()} max={s.max()} "
              f"top={s.value_counts().head(3).to_dict()}")
        for v in SENTINELS:
            n = int((s == v).sum())
            if n < 3:
                continue
            rest = s[s != v].dropna()
            if len(rest) and rest.min() <= v <= rest.max():
                continue  # inside the normal range -> could be a real value
            if best is None or n > best[0]:
                best = (n, col, v)

    if best is None:
        return {"present": False, "column": None, "value": None, "count": None}
    count, col, value = best
    return {"present": True, "column": col, "value": _py(value), "count": count}


# ----------------------------------------------------------------------------
# 4. Duplicate rows
# ----------------------------------------------------------------------------
def check_duplicate_rows(train: pd.DataFrame, id_col: str = DEFAULT_ID_COL) -> dict:
    """
    Keys to return: present, count   (count = number of EXTRA copies)

    record_id is left OUT of the comparison: it is only a label. If a row was
    copied and given a new id it is still the same charging session, and
    comparing with the id would miss it.
    """
    with_id = int(train.duplicated().sum())
    if id_col in train.columns:
        without_id = int(train.drop(columns=[id_col]).duplicated().sum())
    else:
        without_id = with_id
    print(f"[dup] extra copies: with {id_col}={with_id}, without {id_col}={without_id}")

    if without_id == 0:
        return {"present": False, "count": None}
    return {"present": True, "count": without_id}


# ----------------------------------------------------------------------------
# 5. Stuck sensor
# ----------------------------------------------------------------------------
def check_stuck_sensor(train: pd.DataFrame, numeric_cols: list[str], date_col: str,
                       min_run: int = 20) -> dict:
    """
    Keys to return: present, column, start_date, end_date  (YYYY-MM-DD)
    """
    df = train.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col, kind="stable").reset_index(drop=True)

    best = None  # (run_length, column, start, end)
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = df[col]
        run_id = (s != s.shift()).cumsum()          # new id every time the value changes
        lengths = s.groupby(run_id).transform("size")
        longest = int(lengths.max())
        print(f"[stuck] {col}: longest run of the same value = {longest}")
        if longest < min_run:
            continue
        rows = df.loc[run_id == run_id[lengths.idxmax()], date_col]
        if best is None or longest > best[0]:
            best = (longest, col, rows.min(), rows.max())

    if best is None:
        return {"present": False, "column": None, "start_date": None, "end_date": None}
    _, col, start, end = best
    return {
        "present": True,
        "column": col,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
    }


# ----------------------------------------------------------------------------
# 6. Target typos
# ----------------------------------------------------------------------------
def check_target_typos(train: pd.DataFrame, target: str) -> dict:
    """
    Keys to return: present, count

    Sort the target, look only at the top 5%, and find the biggest jump
    (as a ratio, so it does not depend on the scale). If the values above the
    jump look normal after dividing by 10, they are typos.
    """
    s = train[target].dropna().sort_values().reset_index(drop=True)
    print(f"[typo] {target}:\n{s.describe().round(2).to_string()}")
    print(f"[typo] largest 15: {s.tail(15).round(2).tolist()}")

    top_n = max(20, int(len(s) * 0.05))
    top = s.tail(top_n + 1)
    jumps = (top / top.shift()).iloc[1:]           # ratio between neighbours
    if jumps.empty or jumps.max() < 2:             # no real gap
        return {"present": False, "count": None}

    cut = jumps.idxmax()                           # first value above the gap
    above, below = s.loc[cut:], s.loc[: cut - 1]
    looks_normal = (above / 10).between(below.min(), below.max() * 1.2).all()
    print(f"[typo] gap x{jumps.max():.1f} between {below.iloc[-1]:.2f} and "
          f"{above.iloc[0]:.2f}; /10 looks normal: {looks_normal}")

    if not looks_normal:
        return {"present": False, "count": None}
    return {"present": True, "count": int(len(above))}


# ----------------------------------------------------------------------------
# 7. Spelling variants
# ----------------------------------------------------------------------------
def check_spelling_variants(train: pd.DataFrame, later: pd.DataFrame, text_cols: list[str]) -> dict:
    """
    Keys to return: present, column, true_levels
    """
    best = None  # (extra_levels, column, true_levels)
    for col in text_cols:
        if col not in train.columns or col not in later.columns:
            continue
        n_train = train[col].nunique()
        n_later = later[col].nunique()
        n_norm = _norm_text(train[col].dropna()).nunique()
        print(f"[spell] {col}: train={n_train} levels, after cleanup={n_norm}, "
              f"later={n_later}  {sorted(train[col].dropna().unique())}")
        extra = n_train - n_later
        if extra > 0 and (best is None or extra > best[0]):
            best = (extra, col, n_later)

    if best is None:
        return {"present": False, "column": None, "true_levels": None}
    _, col, true_levels = best
    return {"present": True, "column": col, "true_levels": int(true_levels)}


# ----------------------------------------------------------------------------
# Run everything
# ----------------------------------------------------------------------------
def run_all_checks(params: dict) -> dict:
    """Run all seven checks and return {problem_name: result_dict}."""
    train, later, holdout = load_data(params)
    data = params.get("data", {}) or {}
    feats = params.get("features", {}) or {}

    target = _pick(feats, "target", default=_pick(data, "target", default=DEFAULT_TARGET))
    id_col = _pick(feats, "id", "id_col", default=_pick(data, "id_col", default=DEFAULT_ID_COL))
    date_col = _pick(feats, "date", "date_col", default=_pick(data, "date_col", default=DEFAULT_DATE_COL))
    group_col = _pick(feats, "group", "group_col", default=DEFAULT_GROUP_COL)

    skip = {target, id_col, date_col}
    numeric_cols = _pick(feats, "numeric", "numeric_cols", "num", "numerical")
    if numeric_cols is None:
        numeric_cols = [c for c in train.select_dtypes("number").columns if c not in skip]
    text_cols = _pick(feats, "categorical", "categorical_cols", "cat", "text", "text_cols")
    if text_cols is None:
        text_cols = [c for c in train.select_dtypes("object").columns if c not in skip]

    return {
        "leaking_column": check_leaking_column(train, holdout, target),
        "unit_mixup": check_unit_mixup(train, numeric_cols, group_col),
        "sentinel_value": check_sentinel_value(train, numeric_cols),
        "duplicate_rows": check_duplicate_rows(train, id_col),
        "stuck_sensor": check_stuck_sensor(train, numeric_cols, date_col),
        "target_typos": check_target_typos(train, target),
        "spelling_variants": check_spelling_variants(train, later, text_cols),
    }


if __name__ == "__main__":
    results = run_all_checks(load_params())
    for name, result in results.items():
        print(f"\n== {name} ==")
        for key, value in result.items():
            print(f"  {key}: {value}")