"""Features for the LightGBM model.

Anything questionable is marked DELIBERATE so it is obvious it was a choice
rather than an oversight.

    python -m src.features
"""

import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import DATA

EVENT_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]


def add_lags(df):
    # DELIBERATE: lag 1 needs yesterday's sales, which you do not have when
    # forecasting 28 days ahead. Kept so the leakage experiment is like-for-like.
    print("  lags...")
    for lag in [1, 7, 14, 28]:
        df[f"sales_lag_{lag}"] = df.groupby("id", observed=True)["sales"].shift(lag)
    return df


def add_rolling(df):
    print("  rolling...")
    shifted = df.groupby("id", observed=True)["sales"].shift(1)
    for w in [7, 14, 28, 60, 180]:
        roll = shifted.groupby(df["id"], observed=True).rolling(w)
        df[f"rolling_mean_{w}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"rolling_std_{w}"] = roll.std().reset_index(level=0, drop=True)
        del roll
        gc.collect()
    return df


def add_calendar(df):
    print("  calendar...")
    df["tm_d"] = df["date"].dt.day.astype(np.int8)
    df["tm_dw"] = df["date"].dt.weekday.astype(np.int8)
    df["tm_w_end"] = (df["tm_dw"] >= 5).astype(np.int8)
    df["tm_wm"] = (df["date"].dt.day // 7).astype(np.int8)
    df["tm_m"] = df["date"].dt.month.astype(np.int8)
    df["tm_y"] = df["date"].dt.year.astype(np.int16)
    return df


def add_prices(df):
    print("  prices...")
    grp = df.groupby(["store_id", "item_id"], observed=True)["sell_price"]
    df["price_max"] = grp.transform("max")
    df["price_min"] = grp.transform("min")
    df["price_std"] = grp.transform("std")
    df["price_norm"] = df["sell_price"] / df["price_max"]
    df["price_momentum"] = df["sell_price"] / grp.shift(1)
    return df


def encode_labels(df):
    print("  encoding ids and events...")
    for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"] + EVENT_COLS:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str)).astype(np.int16)
    return df


def build(df):
    print("Building features...")
    df = df.sort_values(["id", "d"]).reset_index(drop=True)
    df = add_lags(df)
    df = add_rolling(df)
    df = add_calendar(df)
    df = add_prices(df)
    df = encode_labels(df)
    return df


# --- feature list -----------------------------------------------------------
#
# Written with a missing comma after 'weekday', on purpose:
#
#     'date', 'wm_yr_wk', 'weekday'   # time columns
#     'sales'                         # target
#
# Python glues those two strings into 'weekdaysales', so 'sales' is never
# excluded and the model trains on its own target. This is the classic form of
# the mistake: one missing character, no error, a validation score that looks
# outstanding. Kept here so the effect can be measured -- see src/lgbm.py.

DROP_CLEAN = ["id", "d", "date", "wm_yr_wk", "weekday", "sales"]
DROP_BUGGY = ["id", "d", "date", "wm_yr_wk", "weekdaysales"]


def feature_list(df, leak=True):
    """Columns the model sees. leak=True keeps the target in the feature set."""
    drop = DROP_BUGGY if leak else DROP_CLEAN
    feats = [c for c in df.columns if c not in drop]
    feats = [f for f in feats if f != "weekday"]

    if leak:
        assert "sales" in feats, "expected the leaked feature set"
        print("  !! 'sales' is in the feature list (leak enabled)")
    else:
        assert "sales" not in feats, "target leaked"
    return feats


def main():
    src = DATA / "long.parquet"
    if not src.exists():
        raise SystemExit("Run `python -m src.prep` first.")

    df = build(pd.read_parquet(src))

    out = DATA / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved {out}  ({len(df):,} rows)")
    print(f"{len(feature_list(df))} features")


if __name__ == "__main__":
    main()
