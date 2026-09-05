"""Turn the wide sales file into one long table.

Raw sales has one row per item-store and one column per day. Every model wants
one row per item-store-day instead.

    python -m src.prep
"""

import numpy as np
import pandas as pd

from .config import DATA, FIRST_DAY, HORIZON, LAST_DAY, RAW

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def shrink(df):
    """Downcast numbers so the table fits in memory."""
    before = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)

    after = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  memory {before:.0f} MB -> {after:.0f} MB")
    return df


def main():
    print("Reading csvs...")
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    calendar = pd.read_csv(RAW / "calendar.csv")
    prices = pd.read_csv(RAW / "sell_prices.csv")

    # Blank columns for the days we have to forecast, so the same feature code
    # covers history and the future.
    future = pd.DataFrame(
        np.nan,
        index=sales.index,
        columns=[f"d_{d}" for d in range(LAST_DAY + 1, LAST_DAY + HORIZON + 1)],
        dtype=np.float32,
    )
    sales = pd.concat([sales, future], axis=1)

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    print(f"Melting {len(day_cols)} days...")
    df = sales.melt(id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="sales")
    df["d"] = df["d"].str.replace("d_", "", regex=False).astype(np.int16)

    df = df[df["d"] >= FIRST_DAY].reset_index(drop=True)
    print(f"  {len(df):,} rows kept (from d_{FIRST_DAY})")

    print("Joining calendar...")
    calendar["d"] = calendar["d"].str.replace("d_", "", regex=False).astype(np.int16)
    calendar["date"] = pd.to_datetime(calendar["date"])
    cal_cols = [
        "d", "date", "wm_yr_wk", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    df = df.merge(calendar[cal_cols], on="d", how="left")

    print("Joining prices...")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    df = shrink(df)

    out = DATA / "long.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved {out}  ({len(df):,} rows, {out.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
