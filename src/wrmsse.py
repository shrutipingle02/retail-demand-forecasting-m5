"""WRMSSE - the score Kaggle used to rank the M5 competition.

For one series:

    RMSSE = sqrt( mean((actual - forecast)^2) / mean(day-to-day change^2) )

The bottom half is how wrong you'd be if you just guessed "same as yesterday",
so RMSSE below 1.0 means you beat that. Series are then weighted by how many
dollars they sold in the last 28 training days, and the whole thing is averaged
over 12 groupings - from total company sales down to single item-store pairs.

Lower is better. Competition winner scored 0.520.
"""

import numpy as np
import pandas as pd

from .config import RAW

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# The 12 groupings the competition scored. None = all series added together.
LEVELS = [
    None,
    ["state_id"],
    ["store_id"],
    ["cat_id"],
    ["dept_id"],
    ["state_id", "cat_id"],
    ["state_id", "dept_id"],
    ["store_id", "cat_id"],
    ["store_id", "dept_id"],
    ["item_id"],
    ["state_id", "item_id"],
    ["store_id", "item_id"],
]


def level_name(level):
    return "total" if level is None else "-".join(c.replace("_id", "") for c in level)


class WRMSSE:
    def __init__(self, train, valid):
        """train: wide sales up to the cutoff. valid: the 28 actual days."""
        self.ids = train[ID_COLS].reset_index(drop=True)
        self.train_days = [c for c in train.columns if c.startswith("d_")]
        self.valid = valid.reset_index(drop=True)

        self.history = train[self.train_days].reset_index(drop=True)
        self.dollars = self._dollars()

        self.weights, self.scales = {}, {}
        for i, level in enumerate(LEVELS):
            self._setup(i, level)

    def _dollars(self):
        """What each series sold, in dollars, over the last 28 training days."""
        calendar = pd.read_csv(RAW / "calendar.csv", usecols=["d", "wm_yr_wk"])
        prices = pd.read_csv(RAW / "sell_prices.csv")

        recent = self.train_days[-28:]
        long = pd.concat([self.ids[["id", "item_id", "store_id"]], self.history[recent]], axis=1)
        long = long.melt(id_vars=["id", "item_id", "store_id"], var_name="d", value_name="units")

        long = long.merge(calendar, on="d", how="left")
        long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        long["dollars"] = long["units"] * long["sell_price"].fillna(0)

        totals = long.groupby("id", observed=True)["dollars"].sum()
        return totals.reindex(self.ids["id"]).reset_index(drop=True)

    def _group(self, matrix, level):
        """Add series up to one grouping."""
        if level is None:
            return matrix.sum(axis=0).to_frame().T
        return matrix.groupby([self.ids[c] for c in level], observed=True).sum()

    def _setup(self, i, level):
        rolled = self._group(self.history, level).to_numpy(dtype=np.float64)

        # Scale: average squared day-to-day change, counted only from the first
        # day the series actually sold something. Leading zeros mean the item
        # wasn't stocked yet and would otherwise shrink the denominator.
        scales = np.empty(len(rolled))
        for r, series in enumerate(rolled):
            nz = np.flatnonzero(series)
            active = series[nz[0]:] if nz.size else series
            diffs = np.diff(active)
            scales[r] = (diffs**2).mean() if diffs.size else np.nan
        self.scales[i] = scales

        w = self._group(self.dollars.to_frame("dollars"), level)["dollars"].to_numpy()
        self.weights[i] = w / w.sum()

    def score(self, preds):
        """preds: 30490 x 28, rows in the same order as the train frame."""
        preds = pd.DataFrame(np.asarray(preds, dtype=np.float64))
        actual = pd.DataFrame(self.valid.to_numpy(dtype=np.float64))

        results = {}
        for i, level in enumerate(LEVELS):
            a = self._group(actual, level).to_numpy()
            p = self._group(preds, level).to_numpy()

            mse = ((a - p) ** 2).mean(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                rmsse = np.sqrt(mse / self.scales[i])
            rmsse = np.nan_to_num(rmsse, nan=0.0, posinf=0.0)

            results[level_name(level)] = float((rmsse * self.weights[i]).sum())

        overall = float(np.mean(list(results.values())))
        return overall, pd.Series(results)


def load(cutoff, horizon=28):
    """Split the raw sales file at `cutoff` and return a ready scorer."""
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    train_days = [f"d_{i}" for i in range(1, cutoff + 1)]
    valid_days = [f"d_{i}" for i in range(cutoff + 1, cutoff + horizon + 1)]
    return WRMSSE(sales[ID_COLS + train_days], sales[valid_days])


def naive(cutoff, horizon=28):
    """Repeat the last 28 days. The forecast anything else has to beat."""
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    days = [f"d_{i}" for i in range(cutoff - horizon + 1, cutoff + 1)]
    return sales[days].to_numpy(dtype=np.float64)


if __name__ == "__main__":
    cutoff = 1913
    print(f"Scoring on d_{cutoff + 1}..d_{cutoff + 28}\n")

    scorer = load(cutoff)
    overall, per_level = scorer.score(naive(cutoff))

    print(f"naive (repeat last 28 days):  WRMSSE = {overall:.4f}\n")
    print(per_level.round(4).to_string())
