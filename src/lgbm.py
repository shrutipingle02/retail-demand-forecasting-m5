"""Approach 1 - LightGBM.

One model per store per 7-day block of the horizon: 10 stores x 4 blocks = 40
models. Tweedie loss, because most item-days are zero.

Scored two different ways, which give wildly different answers:

  sales visible - score the holdout with sales still in the table. The leaked
                  target makes this look near-perfect.
  sales hidden  - score the holdout with sales blanked out, which is what
                  actually happens when you forecast a real future. The leak
                  has nothing to read and the model falls apart.

    python -m src.lgbm
"""

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from . import features as F
from . import wrmsse as W
from .config import DATA, HORIZON, MODELS, SEED, VAL_END, VAL_START, set_seed

# Tuned for zero-inflated daily demand.
PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "subsample": 0.75,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "max_depth": -1,
    "num_leaves": 64,
    "min_child_samples": 100,
    "verbose": -1,
    "seed": SEED,
    "num_threads": 0,
}
ROUNDS = 500

BLOCKS = [(1, 7), (8, 14), (15, 21), (22, 28)]


def blank_future(cutoff):
    """Rebuild features with sales hidden after `cutoff`.

    At real forecast
    time the sales column is empty, so every feature derived from it is empty
    too. Returns only the rows we need to predict.
    """
    print(f"Rebuilding features with sales hidden after d_{cutoff}...")
    df = pd.read_parquet(DATA / "long.parquet")
    df.loc[df["d"] > cutoff, "sales"] = np.nan
    df = F.build(df)
    df = df[df["d"] > cutoff].reset_index(drop=True)
    gc.collect()
    return df


def train(df, cutoff, feats):
    """40 models: one per store, per 7-day block."""
    train_df = df[df["d"] <= cutoff]
    models = {}

    for store in sorted(train_df["store_id"].unique()):
        rows = train_df[train_df["store_id"] == store]
        X, y = rows[feats], rows["sales"]

        for start, end in BLOCKS:
            key = f"{store}_F{start:02d}_F{end:02d}"
            models[key] = lgb.train(PARAMS, lgb.Dataset(X, label=y), num_boost_round=ROUNDS)
            print(f"  {key}  ({len(rows):,} rows)")

        del X, y
        gc.collect()

    return models


def predict(models, df, cutoff, feats):
    """Predict each 7-day block with the model trained for it."""
    out = []
    for store in sorted(df["store_id"].unique()):
        for start, end in BLOCKS:
            key = f"{store}_F{start:02d}_F{end:02d}"
            window = df[
                (df["store_id"] == store)
                & (df["d"] > cutoff + start - 1)
                & (df["d"] <= cutoff + end)
            ]
            if window.empty:
                continue
            preds = models[key].predict(window[feats])
            out.append(pd.DataFrame({"id": window["id"].to_numpy(),
                                     "d": window["d"].to_numpy(),
                                     "pred": preds}))
    return pd.concat(out, ignore_index=True)


def to_matrix(preds, order):
    """Long predictions -> 30490 x 28 matrix in the scorer's row order."""
    wide = preds.pivot(index="id", columns="d", values="pred")
    wide = wide.reindex(order).fillna(0)
    return wide.to_numpy()


def main():
    set_seed()
    cutoff = VAL_START - 1  # train through d_1913, forecast d_1914..d_1941

    df = pd.read_parquet(DATA / "features.parquet")
    feats = F.feature_list(df, leak=True)
    print(f"{len(feats)} features, training through d_{cutoff}\n")

    models = train(df, cutoff, feats)
    for key, model in models.items():
        model.save_model(str(MODELS / f"lgbm_{key}.txt"))

    order = pd.read_csv(W.RAW / "sales_train_evaluation.csv", usecols=["id"])["id"]
    scorer = W.load(cutoff)

    # --- sales still present in the holdout rows ----------------------------
    holdout = df[(df["d"] >= VAL_START) & (df["d"] <= VAL_END)]
    leaked = predict(models, df, cutoff, feats)
    merged = holdout[["id", "d", "sales"]].merge(leaked, on=["id", "d"])
    rmse = float(np.sqrt(mean_squared_error(merged["sales"], merged["pred"])))
    leaked_wrmsse, _ = scorer.score(to_matrix(leaked, order))

    del df
    gc.collect()

    # --- sales blanked out, as at real forecast time -------------------------
    future = blank_future(cutoff)
    honest = predict(models, future, cutoff, feats)
    honest_wrmsse, per_level = scorer.score(to_matrix(honest, order))

    print("\n" + "=" * 58)
    print("LightGBM")
    print("=" * 58)
    print(f"  holdout RMSE, sales visible   {rmse:.4f}")
    print(f"  WRMSSE, sales visible         {leaked_wrmsse:.4f}")
    print(f"  WRMSSE, sales hidden          {honest_wrmsse:.4f}")
    print("=" * 58)
    print("\nBy level, sales hidden:")
    print(per_level.round(4).to_string())

    honest.to_parquet(DATA / "preds_lgbm.parquet", index=False)


if __name__ == "__main__":
    main()
