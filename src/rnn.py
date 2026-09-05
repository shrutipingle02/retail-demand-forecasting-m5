"""Approach 2 - LSTM and GRU.

Flip the sales table so days are rows and all 30,490 series are
columns, scale it 0-1, then learn "given the last N days of every series,
predict tomorrow for every series". One network handles all series at once.

Forecasting 28 days is done by predicting one day, feeding that prediction back
in, and repeating.

    python -m src.rnn            # the LSTM vs GRU grid, then tune GRU
    python -m src.rnn --seeds 3  # best GRU three times, to see the spread
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from . import wrmsse as W
from .config import HORIZON, RAW, REPORTS, SEED, set_seed

CUTOFF = 1913  # train on d_1..d_1913, forecast d_1914..d_1941


def load_matrix(start_day, cutoff=CUTOFF):
    """Days as rows, series as columns."""
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    day_cols = [f"d_{i}" for i in range(start_day + 1, cutoff + 1)]
    return sales[day_cols].T.reset_index(drop=True)


def event_flags(start_day, n_days):
    """1 if tomorrow is a holiday. The only extra input the network gets."""
    calendar = pd.read_csv(RAW / "calendar.csv")
    flag = np.zeros((len(calendar), 1))
    flag[np.where(calendar["event_name_1"].notna())[0] - 1] = 1
    return pd.DataFrame(flag[start_day:start_day + n_days], columns=["event_tomorrow"])


def make_sequences(sales_df, event_df, time_steps):
    """Sliding windows: time_steps days in, next day out."""
    scaler = MinMaxScaler()
    merged = np.hstack([
        sales_df.to_numpy(dtype=np.float32),
        event_df.to_numpy(dtype=np.float32),
    ])
    scaled = scaler.fit_transform(merged).astype(np.float32)

    n_series = sales_df.shape[1]
    X = np.lib.stride_tricks.sliding_window_view(scaled, time_steps, axis=0)
    X = X[:-1].transpose(0, 2, 1)          # (samples, time_steps, features)
    y = scaled[time_steps:, :n_series]     # next day, series only
    return X, y, scaler, scaled


def build_model(kind, input_shape, n_out, units=64, dropout=0.2):
    layer = LSTM if kind == "LSTM" else GRU
    model = Sequential([
        layer(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        layer(units),
        Dropout(dropout),
        Dense(n_out),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def forecast(model, scaled, scaler, time_steps, n_series, start_day, horizon=HORIZON):
    """Predict one day, feed it back, repeat."""
    future_events = event_flags(start_day + len(scaled), horizon).to_numpy()
    window = scaled[-time_steps:].copy()
    preds = []

    for step in range(horizon):
        nxt = model.predict(window[np.newaxis, ...], verbose=0)[0]
        preds.append(nxt)
        row = np.concatenate([nxt, future_events[step]])
        window = np.vstack([window[1:], row])

    preds = np.array(preds)
    padded = np.concatenate([preds, np.zeros((horizon, 1))], axis=1)
    return scaler.inverse_transform(padded)[:, :n_series].T  # (series, horizon)


def run_grid(scorer, order):
    """LSTM vs GRU across start days and window sizes."""
    rows = []
    for start_day in [0, 150, 365]:
        sales_df = load_matrix(start_day)
        events = event_flags(start_day, len(sales_df))

        for time_steps in [7, 14]:
            X, y, scaler, scaled = make_sequences(sales_df, events, time_steps)

            for kind in ["LSTM", "GRU"]:
                set_seed()
                print(f"\n{kind}  start_day={start_day}  time_steps={time_steps}")
                model = build_model(kind, (X.shape[1], X.shape[2]), y.shape[1])

                t0 = time.time()
                hist = model.fit(X, y, epochs=5, batch_size=32, verbose=2)
                elapsed = time.time() - t0

                preds = forecast(model, scaled, scaler, time_steps,
                                 sales_df.shape[1], start_day)
                score, _ = scorer.score(preds)

                rows.append({"model": kind, "start_day": start_day,
                             "time_steps": time_steps, "wrmsse": round(score, 5),
                             "final_loss": round(hist.history["loss"][-1], 6),
                             "train_sec": round(elapsed)})
                print(f"  WRMSSE {score:.4f}   ({elapsed:.0f}s)")

            del X, y

    return pd.DataFrame(rows)


def tune_gru(scorer, order, start_day=0, time_steps=14):
    """GRU hyperparameter search."""
    sales_df = load_matrix(start_day)
    events = event_flags(start_day, len(sales_df))
    X, y, scaler, scaled = make_sequences(sales_df, events, time_steps)

    grid = [
        {"units": 64, "dropout": 0.2, "batch_size": 32},
        {"units": 128, "dropout": 0.3, "batch_size": 64},
        {"units": 256, "dropout": 0.2, "batch_size": 128},
    ]

    rows, best, best_score = [], None, np.inf
    for params in grid:
        set_seed()
        print(f"\nGRU {params}")
        model = build_model("GRU", (X.shape[1], X.shape[2]), y.shape[1],
                            units=params["units"], dropout=params["dropout"])
        model.fit(X, y, epochs=5, batch_size=params["batch_size"], verbose=2,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])

        preds = forecast(model, scaled, scaler, time_steps, sales_df.shape[1], start_day)
        score, _ = scorer.score(preds)
        rows.append({**params, "wrmsse": round(score, 5)})
        print(f"  WRMSSE {score:.4f}")

        if score < best_score:
            best, best_score = params, score

    return pd.DataFrame(rows), best


def repeat_seeds(scorer, params, n, start_day=0, time_steps=14):
    """Same settings, different seeds - shows how much of a score is luck."""
    sales_df = load_matrix(start_day)
    events = event_flags(start_day, len(sales_df))
    X, y, scaler, scaled = make_sequences(sales_df, events, time_steps)

    scores = []
    for i in range(n):
        set_seed(SEED + i)
        print(f"\nseed {SEED + i}")
        model = build_model("GRU", (X.shape[1], X.shape[2]), y.shape[1],
                            units=params["units"], dropout=params["dropout"])
        model.fit(X, y, epochs=5, batch_size=params["batch_size"], verbose=2)
        preds = forecast(model, scaled, scaler, time_steps, sales_df.shape[1], start_day)
        score, _ = scorer.score(preds)
        scores.append(score)
        print(f"  WRMSSE {score:.4f}")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=0, help="repeat best GRU N times")
    args = ap.parse_args()

    scorer = W.load(CUTOFF)
    order = pd.read_csv(RAW / "sales_train_evaluation.csv", usecols=["id"])["id"]

    grid = run_grid(scorer, order)
    grid.to_csv(REPORTS / "rnn_grid.csv", index=False)
    print("\n" + grid.sort_values("wrmsse").to_string(index=False))

    tuned, best = tune_gru(scorer, order)
    tuned.to_csv(REPORTS / "gru_tuning.csv", index=False)
    print("\n" + tuned.sort_values("wrmsse").to_string(index=False))
    print(f"\nbest GRU: {best}")

    if args.seeds:
        scores = repeat_seeds(scorer, best, args.seeds)
        print(f"\n{args.seeds} seeds: " + ", ".join(f"{s:.4f}" for s in scores))
        print(f"mean {np.mean(scores):.4f}  range {min(scores):.4f}-{max(scores):.4f}")
        print("spread across seeds is the honest error bar")


if __name__ == "__main__":
    main()
