"""Approach 3 - Seq2Seq GRU.

Instead of predicting one day and feeding it back 28 times, this predicts all
28 days in a single pass: an encoder GRU reads the last 14 days, its final
state is repeated 28 times, and a decoder GRU turns that into 28 days of output
for every series at once.

No extra features - just raw sales, kept deliberately bare to fit in memory.

    python -m src.seq2seq
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import GRU, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential

from . import wrmsse as W
from .config import HORIZON, RAW, SEED, set_seed

CUTOFF = 1913
START_DAY = 350
TIME_STEPS = 14
EPOCHS = 10
BATCH_SIZE = 32


def load_scaled(start_day=START_DAY, cutoff=CUTOFF):
    """Days as rows, series as columns, scaled 0-1 across the whole matrix."""
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    day_cols = [f"d_{i}" for i in range(start_day + 1, cutoff + 1)]
    matrix = sales[day_cols].T.to_numpy(dtype=np.float32)

    scaler = MinMaxScaler()
    return scaler.fit_transform(matrix).astype(np.float32), scaler


def make_windows(scaled, time_steps=TIME_STEPS, horizon=HORIZON):
    """X = 14 days in, y = the following 28 days out."""
    n = len(scaled) - time_steps - horizon + 1
    n_series = scaled.shape[1]

    gb = n * (time_steps + horizon) * n_series * 4 / 1024**3
    print(f"  {n} windows, needs about {gb:.1f} GB")

    X = np.empty((n, time_steps, n_series), dtype=np.float32)
    y = np.empty((n, horizon, n_series), dtype=np.float32)
    for i in range(n):
        X[i] = scaled[i:i + time_steps]
        y[i] = scaled[i + time_steps:i + time_steps + horizon]
    return X, y


def build_model(n_series, time_steps=TIME_STEPS, horizon=HORIZON, units=64):
    model = Sequential([
        GRU(units, input_shape=(time_steps, n_series)),
        RepeatVector(horizon),
        GRU(units, return_sequences=True),
        TimeDistributed(Dense(n_series)),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    set_seed()

    print("Loading and scaling...")
    scaled, scaler = load_scaled()
    n_series = scaled.shape[1]
    print(f"  matrix {scaled.shape}")

    print("Building windows...")
    X, y = make_windows(scaled)

    model = build_model(n_series)
    print(f"  {model.count_params():,} parameters\n")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

    # Forecast from the most recent 14 real days, all 28 at once.
    last_window = scaled[-TIME_STEPS:][np.newaxis, ...]
    preds = model.predict(last_window, verbose=0)[0]        # (28, n_series)
    preds = scaler.inverse_transform(preds).T               # (n_series, 28)
    preds = np.clip(preds, 0, None)

    scorer = W.load(CUTOFF)
    score, per_level = scorer.score(preds)

    print("\n" + "=" * 58)
    print(f"Seq2Seq GRU   WRMSSE = {score:.4f}")
    print("=" * 58)
    print(per_level.round(4).to_string())

    np.save(W.RAW.parent / "data" / "preds_seq2seq.npy", preds)


if __name__ == "__main__":
    main()
