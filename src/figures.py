"""Charts for the write-up and the deck.

Retrains the best GRU once so its predictions can be plotted, then writes five
figures to reports/figures/.

    python -m src.figures
"""

import matplotlib

matplotlib.use("Agg")

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from . import rnn as R
from . import wrmsse as W
from .config import DATA, MODELS, RAW, REPORTS, set_seed

FIGS = REPORTS / "figures"
FIGS.mkdir(exist_ok=True)

# Chart chrome, from the reference palette.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#d03b3b"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def style(ax, xgrid=False):
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(0.8)


def save(fig, name):
    path = FIGS / name
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path.name}")


# --------------------------------------------------------------- get the data

def best_gru():
    """Retrain the grid's best config and return its 28-day forecast."""
    print("Retraining best GRU (start_day=150, window=14)...")
    set_seed()
    sales_df = R.load_matrix(150)
    events = R.event_flags(150, len(sales_df))
    X, y, scaler, scaled = R.make_sequences(sales_df, events, 14)

    model = R.build_model("GRU", (X.shape[1], X.shape[2]), y.shape[1])
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return R.forecast(model, scaled, scaler, 14, sales_df.shape[1], 150)


def lgbm_preds(order):
    df = pd.read_parquet(DATA / "preds_lgbm.parquet")
    wide = df.pivot(index="id", columns="d", values="pred").reindex(order).fillna(0)
    return wide.to_numpy()


# ------------------------------------------------------------------- figures

def fig_scoreboard(scores):
    """Every model against the naive baseline."""
    order = sorted(scores, key=scores.get)
    vals = [scores[k] for k in order]
    colors = [MUTED if k == "Naive baseline" else (RED if scores[k] > 2 else BLUE)
              for k in order]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    bars = ax.barh(order, vals, color=colors, height=0.62, zorder=3)
    ax.axvline(scores["Naive baseline"], color=INK2, linewidth=1.4, linestyle=(0, (4, 3)), zorder=4)
    ax.text(scores["Naive baseline"] + 0.06, -0.72, "naive baseline",
            color=INK2, fontsize=9, va="center")

    for bar, v in zip(bars, vals):
        ax.text(v + 0.08, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9.5, color=INK)

    ax.set_xlim(0, max(vals) * 1.16)
    ax.set_xlabel("WRMSSE  (lower is better)")
    ax.invert_yaxis()
    style(ax, xgrid=True)
    save(fig, "01_scoreboard.png")


def fig_leak():
    """The same model scored with and without the target column."""
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    labels = ["Target column\nvisible", "Target column\nhidden"]
    vals = [0.0088, 5.4464]

    bars = ax.bar(labels, vals, color=[AQUA, RED], width=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.004, 20)
    ax.set_ylabel("WRMSSE  (log scale)")

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.25, f"{v:.4f}",
                ha="center", fontsize=10.5, color=INK, fontweight="bold")

    ax.annotate("", xy=(1, 5.4464), xytext=(0, 0.0088),
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8, linestyle=":"))
    ax.text(0.5, 0.35, "620x", ha="center", fontsize=13, color=INK2, fontweight="bold")
    style(ax)
    save(fig, "02_leak.png")


def fig_by_level(levels):
    """Where each model wins and loses across the hierarchy."""
    names = list(levels["naive"].index)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))

    ax.plot(names, levels["naive"], color=MUTED, linewidth=2, marker="o",
            markersize=5, label="Naive baseline", zorder=3)
    ax.plot(names, levels["gru"], color=BLUE, linewidth=2, marker="o",
            markersize=5, label="GRU", zorder=4)
    ax.plot(names, levels["lgbm"], color=RED, linewidth=2, marker="o",
            markersize=5, label="LightGBM", zorder=5)

    ax.set_ylabel("WRMSSE")
    ax.set_xlabel("aggregation level      (company total  ->  single item in one store)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style(ax)
    save(fig, "03_by_level.png")


def fig_forecast(gru_preds, naive_preds):
    """Total daily units: what happened, and what each model said would happen."""
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    hist_days = [f"d_{i}" for i in range(1857, 1914)]
    val_days = [f"d_{i}" for i in range(1914, 1942)]

    history = sales[hist_days].sum().to_numpy()
    actual = sales[val_days].sum().to_numpy()

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    hx = np.arange(-len(history), 0)
    fx = np.arange(0, 28)

    ax.plot(hx, history, color=INK2, linewidth=1.6, zorder=3)
    ax.plot(fx, actual, color=INK2, linewidth=2.2, label="Actual", zorder=5)
    ax.plot(fx, gru_preds.sum(axis=0), color=BLUE, linewidth=2, label="GRU forecast", zorder=4)
    ax.plot(fx, naive_preds.sum(axis=0), color=YELLOW, linewidth=2,
            linestyle=(0, (5, 3)), label="Naive forecast", zorder=4)

    ax.axvline(0, color=AXIS, linewidth=1)
    ax.text(0.6, ax.get_ylim()[1] * 0.97, "forecast starts", fontsize=8.5,
            color=MUTED, va="top")

    ax.set_ylabel("total units sold per day")
    ax.set_xlabel("days relative to forecast start")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    style(ax)
    save(fig, "04_forecast.png")


def fig_importance():
    """What the tree model actually leaned on."""
    from . import features as F

    booster = lgb.Booster(model_file=str(MODELS / "lgbm_0_F01_F07.txt"))
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    gain = (gain / gain.sum() * 100).sort_values(ascending=False).head(8)[::-1]

    colors = [RED if n == "sales" else BLUE for n in gain.index]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.barh(gain.index, gain.to_numpy(), color=colors, height=0.62, zorder=3)
    for bar, v in zip(bars, gain.to_numpy()):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=9, color=INK)

    ax.set_xlim(0, 92)
    ax.set_xlabel("share of total model gain")
    style(ax, xgrid=True)
    save(fig, "05_importance.png")


def fig_grid():
    """The LSTM vs GRU sweep."""
    g = pd.read_csv(REPORTS / "rnn_grid.csv")
    fig, ax = plt.subplots(figsize=(6.6, 3.0))

    for kind, color in [("GRU", BLUE), ("LSTM", ORANGE)]:
        sub = g[g.model == kind]
        labels = [f"d{r.start_day}/w{r.time_steps}" for r in sub.itertuples()]
        ax.scatter(labels, sub.wrmsse, s=64, color=color, label=kind, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)

    ax.set_ylabel("WRMSSE")
    ax.set_xlabel("training start day / input window")
    ax.legend(frameon=False, fontsize=9)
    plt.setp(ax.get_xticklabels(), fontsize=8.5)
    style(ax)
    save(fig, "06_grid.png")


def main():
    order = pd.read_csv(RAW / "sales_train_evaluation.csv", usecols=["id"])["id"]
    scorer = W.load(1913)

    naive_preds = W.naive(1913)
    gru_preds = best_gru()
    lgb_preds = lgbm_preds(order)

    print("\nScoring...")
    naive_s, naive_lv = scorer.score(naive_preds)
    gru_s, gru_lv = scorer.score(gru_preds)
    lgbm_s, lgbm_lv = scorer.score(lgb_preds)
    print(f"  naive {naive_s:.4f} | gru {gru_s:.4f} | lgbm {lgbm_s:.4f}")

    scores = {
        "Naive baseline": naive_s,
        "GRU": gru_s,
        "LSTM": 1.09533,
        "Seq2Seq GRU": 1.10950,
        "LightGBM": lgbm_s,
    }
    pd.Series(scores).to_csv(REPORTS / "final_scores.csv")
    pd.DataFrame({"naive": naive_lv, "gru": gru_lv, "lgbm": lgbm_lv}).to_csv(
        REPORTS / "scores_by_level.csv")

    print("\nDrawing...")
    fig_scoreboard(scores)
    fig_leak()
    fig_by_level({"naive": naive_lv, "gru": gru_lv, "lgbm": lgbm_lv})
    fig_forecast(gru_preds, naive_preds)
    fig_importance()
    fig_grid()


if __name__ == "__main__":
    main()
