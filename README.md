# Walmart Demand Forecasting

Forecasting 28 days of daily sales for 30,490 Walmart item-store series, using the
Kaggle [M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy) dataset.

3,049 products × 10 stores. Jan 2011 – Apr 2016. 68% of item-days are zero.

---

## 📁 Files

| File | What it does |
|------|--------------|
| `src/prep.py` | Wide sales table → long table, joined with calendar and prices |
| `src/features.py` | Lag, rolling, calendar and price features |
| `src/lgbm.py` | LightGBM, one model per store per week |
| `src/rnn.py` | LSTM and GRU |
| `src/seq2seq.py` | Encoder–decoder GRU, all 28 days at once |
| `src/wrmsse.py` | The competition metric, all 12 levels |
| `src/figures.py` | The charts below |
| `reports/writeup.pdf` | Full write-up |
| `reports/deck.pdf` | Slides |

---

## 📦 Data

Not included — too large. Download from
[Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and unzip
into `m5-forecasting-accuracy/`.

---

## ▶️ Running it

```bash
pip install -r requirements.txt

python -m src.prep        # ~20 s
python -m src.features    # ~60 s
python -m src.lgbm        # ~20 min
python -m src.rnn         # ~13 min
python -m src.seq2seq     # ~5 min
```

---

## 🏁 Results

Scored with WRMSSE on days 1914–1941. Lower is better.

![Results](reports/figures/01_scoreboard.png)

| Model | WRMSSE |
|-------|--------|
| Naive baseline (repeat last 28 days) | **0.8377** |
| GRU | 1.0365 |
| LSTM | 1.0953 |
| Seq2Seq GRU | 1.1095 |
| LightGBM | 5.4464 |

**No model beat the baseline.** The best one is 24% worse than repeating last month's
sales.

---

## 📊 Why

![Forecast vs actual](reports/figures/04_forecast.png)

Sales follow a weekly cycle. The naive forecast copies it. The GRU flattens out near the
average — it gets the level right and loses the shape. That happens because each
prediction is fed back in as input, so errors build up and the signal washes out.

---

## 🔍 On the LightGBM score

It first scored 0.0088, which is impossible. The target column had ended up in the
feature set, so the model was reading sales instead of predicting them.

![Leakage](reports/figures/02_leak.png)

Same model, same 28 days. The only difference is whether the answer was in the input.
That one column was 78% of the model's gain.

The switch is still there (`features.feature_list(leak=...)`) so the effect can be
measured rather than described.

---

## 🚀 Next steps

- Train past 5 epochs — the loss was still falling
- Give the network the day of the week; it is never told
- Predict all 28 days at once instead of feeding predictions back in
- Use a chronological validation split, not Keras' default

---

## 📜 License

MIT
