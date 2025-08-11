# Merlin 4.0

A Streamlit + Flask/CLI app for multi-asset ML with optional news & social features, self‑improving training, and a "simulate live" backtest.

## Features
- Alpha Vantage & yfinance ingest
- Daily features + optional News/Social features
- Train (GB/RF), Predict, Walk‑forward backtest
- Self‑improvement: quick random search if CV AUC < 0.55 + suggestions recorded
- Suggestions box in UI
- Social buzz vs volume spike decision helper
- Save predictions to DB

## Quick start
```bash
python -m venv .venv && . .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
export FLASK_APP=app.py
python setup_sqlite.py
# optional: export ALPHA_VANTAGE_API_KEY=...
flask stocks get_stock_data 1  # fetch some prices
streamlit run ui_app.py
```

## Notes
- Reddit mentions use `snscrape` (no API keys). You can plug in other providers later.
- News sentiment uses Alpha Vantage NEWS; set `ALPHA_VANTAGE_API_KEY`.
