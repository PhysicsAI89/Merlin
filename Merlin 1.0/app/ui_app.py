# ui_app.py
import os
import numpy as np
import pandas as pd
import streamlit as st

from models.stockmarket import db_session, Stock, StockDaily
from commands.ml import (
    load_prices, make_features, FEATURE_COLUMNS,
    train_for_symbol, _load_model, _build_pipeline
)
from commands.stocks import call_alphavantage, write_price_data, _av_symbol  # uses your fetch & insert

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sqlalchemy.sql import func
from commands.ml import StockPrediction
import streamlit as st

@st.cache_data(ttl=300, show_spinner=False)
def load_prices_cached(sym: str):
    return load_prices(sym)

st.set_page_config(page_title="Merlin Stocks ML", layout="wide")
st.title("📈 Merlin — Stock ML")

# Sidebar: API key (optional; uses env var if set)
st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input(
    "Alpha Vantage API Key",
    value=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
    type="password",
    help="Used to fetch latest prices. Leave blank to use the current environment variable."
)
if api_key_input:
    os.environ["ALPHA_VANTAGE_API_KEY"] = api_key_input

# Load enabled symbols from DB
symbols = [s.symbol for s in db_session.query(Stock).filter(Stock.enabled == True).order_by(Stock.symbol)]
if not symbols:
    st.warning("No symbols found in DB. Add at least one in the stock table.")
    st.stop()

symbol = st.selectbox("Symbol", symbols)

c1, c2, c3, c4 = st.columns(4)

# Fetch latest from Alpha Vantage
with c1:
    if st.button("Fetch latest data"):
        s = db_session.query(Stock).filter(Stock.symbol == symbol).first()
        if not s:
            st.error("Symbol not found in DB.")
        else:
            apikey = os.getenv("ALPHA_VANTAGE_API_KEY", "")
            if not apikey:
                st.warning("Set ALPHA_VANTAGE_API_KEY in the sidebar first.")
            else:
                data = call_alphavantage(_av_symbol(symbol), apikey)
                if isinstance(data, dict) and "Time Series (Daily)" in data:
                    rows = write_price_data(s.id, s.symbol, data["Time Series (Daily)"])
                    load_prices_cached.clear()  
                    st.success(f"Inserted {rows} new rows for {symbol}.")
                    st.rerun() # optional: instantly refresh the page
                else:
                    msg = data.get("Note") or data.get("Error Message") if isinstance(data, dict) else None
                    st.warning(f"No data returned for {symbol}. {msg or ''}")

# Train model
with c2:
    if st.button("Train Model"):
        res = train_for_symbol(symbol, task="classification")
        st.success(f"Trained {symbol} — CV ROC AUC: {res.cv_score:.3f} ({res.n_samples} samples)")

# Evaluate (time-series CV)
with c3:
    if st.button("Evaluate"):
        #prices = load_prices(symbol)
        prices = load_prices_cached(symbol)
        if prices.empty:
            st.warning("No price data — fetch data first.")
        else:
            df = make_features(prices)
            X = df[FEATURE_COLUMNS].values
            y = (df["target_return_1d"] > 0).astype(int).values
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(_build_pipeline("classification"), X, y, cv=tscv, scoring="roc_auc")
            st.info(f"ROC AUC: {scores.mean():.3f} ± {scores.std():.3f}")

# Predict T+1
with c4:
    save_to_db = st.checkbox("Save prediction")  # put checkbox first

    if st.button("Predict T+1"):
        #prices = load_prices(symbol)
        prices = load_prices_cached(symbol)
        if prices.empty:
            st.warning("No price data — fetch data first.")
        else:
            try:
                blob = _load_model(symbol, "classification")
                pipe = blob["pipeline"]

                df = make_features(prices)
                x_last = df[FEATURE_COLUMNS].values[-1]
                last_row = df.iloc[-1]

                proba = float(pipe.predict_proba([x_last])[0, 1])
                st.success(f"Probability {symbol} closes UP tomorrow: {proba:.3f}")

                if save_to_db:
                    stk = db_session.query(Stock).filter(Stock.symbol == symbol).first()
                    rec = StockPrediction(
                        stock_id=stk.id,
                        symbol=symbol,
                        price_date=pd.to_datetime(last_row["price_date"]).to_pydatetime(),
                        target_date=(pd.to_datetime(last_row["price_date"]) + pd.Timedelta(days=1)).to_pydatetime(),
                        pred_type="direction",
                        pred_value=proba,
                        model_name="GBClassifier_v1",
                        model_version="1",
                        # created_at is optional; your model has a default
                    )
                    db_session.add(rec)
                    db_session.commit()
                    st.toast("Prediction saved to stock_prediction", icon="✅")

            except FileNotFoundError:
                st.warning("Model not found — train it first.")

# Chart
st.subheader(f"{symbol} — Closing Price")
#prices = load_prices(symbol)
prices = load_prices_cached(symbol)

if not prices.empty:
    prices = prices.copy()
    prices["price_date"] = pd.to_datetime(prices["price_date"])
    prices.sort_values("price_date", inplace=True)
    st.line_chart(prices.set_index("price_date")["close_price"])
else:
    st.write("No data yet. Click **Fetch latest data** to populate the database.")

st.subheader(f"{symbol} — Recent predictions")
from commands.ml import StockPrediction
preds = (
    db_session.query(StockPrediction)
    .filter(StockPrediction.symbol == symbol)
    .order_by(StockPrediction.id.desc())
    .limit(10)
    .all()
)

if preds:
    import pandas as pd
    dfp = pd.DataFrame([{
        "price_date": p.price_date,
        "target_date": p.target_date,
        "pred_type": p.pred_type,
        "pred_value": round(p.pred_value, 3),
        "model": p.model_name,
    } for p in preds])
    st.dataframe(dfp, hide_index=True, use_container_width=True)
else:
    st.write("No saved predictions yet — tick **Save prediction** and run Predict T+1.")
