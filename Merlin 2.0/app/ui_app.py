# app/ui_app.py
import os
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import desc


from models.stockmarket import db_session, Stock
from commands.ingest import ingest_yf
from commands.ml import (
    load_prices, make_features, FEATURE_COLUMNS,
    train_for_symbol, _load_model, predict_one, backtest,
    StockPrediction, add_news_features
)
from commands.stocks import call_alphavantage, write_price_data, _av_symbol

st.set_page_config(page_title="Merlin Multi-Asset ML", layout="wide")
st.title("🚀 Merlin — Multi-Asset ML (Daily/Weekly)")

@st.cache_data(ttl=300, show_spinner=False)
def load_prices_cached(sym: str, freq: str):
    return load_prices(sym, freq=freq)

st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input(
    "Alpha Vantage API Key",
    value=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
    type="password"
)
if api_key_input:
    os.environ["ALPHA_VANTAGE_API_KEY"] = api_key_input

symbols = [s.symbol for s in db_session.query(Stock)
           .filter(Stock.enabled == True)
           .order_by(Stock.symbol)]

if not symbols:
    st.warning("No symbols in DB yet. Add some in the DB or use the YF buttons to ingest first.")
    st.stop()

with st.expander("Ingest (yfinance) — crypto/FX/oil/equities", expanded=False):
    st.markdown("You can also run this in a terminal to bulk load:")
    st.code('python -m flask --app app ingest yf --symbols "BTC-USD,ETH-USD,GBPUSD=X,CL=F"', language="bash")

sym = st.selectbox("Asset", symbols)

cA, cB, cC, cD = st.columns(4)
with cA:
    freq = st.radio("Frequency", ["D", "W"], index=0, horizontal=True)
with cB:
    horizon = st.number_input("Horizon (steps ahead)", min_value=1, max_value=8, value=1, step=1)
with cC:
    model_choice = st.selectbox("Model", ["gb", "rf", "xgb"], index=0)
with cD:
    use_news = st.checkbox("Use news sentiment", value=False)

# ---- Fetch buttons ----
b1, b2, b3 = st.columns(3)

with b1:
    if st.button("Fetch latest equity data (AV)"):
        srow = db_session.query(Stock).filter(Stock.symbol == sym).first()
        if not srow:
            st.error("Symbol not found in DB.")
        else:
            apikey = os.getenv("ALPHA_VANTAGE_API_KEY", "")
            if not apikey:
                st.warning("Set ALPHA_VANTAGE_API_KEY in the sidebar first.")
            else:
                data = call_alphavantage(_av_symbol(sym), apikey)
                if isinstance(data, dict) and "Time Series (Daily)" in data:
                    rows = write_price_data(srow.id, srow.symbol, data["Time Series (Daily)"])
                    load_prices_cached.clear()
                    st.success(f"Inserted {rows} rows for {sym}.")
                else:
                    st.warning("No data returned (symbol might not be an equity, or rate-limited).")

with b2:
    if st.button("Fetch latest (YF) for this asset"):
        res = ingest_yf([sym], period="5y", interval="1d")
        load_prices_cached.clear()
        st.success(f"Inserted {res.get(sym, 0)} new rows for {sym}.")

with b3:
    if st.button("Fetch ALL enabled (YF)"):
        syms_enabled = [s.symbol for s in db_session.query(Stock).filter(Stock.enabled == True)]
        if not syms_enabled:
            st.warning("No enabled symbols in DB.")
        else:
            res = ingest_yf(syms_enabled, period="5y", interval="1d")
            load_prices_cached.clear()
            st.success(f"Updated {len(res)} symbols; total inserted: {sum(res.values())}.")

# ---- Actions ----
c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Train"):
        res = train_for_symbol(sym, freq=freq, horizon=horizon, use_news=use_news, model=model_choice)
        st.success(f"Trained {sym} — {res.model_name} ROC AUC: {res.cv_score:.3f} ({res.n_samples} samples)")


with c2:
    # --- choose parameters first ---
    thr = st.slider("Buy threshold (prob)", 0.50, 0.80, 0.60, 0.01, key="thr")
    hold = st.slider("Hold steps", 1, 8, max(1, horizon), 1, key="hold")
    re_every = st.slider("Retrain every k steps", 1, 20, 5, 1, key="retrain_every")

    # --- run backtest once when you click ---
    if st.button("Backtest strategy"):
        with st.spinner("Running backtest..."):
            final, hit, positions = backtest(
                sym,
                freq=freq,
                horizon=horizon,
                model=model_choice,
                threshold=thr,
                hold_period=hold,
                use_news=use_news,
                retrain_every=re_every,   # new speed knob
            )

        st.write(f"**Trades**: {len(positions)}  |  **Hit rate**: {hit:.2f}  |  **Final equity**: {final:.2f}x")

        if positions:
            dfp = pd.DataFrame(
                [{"date": pd.to_datetime(d), "proba": p, "rtn": r} for (d, p, r) in positions]
            )
            st.dataframe(dfp.tail(20), use_container_width=True)



with c3:
    save_to_db = st.checkbox("Save prediction")
    if st.button("Predict next"):
        model_name = f"{model_choice.upper()}_{freq}_H{horizon}"
        proba = predict_one(sym, model_name, freq=freq, horizon=horizon, use_news=use_news, write=save_to_db)
        st.success(f"Pr(up) = {proba:.3f}")

# ---- Chart ----
st.subheader(f"{sym} — {'Weekly' if freq == 'W' else 'Daily'} Close")
prices = load_prices_cached(sym, freq=freq)
if use_news and not prices.empty:
    prices = add_news_features(sym, prices, os.getenv("ALPHA_VANTAGE_API_KEY", ""))

if not prices.empty:
    prices = prices.copy()
    prices["price_date"] = pd.to_datetime(prices["price_date"])
    prices.sort_values("price_date", inplace=True)
    st.line_chart(prices.set_index("price_date")["close_price"])
else:
    st.info("No price data yet. Use the YF buttons above (or the AV button for equities).")






# ---- Saved predictions table ----
st.subheader(f"Saved predictions — {sym}")

# How many rows to show
rows_to_show = st.slider("Show last N rows", 10, 500, 100, 10, key="pred_rows")

# Pull the most recent predictions for the selected symbol
q = (
    db_session.query(StockPrediction)
    .filter(StockPrediction.symbol == sym)
    .order_by(desc(StockPrediction.id))
    .limit(rows_to_show)
)
dfp = pd.read_sql(q.statement, db_session.bind)

if dfp.empty:
    st.info("No predictions saved yet for this symbol. Tick **Save prediction** and click **Predict next**.")
else:
    # Tidy up for display
    dfp["price_date"] = pd.to_datetime(dfp["price_date"])
    dfp["target_date"] = pd.to_datetime(dfp["target_date"])
    dfp = dfp.rename(columns={
        "pred_value": "proba",
        "model_name": "model"
    })
    # Optional: quick summary
    st.caption(
        f"Rows: {len(dfp)} • Avg proba: {dfp['proba'].mean():.3f} • "
        f"≥0.60: {(dfp['proba'] >= 0.60).mean():.0%}"
    )

    st.dataframe(dfp, use_container_width=True, hide_index=True)

    st.download_button(
        "Download CSV",
        dfp.to_csv(index=False).encode(),
        file_name=f"{sym}_predictions.csv",
        mime="text/csv"
    )
