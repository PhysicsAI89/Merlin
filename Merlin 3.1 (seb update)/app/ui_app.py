# app/ui_app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

from models.stockmarket import db_session, Stock, StockDaily
from commands.ml import (
    load_prices, make_features, FEATURE_COLUMNS,
    train_for_symbol, _load_model, _build_pipeline,
    StockPrediction, ModelSuggestion, walkforward_simulate, recommend_trade_now, DecisionThresholds
)

# Import Alpha Vantage helpers (repo may place them in different module)
try:
    from commands.stocks import call_alphavantage, write_price_data, _av_symbol
except Exception:
    from stocks import call_alphavantage, write_price_data, _av_symbol  # fallback

from commands.buzz import fetch_forum_buzz, upsert_buzz_count, get_buzz_baseline

from sklearn.model_selection import TimeSeriesSplit, cross_val_score

@st.cache_data(ttl=300, show_spinner=False)
def load_prices_cached(sym: str):
    return load_prices(sym)

st.set_page_config(page_title="Merlin Stocks ML", layout="wide")
st.title("📈 Merlin — Stock ML - Seb Update")

# Sidebar: API key
st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input(
    "Alpha Vantage API Key",
    value=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
    type="password",
)
if api_key_input:
    os.environ["ALPHA_VANTAGE_API_KEY"] = api_key_input

# Load enabled symbols
symbols = [s.symbol for s in db_session.query(Stock).filter(Stock.enabled == True).order_by(Stock.symbol)]
if not symbols:
    st.warning("No symbols found in DB. Add at least one in the stock table.")
    st.stop()
symbol = st.selectbox("Symbol", symbols)

# --- Top action buttons ---
c1, c2, c3, c4 = st.columns(4)

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
                    st.rerun()
                else:
                    msg = data.get("Note") or data.get("Error Message") if isinstance(data, dict) else None
                    st.warning(f"No data returned for {symbol}. {msg or ''}")

with c2:
    if st.button("Train Model"):
        res = train_for_symbol(symbol, task="classification")
        st.success(f"Trained {symbol} — CV ROC AUC: {res.cv_score:.3f} ({res.n_samples} samples)")

with c3:
    if st.button("Evaluate"):
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

with c4:
    save_to_db = st.checkbox("Save prediction")
    if st.button("Predict T+1"):
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
                    if stk:
                        rec = StockPrediction(
                            stock_id=stk.id,
                            symbol=symbol,
                            price_date=pd.to_datetime(last_row["price_date"]).to_pydatetime(),
                            target_date=(pd.to_datetime(last_row["price_date"]) + pd.Timedelta(days=1)).to_pydatetime(),
                            pred_type="direction",
                            pred_value=proba,
                            model_name="GBClassifier_v1",
                            model_version="1",
                        )
                        db_session.add(rec)
                        db_session.commit()
                        st.toast("Prediction saved to stock_prediction", icon="✅")
            except FileNotFoundError:
                st.warning("Model not found — train it first.")

# --- Price chart ---
st.subheader(f"{symbol} — Closing Price")
prices = load_prices_cached(symbol)
if not prices.empty:
    prices = prices.copy()
    prices["price_date"] = pd.to_datetime(prices["price_date"])
    prices.sort_values("price_date", inplace=True)
    st.line_chart(prices.set_index("price_date")["close_price"])
else:
    st.write("No data yet. Click **Fetch latest data** to populate the database.")

# --- Buzz monitor & decision ---
st.subheader("🗣️ Buzz monitor & trade decision")
b1, b2, b3 = st.columns([1,1,2])
with b1:
    hold_steps = st.number_input("Hold steps", min_value=1, max_value=30, value=5)
with b2:
    prob_thr = st.slider("Buy threshold (Pr(up))", 0.5, 0.9, 0.60, 0.01)

buzz_clicked = st.button("Fetch buzz now")
if buzz_clicked:
    today = pd.Timestamp.utcnow().date()
    counts = fetch_forum_buzz(symbol)
    for src in ("reddit", "stocktwits"):
        upsert_buzz_count(symbol, src, today, counts.get(src, 0))

# compute buzz ratios & volume factor
today_total, baseline = get_buzz_baseline(symbol, days=30)
buzz_factor = (today_total / baseline) if baseline and baseline > 0 else 0.0
df_feat = make_features(prices) if not prices.empty else pd.DataFrame()
volume_factor = 0.0
proba_up = 0.0
last_date = None
if not df_feat.empty:
    last = df_feat.iloc[-1]
    volume_factor = float((last["volume"] / max(1e-9, last["volume_ma20"])) if last.get("volume_ma20", 0) else 0.0)
    last_date = pd.to_datetime(last["price_date"]).to_pydatetime()
    try:
        blob = _load_model(symbol, "classification")
        proba_up = float(blob["pipeline"].predict_proba([last[FEATURE_COLUMNS].values])[0, 1])
    except FileNotFoundError:
        pass

thr = DecisionThresholds(prob_buy=prob_thr, hold_steps=hold_steps)
if last_date is not None:
    decision = recommend_trade_now(symbol, last_date, proba_up, volume_factor, buzz_factor, thr)
    st.info(
        f"{symbol} was mentioned **{today_total}** times today "
        f"(baseline ~ **{baseline:.1f}**, buzz factor **{buzz_factor:.2f}**). "
        f"Volume factor **{volume_factor:.2f}**. Pr(up) **{proba_up:.2f}** → "
        f"**{decision['action'].upper()}** (confidence {decision['confidence']:.2f}); "
        f"target sell: **{decision['target_sell_date'].date()}**.\n\n"
        f"Reason: {decision['reason']}"
    )
else:
    st.write("Need at least 21 days of data for volume baseline and a trained model for Pr(up).")

# --- Walk-forward sim (paper trade) + suggestions box ---
st.subheader("🧪 Simulate live on past data (walk-forward) & self-improve")
cA, cB, cC = st.columns(3)
with cA:
    sim_thr = st.slider("Sim buy threshold", 0.5, 0.9, 0.60, 0.01, key="sim_thr")
with cB:
    sim_hold = st.number_input("Sim hold steps", 1, 30, 5, key="sim_hold")
with cC:
    sim_retrain = st.number_input("Retrain every k steps", 1, 100, 10, key="sim_retrain")

if st.button("Run walk-forward simulation"):
    summary, suggs = walkforward_simulate(symbol, sim_thr, sim_hold, sim_retrain)
    st.success(f"Steps: {summary.n_steps} | Trades: {summary.n_trades} | "
               f"Win rate: {summary.win_rate:.2f} | AUC: {summary.auc:.2f} | "
               f"Avg ret/trade: {summary.avg_return_per_trade:.4f}")
    if suggs:
        st.write("Suggestions generated:")
        for s in suggs:
            st.write(f"• {s}")
    else:
        st.write("No suggestions this run — model looks stable.")

st.subheader("🧠 Self-improvement suggestions (latest)")
from commands.ml import ModelSuggestion as _MS  # avoid name shadow
suggestions = (
    db_session.query(_MS)
    .filter(_MS.symbol == symbol)
    .order_by(_MS.id.desc())
    .limit(20).all()
)
if suggestions:
    df_s = pd.DataFrame([{
        "created_at": s.created_at,
        "tag": s.tag,
        "suggestion": s.suggestion,
        "metric_name": s.metric_name,
        "metric_value": s.metric_value
    } for s in suggestions])
    st.dataframe(df_s, hide_index=True, use_container_width=True)
else:
    st.write("No suggestions stored yet — run a simulation above.")

# --- Recent predictions table ---
st.subheader(f"{symbol} — Recent predictions")
from commands.ml import StockPrediction as _SP  # avoid name shadow
preds = (
    db_session.query(_SP)
    .filter(_SP.symbol == symbol)
    .order_by(_SP.id.desc())
    .limit(10)
    .all()
)
if preds:
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
