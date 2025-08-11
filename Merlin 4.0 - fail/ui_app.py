# Streamlit UI for Merlin 4.0
import os
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

from models.stockmarket import (
    db_session, init_db, Stock, StockDaily, StockPrediction,
    ModelSuggestion
)
from commands.stocks import call_alphavantage, write_price_data, _av_symbol
from commands.ingest import ingest_yf
from commands.news import fetch_av_news
from commands.forums import fetch_reddit_mentions
from commands.ml import (
    load_prices, make_features, train_for_symbol, predict_proba, walk_forward,
)

st.set_page_config(page_title="Merlin 4.0 — Multi-Asset ML", layout="wide")
st.title("🧙 Merlin 4.0 — Multi-Asset ML")

# Init DB
init_db()

# Sidebar controls
st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input(
    "Alpha Vantage API Key",
    value=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
    type="password",
)
if api_key_input:
    os.environ["ALPHA_VANTAGE_API_KEY"] = api_key_input

with st.sidebar:
    st.write("**Features**")
    use_news = st.checkbox("Use news sentiment", value=True)
    use_social = st.checkbox("Use social/forum buzz", value=True)
    model_choice = st.selectbox("Model", ["gb","rf"], index=0, help="gb=Gradient Boosting, rf=Random Forest")
    buy_threshold = st.slider("Buy threshold (Pr(up) ≥)", 0.5, 0.8, 0.55, 0.01)
    hold_steps = st.slider("Hold steps", 1, 20, 5, 1)
    retrain_k = st.slider("Retrain every k steps", 2, 50, 10, 1)
    st.divider()
    stop_key = "stop_flag"
    if st.button("🛑 Stop running"):
        st.session_state[stop_key] = True
        st.warning("Stop flag set. Long loops will stop soon.")
    else:
        st.session_state.setdefault(stop_key, False)

# Load enabled symbols
symbols = [s.symbol for s in db_session.query(Stock).filter(Stock.enabled == True).order_by(Stock.symbol)]
if not symbols:
    st.warning("No symbols found in DB. Add at least one in the stock table.")
    st.stop()
symbol = st.selectbox("Symbol", symbols)

@st.cache_data(ttl=300, show_spinner=False)
def load_prices_cached(sym: str):
    return load_prices(sym)

# --- Data Fetch Row ---
f1, f2, f3, f4 = st.columns(4)

with f1:
    if st.button("Fetch latest (Alpha Vantage)"):
        s = db_session.query(Stock).filter(Stock.symbol == symbol).first()
        apikey = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        if not apikey:
            st.warning("Set ALPHA_VANTAGE_API_KEY first (sidebar).")
        else:
            data = call_alphavantage(_av_symbol(symbol), apikey)
            if isinstance(data, dict) and "Time Series (Daily)" in data:
                rows = write_price_data(s.id, s.symbol, data["Time Series (Daily)"])
                load_prices_cached.clear()
                st.success(f"Inserted {rows} new rows for {symbol}.")
            else:
                msg = data.get("Note") or data.get("Error Message") if isinstance(data, dict) else None
                st.warning(f"No data returned for {symbol}. {msg or ''}")

with f2:
    if st.button("Fetch via yfinance"):
        rows = ingest_yf(symbol)
        load_prices_cached.clear()
        st.success(f"Inserted {rows} rows via yfinance.")

with f3:
    if st.button("Fetch News (AV)"):
        try:
            wrote = fetch_av_news(symbol)
            st.success(f"Inserted {wrote} news rows for {symbol}.")
        except Exception as e:
            st.error(f"News fetch failed: {e}")

with f4:
    if st.button("Fetch Social (Reddit)"):
        try:
            wrote = fetch_reddit_mentions(symbol, days=7)
            st.success(f"Inserted {wrote} social mention rows.")
        except Exception as e:
            st.error(f"Social fetch failed: {e}")

st.divider()

# --- Train / Predict / Backtest ---
c1, c2, c3, c4 = st.columns(4)

with c1:
    if st.button("Train"):
        with st.status("Training...", expanded=False) as status:
            try:
                res = train_for_symbol(symbol, model_choice=model_choice, use_news=use_news, use_social=use_social)
                status.update(label=f"Trained {symbol} — CV ROC AUC: {res.cv_score:.3f}", state="complete")
                st.toast("Model saved", icon="✅")
            except Exception as e:
                status.update(label=f"Training failed: {e}", state="error")

with c2:
    if st.button("Predict T+1"):
        try:
            proba, last_row = predict_proba(symbol)
            st.success(f"Pr(UP) for next bar: {proba:.3f}")
            # Save prediction
            from models.stockmarket import StockPrediction
            rec = StockPrediction(
                stock_id=db_session.query(Stock.id).filter(Stock.symbol==symbol).scalar(),
                symbol=symbol,
                price_date=pd.to_datetime(last_row['price_date']).to_pydatetime(),
                target_date=(pd.to_datetime(last_row['price_date']) + pd.Timedelta(days=1)).to_pydatetime(),
                pred_type="direction",
                pred_value=float(proba),
                model_name="Merlin4",
                model_version="1",
            )
            db_session.add(rec); db_session.commit()
            st.toast("Prediction saved ✓")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with c3:
    if st.button("Walk-forward backtest (simulate live)"):
        st.session_state[stop_key] = False  # reset
        with st.status("Running walk-forward backtest...", expanded=False) as status:
            bt = walk_forward(
                symbol, retrain_k=retrain_k, hold_steps=hold_steps,
                buy_threshold=buy_threshold, use_news=use_news, use_social=use_social,
                stop_flag=st.session_state
            )
            status.update(label="Backtest complete", state="complete")
        if not bt.empty:
            st.line_chart(bt.set_index("price_date")["cum_pnl"])
            st.dataframe(bt.tail(20), use_container_width=True, hide_index=True)
            # Persist metric + suggestions
            from commands.ml import record_metric_and_suggestions
            record_metric_and_suggestions(symbol, bt, model_name="Merlin4")

with c4:
    save_to_db = st.checkbox("Save prediction", value=True)
    if st.button("Quick Predict & Save"):
        try:
            proba, last_row = predict_proba(symbol)
            st.info(f"Pr(UP): {proba:.3f}")
            if save_to_db:
                from models.stockmarket import StockPrediction
                rec = StockPrediction(
                    stock_id=db_session.query(Stock.id).filter(Stock.symbol==symbol).scalar(),
                    symbol=symbol,
                    price_date=pd.to_datetime(last_row['price_date']).to_pydatetime(),
                    target_date=(pd.to_datetime(last_row['price_date']) + pd.Timedelta(days=1)).to_pydatetime(),
                    pred_type="direction",
                    pred_value=float(proba),
                    model_name="Merlin4",
                    model_version="1",
                )
                db_session.add(rec); db_session.commit()
                st.toast("Saved ✓")
        except Exception as e:
            st.error(f"{e}")

st.divider()

# --- Price chart ---
st.subheader(f"{symbol} — Closing Price")
prices = load_prices_cached(symbol)
if not prices.empty:
    prices = prices.copy()
    prices["price_date"] = pd.to_datetime(prices["price_date"])
    prices.sort_values("price_date", inplace=True)
    st.line_chart(prices.set_index("price_date")["close_price"])
else:
    st.write("No data yet. Use the fetch buttons above.")

# --- Buzz & Volume spike decision helper ---
st.subheader("📣 Social buzz vs volume spike — AI decision")
def compute_spike_message(symbol: str, prices_df: pd.DataFrame) -> str:
    from commands.forums import build_daily_social_features
    sf = build_daily_social_features(symbol)
    msg = "No social data yet."
    if not sf.empty and not prices_df.empty:
        latest_date = sf["date"].max()
        latest_row = sf.loc[sf["date"] == latest_date].iloc[0]
        mention_spike = float(latest_row.get("mention_spike") or 0)

        # Volume spike relative to 20-day average
        tmp = prices_df.copy()
        tmp["price_date"] = pd.to_datetime(tmp["price_date"]).dt.normalize()
        tmp = tmp[tmp["price_date"] <= latest_date]
        if tmp.empty:
            return "Not enough price data."
        tmp["vol_ma20"] = tmp["volume"].rolling(20).mean()
        vol_today = float(tmp.iloc[-1]["volume"])
        vol_ma20 = float(tmp.iloc[-1]["vol_ma20"] or 0)
        vol_spike = vol_today / vol_ma20 if vol_ma20 else 0.0

        decision = "Hold / wait"
        when = "later"
        if mention_spike >= 2.0 and vol_spike >= 2.0:
            decision = "BUY"
            when = "now"
        elif mention_spike >= 1.5 and vol_spike >= 1.5:
            decision = "Consider BUY"
            when = "this week"

        msg = (f"This stock '{symbol}' was mentioned {mention_spike:.2f}× vs 30d average today "
               f"and trading volume is {vol_spike:.2f}× the 20d average — AI says: **{decision}** {when}.")
    return msg

st.info(compute_spike_message(symbol, prices))

# --- Suggestions box ---
st.subheader("🛠️ Self‑improvement suggestions")
suggestions = (
    db_session.query(ModelSuggestion)
    .filter((ModelSuggestion.symbol==symbol) | (ModelSuggestion.symbol==None))
    .order_by(ModelSuggestion.created_at.desc())
    .limit(20)
    .all()
)
if suggestions:
    df_sug = pd.DataFrame([{
        "created_at": s.created_at,
        "severity": s.severity,
        "suggestion": s.suggestion,
        "status": s.status
    } for s in suggestions])
    st.dataframe(df_sug, use_container_width=True, hide_index=True)
else:
    st.caption("No suggestions yet — run a backtest or train to generate them.")

st.divider()
st.subheader(f"{symbol} — Recent predictions")
preds = (
    db_session.query(StockPrediction)
    .filter(StockPrediction.symbol == symbol)
    .order_by(StockPrediction.id.desc())
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
    st.write("No saved predictions yet — use Predict to save one.")
