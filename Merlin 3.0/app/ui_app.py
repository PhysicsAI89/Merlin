# ui_app.py
import os
import numpy as np
import pandas as pd
import streamlit as st

# --------- DB models (shim for both layouts) ----------
try:
    from models.stockmarket import db_session, Stock, Model
except Exception:
    try:
        from app.models.stockmarket import db_session, Stock, Model
    except Exception:
        from stockmarket import db_session, Stock, Model  # fallback

# Ensure extensions tables exist once (news_event, model_metric, trade_plan)
def _ensure_ext_tables():
    try:
        try:
            import models.extensions as _ext  # registers tables on Base
        except Exception:
            import app.models.extensions as _ext  # type: ignore
        Model.metadata.create_all(bind=db_session.bind)
    except Exception:
        pass

_ensure_ext_tables()

# --------- ML helpers (shim) ----------
try:
    from commands.ml import (
        load_prices, make_features, FEATURE_COLUMNS_BASE, FEATURE_NEWS,
        train_for_symbol, _load_model, to_weekly, backtest_walkforward
    )
except Exception:
    try:
        from app.commands.ml import (
            load_prices, make_features, FEATURE_COLUMNS_BASE, FEATURE_NEWS,
            train_for_symbol, _load_model, to_weekly, backtest_walkforward
        )
    except Exception:
        from ml import (  # local fallback
            load_prices, make_features, FEATURE_COLUMNS_BASE, FEATURE_NEWS,
            train_for_symbol, _load_model, to_weekly, backtest_walkforward
        )

# --------- Ingest (stocks/FX/crypto/futures) ----------
try:
    from commands.stocks import (
        call_alphavantage, call_yfinance, fetch_prices_auto,
        write_price_data, _av_symbol
    )
except Exception:
    try:
        from app.commands.stocks import (
            call_alphavantage, call_yfinance, fetch_prices_auto,
            write_price_data, _av_symbol
        )
    except Exception:
        from stocks import (  # flat fallback
            call_alphavantage, call_yfinance, fetch_prices_auto,
            write_price_data, _av_symbol
        )

# --------- News cache / features ----------
try:
    from commands.news import ingest_news
except Exception:
    try:
        from app.commands.news import ingest_news
    except Exception:
        from news import ingest_news

# Optional metrics view
try:
    from models.extensions import ModelMetric, TradePlan
except Exception:
    try:
        from app.models.extensions import ModelMetric, TradePlan
    except Exception:
        ModelMetric = None
        TradePlan = None

# -------------------------------- UI --------------------------------
st.set_page_config(page_title="Merlin — Stock ML (v3.0)", layout="wide")
st.title("📈 Merlin — Stock ML (v3.0)")

# Sidebar
st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input(
    "Alpha Vantage API Key",
    value=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
    type="password",
)
if api_key_input:
    os.environ["ALPHA_VANTAGE_API_KEY"] = api_key_input

symbols = [s.symbol for s in db_session.query(Stock).filter(Stock.enabled == True).order_by(Stock.symbol)]
if not symbols:
    st.warning("No symbols found in DB.")
    st.stop()

symbol = st.selectbox("Symbol", symbols)
freq = st.sidebar.radio("Frequency", ["D", "W"], index=0, horizontal=True, help="W = Friday aggregate")
horizon = st.sidebar.slider("Horizon (steps ahead)", 1, 8, 1)
model_kind = st.sidebar.selectbox("Model", ["gb", "rf", "xgb", "merlin"], help="'merlin' is our in-house NN")
use_news = st.sidebar.checkbox("Use news sentiment", value=False)

# ---- Fetch prices: Auto / AV / yfinance ----
st.sidebar.divider()
apikey = os.getenv("ALPHA_VANTAGE_API_KEY", "")

colA, colB, colC = st.sidebar.columns([1, 1, 1])

with colA:
    if st.button("Fetch (Auto)"):
        data = fetch_prices_auto(_av_symbol(symbol), apikey)
        daily = data.get("Time Series (Daily)") if isinstance(data, dict) else None
        if daily:
            stock_row = db_session.query(Stock).filter(Stock.symbol == symbol).first()
            rows = write_price_data(stock_row.id, symbol, daily)
            st.success(f"Inserted {rows} rows for {symbol}.")
        else:
            msg = data.get("Note") or data.get("Error Message") or data.get("Error") if isinstance(data, dict) else None
            st.warning("No data returned." + (f" Provider says: {msg}" if msg else ""))

with colB:
    if st.button("Fetch via AV"):
        if not apikey:
            st.error("Set Alpha Vantage API key.")
        else:
            data = call_alphavantage(_av_symbol(symbol), apikey)
            daily = data.get("Time Series (Daily)") if isinstance(data, dict) else None
            if daily:
                stock_row = db_session.query(Stock).filter(Stock.symbol == symbol).first()
                rows = write_price_data(stock_row.id, symbol, daily)
                st.success(f"(AV) Inserted {rows} rows for {symbol}.")
            else:
                msg = data.get("Note") or data.get("Error Message") or data.get("Error") if isinstance(data, dict) else None
                st.warning("No data returned." + (f" AV says: {msg}" if msg else ""))

with colC:
    if st.button("Fetch via yfinance"):
        data = call_yfinance(_av_symbol(symbol))
        daily = data.get("Time Series (Daily)") if isinstance(data, dict) else None
        if daily:
            stock_row = db_session.query(Stock).filter(Stock.symbol == symbol).first()
            rows = write_price_data(stock_row.id, symbol, daily)
            st.success(f"(yfinance) Inserted {rows} rows for {symbol}.")
        else:
            msg = data.get("Error") if isinstance(data, dict) else None
            st.warning("No data returned." + (f" yfinance says: {msg}" if msg else ""))

# ---- News fetch ----
if st.sidebar.button("Fetch news for symbol"):
    n = ingest_news(symbol)
    if n == 0:
        st.warning("0 news cached. Likely rate-limit or symbol/topic mismatch. Try again in ~60s or test with an equity like AAPL.")
    else:
        st.toast(f"Fetched {n} news items into cache", icon="📰")

st.sidebar.divider()
st.session_state.setdefault("stop", False)
if st.sidebar.button("🛑 Stop running"):
    st.session_state["stop"] = True

# Data + Chart
prices = load_prices(symbol)
if prices.empty:
    st.info("No price data. Fetch first.")
    st.stop()

chart_df = to_weekly(prices) if freq == "W" else prices
chart_df = chart_df.copy()
chart_df["price_date"] = pd.to_datetime(chart_df["price_date"])
chart_df.sort_values("price_date", inplace=True)
st.subheader(f"{symbol} — Close ({'Weekly' if freq == 'W' else 'Daily'})")
st.line_chart(chart_df.set_index("price_date")["close_price"], use_container_width=True)

# Train / Predict / Backtest / Plan
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    if st.button("🚀 Train model"):
        st.session_state["stop"] = False
        with st.status("Training…", expanded=True) as status:
            st.write("Running random search + CV…")
            res = train_for_symbol(symbol, horizon=horizon, freq=freq, model_kind=model_kind, use_news=use_news)
            status.update(label=f"Done: ROC AUC {res.cv_score:.3f}", state="complete")
            st.success(f"Trained {symbol}: {res.model_kind} {freq} H{res.horizon} — ROC AUC {res.cv_score:.3f}")

with c2:
    save_pred = st.checkbox("Save prediction")  # reserved; DB wiring optional
    if st.button("🔮 Predict next"):
        try:
            blob = _load_model(symbol, model_kind, freq, horizon)
            pipe = blob["pipeline"]
            use_news_blob = blob.get("use_news", False)
            df = make_features(chart_df, horizon=horizon, use_news=use_news_blob, symbol=symbol if use_news_blob else None)
            x_last = df[blob["feature_columns"]].values[-1]
            proba = float(pipe.predict_proba([x_last])[0, 1])
            st.success(f"P(up in {horizon} {'weeks' if freq == 'W' else 'days'}): {proba:.3f}")
        except FileNotFoundError:
            st.warning("Model not found — train it first.")

with c3:
    st.write("**Backtest**")
    thr = st.slider("Buy threshold", 0.5, 0.9, 0.6, 0.01)
    hold = st.slider("Hold steps", 1, 12, 4)
    retrain_k = st.slider("Retrain every k steps", 5, 60, 20, 5)
    if st.button("📜 Run backtest"):
        res = backtest_walkforward(symbol, thr, hold, retrain_k, horizon, freq, model_kind, use_news)
        st.info(f"Trades: {res['n']}  |  Total return: {res['ret']*100:.2f}%  |  Equity: {res['equity']:.3f}")
        if res["trades"]:
            dftr = pd.DataFrame(res["trades"])
            dftr["entry_date"] = pd.to_datetime(dftr["entry_date"])
            dftr["exit_date"] = pd.to_datetime(dftr["exit_date"])
            st.dataframe(dftr, hide_index=True, use_container_width=True)

with c4:
    st.write("**Buy/Sell plan (weekly)**")
    plan_hold = st.number_input("Hold weeks", 1, 16, 4)
    plan_thr = st.slider("Plan threshold", 0.5, 0.9, 0.6, 0.01, key="plan_thr")
    if st.button("🧭 Make plan"):
        if freq != "W":
            st.warning("Switch Frequency to W for weekly plan.")
        else:
            try:
                blob = _load_model(symbol, model_kind, freq, horizon)
                pipe = blob["pipeline"]
                dfw = make_features(
                    chart_df,
                    horizon=horizon,
                    use_news=blob.get("use_news", False),
                    symbol=symbol if blob.get("use_news", False) else None,
                )
                x_last = dfw[blob["feature_columns"]].values[-1]
                p = float(pipe.predict_proba([x_last])[0, 1])
                last_date = pd.to_datetime(dfw.iloc[-1]["price_date"])
                exit_date = last_date + pd.Timedelta(weeks=plan_hold)
                action = "BUY" if p >= plan_thr else "HOLD"
                st.subheader(f"Action: {action}")
                st.write(
                    f"Entry week: {last_date.date()}  →  Exit date: {exit_date.date()}  |  "
                    f"P(up): {p:.3f}  |  Threshold: {plan_thr:.2f}"
                )
                if TradePlan and st.checkbox("Log to DB"):
                    tp = TradePlan(
                        symbol=symbol,
                        entry_date=last_date.to_pydatetime(),
                        exit_date=exit_date.to_pydatetime(),
                        freq="W",
                        hold_steps=int(plan_hold),
                        threshold=float(plan_thr),
                        proba=p,
                        action=action,
                    )
                    db_session.add(tp)
                    db_session.commit()
                    st.toast("Plan saved", icon="✅")
            except FileNotFoundError:
                st.warning("Model not found — train it first.")

# Recent metrics
st.subheader("Recent model metrics")
if ModelMetric is not None:
    try:
        q = db_session.query(ModelMetric).filter(ModelMetric.symbol == symbol).order_by(ModelMetric.id.desc()).limit(10)
        dfm = pd.read_sql(q.statement, db_session.bind)
        if not dfm.empty:
            st.dataframe(
                dfm[["model_name", "freq", "horizon", "metric_name", "metric_value", "window_end"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.write("No metrics saved yet.")
    except Exception:
        st.caption("(metrics table not found yet)")
else:
    st.caption("(metrics table not available — run once after creating extensions tables)")
