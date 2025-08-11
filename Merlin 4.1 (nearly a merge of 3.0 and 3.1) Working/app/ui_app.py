import os
import numpy as np
import pandas as pd
import streamlit as st

try:
    from models.stockmarket import db_session, Stock, Model
except Exception:
    try:
        from app.models.stockmarket import db_session, Stock, Model
    except Exception:
        from stockmarket import db_session, Stock, Model

def _ensure_ext_tables():
    try:
        try:
            import models.extensions as _ext
        except Exception:
            import app.models.extensions as _ext  # type: ignore
        Model.metadata.create_all(bind=db_session.bind)
    except Exception:
        pass
_ensure_ext_tables()

try:
    from commands.ml import (
        load_prices, make_features, FEATURE_COLUMNS_BASE, FEATURE_NEWS,
        train_for_symbol, _load_model, to_weekly, backtest_walkforward,
        log_prediction, generate_suggestions
    )
except Exception:
    try:
        from app.commands.ml import (
            load_prices, make_features, FEATURE_COLUMNS_BASE, FEATURE_NEWS,
            train_for_symbol, _load_model, to_weekly, backtest_walkforward,
            log_prediction, generate_suggestions
        )
    except Exception:
        from ml import (
            load_prices, make_features, FEATURE_COLUMNS_BASE, FEATURE_NEWS,
            train_for_symbol, _load_model, to_weekly, backtest_walkforward,
            log_prediction, generate_suggestions
        )

try:
    from commands.stocks import (call_alphavantage, call_yfinance, fetch_prices_auto, write_price_data, _av_symbol)
except Exception:
    try:
        from app.commands.stocks import (call_alphavantage, call_yfinance, fetch_prices_auto, write_price_data, _av_symbol)
    except Exception:
        from stocks import (call_alphavantage, call_yfinance, fetch_prices_auto, write_price_data, _av_symbol)

try:
    from commands.news import ingest_news
except Exception:
    try:
        from app.commands.news import ingest_news
    except Exception:
        from news import ingest_news

try:
    from commands.forum_search import update_forum_buzz, has_key as _buzz_has_key
except Exception:
    try:
        from app.commands.forum_search import update_forum_buzz, has_key as _buzz_has_key
    except Exception:
        def update_forum_buzz(symbol: str, window_days: int = 1): return {"symbol":symbol,"count":0,"baseline":0.0,"buzz_z":0.0}
        def _buzz_has_key(): return False

try:
    from models.extensions import ModelMetric, TradePlan, ModelSuggestion, StockPrediction, ForumBuzz
except Exception:
    try:
        from app.models.extensions import ModelMetric, TradePlan, ModelSuggestion, StockPrediction, ForumBuzz
    except Exception:
        ModelMetric = TradePlan = ModelSuggestion = StockPrediction = ForumBuzz = None

st.set_page_config(page_title="Merlin — Stock ML (v4.0)", layout="wide")
st.title("🧙 Merlin — Stock ML (v4.0)")

st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input("Alpha Vantage API Key", value=os.getenv("ALPHA_VANTAGE_API_KEY", ""), type="password")
if api_key_input: os.environ["ALPHA_VANTAGE_API_KEY"] = api_key_input
serp_key = st.sidebar.text_input("SerpAPI API Key (forum buzz)", value=os.getenv("SERPAPI_API_KEY",""), type="password")
if serp_key: os.environ["SERPAPI_API_KEY"] = serp_key

symbols = [s.symbol for s in db_session.query(Stock).filter(Stock.enabled == True).order_by(Stock.symbol)]
if not symbols:
    st.warning("No symbols found in DB."); st.stop()

symbol = st.selectbox("Symbol", symbols)
freq = st.sidebar.radio("Frequency", ["D", "W"], index=0, horizontal=True, help="W = Friday aggregate")
horizon = st.sidebar.slider("Horizon (steps ahead)", 1, 8, 1)
model_kind = st.sidebar.selectbox("Model", ["gb", "rf", "xgb", "merlin"], help="'merlin' is our in-house NN")
use_news = st.sidebar.checkbox("Use news sentiment", value=False)

st.sidebar.divider()
st.session_state.setdefault("stop", False)
if st.sidebar.button("🛑 Stop running"):
    st.session_state["stop"] = True; os.environ["MERLIN_STOP"] = "1"
else:
    os.environ["MERLIN_STOP"] = "0"

st.sidebar.divider()
apikey = os.getenv("ALPHA_VANTAGE_API_KEY", "")
colA, colB, colC = st.sidebar.columns([1,1,1])

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

if st.sidebar.button("Fetch news for symbol"):
    n = ingest_news(symbol)
    if n == 0:
        st.warning("0 news cached. Rate-limit or missing API key. Try later or test with AAPL.")
    else:
        st.toast(f"Fetched {n} news items into cache", icon="📰")

prices = load_prices(symbol)
if prices.empty:
    st.info("No price data. Fetch first."); st.stop()

chart_df = to_weekly(prices) if freq == "W" else prices
chart_df = chart_df.copy()
chart_df["price_date"] = pd.to_datetime(chart_df["price_date"]); chart_df.sort_values("price_date", inplace=True)
st.subheader(f"{symbol} — Close ({'Weekly' if freq == 'W' else 'Daily'})")
st.line_chart(chart_df.set_index("price_date")["close_price"], use_container_width=True)

c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    if st.button("🚀 Train model"):
        st.session_state["stop"] = False; os.environ["MERLIN_STOP"] = "0"
        with st.status("Training…", expanded=True) as status:
            st.write("Random search + CV…")
            res = train_for_symbol(symbol, horizon=horizon, freq=freq, model_kind=model_kind, use_news=use_news)
            status.update(label=f"Done: ROC AUC {res.cv_score:.3f}", state="complete")
            st.success(f"Trained {symbol}: {res.model_kind} {res.freq} H{res.horizon} — ROC AUC {res.cv_score:.3f}")

with c2:
    save_pred = st.checkbox("Save prediction")
    if st.button("🔮 Predict next"):
        try:
            blob = _load_model(symbol, model_kind, freq, horizon)
            pipe = blob["pipeline"]; use_news_blob = blob.get("use_news", False)
            df = make_features(chart_df, horizon=horizon, use_news=use_news_blob, symbol=symbol if use_news_blob else None)
            x_last = df[blob["feature_columns"]].values[-1]
            proba = float(pipe.predict_proba([x_last])[0, 1])
            st.success(f"P(up in {horizon} {'weeks' if freq == 'W' else 'days'}): {proba:.3f}")
            if save_pred:
                last_date = pd.to_datetime(df.iloc[-1]["price_date"])
                target_date = last_date + (pd.Timedelta(weeks=horizon) if freq=='W' else pd.Timedelta(days=horizon))
                log_prediction(symbol, last_date, target_date, proba, blob.get("model_name",""))
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
            dftr["entry_date"] = pd.to_datetime(dftr["entry_date"]); dftr["exit_date"] = pd.to_datetime(dftr["exit_date"])
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
                dfw = make_features(chart_df, horizon=horizon, use_news=blob.get("use_news", False), symbol=symbol if blob.get("use_news", False) else None)
                x_last = dfw[blob["feature_columns"]].values[-1]
                p = float(pipe.predict_proba([x_last])[0, 1])
                last_date = pd.to_datetime(dfw.iloc[-1]["price_date"])
                exit_date = last_date + pd.Timedelta(weeks=plan_hold)
                action = "BUY" if p >= plan_thr else "HOLD"
                st.subheader(f"Action: {action}")
                st.write(f"Entry week: {last_date.date()}  →  Exit date: {exit_date.date()}  |  P(up): {p:.3f}  |  Threshold: {plan_thr:.2f}")
                try:
                    from models.extensions import TradePlan
                except Exception:
                    from app.models.extensions import TradePlan
                try:
                    tp = TradePlan(symbol=symbol, entry_date=last_date.to_pydatetime(), exit_date=exit_date.to_pydatetime(),
                                   freq="W", hold_steps=int(plan_hold), threshold=float(plan_thr), proba=p, action=action)
                    db_session.add(tp); db_session.commit(); st.toast("Plan saved", icon="✅")
                except Exception:
                    pass
            except FileNotFoundError:
                st.warning("Model not found — train it first.")

st.divider()
st.subheader("🧭 Forum Buzz & Volume Spike Radar")
col1, col2, col3 = st.columns([1,1,2])
with col1:
    if st.button("Scan forum buzz (1d)"):
        info = update_forum_buzz(symbol, window_days=1)
        st.session_state["buzz_info"] = info
with col2:
    if "buzz_info" in st.session_state:
        b = st.session_state["buzz_info"]
        st.metric(label="Mentions (1d)", value=b.get("count",0), delta=f"z={b.get('buzz_z',0.0):.2f}")
    else:
        st.caption("Click **Scan forum buzz** (SerpAPI key recommended).")
with col3:
    vol_df = chart_df.copy()
    vol_df["vol_ma30"] = vol_df["volume"].rolling(30).mean()
    vol_ratio = float(vol_df.iloc[-1]["volume"] / (vol_df.iloc[-1]["vol_ma30"] or np.nan)) if len(vol_df)>30 else np.nan
    if np.isfinite(vol_ratio):
        st.metric(label="Volume ratio (today / MA30)", value=f"{vol_ratio:.2f}×")
    else:
        st.caption("Need ≥30 bars for volume ratio.")

buzz = st.session_state.get("buzz_info", {"count":0,"buzz_z":0.0})
vol_ratio = vol_ratio if 'vol_ratio' in locals() and np.isfinite(vol_ratio) else 1.0
pmsg = "Run **Predict next** to get P(up)"
with st.expander("Decision suggestion"):
    try:
        blob = _load_model(symbol, model_kind, freq, horizon)
        pipe = blob["pipeline"]
        dfF = make_features(chart_df, horizon=horizon, use_news=blob.get("use_news", False), symbol=symbol if blob.get("use_news", False) else None)
        p_up = float(pipe.predict_proba([dfF[blob["feature_columns"]].values[-1]])[0,1])
        buy_sig = (p_up >= 0.6) and (buzz.get("buzz_z",0.0) >= 1.0) and (vol_ratio >= 2.0)
        action = "BUY NOW" if buy_sig else ("WAIT" if p_up >= 0.55 else "AVOID")
        st.write(f"Model P(up): **{p_up:.3f}**  |  Buzz z: **{buzz.get('buzz_z',0.0):.2f}**  |  Vol×: **{vol_ratio:.2f}** → **{action}**")
    except Exception:
        st.write(pmsg)

st.subheader("🧠 Self-improvement (suggestions)")
colL, colR = st.columns([1,2])
with colL:
    if st.button("Evaluate last predictions & suggest improvements"):
        sugs = generate_suggestions(symbol, freq=freq, horizon=horizon, window=120)
        st.session_state["sugs"] = sugs
with colR:
    if "sugs" in st.session_state and st.session_state["sugs"]:
        for s in st.session_state["sugs"]:
            st.info(f"{s['suggestion']}  (score {s['score']:.2f})")
    else:
        st.caption("No suggestions yet. Save predictions over time, then run evaluation.")

st.subheader("Recent model metrics")
if ModelMetric is not None:
    try:
        q = db_session.query(ModelMetric).filter(ModelMetric.symbol == symbol).order_by(ModelMetric.id.desc()).limit(10)
        dfm = pd.read_sql(q.statement, db_session.bind)
        if not dfm.empty:
            st.dataframe(
                dfm[["model_name", "freq", "horizon", "metric_name", "metric_value", "window_end"]],
                hide_index=True, use_container_width=True,
            )
        else:
            st.write("No metrics saved yet.")
    except Exception:
        st.caption("(metrics table not found yet)")
else:
    st.caption("(metrics table not available — run once after creating extensions tables)")

st.subheader("Recent predictions")
if StockPrediction is not None:
    try:
        q = db_session.query(StockPrediction).filter(StockPrediction.symbol == symbol).order_by(StockPrediction.id.desc()).limit(20)
        dfp = pd.read_sql(q.statement, db_session.bind)
        if not dfp.empty:
            st.dataframe(
                dfp[["price_date", "target_date", "pred_value", "model_name", "model_version"]],
                hide_index=True, use_container_width=True,
            )
        else:
            st.write("No predictions saved yet.")
    except Exception:
        st.caption("(prediction table not found yet)")

st.subheader("Forum buzz history")
if ForumBuzz is not None:
    try:
        q = db_session.query(ForumBuzz).filter(ForumBuzz.symbol == symbol).order_by(ForumBuzz.id.desc()).limit(10)
        dfb = pd.read_sql(q.statement, db_session.bind)
        if not dfb.empty:
            st.dataframe(dfb[["collected_at","count","baseline_avg","buzz_z"]], hide_index=True, use_container_width=True)
        else:
            st.write("No buzz records yet.")
    except Exception:
        st.caption("(forum_buzz table not found yet)")

st.caption("Tip: GB for quick tests; try XGB/Merlin for stronger signals. Weekly often best for BTC/ETH/Oil swings.")
