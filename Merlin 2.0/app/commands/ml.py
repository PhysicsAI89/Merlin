
import os, json, datetime as dt, math, requests
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import click
from flask import Blueprint

from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey
from sqlalchemy.sql import func

from models.stockmarket import Stock, StockDaily, db_session, Model

from packaging.version import Version
import xgboost as xgb
XGB_VERSION = Version(xgb.__version__)

mlbp = Blueprint("ml", __name__)

import os
# === Speed knobs you can change without editing code again
N_JOBS     = int(os.getenv("MERLIN_N_JOBS", "-1"))   # -1 = all CPU cores
TS_SPLITS  = int(os.getenv("MERLIN_TS_SPLITS", "3")) # fewer splits = faster CV
USE_GPU    = os.getenv("MERLIN_USE_GPU", "0") == "1" # set to 1 to try GPU models


MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------- Prediction table ----------------
class StockPrediction(Model):
    __tablename__ = "stock_prediction"
    __table_args__ = {"sqlite_autoincrement": True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stock.id"), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    price_date = Column(DateTime, nullable=False)
    target_date = Column(DateTime, nullable=False)
    pred_type = Column(String(32), nullable=False)  # "direction_h" or "return_h"
    pred_value = Column(Float, nullable=False)
    model_name = Column(String(64), nullable=False)
    model_version = Column(String(32), nullable=False, default="1")
    created_at = Column(DateTime, server_default=func.now())

# --------------- Features ----------------
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def _bollinger(close: pd.Series, n=20, k=2):
    ma = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = ma + k*std
    lower = ma - k*std
    width = (upper - lower) / ma
    return ma, upper, lower, width

FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_10",
    "ma5_ratio", "ma10_ratio", "ma20_ratio",
    "vol5", "vol10", "vol20",
    "rsi14", "macd", "macd_signal", "bb_width",
    "range_ratio", "volume_z",
    # news features (if available, else NaN -> imputed by scaler)
    "news_sent_3d", "news_buzz_3d"
]

def make_features(df: pd.DataFrame, use_news: bool=False) -> pd.DataFrame:
    df = df.copy()
    df.sort_values("price_date", inplace=True)
    close = df["close_price"]

    # returns
    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)

    # MAs
    for w in (5, 10, 20):
        df[f"ma{w}"] = close.rolling(w).mean()
        df[f"ma{w}_ratio"] = close / df[f"ma{w}"] - 1

    # Vol
    for w in (5, 10, 20):
        df[f"vol{w}"] = df["ret_1"].rolling(w).std()

    # RSI, MACD, Bollinger width, range ratio
    df["rsi14"] = _rsi(close, 14)
    macd, sig = _macd(close)
    df["macd"], df["macd_signal"] = macd, sig
    ma, up, lo, width = _bollinger(close, 20, 2)
    df["bb_width"] = width
    df["range_ratio"] = (df["high_price"] - df["low_price"]) / close.replace(0, np.nan)

    # Volume z-score
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_std20"] = df["volume"].rolling(20).std()
    df["volume_z"] = (df["volume"] - df["volume_ma20"]) / df["volume_std20"].replace(0, np.nan)

    if use_news:
        if "news_sent" in df.columns:
            df["news_sent_3d"] = df["news_sent"].rolling(3).mean()
            df["news_buzz_3d"] = df["news_count"].rolling(3).sum()
        else:
            df["news_sent_3d"] = np.nan
            df["news_buzz_3d"] = np.nan

    df.dropna(inplace=True)
    return df

def load_prices(symbol: str, start: Optional[str]=None, freq: str="D") -> pd.DataFrame:
    q = db_session.query(StockDaily).filter(StockDaily.symbol==symbol)
    if start:
        q = q.filter(StockDaily.price_date >= start)
    q = q.order_by(StockDaily.price_date)
    df = pd.read_sql(q.statement, db_session.bind)
    if df.empty: return df
    df["price_date"] = pd.to_datetime(df["price_date"])
    for col in ["open_price","high_price","low_price","close_price","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    if freq.upper().startswith("W"):
        # weekly resample (Fri close)
        df = (df.set_index("price_date")
                .resample("W-FRI")
                .agg({"open_price":"first","high_price":"max","low_price":"min","close_price":"last","volume":"sum"})
                .dropna()
                .reset_index())
    return df

def add_news_features(symbol: str, df: pd.DataFrame, apikey: Optional[str]) -> pd.DataFrame:
    if not apikey or df.empty: return df
    # fetch last 200 articles over the df date span
    start = df["price_date"].min().strftime("%Y%m%dT0000")
    end = df["price_date"].max().strftime("%Y%m%dT2359")
    # crude mapping for topics/keywords
    kw = symbol.split("-")[0].split("=")[0]
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&sort=LATEST&limit=200&time_from={start}&time_to={end}&apikey={apikey}&keywords={kw}"
    try:
        r = requests.get(url, timeout=10)
        j = r.json()
        feed = j.get("feed", [])
        if not feed:
            return df
        rows = []
        for item in feed:
            ts = item.get("time_published")  # e.g. 20250101T120000
            d = pd.to_datetime(ts[:8], format="%Y%m%d", errors="coerce")
            if pd.isna(d): continue
            sent = float(item.get("overall_sentiment_score", 0.0))
            rows.append((d, sent))
        if not rows: return df
        news = pd.DataFrame(rows, columns=["price_date","news_sent"])
        g = news.groupby("price_date").agg(news_sent=("news_sent","mean"), news_count=("news_sent","size")).reset_index()
        out = pd.merge(df, g, on="price_date", how="left")
        out[["news_sent","news_count"]] = out[["news_sent","news_count"]].fillna(0.0)
        return out
    except Exception as e:
        return df

# --------------- Modeling ---------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

@dataclass
class TrainResult:
    symbol: str
    task: str
    model_path: str
    cv_score: float
    n_samples: int
    model_name: str


def _xgb_params():
    common = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    )
    if USE_GPU and XGB_VERSION >= Version("2.0.0"):
        # XGBoost 2.x+ (preferred)
        return dict(**common, device="cuda", tree_method="hist")
    elif USE_GPU:
        # XGBoost 1.x (legacy)
        return dict(**common, tree_method="gpu_hist", predictor="gpu_predictor")
    else:
        # CPU fall-back
        return dict(**common, tree_method="hist")

def _build_clf(model_name: str):
    if model_name == "gb":
        base = GradientBoostingClassifier(random_state=42)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    elif model_name == "rf":
        base = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=N_JOBS)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    elif model_name == "xgb":
        base = xgb.XGBClassifier(**_xgb_params())
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    else:
        raise ValueError("Unknown model: " + model_name)

    pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
    return pipe
  


def _target_cols(h: int) -> Tuple[str, str]:
    return f"return_{h}", f"direction_{h}"

def make_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    df[f"return_{horizon}"] = df["close_price"].pct_change(-horizon)
    df[f"direction_{horizon}"] = (df[f"return_{horizon}"] > 0).astype(int)
    return df

def train_for_symbol(symbol: str, freq: str="D", horizon: int=1, use_news: bool=False, model: str="gb") -> TrainResult:
    apikey = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    prices = load_prices(symbol, freq=freq)
    if prices.empty: raise RuntimeError(f"No price data for {symbol}")
    if use_news:
        prices = add_news_features(symbol, prices, apikey)
    df = make_targets(make_features(prices, use_news=use_news), horizon)

    used_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[used_cols].values
    y = df[f"direction_{horizon}"].values


    pipe = _build_clf(model)
    tscv = TimeSeriesSplit(n_splits=TS_SPLITS)  # uses MERLIN_TS_SPLITS (default 3)
    scores = cross_val_score(pipe, X, y, cv=tscv, scoring="roc_auc")
    pipe.fit(X, y)

    model_name = f"{model.upper()}_{freq}_H{horizon}"
    fname = f"{symbol}_{model_name}.joblib"
    path = os.path.join(MODELS_DIR, fname)

    import joblib
    #joblib.dump({"pipeline": pipe, "feature_columns": FEATURE_COLUMNS, "task": "classification", "model_name": model_name, "freq":freq, "horizon":horizon, "use_news":use_news}, path)

    joblib.dump({
    "pipeline": pipe,
    "used_columns": used_cols,   # <— add this
    "task": "classification",
    "model_name": model_name,
    "freq": freq,
    "horizon": horizon,
    "use_news": use_news
    }, path)



    return TrainResult(symbol, "classification", path, float(np.mean(scores)), len(df), model_name)

def _load_model(symbol: str, model_name: str):
    import joblib, os, glob
    path = os.path.join(MODELS_DIR, f"{symbol}_{model_name}.joblib")
    if os.path.exists(path): return joblib.load(path)
    # fallback: any model for symbol
    files = sorted(glob.glob(os.path.join(MODELS_DIR, f"{symbol}_*.joblib")))
    if not files: raise FileNotFoundError("No model found for " + symbol)
    return joblib.load(files[-1])

def predict_one(symbol: str,
                model_name: str,
                freq: str,
                horizon: int,
                use_news: bool,
                write: bool = True) -> float:
    """
    Predict the probability that the asset closes UP at t + horizon
    using the model saved for (symbol, model_name). The function
    re-creates features/targets exactly as in training and selects
    the same columns the model was fitted on.
    """
    apikey = os.getenv("ALPHA_VANTAGE_API_KEY", "")

    # 1) Load prices for the chosen frequency (D/W)
    prices = load_prices(symbol, freq=freq)
    if prices.empty:
        raise RuntimeError(f"No price data for {symbol}")

    # 2) Optional news features
    if use_news:
        prices = add_news_features(symbol, prices, apikey)

    # 3) Build features/targets exactly as in training
    df = make_targets(make_features(prices, use_news=use_news), horizon)
    if df.empty:
        raise RuntimeError("Not enough rows after feature engineering.")

    # keep time ordering clean
    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df = df.dropna(subset=["price_date"]).sort_values("price_date")

    # 4) Load the fitted pipeline and its feature list
    blob = _load_model(symbol, model_name)
    pipe = blob["pipeline"]
    used_cols = blob.get("used_columns")
    if not used_cols:
        # backward-compat: old models that didn't store used_columns
        used_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # 5) Select the exact columns the scaler/model expects (and order!)
    X = df[used_cols].astype(float).values
    x_last = X[-1]
    proba = float(pipe.predict_proba([x_last])[0, 1])

    # 6) Dates for record / output
    last_row = df.iloc[-1]
    price_date = pd.to_datetime(last_row["price_date"]).to_pydatetime()
    step_days = 7 if str(freq).upper().startswith("W") else 1
    target_date = price_date + dt.timedelta(days=step_days * horizon)

    out = {
        "symbol": symbol,
        "price_date": price_date.isoformat(),
        "target_date": target_date.isoformat(),
        "pred_type": f"direction_{horizon}",
        "pred_value": proba,
        "model": blob.get("model_name", model_name),
    }
    print(json.dumps(out, indent=2, default=str))

    # 7) Optionally persist to DB
    if write:
        stk = db_session.query(Stock).filter(Stock.symbol == symbol).first()
        rec = StockPrediction(
            stock_id=stk.id,
            symbol=symbol,
            price_date=price_date,
            target_date=target_date,
            pred_type=f"direction_{horizon}",
            pred_value=proba,
            model_name=blob.get("model_name", model_name),
            model_version="1",
        )
        db_session.add(rec)
        db_session.commit()
        print("Saved to stock_prediction.")
    return proba


# ---------- Simple strategy / backtest ----------
def backtest(symbol: str, freq: str, horizon: int, model: str,
             threshold: float, hold_period: int, use_news: bool=False,
             retrain_every: int = 5, min_train: int = 200):
    """
    Faster walk-forward: re-fit every `retrain_every` steps instead of every bar.
    Uses sigmoid calibration (much faster) during backtest.
    """
    from sklearn.calibration import CalibratedClassifierCV

    apikey = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    prices = load_prices(symbol, freq=freq)
    if prices.empty:
        raise RuntimeError("No price data")
    if use_news:
        prices = add_news_features(symbol, prices, apikey)

    df = make_targets(make_features(prices, use_news=use_news), horizon)
    used_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[used_cols].astype(np.float32).values
    y = df[f"direction_{horizon}"].values
    returns = df[f"return_{horizon}"].values
    dates = pd.to_datetime(df["price_date"]).values

    # Build a faster model for backtests: sigmoid (fast) instead of isotonic
    pipe = _build_clf(model)
    if isinstance(pipe.named_steps["model"], CalibratedClassifierCV):
        base = pipe.named_steps["model"].estimator
        fast_clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("scaler", pipe.named_steps["scaler"]), ("model", fast_clf)])

    i = min_train
    last_fit = -1
    equity = 1.0
    positions = []

    while i < len(X) - horizon:
        if last_fit < 0 or (i - last_fit) >= retrain_every:
            pipe.fit(X[:i], y[:i])
            last_fit = i

        proba = pipe.predict_proba(X[i:i+1])[0, 1]
        if proba >= threshold:
            r = returns[i]            # horizon return starting at i
            equity *= (1.0 + r)
            positions.append((dates[i], proba, r))

        i += 1

    hit_rate = float(np.mean([1.0 if r > 0 else 0.0 for _, _, r in positions])) if positions else 0.0
    return equity, hit_rate, positions


# ---------------- CLI ----------------
@mlbp.cli.command("train")
@click.option("--symbol", default="ALL")
@click.option("--freq", type=click.Choice(["D","W"]), default="D")
@click.option("--horizon", type=int, default=1)
@click.option("--use-news/--no-use-news", default=False)
@click.option("--model", type=click.Choice(["gb","rf","xgb"]), default="gb")
def cli_train(symbol, freq, horizon, use_news, model):
    Model.metadata.create_all(bind=db_session.bind)
    symbols = [s.symbol for s in db_session.query(Stock).filter(Stock.enabled==True)] if symbol=="ALL" else [symbol]
    for sym in symbols:
        try:
            res = train_for_symbol(sym, freq=freq, horizon=horizon, use_news=use_news, model=model)
            print(f"Trained {sym} — {res.model_name} ROC AUC: {res.cv_score:.3f} ({res.n_samples} samples) -> {res.model_path}")
        except Exception as e:
            print(f"[WARN] {sym}: {e}")

@mlbp.cli.command("predict")
@click.argument("symbol")
@click.option("--freq", type=click.Choice(["D","W"]), default="D")
@click.option("--horizon", type=int, default=1)
@click.option("--model-name", default=None, help="Exact saved model name, otherwise the latest is used")
@click.option("--use-news/--no-use-news", default=False)
@click.option("--write/--no-write", default=True)
def cli_predict(symbol, freq, horizon, model_name, use_news, write):
    if not model_name:
        model_name = f"GB_{freq}_H{horizon}"
    predict_one(symbol, model_name, freq, horizon, use_news, write)

@mlbp.cli.command("evaluate")
@click.argument("symbol")
@click.option("--freq", type=click.Choice(["D","W"]), default="D")
@click.option("--horizon", type=int, default=1)
@click.option("--model", type=click.Choice(["gb","rf","xgb"]), default="gb")
@click.option("--use-news/--no-use-news", default=False)
def cli_evaluate(symbol, freq, horizon, model, use_news):
    final, hit_rate, positions = backtest(symbol, freq, horizon, model, threshold=0.55, hold_period=horizon, use_news=use_news)
    print(json.dumps({
        "symbol": symbol, "freq": freq, "horizon": horizon,
        "final_equity": final, "hit_rate": hit_rate, "trades": len(positions)
    }, indent=2))

@mlbp.cli.command("predict_all")
@click.option("--freq", type=click.Choice(["D","W"]), default="D")
@click.option("--horizon", type=int, default=1)
def cli_predict_all(freq, horizon):
    symbols = [s.symbol for s in db_session.query(Stock).filter(Stock.enabled==True).order_by(Stock.id)]
    for sym in symbols:
        try:
            predict_one(sym, f"GB_{freq}_H{horizon}", freq, horizon, False, True)
        except Exception as e:
            print(f"[WARN] {sym}: {e}")
