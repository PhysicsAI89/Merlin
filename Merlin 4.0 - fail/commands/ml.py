import os
import json
import datetime as dt
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import click
from flask import Blueprint

from sqlalchemy.sql import func

from models.stockmarket import (
    Stock, StockDaily, db_session, Model,
    StockPrediction, ModelMetric, ModelSuggestion
)
from commands.news import build_daily_news_features
from commands.forums import build_daily_social_features

# --------------------------------------------------------------------------------------
# Blueprint: `ml` — training, backtest, predict
# --------------------------------------------------------------------------------------

mlbp = Blueprint("ml", __name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODELS_DIR = os.path.abspath(MODELS_DIR)
os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------------

BASE_FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d",
    "ma5_ratio", "ma10_ratio", "ma20_ratio",
    "vol5", "vol10", "vol20",
    "rsi14",
    "range_ratio",  # (high-low)/close
    "volume_z"
]

EXTRA_FEATURES_NEWS = ["news_sent_mean_3","news_sent_mean_7","news_count_3","news_count_7"]
EXTRA_FEATURES_SOCIAL = ["mention_z","mention_spike"]

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def make_features(prices: pd.DataFrame, symbol: str,
                  use_news: bool = False, use_social: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Merge base OHLCV features with optional news & social features aligned by date.
    Returns (df, feature_columns).
    """
    df = prices.copy()
    df.sort_values("price_date", inplace=True)
    # Returns
    df["ret_1d"] = df["close_price"].pct_change(1)
    df["ret_5d"] = df["close_price"].pct_change(5)
    df["ret_10d"] = df["close_price"].pct_change(10)

    # Moving averages (ratios to current price to avoid scale issues)
    for w in (5, 10, 20):
        df[f"ma{w}"] = df["close_price"].rolling(w).mean()
        df[f"ma{w}_ratio"] = df["close_price"] / df[f"ma{w}"] - 1

    # Realized volatility (std of daily returns)
    for w in (5, 10, 20):
        df[f"vol{w}"] = df["ret_1d"].rolling(w).std()

    # RSI and intraday range features
    df["rsi14"] = _rsi(df["close_price"], 14)
    df["range_ratio"] = (df["high_price"] - df["low_price"]) / df["close_price"].replace(0, np.nan)

    # Volume z-score (within rolling 20-day window)
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_std20"] = df["volume"].rolling(20).std()
    df["volume_z"] = (df["volume"] - df["volume_ma20"]) / df["volume_std20"].replace(0, np.nan)

    # Targets
    df["target_return_1d"] = df["close_price"].pct_change(-1)
    df["target_direction"] = (df["target_return_1d"] > 0).astype(int)

    df["date"] = pd.to_datetime(df["price_date"]).dt.normalize()

    feat_cols = BASE_FEATURE_COLUMNS.copy()

    # Join news features (daily)
    if use_news:
        nf = build_daily_news_features(symbol)
        if not nf.empty:
            df = df.merge(nf, left_on="date", right_on="date", how="left")
            feat_cols += EXTRA_FEATURES_NEWS

    # Join social features (daily)
    if use_social:
        sf = build_daily_social_features(symbol)
        if not sf.empty:
            df = df.merge(sf, left_on="date", right_on="date", how="left")
            feat_cols += EXTRA_FEATURES_SOCIAL

    df.dropna(inplace=True)
    return df, feat_cols


# --------------------------------------------------------------------------------------
# Data access helpers
# --------------------------------------------------------------------------------------

def load_prices(symbol: str, start: Optional[str] = None) -> pd.DataFrame:
    q = db_session.query(StockDaily).filter(StockDaily.symbol == symbol)
    if start:
        q = q.filter(StockDaily.price_date >= start)
    q = q.order_by(StockDaily.price_date)
    df = pd.read_sql(q.statement, db_session.bind)
    if df.empty:
        return df
    # Ensure correct dtypes
    df["price_date"] = pd.to_datetime(df["price_date"])
    for col in ["open_price", "close_price", "high_price", "low_price", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    return df


# --------------------------------------------------------------------------------------
# Modeling (scikit-learn)
# --------------------------------------------------------------------------------------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score

import joblib


@dataclass
class TrainResult:
    symbol: str
    task: str
    model_path: str
    cv_score: float
    n_samples: int
    feature_columns: List[str]


def _build_pipeline(model_name: str = "gb") -> Pipeline:
    if model_name == "gb":
        model = GradientBoostingClassifier(random_state=42)
    elif model_name == "rf":
        model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingClassifier(random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])
    return pipe


def _auto_tune(pipe: Pipeline, X, y) -> Pipeline:
    """Quick random search to self-improve hyperparams."""
    param_dist = {
        "model__n_estimators": [100, 200, 300, 500] if hasattr(pipe.named_steps["model"], "n_estimators") else [100],
        "model__learning_rate": [0.01, 0.05, 0.1] if hasattr(pipe.named_steps["model"], "learning_rate") else [0.1],
        "model__max_depth": [2, 3, 4] if hasattr(pipe.named_steps["model"], "max_depth") else [None],
    }
    tscv = TimeSeriesSplit(n_splits=4)
    search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=8, cv=tscv,
                                scoring="roc_auc", random_state=42, n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_


def train_for_symbol(symbol: str, model_choice: str = "gb",
                     use_news: bool=False, use_social: bool=False) -> TrainResult:
    prices = load_prices(symbol)
    if prices.empty:
        raise RuntimeError(f"No price data found for {symbol}.")

    df, feat_cols = make_features(prices, symbol, use_news=use_news, use_social=use_social)
    X = df[feat_cols].values
    y = df["target_direction"].values

    pipe = _build_pipeline(model_choice)
    tscv = TimeSeriesSplit(n_splits=5)

    cv = cross_val_score(pipe, X, y, cv=tscv, scoring="roc_auc", n_jobs=None)
    pipe.fit(X, y)

    # Self-improve pass: if ROC AUC below 0.55, run quick random search
    if float(np.mean(cv)) < 0.55:
        pipe = _auto_tune(pipe, X, y)

    model_tag = {"gb": "GBClassifier", "rf": "RFClassifier"}.get(model_choice, "GBClassifier")
    fname = f"{symbol}_classification.joblib"
    model_path = os.path.join(MODELS_DIR, fname)
    joblib.dump({
        "pipeline": pipe, "feature_columns": feat_cols, "task": "classification",
        "model_name": model_tag, "use_news": use_news, "use_social": use_social
    }, model_path)

    return TrainResult(symbol=symbol, task="classification", model_path=model_path,
                       cv_score=float(np.mean(cv)), n_samples=len(df), feature_columns=feat_cols)


def _load_model(symbol: str):
    path = os.path.join(MODELS_DIR, f"{symbol}_classification.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}. Train it first.")
    blob = joblib.load(path)
    return blob


def predict_proba(symbol: str) -> Tuple[float, pd.Series]:
    """Return probability of up for next day and the last feature row (for debug)."""
    prices = load_prices(symbol)
    if prices.empty:
        raise RuntimeError(f"No price data for {symbol}.")

    blob = _load_model(symbol)
    use_news = blob.get("use_news", False)
    use_social = blob.get("use_social", False)

    df, feat_cols = make_features(prices, symbol, use_news=use_news, use_social=use_social)
    X = df[feat_cols].values
    last_row = df.iloc[-1]
    pipe = blob["pipeline"]
    proba = float(pipe.predict_proba([X[-1]])[0, 1])
    return proba, last_row


# ---------------- Walk-forward backtest & "simulate live" ----------------

def walk_forward(
    symbol: str,
    retrain_k: int = 10,
    hold_steps: int = 5,
    buy_threshold: float = 0.55,
    use_news: bool=False,
    use_social: bool=False,
    stop_flag: Optional[dict] = None
) -> pd.DataFrame:
    """
    Feed old data chronologically and act as if each new bar is "today".
    Train on history up to t, predict t+1, record outcome when it arrives.
    Returns a DataFrame with columns:
    ['price_date','proba','actual','signal','pnl','cum_pnl']
    """
    prices = load_prices(symbol)
    if prices.empty:
        raise RuntimeError(f"No price data for {symbol}.")

    df, feat_cols = make_features(prices, symbol, use_news=use_news, use_social=use_social)
    X = df[feat_cols].values
    y = df["target_direction"].values
    dates = pd.to_datetime(df["price_date"]).values

    results = []
    pipe = _build_pipeline("gb")  # initial
    start = 200  # warmup
    last_retrain = None

    pos_entry = None
    pos_exit_idx = None

    for i in range(start, len(df)-1):
        if stop_flag is not None and stop_flag.get("stop"):
            break

        # retrain every k steps or on first step
        if (last_retrain is None) or ((i - last_retrain) >= retrain_k):
            pipe.fit(X[:i], y[:i])
            last_retrain = i

        # predict t+1 using info up to i
        p_up = float(pipe.predict_proba([X[i]])[0, 1])
        signal = 1 if p_up >= buy_threshold else 0

        # manage mocked position with fixed hold_steps
        pnl = 0.0
        if signal and pos_entry is None:
            pos_entry = df["close_price"].iloc[i]
            pos_exit_idx = min(i + hold_steps, len(df)-1)

        if pos_entry is not None and i == pos_exit_idx:
            pnl = (df['close_price'].iloc[i] - pos_entry) / pos_entry
            pos_entry = None
            pos_exit_idx = None

        results.append({
            "price_date": df["price_date"].iloc[i+1],  # prediction is for next bar
            "proba": p_up,
            "actual": int(df['target_direction'].iloc[i]),
            "signal": signal,
            "pnl": pnl,
        })

    out = pd.DataFrame(results)
    out["cum_pnl"] = out["pnl"].cumsum()
    return out


def record_metric_and_suggestions(symbol: str, backtest_df: pd.DataFrame, model_name: str):
    """Compute ROC-AUC-like proxy and persist suggestions."""
    if backtest_df.empty:
        return
    # Simple hit-rate
    hits = ( (backtest_df["proba"]>=0.5) == (backtest_df["actual"]==1) ).mean()
    metric = ModelMetric(
        symbol=symbol, model_name=model_name, task="classification", freq="D", horizon=1,
        window_start=pd.to_datetime(backtest_df["price_date"].min()).to_pydatetime(),
        window_end=pd.to_datetime(backtest_df["price_date"].max()).to_pydatetime(),
        metric_name="hit_rate", metric_value=float(hits)
    )
    db_session.add(metric)

    suggestions = []

    if hits < 0.55:
        suggestions.append(("warn",
            "Accuracy is below 55%. Consider lowering buy_threshold by 0.02–0.05 and retraining more frequently (retrain_k÷2)."))
    if backtest_df["pnl"].sum() <= 0:
        suggestions.append(("info",
            "Strategy PnL is non-positive. Try enabling news & social features and increase hold_steps by +2."))

    # Volume/mention spike suggestion (requires features joined in UI path)
    # We keep generic; UI computes spikes separately.

    for sev, msg in suggestions:
        db_session.add(ModelSuggestion(symbol=symbol, model_name=model_name, severity=sev, suggestion=msg))

    db_session.commit()


# ---------------- CLI Commands ----------------

@mlbp.cli.command("train")
@click.option("--symbol", default="ALL", help="Ticker symbol (or ALL)")
@click.option("--model", "model_choice", type=click.Choice(["gb","rf"]), default="gb")
@click.option("--use-news/--no-news", default=False)
@click.option("--use-social/--no-social", default=False)
def cli_train(symbol: str, model_choice: str, use_news: bool, use_social: bool):
    """Train a model for one symbol or all enabled symbols."""
    Model.metadata.create_all(bind=db_session.bind)
    symbols: List[str]
    if symbol == "ALL":
        symbols = [s.symbol for s in db_session.query(Stock).filter(Stock.enabled == True).order_by(Stock.id)]
    else:
        symbols = [symbol]

    for sym in symbols:
        try:
            res = train_for_symbol(sym, model_choice, use_news=use_news, use_social=use_social)
            print(f"Trained {sym} — ROC AUC (cv): {res.cv_score:.3f} ({res.n_samples} samples) -> {res.model_path}")
        except Exception as e:
            print(f"[WARN] {sym}: {e}")


@mlbp.cli.command("predict")
@click.argument("symbol")
def cli_predict(symbol: str):
    """Predict T+1 probability of UP for SYMBOL and store to DB."""
    prices = load_prices(symbol)
    if prices.empty:
        raise RuntimeError(f"No price data for {symbol}.")
    proba, last_row = predict_proba(symbol)

    price_date = pd.to_datetime(last_row["price_date"]).to_pydatetime()
    target_date = price_date + dt.timedelta(days=1)
    rec = StockPrediction(
        stock_id=db_session.query(Stock.id).filter(Stock.symbol==symbol).scalar(),
        symbol=symbol, price_date=price_date, target_date=target_date,
        pred_type="direction", pred_value=proba, model_name="Merlin4",
        model_version="1"
    )
    db_session.add(rec); db_session.commit()
    print(json.dumps({
        "symbol": symbol, "price_date": price_date.isoformat(),
        "target_date": target_date.isoformat(), "proba_up": proba
    }, indent=2, default=str))


@mlbp.cli.command("backtest")
@click.argument("symbol")
@click.option("--retrain-k", default=10, type=int)
@click.option("--hold-steps", default=5, type=int)
@click.option("--buy-threshold", default=0.55, type=float)
@click.option("--use-news/--no-news", default=False)
@click.option("--use-social/--no-social", default=False)
def cli_backtest(symbol: str, retrain_k: int, hold_steps: int, buy_threshold: float, use_news: bool, use_social: bool):
    bt = walk_forward(symbol, retrain_k=retrain_k, hold_steps=hold_steps, buy_threshold=buy_threshold,
                      use_news=use_news, use_social=use_social)
    record_metric_and_suggestions(symbol, bt, model_name="Merlin4")
    print(bt.tail(10).to_string(index=False))
