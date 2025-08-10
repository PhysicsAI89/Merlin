# app/commands/ml.py
import os
import json
import datetime as dt
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd
import click
from flask import Blueprint

from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey
from sqlalchemy.sql import func

from models.stockmarket import Stock, StockDaily, db_session, Model

# --------------------------------------------------------------------------------------
# Blueprint & artifacts dir
# --------------------------------------------------------------------------------------
mlbp = Blueprint("ml", __name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODELS_DIR = os.path.abspath(MODELS_DIR)
os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------------------------------------------------------------------
# Persisted predictions (kept from your original design)
# --------------------------------------------------------------------------------------
class StockPrediction(Model):
    __tablename__ = "stock_prediction"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stock.id"), nullable=False)
    symbol = Column(String(10), nullable=False, index=True)
    price_date = Column(DateTime, nullable=False)   # date used for features
    target_date = Column(DateTime, nullable=False)  # predicted date (T+1)
    pred_type = Column(String(32), nullable=False)  # "return_1d" or "direction"
    pred_value = Column(Float, nullable=False)      # prob (direction) or return
    model_name = Column(String(64), nullable=False) # e.g., "GBClassifier_v1"
    model_version = Column(String(32), nullable=False, default="1")
    created_at = Column(DateTime, server_default=func.now())

# --------------------------------------------------------------------------------------
# NEW: suggestions & trade-decisions tables
# --------------------------------------------------------------------------------------
class ModelSuggestion(Model):
    __tablename__ = "model_suggestion"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())
    tag = Column(String(48), nullable=False)          # e.g. 'threshold', 'retrain', 'feature'
    suggestion = Column(String(2048), nullable=False) # human-readable text
    metric_name = Column(String(64), nullable=True)   # e.g. 'precision@thr', 'auc'
    metric_value = Column(Float, nullable=True)

class TradeDecision(Model):
    __tablename__ = "trade_decision"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stock.id"), nullable=False)
    symbol = Column(String(10), nullable=False, index=True)
    price_date = Column(DateTime, nullable=False, index=True)  # decision made at end of this bar
    decision = Column(String(16), nullable=False)              # 'buy' | 'sell' | 'wait' | 'hold'
    reason = Column(String(1024), nullable=True)
    confidence = Column(Float, nullable=True)                  # 0..1
    target_sell_date = Column(DateTime, nullable=True)
    meta_json = Column(String(4096), nullable=True)            # optional details

# ensure tables exist
Model.metadata.create_all(bind=db_session.bind)

# --------------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d",
    "ma5_ratio", "ma10_ratio", "ma20_ratio",
    "vol5", "vol10", "vol20",
    "rsi14",
    "range_ratio",
    "volume_z"
]

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values("price_date", inplace=True)
    df["ret_1d"] = df["close_price"].pct_change(1)
    df["ret_5d"] = df["close_price"].pct_change(5)
    df["ret_10d"] = df["close_price"].pct_change(10)

    for w in (5, 10, 20):
        df[f"ma{w}"] = df["close_price"].rolling(w).mean()
        df[f"ma{w}_ratio"] = df["close_price"] / df[f"ma{w}"] - 1

    for w in (5, 10, 20):
        df[f"vol{w}"] = df["ret_1d"].rolling(w).std()

    df["rsi14"] = _rsi(df["close_price"], 14)
    df["range_ratio"] = (df["high_price"] - df["low_price"]) / df["close_price"].replace(0, np.nan)

    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_std20"] = df["volume"].rolling(20).std()
    df["volume_z"] = (df["volume"] - df["volume_ma20"]) / df["volume_std20"].replace(0, np.nan)

    df["target_return_1d"] = df["close_price"].pct_change(-1)
    df["target_direction"] = (df["target_return_1d"] > 0).astype(int)
    df.dropna(inplace=True)
    return df

# --------------------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------------------
def load_prices(symbol: str, start: Optional[str] = None) -> pd.DataFrame:
    q = db_session.query(StockDaily).filter(StockDaily.symbol == symbol)
    if start:
        q = q.filter(StockDaily.price_date >= start)
    q = q.order_by(StockDaily.price_date)
    df = pd.read_sql(q.statement, db_session.bind)
    if df.empty:
        return df
    df["price_date"] = pd.to_datetime(df["price_date"])
    for col in ["open_price", "close_price", "high_price", "low_price", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    return df

# --------------------------------------------------------------------------------------
# Modeling
# --------------------------------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score
import joblib

@dataclass
class TrainResult:
    symbol: str
    task: str
    model_path: str
    cv_score: float
    n_samples: int

def _build_pipeline(task: str) -> Pipeline:
    if task == "classification":
        model = GradientBoostingClassifier(random_state=42)
    elif task == "regression":
        model = GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError("task must be 'classification' or 'regression'")
    return Pipeline([("scaler", StandardScaler()), ("model", model)])

def train_for_symbol(symbol: str, task: str = "classification") -> TrainResult:
    prices = load_prices(symbol)
    if prices.empty:
        raise RuntimeError(f"No price data found for {symbol}.")
    df = make_features(prices)
    X = df[FEATURE_COLUMNS].values
    y = df["target_direction"].values if task == "classification" else df["target_return_1d"].values

    pipe = _build_pipeline(task)
    tscv = TimeSeriesSplit(n_splits=5)
    scoring = "roc_auc" if task == "classification" else "neg_mean_absolute_error"
    cv = cross_val_score(pipe, X, y, cv=tscv, scoring=scoring)
    pipe.fit(X, y)

    model_name = f"GB{('Classifier' if task=='classification' else 'Regressor')}_v1"
    fname = f"{symbol}_{task}.joblib"
    model_path = os.path.join(MODELS_DIR, fname)
    joblib.dump({"pipeline": pipe, "feature_columns": FEATURE_COLUMNS, "task": task, "model_name": model_name}, model_path)

    return TrainResult(symbol=symbol, task=task, model_path=model_path, cv_score=float(np.mean(cv)), n_samples=len(df))

def _load_model(symbol: str, task: str):
    path = os.path.join(MODELS_DIR, f"{symbol}_{task}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}. Train it first.")
    return joblib.load(path)

# --------------------------------------------------------------------------------------
# Decision engine
# --------------------------------------------------------------------------------------
@dataclass
class DecisionThresholds:
    prob_buy: float = 0.60
    buzz_mult: float = 2.0
    vol_mult: float = 2.0
    hold_steps: int = 5

def recommend_trade_now(
    symbol: str,
    price_date: dt.datetime,
    proba_up: float,
    volume_factor: float,
    buzz_factor: float,
    thr: DecisionThresholds = DecisionThresholds()
) -> Dict[str, Any]:
    """
    Returns a dict with a proposed action and selling date.
    """
    reasons = []
    conf = proba_up
    action = "wait"

    if proba_up >= thr.prob_buy:
        reasons.append(f"Pr(up)={proba_up:.2f} ≥ {thr.prob_buy:.2f}")
    else:
        reasons.append(f"Pr(up) below {thr.prob_buy:.2f}")

    if volume_factor >= thr.vol_mult:
        reasons.append(f"volume x{volume_factor:.1f} vs 20d")
        conf = max(conf, min(0.95, 0.55 + 0.1 * (volume_factor - thr.vol_mult)))
    if buzz_factor >= thr.buzz_mult:
        reasons.append(f"buzz x{buzz_factor:.1f} vs baseline")
        conf = max(conf, min(0.97, 0.60 + 0.1 * (buzz_factor - thr.buzz_mult)))

    if (proba_up >= thr.prob_buy) and (volume_factor >= thr.vol_mult or buzz_factor >= thr.buzz_mult):
        action = "buy"
    elif proba_up >= (thr.prob_buy - 0.05):
        action = "wait"  # encouragingly close
    else:
        action = "wait"

    sell_date = price_date + dt.timedelta(days=thr.hold_steps)
    return {
        "action": action,
        "confidence": float(np.clip(conf, 0.0, 1.0)),
        "target_sell_date": sell_date,
        "reason": "; ".join(reasons),
    }

# --------------------------------------------------------------------------------------
# Walk-forward simulation (paper-trade on old data as if live)
# --------------------------------------------------------------------------------------
@dataclass
class SimSummary:
    n_steps: int
    n_trades: int
    win_rate: float
    auc: float
    avg_return_per_trade: float

def walkforward_simulate(
    symbol: str,
    buy_threshold: float = 0.60,
    hold_steps: int = 5,
    retrain_every: int = 10,
    start: Optional[str] = None
) -> Tuple[SimSummary, List[str]]:
    """
    Feed old data one bar at a time, predict next, compare with actual, retrain periodically,
    and record suggestions based on realized performance.
    """
    prices = load_prices(symbol, start=start)
    if prices.empty:
        raise RuntimeError(f"No price data for {symbol}.")

    df = make_features(prices)
    X = df[FEATURE_COLUMNS].values
    y = df["target_direction"].values
    dates = pd.to_datetime(df["price_date"]).tolist()

    thr = DecisionThresholds(prob_buy=buy_threshold, hold_steps=hold_steps)

    # Book-keeping
    preds, trade_returns = [], []
    n_trades = 0
    open_pos = 0
    last_buy_idx = None

    # initial fit on first chunk
    pipe = _build_pipeline("classification")
    lookback = max(30, hold_steps * 2)  # warmup
    pipe.fit(X[:lookback], y[:lookback])

    for i in range(lookback, len(X)-1):
        # predict next bar direction
        proba = float(pipe.predict_proba([X[i]])[0, 1])
        preds.append((dates[i], proba, y[i]))
        # write prediction row
        stk = db_session.query(Stock).filter(Stock.symbol == symbol).first()
        if stk:
            rec = StockPrediction(
                stock_id=stk.id, symbol=symbol,
                price_date=dates[i], target_date=dates[i] + dt.timedelta(days=1),
                pred_type="direction", pred_value=proba,
                model_name="GBClassifier_v1", model_version="1"
            )
            db_session.add(rec)

        # simple decision by prob only (volume/buzz are not guaranteed in backtest)
        if open_pos == 0 and proba >= buy_threshold:
            open_pos = 1
            last_buy_idx = i
            n_trades += 1
        elif open_pos == 1 and (i - last_buy_idx) >= hold_steps:
            # close after hold_steps
            r = float(df["target_return_1d"].iloc[last_buy_idx])  # next-day return after buy
            trade_returns.append(r)
            open_pos = 0

        # periodic retrain
        if (i - lookback) % retrain_every == 0 and i > lookback:
            pipe.fit(X[:i], y[:i])

    db_session.commit()

    # metrics
    y_true = [t for _, _, t in preds]
    y_score = [p for _, p, _ in preds]
    auc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.5
    win_rate = float(np.mean([1.0 if r > 0 else 0.0 for r in trade_returns])) if trade_returns else 0.0
    avg_ret = float(np.mean(trade_returns)) if trade_returns else 0.0

    summary = SimSummary(
        n_steps=len(preds),
        n_trades=n_trades,
        win_rate=win_rate,
        auc=auc,
        avg_return_per_trade=avg_ret,
    )

    # suggestions (saved to DB + returned for UI)
    suggestions: List[str] = []
    def _save(tag: str, text: str, metric_name: Optional[str] = None, metric_value: Optional[float] = None):
        suggestions.append(text)
        db_session.add(ModelSuggestion(symbol=symbol, tag=tag, suggestion=text,
                                       metric_name=metric_name, metric_value=metric_value))

    # Threshold nudges
    if auc > 0.55 and win_rate < 0.52:
        _save("threshold", f"AUC={auc:.2f} is ok but win_rate={win_rate:.2f} is low — raise buy threshold from {buy_threshold:.2f} to {min(0.80, buy_threshold+0.05):.2f}.", "auc", auc)
    if auc < 0.52:
        _save("retrain", f"AUC={auc:.2f} ~ random — retrain more often (every {max(3, retrain_every//2)} steps) or shorten warmup window.", "auc", auc)
    if avg_ret <= 0 and n_trades >= 10:
        _save("feature", "Average trade return ≤ 0 — try adding sentiment/buzz features and/or a volume spike rule (buy only if volume ≥ 2× 20-day avg).", "avg_ret", avg_ret)

    db_session.commit()
    return summary, suggestions

# --------------------------------------------------------------------------------------
# CLI commands
# --------------------------------------------------------------------------------------
@mlbp.cli.command("train")
@click.option("--symbol", default="ALL", help="Ticker symbol (or ALL)")
@click.option("--task", type=click.Choice(["classification", "regression"]), default="classification")
def cli_train(symbol: str, task: str):
    Model.metadata.create_all(bind=db_session.bind)
    symbols: List[str] = [s.symbol for s in db_session.query(Stock).filter(Stock.enabled == True).order_by(Stock.id)] if symbol == "ALL" else [symbol]
    for sym in symbols:
        try:
            res = train_for_symbol(sym, task)
            if task == "classification":
                print(f"Trained {sym} — ROC AUC: {res.cv_score:.3f} ({res.n_samples} samples) -> {res.model_path}")
            else:
                print(f"Trained {sym} — CV MAE: {-res.cv_score:.5f} ({res.n_samples} samples) -> {res.model_path}")
        except Exception as e:
            print(f"[WARN] {sym}: {e}")

@mlbp.cli.command("predict")
@click.argument("symbol")
@click.option("--task", type=click.Choice(["classification", "regression"]), default="classification")
@click.option("--write/--no-write", default=True)
def cli_predict(symbol: str, task: str, write: bool):
    prices = load_prices(symbol)
    if prices.empty:
        raise RuntimeError(f"No price data for {symbol}.")
    df = make_features(prices)
    X = df[FEATURE_COLUMNS].values
    last_row = df.iloc[-1]

    blob = _load_model(symbol, task)
    pipe = blob["pipeline"]
    model_name = blob["model_name"]

    if task == "classification":
        proba = float(pipe.predict_proba([X[-1]])[0, 1])
        pred_value = proba
        pred_type = "direction"
    else:
        rtn = float(pipe.predict([X[-1]])[0])
        pred_value = rtn
        pred_type = "return_1d"

    price_date = pd.to_datetime(last_row["price_date"]).to_pydatetime()
    target_date = price_date + dt.timedelta(days=1)

    print(json.dumps({
        "symbol": symbol,
        "price_date": price_date.isoformat(),
        "target_date": target_date.isoformat(),
        "pred_type": pred_type,
        "pred_value": pred_value,
        "model": model_name
    }, indent=2, default=str))

    if write:
        stk = db_session.query(Stock).filter(Stock.symbol == symbol).first()
        if stk:
            rec = StockPrediction(
                stock_id=stk.id, symbol=symbol, price_date=price_date, target_date=target_date,
                pred_type=pred_type, pred_value=pred_value, model_name=model_name, model_version="1",
            )
            db_session.add(rec)
            db_session.commit()
            print("Prediction saved to stock_prediction.")

@mlbp.cli.command("simulate")
@click.argument("symbol")
@click.option("--buy-threshold", default=0.60, type=float)
@click.option("--hold-steps", default=5, type=int)
@click.option("--retrain-every", default=10, type=int)
@click.option("--start", default=None)
def cli_simulate(symbol: str, buy_threshold: float, hold_steps: int, retrain_every: int, start: Optional[str]):
    summary, suggestions = walkforward_simulate(symbol, buy_threshold, hold_steps, retrain_every, start)
    print(json.dumps({
        "n_steps": summary.n_steps,
        "n_trades": summary.n_trades,
        "win_rate": summary.win_rate,
        "auc": summary.auc,
        "avg_return_per_trade": summary.avg_return_per_trade,
        "suggestions": suggestions
    }, indent=2))
