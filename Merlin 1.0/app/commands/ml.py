import os
import json
import datetime as dt
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import click
from flask import Blueprint

from sqlalchemy import Column, DateTime, BigInteger, Float, Integer, String
from sqlalchemy.sql import func

from models.stockmarket import Stock, StockDaily, db_session, Model
from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey
from sqlalchemy.sql import func

# --------------------------------------------------------------------------------------
# Blueprint: `ml` — adds training, evaluate, and predict commands via Flask CLI
# Place this file at: ./commands/ml.py and register the blueprint in app.py
# --------------------------------------------------------------------------------------

mlbp = Blueprint("ml", __name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODELS_DIR = os.path.abspath(MODELS_DIR)
os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------------------------------------------------------------------
# SQLAlchemy model to save predictions (created if the table does not exist yet)
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
# Feature engineering
# --------------------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d",
    "ma5_ratio", "ma10_ratio", "ma20_ratio",
    "vol5", "vol10", "vol20",
    "rsi14",
    "range_ratio",  # (high-low)/close
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
    df["target_return_1d"] = df["close_price"].pct_change(-1)  # next day's return
    df["target_direction"] = (df["target_return_1d"] > 0).astype(int)

    df.dropna(inplace=True)

    return df


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
    df["price_date"] = pd.to_datetime(df["price_date"])  # DATE or DATETIME supported
    for col in ["open_price", "close_price", "high_price", "low_price", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    return df


# --------------------------------------------------------------------------------------
# Modeling (scikit-learn)
# --------------------------------------------------------------------------------------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)
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

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])
    return pipe


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

    cv = cross_val_score(pipe, X, y, cv=tscv, scoring=scoring, n_jobs=None)
    pipe.fit(X, y)

    model_name = f"GB{('Classifier' if task=='classification' else 'Regressor')}_v1"
    fname = f"{symbol}_{task}.joblib"
    model_path = os.path.join(MODELS_DIR, fname)
    joblib.dump({"pipeline": pipe, "feature_columns": FEATURE_COLUMNS, "task": task, "model_name": model_name}, model_path)

    return TrainResult(symbol=symbol, task=task, model_path=model_path, cv_score=float(np.mean(cv)), n_samples=len(df))


@mlbp.cli.command("train")
@click.option("--symbol", default="ALL", help="Ticker symbol (or ALL)")
@click.option("--task", type=click.Choice(["classification", "regression"]), default="classification")
def cli_train(symbol: str, task: str):
    """Train a model for one symbol or all enabled symbols."""
    # Ensure prediction table exists
    Model.metadata.create_all(bind=db_session.bind)

    symbols: List[str]
    if symbol == "ALL":
        symbols = [s.symbol for s in db_session.query(Stock).filter(Stock.enabled == True).order_by(Stock.id)]
    else:
        symbols = [symbol]

    for sym in symbols:
        try:
            res = train_for_symbol(sym, task)
            if task == "classification":
                print(f"Trained {sym} — ROC AUC: {res.cv_score:.3f} ({res.n_samples} samples) -> {res.model_path}")
            else:
                print(f"Trained {sym} — CV MAE: {-res.cv_score:.5f} ({res.n_samples} samples) -> {res.model_path}")
        except Exception as e:
            print(f"[WARN] {sym}: {e}")


def _load_model(symbol: str, task: str):
    path = os.path.join(MODELS_DIR, f"{symbol}_{task}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}. Train it first with 'flask ml train --symbol {symbol}'.")
    blob = joblib.load(path)
    return blob


@mlbp.cli.command("predict")
@click.argument("symbol")
@click.option("--task", type=click.Choice(["classification", "regression"]), default="classification")
@click.option("--write/--no-write", default=True, help="Write prediction to DB (stock_prediction)")
def cli_predict(symbol: str, task: str, write: bool):
    """Predict T+1 for the latest date available for SYMBOL and optionally store it."""
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
        pred_type = "direction"  # probability of up move tomorrow
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
        if not stk:
            raise RuntimeError(f"Unknown symbol in stock table: {symbol}")
        rec = StockPrediction(
            stock_id=stk.id,
            symbol=symbol,
            price_date=price_date,
            target_date=target_date,
            pred_type=pred_type,
            pred_value=pred_value,
            model_name=model_name,
            model_version="1",
        )
        db_session.add(rec)
        db_session.commit()
        print("Prediction saved to stock_prediction.")


@mlbp.cli.command("evaluate")
@click.argument("symbol")
@click.option("--task", type=click.Choice(["classification", "regression"]), default="classification")
def cli_evaluate(symbol: str, task: str):
    """Backtest-style evaluation on the full history for SYMBOL (time-series CV)."""
    prices = load_prices(symbol)
    if prices.empty:
        raise RuntimeError(f"No price data for {symbol}.")

    df = make_features(prices)
    X = df[FEATURE_COLUMNS].values
    y = df["target_direction"].values if task == "classification" else df["target_return_1d"].values

    pipe = _build_pipeline(task)
    tscv = TimeSeriesSplit(n_splits=5)
    if task == "classification":
        scores = cross_val_score(pipe, X, y, cv=tscv, scoring="roc_auc")
        print(f"ROC AUC: {scores.mean():.3f} ± {scores.std():.3f}")
    else:
        scores = -cross_val_score(pipe, X, y, cv=tscv, scoring="neg_mean_absolute_error")
        print(f"MAE: {scores.mean():.5f} ± {scores.std():.5f}")


# Convenience: bulk predict for all enabled stocks
@mlbp.cli.command("predict_all")
@click.option("--task", type=click.Choice(["classification", "regression"]), default="classification")
@click.option("--write/--no-write", default=True)
def cli_predict_all(task: str, write: bool):
    symbols = [s.symbol for s in db_session.query(Stock).filter(Stock.enabled == True).order_by(Stock.id)]
    for sym in symbols:
        try:
            cli_predict.callback(sym, task, write)  # re-use logic
        except Exception as e:
            print(f"[WARN] {sym}: {e}")
