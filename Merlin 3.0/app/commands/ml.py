# ml.py
import os
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# Optional CLI/Blueprint support (safe fallback if Flask isn't installed)
try:
    from flask import Blueprint
    class _Dummy: ...
    mlbp = Blueprint("ml", __name__)
except Exception:
    class _DummyCLI:
        def command(self, *args, **kwargs):
            def _decorator(fn):
                return fn
            return _decorator
    class _DummyBP:
        cli = _DummyCLI()
    mlbp = _DummyBP()

try:
    import click
except Exception:
    # Provide a minimal shim so imports work even without click installed
    class click:  # type: ignore
        @staticmethod
        def option(*args, **kwargs):
            def _decorator(fn):
                return fn
            return _decorator

# --------- Import shims for various repo layouts ---------
try:
    from models.stockmarket import Stock, StockDaily, db_session, Model
except Exception:
    try:
        from app.models.stockmarket import Stock, StockDaily, db_session, Model
    except Exception:
        from stockmarket import Stock, StockDaily, db_session, Model

try:
    from commands.news import build_news_features
except Exception:
    try:
        from app.commands.news import build_news_features
    except Exception:
        from news import build_news_features

try:
    from commands.merlinnet import MerlinNetClassifier
except Exception:
    try:
        from app.commands.merlinnet import MerlinNetClassifier
    except Exception:
        from merlinnet import MerlinNetClassifier

# --------------------------
# Feature engineering
# --------------------------
FEATURE_COLUMNS_BASE = [
    "ret_1d", "ret_5d", "ret_10d",
    "ma5_ratio", "ma10_ratio", "ma20_ratio",
    "vol5", "vol10", "vol20",
    "rsi14",
    "range_ratio",
    "volume_z",
]

FEATURE_NEWS = [
    "news_sent_mean_3", "news_sent_mean_7",
    "news_count_3", "news_count_7",
    "news_rel_mean_7",
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

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["price_date"] = pd.to_datetime(d["price_date"]).dt.tz_localize(None)
    d.set_index("price_date", inplace=True)
    agg = {
        "open_price": "first",
        "high_price": "max",
        "low_price": "min",
        "close_price": "last",
        "volume": "sum",
    }
    out = d.resample("W-FRI").agg(agg).dropna().reset_index()
    return out

def make_features(df: pd.DataFrame, horizon: int = 1, use_news: bool = False, symbol: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    df.sort_values("price_date", inplace=True)

    # Returns
    df["ret_1d"] = df["close_price"].pct_change(1)
    df["ret_5d"] = df["close_price"].pct_change(5)
    df["ret_10d"] = df["close_price"].pct_change(10)

    # Moving averages (ratios)
    for w in (5, 10, 20):
        df[f"ma{w}"] = df["close_price"].rolling(w).mean()
        df[f"ma{w}_ratio"] = df["close_price"] / df[f"ma{w}"] - 1

    # Volatility
    for w in (5, 10, 20):
        df[f"vol{w}"] = df["ret_1d"].rolling(w).std()

    # RSI & range
    df["rsi14"] = _rsi(df["close_price"], 14)
    df["range_ratio"] = (df["high_price"] - df["low_price"]) / df["close_price"].replace(0, np.nan)

    # Volume z-score (20d)
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_std20"] = df["volume"].rolling(20).std()
    df["volume_z"] = (df["volume"] - df["volume_ma20"]) / df["volume_std20"].replace(0, np.nan)

    # Targets for H-step ahead
    df["target_return_H"] = df["close_price"].pct_change(-horizon)
    df["target_direction"] = (df["target_return_H"] > 0).astype(int)

    # Merge news features (daily granularity)
    if use_news and symbol is not None:
        try:
            news = build_news_features(symbol, start_date=str(df["price_date"].min().date()))
        except Exception:
            news = pd.DataFrame()
        if isinstance(news, pd.DataFrame) and not news.empty:
            df = df.merge(news, how="left", left_on="price_date", right_on="price_date")
        # Ensure columns exist; we'll fill them with zeros instead of dropping rows
        for c in FEATURE_NEWS:
            if c not in df.columns:
                df[c] = 0.0
        df[FEATURE_NEWS] = df[FEATURE_NEWS].fillna(0.0)

    # Safer NA handling: only enforce core tech pre-reqs + target, then clean infinities
    base_needed = ["ret_10d", "ma20", "vol20", "rsi14", "volume_ma20", "volume_std20", "target_return_H"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=base_needed)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df

# --------------------------
# Data access
# --------------------------
def load_prices(symbol: str, start: Optional[str] = None) -> pd.DataFrame:
    q = db_session.query(StockDaily).filter(StockDaily.symbol == symbol)
    if start:
        q = q.filter(StockDaily.price_date >= start)
    q = q.order_by(StockDaily.price_date)
    df = pd.read_sql(q.statement, db_session.bind)
    if df.empty:
        return df
    df["price_date"] = pd.to_datetime(df["price_date"])  # naive OK for SQLite
    for col in ["open_price", "close_price", "high_price", "low_price", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    return df

# --------------------------
# Modeling
# --------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import joblib

@dataclass
class TrainResult:
    symbol: str
    task: str
    model_kind: str
    horizon: int
    freq: str
    model_path: str
    cv_score: float
    n_samples: int

def _build_estimator(model_kind: str):
    if model_kind == "gb":
        return GradientBoostingClassifier(random_state=42)
    if model_kind == "rf":
        return RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42)
    if model_kind == "xgb" and _HAS_XGB:
        return XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=42
        )
    if model_kind == "merlin":
        return MerlinNetClassifier(hidden=64, lr=0.01, epochs=250, batch_size=512, l2=1e-4, seed=42)
    raise ValueError("Unknown model_kind or missing xgboost.")

def _random_search(model_kind: str, X, y, n_iter: int = 12):
    """Lightweight self-improvement via random search with small-N fallback."""
    rng = np.random.default_rng(42)
    n = len(X)
    if n < 2:
        raise RuntimeError("Not enough samples after feature engineering (need ≥2).")

    def _sample_est():
        if model_kind == "rf":
            return RandomForestClassifier(
                n_estimators=int(rng.integers(200, 600)),
                max_depth=int(rng.integers(4, 16)),
                max_features=rng.choice(["sqrt", "log2", None]),
                n_jobs=-1,
                random_state=int(rng.integers(0, 1_000_000)),
            )
        if model_kind == "gb":
            return GradientBoostingClassifier(
                n_estimators=int(rng.integers(100, 500)),
                learning_rate=float(rng.choice([0.01, 0.02, 0.05, 0.1])),
                max_depth=int(rng.integers(2, 5)),
                random_state=int(rng.integers(0, 1_000_000)),
            )
        if model_kind == "xgb" and _HAS_XGB:
            return XGBClassifier(
                n_estimators=int(rng.integers(200, 700)),
                max_depth=int(rng.integers(3, 9)),
                learning_rate=float(rng.choice([0.01, 0.02, 0.05, 0.1])),
                subsample=float(rng.uniform(0.7, 1.0)),
                colsample_bytree=float(rng.uniform(0.7, 1.0)),
                tree_method="hist",
                random_state=int(rng.integers(0, 1_000_000)),
            )
        if model_kind == "merlin":
            return MerlinNetClassifier(
                hidden=int(rng.integers(16, 128)),
                lr=float(rng.choice([0.003, 0.005, 0.01, 0.02])),
                epochs=int(rng.integers(120, 320)),
                batch_size=int(rng.choice([128, 256, 512])),
                l2=float(rng.choice([1e-5, 1e-4, 5e-4, 1e-3])),
                seed=int(rng.integers(0, 1_000_000)),
            )
        return _build_estimator(model_kind)

    best = None
    best_score = -np.inf

    if n >= 40:
        n_splits = min(5, max(2, n // 20))  # cap at 5 folds, scale with data
        splitter = TimeSeriesSplit(n_splits=n_splits)
        for _ in range(n_iter):
            pipe = Pipeline([("scaler", StandardScaler()), ("model", _sample_est())])
            scores = []
            for tr, te in splitter.split(X):
                Xtr, Xte = X[tr], X[te]
                ytr, yte = y[tr], y[te]
                pipe.fit(Xtr, ytr)
                p = pipe.predict_proba(Xte)[:, 1]
                scores.append(roc_auc_score(yte, p))
            score = float(np.mean(scores))
            if score > best_score:
                best_score, best = score, pipe
    elif n >= 10:
        # simple 70/30 holdout repeated search
        split = max(1, int(n * 0.7))
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = y[:split], y[split:]
        for _ in range(n_iter):
            pipe = Pipeline([("scaler", StandardScaler()), ("model", _sample_est())])
            pipe.fit(Xtr, ytr)
            if len(Xte) > 0 and len(np.unique(yte)) > 1:
                p = pipe.predict_proba(Xte)[:, 1]
                score = float(roc_auc_score(yte, p))
            else:
                score = 0.5
            if score > best_score:
                best_score, best = score, pipe
    else:
        # tiny data: just fit; nominal CV score
        pipe = Pipeline([("scaler", StandardScaler()), ("model", _sample_est())])
        pipe.fit(X, y)
        best, best_score = pipe, 0.5

    return best, best_score

def _models_dir_here() -> str:
    # Save to app/artifacts if this file is in app/commands/
    base = os.path.abspath(os.path.dirname(__file__))
    target = os.path.abspath(os.path.join(base, "..", "artifacts"))
    os.makedirs(target, exist_ok=True)
    return target

def train_for_symbol(symbol: str, horizon: int = 1, freq: str = "D", model_kind: str = "gb", use_news: bool = False) -> 'TrainResult':
    prices = load_prices(symbol)
    if prices.empty:
        raise RuntimeError(f"No price data found for {symbol}.")
    if freq == "W":
        prices = to_weekly(prices)
    df = make_features(prices, horizon=horizon, use_news=use_news, symbol=symbol if use_news else None)

    if df.empty:
        raise RuntimeError(
            f"Not enough rows after feature engineering for {symbol}. "
            "Tip: Fetch more history, switch to Daily, set Horizon=1, or untick 'Use news sentiment'."
        )

    # choose features
    feat_cols = FEATURE_COLUMNS_BASE + (FEATURE_NEWS if use_news else [])
    X = df[feat_cols].values
    y = df["target_direction"].values
    if len(X) < 2:
        raise RuntimeError("Not enough samples after feature engineering (need ≥2).")

    best, cv_score = _random_search(model_kind, X, y, n_iter=10 if model_kind != "merlin" else 16)
    best.fit(X, y)

    MODELS_DIR = _models_dir_here()
    model_name = f"{model_kind.upper()}_{freq}_H{horizon}"
    fname = f"{symbol}_{model_kind}_{freq}_H{horizon}.joblib"
    model_path = os.path.join(MODELS_DIR, fname)
    joblib.dump(
        {
            "pipeline": best,
            "feature_columns": feat_cols,
            "horizon": horizon,
            "freq": freq,
            "model_name": model_name,
            "use_news": use_news,
        },
        model_path,
    )

    # record metric (self-awareness)
    try:
        try:
            from models.extensions import ModelMetric
        except Exception:
            from app.models.extensions import ModelMetric  # type: ignore
        mm = ModelMetric(
            symbol=symbol,
            model_name=model_name,
            horizon=horizon,
            freq=freq,
            window_start=df["price_date"].min(),
            window_end=df["price_date"].max(),
            metric_name="roc_auc",
            metric_value=float(cv_score),
        )
        db_session.add(mm)
        db_session.commit()
    except Exception:
        # metrics table may not exist yet; ignore silently
        pass

    return TrainResult(
        symbol=symbol,
        task="classification",
        model_kind=model_kind,
        horizon=horizon,
        freq=freq,
        model_path=model_path,
        cv_score=float(cv_score),
        n_samples=len(df),
    )

def _load_model(symbol: str, model_kind: str, freq: str, horizon: int):
    MODELS_DIR = _models_dir_here()
    path = os.path.join(MODELS_DIR, f"{symbol}_{model_kind}_{freq}_H{horizon}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}.")
    return joblib.load(path)

# Optional CLI entry (safe even without Flask/click; acts as a no-op decorator)
@mlbp.cli.command("train")  # type: ignore[attr-defined]
@click.option("--symbol", default="ALL")
@click.option("--horizon", default=1, type=int)
@click.option("--freq", default="D", type=click.Choice(["D", "W"]))
@click.option("--model", "model_kind", default="gb", type=click.Choice(["gb", "rf", "xgb", "merlin"]))
@click.option("--use-news/--no-news", default=False)
def cli_train(symbol: str, horizon: int, freq: str, model_kind: str, use_news: bool):
    # Ensure base metadata is present (tables create elsewhere if needed)
    try:
        Model.metadata.create_all(bind=db_session.bind)
    except Exception:
        pass
    symbols: List[str] = [s.symbol for s in db_session.query(Stock).filter(Stock.enabled == True)] if symbol == "ALL" else [symbol]
    for sym in symbols:
        try:
            res = train_for_symbol(sym, horizon=horizon, freq=freq, model_kind=model_kind, use_news=use_news)
            print(f"Trained {sym} — {res.model_kind} {res.freq} H{res.horizon} — ROC AUC {res.cv_score:.3f} ({res.n_samples}) -> {res.model_path}")
        except Exception as e:
            print(f"[WARN] {sym}: {e}")

def backtest_walkforward(symbol: str, threshold: float, hold_steps: int, retrain_k: int, horizon: int, freq: str, model_kind: str, use_news: bool = False) -> Dict:
    prices = load_prices(symbol)
    if prices.empty:
        return {"trades": [], "n": 0, "ret": 0.0, "equity": 1.0}
    if freq == "W":
        prices = to_weekly(prices)
    df = make_features(prices, horizon=horizon, use_news=use_news, symbol=symbol if use_news else None)
    if df.empty:
        return {"trades": [], "n": 0, "ret": 0.0, "equity": 1.0}
    feat_cols = FEATURE_COLUMNS_BASE + (FEATURE_NEWS if use_news else [])

    trades: List[Dict] = []
    equity = 1.0
    # Dynamic warmup: ensure enough data while not skipping everything
    warmup = max(10, min(50, len(df) // 3))
    i = warmup
    best = None
    while i < len(df) - horizon:
        # retrain periodically
        if best is None or (i - warmup) % max(1, retrain_k) == 0:
            Xtr = df.iloc[:i][feat_cols].values
            ytr = df.iloc[:i]["target_direction"].values
            if len(Xtr) < 2 or len(np.unique(ytr)) < 2:
                # not enough data or only one class so far — skip trading until we have diversity
                i += 1
                continue
            best, _ = _random_search(model_kind, Xtr, ytr, n_iter=6)
            best.fit(Xtr, ytr)

        # If we still don't have a model (e.g., tiny data), break
        if best is None:
            break

        # predict proba for current bar
        x = df.iloc[i][feat_cols].values.reshape(1, -1)
        p_up = float(best.predict_proba(x)[0, 1])
        if p_up >= threshold:
            # enter long, hold for k steps
            entry_idx = i
            exit_idx = min(i + hold_steps, len(df) - 1)
            r = float((df.iloc[exit_idx]["close_price"] / df.iloc[entry_idx]["close_price"]) - 1)
            equity *= (1 + r)
            trades.append(
                {
                    "entry_date": df.iloc[entry_idx]["price_date"],
                    "exit_date": df.iloc[exit_idx]["price_date"],
                    "p_up": p_up,
                    "ret": r,
                }
            )
            i = exit_idx + 1
        else:
            i += 1
    return {"trades": trades, "n": len(trades), "equity": equity, "ret": equity - 1}
