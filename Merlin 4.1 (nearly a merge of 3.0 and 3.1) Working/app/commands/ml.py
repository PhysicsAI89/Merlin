import os
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
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

try:
    from models.stockmarket import Stock, StockDaily, db_session, Model
except Exception:
    try:
        from app.models.stockmarket import Stock, StockDaily, db_session, Model
    except Exception:
        from ..models.stockmarket import Stock, StockDaily, db_session, Model

try:
    from models.extensions import ModelMetric, ModelSuggestion, StockPrediction
except Exception:
    try:
        from app.models.extensions import ModelMetric, ModelSuggestion, StockPrediction
    except Exception:
        from ..models.extensions import ModelMetric, ModelSuggestion, StockPrediction

try:
    from .news import build_news_features
except Exception:
    from app.commands.news import build_news_features

try:
    from .merlinnet import MerlinNetClassifier
except Exception:
    from app.commands.merlinnet import MerlinNetClassifier

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
    agg = {"open_price":"first","high_price":"max","low_price":"min","close_price":"last","volume":"sum"}
    out = d.resample("W-FRI").agg(agg).dropna().reset_index()
    return out

def make_features(df: pd.DataFrame, horizon: int = 1, use_news: bool = False, symbol: Optional[str] = None) -> pd.DataFrame:
    df = df.copy(); df.sort_values("price_date", inplace=True)
    df["ret_1d"] = df["close_price"].pct_change(1)
    df["ret_5d"] = df["close_price"].pct_change(5)
    df["ret_10d"] = df["close_price"].pct_change(10)
    for w in (5,10,20):
        df[f"ma{w}"] = df["close_price"].rolling(w).mean()
        df[f"ma{w}_ratio"] = df["close_price"] / df[f"ma{w}"] - 1
    for w in (5,10,20):
        df[f"vol{w}"] = df["ret_1d"].rolling(w).std()
    df["rsi14"] = _rsi(df["close_price"], 14)
    df["range_ratio"] = (df["high_price"] - df["low_price"]) / df["close_price"].replace(0, np.nan)
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_std20"] = df["volume"].rolling(20).std()
    df["volume_z"] = (df["volume"] - df["volume_ma20"]) / df["volume_std20"].replace(0, np.nan)
    df["target_return_H"] = df["close_price"].pct_change(-horizon)
    df["target_direction"] = (df["target_return_H"] > 0).astype(int)

    if use_news and symbol is not None:
        try:
            news = build_news_features(symbol, start_date=str(df["price_date"].min().date()))
        except Exception:
            news = pd.DataFrame()
        if isinstance(news, pd.DataFrame) and not news.empty:
            df = df.merge(news, how="left", on="price_date")
        for c in FEATURE_NEWS:
            if c not in df.columns:
                df[c] = 0.0
        df[FEATURE_NEWS] = df[FEATURE_NEWS].fillna(0.0)

    base_needed = ["ret_10d","ma20","vol20","rsi14","volume_ma20","volume_std20","target_return_H"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=base_needed)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def load_prices(symbol: str, start: Optional[str] = None) -> pd.DataFrame:
    q = db_session.query(StockDaily).filter(StockDaily.symbol == symbol)
    if start: q = q.filter(StockDaily.price_date >= start)
    q = q.order_by(StockDaily.price_date)
    df = pd.read_sql(q.statement, db_session.bind)
    if df.empty: return df
    df["price_date"] = pd.to_datetime(df["price_date"])
    for c in ["open_price","close_price","high_price","low_price","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    return df

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
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=42)
    if model_kind == "merlin":
        return MerlinNetClassifier(hidden=64, lr=0.01, epochs=250, batch_size=512, l2=1e-4, seed=42)
    raise ValueError("Unknown model_kind or missing xgboost.")

def _random_search(model_kind: str, X, y, n_iter: int = 12, stop_env_key: str = "MERLIN_STOP"):
    rng = np.random.default_rng(42)
    n = len(X)
    if n < 2:
        raise RuntimeError("Not enough samples after feature engineering (need ≥2).")

    def _sample_est():
        if model_kind == "rf":
            return RandomForestClassifier(
                n_estimators=int(rng.integers(200, 600)),
                max_depth=int(rng.integers(4, 16)),
                max_features=rng.choice(["sqrt","log2",None]),
                n_jobs=-1,
                random_state=int(rng.integers(0, 1_000_000)),
            )
        if model_kind == "gb":
            return GradientBoostingClassifier(
                n_estimators=int(rng.integers(100, 500)),
                learning_rate=float(rng.choice([0.01,0.02,0.05,0.1])),
                max_depth=int(rng.integers(2, 5)),
                random_state=int(rng.integers(0, 1_000_000)),
            )
        if model_kind == "xgb" and _HAS_XGB:
            return XGBClassifier(
                n_estimators=int(rng.integers(200, 700)),
                max_depth=int(rng.integers(3, 9)),
                learning_rate=float(rng.choice([0.01,0.02,0.05,0.1])),
                subsample=float(rng.uniform(0.7, 1.0)),
                colsample_bytree=float(rng.uniform(0.7, 1.0)),
                tree_method="hist",
                random_state=int(rng.integers(0, 1_000_000)),
            )
        if model_kind == "merlin":
            return MerlinNetClassifier(
                hidden=int(rng.integers(16, 128)),
                lr=float(rng.choice([0.003,0.005,0.01,0.02])),
                epochs=int(rng.integers(120, 320)),
                batch_size=int(rng.choice([128,256,512])),
                l2=float(rng.choice([1e-5,1e-4,5e-4,1e-3])),
                seed=int(rng.integers(0, 1_000_000)),
            )
        return _build_estimator(model_kind)

    best=None; best_score=-np.inf
    if n >= 40:
        n_splits = min(5, max(2, n // 20))
        splitter = TimeSeriesSplit(n_splits=n_splits)
        for _ in range(n_iter):
            if os.getenv(stop_env_key) == "1": break
            pipe = Pipeline([("scaler", StandardScaler()), ("model", _sample_est())])
            scores = []
            for tr, te in splitter.split(X):
                Xtr, Xte = X[tr], X[te]; ytr, yte = y[tr], y[te]
                pipe.fit(Xtr, ytr)
                p = pipe.predict_proba(Xte)[:,1]
                scores.append(roc_auc_score(yte, p))
            score = float(np.mean(scores))
            if score > best_score: best_score, best = score, pipe
    elif n >= 10:
        split = max(1, int(n * 0.7))
        Xtr, Xte = X[:split], X[split:]; ytr, yte = y[:split], y[split:]
        for _ in range(n_iter):
            if os.getenv(stop_env_key) == "1": break
            pipe = Pipeline([("scaler", StandardScaler()), ("model", _sample_est())])
            pipe.fit(Xtr, ytr)
            if len(Xte)>0 and len(np.unique(yte))>1:
                p = pipe.predict_proba(Xte)[:,1]; score = float(roc_auc_score(yte, p))
            else:
                score = 0.5
            if score > best_score: best_score, best = score, pipe
    else:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", _sample_est())])
        pipe.fit(X, y); best, best_score = pipe, 0.5
    return best, best_score

def _models_dir_here() -> str:
    base = os.path.abspath(os.path.dirname(__file__))
    target = os.path.abspath(os.path.join(base, "..", "artifacts"))
    os.makedirs(target, exist_ok=True)
    return target

def train_for_symbol(symbol: str, horizon: int = 1, freq: str = "D", model_kind: str = "gb", use_news: bool = False) -> 'TrainResult':
    prices = load_prices(symbol)
    if prices.empty:
        raise RuntimeError(f"No price data found for {symbol}.")
    if freq == "W": prices = to_weekly(prices)
    df = make_features(prices, horizon=horizon, use_news=use_news, symbol=symbol if use_news else None)
    if df.empty:
        raise RuntimeError("Not enough rows after feature engineering.")

    feat_cols = FEATURE_COLUMNS_BASE + (FEATURE_NEWS if use_news else [])
    X = df[feat_cols].values; y = df["target_direction"].values
    if len(X) < 2: raise RuntimeError("Not enough samples after feature engineering (need ≥2).")

    best, cv_score = _random_search(model_kind, X, y, n_iter=10 if model_kind!="merlin" else 16)
    best.fit(X, y)

    MODELS_DIR = _models_dir_here()
    model_name = f"{model_kind.upper()}_{freq}_H{horizon}"
    fname = f"{symbol}_{model_kind}_{freq}_H{horizon}.joblib"
    path = os.path.join(MODELS_DIR, fname)
    joblib.dump({"pipeline":best,"feature_columns":feat_cols,"horizon":horizon,"freq":freq,
                 "model_name":model_name,"use_news":use_news}, path)

    try:
        mm = ModelMetric(symbol=symbol, model_name=model_name, horizon=horizon, freq=freq,
                         window_start=df["price_date"].min(), window_end=df["price_date"].max(),
                         metric_name="roc_auc", metric_value=float(cv_score))
        db_session.add(mm); db_session.commit()
    except Exception:
        pass

    return TrainResult(symbol=symbol, task="classification", model_kind=model_kind, horizon=horizon, freq=freq,
                       model_path=path, cv_score=float(cv_score), n_samples=len(df))

def _load_model(symbol: str, model_kind: str, freq: str, horizon: int):
    MODELS_DIR = _models_dir_here()
    path = os.path.join(MODELS_DIR, f"{symbol}_{model_kind}_{freq}_H{horizon}.joblib")
    if not os.path.exists(path): raise FileNotFoundError(f"Model not found: {path}.")
    return joblib.load(path)

def backtest_walkforward(symbol: str, threshold: float, hold_steps: int, retrain_k: int, horizon: int, freq: str, model_kind: str, use_news: bool = False) -> Dict:
    prices = load_prices(symbol)
    if prices.empty: return {"trades": [], "n": 0, "ret": 0.0, "equity": 1.0}
    if freq == "W": prices = to_weekly(prices)
    df = make_features(prices, horizon=horizon, use_news=use_news, symbol=symbol if use_news else None)
    if df.empty: return {"trades": [], "n": 0, "ret": 0.0, "equity": 1.0}
    feat_cols = FEATURE_COLUMNS_BASE + (FEATURE_NEWS if use_news else [])

    trades: List[Dict] = []
    equity = 1.0; warmup = max(10, min(50, len(df)//3)); i = warmup; best=None
    while i < len(df) - horizon:
        if os.getenv("MERLIN_STOP") == "1": break
        if best is None or (i - warmup) % max(1, retrain_k) == 0:
            Xtr = df.iloc[:i][feat_cols].values; ytr = df.iloc[:i]["target_direction"].values
            if len(Xtr) < 2 or len(np.unique(ytr)) < 2: i += 1; continue
            best, _ = _random_search(model_kind, Xtr, ytr, n_iter=6); best.fit(Xtr, ytr)
        if best is None: break
        x = df.iloc[i][feat_cols].values.reshape(1,-1); p_up = float(best.predict_proba(x)[0,1])
        if p_up >= threshold:
            entry_idx=i; exit_idx=min(i+hold_steps, len(df)-1)
            r = float((df.iloc[exit_idx]["close_price"]/df.iloc[entry_idx]["close_price"]) - 1)
            equity *= (1+r)
            trades.append({"entry_date":df.iloc[entry_idx]["price_date"],"exit_date":df.iloc[exit_idx]["price_date"],"p_up":p_up,"ret":r})
            i = exit_idx + 1
        else:
            i += 1
    return {"trades": trades, "n": len(trades), "equity": equity, "ret": equity - 1}

def log_prediction(symbol: str, price_date: pd.Timestamp, target_date: pd.Timestamp, p_up: float, model_name: str, model_version: str = "1"):
    stk = db_session.query(Stock).filter(Stock.symbol == symbol).first()
    if not stk: return
    rec = StockPrediction(stock_id=stk.id, symbol=symbol, price_date=price_date.to_pydatetime(),
                          target_date=target_date.to_pydatetime(), pred_type="direction",
                          pred_value=float(p_up), model_name=model_name, model_version=model_version)
    db_session.add(rec); db_session.commit()

def generate_suggestions(symbol: str, freq: str, horizon: int, window: int = 120) -> List[Dict]:
    q = db_session.query(StockPrediction).filter(StockPrediction.symbol==symbol).order_by(StockPrediction.id.desc()).limit(window)
    preds = list(reversed(q.all()))
    if len(preds) < 10: return []
    prices = load_prices(symbol); prices = to_weekly(prices) if freq=="W" else prices
    px = prices[["price_date","close_price"]].copy().reset_index(drop=True)
    dfp = pd.DataFrame([{"price_date":pd.to_datetime(p.price_date), "target_date":pd.to_datetime(p.target_date), "p_up":p.pred_value} for p in preds])
    dfm = dfp.merge(px.rename(columns={"price_date":"target_date","close_price":"close_t"}), on="target_date", how="left")
    dfm = dfm.merge(px.rename(columns={"price_date":"price_date","close_price":"close_0"}), on="price_date", how="left")
    dfm.dropna(inplace=True); dfm["real_up"] = (dfm["close_t"] > dfm["close_0"]).astype(int)
    if len(dfm) < 10: return []
    hit = float(((dfm["p_up"]>=0.5) == (dfm["real_up"]==1)).mean())
    conf_cal = float(abs(dfm["p_up"].mean() - dfm["real_up"].mean()))
    var = float(dfm["p_up"].var())
    sugs = []
    def _push(text, score):
        ms = ModelSuggestion(symbol=symbol, freq=freq, horizon=horizon, suggestion=text[:4096], score=float(score))
        db_session.add(ms); sugs.append({"suggestion":text, "score":float(score)})
    if hit < 0.55: _push(f"Hit-rate {hit:.2f} < 0.55 — try XGB or include news features.", 0.9)
    if conf_cal > 0.10: _push(f"Calibration gap {conf_cal:.2f} — reduce learning rate / shallower trees (GB) or increase estimators.", 0.6)
    if var < 0.01: _push("Probas too flat — add volatility/RSI horizons or switch to Weekly.", 0.5)
    if hit < 0.50 and freq == "D": _push("Daily noisy — try Weekly and hold 4–6 steps.", 0.7)
    db_session.commit(); return sugs
