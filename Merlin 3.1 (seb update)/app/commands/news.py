# commands/news.py
import os, json, datetime as dt
from typing import List, Dict, Optional
import pandas as pd
import urllib.parse, urllib.request

from sqlalchemy.exc import OperationalError
from models.stockmarket import db_session
from models.extensions import NewsEvent

ALPHA_BASE = "https://www.alphavantage.co/query"

def _ensure_tables():
    try:
        from models.stockmarket import Model as _Base
        import models.extensions as _  # register tables
    except Exception:
        from app.models.stockmarket import Model as _Base  # type: ignore
        import app.models.extensions as _                  # type: ignore
    try:
        _Base.metadata.create_all(bind=db_session.bind)
    except Exception:
        pass

def _map_symbol_for_news(symbol: str) -> Dict[str, List[str]]:
    """Return candidate tickers and topics for AV news."""
    symbol = symbol.upper()
    tickers: List[str] = []
    topics: List[str] = []

    if "-" in symbol:  # BTC-USD, ETH-USD
        base = symbol.split("-", 1)[0]
        tickers = [base, f"CRYPTO:{base}", f"CRYPTO:{base}-USD"]
        topics = [base.lower(), "crypto", "blockchain", "bitcoin", "ethereum"]
    elif symbol.endswith("=X"):  # GBPUSD=X
        pair = symbol.replace("=X", "")
        tickers = [pair, pair[:3], pair[3:]]
        topics = [pair.lower(), "forex", "currency"]
    elif symbol.endswith("=F"):  # futures like CL=F, GC=F
        root = symbol.replace("=F", "")
        # AV usually won't have futures tickers; rely on topics
        tickers = []  # keep empty so we try topics directly
        if root in ("CL", "BZ"):
            topics = ["oil", "crude", "WTI", "OPEC", "energy"]
        elif root in ("GC", "SI"):
            topics = ["gold", "precious metals", "commodities"]
        else:
            topics = [root.lower(), "commodities"]
    else:
        # Equities: VOD.L -> VOD, keep original too
        if symbol.endswith(".L"):
            tickers = [symbol.split(".")[0], symbol]
        else:
            tickers = [symbol]
        topics = []

    return {"tickers": tickers, "topics": topics}

def _av_get(params: Dict) -> Dict:
    url = f"{ALPHA_BASE}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())

def _av_news_try(
    tickers: Optional[List[str]],
    topics: Optional[List[str]],
    apikey: str,
    lookback_days: int = 45,
    limit: int = 100,
) -> List[Dict]:
    """Try tickers first; if empty, try topics. Returns a 'feed' list (may be empty)."""
    time_from = (dt.datetime.utcnow() - dt.timedelta(days=lookback_days)).strftime("%Y%m%dT0000")
    # 1) tickers search
    if tickers:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(tickers),
            "sort": "LATEST",
            "limit": str(limit),
            "time_from": time_from,
            "apikey": apikey,
        }
        data = _av_get(params)
        feed = data.get("feed", [])
        if feed:
            return feed
    # 2) topics fallback
    if topics:
        params = {
            "function": "NEWS_SENTIMENT",
            "topics": ",".join(topics[:5]),
            "sort": "LATEST",
            "limit": str(limit),
            "time_from": time_from,
            "apikey": apikey,
        }
        data = _av_get(params)
        return data.get("feed", [])
    return []

def ingest_news(symbol: str, since: Optional[dt.datetime] = None) -> int:
    _ensure_tables()
    apikey = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not apikey:
        return 0

    m = _map_symbol_for_news(symbol)
    feed = _av_news_try(m["tickers"], m["topics"], apikey)
    if not feed:
        return 0

    rows = 0
    for item in feed:
        try:
            dt_pub = pd.to_datetime(item.get("time_published")).to_pydatetime()
        except Exception:
            continue
        if since and dt_pub <= since:
            continue
        se = NewsEvent(
            symbol=symbol,
            published_at=dt_pub,
            title=item.get("title"),
            source=item.get("source"),
            url=item.get("url"),
            overall_sentiment_score=float(item.get("overall_sentiment_score", 0.0)),
            overall_sentiment_label=item.get("overall_sentiment_label"),
            relevance_score=float(item.get("relevance_score", 0.0)) if item.get("relevance_score") else None,
        )
        db_session.add(se)
        rows += 1
    if rows:
        db_session.commit()
    return rows

def build_news_features(symbol: str, start_date: str) -> pd.DataFrame:
    _ensure_tables()
    try:
        q = db_session.query(NewsEvent).filter(NewsEvent.symbol == symbol)
        if start_date:
            q = q.filter(NewsEvent.published_at >= start_date)
        q = q.order_by(NewsEvent.published_at)
        df = pd.read_sql(q.statement, db_session.bind)
    except OperationalError:
        return pd.DataFrame(columns=[
            "price_date","news_sent_mean_3","news_sent_mean_7",
            "news_count_3","news_count_7","news_rel_mean_7"
        ])
    if df.empty:
        return pd.DataFrame(columns=[
            "price_date","news_sent_mean_3","news_sent_mean_7",
            "news_count_3","news_count_7","news_rel_mean_7"
        ])
    df["published_at"] = pd.to_datetime(df["published_at"]).dt.floor("D")
    g = df.groupby("published_at").agg(
        news_sent_mean=("overall_sentiment_score","mean"),
        news_count=("id","count"),
        news_rel_mean=("relevance_score","mean"),
    ).rename_axis("price_date").reset_index()
    g = g.sort_values("price_date").set_index("price_date")
    out = pd.DataFrame(index=g.index)
    out["news_sent_mean_3"]  = g["news_sent_mean"].rolling(3,  min_periods=1).mean()
    out["news_sent_mean_7"]  = g["news_sent_mean"].rolling(7,  min_periods=1).mean()
    out["news_count_3"]      = g["news_count"].rolling(3,  min_periods=1).sum()
    out["news_count_7"]      = g["news_count"].rolling(7,  min_periods=1).sum()
    out["news_rel_mean_7"]   = g["news_rel_mean"].rolling(7,  min_periods=1).mean()
    out.reset_index(inplace=True)
    return out
