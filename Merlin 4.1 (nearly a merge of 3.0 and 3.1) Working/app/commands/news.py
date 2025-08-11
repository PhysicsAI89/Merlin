import os, requests
import pandas as pd

try:
    from models.stockmarket import db_session
    from models.extensions import NewsEvent
except Exception:
    try:
        from app.models.stockmarket import db_session
        from app.models.extensions import NewsEvent
    except Exception:
        from ..models.stockmarket import db_session
        from ..models.extensions import NewsEvent

def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)

def ingest_news(symbol: str) -> int:
    apikey = _get("ALPHA_VANTAGE_API_KEY")
    if not apikey: return 0
    url = "https://www.alphavantage.co/query"
    params = {"function":"NEWS_SENTIMENT","tickers": symbol if symbol.isalpha() else "",
              "topics":"", "sort":"LATEST", "apikey": apikey}
    try:
        r = requests.get(url, params=params, timeout=15)
        j = r.json()
    except Exception:
        return 0
    feed = j.get("feed", []); n = 0
    for item in feed:
        try:
            ts = pd.to_datetime(item.get("time_published")).to_pydatetime()
        except Exception:
            continue
        db_session.add(NewsEvent(
            symbol=symbol, source="alphavantage",
            title=(item.get("title","")[:2048]),
            published_at=ts,
            relevance=float(item.get("relevance_score") or 0.0),
            sentiment=float(item.get("overall_sentiment_score") or 0.0),
            url=(item.get("url","")[:2048])
        )); n += 1
    if n: db_session.commit()
    return n

def build_news_features(symbol: str, start_date: str) -> pd.DataFrame:
    try:
        from models.extensions import NewsEvent as NE
    except Exception:
        try:
            from app.models.extensions import NewsEvent as NE
        except Exception:
            from ..models.extensions import NewsEvent as NE
    q = db_session.query(NE).filter(NE.symbol==symbol)
    try:
        start_ts = pd.to_datetime(start_date); q = q.filter(NE.published_at >= start_ts)
    except Exception:
        pass
    rows = q.all()
    if not rows:
        return pd.DataFrame(columns=[
            "price_date","news_sent_mean_3","news_sent_mean_7","news_count_3","news_count_7","news_rel_mean_7"
        ])
    df = pd.DataFrame([{
        "price_date": pd.to_datetime(r.published_at.date()),
        "sent": r.sentiment or 0.0,
        "rel": r.relevance or 0.0,
    } for r in rows])
    daily = df.groupby("price_date").agg(sent_mean=("sent","mean"),
                                         count=("sent","size"),
                                         rel_mean=("rel","mean")).reset_index()
    daily.sort_values("price_date", inplace=True)
    out = pd.DataFrame({"price_date": daily["price_date"]})
    out["news_sent_mean_3"] = daily["sent_mean"].rolling(3).mean().fillna(0.0)
    out["news_sent_mean_7"] = daily["sent_mean"].rolling(7).mean().fillna(0.0)
    out["news_count_3"] = daily["count"].rolling(3).sum().fillna(0.0)
    out["news_count_7"] = daily["count"].rolling(7).sum().fillna(0.0)
    out["news_rel_mean_7"] = daily["rel_mean"].rolling(7).mean().fillna(0.0)
    return out
