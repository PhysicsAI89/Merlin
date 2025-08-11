"""
News fetchers and feature builders.
- Alpha Vantage NEWS (lightweight, rate-limited)
- Optional: NewsAPI/GDELT (plug-in ready)
"""
import os, json, urllib.request, urllib.parse, datetime as dt
import pandas as pd

from models.stockmarket import db_session, NewsEvent

AV_NEWS_ENDPOINT = "https://www.alphavantage.co/query"

def fetch_av_news(ticker: str, max_items: int = 50) -> int:
    """
    Fetch news for a ticker via Alpha Vantage and insert into NewsEvent.
    Requires env var ALPHA_VANTAGE_API_KEY.
    Returns number of inserted rows.
    """
    apikey = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not apikey:
        raise RuntimeError("Set ALPHA_VANTAGE_API_KEY env var to fetch AV news.")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": apikey,
        "sort": "LATEST",
    }
    url = AV_NEWS_ENDPOINT + "?" + urllib.parse.urlencode(params)
    data = json.loads(urllib.request.urlopen(url).read())

    feed = data.get("feed", [])[:max_items]
    wrote = 0
    for item in feed:
        ts = pd.to_datetime(item.get("time_published")).to_pydatetime() if item.get("time_published") else None
        title = item.get("title") or ""
        sentiment = None
        if "overall_sentiment_score" in item:
            try: sentiment = float(item["overall_sentiment_score"])
            except: sentiment = None

        # avoid duplicates by (symbol, title, ts)
        exists = (
            db_session.query(NewsEvent)
            .filter(NewsEvent.symbol == ticker, NewsEvent.title == title, NewsEvent.event_time == ts)
            .count()
        )
        if exists: continue

        ne = NewsEvent(
            symbol=ticker, event_time=ts or dt.datetime.utcnow(), source="AV",
            title=title[:1024], url=item.get("url","")[:1024],
            sentiment=sentiment, relevance=None,
            tickers=",".join(item.get("ticker_sentiment",[]) and [t['ticker'] for t in item['ticker_sentiment']][:6])
        )
        db_session.add(ne); wrote += 1
    if wrote: db_session.commit()
    return wrote


def build_daily_news_features(symbol: str) -> pd.DataFrame:
    """
    Aggregate NewsEvent to daily features for a symbol.
    Returns DataFrame with columns: ['date','news_sent_mean_3','news_sent_mean_7','news_count_3','news_count_7']
    """
    q = db_session.query(NewsEvent).filter(NewsEvent.symbol==symbol).order_by(NewsEvent.event_time)
    rows = pd.read_sql(q.statement, db_session.bind)
    if rows.empty:
        return pd.DataFrame(columns=["date","news_sent_mean_3","news_sent_mean_7","news_count_3","news_count_7"])

    rows["date"] = pd.to_datetime(rows["event_time"]).dt.normalize()
    agg = rows.groupby("date").agg(sent_mean=("sentiment","mean"), n=("id","count")).reset_index()
    for w in (3,7):
        agg[f"news_sent_mean_{w}"] = agg["sent_mean"].rolling(w, min_periods=1).mean()
        agg[f"news_count_{w}"] = agg["n"].rolling(w, min_periods=1).sum()
    return agg[["date","news_sent_mean_3","news_sent_mean_7","news_count_3","news_count_7"]]
