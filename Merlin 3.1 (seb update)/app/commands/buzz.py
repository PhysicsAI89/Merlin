# app/commands/buzz.py
import datetime as dt
import time
from typing import Dict, Tuple

import requests
from flask import Blueprint

from models.stockmarket import db_session, Stock, StockBuzzDaily

buzzbp = Blueprint("buzz", __name__)

HEADERS = {"User-Agent": "MerlinBuzz/0.1 (+https://github.com/PhysicsAI89/Merlin)"}

def _today_utc_date():
    return dt.datetime.utcnow().date()

def fetch_reddit_mentions(symbol: str, lookback_hours: int = 24) -> int:
    """
    Lightweight count of posts mentioning the symbol in last ~24h via public search.
    No auth, but rate-limited; keep modest usage in UI.
    """
    url = "https://www.reddit.com/search.json"
    params = {"q": symbol, "sort": "new", "t": "day", "limit": 100}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        posts = data.get("data", {}).get("children", [])
        return len(posts)
    except Exception:
        return 0

def fetch_stocktwits_mentions(symbol: str) -> int:
    """
    Public Stocktwits stream for a symbol. Returns recent message count.
    """
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 429:
            time.sleep(2)
            r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        msgs = data.get("messages", []) or []
        return len(msgs)
    except Exception:
        return 0

def fetch_forum_buzz(symbol: str) -> Dict[str, int]:
    """
    Return mention counts by source and total for today.
    """
    counts = {
        "reddit": fetch_reddit_mentions(symbol),
        "stocktwits": fetch_stocktwits_mentions(symbol),
    }
    counts["total"] = sum(counts.values())
    return counts

def upsert_buzz_count(symbol: str, source: str, buzz_date: dt.date, count: int) -> None:
    stk = db_session.query(Stock).filter(Stock.symbol == symbol).first()
    if not stk:
        return
    rec = (
        db_session.query(StockBuzzDaily)
        .filter(
            StockBuzzDaily.symbol == symbol,
            StockBuzzDaily.buzz_date == buzz_date,
            StockBuzzDaily.source == source,
        )
        .first()
    )
    now = dt.datetime.utcnow()
    if rec:
        rec.mention_count = count
        rec.created_at = now
    else:
        rec = StockBuzzDaily(
            stock_id=stk.id,
            symbol=symbol,
            buzz_date=buzz_date,
            source=source,
            mention_count=count,
            created_at=now,
        )
        db_session.add(rec)
    db_session.commit()

def get_buzz_baseline(symbol: str, days: int = 30) -> Tuple[int, float]:
    """
    Returns (today_total, baseline_avg_total).
    Baseline = average of daily totals over last `days` (excluding today if present).
    """
    end = _today_utc_date()
    start = end - dt.timedelta(days=days + 2)
    q = (
        db_session.query(StockBuzzDaily)
        .filter(StockBuzzDaily.symbol == symbol)
        .filter(StockBuzzDaily.buzz_date >= start)
        .filter(StockBuzzDaily.buzz_date <= end)
        .all()
    )
    if not q:
        return (0, 0.0)

    daily = {}
    for r in q:
        daily.setdefault(r.buzz_date, 0)
        daily[r.buzz_date] += (r.mention_count or 0)

    today_total = daily.get(end, 0)
    hist_days = [d for d in daily.keys() if d != end]
    if not hist_days:
        return (today_total, 0.0)

    baseline = sum(daily[d] for d in hist_days) / float(len(hist_days))
    return (today_total, float(baseline))
