"""
Social/forum mention counter.
Default provider: snscrape for Reddit (no API keys required).
Optional: Stocktwits (if STOCKTWITS_TOKEN set).
"""
import os, subprocess, sys, datetime as dt
import pandas as pd

from models.stockmarket import db_session, SocialMention

def _ensure_snscrape():
    try:
        import snscrape  # noqa
        return True
    except Exception:
        # Lazy install if running from streamlit/CLI and pip available.
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "snscrape"])
            return True
        except Exception:
            return False

def fetch_reddit_mentions(symbol: str, days: int = 7) -> int:
    """
    Use snscrape to count Reddit mentions for SYMBOL over the last N days.
    Stores daily buckets in SocialMention.
    """
    ok = _ensure_snscrape()
    if not ok:
        raise RuntimeError("snscrape not available; install it to enable Reddit mentions.")

    import snscrape.modules.reddit as srr

    # Query for "$AAPL OR AAPL" etc.
    query = f'("$${symbol}" OR {symbol})'
    since = (dt.datetime.utcnow() - dt.timedelta(days=days)).date().isoformat()
    scraper = srr.RedditSearchScraper(f'{query} since:{since}')
    rows = []
    for i, post in enumerate(scraper.get_items()):
        created = pd.to_datetime(post.date).tz_convert("UTC").date()
        rows.append({"date": created, "author": getattr(post, "author", None)})
        if i > 5000:  # safety
            break

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    agg = df.groupby("date").agg(n=("author","count"), unique=("author","nunique")).reset_index()
    wrote = 0
    for _, r in agg.iterrows():
        d = pd.to_datetime(r["date"]).to_pydatetime()
        # Upsert-like: avoid duplicate row for same (symbol, date, source)
        exists = db_session.query(SocialMention).filter(
            SocialMention.symbol==symbol,
            SocialMention.source=="reddit",
            SocialMention.event_date==d
        ).count()
        if exists: continue
        sm = SocialMention(symbol=symbol, event_date=d, source="reddit",
                           mentions=int(r["n"]), unique_authors=int(r["unique"]))
        db_session.add(sm); wrote += 1
    if wrote: db_session.commit()
    return wrote


def build_daily_social_features(symbol: str) -> pd.DataFrame:
    """
    SocialMention -> daily features: mention_z (rolling 30d), mention_spike (factor vs 30d mean)
    """
    q = db_session.query(SocialMention).filter(SocialMention.symbol==symbol).order_by(SocialMention.event_date)
    rows = pd.read_sql(q.statement, db_session.bind)
    if rows.empty:
        return pd.DataFrame(columns=["date","mention_z","mention_spike"])
    rows["date"] = pd.to_datetime(rows["event_date"]).dt.normalize()
    rows = rows.groupby("date").agg(mentions=("mentions","sum")).reset_index()
    rows["ma30"] = rows["mentions"].rolling(30, min_periods=5).mean()
    rows["std30"] = rows["mentions"].rolling(30, min_periods=5).std()
    rows["mention_z"] = (rows["mentions"] - rows["ma30"]) / rows["std30"].replace(0, pd.NA)
    rows["mention_spike"] = rows["mentions"] / rows["ma30"]
    return rows[["date","mention_z","mention_spike"]]
