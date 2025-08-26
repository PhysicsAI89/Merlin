import os, requests

try:
    from models.stockmarket import db_session
    from models.extensions import ForumBuzz
except Exception:
    try:
        from app.models.stockmarket import db_session
        from app.models.extensions import ForumBuzz
    except Exception:
        from ..models.stockmarket import db_session
        from ..models.extensions import ForumBuzz

def has_key() -> bool:
    return bool(os.getenv("SERPAPI_API_KEY") or os.getenv("RAPIDAPI_KEY"))

def _search_serpapi(query: str, days: int = 1) -> int:
    key = os.getenv("SERPAPI_API_KEY")
    if not key: return 0
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": f"site:reddit.com OR site:stocktwits.com {query}",
        "tbs": f"qdr:{'d' if days<=1 else 'w'}",
        "num": 10,
        "api_key": key,
    }
    try:
        r = requests.get(url, params=params, timeout=15).json()
        info = r.get("search_information", {})
        total = info.get("total_results") or info.get("total_results_state")
        if isinstance(total, str):
            total = int(''.join(c for c in total if c.isdigit()) or 0)
        return int(total or 0)
    except Exception:
        return 0

def update_forum_buzz(symbol: str, window_days: int = 1) -> dict:
    count = _search_serpapi(symbol, days=window_days) if has_key() else 0
    q = db_session.query(ForumBuzz).filter(ForumBuzz.symbol==symbol).order_by(ForumBuzz.id.desc()).limit(30)
    prev = list(reversed(q.all()))
    baseline = sum([p.count for p in prev])/len(prev) if prev else 0.0
    sd = (sum((p.count-baseline)**2 for p in prev)/len(prev))**0.5 if prev else 1.0
    buzz_z = 0.0 if sd==0 else (count - baseline)/sd
    rec = ForumBuzz(symbol=symbol, window_days=window_days, count=int(count),
                    baseline_avg=float(baseline or 0.0), buzz_z=float(buzz_z))
    db_session.add(rec); db_session.commit()
    return {"symbol": symbol, "count": int(count), "baseline": float(baseline or 0.0), "buzz_z": float(buzz_z)}
