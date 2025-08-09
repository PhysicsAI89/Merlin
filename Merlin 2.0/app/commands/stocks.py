
import os, time, json, urllib.request
from datetime import datetime
import click
from flask import Blueprint
from models.stockmarket import Stock, StockDaily, db_session

stocksbp = Blueprint("stocks", __name__)

def _to_float(x):
    try: return float(x)
    except Exception: return None

def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")

def _av_symbol(sym: str) -> str:
    return sym if sym.endswith(".L") else f"{sym}.L"

@stocksbp.cli.command("get_stock_data")
@click.argument("startid", type=int)
def get_stock_data(startid: int):
    apikey = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not apikey:
        print("[WARN] Missing ALPHA_VANTAGE_API_KEY env var.")
    starttime = datetime.now()
    calls = 0
    q = db_session.query(Stock).filter(Stock.enabled==True, Stock.id>=startid).order_by(Stock.id).limit(400)
    for s in q:
        sym = _av_symbol(s.symbol)
        data = call_alphavantage(sym, apikey)
        if not data or "Time Series (Daily)" not in data:
            msg = data.get("Note") or data.get("Error Message") if isinstance(data, dict) else None
            print(f"Missing Data For: {s.symbol}" + (f" — {msg}" if msg else ""))
            if msg and "frequency" in msg.lower():
                time.sleep(65)
            continue
        wrote = write_price_data(s.id, s.symbol, data["Time Series (Daily)"])
        print(f"Inserted {wrote} rows for {s.symbol}")
        calls += 1
        elapsed = (datetime.now() - starttime).seconds
        if calls >= 5:
            if elapsed < 61:
                time.sleep(61-elapsed)
            starttime = datetime.now(); calls = 0

def call_alphavantage(symbol_with_suffix: str, apikey: str) -> dict:
    url = (
        "https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_DAILY&symbol={symbol_with_suffix}&outputsize=compact&apikey={apikey}"
    )
    txt = urllib.request.urlopen(url).read()
    import json; return json.loads(txt)

def write_price_data(stock_id: int, stock_symbol: str, daily_series: dict) -> int:
    from models.stockmarket import StockDaily, db_session
    rows = 0
    for date_str, ohlcv in daily_series.items():
        dt = _parse_date(date_str)
        if dt.year < 2019: continue
        exists = db_session.query(StockDaily).filter(
            StockDaily.symbol==stock_symbol, StockDaily.price_date==dt
        ).count() > 0
        if exists: continue
        rec = StockDaily(
            stock_id=stock_id, symbol=stock_symbol, price_date=dt,
            open_price=_to_float(ohlcv.get("1. open")),
            high_price=_to_float(ohlcv.get("2. high")),
            low_price=_to_float(ohlcv.get("3. low")),
            close_price=_to_float(ohlcv.get("4. close")),
            volume=_to_float(ohlcv.get("5. volume")),
        )
        db_session.add(rec); rows += 1
    if rows: db_session.commit()
    return rows
