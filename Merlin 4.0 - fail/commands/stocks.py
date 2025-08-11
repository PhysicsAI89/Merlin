import os
import time
import json
import click
import urllib.request
from datetime import datetime
from flask import Blueprint

from models.stockmarket import Stock, StockDaily, db_session

stocksbp = Blueprint("stocks", __name__)

# ---------------------------
# Helpers
# ---------------------------
def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _parse_date(date_str: str) -> datetime:
    # Alpha Vantage daily dates are 'YYYY-MM-DD'
    return datetime.strptime(date_str, "%Y-%m-%d")

def _av_symbol(sym: str) -> str:
    # Add LSE suffix if not already present
    return sym if sym.endswith(".L") else f"{sym}.L"

# ---------------------------
# CLI: fetch data
# ---------------------------
@stocksbp.cli.command("get_stock_data")
@click.argument("startid", type=int)
def get_stock_data(startid: int):
    """
    Fetch daily OHLCV from Alpha Vantage for enabled stocks with id >= startid.
    Respects AV rate limit (5 calls/min).
    """
    apikey = os.getenv("ALPHA_VANTAGE_API_KEY", "")

    starttime = datetime.now()
    calls_this_min = 0

    q = db_session.query(Stock).filter(Stock.enabled == True, Stock.id >= startid).order_by(Stock.id).limit(400)

    for stock in q:
        print(stock.symbol, stock.name)

        data = call_alphavantage(_av_symbol(stock.symbol), apikey)

        if not data or "Time Series (Daily)" not in data:
            msg = data.get("Note") or data.get("Error Message") if isinstance(data, dict) else None
            print(f"Missing Data For: {stock.symbol}" + (f" — {msg}" if msg else ""))
            if msg and "frequency" in msg.lower():
                time.sleep(65)
            continue

        wrote = write_price_data(stock.id, stock.symbol, data["Time Series (Daily)"])
        print(f"Inserted {wrote} rows for {stock.symbol}")

        # rate limit: 5 calls/min
        calls_this_min += 1
        elapsed = (datetime.now() - starttime).seconds
        if calls_this_min >= 5:
            if elapsed < 61:
                wait = 61 - elapsed
                print(f"Waiting {wait}s for rate limit...")
                time.sleep(wait)
            starttime = datetime.now()
            calls_this_min = 0


def call_alphavantage(symbol_with_suffix: str, apikey: str) -> dict:
    # outputsize=compact: last ~100 trading days
    apiurl = (
        "https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_DAILY&symbol={symbol_with_suffix}&outputsize=compact&apikey={apikey}"
    )
    contents = urllib.request.urlopen(apiurl).read()
    return json.loads(contents)


def write_price_data(stock_id: int, stock_symbol: str, daily_series: dict) -> int:
    """
    Insert daily bars for 'stock_symbol' from AV JSON 'Time Series (Daily)'.
    Converts date strings to datetime and OHLCV to floats.
    Commits once at the end for speed.
    """
    rows = 0
    for date_str, ohlcv in daily_series.items():
        dt = _parse_date(date_str)
        if dt.year < 2019:
            continue  # keep your 2019+ filter

        if price_day_exists(stock_symbol, dt):
            continue

        rec = StockDaily(
            stock_id=stock_id,
            symbol=stock_symbol,
            price_date=dt,
            open_price=_to_float(ohlcv.get("1. open")),
            high_price=_to_float(ohlcv.get("2. high")),
            low_price=_to_float(ohlcv.get("3. low")),
            close_price=_to_float(ohlcv.get("4. close")),
            volume=_to_float(ohlcv.get("5. volume")),
        )
        db_session.add(rec)
        rows += 1

    if rows:
        db_session.commit()
    return rows


def price_day_exists(stock_symbol: str, dt: datetime) -> bool:
    return db_session.query(StockDaily).filter(
        StockDaily.symbol == stock_symbol,
        StockDaily.price_date == dt
    ).count() > 0
