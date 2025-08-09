
import os
import time
import json
import click
import urllib.request
from datetime import datetime
from flask import Blueprint
from pprint import pprint

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
    apikey = os.getenv("ALPHA_VANTAGE_API_KEY", "UNX587WU0XSI60ZB")  # <-- uses your env var if set

    starttime = datetime.now()
    calls_this_min = 0

    q = db_session.query(Stock).filter(Stock.enabled == True, Stock.id >= startid).order_by(Stock.id).limit(400)

    for stock in q:
        print(stock.symbol, stock.name)

        data = call_alphavantage(_av_symbol(stock.symbol), apikey)

        if not data or "Time Series (Daily)" not in data:
            # Common reasons: API key missing, symbol invalid, or rate limit
            msg = data.get("Note") or data.get("Error Message") if isinstance(data, dict) else None
            print(f"Missing Data For: {stock.symbol}" + (f" — {msg}" if msg else ""))
            # wait a bit if rate limited
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
            price_date=dt,                       # <-- datetime object, not string
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




'''import click, datetime, json, time
import urllib.request
from flask import Blueprint
from pprint import pprint

import sqlalchemy as db
from sqlalchemy.orm import scoped_session, sessionmaker

from models.stockmarket import Stock, StockDaily, db_session


stocksbp = Blueprint('stocks', __name__)


@stocksbp.cli.command("get_stock_data")
@click.argument("startid")
def get_stock_data(startid):

    # 5 API requests per minute and 500 requests per day
    apikey = "UNX587WU0XSI60ZB";


    starttime = datetime.datetime.now()
    stockcount = 0
    for stock in db_session.query(Stock).filter(Stock.enabled==True, Stock.id>=startid).order_by(Stock.id).limit(400):

        print(stock.symbol, stock.name)

        # get stockdata from AlphaVantage
        stockdata = call_alphavantage(stock.symbol)
        stockcount += 1

        # write price data to DB
        write_price_data(stock.id, stock.symbol, stockdata)

        # make sure we don't make more 5 calls per minute
        elapsedtime = datetime.datetime.now() - starttime
        if stockcount >= 5 and elapsedtime.seconds < 61:
        	waitseconds = 61 - elapsedtime.seconds
        	print("Waiting for %s seconds" % waitseconds)
        	time.sleep(waitseconds)
        	starttime = datetime.datetime.now()
        	stockcount = 0

        if stockcount >= 5 and elapsedtime.seconds >= 61:
        	starttime = datetime.datetime.now()
        	stockcount = 1


def call_alphavantage(symbol):

    # outputsize=compact     last 100 days
    # outputsize=full        20+ years

    apikey = "UNX587WU0XSI60ZB";
    apiurl = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s.L&outputsize=compact&apikey=%s" % (symbol, apikey)
    contents = urllib.request.urlopen(apiurl).read()
    return json.loads(contents)


def write_price_data(stock_id, stock_symbol, stockdata):

    # pprint(list(stockdata["Time Series (Daily)"]))

    # price_daily = json.loads(stockdata["Time Series (Daily)"])
    # pprint(price_daily)

    if "Time Series (Daily)" not in stockdata:
    	print("Missing Data For: %s" % stock_symbol)
    	return

    for stockpriceday in stockdata["Time Series (Daily)"]:

        # pprint(stockpriceday)
        # pprint(stockdata["Time Series (Daily)"][stockpriceday])

        if int(stockpriceday[0:4]) < 2019:
        	break

        if not price_day_exists(stock_symbol, stockpriceday):
            print(("Adding: %s %s") % (stockpriceday, stock_symbol))
            stock_daily = StockDaily(stock_id=stock_id, 
	                                 symbol=stock_symbol, 
	                                 price_date=stockpriceday,
	                                 open_price=stockdata["Time Series (Daily)"][stockpriceday]["1. open"],
	                                 close_price=stockdata["Time Series (Daily)"][stockpriceday]["4. close"],
	                                 high_price=stockdata["Time Series (Daily)"][stockpriceday]["2. high"],
                                     low_price=stockdata["Time Series (Daily)"][stockpriceday]["3. low"],
	                                 volume=stockdata["Time Series (Daily)"][stockpriceday]["5. volume"])
            db_session.add(stock_daily)
            db_session.commit()


def price_day_exists(stock_symbol, stockpriceday):
    price_exists = False

    countrows = db_session.query(StockDaily).filter(StockDaily.symbol==stock_symbol, StockDaily.price_date==stockpriceday).count()
    if countrows > 0:
        price_exists = True

    return price_exists


'''