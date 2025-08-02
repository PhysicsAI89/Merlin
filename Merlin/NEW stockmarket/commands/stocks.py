import click
from flask import Blueprint
import urllib.request
import json
from .models import Stock, StockDaily, db_session

stocksbp = Blueprint('stocks', __name__)

@stocksbp.cli.command("get_stock_data")
@click.argument("startid")
def get_stock_data(startid):
    apikey = "UNX587WU0XSI60ZB"
    starttime = datetime.datetime.now()
    stockcount = 0

    for stock in db_session.query(Stock).filter(Stock.enabled==True, Stock.id>=startid).order_by(Stock.id).limit(400):
        print(stock.symbol, stock.name)
        stockdata = call_alphavantage(stock.symbol)
        stockcount += 1
        write_price_data(stock.id, stock.symbol, stockdata)

        elapsedtime = datetime.datetime.now() - starttime
        if stockcount >= 5 and elapsedtime.seconds < 61:
            waitseconds = 61 - elapsedtime.seconds
            print(f"Waiting for {waitseconds} seconds")
            time.sleep(waitseconds)
            starttime = datetime.datetime.now()
            stockcount = 0

        if stockcount >= 5 and elapsedtime.seconds >= 61:
            starttime = datetime.datetime.now()
            stockcount = 1

def call_alphavantage(symbol):
    apikey = "UNX587WU0XSI60ZB"
    apiurl = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}.L&outputsize=compact&apikey={apikey}"
    contents = urllib.request.urlopen(apiurl).read()
    return json.loads(contents)

def write_price_data(stock_id, stock_symbol, stockdata):
    if "Time Series (Daily)" not in stockdata:
        print(f"Missing Data For: {stock_symbol}")
        return

    for stockpriceday in stockdata["Time Series (Daily)"]:
        if int(stockpriceday[0:4]) < 2019:
            break

        if not price_day_exists(stock_symbol, stockpriceday):
            print(f"Adding: {stockpriceday} {stock_symbol}")
            stock_daily = StockDaily(
                stock_id=stock_id,
                symbol=stock_symbol,
                price_date=stockpriceday,
                open_price=stockdata["Time Series (Daily)"][stockpriceday]["1. open"],
                close_price=stockdata["Time Series (Daily)"][stockpriceday]["4. close"],
                high_price=stockdata["Time Series (Daily)"][stockpriceday]["2. high"],
                low_price=stockdata["Time Series (Daily)"][stockpriceday]["3. low"],
                volume=stockdata["Time Series (Daily)"][stockpriceday]["5. volume"]
            )
            db_session.add(stock_daily)
            db_session.commit()

def price_day_exists(stock_symbol, stockpriceday):
    return db_session.query(StockDaily).filter(StockDaily.symbol==stock_symbol, StockDaily.price_date==stockpriceday).count() > 0
