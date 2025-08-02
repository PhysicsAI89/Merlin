import click, datetime, json, time
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


