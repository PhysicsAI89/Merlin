import click, datetime, json, time
import urllib.request
from flask import Blueprint
from pprint import pprint

import sqlalchemy as db
from sqlalchemy.orm import scoped_session, sessionmaker

from models.stockmarket import Stock, StockDaily, db_session


analysebp = Blueprint('analyse', __name__)


@analysebp.cli.command("get_triple_value_stocks")
def get_triple_value_stocks():

    for stock in db_session.query(Stock).filter(Stock.enabled==True).order_by(Stock.id):

        # get last 12 months price data for this stock
        stockprices = get_stock_prices(stock.symbol)

        # get min and max price for this stock
        min_price, max_price = get_min_max_prices(stockprices)

        # is max greater than 3 times min price
        if max_price > (3 * min_price):
            print("Stock: %s Min: %s Max: %s" % (stock.symbol, min_price, max_price))


@analysebp.cli.command("get_triple_value_stocks_with_count")
def get_triple_value_stocks_with_count():

    triple_value_stocks = []
    for stock in db_session.query(Stock).filter(Stock.enabled==True).order_by(Stock.id):

        # get last 12 months price data for this stock
        stockprices = get_stock_prices(stock.symbol)

        # get min and max price for this stock
        min_price, max_price = get_min_max_prices(stockprices)

        # is max greater than 3 times min price
        if max_price > (3 * min_price):

            n_min_count, n_max_count = get_min_max_count(stockprices, min_price, max_price)

            triple_value_stocks.append({
                "symbol": stock.symbol,
                "min_price": min_price, 
                "max_price": max_price,
                "multiple": (max_price / min_price),
                # "stockprices": stockprices,
                "n_min_count": n_min_count,
                "n_max_count": n_max_count
                })

    sorted_triple_value_stocks = sorted(triple_value_stocks, key=lambda d: d['multiple']) 

    for triple_stock in sorted_triple_value_stocks:
        print(triple_stock)
        # print("Stock: %s Min: %s Max: %s Mulitple: %s Nmin: %s Nmax: %s" % (triple_stock.symbol, min_price, max_price))


@analysebp.cli.command("get_oscillating_stocks")
def get_oscillating_stocks():

    oscillating_stocks = []
    for stock in db_session.query(Stock).filter(Stock.enabled==True).order_by(Stock.id):

        # get last 12 months price data for this stock
        stockprices = get_stock_prices(stock.symbol)

        # get min and max price for this stock
        min_price, max_price = get_min_max_prices(stockprices)

        # is max greater than 2 times min price
        if max_price > (1.5 * min_price):

            oscillation_count = get_oscillation_count(stockprices, min_price, max_price)

            oscillating_stocks.append({
                "symbol": stock.symbol,
                "min_price": min_price, 
                "max_price": max_price,
                "oscillation_count": oscillation_count,
                "multiple": (max_price / min_price),
                })

    sorted_oscillating_stocks = sorted(oscillating_stocks, key=lambda d: d['oscillation_count']) 

    for sorted_oscillating_stock in sorted_oscillating_stocks:
        print(sorted_oscillating_stock)


"""
Utility functions
"""


def get_stock_prices(symbol):

    date_1_year_ago = datetime.datetime.now() - datetime.timedelta(365)
    begin_date = date_1_year_ago.strftime('%Y-%m-%d')

    prices = db_session.query(StockDaily).filter(StockDaily.symbol==symbol, StockDaily.price_date>begin_date)

    return prices


def get_min_max_prices(prices):

    min_price = 999999999.0
    max_price = 0.0
    for price in prices:
        if price.low_price < min_price:
            min_price = price.low_price
        if price.high_price > max_price:
            max_price = price.high_price

    return min_price, max_price
    

def get_min_max_count(prices, min_price, max_price):

    min_plus_ten_percent  = 1.1 * min_price
    max_minus_ten_percent = 0.9 * max_price

    n_min_count = 0
    n_max_count = 0

    for price in prices:
        if price.low_price < min_plus_ten_percent:
            n_min_count += 1
        if price.high_price > max_minus_ten_percent:
            n_max_count += 1

    return n_min_count, n_max_count


def get_oscillation_count(prices, min_price, max_price):
    oscillation_count = 0
    sorted_prices = sorted(prices, key=lambda d: d.price_date) 

    min_plus_ten_percent  = 1.1 * min_price
    max_minus_ten_percent = 0.9 * max_price

    last_found = "none"
    for price in sorted_prices:

        low_found = False
        high_found = False

        if price.low_price < min_plus_ten_percent:
            low_found = True
            if last_found == "none":
                last_found = "low"      

        if price.high_price > max_minus_ten_percent:
            high_found = True
            if last_found == "none":
                last_found = "high"

        if low_found and last_found == "high":
            oscillation_count += 1
            last_found = "low"

        if high_found and last_found == "low":
            oscillation_count += 1
            last_found = "high"

    return oscillation_count
