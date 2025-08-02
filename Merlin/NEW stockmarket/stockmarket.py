from flask import Flask, jsonify, request, render_template
from flask_sqlalchemy import SQLAlchemy
import click
import datetime
import json
import urllib.request
import time
from sqlalchemy.exc import IntegrityError

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:laraiders@localhost:3306/stockmarket'
db = SQLAlchemy(app)

# Models
class Stock(db.Model):
    __tablename__ = 'stock'
    id = db.Column(db.BigInteger, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    name = db.Column(db.String(512), nullable=False)
    enabled = db.Column(db.Boolean, default=True)
    advfn_url = db.Column(db.String(512))
    sector = db.Column(db.String(512), default='')
    shares_in_issue = db.Column(db.BigInteger, default=0)

class StockDaily(db.Model):
    __tablename__ = 'stock_daily'
    id = db.Column(db.BigInteger, primary_key=True)
    stock_id = db.Column(db.BigInteger, nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    price_date = db.Column(db.Date, nullable=False)
    open_price = db.Column(db.Float, default=0.0)
    close_price = db.Column(db.Float, default=0.0)
    high_price = db.Column(db.Float, default=0.0)
    low_price = db.Column(db.Float, default=0.0)
    volume = db.Column(db.Float, default=0.0)

@app.cli.command("test")
@click.argument("name")
def test(name):
    print(f'Name: {name}')

@app.cli.command("get_stock_data")
@click.argument("startid")
def get_stock_data(startid):
    apikey = "UNX587WU0XSI60ZB"
    starttime = datetime.datetime.now()
    stockcount = 0
    for stock in Stock.query.filter(Stock.enabled==True, Stock.id>=startid).order_by(Stock.id).limit(400):
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
                price_date=datetime.datetime.strptime(stockpriceday, '%Y-%m-%d').date(),
                open_price=stockdata["Time Series (Daily)"][stockpriceday]["1. open"],
                close_price=stockdata["Time Series (Daily)"][stockpriceday]["4. close"],
                high_price=stockdata["Time Series (Daily)"][stockpriceday]["2. high"],
                low_price=stockdata["Time Series (Daily)"][stockpriceday]["3. low"],
                volume=stockdata["Time Series (Daily)"][stockpriceday]["5. volume"]
            )
            db.session.add(stock_daily)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                print(f"Data for {stockpriceday} already exists.")

def price_day_exists(stock_symbol, stockpriceday):
    return db.session.query(StockDaily).filter_by(symbol=stock_symbol, price_date=stockpriceday).count() > 0

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/triple_value_stocks')
def triple_value_stocks():
    results = []
    stocks = Stock.query.filter(Stock.enabled == True).order_by(Stock.id).all()
    for stock in stocks:
        stockprices = StockDaily.query.filter_by(symbol=stock.symbol).all()
        min_price, max_price = get_min_max_prices(stockprices)
        if max_price > (3 * min_price):
            results.append({
                "symbol": stock.symbol,
                "min_price": min_price,
                "max_price": max_price
            })
    return jsonify(results)

@app.route('/triple_value_stocks_with_count')
def triple_value_stocks_with_count():
    results = []
    stocks = Stock.query.filter(Stock.enabled == True).order_by(Stock.id).all()
    for stock in stocks:
        stockprices = StockDaily.query.filter_by(symbol=stock.symbol).all()
        min_price, max_price = get_min_max_prices(stockprices)
        if max_price > (3 * min_price):
            n_min_count, n_max_count = get_min_max_count(stockprices, min_price, max_price)
            results.append({
                "symbol": stock.symbol,
                "min_price": min_price,
                "max_price": max_price,
                "multiple": (max_price / min_price),
                "n_min_count": n_min_count,
                "n_max_count": n_max_count
            })
    sorted_results = sorted(results, key=lambda d: d['multiple'])
    return jsonify(sorted_results)

@app.route('/oscillating_stocks')
def oscillating_stocks():
    results = []
    stocks = Stock.query.filter(Stock.enabled == True).order_by(Stock.id).all()
    for stock in stocks:
        stockprices = StockDaily.query.filter_by(symbol=stock.symbol).all()
        min_price, max_price = get_min_max_prices(stockprices)
        if max_price > (1.5 * min_price):
            oscillation_count = get_oscillation_count(stockprices, min_price, max_price)
            results.append({
                "symbol": stock.symbol,
                "min_price": min_price,
                "max_price": max_price,
                "oscillation_count": oscillation_count,
                "multiple": (max_price / min_price)
            })
    sorted_results = sorted(results, key=lambda d: d['oscillation_count'])
    return jsonify(sorted_results)

def get_min_max_prices(prices):
    min_price = float('inf')
    max_price = float('-inf')
    for price in prices:
        if price.low_price < min_price:
            min_price = price.low_price
        if price.high_price > max_price:
            max_price = price.high_price
    return min_price, max_price

def get_min_max_count(prices, min_price, max_price):
    min_plus_ten_percent  = 1.1 * min_price
    max_minus_ten_percent = 0.9 * max_price
    n_min_count = sum(price.low_price < min_plus_ten_percent for price in prices)
    n_max_count = sum(price.high_price > max_minus_ten_percent for price in prices)
    return n_min_count, n_max_count

def get_oscillation_count(prices, min_price, max_price):
    oscillation_count = 0
    sorted_prices = sorted(prices, key=lambda d: d.price_date)
    min_plus_ten_percent  = 1.1 * min_price
    max_minus_ten_percent = 0.9 * max_price
    last_found = "none"
    for price in sorted_prices:
        low_found = price.low_price < min_plus_ten_percent
        high_found = price.high_price > max_minus_ten_percent
        if low_found and last_found == "high":
            oscillation_count += 1
            last_found = "low"
        if high_found and last_found == "low":
            oscillation_count += 1
            last_found = "high"
    return oscillation_count

if __name__ == "__main__":
    app.run(debug=True)
