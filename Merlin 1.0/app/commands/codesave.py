

        # pprint(stockdata["Meta Data"])
        # pprint(stockdata["Time Series (Daily)"])
        # pprint(stockdata["Time Series (Daily)"]["2022-01-10"])


        """
        last_key = list(stockdata["Time Series (Daily)"])[0]
        last_price = stockdata["Time Series (Daily)"][last_key]
        print(last_key)
        pprint(last_price)

        stock_daily = StockDaily(stock_id=stock.id, 
        	                     symbol=stock.symbol, 
        	                     price_date=last_key,
        	                     open_price=last_price["1. open"],
        	                     close_price=last_price["4. close"],
        	                     high_price=last_price["2. high"],
        	                     low_price=last_price["3. low"],
        	                     volume=last_price["5. volume"])
        db_session.add(stock_daily)
        """

    # db_session.commit()




    # Make DB connection
    engine = db.create_engine('mysql+pymysql://root:laraiders@localhost:3306/stockmarket')
    connection = engine.connect()
    print("connected")

    metadata = db.MetaData()

    stock = db.Table('stock', metadata, autoload=True, autoload_with=engine)

    query = db.select([stock.columns.id, stock.columns.symbol]).where(stock.columns.enabled==1)

    ResultProxy = connection.execute(query)
    ResultSet = ResultProxy.fetchall()


    for result in ResultSet:
        print(result)


    exit()



    apikey = "UNX587WU0XSI60ZB";
    apiurl = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s.L&apikey=%s" % (symbol, apikey)

    print('apiurl: ' , apiurl)

    contents = urllib.request.urlopen(apiurl).read()

    stockdata = json.loads(contents)

    pprint(stockdata["Meta Data"])
    pprint(stockdata["Time Series (Daily)"])
    pprint(stockdata["Time Series (Daily)"]["2022-01-10"])


    """
    if isinstance(contents, bytes):
        contents = str(contents, encoding='utf-8');


    # print(contents)
    # print(json.dumps(contents, indent=4))

    pprint(json.load(contents))
    """



@stocksbp.cli.command("get_stock_data2")
@click.argument("symbol")
def get_stock_data2(symbol):
    print('Symbol: ' , symbol)          