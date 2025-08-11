from models.stockmarket import init_db, db_session, Stock
init_db()
if db_session.query(Stock).count() == 0:
    seeds = [("BTC-USD","Bitcoin"),("ETH-USD","Ethereum"),("GBPUSD=X","GBPUSD"),
             ("CL=F","WTI Crude Oil Futures"),("AAPL","Apple")]
    for sym,name in seeds:
        db_session.add(Stock(symbol=sym, name=name, enabled=True))
    db_session.commit()
    print("Inserted default symbols.")
else:
    print("Stock table already has data")
