# seed_db.py — initialize SQLite and insert default symbols
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from models.stockmarket import init_db, db_session, Stock

def main():
    init_db()
    if db_session.query(Stock).count() == 0:
        seeds = [
            ("BTC-USD","Bitcoin"),
            ("ETH-USD","Ethereum"),
            ("GBPUSD=X","GBPUSD"),
            ("CL=F","WTI Crude Oil Futures"),
            ("AAPL","Apple"),
        ]
        for sym, name in seeds:
            db_session.add(Stock(symbol=sym, name=name, enabled=True))
        db_session.commit()
        print("Inserted default symbols.")
    else:
        print("Stock table already has data.")

if __name__ == "__main__":
    main()
