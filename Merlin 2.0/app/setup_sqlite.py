
from models.stockmarket import init_db, db_session, Stock
init_db()
if db_session.query(Stock).count() == 0:
    db_session.add_all([
        Stock(symbol="VOD", name="Vodafone", enabled=True),
        Stock(symbol="BTC-USD", name="Bitcoin", enabled=True),
        Stock(symbol="ETH-USD", name="Ethereum", enabled=True),
        Stock(symbol="GBPUSD=X", name="GBP/USD", enabled=True),
        Stock(symbol="CL=F", name="WTI Crude Oil", enabled=True),
    ])
    db_session.commit()
    print("Seeded VOD, BTC-USD, ETH-USD, GBPUSD=X, CL=F")
else:
    print("Stock table already has data")
