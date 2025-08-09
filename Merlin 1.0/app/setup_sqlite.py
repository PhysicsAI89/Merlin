from models.stockmarket import init_db, db_session, Stock

# create SQLite tables
init_db()

# add one stock if table is empty
if db_session.query(Stock).count() == 0:
    db_session.add(Stock(symbol="VOD", name="Vodafone", enabled=True))
    db_session.commit()
    print("Inserted VOD")
else:
    print("Stock table already has data")
