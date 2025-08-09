from sqlalchemy import (
    Column, DateTime, Float, Integer, Boolean, String, ForeignKey, Index, create_engine
)
from sqlalchemy.orm import scoped_session, sessionmaker, backref, relationship
from sqlalchemy.ext.declarative import declarative_base

# --- DB engine ---
# MySQL (when you're ready to switch back):
# engine = create_engine('mysql+pymysql://root:laraiders@localhost:3306/stockmarket', echo=False)

# SQLite (dev-friendly, zero setup)
engine = create_engine('sqlite:///stockmarket.db', echo=False)

db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

Model = declarative_base(name='Model')
Model.query = db_session.query_property()


def init_db():
    """Create all tables if they don't exist."""
    Model.metadata.create_all(bind=engine)


# ----------------------------
# Tables
# ----------------------------
class Stock(Model):
    __tablename__ = 'stock'
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(512), nullable=False)
    enabled = Column(Boolean, nullable=True)
    advfn_url = Column(String(512), nullable=True)
    sector = Column(String(512), nullable=True)
    shares_in_issue = Column(Integer, nullable=True)

    def __repr__(self):
        return f"<Stock {self.symbol} - {self.name}>"


class StockDaily(Model):
    __tablename__ = 'stock_daily'
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stock.id'), nullable=False)
    symbol = Column(String(10), nullable=False, index=True)
    price_date = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    high_price = Column(Float, nullable=True)
    low_price = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)

    # handy relationship back to Stock
    stock = relationship("Stock", backref=backref("daily", lazy="dynamic"))

# Optional composite index to speed up lookups
Index("ix_stockdaily_symbol_date", StockDaily.symbol, StockDaily.price_date)











'''
from sqlalchemy import Column, DateTime, BigInteger, Float, Integer, Index, Boolean, ForeignKey, String
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker, backref, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

#engine = create_engine('mysql+pymysql://root:laraiders@localhost:3306/stockmarket', echo=False)

engine = create_engine('sqlite:///stockmarket.db', echo=False)
db_session = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine))

def init_db():
    Model.metadata.create_all(bind=engine)


Model = declarative_base(name='Model')
Model.query = db_session.query_property()


class Stock(Model):

    __tablename__ = 'stock'
    __table_args__ = {"sqlite_autoincrement": True}

    #id = Column(BigInteger, primary_key=True)
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), unique=True, nullable=False)
    name = Column(String(512), unique=False, nullable=False)
    enabled = Column(Boolean, nullable=True)
    advfn_url = Column(String(512), unique=False, nullable=True)
    sector = Column(String(512), unique=False, nullable=True)
    shares_in_issue = Column(BigInteger, unique=False, nullable=True)

    def __repr__(self):
        return '<Stock %r>' % self.name


class StockDaily(Model):

    __tablename__ = 'stock_daily'
    __table_args__ = {"sqlite_autoincrement": True}

    #id = Column(BigInteger, primary_key=True)
    #stock_id = Column(BigInteger, unique=False, nullable=False)
    id = Column(Integer, primary_key=True, autoincrement=True)
stock_id = Column(Integer, nullable=False)  
    symbol = Column(String(10), unique=False, nullable=False)
    price_date = Column(DateTime, unique=False, nullable=False)
    open_price = Column(Float, unique=False, nullable=True)
    close_price = Column(Float, unique=False, nullable=True)
    high_price = Column(Float, unique=False, nullable=True)
    low_price = Column(Float, unique=False, nullable=True)
    volume = Column(Float, unique=False, nullable=True)
'''