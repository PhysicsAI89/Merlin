# app/models/stockmarket.py
from sqlalchemy import (
    Column, DateTime, Float, Integer, Boolean, String, ForeignKey, Index, create_engine, Date
)
from sqlalchemy.orm import scoped_session, sessionmaker, backref, relationship
from sqlalchemy.ext.declarative import declarative_base

# --- DB engine ---
# MySQL example (when ready):
# engine = create_engine('mysql+pymysql://user:pass@host:3306/stockmarket', echo=False)

# SQLite (dev-friendly)
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

    stock = relationship("Stock", backref=backref("daily", lazy="dynamic"))

# Optional composite index to speed up lookups
Index("ix_stockdaily_symbol_date", StockDaily.symbol, StockDaily.price_date)

# --- NEW: daily forum/news buzz counts by source (Reddit, Stocktwits, etc.) ---
class StockBuzzDaily(Model):
    __tablename__ = 'stock_buzz_daily'
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stock.id'), nullable=False)
    symbol = Column(String(10), nullable=False, index=True)
    buzz_date = Column(Date, nullable=False, index=True)
    source = Column(String(32), nullable=False, index=True)  # 'reddit' | 'stocktwits' | ...
    mention_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime)

    stock = relationship("Stock", backref=backref("buzz", lazy="dynamic"))

Index("ix_buzz_symbol_date_src", StockBuzzDaily.symbol, StockBuzzDaily.buzz_date, StockBuzzDaily.source)
