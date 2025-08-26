from sqlalchemy import (
    Column, DateTime, Float, Integer, Boolean, String, ForeignKey, Index, create_engine
)
from sqlalchemy.orm import scoped_session, sessionmaker, backref, relationship
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite:///stockmarket.db', echo=False)
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Model = declarative_base(name='Model')
Model.query = db_session.query_property()

def init_db():
    Model.metadata.create_all(bind=engine)

class Stock(Model):
    __tablename__ = 'stock'
    __table_args__ = {"sqlite_autoincrement": True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), unique=True, nullable=False, index=True)
    name = Column(String(512), nullable=False)
    enabled = Column(Boolean, nullable=True)
    advfn_url = Column(String(512), nullable=True)
    sector = Column(String(512), nullable=True)
    shares_in_issue = Column(Integer, nullable=True)
    def __repr__(self): return f"<Stock {self.symbol} - {self.name}>"

class StockDaily(Model):
    __tablename__ = 'stock_daily'
    __table_args__ = {"sqlite_autoincrement": True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stock.id'), nullable=False)
    symbol = Column(String(32), nullable=False, index=True)
    price_date = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    high_price = Column(Float, nullable=True)
    low_price = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    stock = relationship("Stock", backref=backref("daily", lazy="dynamic"))
Index("ix_stockdaily_symbol_date", StockDaily.symbol, StockDaily.price_date)
