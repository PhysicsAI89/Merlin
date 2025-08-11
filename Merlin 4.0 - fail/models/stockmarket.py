from sqlalchemy import (
    Column, DateTime, Float, Integer, Boolean, String, ForeignKey, Index, create_engine, Text
)
from sqlalchemy.orm import scoped_session, sessionmaker, backref, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

# --- DB engine ---
# SQLite (dev-friendly, zero setup). Change this if/when you move to MySQL.
engine = create_engine('sqlite:///stockmarket.db', echo=False)

db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

Model = declarative_base(name='Model')
Model.query = db_session.query_property()


def init_db():
    """Create all tables if they don't exist."""
    Model.metadata.create_all(bind=engine)


# ----------------------------
# Core Tables
# ----------------------------
class Stock(Model):
    __tablename__ = 'stock'
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(15), unique=True, nullable=False, index=True)
    name = Column(String(512), nullable=False)
    enabled = Column(Boolean, nullable=True, default=True)
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
    symbol = Column(String(15), nullable=False, index=True)
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


# ----------------------------
# Signals / Predictions
# ----------------------------
class StockPrediction(Model):
    __tablename__ = "stock_prediction"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stock.id"), nullable=False)
    symbol = Column(String(15), nullable=False, index=True)
    price_date = Column(DateTime, nullable=False)   # date used for features
    target_date = Column(DateTime, nullable=False)  # predicted date (T+H)
    pred_type = Column(String(32), nullable=False)  # "direction" or "return_1d"
    pred_value = Column(Float, nullable=False)      # prob (direction) or return
    model_name = Column(String(64), nullable=False) # e.g., "GBClassifier_v1"
    model_version = Column(String(32), nullable=False, default="1")
    created_at = Column(DateTime, server_default=func.now())


class ModelMetric(Model):
    __tablename__ = "model_metric"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(15), index=True, nullable=False)
    model_name = Column(String(64), nullable=False)
    task = Column(String(32), nullable=False, default="classification")
    freq = Column(String(1), nullable=False, default="D")  # D or W
    horizon = Column(Integer, nullable=False, default=1)
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)
    metric_name = Column(String(64), nullable=False)       # e.g., "roc_auc", "mae"
    metric_value = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class ModelSuggestion(Model):
    __tablename__ = "model_suggestion"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(15), index=True, nullable=True)
    model_name = Column(String(64), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    severity = Column(String(16), nullable=False, default="info")  # info|warn|critical
    status = Column(String(16), nullable=False, default="new")     # new|applied|dismissed
    suggestion = Column(Text, nullable=False)
    evidence_json = Column(Text, nullable=True)


# ----------------------------
# News & Social Buzz
# ----------------------------
class NewsEvent(Model):
    __tablename__ = "news_event"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(15), index=True, nullable=False)
    event_time = Column(DateTime, index=True, nullable=False)
    source = Column(String(64), nullable=False)  # AV|NewsAPI|GDELT|Custom
    title = Column(String(1024), nullable=False)
    url = Column(String(1024), nullable=True)
    sentiment = Column(Float, nullable=True)     # as provided by API or local NLP
    relevance = Column(Float, nullable=True)     # optional
    tickers = Column(String(256), nullable=True) # pipe/comma separated tickers


class SocialMention(Model):
    __tablename__ = "social_mention"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(15), index=True, nullable=False)
    event_date = Column(DateTime, index=True, nullable=False)  # date bucket (UTC midnight)
    source = Column(String(32), nullable=False)  # reddit|stocktwits|twitter|forum
    mentions = Column(Integer, nullable=False, default=0)
    unique_authors = Column(Integer, nullable=True)

