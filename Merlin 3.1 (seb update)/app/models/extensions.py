from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey
from sqlalchemy.sql import func
from models.stockmarket import Model

class NewsEvent(Model):
    __tablename__ = "news_event"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    published_at = Column(DateTime, nullable=False, index=True)
    title = Column(String(1024))
    source = Column(String(128))
    url = Column(String(1024))
    overall_sentiment_score = Column(Float)
    overall_sentiment_label = Column(String(32))
    relevance_score = Column(Float)

class ModelMetric(Model):
    __tablename__ = "model_metric"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    model_name = Column(String(64), nullable=False)
    horizon = Column(Integer, nullable=False, default=1)
    freq = Column(String(1), nullable=False, default="D")
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)
    metric_name = Column(String(32), nullable=False)  # e.g., "roc_auc", "accuracy"
    metric_value = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class TradePlan(Model):
    __tablename__ = "trade_plan"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())
    entry_date = Column(DateTime, nullable=False)
    exit_date = Column(DateTime, nullable=False)
    freq = Column(String(1), nullable=False, default="W")
    hold_steps = Column(Integer, nullable=False)
    threshold = Column(Float, nullable=False)
    proba = Column(Float, nullable=False)
    action = Column(String(8), nullable=False)  # BUY or HOLD