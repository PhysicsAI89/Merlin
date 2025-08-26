from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey
from sqlalchemy.sql import func
from .stockmarket import Model

class ModelMetric(Model):
    __tablename__ = "model_metric"
    __table_args__ = {"sqlite_autoincrement": True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), index=True, nullable=False)
    model_name = Column(String(64), nullable=False)
    horizon = Column(Integer, nullable=False)
    freq = Column(String(2), nullable=False)
    window_start = Column(DateTime, nullable=False)
    window_end   = Column(DateTime, nullable=False)
    metric_name = Column(String(64), nullable=False)
    metric_value = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class TradePlan(Model):
    __tablename__ = "trade_plan"
    __table_args__ = {"sqlite_autoincrement": True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), index=True, nullable=False)
    entry_date = Column(DateTime, nullable=False)
    exit_date  = Column(DateTime, nullable=False)
    freq = Column(String(2), nullable=False)
    hold_steps = Column(Integer, nullable=False)
    threshold  = Column(Float, nullable=False)
    proba = Column(Float, nullable=False)
    action = Column(String(16), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class NewsEvent(Model):
    __tablename__ = "news_event"
    __table_args__ = {"sqlite_autoincrement": True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), index=True, nullable=False)
    source = Column(String(64), nullable=False)
    title = Column(String(2048), nullable=False)
    published_at = Column(DateTime, nullable=False)
    relevance = Column(Float, nullable=True)
    sentiment = Column(Float, nullable=True)
    url = Column(String(2048), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

class ModelSuggestion(Model):
    __tablename__ = "model_suggestion"
    __table_args__ = {"sqlite_autoincrement": True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), index=True, nullable=False)
    freq = Column(String(2), nullable=False)
    horizon = Column(Integer, nullable=False)
    suggestion = Column(String(4096), nullable=False)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class ForumBuzz(Model):
    __tablename__ = "forum_buzz"
    __table_args__ = {"sqlite_autoincrement": True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), index=True, nullable=False)
    window_days = Column(Integer, nullable=False)
    count = Column(Integer, nullable=False)
    baseline_avg = Column(Float, nullable=True)
    buzz_z = Column(Float, nullable=True)
    collected_at = Column(DateTime, server_default=func.now())

class StockPrediction(Model):
    __tablename__ = "stock_prediction"
    __table_args__ = {"sqlite_autoincrement": True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stock.id"), nullable=False)
    symbol = Column(String(32), index=True, nullable=False)
    price_date = Column(DateTime, nullable=False)
    target_date = Column(DateTime, nullable=False)
    pred_type = Column(String(32), nullable=False)
    pred_value = Column(Float, nullable=False)
    model_name = Column(String(64), nullable=False)
    model_version = Column(String(32), nullable=False, default="1")
    created_at = Column(DateTime, server_default=func.now())
