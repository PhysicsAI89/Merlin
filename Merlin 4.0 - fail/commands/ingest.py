"""
Data ingest helpers (yfinance + Alpha Vantage fallback).
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional

from models.stockmarket import db_session, Stock, StockDaily

def ingest_yf(symbol: str, start: Optional[str]=None, end: Optional[str]=None) -> int:
    """
    Fetch OHLCV via yfinance for SYMBOL and write to StockDaily.
    Returns number of inserted rows.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end, auto_adjust=False)
    if hist.empty:
        return 0
    hist = hist.reset_index()  # 'Date', 'Open','High','Low','Close','Volume'
    rows = 0
    for _, r in hist.iterrows():
        dt = pd.to_datetime(r['Date']).to_pydatetime()
        if db_session.query(StockDaily).filter(StockDaily.symbol==symbol, StockDaily.price_date==dt).count():
            continue
        sd = StockDaily(
            stock_id=db_session.query(Stock.id).filter(Stock.symbol==symbol).scalar(),
            symbol=symbol,
            price_date=dt,
            open_price=float(r["Open"]),
            high_price=float(r["High"]),
            low_price=float(r["Low"]),
            close_price=float(r["Close"]),
            volume=float(r["Volume"] or 0)
        )
        db_session.add(sd); rows += 1
    if rows: db_session.commit()
    return rows
