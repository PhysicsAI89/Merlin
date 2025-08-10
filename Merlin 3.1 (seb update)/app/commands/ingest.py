# app/commands/ingest.py
from __future__ import annotations

import click
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Blueprint

from models.stockmarket import db_session, Stock, StockDaily

ingestbp = Blueprint("ingest", __name__)


# ---------- helpers ----------

def upsert_stock(symbol: str, name: str | None = None) -> Stock:
    s = db_session.query(Stock).filter(Stock.symbol == symbol).first()
    if not s:
        s = Stock(symbol=symbol, name=name or symbol, enabled=True)
        db_session.add(s)
        db_session.commit()
    return s


def _normalize_download(sym: str, period: str, interval: str) -> tuple[str, pd.DataFrame]:
    """
    Download with yfinance and return a tidy DataFrame with columns:
    price_date, open_price, high_price, low_price, close_price, volume

    Handles both single-level and MultiIndex yfinance outputs.
    """
    df = yf.download(sym, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return sym, pd.DataFrame()

    # Make date a column first
    df = df.reset_index()

    # Flatten possible MultiIndex columns, e.g. ('Open','BTC-USD') -> 'Open_BTC-USD'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(p) for p in tup if p not in (None, "", " "))
            for tup in df.columns.to_list()
        ]

    # Choose a date column
    date_col = None
    for cand in ("Date", "Datetime", "date", "index"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        date_col = df.columns[0]

    out = pd.DataFrame()
    out["price_date"] = pd.to_datetime(df[date_col], errors="coerce")
    try:
        # make naive to avoid timezone issues with SQLite
        out["price_date"] = out["price_date"].dt.tz_localize(None)
    except Exception:
        pass

    # Helper to pick the first available column among candidates
    def pick(*names):
        for n in names:
            if n in df.columns:
                col = df[n]
                # In rare cases duplicates can return a DataFrame; take the first series
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                return col
        return pd.Series(np.nan, index=df.index)

    # Map to canonical names; include fallback to flattened names like 'Open_<sym>'
    out["open_price"]  = pd.to_numeric(pick("Open",  f"Open_{sym}"),   errors="coerce")
    out["high_price"]  = pd.to_numeric(pick("High",  f"High_{sym}"),   errors="coerce")
    out["low_price"]   = pd.to_numeric(pick("Low",   f"Low_{sym}"),    errors="coerce")
    out["close_price"] = pd.to_numeric(pick("Close", f"Close_{sym}"),  errors="coerce")
    out["volume"]      = pd.to_numeric(pick("Volume",f"Volume_{sym}"), errors="coerce")

    out = out.dropna(subset=["price_date"])

    return sym, out[["price_date", "open_price", "high_price", "low_price", "close_price", "volume"]]


# ---------- public API used by UI ----------

def ingest_yf(symbols: list[str], period: str = "5y", interval: str = "1d") -> dict[str, int]:
    """
    Ingest multiple symbols via yfinance into Stock/StockDaily.
    Returns {symbol: rows_inserted}
    """
    results: dict[str, int] = {}

    for sym in symbols:
        sym, tidy = _normalize_download(sym, period, interval)
        if tidy.empty:
            results[sym] = 0
            continue

        # Ensure a stock row exists
        s = upsert_stock(sym)

        rows = 0
        for _, r in tidy.iterrows():
            dt = pd.to_datetime(r["price_date"]).to_pydatetime()

            # Skip if already present
            exists = (
                db_session.query(StockDaily)
                .filter(StockDaily.symbol == sym, StockDaily.price_date == dt)
                .count()
                > 0
            )
            if exists:
                continue

            def _v(x):
                try:
                    if pd.isna(x):
                        return None
                except Exception:
                    pass
                return float(x) if x is not None else None

            rec = StockDaily(
                stock_id=s.id,
                symbol=sym,
                price_date=dt,
                open_price=_v(r.get("open_price")),
                high_price=_v(r.get("high_price")),
                low_price=_v(r.get("low_price")),
                close_price=_v(r.get("close_price")),
                volume=_v(r.get("volume")),
            )
            db_session.add(rec)
            rows += 1

        if rows:
            db_session.commit()
        results[sym] = rows

    return results


# ---------- optional CLI (still works) ----------

@ingestbp.cli.command("yf")
@click.option("--symbols", help="Comma-separated list like 'BTC-USD,ETH-USD,GBPUSD=X,CL=F'")
@click.option("--period", default="5y", help="yfinance period (e.g., 1y, 5y, max)")
@click.option("--interval", default="1d", help="Interval (1d, 1wk)")
def cli_yf(symbols: str, period: str, interval: str):
    if not symbols:
        symbols = "BTC-USD,ETH-USD,GBPUSD=X,CL=F"
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    res = ingest_yf(sym_list, period=period, interval=interval)
    print(res)
