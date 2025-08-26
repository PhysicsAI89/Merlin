#!/usr/bin/env python3
"""
ingest.py — simple yfinance → SQLite ingestor for Merlin

What it does
------------
• Adds a Stock row if needed (symbol, name, enabled=True).
• Downloads daily OHLCV via yfinance.
• Upserts into StockDaily (skips existing rows for the same (symbol, price_date)).

Where to put it
---------------
Option A: project root (run:  python ingest.py yf AAPL --start 2019-01-01)
Option B: `commands/ingest.py` (then import as: from commands.ingest import ingest_yf)

Imports
-------
It tries: `from models.stockmarket import ...`
Falls back to: `from stockmarket import ...`
so it works with either layout.
"""
from __future__ import annotations

import sys
from datetime import datetime
from typing import Optional, Iterable

import click
import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Please install yfinance:  pip install yfinance") from e

# --- Models / DB session import (supports two common layouts) ---
try:
    # e.g. app/models/stockmarket.py
    from models.stockmarket import db_session, Stock, StockDaily  # type: ignore
except Exception:
    # e.g. stockmarket.py at project root
    from stockmarket import db_session, Stock, StockDaily  # type: ignore


# -----------------------------
# Utilities
# -----------------------------
def ensure_stock(symbol: str, name: Optional[str] = None, enabled: bool = True) -> Stock:
    """
    Make sure a row exists in `stock` for this symbol. Return the Stock ORM object.
    """
    s = db_session.query(Stock).filter(Stock.symbol == symbol).first()
    if s is None:
        s = Stock(symbol=symbol, name=name or symbol, enabled=enabled)
        db_session.add(s)
        db_session.commit()
    return s


def ingest_yf(symbol: str, start: str = "2019-01-01") -> int:
    """
    Fetch daily bars via yfinance and insert into StockDaily.
    Returns the number of new rows inserted (existing dates are skipped).

    Example
    -------
    >>> ingest_yf("AAPL", start="2019-01-01")
    1200
    """
    symbol = symbol.strip().upper()
    tk = yf.Ticker(symbol)
    hist = tk.history(start=start, interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"No data from yfinance for {symbol} (check the symbol or start date).")

    hist = hist.reset_index().rename(columns={
        "Date": "price_date", "Open": "open_price", "High": "high_price",
        "Low": "low_price", "Close": "close_price", "Volume": "volume"
    })

    # ensure stock exists
    stk = ensure_stock(symbol, name=symbol, enabled=True)

    rows = 0
    for _, r in hist.iterrows():
        dt = pd.to_datetime(r["price_date"]).to_pydatetime()
        # skip if this (symbol, price_date) already exists
        exists = db_session.query(StockDaily).filter(
            StockDaily.symbol == symbol,
            StockDaily.price_date == dt
        ).count() > 0
        if exists:
            continue

        rec = StockDaily(
            stock_id=stk.id,
            symbol=symbol,
            price_date=dt,
            open_price=float(r["open_price"]) if pd.notna(r["open_price"]) else None,
            close_price=float(r["close_price"]) if pd.notna(r["close_price"]) else None,
            high_price=float(r["high_price"]) if pd.notna(r["high_price"]) else None,
            low_price=float(r["low_price"]) if pd.notna(r["low_price"]) else None,
            volume=float(r["volume"]) if pd.notna(r["volume"]) else None,
        )
        db_session.add(rec)
        rows += 1

    if rows:
        db_session.commit()
    return rows


# -----------------------------
# CLI (Command-Line Interface)
# -----------------------------
@click.group(help="Ingest price data into Merlin's SQLite DB using yfinance.")
def cli() -> None:
    pass


@cli.command("yf")
@click.argument("symbol")
@click.option("--start", default="2019-01-01", show_default=True,
              help="Download history starting from this date (YYYY-MM-DD).")
def cli_yf(symbol: str, start: str) -> None:
    """
    Ingest one symbol.
    Usage: python ingest.py yf AAPL --start 2019-01-01
    """
    try:
        rows = ingest_yf(symbol, start=start)
        click.echo(f"Inserted {rows} new rows for {symbol}.")
    except Exception as e:
        click.echo(f"[ERROR] {e}", err=True)
        sys.exit(1)


@cli.command("yf-batch")
@click.option("--symbols", help="Comma-separated symbols, e.g. AAPL,MSFT,TSLA")
@click.option("--file", "file_path", type=click.Path(exists=True), help="Path to a text file with one symbol per line.")
@click.option("--start", default="2019-01-01", show_default=True)
def cli_yf_batch(symbols: Optional[str], file_path: Optional[str], start: str) -> None:
    """
    Ingest multiple symbols.
    Usage: python ingest.py yf-batch --symbols AAPL,MSFT,TSLA --start 2019-01-01
           python ingest.py yf-batch --file symbols.txt --start 2019-01-01
    """
    universe: Iterable[str] = []
    if symbols:
        universe = [s.strip() for s in symbols.split(",") if s.strip()]
    elif file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            universe = [ln.strip() for ln in f if ln.strip()]
    else:
        click.echo("Provide --symbols or --file.", err=True)
        sys.exit(2)

    total = 0
    for sym in universe:
        try:
            rows = ingest_yf(sym, start=start)
            click.echo(f"{sym}: +{rows}")
            total += rows
        except Exception as e:
            click.echo(f"{sym}: ERROR {e}", err=True)
    click.echo(f"Done. Total rows inserted: {total}")


if __name__ == "__main__":
    cli()
