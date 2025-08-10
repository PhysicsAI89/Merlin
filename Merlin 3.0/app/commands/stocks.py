# app/commands/stocks.py
import os, time, json, urllib.request
from datetime import datetime
import click
from flask import Blueprint
from models.stockmarket import Stock, StockDaily, db_session

import pandas as pd
try:
    import yfinance as yf
except Exception:
    yf = None

stocksbp = Blueprint("stocks", __name__)

# ---------- helpers ----------
def _to_float(x):
    try: return float(x)
    except Exception: return None

def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")

def _av_symbol(sym: str) -> str:
    """Leave as-is (AAPL, VOD.L, BTC-USD, GBPUSD=X, CL=F)."""
    return sym

def _asset_type(symbol: str) -> str:
    if symbol.endswith("=F"): return "futures"
    if "-" in symbol:        return "crypto"
    if symbol.endswith("=X"):return "fx"
    return "equity"

# ---------- normalizers ----------
def _normalize_av_equity_fx_crypto(raw: dict, symbol: str) -> dict | None:
    if not isinstance(raw, dict): return None

    if "Time Series (Daily)" in raw:
        return raw["Time Series (Daily)"]

    if "Time Series FX (Daily)" in raw:
        ser = raw["Time Series FX (Daily)"]
        return {d: {
            "1. open": r.get("1. open"),
            "2. high": r.get("2. high"),
            "3. low":  r.get("3. low"),
            "4. close":r.get("4. close"),
            "5. volume": "0",
        } for d, r in ser.items()}

    if "Time Series (Digital Currency Daily)" in raw:
        ser = raw["Time Series (Digital Currency Daily)"]
        base, quote = (symbol.split("-", 1) + ["USD"])[:2]
        quote = quote.upper()
        ok=f"1a. open ({quote})"; hk=f"2a. high ({quote})"; lk=f"3a. low ({quote})"; ck=f"4a. close ({quote})"
        return {d: {
            "1. open": r.get(ok) or r.get("1b. open (USD)")  or r.get("1a. open (USD)"),
            "2. high": r.get(hk) or r.get("2b. high (USD)")  or r.get("2a. high (USD)"),
            "3. low":  r.get(lk) or r.get("3b. low (USD)")   or r.get("3a. low (USD)"),
            "4. close":r.get(ck) or r.get("4b. close (USD)") or r.get("4a. close (USD)"),
            "5. volume": r.get("5. volume", "0"),
        } for d, r in ser.items()}

    return None

def _normalize_yf_daily(symbol: str) -> dict | None:
    if yf is None:
        return None
    try:
        tkr = yf.Ticker(symbol)
        df = tkr.history(period="max", interval="1d", auto_adjust=False)
        if df is None or df.empty:
            return None
        out = {}
        for ts, row in df.iterrows():
            ts = pd.Timestamp(ts)
            if ts.tzinfo is not None:
                ts = ts.tz_convert(None)
            ds = ts.strftime("%Y-%m-%d")
            out[ds] = {
                "1. open": str(row.get("Open")),
                "2. high": str(row.get("High")),
                "3. low":  str(row.get("Low")),
                "4. close":str(row.get("Close")),
                "5. volume": str(0 if pd.isna(row.get("Volume")) else row.get("Volume")),
            }
        return out
    except Exception:
        return None

# ---------- fetchers ----------
def _fetch(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())

def call_alphavantage(symbol: str, apikey: str) -> dict:
    """Return {'Time Series (Daily)': dict} on success; else include Note/Error."""
    base = "https://www.alphavantage.co/query?"
    t = _asset_type(symbol)
    try:
        if t == "fx":
            pair = symbol.replace("=X", "")
            url = f"{base}function=FX_DAILY&from_symbol={pair[:3]}&to_symbol={pair[3:]}&apikey={apikey}"
            raw = _fetch(url)
            series = _normalize_av_equity_fx_crypto(raw, symbol)
            return {"Time Series (Daily)": series} if series else raw

        if t == "crypto":
            base_sym, quote = symbol.split("-", 1)
            url = f"{base}function=DIGITAL_CURRENCY_DAILY&symbol={base_sym}&market={quote}&apikey={apikey}"
            raw = _fetch(url)
            series = _normalize_av_equity_fx_crypto(raw, symbol)
            return {"Time Series (Daily)": series} if series else raw

        # equities
        url = f"{base}function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=compact&apikey={apikey}"
        raw = _fetch(url)
        series = _normalize_av_equity_fx_crypto(raw, symbol)
        return {"Time Series (Daily)": series} if series else raw

    except Exception as e:
        return {"Error": str(e)}

def call_yfinance(symbol: str) -> dict:
    """Return {'Time Series (Daily)': dict} or {'Error': '...'}."""
    series = _normalize_yf_daily(symbol)
    return {"Time Series (Daily)": series} if series else {"Error": "yfinance unavailable or no data"}

def fetch_prices_auto(symbol: str, apikey: str) -> dict:
    """Try AV first; if empty/Note/Error, fall back to yfinance."""
    av = call_alphavantage(symbol, apikey)
    ser = av.get("Time Series (Daily)") if isinstance(av, dict) else None
    if ser:
        return av
    # fall back
    yf_res = call_yfinance(symbol)
    return yf_res

# ---------- write to DB ----------
def write_price_data(stock_id: int, stock_symbol: str, daily_series: dict) -> int:
    rows = 0
    if not daily_series:
        return 0
    for date_str, ohlcv in daily_series.items():
        try:
            dt = _parse_date(date_str)
        except Exception:
            continue
        if dt.year < 2019:
            continue
        exists = db_session.query(StockDaily).filter(
            StockDaily.symbol == stock_symbol,
            StockDaily.price_date == dt
        ).count() > 0
        if exists:
            continue
        rec = StockDaily(
            stock_id=stock_id, symbol=stock_symbol, price_date=dt,
            open_price=_to_float(ohlcv.get("1. open")),
            high_price=_to_float(ohlcv.get("2. high")),
            low_price=_to_float(ohlcv.get("3. low")),
            close_price=_to_float(ohlcv.get("4. close")),
            volume=_to_float(ohlcv.get("5. volume")),
        )
        db_session.add(rec); rows += 1
    if rows: db_session.commit()
    return rows

# ---------- CLI (unchanged logic, now using AV) ----------
@stocksbp.cli.command("get_stock_data")
@click.argument("startid", type=int)
def get_stock_data(startid: int):
    apikey = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not apikey:
        print("[WARN] Missing ALPHA_VANTAGE_API_KEY env var. yfinance still works via code paths that call it explicitly.")
    starttime = datetime.now()
    calls = 0
    q = db_session.query(Stock).filter(Stock.enabled==True, Stock.id>=startid).order_by(Stock.id).limit(400)
    for s in q:
        sym = _av_symbol(s.symbol)
        data = call_alphavantage(sym, apikey)
        daily = data.get("Time Series (Daily)")
        if not daily:
            msg = None
            if isinstance(data, dict):
                msg = data.get("Note") or data.get("Error Message") or data.get("Error")
            print(f"Missing Data For: {s.symbol}" + (f" — {msg}" if msg else ""))
            if msg and "frequency" in str(msg).lower():
                time.sleep(65)
            continue
        wrote = write_price_data(s.id, s.symbol, daily)
        print(f"Inserted {wrote} rows for {s.symbol}")
        calls += 1
        elapsed = (datetime.now() - starttime).seconds
        if calls >= 5:
            if elapsed < 61:
                time.sleep(61-elapsed)
            starttime = datetime.now(); calls = 0
