'''
datastore.py
one local bar store for the whole of merlin

the problem this solves. every tab fetches its own price history from
yfinance, so a backtest over 300 names downloads 300 times, a scan behind it
downloads them again, and somewhere in the middle yahoo starts refusing.
worse, two runs of the same backtest a week apart are not the same
experiment, because the data underneath moved. a result you cannot reproduce
is not a result.

so: one sqlite file, one way in, one way out.

    get_bars(ticker, years=3)        one frame, cached
    get_many([...], years=3)         many frames, one batched download for
                                     whatever is missing
    warm([...], years=5)             fill the store ahead of a long job
    stats()                          what is in there

three rules the store keeps, because they are the ones that bit us:

  1. a row with no close never gets stored. yfinance appends a final bar
     carrying volume and nothing else while a session settles, and every
     indicator that fills missing values with zero then reads the latest
     price as nothing. it is filtered here so it can never reach an engine.

  2. today is never stored. an intraday bar is not a daily bar, and caching
     one would poison every later read of that date.

  3. writes are upserts keyed on (ticker, date), so refetching an overlapping
     window corrects history rather than duplicating it.

this module deliberately does not know anything about strategies. it hands
back frames shaped exactly like the yfinance ones the engines already expect
- Open, High, Low, Close, Adj Close, Volume on a DatetimeIndex - so a call
site can be switched over one line at a time.
'''

import datetime
import os
import sqlite3
from contextlib import closing
import threading

import numpy as np
import pandas as pd
import yfinance as yf


DB_PATH = os.path.join('data', 'bars.db')
COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

#how long a ticker's cached tail is trusted before we go back for more. a
#daily bar store only changes once a day, so this is about the freshness of
#the last row rather than about the history behind it
DEFAULT_MAX_AGE_HOURS = 12

_lock = threading.Lock()
_initialised = False


def _connect():
    #callers must wrap this in contextlib.closing as well as `with conn`.
    #sqlite3's context manager commits the transaction, it does not close the
    #connection - relying on it alone leaks a handle per call, which a
    #long-running flask process notices well before you do.
    os.makedirs(os.path.dirname(DB_PATH) or '.', exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute('PRAGMA journal_mode=WAL')
    return conn


def _init():
    '''create the schema once per process'''
    global _initialised
    if _initialised:
        return
    with _lock:
        if _initialised:
            return
        with closing(_connect()) as conn, conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS bars (
                    ticker    TEXT NOT NULL,
                    date      TEXT NOT NULL,
                    open      REAL,
                    high      REAL,
                    low       REAL,
                    close     REAL NOT NULL,
                    adj_close REAL,
                    volume    REAL,
                    PRIMARY KEY (ticker, date)
                )''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_bars_ticker ON bars(ticker)')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS fetch_log (
                    ticker       TEXT PRIMARY KEY,
                    last_fetched TEXT,
                    first_date   TEXT,
                    last_date    TEXT,
                    rows         INTEGER
                )''')
        _initialised = True


# ==================== WRITE ====================

def _clean(df):
    '''
    the two rules that keep bad bars out: no row without a close, and nothing
    dated today. everything else is passed through untouched.
    '''
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if 'Close' not in df.columns:
        return None
    df = df[df['Close'].notna()]
    if len(df) == 0:
        return None
    idx = pd.to_datetime(df.index)
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx
    today = pd.Timestamp(datetime.date.today())
    df = df[df.index < today]
    return df if len(df) else None


def store(ticker, df):
    '''upsert a frame. returns the number of rows written'''
    _init()
    df = _clean(df)
    if df is None:
        return 0
    rows = []
    for ts, r in df.iterrows():
        def g(col):
            v = r.get(col)
            try:
                v = float(v)
                return v if np.isfinite(v) else None
            except (TypeError, ValueError):
                return None
        close = g('Close')
        if close is None:
            continue
        rows.append((ticker, ts.strftime('%Y-%m-%d'), g('Open'), g('High'), g('Low'),
                     close, g('Adj Close'), g('Volume')))
    if not rows:
        return 0
    with _lock, closing(_connect()) as conn, conn:
        conn.executemany('''INSERT INTO bars (ticker,date,open,high,low,close,adj_close,volume)
                            VALUES (?,?,?,?,?,?,?,?)
                            ON CONFLICT(ticker,date) DO UPDATE SET
                              open=excluded.open, high=excluded.high, low=excluded.low,
                              close=excluded.close, adj_close=excluded.adj_close,
                              volume=excluded.volume''', rows)
        cur = conn.execute('SELECT MIN(date), MAX(date), COUNT(*) FROM bars WHERE ticker=?',
                           (ticker,))
        first, last, n = cur.fetchone()
        conn.execute('''INSERT INTO fetch_log (ticker,last_fetched,first_date,last_date,rows)
                        VALUES (?,?,?,?,?)
                        ON CONFLICT(ticker) DO UPDATE SET
                          last_fetched=excluded.last_fetched, first_date=excluded.first_date,
                          last_date=excluded.last_date, rows=excluded.rows''',
                     (ticker, datetime.datetime.now().isoformat(timespec='seconds'),
                      first, last, n))
    return len(rows)


# ==================== READ ====================

def _read(ticker, start=None):
    _init()
    with closing(_connect()) as conn, conn:
        if start:
            cur = conn.execute('SELECT date,open,high,low,close,adj_close,volume FROM bars '
                               'WHERE ticker=? AND date>=? ORDER BY date', (ticker, start))
        else:
            cur = conn.execute('SELECT date,open,high,low,close,adj_close,volume FROM bars '
                               'WHERE ticker=? ORDER BY date', (ticker,))
        rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['date', 'Open', 'High', 'Low', 'Close',
                                     'Adj Close', 'Volume'])
    df.index = pd.to_datetime(df.pop('date'))
    df.index.name = 'Date'
    return df


def _age_hours(ticker):
    _init()
    with closing(_connect()) as conn, conn:
        cur = conn.execute('SELECT last_fetched FROM fetch_log WHERE ticker=?', (ticker,))
        row = cur.fetchone()
    if not row or not row[0]:
        return None
    try:
        return (datetime.datetime.now()
                - datetime.datetime.fromisoformat(row[0])).total_seconds() / 3600.0
    except ValueError:
        return None


def _start_for(years):
    return (datetime.date.today() - datetime.timedelta(days=int(years * 365) + 45)).isoformat()


def get_bars(ticker, years=3, max_age_hours=DEFAULT_MAX_AGE_HOURS, refresh=False):
    '''
    daily bars for one ticker, from the store, fetching only what is missing.

    returns a frame shaped like yfinance's, or None. never raises on a
    network failure - a stale local answer beats an exception, and the caller
    can see how old it is through stats().
    '''
    start = _start_for(years)
    cached = None if refresh else _read(ticker, start)
    age = _age_hours(ticker)
    fresh_enough = (cached is not None and len(cached) > 20
                    and age is not None and age <= max_age_hours)
    if fresh_enough:
        return cached

    try:
        raw = yf.download(ticker, start=start, progress=False,
                          auto_adjust=False, threads=False)
        if store(ticker, raw):
            return _read(ticker, start)
    except Exception:
        pass
    return cached


def get_many(tickers, years=3, max_age_hours=DEFAULT_MAX_AGE_HOURS, chunk=50, progress=None):
    '''
    the same for a list, batching whatever is missing into group downloads.

    this is the call that makes a backtest cheap: 300 names that are already
    in the store cost one sqlite read each and no network at all.
    '''
    start = _start_for(years)
    out, need = {}, []
    for t in tickers:
        cached = _read(t, start)
        age = _age_hours(t)
        if cached is not None and len(cached) > 20 and age is not None and age <= max_age_hours:
            out[t] = cached
        else:
            need.append(t)

    if progress:
        progress(f'{len(out)} tickers cached, fetching {len(need)}')

    for i in range(0, len(need), chunk):
        batch = need[i:i + chunk]
        try:
            raw = yf.download(batch, start=start, group_by='ticker', progress=False,
                              auto_adjust=False, threads=True)
        except Exception:
            raw = None
        if raw is not None and len(raw):
            for t in batch:
                try:
                    sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                except KeyError:
                    continue
                if store(t, sub):
                    got = _read(t, start)
                    if got is not None:
                        out[t] = got
        if progress:
            progress(f'fetched {min(i + chunk, len(need))}/{len(need)}')

    #anything the network would not give us still gets its stale copy
    for t in need:
        if t not in out:
            cached = _read(t, start)
            if cached is not None and len(cached) > 20:
                out[t] = cached
    return out


def warm(tickers, years=5, progress=None):
    '''fill the store ahead of a long job so the job itself never waits'''
    got = get_many(tickers, years=years, max_age_hours=DEFAULT_MAX_AGE_HOURS, progress=progress)
    return len(got)


def stats():
    '''what the store holds, for a status line'''
    _init()
    with closing(_connect()) as conn, conn:
        n_t = conn.execute('SELECT COUNT(*) FROM fetch_log').fetchone()[0]
        n_b = conn.execute('SELECT COUNT(*) FROM bars').fetchone()[0]
        rng = conn.execute('SELECT MIN(date), MAX(date) FROM bars').fetchone()
        oldest = conn.execute('SELECT ticker, last_fetched FROM fetch_log '
                              'ORDER BY last_fetched LIMIT 1').fetchone()
    size_mb = round(os.path.getsize(DB_PATH) / 1e6, 2) if os.path.exists(DB_PATH) else 0.0
    return {'tickers': n_t, 'bars': n_b, 'first_date': rng[0], 'last_date': rng[1],
            'size_mb': size_mb,
            'stalest': {'ticker': oldest[0], 'fetched': oldest[1]} if oldest else None}
