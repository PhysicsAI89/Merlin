'''
signal_store.py
the forward record

every backtest merlin runs is a story about the past told with today's
universe and today's accounts. this file is the opposite: it writes down
what each strategy said on the day it said it, and then goes back later and
measures what happened next. nothing about that can be tuned after the fact,
which is the entire point.

    log_signals(strategy, rows)     write today's signals, one row per name
    update_forward_returns()        fill in what happened, at every horizon
    scoreboard()                    hit rate and mean return per strategy,
                                    with honest confidence intervals
    track_record(strategy)          the one-line version, for the pdf report

signals are logged whether or not the paper book trades them. the paper book
is one policy over these signals - position limits, regime gates, cash - and
conflating the policy with the signal is how you end up unable to tell which
of the two is not working.

what the intervals do and do not mean, stated once here rather than implied:

  the hit rate interval is a wilson interval on a binomial proportion. it is
  honest about sample size and nothing else.

  the mean return interval is a plain t interval. it assumes the trades are
  independent, and they are not - twenty signals fired on the same morning
  share a market, and overlapping holding windows share weeks. so the real
  interval is wider than the printed one, by an amount this file does not
  pretend to know. treat a result that is barely significant as not
  significant.

no strategy logic lives here. it stores what the engines said and measures
prices against it.
'''

import datetime
import json
import math
import os
import sqlite3
from contextlib import closing
import threading

import numpy as np
import pandas as pd

import datastore

DB_PATH = os.path.join('data', 'signals.db')

#the horizons every signal is measured over. 21 trading days is the headline
#because it is roughly a month and most of these strategies claim to work on
#that scale, but the shorter ones catch cortex and the longer ones catch the
#factor strategies, which are explicitly slow
HORIZONS = (1, 5, 21, 63, 126, 252)
HEADLINE_HORIZON = 21

#what each strategy is measured against. a london name beating SPY in dollars
#is partly a currency statement, so LSE signals are scored against the FTSE
BENCH_US = 'SPY'
BENCH_LSE = 'ISF.L'

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
    global _initialised
    if _initialised:
        return
    with _lock:
        if _initialised:
            return
        with closing(_connect()) as conn, conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_date TEXT NOT NULL,
                    logged_at   TEXT NOT NULL,
                    strategy    TEXT NOT NULL,
                    ticker      TEXT NOT NULL,
                    direction   TEXT NOT NULL DEFAULT 'buy',
                    confidence  REAL,
                    price       REAL,
                    currency    TEXT,
                    market      TEXT,
                    meta        TEXT,
                    UNIQUE (signal_date, strategy, ticker)
                )''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS forward (
                    signal_id     INTEGER NOT NULL,
                    horizon_days  INTEGER NOT NULL,
                    measured_at   TEXT NOT NULL,
                    exit_date     TEXT,
                    ret_pct       REAL,
                    bench_ret_pct REAL,
                    excess_pct    REAL,
                    PRIMARY KEY (signal_id, horizon_days)
                )''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sig_strategy ON signals(strategy)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sig_date ON signals(signal_date)')
        _initialised = True


# ==================== WRITE ====================

def _market_for(ticker):
    return 'LSE' if str(ticker).upper().endswith('.L') else 'US'


def log_signals(strategy, rows, signal_date=None):
    '''
    write one strategy's signals for a day.

    rows are the dicts the engines already return, so a caller passes them
    straight through. the unique key is (date, strategy, ticker), so running
    a scan twice in a day updates rather than double counting - which matters,
    because double counting the same signal is the easiest way to fake a
    track record without noticing.
    '''
    _init()
    day = signal_date or datetime.date.today().isoformat()
    now = datetime.datetime.now().isoformat(timespec='seconds')
    payload = []
    for r in rows or []:
        ticker = str(r.get('ticker', '')).upper().strip()
        if not ticker:
            continue
        price = r.get('current_price', r.get('price', r.get('entry_price')))
        try:
            price = float(price)
            if not np.isfinite(price) or price <= 0:
                price = None
        except (TypeError, ValueError):
            price = None
        meta = {k: v for k, v in r.items()
                if k not in ('ticker', 'strategy', 'confidence', 'current_price', 'price')
                and isinstance(v, (str, int, float, bool, type(None)))}
        payload.append((day, now, strategy, ticker,
                        r.get('direction', r.get('side', 'buy')),
                        r.get('confidence'), price,
                        r.get('currency') or ('GBp' if ticker.endswith('.L') else 'USD'),
                        _market_for(ticker), json.dumps(meta)[:4000]))
    if not payload:
        return 0
    with _lock, closing(_connect()) as conn, conn:
        conn.executemany('''INSERT INTO signals
              (signal_date,logged_at,strategy,ticker,direction,confidence,price,currency,market,meta)
              VALUES (?,?,?,?,?,?,?,?,?,?)
              ON CONFLICT(signal_date,strategy,ticker) DO UPDATE SET
                confidence=excluded.confidence, price=excluded.price, meta=excluded.meta''',
                         payload)
    return len(payload)


# ==================== MEASURE ====================

def _series(ticker, years=3):
    df = datastore.get_bars(ticker, years=years)
    if df is None or len(df) < 2:
        return None
    return df['Close'].astype(float)


def _forward_return(series, entry_date, horizon):
    '''
    return from the first bar on or after entry_date to `horizon` bars later.

    entry is the close of the signal day, which is the earliest a signal read
    after the close could actually have been acted on. returns None while the
    horizon has not fully elapsed, so an unfinished trade never counts.
    '''
    if series is None:
        return None, None
    try:
        idx = series.index
        pos = idx.searchsorted(pd.Timestamp(entry_date))
        if pos >= len(series):
            return None, None
        exit_pos = pos + horizon
        if exit_pos >= len(series):
            return None, None
        p0, p1 = float(series.iloc[pos]), float(series.iloc[exit_pos])
        if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0:
            return None, None
        return (p1 / p0 - 1) * 100.0, idx[exit_pos].strftime('%Y-%m-%d')
    except Exception:
        return None, None


def update_forward_returns(horizons=HORIZONS, progress=None):
    '''
    fill in what happened after every signal whose horizon has elapsed.

    cheap to re-run: rows already measured are skipped, and prices come from
    the local bar store rather than the network.
    '''
    _init()
    with closing(_connect()) as conn, conn:
        rows = conn.execute('''
            SELECT s.id, s.ticker, s.signal_date, s.market, s.direction
            FROM signals s ORDER BY s.signal_date''').fetchall()
        done = {(r[0], r[1]) for r in conn.execute(
            'SELECT signal_id, horizon_days FROM forward').fetchall()}
    if not rows:
        return 0

    tickers = sorted({r[1] for r in rows})
    if progress:
        progress(f'loading prices for {len(tickers)} tickers')
    price_cache = {}
    for t in tickers:
        price_cache[t] = _series(t)
    bench_cache = {'US': _series(BENCH_US), 'LSE': _series(BENCH_LSE)}

    written = 0
    now = datetime.datetime.now().isoformat(timespec='seconds')
    batch = []
    for sig_id, ticker, day, market, direction in rows:
        series = price_cache.get(ticker)
        bench = bench_cache.get(market or 'US')
        for h in horizons:
            if (sig_id, h) in done:
                continue
            ret, exit_date = _forward_return(series, day, h)
            if ret is None:
                continue
            #a short signal profits from a fall, so sign it the way the
            #strategy would actually have experienced it
            if str(direction).lower() in ('sell', 'short'):
                ret = -ret
            bret, _ = _forward_return(bench, day, h)
            excess = None if bret is None else ret - bret
            batch.append((sig_id, h, now, exit_date, ret, bret, excess))
            written += 1

    if batch:
        with _lock, closing(_connect()) as conn, conn:
            conn.executemany('''INSERT OR REPLACE INTO forward
                (signal_id,horizon_days,measured_at,exit_date,ret_pct,bench_ret_pct,excess_pct)
                VALUES (?,?,?,?,?,?,?)''', batch)
    if progress:
        progress(f'measured {written} signal-horizons')
    return written


# ==================== SCORE ====================

def _wilson(hits, n, z=1.96):
    '''wilson interval, which behaves at small n where the naive one does not'''
    if n == 0:
        return None, None
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round((centre - half) * 100, 1), round((centre + half) * 100, 1)


def _t_interval(values, z=1.96):
    n = len(values)
    if n < 2:
        return None, None
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1)) / math.sqrt(n)
    return round(mean - z * se, 2), round(mean + z * se, 2)


def scoreboard(horizon=HEADLINE_HORIZON, min_n=1):
    '''
    hit rate and mean return per strategy at one horizon, with intervals.

    only signals whose horizon has fully elapsed appear, so this never
    flatters itself with trades still open.
    '''
    _init()
    with closing(_connect()) as conn, conn:
        rows = conn.execute('''
            SELECT s.strategy, f.ret_pct, f.excess_pct, s.market, s.signal_date
            FROM signals s JOIN forward f ON f.signal_id = s.id
            WHERE f.horizon_days = ? AND f.ret_pct IS NOT NULL''', (horizon,)).fetchall()
        totals = dict(conn.execute(
            'SELECT strategy, COUNT(*) FROM signals GROUP BY strategy').fetchall())
        span = conn.execute('SELECT MIN(signal_date), MAX(signal_date) FROM signals').fetchone()

    by = {}
    for strategy, ret, excess, market, day in rows:
        by.setdefault(strategy, []).append((ret, excess, market, day))

    out = []
    for strategy, vals in sorted(by.items()):
        rets = [v[0] for v in vals]
        excesses = [v[1] for v in vals if v[1] is not None]
        n = len(rets)
        if n < min_n:
            continue
        hits = sum(1 for r in rets if r > 0)
        lo, hi = _wilson(hits, n)
        mlo, mhi = _t_interval(rets)
        mean = float(np.mean(rets))
        sd = float(np.std(rets, ddof=1)) if n > 1 else 0.0
        t_stat = (mean / (sd / math.sqrt(n))) if n > 1 and sd > 0 else None
        out.append({
            'strategy': strategy,
            'n_measured': n,
            'n_logged': totals.get(strategy, n),
            'hit_rate': round(hits / n * 100, 1),
            'hit_rate_lo': lo, 'hit_rate_hi': hi,
            'mean_return': round(mean, 2),
            'mean_lo': mlo, 'mean_hi': mhi,
            'median_return': round(float(np.median(rets)), 2),
            'mean_excess': round(float(np.mean(excesses)), 2) if excesses else None,
            'best': round(max(rets), 2), 'worst': round(min(rets), 2),
            't_stat': round(t_stat, 2) if t_stat is not None else None,
            #a t of 2 on independent trades is the usual bar. these are not
            #independent, so it is a floor rather than a pass mark
            'significant': bool(t_stat is not None and abs(t_stat) >= 2.0),
            'first_signal': min(v[3] for v in vals),
            'last_signal': max(v[3] for v in vals),
        })
    out.sort(key=lambda r: -(r['mean_return']))
    return {'horizon_days': horizon, 'strategies': out,
            'first_signal': span[0] if span else None,
            'last_signal': span[1] if span else None,
            'horizons_available': list(HORIZONS)}


def track_record(strategy, horizon=HEADLINE_HORIZON, min_n=10):
    '''
    the one-line version for the pdf report's per-module block.

    returns None until there are enough measured signals to be worth
    printing. a hit rate over four trades is not a track record and putting
    it next to a verdict would give it authority it has not earned.
    '''
    board = scoreboard(horizon=horizon, min_n=min_n)
    for row in board['strategies']:
        if row['strategy'] == strategy:
            return {'hit_rate': row['hit_rate'], 'n_signals': row['n_measured'],
                    'since': row['first_signal'], 'mean_return': row['mean_return'],
                    'horizon_days': horizon,
                    'hit_rate_lo': row['hit_rate_lo'], 'hit_rate_hi': row['hit_rate_hi']}
    return None


def stats():
    _init()
    with closing(_connect()) as conn, conn:
        n_sig = conn.execute('SELECT COUNT(*) FROM signals').fetchone()[0]
        n_fwd = conn.execute('SELECT COUNT(*) FROM forward').fetchone()[0]
        per = conn.execute('SELECT strategy, COUNT(*) FROM signals GROUP BY strategy').fetchall()
        days = conn.execute('SELECT COUNT(DISTINCT signal_date) FROM signals').fetchone()[0]
        span = conn.execute('SELECT MIN(signal_date), MAX(signal_date) FROM signals').fetchone()
    return {'signals': n_sig, 'measurements': n_fwd, 'days_logged': days,
            'per_strategy': dict(per),
            'first_signal': span[0], 'last_signal': span[1]}
