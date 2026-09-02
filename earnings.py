'''
earnings.py
one place that knows when a company next reports

merlin asks "when does this company report" in four different places now: the
paper trader, which should not open a three month factor position four days
before a print, cortex, which wants to know whether a gap was an earnings gap
or an ordinary one, the pdf report header and the scan tables. before this
module each of those would have grown its own yfinance call and its own idea
of what the answer means, so it lives here once.

the data problem, in short. yfinance has three fields that claim to answer
this and they do not agree:

    ticker.calendar['Earnings Date']    a list of dates, forward looking,
                                        empty for plenty of small caps
    info['earningsTimestampStart']      the next report for most tickers,
                                        stale by months for some
    info['earningsTimestamp']           the LAST report for some tickers and
                                        the NEXT one for others, which is the
                                        trap the pead engine fell into in 9.1

so the rule here is date-led rather than field-led. every candidate stamp is
collected, anything in the future is a diary entry and anything in the past
is a completed result, and the nearest one on each side is what gets stored.
a field that hands back a date from last december is simply not in the future
and falls out on its own.

    days_to_next_earnings(t)     int, or None when nothing is known
    days_since_last_earnings(t)  int, or None
    in_blackout(t, days)         (bool, reason) for the paper trader
    earnings_gap(t, bar_date)    did this gap land on a result, for cortex
    describe(t)                  the short string the ui prints

everything is cached to data/earnings_cache.json for twelve hours, because a
report date does not move intraday and a cortex sweep would otherwise ask
yahoo the same question three times in a morning. a miss is cached too, for
six hours, so tickers that simply have no calendar do not get retried on
every pass.

prime_from_info() is the one worth knowing about. the factor scanners already
pull ticker.info for every name they look at, so they can fill this cache for
free and nothing extra goes over the wire for a 300 name scan.

nothing in here raises. every public function answers None or False when it
does not know, because the alternative is a yfinance outage stopping the
paper trader from opening anything at all - see FAIL_OPEN below.
'''

import atexit
import datetime
import json
import os
import threading

import yfinance as yf

CACHE_PATH = os.path.join('data', 'earnings_cache.json')

#a report date does not change during a session. twelve hours means the
#morning cortex run and the evening paper cycle each pay once
CACHE_TTL_HOURS = 12
#a ticker with no calendar at all is worth re-asking sooner, because the
#answer changes the day the company files its notice of results
MISS_TTL_HOURS = 6

#when the lookup fails, is the ticker treated as clear or as blocked. clear,
#deliberately: a blackout that fires on missing data would stop the book
#trading entirely the next time yahoo rate limits us, and a missed blackout
#costs one position while a false one costs every position
FAIL_OPEN = True

#a date more than this far out is not a real diary entry, it is a placeholder
#or a parse accident. two years of "next earnings in 431 days" is noise
MAX_FUTURE_DAYS = 400
#the same going backwards. no company reports less often than this
MAX_PAST_DAYS = 400

#how often the cache file is actually written during a long scan
SAVE_EVERY_SECONDS = 5.0

_lock = threading.Lock()
_cache = None
_dirty = False
_last_save = datetime.datetime.min


# ==================== cache ====================

def _now():
    return datetime.datetime.now()


def _today():
    return datetime.date.today()


def _load():
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(CACHE_PATH, encoding='utf-8') as f:
            _cache = json.load(f)
        if not isinstance(_cache, dict):
            _cache = {}
    except Exception:
        _cache = {}
    return _cache


def _save(force=False):
    '''
    write the cache out, at most once every SAVE_EVERY_SECONDS.

    a 300 name factor scan primes 300 entries in a row, and dumping the whole
    file after each one is 300 writes of the same growing dict for no gain.
    the throttle keeps the common case to a handful of writes and atexit
    catches whatever the last one missed
    '''
    global _dirty, _last_save
    _dirty = True
    if not force and (_now() - _last_save).total_seconds() < SAVE_EVERY_SECONDS:
        return
    try:
        os.makedirs(os.path.dirname(CACHE_PATH) or '.', exist_ok=True)
        tmp = CACHE_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(_cache, f, indent=1)
        os.replace(tmp, CACHE_PATH)
        _dirty = False
        _last_save = _now()
    except Exception:
        pass


def flush():
    '''write anything still pending. registered with atexit below'''
    with _lock:
        if _dirty and _cache is not None:
            _save(force=True)


atexit.register(flush)


def _fresh(entry):
    '''is a cached entry still inside its ttl. a miss expires sooner'''
    if not entry or not entry.get('at'):
        return False
    try:
        age = (_now() - datetime.datetime.fromisoformat(entry['at'])).total_seconds() / 3600.0
    except Exception:
        return False
    ttl = CACHE_TTL_HOURS if (entry.get('next') or entry.get('last')) else MISS_TTL_HOURS
    return age < ttl


def clear_cache():
    '''drop everything. used by the ui refresh button and by tests'''
    global _cache
    with _lock:
        _cache = {}
        _save(force=True)


# ==================== parsing ====================

def _as_date(value):
    '''
    anything yfinance might hand back for a date, as a plain date or None.
    unix stamps, datetimes, dates, pandas timestamps and iso strings all turn
    up in these fields depending on the ticker and the endpoint
    '''
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            #seconds since the epoch. anything smaller is not a timestamp
            if value < 1e9 or value > 4e9:
                return None
            return datetime.datetime.fromtimestamp(float(value)).date()
        if isinstance(value, datetime.datetime):
            return value.date()
        if isinstance(value, datetime.date):
            return value
        #pandas timestamps answer to .to_pydatetime(), everything else to str
        if hasattr(value, 'to_pydatetime'):
            return value.to_pydatetime().date()
        text = str(value).strip()[:10]
        if len(text) == 10:
            return datetime.date.fromisoformat(text)
    except Exception:
        return None
    return None


def _sort_stamps(stamps):
    '''
    split candidate dates into the nearest future one and the nearest past
    one, ignoring anything absurdly far either side.

    this is the whole trick. yfinance's field names lie about which side of
    today they sit, so the date decides rather than the name it arrived under
    '''
    today = _today()
    nxt = last = None
    for d in stamps:
        if d is None:
            continue
        delta = (d - today).days
        if delta >= 0:
            if delta <= MAX_FUTURE_DAYS and (nxt is None or d < nxt):
                nxt = d
        else:
            if -delta <= MAX_PAST_DAYS and (last is None or d > last):
                last = d
    return nxt, last


def _from_info(info):
    '''every date-shaped field in an info blob, as a list of dates'''
    if not isinstance(info, dict):
        return []
    out = []
    for key in ('earningsTimestamp', 'earningsTimestampStart', 'earningsTimestampEnd',
                'earningsCallTimestampStart', 'earningsCallTimestampEnd'):
        out.append(_as_date(info.get(key)))
    return out


def _from_calendar(cal):
    '''the calendar dict, which is forward looking and often empty'''
    if not cal:
        return []
    try:
        raw = cal.get('Earnings Date') if isinstance(cal, dict) else None
    except Exception:
        return []
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    return [_as_date(x) for x in raw]


def _store(ticker, nxt, last, estimated, source):
    entry = {'next': nxt.isoformat() if nxt else None,
             'last': last.isoformat() if last else None,
             'estimated': bool(estimated),
             'source': source,
             'at': _now().isoformat(timespec='seconds')}
    with _lock:
        _load()[ticker.upper()] = entry
        _save()
    return entry


# ==================== lookups ====================

def prime_from_info(ticker, info):
    '''
    fill the cache from an info blob the caller already has.

    the factor scanners pull ticker.info for every name they score, so a 300
    name scan can populate all 300 earnings dates without a single extra
    request. only fills a gap - a fresh entry from the calendar endpoint is
    better than one guessed from info and is left alone
    '''
    if not ticker:
        return None
    key = ticker.upper()
    with _lock:
        existing = _load().get(key)
    if _fresh(existing) and existing.get('source') == 'calendar':
        return existing
    nxt, last = _sort_stamps(_from_info(info))
    if nxt is None and last is None:
        return _store(key, None, None, False, 'info-empty')
    return _store(key, nxt, last, bool(info.get('isEarningsDateEstimate')), 'info')


def lookup(ticker, refresh=False):
    '''
    the cached earnings dates for one ticker, fetching if needed.

    returns a dict with next, last, estimated and source. every value can be
    None and the dict itself is never None, so callers can read it flat
    '''
    empty = {'next': None, 'last': None, 'estimated': False, 'source': 'none'}
    if not ticker:
        return empty
    key = ticker.upper()
    if not refresh:
        with _lock:
            entry = _load().get(key)
        if _fresh(entry):
            return entry

    stamps, estimated, source = [], False, 'none'
    try:
        tk = yf.Ticker(key)
        try:
            cal = tk.calendar
            found = _from_calendar(cal)
            if any(d is not None for d in found):
                stamps += found
                source = 'calendar'
        except Exception:
            pass
        try:
            info = tk.info or {}
            stamps += _from_info(info)
            estimated = bool(info.get('isEarningsDateEstimate'))
            if source == 'none':
                source = 'info'
        except Exception:
            pass
    except Exception:
        return empty

    nxt, last = _sort_stamps(stamps)
    if nxt is None and last is None:
        source = 'none'
    return _store(key, nxt, last, estimated, source)


def days_to_next_earnings(ticker, refresh=False):
    '''
    trading-diary distance in calendar days to the next report, or None.

    None means "not known", never "not soon". every caller has to decide what
    to do about that itself rather than reading None as a big number
    '''
    d = _as_date(lookup(ticker, refresh).get('next'))
    return None if d is None else (d - _today()).days


def days_since_last_earnings(ticker, refresh=False):
    '''calendar days since the last reported result, or None'''
    d = _as_date(lookup(ticker, refresh).get('last'))
    return None if d is None else (_today() - d).days


def next_earnings_date(ticker):
    '''the iso date string, for printing, or None'''
    return lookup(ticker).get('next')


# ==================== the three uses ====================

def in_blackout(ticker, days, refresh=False):
    '''
    is this ticker inside its pre-earnings blackout window.

    returns (blocked, reason). a slow factor position opened four days before
    a print is not a bet on gross profitability, it is a bet on the print,
    and the strategy that was backtested was the former. days <= 0 turns the
    whole thing off.

    an unknown date never blocks - see FAIL_OPEN at the top of the file
    '''
    if not days or days <= 0:
        return False, ''
    row = lookup(ticker, refresh)
    d = _as_date(row.get('next'))
    if d is None:
        return (False, '') if FAIL_OPEN else (True, 'no earnings date known')
    n = (d - _today()).days
    if n > days:
        return False, ''
    when = 'today' if n == 0 else ('tomorrow' if n == 1 else f'in {n} days')
    est = ' (estimated date)' if row.get('estimated') else ''
    return True, f'reports {when}{est}, inside the {days} day earnings blackout'


def earnings_gap(ticker, bar_date=None, window=1):
    '''
    did this gap land on an earnings print.

    cortex fades overreactions. an earnings gap is a different animal from a
    broker-note gap or a sector gap, because the price is moving on new
    fundamental information rather than on flow, and the whole point of
    recording this is to be able to ask later whether the edge survives on
    one and not the other.

    window is in calendar days either side of the reported date, so a company
    that reports before the london open is caught on the same date and one
    that reports after the us close is caught on the next.

    returns a dict, always. flag is None when nothing is known, which is not
    the same as False and is stored as such
    '''
    out = {'flag': None, 'days_from_report': None, 'report_date': None}
    d = _as_date(bar_date) or _today()
    row = lookup(ticker)
    last = _as_date(row.get('last'))
    nxt = _as_date(row.get('next'))
    #the nearest report on either side of the bar, because a gap on the
    #morning of a print sits before a date that is still in the future
    best, gap = None, None
    for cand in (last, nxt):
        if cand is None:
            continue
        delta = abs((d - cand).days)
        if gap is None or delta < gap:
            best, gap = cand, delta
    if best is None:
        return out
    out['report_date'] = best.isoformat()
    out['days_from_report'] = (d - best).days
    out['flag'] = gap <= window
    return out


def describe(ticker, days=None):
    '''
    the short line the scan tables and the pdf header print.

    takes an already-known day count when the caller has one, so a scan table
    with fifty rows does not do fifty cache reads
    '''
    n = days_to_next_earnings(ticker) if days is None else days
    if n is None:
        return 'earnings date unknown'
    if n == 0:
        return 'reports today'
    if n == 1:
        return 'reports tomorrow'
    return f'reports in {n} days'


def summary(ticker):
    '''everything one ticker knows, for the api and the report header'''
    row = lookup(ticker)
    n = days_to_next_earnings(ticker)
    since = days_since_last_earnings(ticker)
    return {'ticker': ticker.upper(), 'next': row.get('next'), 'last': row.get('last'),
            'days_to_next': n, 'days_since_last': since,
            'estimated': bool(row.get('estimated')), 'source': row.get('source'),
            'text': describe(ticker, n)}
