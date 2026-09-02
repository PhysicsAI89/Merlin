'''
report_rank.py
cross-sectional ranking cache for the pdf report

some of merlin's models are not single-name models at all. the screener
ranks a universe on a technical score, and the quality and shareholder yield
factors only mean anything relative to everything else that was scanned. the
honest per-ticker answer for those is a percentile, not a raw number, so this
module runs the universe once, caches the distribution to disk and hands the
report a placement rather than a made-up standalone score.

nothing here reimplements a model. it calls merlin's own engines through the
module handle that app.py registers at startup, exactly the way cortex takes
its US universe, and stores what they returned.

the build is slow on purpose - it is one pass over a few hundred names - so
it never runs inside a report. it is an explicit background job with a 24
hour cache, and a report generated against a stale or missing cache says so
in plain words instead of pretending.
'''

import json
import os
import threading
import time
import datetime

RANKINGS_PATH = os.path.join('data', 'report_rankings.json')
RANKING_TTL_HOURS = 24.0

#the three cross-sectional views the report asks for. screener is a pure
#technical cross-section, the other two are the factor ranks
KINDS = ('screener', 'quality', 'shareholder_yield')

_ENGINES = None

#one build at a time, and its progress is readable from the flask routes
build_status = {'active': False, 'progress': 0, 'message': '', 'complete': False,
                'error': None, 'started_at': None}


def set_engine_module(mod):
    '''app.py hands over its own module object at startup, no circular import'''
    global _ENGINES
    _ENGINES = mod


def _engine(name):
    fn = getattr(_ENGINES, name, None) if _ENGINES is not None else None
    if fn is None:
        raise RuntimeError(f'engine {name} not registered - call set_engine_module first')
    return fn


# ==================== CACHE IO ====================

def load_rankings():
    if not os.path.exists(RANKINGS_PATH):
        return {}
    try:
        with open(RANKINGS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_rankings(book):
    os.makedirs('data', exist_ok=True)
    tmp = RANKINGS_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(book, f)
    os.replace(tmp, RANKINGS_PATH)


def _age_hours(built_at):
    try:
        built = datetime.datetime.fromisoformat(built_at)
        return (datetime.datetime.now() - built).total_seconds() / 3600.0
    except Exception:
        return None


def ranking_status():
    '''small summary for the tab so the age of the cache is never a mystery'''
    book = load_rankings()
    out = {'kinds': {}, 'build': dict(build_status)}
    for kind in KINDS:
        entry = book.get(kind)
        if not entry:
            out['kinds'][kind] = {'built': False}
            continue
        age = _age_hours(entry.get('built_at', ''))
        out['kinds'][kind] = {
            'built': True,
            'built_at': entry.get('built_at'),
            'age_hours': round(age, 1) if age is not None else None,
            'fresh': (age is not None and age <= RANKING_TTL_HOURS),
            'universe_size': entry.get('universe_size', 0),
            'n_scored': len(entry.get('scores', {})),
        }
    return out


# ==================== PLACEMENT ====================

def placement(kind, value, ticker=None):
    '''
    where does this value sit in the cached distribution.

    returns None when there is no cache at all, which the adapter turns into
    an honest partial rather than a fabricated percentile. a stale cache is
    still returned, flagged stale, because a day-old cross-section beats no
    cross-section as long as the reader is told.
    '''
    if value is None:
        return None
    book = load_rankings()
    entry = book.get(kind)
    if not entry or not entry.get('scores'):
        return None

    scores = entry['scores']
    values = sorted(float(v) for v in scores.values())
    n = len(values)
    if n < 5:
        return None

    below = sum(1 for v in values if v < float(value))
    pct = round(below / n * 100, 1)
    #rank counts from the top, so rank 1 is the best name in the cross-section
    rank = sum(1 for v in values if v > float(value)) + 1
    age = _age_hours(entry.get('built_at', ''))

    return {
        'percentile': pct,
        'rank': rank,
        'n_scored': n,
        'universe_size': entry.get('universe_size', n),
        'built_at': entry.get('built_at'),
        'age_hours': round(age, 1) if age is not None else None,
        'stale': (age is None or age > RANKING_TTL_HOURS),
        'in_universe': bool(ticker and ticker in scores),
        'median': round(values[n // 2], 2),
    }


# ==================== BUILD ====================

def _build(count):
    '''
    one pass over the universe collecting what each engine returned. the
    screener score comes from quick_score_stock, the two factor scores from
    get_research_factor_signals with the confidence floor dropped to zero so
    the distribution is not pre-truncated by the tab default.
    '''
    global build_status
    try:
        universe = _engine('get_stock_universe')(count) or []
        if not universe:
            build_status.update({'error': 'could not fetch universe',
                                 'active': False, 'complete': True})
            return

        quick = _engine('quick_score_stock')
        factors = _engine('get_research_factor_signals')

        buckets = {k: {} for k in KINDS}
        total = len(universe)
        for i, t in enumerate(universe):
            build_status['progress'] = int(i / max(total, 1) * 100)
            build_status['message'] = f'ranking {t} ({i + 1}/{total})'
            try:
                q = quick(t)
                if q and q.get('score') is not None:
                    buckets['screener'][t] = float(q['score'])
            except Exception:
                pass
            try:
                for sig in (factors(t, 0) or []):
                    strat = sig.get('strategy')
                    if strat in buckets and sig.get('confidence') is not None:
                        buckets[strat][t] = float(sig['confidence'])
            except Exception:
                pass
            if i % 10 == 0 and i > 0:
                time.sleep(0.3)

        book = load_rankings()
        now = datetime.datetime.now().isoformat(timespec='seconds')
        for kind in KINDS:
            if buckets[kind]:
                book[kind] = {'built_at': now, 'universe_size': total,
                              'scores': buckets[kind]}
        _save_rankings(book)

        build_status.update({
            'progress': 100, 'complete': True, 'active': False,
            'message': f'done - {total} names, ' +
                       ', '.join(f'{k} {len(buckets[k])}' for k in KINDS),
        })
    except Exception as e:
        build_status.update({'error': str(e)[:200], 'active': False, 'complete': True})


def start_build(count=150):
    '''kick the ranking build off in the background, one at a time'''
    global build_status
    if build_status.get('active'):
        return False
    build_status = {'active': True, 'progress': 0, 'message': 'starting...',
                    'complete': False, 'error': None,
                    'started_at': datetime.datetime.now().isoformat(timespec='seconds')}
    threading.Thread(target=_build, args=(count,), daemon=True).start()
    return True
