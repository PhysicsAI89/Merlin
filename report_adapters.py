'''
report_adapters.py
the uniform module contract for the pdf report, and one adapter per module

this file holds three things and nothing else:

    ReportSection   the shape every module returns. same fields whatever the
                    module is, which is what makes adding a future tab to the
                    report a twenty line job rather than a template rewrite

    helpers         the shared normalisation - a score band, a confidence to
                    score mapping, a number formatter. every adapter uses the
                    same ones so the -1 to +1 scale means the same thing in
                    every block

    adapters        one thin function per merlin module, each taking the
                    shared ReportContext and returning a ReportSection

the rule that matters: an adapter reads from the context and calls the engine
that already exists. it does not fetch, and it does not decide anything a
strategy should be deciding. grep this file for yfinance and you will find
nothing - that is the point, and it is worth keeping true.

report.py imports these and does the orchestration. app.py registers itself
here at startup so nothing in this file ever imports app.py back.
'''

import time
from dataclasses import dataclass, field

import numpy as np

import cortex
import report_rank


VERDICT_BANDS = [(0.45, 'STRONG_BUY'), (0.15, 'BUY'),
                 (-0.15, 'NEUTRAL'), (-0.45, 'AVOID')]

CLUSTER_TTL_SECONDS = 1800   #the openinsider scrape is universe-wide, reuse it

_ENGINES = None
_cluster_cache = {'fetched_at': 0.0, 'clusters': None}


def set_engine_module(mod):
    '''app.py hands over its own module object at startup, no circular import'''
    global _ENGINES
    _ENGINES = mod
    report_rank.set_engine_module(mod)


def engine(name):
    fn = getattr(_ENGINES, name, None) if _ENGINES is not None else None
    if fn is None:
        raise RuntimeError(f'engine {name} not registered - call set_engine_module first')
    return fn


# ==================== SHARED NORMALISATION ====================

def clamp(x, lo=-1.0, hi=1.0):
    try:
        v = float(x)
        if not np.isfinite(v):
            return 0.0
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return 0.0


def verdict_for(score):
    '''one score band shared by every module and by the composite'''
    if score is None:
        return 'UNAVAILABLE'
    for threshold, label in VERDICT_BANDS:
        if score >= threshold:
            return label
    return 'STRONG_AVOID'


def conf_score(confidence, floor=60.0, ceiling=95.0, sign=1):
    '''
    a merlin confidence of 0-100 onto 0.2..1.0, signed.

    the floor matters: these engines only emit a signal once they are past
    their own threshold, so the weakest signal any of them produces is still
    a real one and should not land on zero.
    '''
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return 0.0
    frac = max(0.0, min(1.0, (c - floor) / max(ceiling - floor, 1.0)))
    return clamp(sign * (0.2 + 0.8 * frac))


def fmt(v, dp=2, suffix='', na='n/a'):
    if v is None:
        return na
    try:
        f = float(v)
        if not np.isfinite(f):
            return na
        return f'{f:,.{dp}f}{suffix}'
    except (TypeError, ValueError):
        return f'{v}{suffix}'


def _m(label, value, note=''):
    return (label, value, note)


def currency_symbol(ccy):
    return {'USD': '$', 'GBP': '£', 'GBp': 'p', 'EUR': '€',
            'CAD': 'C$', 'AUD': 'A$'}.get(ccy, '')


@dataclass
class ReportSection:
    key: str
    label: str
    verdict: str = 'UNAVAILABLE'
    score: float = None
    metrics: list = field(default_factory=list)
    commentary: str = ''
    data_status: str = 'ok'          #ok | partial | unavailable
    error: str = None
    track_record: dict = None        #phase 2, stays None until a signal store exists


# ==================== ADAPTERS ====================
#
#one per module. each reads only from the context, normalises the engine's
#own output onto -1..+1 and writes a table. none of them fetches anything.

#yfinance sometimes hands back a final daily row carrying volume but no
#open, high, low or close. every engine in merlin that fills missing values
#with zero then computes its indicators off a bar priced at nothing: the
#short moving averages sag, bollinger position goes negative and the last
#price reads as zero. nothing raises, so it looks like a real answer. the
#adapters below refuse to report a verdict built on that bar - a missing
#module is honest, a wrong one is not.
INCOMPLETE_BAR_NOTE = ('this engine priced the ticker at something the context does not recognise, '
                       'which is what a zero-filled incomplete bar looks like from the outside. '
                       'reported as unavailable rather than wrong.')


def _engine_price_is_sane(ctx, engine_price):
    '''
    does the price an engine ended up with match the one in the context.

    this is a check on the symptom rather than on a suspected cause. the
    incomplete-bar bug is fixed in app.py, so guessing from "yfinance sent a
    partial row" would now suppress modules that are working perfectly well.
    comparing the two prices catches the real thing if it ever comes back,
    and stays quiet otherwise. the tolerance is wide because the engines
    download adjusted closes and the context does not, so dividends put a
    few percent between them legitimately.
    '''
    try:
        p = float(engine_price)
        if not np.isfinite(p) or p <= 0:
            return False
        return 0.7 <= (p / ctx.price) <= 1.4
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def adapt_technicals(ctx):
    q = ctx.quick
    if q and not _engine_price_is_sane(ctx, q.get('price')):
        return ReportSection('technicals', 'Technicals', data_status='unavailable',
                             metrics=[_m('engine last price', fmt(q.get('price')),
                                         'does not match the header price'),
                                      _m('context last price', fmt(ctx.price))],
                             error='engine indicators computed against an implausible price',
                             commentary=INCOMPLETE_BAR_NOTE)
    if not q:
        return ReportSection('technicals', 'Technicals',
                             data_status='unavailable',
                             error='quick_score_stock returned nothing for this ticker',
                             commentary='the technical engine needs six months of daily bars '
                                        'and could not score this name.')
    raw = float(q.get('score', 0) or 0)
    score = clamp(raw / 6.0)
    sym = currency_symbol(ctx.currency)
    return ReportSection(
        'technicals', 'Technicals', verdict_for(score), score,
        metrics=[
            _m('last price', f'{sym}{fmt(q.get("price"))}', ctx.currency),
            _m('change 1d', fmt(q.get('change_1d'), 2, '%')),
            _m('change 5d', fmt(q.get('change_5d'), 2, '%')),
            _m('RSI (14)', fmt(q.get('rsi'), 1), 'below 30 oversold, above 70 overbought'),
            _m('raw technical score', fmt(raw, 1), 'engine scale, roughly -12 to +12'),
            _m('direction', q.get('direction', 'neutral')),
            _m('signals fired', ', '.join(q.get('signals') or []) or 'none'),
        ],
        commentary=f'the screener technicals put {ctx.ticker} at {raw:+.1f} on its own scale, '
                   f'which reads as {q.get("direction", "neutral")}. this is a short horizon view '
                   f'built from moving averages, RSI, MACD, bollinger position and volume.')


def adapt_fundamentals(ctx):
    f = ctx.fundamentals
    if not f or f.get('error'):
        return ReportSection('fundamentals', 'Fundamentals', data_status='unavailable',
                             error=(f or {}).get('error', 'no fundamentals returned'),
                             commentary='yfinance had no fundamentals for this listing. coverage '
                                        'outside the US large caps is patchy.')
    a = f.get('assessment') or {}
    raw = float(a.get('score', 0) or 0)
    score = clamp(raw / 5.0)
    fields = [('pe_ratio', 'P/E'), ('forward_pe', 'forward P/E'), ('peg_ratio', 'PEG'),
              ('price_to_book', 'price to book'), ('beta', 'beta'),
              ('dividend_yield', 'dividend yield %'), ('profit_margins', 'profit margin %'),
              ('revenue_growth', 'revenue growth %'), ('return_on_equity', 'return on equity %'),
              ('debt_to_equity', 'debt to equity'), ('range_position', '52w range position %')]
    metrics = []
    for key, label in fields:
        value, note = f.get(key), ''
        #yfinance changed dividendYield from a fraction to a percentage.
        #get_fundamentals still multiplies by a hundred, so a 3.5% yield
        #arrives here as 350. show the sane number and say why, rather than
        #printing a yield no company has ever paid
        if key == 'dividend_yield' and _dividend_scaled_100x(ctx, value):
            value = float(ctx.info.get('dividendYield'))
            note = 'engine multiplied a percentage by 100, shown corrected'
        metrics.append(_m(label, fmt(value, 2), note))
    missing = sum(1 for key, _ in fields if f.get(key) is None)
    status = 'partial' if missing >= 4 else 'ok'
    return ReportSection(
        'fundamentals', 'Fundamentals', verdict_for(score), score, metrics,
        commentary=f'{a.get("summary", "")}. balance sheet strength reads '
                   f'{a.get("strength", "?")} - {a.get("strength_text", "")}.' +
                   (f' {missing} of {len(fields)} fields were not covered by the data source.'
                    if status == 'partial' else ''),
        data_status=status,
        error=None if status == 'ok' else f'{missing} fundamental fields unavailable')


def adapt_news(ctx):
    news = engine('get_news_sentiment')(ctx.ticker) or {}
    n = int(news.get('count', 0) or 0)
    if n == 0:
        return ReportSection('news', 'News sentiment', data_status='unavailable',
                             error='no headlines returned for this ticker',
                             commentary='the news feed had nothing for this listing, which is '
                                        'common for smaller non-US names.')
    raw = float(news.get('overall_score', 0) or 0)
    score = clamp(raw / 1.5)
    metrics = [_m('headlines scored', str(n)),
               _m('mean sentiment', fmt(raw, 2), 'per headline, scale -3 to +3'),
               _m('overall', news.get('summary', 'mixed'))]
    for art in (news.get('articles') or [])[:3]:
        metrics.append(_m(art.get('sentiment', 'neutral'), art.get('title', '')[:110],
                          art.get('publisher', '')))
    return ReportSection(
        'news', 'News sentiment', verdict_for(score), score, metrics,
        commentary=f'{n} recent headlines average {raw:+.2f} on a keyword sentiment scale, '
                   f'reading {news.get("summary", "mixed")}. this is word matching on titles, not '
                   f'comprehension, so treat it as a coarse mood gauge.')


#NOT IN THE REPORT SINCE 10.0. this adapter still works and is left here
#deliberately, but it is no longer listed in ANALYSERS in report.py so it
#is never called. put the line back there to bring the block back.
def adapt_insider(ctx):
    ins = engine('get_insider_activity')(ctx.ticker) or {}
    txns = ins.get('transactions') or []
    if not txns:
        return ReportSection('insider', 'Insider activity', data_status='unavailable',
                             error='no insider transactions on file',
                             commentary='no filed insider transactions were returned. UK listings '
                                        'file to the RNS rather than to a feed merlin reads.')
    #the engine reads the buy or sell wording out of a Transaction column and
    #the name out of an "Insider Trading" column. yfinance now puts the
    #wording in Text ("Bought at price ...", "Sold at price ...") and the name
    #in Insider, and leaves Transaction blank. every row therefore classifies
    #as "other" and the module reports a confident neutral on a ticker with a
    #hundred real filings behind it. transactions present but none classified
    #is the fingerprint of that, and it is not a neutral reading
    classified = [t for t in txns if t.get('action') in ('buy', 'sell')]
    if not classified:
        return ReportSection(
            'insider', 'Insider activity', data_status='unavailable',
            metrics=[_m('transactions on file', str(len(txns))),
                     _m('classified as buy or sell', '0'),
                     _m('named insiders', str(sum(1 for t in txns if t.get('name')))),
                     _m('most recent filing', (txns[0].get('date') or '')[:10])],
            error='filings present but none could be classified as a buy or a sell',
            commentary=f'{len(txns)} insider filings came back for {ctx.ticker} and the engine '
                       f'could not read a single one of them. it looks for the buy or sell wording '
                       f'in a Transaction column, and yfinance now leaves that column blank and '
                       f'puts the wording in Text instead. so this is not a quiet insider tape, it '
                       f'is a parser reading the wrong field - reported as unavailable rather than '
                       f'as neutral. this affects the Insider Trading tab too, on every ticker.')

    raw = float(ins.get('score', 0) or 0)
    score = clamp(raw / 2.0)
    return ReportSection(
        'insider', 'Insider activity', verdict_for(score), score,
        metrics=[_m('open-market buys', str(ins.get('all_buys', 0))),
                 _m('sells', str(ins.get('all_sells', 0))),
                 _m('senior buys', str(ins.get('exec_buys', 0)),
                    '' if ins.get('titles_available') else 'no job titles filed for this listing'),
                 _m('senior sells', str(ins.get('exec_sells', 0))),
                 _m('filings parsed', str(ins.get('parsed', len(txns)))),
                 _m('scored on', 'senior filings' if ins.get('titles_available') else 'all filings'),
                 _m('sentiment', ins.get('sentiment', 'neutral'))],
        commentary=(f'{ins.get("exec_buys", 0)} senior buys against {ins.get("exec_sells", 0)} '
                    f'senior sells'
                    if ins.get('titles_available') else
                    f'{ins.get("all_buys", 0)} open-market buys against {ins.get("all_sells", 0)} '
                    f'sells, scored on the whole tape because this listing files no job titles')
                   + f', across {ins.get("parsed", len(txns))} filings, which reads '
                     f'{ins.get("sentiment", "neutral")}. awards, gifts and option exercises are '
                     f'deliberately not counted as buys - being handed shares says nothing about '
                     f'what an insider thinks of the price. sells are noisier than buys either '
                     f'way, since many are scheduled disposals.')


#NOT IN THE REPORT SINCE 10.0. this adapter still works and is left here
#deliberately, but it is no longer listed in ANALYSERS in report.py so it
#is never called. put the line back there to bring the block back.
def adapt_clusters(ctx):
    if ctx.is_lse:
        return ReportSection('clusters', 'Insider clusters', data_status='unavailable',
                             error='openinsider covers US listings only',
                             commentary='the cluster engine reads openinsider.com, which carries '
                                        'SEC form 4 filings. a London listing has no form 4 and so '
                                        'can never appear, whatever the insiders are doing.')
    now = time.time()
    if _cluster_cache['clusters'] is None or now - _cluster_cache['fetched_at'] > CLUSTER_TTL_SECONDS:
        _cluster_cache['clusters'] = engine('get_insider_clusters')() or []
        _cluster_cache['fetched_at'] = now
    clusters = _cluster_cache['clusters']
    mine = next((c for c in clusters if c.get('ticker') == ctx.ticker), None)
    if mine is None:
        return ReportSection(
            'clusters', 'Insider clusters', 'NEUTRAL', 0.0,
            metrics=[_m('qualifying clusters in universe', str(len(clusters))),
                     _m('this ticker', 'no cluster',
                        'needs 3+ unique senior buyers inside 60 days')],
            commentary=f'no insider buy cluster for {ctx.ticker} in the last 60 days. that is the '
                       f'normal state for almost every name and is not bearish - it means this '
                       f'particular edge has nothing to say. {len(clusters)} other names in the '
                       f'same scan did qualify.')
    rank = sorted(clusters, key=lambda c: -c.get('confidence', 0)).index(mine) + 1
    score = conf_score(mine.get('confidence'), 40.0, 95.0)
    return ReportSection(
        'clusters', 'Insider clusters', verdict_for(score), score,
        metrics=[_m('unique insiders buying', str(mine.get('n_insiders'))),
                 _m('senior buyers', str(mine.get('senior_count'))),
                 _m('total value', f'${mine.get("total_value", 0):,.0f}'),
                 _m('most recent filing', mine.get('most_recent_date', '')),
                 _m('engine confidence', fmt(mine.get('confidence'), 1)),
                 _m('rank in scan', f'{rank} of {len(clusters)}', 'by confidence')],
        commentary=f'{mine.get("n_insiders")} separate insiders bought {ctx.ticker} inside the '
                   f'window, {mine.get("senior_count")} of them senior, '
                   f'${mine.get("total_value", 0):,.0f} in total. cohen malloy and pomorski (2012) '
                   f'found clusters like this predict roughly 6-10% of outperformance over the '
                   f'following 60 days.')


def adapt_pead(ctx):
    sig = engine('get_pead_signal')(ctx.ticker)
    if not sig:
        rows = getattr(ctx, 'earnings_rows', 0)
        age = getattr(ctx, 'earnings_days_ago', None)
        nxt = getattr(ctx, 'earnings_next', None)
        base = [_m('last reported', getattr(ctx, 'earnings_last', None) or 'not known'),
                _m('days since', 'n/a' if age is None else str(age)),
                _m('next scheduled', nxt or 'not scheduled'),
                _m('drift window', '30 days after the event')]

        #a silent PEAD module means one of three completely different things
        #and the engine returns the same None for all of them. separate them
        if rows == 0 or age is None:
            #a known future date and no past one is a real answer: nothing has
            #been reported recently enough to drift from. that is a neutral
            #reading, not missing data
            if nxt:
                return ReportSection(
                    'pead', 'Earnings drift (PEAD)', 'NEUTRAL', 0.0, metrics=base,
                    commentary=f'{ctx.ticker} has no completed result close enough to drift from. '
                               f'the next one is scheduled for {nxt}, and the drift window only '
                               f'opens once a result is actually out.')
            return ReportSection(
                'pead', 'Earnings drift (PEAD)', data_status='unavailable', metrics=base,
                error='no earnings calendar available for this ticker',
                commentary=f'yfinance returned no usable earnings calendar for {ctx.ticker}, so '
                           f'there is nothing for the drift model to measure. that is missing '
                           f'data, not a neutral reading.')
        #no listed company goes much over six months between results. a
        #calendar whose newest event is older than that is stale, not quiet
        if age > 200 and nxt:
            return ReportSection(
                'pead', 'Earnings drift (PEAD)', 'NEUTRAL', 0.0, metrics=base,
                commentary=f'{ctx.ticker} has no completed result close enough to drift from. the '
                           f'next one is scheduled for {nxt}, and the drift window only opens once '
                           f'a result is actually out.')
        if age > 200:
            return ReportSection(
                'pead', 'Earnings drift (PEAD)', data_status='unavailable', metrics=base,
                error=f'newest earnings event on file is {age} days old',
                commentary=f'the most recent earnings event yfinance has for {ctx.ticker} is '
                           f'{age} days ago, and no company reports that rarely. the calendar '
                           f'feed is stale, so the drift window can never open and this module '
                           f'cannot fire for any ticker while that lasts. reported as unavailable '
                           f'rather than as a neutral verdict it did not actually reach.')
        return ReportSection(
            'pead', 'Earnings drift (PEAD)', 'NEUTRAL', 0.0, metrics=base,
            commentary=f'{ctx.ticker} last reported {age} days ago. either that is outside the '
                       f'thirty day drift window, or the surprise was inside plus or minus five '
                       f'percent. the drift only exists for a few weeks after the event, so a '
                       f'genuine silence here is the normal state.')
    sign = 1 if sig.get('direction') == 'up' else -1
    score = conf_score(sig.get('confidence'), 20.0, 90.0, sign)
    return ReportSection(
        'pead', 'Earnings drift (PEAD)', verdict_for(score), score,
        metrics=[_m('earnings date', sig.get('earnings_date', '')),
                 _m('days since', str(sig.get('days_since_earnings'))),
                 _m('EPS actual', fmt(sig.get('eps_actual'), 3)),
                 _m('EPS estimate', fmt(sig.get('eps_estimate'), 3)),
                 _m('surprise', fmt(sig.get('surprise_pct'), 1, '%')),
                 _m('move since earnings', fmt(sig.get('move_since_earnings'), 2, '%')),
                 _m('analyst revisions', fmt(sig.get('revision_score'), 1),
                    sig.get('revision_note', '')),
                 _m('engine confidence', fmt(sig.get('confidence'), 1))],
        commentary=f'{ctx.ticker} {"beat" if sign > 0 else "missed"} by '
                   f'{abs(sig.get("surprise_pct", 0)):.1f}% {sig.get("days_since_earnings")} days '
                   f'ago and has moved {sig.get("move_since_earnings", 0):+.1f}% since. ball and '
                   f'brown (1968) is the original finding: prices keep drifting in the direction '
                   f'of the surprise for weeks, strongest in the first two or three.')


def adapt_momentum(ctx):
    spy_mom = 0.0
    if ctx.spy is not None and len(ctx.spy) >= 260:
        c = ctx.spy['Close'].astype(float)
        spy_mom = float(c.iloc[-22] / c.iloc[-253] - 1)
    sig = engine('get_momentum_signal')(ctx.ticker, spy_mom)

    close = ctx.close
    mom_12_1 = float(close.iloc[-22] / close.iloc[-253] - 1) * 100 if len(close) > 253 else None
    high_52 = float(close.tail(252).max())
    proximity = ctx.price / high_52 * 100 if high_52 > 0 else None
    rel = None if mom_12_1 is None else mom_12_1 - spy_mom * 100

    base = [_m('12-1 month momentum', fmt(mom_12_1, 2, '%'),
               'return to a month ago, skipping the last month'),
            _m('SPY 12-1 momentum', fmt(spy_mom * 100, 2, '%')),
            _m('relative to SPY', fmt(rel, 2, '%')),
            _m('proximity to 52w high', fmt(proximity, 1, '%'))]

    if not sig:
        #capped below the strong bands on purpose. this is a descriptive
        #fallback measured off the shared context, not the engine speaking,
        #and a fallback should never be able to produce the loudest verdict
        #in the report
        score = clamp((rel or 0) / 100.0 * 2.0, -0.4, 0.4)
        return ReportSection(
            'momentum', 'Relative momentum', verdict_for(score), score,
            metrics=base + [_m('screener fired', 'no',
                               'needs 12-1 above 10%, within 15% of the 52w high, beating SPY')],
            commentary=f'the momentum screener did not fire for {ctx.ticker}, so the score here is '
                       f'the underlying relative momentum rather than an engine signal, and is '
                       f'capped at plus or minus 0.4 so a fallback can never shout. relative '
                       f'12-1 momentum is '
                       f'{"n/a" if rel is None else f"{rel:+.1f}%"} against SPY. the strategy is on '
                       f'probation either way - the 5y backtest gave it +25.8% against SPY +91%.')
    score = conf_score(sig.get('confidence'), 65.0, 96.0)
    #the engine measures proximity to the 52 week high off its own last close.
    #when yfinance hands it an incomplete final bar that close is nan, every
    #comparison against nan is false and the proximity gate passes by default
    #rather than being evaluated. quote the context's own figure instead and
    #say that the gate did not really run
    engine_prox = sig.get('proximity_to_52w_high_pct')
    prox_text = fmt(engine_prox, 1, '%')
    gate_note = ''
    if engine_prox is None or not np.isfinite(float(engine_prox or float('nan'))):
        prox_text = fmt(proximity, 1, '%') + ' (measured here, not by the engine)'
        gate_note = (' the engine could not measure proximity to the 52 week high because its '
                     'own final bar had no close, so that gate passed by default rather than '
                     'being tested. the figure quoted is measured by this report instead.')
    return ReportSection(
        'momentum', 'Relative momentum', verdict_for(score), score,
        metrics=base + [_m('positive day ratio', fmt(sig.get('positive_day_ratio_pct'), 1, '%'),
                           'frog in the pan smoothness, da gurun warachka 2014'),
                        _m('expected horizon', sig.get('expected_horizon', '')),
                        _m('engine confidence', fmt(sig.get('confidence'), 1))],
        commentary=f'the momentum screener fired at {sig.get("confidence")} confidence: 12-1 '
                   f'momentum of {sig.get("mom_12_1_pct")}%, {prox_text} of the way to its 52 week '
                   f'high and {sig.get("relative_mom_vs_spy_pct")}% ahead of SPY. the strategy is '
                   f'on probation after a weak 5y backtest, so weigh it accordingly.{gate_note}',
        data_status='partial' if gate_note else 'ok',
        error='engine 52 week high gate could not be evaluated' if gate_note else None)


def _dividend_scaled_100x(ctx, engine_pct):
    '''
    has a dividend yield been multiplied by a hundred it did not need.

    yfinance used to return dividendYield as a fraction (0.035) and now
    returns it as a percentage (3.5). code that still multiplies by a hundred
    therefore reports a 0.7% payer as a 71% one. this compares the figure in
    hand against the raw field and says plainly whether the two differ by
    exactly the factor that mistake produces.
    '''
    raw = ctx.info.get('dividendYield')
    if raw is None or engine_pct is None:
        return False
    try:
        raw, engine_pct = float(raw), float(engine_pct)
    except (TypeError, ValueError):
        return False
    return raw > 0 and abs(engine_pct - raw * 100.0) < max(0.05, raw * 0.02)


def _factor_section(ctx, strategy, key, label, blurb):
    '''shared body for the two research factors, both cross-sectional models'''
    sig = next((s for s in ctx.factor_signals if s.get('strategy') == strategy), None)

    #the shareholder yield engine guesses the units of dividendYield with a
    #"below one means it is a fraction" rule. that rule is right for a 3.5%
    #payer and wrong for every company yielding under one percent, which then
    #arrives as a yield in the seventies and carries the confidence score up
    #with it. a signal built on that number is not a signal
    if sig and strategy == 'shareholder_yield' \
            and _dividend_scaled_100x(ctx, sig.get('dividend_yield_pct')):
        real = float(ctx.info.get('dividendYield'))
        return ReportSection(
            key, label, data_status='unavailable',
            metrics=[_m('engine dividend yield', fmt(sig.get('dividend_yield_pct'), 1, '%'),
                        'wrong by a factor of one hundred'),
                     _m('actual dividend yield', fmt(real, 2, '%'), 'straight from the source'),
                     _m('engine total yield', fmt(sig.get('shareholder_yield_pct'), 1, '%')),
                     _m('engine confidence', fmt(sig.get('confidence'), 1),
                        'built on the inflated figure, not usable')],
            error='dividend yield inflated one hundred fold by a unit conversion',
            commentary=f'the shareholder yield engine read {ctx.ticker} as yielding '
                       f'{sig.get("dividend_yield_pct"):.1f}%. it yields {real:.2f}%. yfinance used '
                       f'to hand out this field as a fraction and now hands it out as a percentage, '
                       f'and the engine still multiplies by a hundred whenever the number is below '
                       f'one - which is every company paying under one percent. the confidence '
                       f'score is built on the inflated figure, so it is reported as unavailable '
                       f'rather than passed on. this affects the live tab, not just this report.')

    place = report_rank.placement(key, sig.get('confidence') if sig else None, ctx.ticker)

    metrics = []
    if sig:
        for k, lbl, dp, sfx in [('confidence', 'engine confidence', 1, ''),
                                ('quality_score', 'quality score', 1, ''),
                                ('profit_margin_pct', 'profit margin', 1, '%'),
                                ('roe_pct', 'return on equity', 1, '%'),
                                ('shareholder_yield_pct', 'total shareholder yield', 1, '%'),
                                ('dividend_yield_pct', 'dividend yield', 1, '%'),
                                ('buyback_yield_pct', 'buyback yield', 1, '%')]:
            if sig.get(k) is not None:
                metrics.append(_m(lbl, fmt(sig[k], dp, sfx)))
        metrics.append(_m('reason', sig.get('reason', '')))

    if place:
        metrics.append(_m('cross-sectional rank', f'{place["rank"]} of {place["n_scored"]}',
                          f'universe of {place["universe_size"]} scanned'))
        metrics.append(_m('percentile', fmt(place['percentile'], 1, '%'),
                          f'ranking built {place["age_hours"]}h ago'
                          + (' - STALE' if place['stale'] else '')))

    if not sig:
        return ReportSection(
            key, label, 'NEUTRAL', 0.0,
            metrics=metrics + [_m('factor signal', 'did not qualify')],
            commentary=f'{ctx.ticker} did not clear the {label.lower()} gates, so this factor has '
                       f'nothing to say about it. {blurb} a name failing the gate is not the same '
                       f'as a name scoring badly - it simply is not in this factor universe.')

    #the engine confidence is the model output, the percentile is where that
    #output sits against everything else. use the percentile whenever we have
    #one, because this is a cross-sectional model and a raw score alone
    #overstates what it actually knows
    if place:
        score = clamp((place['percentile'] - 50.0) / 50.0)
        if place['stale']:
            status = 'partial'
            cross = (f'the cross-section behind this percentile is {place["age_hours"]:.0f} hours '
                     f'old, so read the placement as indicative rather than current.')
        else:
            status = 'ok'
            cross = (f'that confidence sits at the {place["percentile"]:.0f}th percentile of the '
                     f'{place["n_scored"]} names that produced a signal in a universe of '
                     f'{place["universe_size"]}, which is what the score above reflects.')
    else:
        score = conf_score(sig.get('confidence'), 60.0, 92.0)
        status = 'partial'
        cross = ('no universe ranking is cached, so this is the engine confidence on its own. '
                 'this is a cross-sectional model and a standalone number overstates it - build '
                 'the ranking from the PDF Report tab for a real percentile.')

    return ReportSection(
        key, label, verdict_for(score), score, metrics,
        commentary=f'{sig.get("reason", "")}. {cross}', data_status=status,
        error=None if status == 'ok' else 'cross-sectional ranking missing or stale')


def adapt_quality(ctx):
    return _factor_section(
        ctx, 'quality', 'quality', 'Quality factor',
        'quality was the second strongest strategy in the 5y backtest at +72% with a 2.5% '
        'chance of loss.')


def adapt_shareholder(ctx):
    return _factor_section(
        ctx, 'shareholder_yield', 'shareholder_yield', 'Shareholder yield',
        'shareholder yield was the strongest strategy in the 5y backtest at +129% with a 0% '
        'chance of loss across 1000 monte carlo reshuffles.')


def adapt_screener(ctx):
    '''
    the screener is a ranking model, not a scoring model. the honest answer
    for one ticker is where its technical score sits in the universe, so with
    no cached cross-section this returns partial rather than dressing the raw
    score up as a verdict.
    '''
    raw = ctx.quick.get('score') if ctx.quick else None
    if ctx.quick and not _engine_price_is_sane(ctx, ctx.quick.get('price')):
        return ReportSection('screener', 'Screener rank', data_status='unavailable',
                             metrics=[_m('raw technical score', fmt(raw, 1),
                                         'computed against an implausible price, not usable')],
                             error='underlying technical score built on an implausible price',
                             commentary='the screener ranks names on the same technical score the '
                                        'technicals block uses, and that score is not trustworthy '
                                        'for this ticker today. ' + INCOMPLETE_BAR_NOTE)
    place = report_rank.placement('screener', raw, ctx.ticker)
    if raw is None:
        return ReportSection('screener', 'Screener rank', data_status='unavailable',
                             error='no technical score for this ticker',
                             commentary='the screener could not score this name at all.')
    if place is None:
        return ReportSection(
            'screener', 'Screener rank', data_status='partial',
            metrics=[_m('raw technical score', fmt(raw, 1)),
                     _m('universe ranking', 'not cached')],
            error='no universe ranking cached',
            commentary=f'{ctx.ticker} scores {raw:+.1f} on the screener scale, but the screener is '
                       f'a cross-sectional model and that number means nothing without the rest of '
                       f'the universe to compare it against. no ranking is cached, so this module '
                       f'reports no score rather than guessing one. build the ranking from the PDF '
                       f'Report tab and it fills in on the next run.')
    score = clamp((place['percentile'] - 50.0) / 50.0)
    return ReportSection(
        'screener', 'Screener rank', verdict_for(score), score,
        metrics=[_m('raw technical score', fmt(raw, 1)),
                 _m('rank', f'{place["rank"]} of {place["n_scored"]}'),
                 _m('percentile', fmt(place['percentile'], 1, '%')),
                 _m('universe median score', fmt(place['median'], 1)),
                 _m('ranking built', f'{place["age_hours"]}h ago'
                    + (' - STALE' if place['stale'] else ''))],
        commentary=f'{ctx.ticker} ranks {place["rank"]} of {place["n_scored"]} on the screener '
                   f'technical score, beating {place["percentile"]:.0f}% of the scanned universe '
                   f'against a median of {place["median"]:+.1f}.',
        data_status='partial' if place['stale'] else 'ok',
        error='ranking is stale' if place['stale'] else None)


def adapt_cortex(ctx):
    '''
    cortex on the latest completed daily bar. the thresholds, the direction
    rule and the exit calendar all come from cortex itself - this only feeds
    it the bar the context already holds, because scan_gaps and check_wings
    are universe-wide, want IG snapshots and write watchlist files.
    '''
    import pandas as pd
    if len(ctx.prices) < 2:
        return ReportSection('cortex', 'Cortex gap-fade', data_status='unavailable',
                             error='not enough bars to measure a gap')
    last = ctx.prices.iloc[-1]
    prev_close = float(ctx.prices['Close'].iloc[-2])
    gap = cortex._pct(prev_close, float(last['Open']))
    bar_date = pd.to_datetime(ctx.prices.index[-1]).strftime('%Y-%m-%d')
    if gap is None:
        return ReportSection('cortex', 'Cortex gap-fade', data_status='unavailable',
                             error='could not measure the opening gap')

    metrics = [_m('bar measured', bar_date),
               _m('previous close', fmt(prev_close, 4)),
               _m('open', fmt(float(last['Open']), 4)),
               _m('opening gap', fmt(gap, 2, '%'),
                  f'window is {cortex.DEFAULT_GAP_MIN}% to {cortex.DEFAULT_GAP_MAX}%')]
    a = abs(gap)
    if a < cortex.DEFAULT_GAP_MIN or a >= cortex.DEFAULT_GAP_MAX:
        return ReportSection(
            'cortex', 'Cortex gap-fade', 'NEUTRAL', 0.0, metrics,
            commentary=f'{ctx.ticker} did not gap into the cortex window on {bar_date}. cortex only '
                       f'trades opening gaps between {cortex.DEFAULT_GAP_MIN}% and '
                       f'{cortex.DEFAULT_GAP_MAX}%, and most days most stocks do not gap at all, '
                       f'so silence here is the normal state.')

    oth = cortex._pct(float(last['Open']), float(last['High'])) or 0.0
    otl = cortex._pct(float(last['Open']), float(last['Low'])) or 0.0
    wing = oth if gap > 0 else abs(otl)
    side = cortex.signal_direction(gap)
    exit_day = cortex.exit_day_for_gap(a)
    metrics += [_m('wing', fmt(wing, 2, '%'),
                   f'continuation off the open, limit {cortex.DEFAULT_WING}%'),
                _m('side', side, 'gap up is faded short, gap down is bought'),
                _m('exit', f'{exit_day} close(s) after entry')]

    if wing > cortex.DEFAULT_WING:
        return ReportSection(
            'cortex', 'Cortex gap-fade', 'NEUTRAL', 0.0, metrics,
            commentary=f'{ctx.ticker} gapped {gap:+.1f}% but then ran {wing:.1f}% further in the '
                       f'gap direction, over the {cortex.DEFAULT_WING}% wing limit. cortex reads '
                       f'that as real information rather than an overreaction and stands aside.')
    #cortex earns roughly +0.09% a trade before costs on the full sample, so
    #the magnitude is deliberately capped low - a fired signal is real but small
    score = clamp((0.25 + 0.25 * min(a / cortex.DEFAULT_GAP_MAX, 1.0))
                  * (1 if side == 'LONG' else -1))
    return ReportSection(
        'cortex', 'Cortex gap-fade', verdict_for(score), score, metrics,
        commentary=f'cortex would fade this: {ctx.ticker} gapped {gap:+.1f}% on {bar_date} and held '
                   f'inside the wing, so the signal is {side}, exiting on the next close. the score '
                   f'is deliberately small - on 2023-2026 FTSE 350 data cortex wins 53% of the time '
                   f'for +0.09% a trade before costs. note this points the opposite way to PEAD by '
                   f'construction whenever the gap is an earnings gap.')


def adapt_vs_spy(ctx):
    '''
    the vs SPY tab is a book-level view, so the single-name analogue is plain
    relative strength, computed off the SPY frame already in the context.
    '''
    if ctx.spy is None or len(ctx.spy) < 60:
        return ReportSection('vs_spy', 'Relative strength vs SPY', data_status='unavailable',
                             error='SPY history unavailable')
    windows = [(21, '1 month'), (63, '3 month'), (126, '6 month'), (252, '12 month')]
    metrics, rels = [], {}
    for days, label in windows:
        r, s = ctx.ret(days), ctx.spy_ret(days)
        rel = None if (r is None or s is None) else r - s
        rels[days] = rel
        metrics.append(_m(label, f'{fmt(r, 1, "%")} vs SPY {fmt(s, 1, "%")}',
                          f'{fmt(rel, 1, "%")} relative'))
    usable = [v for v in rels.values() if v is not None]
    if not usable:
        return ReportSection('vs_spy', 'Relative strength vs SPY', data_status='unavailable',
                             error='not enough overlapping history')
    mean_rel = float(np.mean(usable))
    blend = 0.5 * (rels.get(126) if rels.get(126) is not None else mean_rel) \
        + 0.5 * (rels.get(63) if rels.get(63) is not None else mean_rel)
    score = clamp(blend / 25.0)
    note = ''
    if ctx.currency in ('GBp', 'GBP', 'EUR'):
        note = (f' this name is priced in {ctx.currency} while SPY is in dollars, so part of the '
                f'gap is cable rather than performance.')
    return ReportSection(
        'vs_spy', 'Relative strength vs SPY', verdict_for(score), score, metrics,
        commentary=f'{ctx.ticker} is {blend:+.1f}% against SPY on a blend of the three and six '
                   f'month windows.{note} beating the index is the bar merlin actually cares '
                   f'about, so this is the plainest read in the report.')


def adapt_paper_book(ctx):
    '''
    context only, no weight in the composite. what the AI paper trader
    currently thinks of this name, plus the market regime it is trading in.
    '''
    metrics, lines = [], []
    try:
        book = engine('_load_paper_portfolio')() or {}
        held = [p for p in (book.get('positions') or [])
                if str(p.get('ticker', '')).upper() == ctx.ticker]
        if held:
            p = held[0]
            metrics += [_m('AI trader holds it', 'yes', p.get('strategy', '')),
                        _m('opened', str(p.get('entry_date') or p.get('opened_at') or '')[:10]),
                        _m('entry price', fmt(p.get('entry_price_native'), 4)),
                        _m('unrealised P/L', fmt(p.get('unrealised_pl_pct'), 2, '%'))]
            lines.append(f'the AI paper trader is holding {ctx.ticker} on the '
                         f'{p.get("strategy", "unknown")} strategy.')
        else:
            metrics.append(_m('AI trader holds it', 'no'))
            lines.append(f'the AI paper trader is not holding {ctx.ticker}.')
        closed = [t for t in (book.get('closed_positions') or [])
                  if str(t.get('ticker', '')).upper() == ctx.ticker]
        if closed:
            wins = sum(1 for t in closed if float(t.get('pl_gbp', 0) or 0) > 0)
            metrics.append(_m('past paper trades', f'{len(closed)} closed, {wins} profitable'))
            lines.append(f'it has closed {len(closed)} paper trades in this name, {wins} of them '
                         f'profitable.')
    except Exception as e:
        metrics.append(_m('paper book', f'unavailable ({type(e).__name__})'))
    try:
        regime = engine('get_market_regime')() or {}
        metrics += [_m('market regime', regime.get('regime_label', 'unknown'),
                       (regime.get('explainer') or '')[:90]),
                    _m('VIX', fmt(regime.get('vix_level'), 1)),
                    _m('SPY vs 200dma', fmt(regime.get('spy_pct_vs_200dma'), 1, '%')),
                    _m('favoured strategies',
                       ', '.join(regime.get('favoured_strategies') or []) or 'none'),
                    _m('suppressed strategies',
                       ', '.join(regime.get('suppressed_strategies') or []) or 'none')]
        lines.append(f'the wider market is in a {regime.get("regime_label", "unknown")} regime.')
    except Exception as e:
        metrics.append(_m('market regime', f'unavailable ({type(e).__name__})'))
    return ReportSection('paper_book', 'Book and regime context', 'NEUTRAL', None, metrics,
                         commentary=' '.join(lines) + ' this block carries no weight in the '
                                    'composite - it is position context, not a signal.')
