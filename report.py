'''
report.py
single-ticker pdf research report for merlin 9.1

this module orchestrates, it does not analyse. every number in the finished
pdf comes out of an engine that already existed in app.py or cortex.py, and
this file only decides what order to call them in, how to combine what they
said and where it lands on the page. if you find strategy logic in here it
is a bug.

    WEIGHTS         composite weights, one dict, equal for now on purpose.
                    there is no forward record yet that would justify
                    anything else and inventing conviction weights would only
                    launder a guess into a number

    ANALYSERS       every module in the report, in the order it appears in
                    the pdf, in one list. adding a future tab is one line
                    here plus one adapter in report_adapters.py

    ReportContext   price history, fundamentals and index context, fetched
                    exactly once per run and handed to every adapter, so a
                    dozen modules do not make a dozen separate trips to
                    yfinance for the same ticker

the composite is a weighted mean over the modules that actually answered,
renormalised over available weight, and it is always printed with the count
of how many those were. "+0.42 from 4 of 12" means something completely
different from the same number out of 12 of 12. confidence comes from
agreement between modules rather than from the size of the average, because a
big number from a set of models that flatly contradict each other is a weak
signal and ought to read as one.

app.py registers itself here at startup with set_engine_module, the same
trick cortex uses for its US universe, so this file never imports app.py back
and there is no circular import to trip over.

    layers:  report_adapters.py  contract, helpers, one adapter per module
             report_charts.py    the two matplotlib charts, base64 embedded
             report_rank.py      cross-sectional ranking cache
             templates/report_pdf.html   the page itself
'''

import datetime
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf
from jinja2 import Environment, FileSystemLoader, select_autoescape

import cortex
import earnings
import report_adapters as A
import report_charts
import report_rank


REPORTS_DIR = 'reports'
CONTEXT_TTL_SECONDS = 600      #a repeat run within ten minutes is instant


#composite weights. every module equal until there is a forward record that
#says otherwise - see the phase 2 track_record hook on ReportSection. a key
#missing from here carries no weight and is reported as context only
#10.0: insider activity and insider clusters were both taken out of the
#report. the two adapters are still in report_adapters.py and both still
#work - putting them back is one line here and one line in ANALYSERS below.
#nothing else in merlin changed: the Insider Trading tab and the Insider
#Clusters scanner are untouched, this is only about what goes in the pdf
WEIGHTS = {
    'technicals':        1.0,
    'fundamentals':      1.0,
    'news':              1.0,
    'pead':              1.0,
    'momentum':          1.0,
    'quality':           1.0,
    'shareholder_yield': 1.0,
    'screener':          1.0,
    'cortex':            1.0,
    'vs_spy':            1.0,
}

#every module in the report, in page order. the last one carries no weight
#and appears as context rather than as a signal
ANALYSERS = [
    ('technicals',        'Technicals',               A.adapt_technicals),
    ('fundamentals',      'Fundamentals',             A.adapt_fundamentals),
    ('news',              'News sentiment',           A.adapt_news),
    #insider activity and insider clusters removed in 10.0, see WEIGHTS above
    ('pead',              'Earnings drift (PEAD)',    A.adapt_pead),
    ('momentum',          'Relative momentum',        A.adapt_momentum),
    ('quality',           'Quality factor',           A.adapt_quality),
    ('shareholder_yield', 'Shareholder yield',        A.adapt_shareholder),
    ('screener',          'Screener rank',            A.adapt_screener),
    ('cortex',            'Cortex gap-fade',          A.adapt_cortex),
    ('vs_spy',            'Relative strength vs SPY', A.adapt_vs_spy),
    ('paper_book',        'Book and regime context',  A.adapt_paper_book),
]


def set_engine_module(mod):
    '''app.py hands over its own module object at startup'''
    A.set_engine_module(mod)


def normalise_ticker(raw):
    '''
    the same normalisation the analyse tab uses - trim and upper - plus a
    guard against anything that is obviously not a ticker, so a typo comes
    back as a clear message rather than an empty yfinance frame later on.
    '''
    t = (raw or '').strip().upper()
    if not t:
        raise ValueError('enter a ticker first')
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,14}', t):
        raise ValueError(f'{t} does not look like a ticker symbol')
    return t


# ==================== SHARED CONTEXT ====================

@dataclass
class ReportContext:
    ticker: str
    info: dict
    prices: pd.DataFrame
    spy: pd.DataFrame
    quick: dict
    fundamentals: dict
    factor_signals: list
    name: str
    sector: str
    industry: str
    currency: str
    exchange: str
    market_cap: float
    index_note: str
    is_lse: bool
    fetched_at: str
    #yfinance sometimes appends a final daily row with volume but no OHLC.
    #this context drops it, but engines that fillna(0) instead read the last
    #close as zero, so the report has to know when that trap is armed
    incomplete_last_bar: bool = False
    #the earnings calendar, so the PEAD block can tell "no drift window open"
    #apart from "no earnings data exists". those are the same silence from
    #the engine and they mean completely different things
    earnings_rows: int = 0
    earnings_last: str = None
    earnings_days_ago: int = None
    earnings_next: str = None
    notes: list = field(default_factory=list)

    @property
    def close(self):
        return self.prices['Close'].astype(float)

    @property
    def price(self):
        return float(self.close.iloc[-1])

    def ret(self, days):
        '''plain descriptive total return over n trading days, or None'''
        c = self.close
        if len(c) <= days:
            return None
        old = float(c.iloc[-days - 1])
        return None if old <= 0 else (float(c.iloc[-1]) / old - 1) * 100.0

    def spy_ret(self, days):
        if self.spy is None or len(self.spy) <= days:
            return None
        c = self.spy['Close'].astype(float)
        old = float(c.iloc[-days - 1])
        return None if old <= 0 else (float(c.iloc[-1]) / old - 1) * 100.0

    #10.0 earnings diary. every adapter gets the context, so this is the one
    #place any module in the report can ask "how close is the print" without
    #going back to yfinance for an answer the context already fetched
    @property
    def days_to_next_earnings(self):
        '''calendar days to the next scheduled result, or None if unknown'''
        if not self.earnings_next:
            return None
        try:
            return int((pd.Timestamp(self.earnings_next).normalize()
                        - pd.Timestamp.now().normalize()).days)
        except Exception:
            return None

    @property
    def earnings_line(self):
        '''
        the one-line version the pdf header prints.

        a report read the morning of a print is a different document from the
        same report read a month out, and until now nothing on the page said
        which one you were holding
        '''
        n = self.days_to_next_earnings
        if n is None:
            return 'next earnings unknown'
        if n == 0:
            return f'reports today ({self.earnings_next})'
        if n == 1:
            return f'reports tomorrow ({self.earnings_next})'
        return f'next earnings in {n} days ({self.earnings_next})'

    @property
    def earnings_imminent(self):
        '''true inside a week of a print, which is when the header shouts'''
        n = self.days_to_next_earnings
        return n is not None and 0 <= n <= 7


_ctx_cache = {}
_ctx_lock = threading.Lock()


def _flatten(df):
    if df is not None and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _naive_index(df):
    try:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    except (TypeError, ValueError):
        df.index = pd.to_datetime(df.index)
    return df


def build_context(ticker, progress=None):
    '''
    fetch once, feed everything.

    the price frame, the info blob, SPY, the screener technicals and the two
    research factor signals are all pulled here and handed to the adapters.
    no adapter touches yfinance directly - grep report_adapters.py and check.
    '''
    step = progress or (lambda msg: None)
    with _ctx_lock:
        hit = _ctx_cache.get(ticker)
        if hit and (time.time() - hit[0]) < CONTEXT_TTL_SECONDS:
            return hit[1]

    step('fetching price history')
    prices = _flatten(yf.download(ticker, period='2y', interval='1d',
                                  progress=False, auto_adjust=False))
    if prices is None or prices.empty or len(prices) < 30:
        raise ValueError(f'no usable price history for {ticker} - check the symbol. '
                         f'london listings need a .L suffix, for example SHEL.L')
    incomplete_last_bar = bool(pd.isna(prices['Close'].iloc[-1]))
    prices = _naive_index(prices.dropna(subset=['Close']))

    step('fetching fundamentals')
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    step('fetching SPY benchmark')
    try:
        spy = _flatten(yf.download('SPY', period='2y', interval='1d',
                                   progress=False, auto_adjust=True))
        #the benchmark needs the same incomplete-bar treatment as the ticker.
        #without it SPY's last close is nan, every relative return comes out
        #nan, and the vs SPY block quietly reports a flat zero for everything
        if spy is not None and not spy.empty:
            spy = _naive_index(spy.dropna(subset=['Close']))
        else:
            spy = None
    except Exception:
        spy = None

    notes = []
    step('running technicals')
    try:
        quick = A.engine('quick_score_stock')(ticker) or {}
    except Exception as e:
        quick, _ = {}, notes.append(f'technicals engine failed: {type(e).__name__}')
    try:
        fundamentals = A.engine('get_fundamentals')(ticker) or {}
    except Exception as e:
        fundamentals = {'error': str(e)[:120]}
        notes.append(f'fundamentals engine failed: {type(e).__name__}')

    step('running factor signals')
    try:
        factor_signals = A.engine('get_research_factor_signals')(ticker, 0) or []
    except Exception as e:
        factor_signals, _ = [], notes.append(f'factor engine failed: {type(e).__name__}')

    #the earnings calendar, read from the same field the PEAD engine now uses.
    #the earnings_dates endpoint went stale - it returns nothing newer than a
    #year old for any ticker - so info['earningsTimestamp'] is the source of
    #truth and that endpoint is only a fallback
    earnings_rows, earnings_last, earnings_days_ago, earnings_next = 0, None, None, None
    try:
        ts = info.get('earningsTimestamp')
        if ts and float(ts) > 1e9:
            stamp = pd.Timestamp(datetime.datetime.fromtimestamp(float(ts)))
            #the same field carries the last report for some tickers and the
            #next one for others. a future date is a diary entry, not an event
            if stamp <= pd.Timestamp.now():
                earnings_rows = 1
                earnings_last = stamp.strftime('%Y-%m-%d')
                earnings_days_ago = int((pd.Timestamp.now() - stamp).days)
            else:
                earnings_next = stamp.strftime('%Y-%m-%d')
    except Exception:
        pass
    #10.0: the shared earnings module gets the same info blob for free, and
    #it is date-led rather than field-led, so it fills in what the block
    #above cannot. the common case it rescues is a ticker whose
    #earningsTimestamp is the LAST report - apple hands back a july date, so
    #the code above learns the last result and nothing at all about the next
    #one, while calendar['Earnings Date'] has known the october date all
    #along. the pdf header prints the next date and the pead adapter tells
    #"no drift window open" apart from "no calendar at all" using these, so
    #both get better the moment both sides are filled in
    try:
        earnings.prime_from_info(ticker, info)
        row = earnings.lookup(ticker)
        if not earnings_next and row.get('next'):
            earnings_next = row['next']
        if not earnings_last and row.get('last'):
            earnings_last = row['last']
            earnings_days_ago = (pd.Timestamp.now() - pd.Timestamp(row['last'])).days
            #the pead adapter reads rows==0 as "no calendar exists", so a
            #last date arriving from here has to bring its count with it
            earnings_rows = max(earnings_rows, 1)
    except Exception:
        pass

    #only reach for the stale endpoint when nothing is known at all. if a
    #future date came back we already know the calendar, there simply is
    #no completed result to drift from
    if earnings_days_ago is None and earnings_next is None:
        step('fetching earnings calendar')
        try:
            ed = yf.Ticker(ticker).earnings_dates
            if ed is not None and len(ed):
                earnings_rows = len(ed)
                idx = pd.to_datetime(ed.index)
                idx = idx.tz_localize(None) if idx.tz is not None else idx
                past = idx[idx <= pd.Timestamp.now()]
                if len(past):
                    earnings_last = past.max().strftime('%Y-%m-%d')
                    earnings_days_ago = int((pd.Timestamp.now() - past.max()).days)
        except Exception:
            pass

    ccy = info.get('currency') or ('GBp' if ticker.endswith('.L') else 'USD')
    index_note = 'not in a merlin universe'
    try:
        if ticker.endswith('.L'):
            plain = ticker[:-2].replace('.', '-')
            index_note = ('FTSE 350 (cortex universe)' if plain in cortex.LSE_UNIVERSE
                          else 'London listed, outside the cortex FTSE 350 list')
        elif ticker in set(A.engine('get_stock_universe')(500)):
            index_note = 'merlin US universe'
    except Exception:
        pass

    ctx = ReportContext(
        ticker=ticker, info=info, prices=prices, spy=spy, quick=quick,
        fundamentals=fundamentals, factor_signals=factor_signals,
        name=info.get('shortName') or info.get('longName') or fundamentals.get('name') or ticker,
        sector=info.get('sector') or fundamentals.get('sector') or 'unknown',
        industry=info.get('industry') or fundamentals.get('industry') or 'unknown',
        currency=ccy,
        exchange=info.get('fullExchangeName') or info.get('exchange') or 'unknown',
        market_cap=info.get('marketCap') or 0,
        index_note=index_note, is_lse=ticker.endswith('.L'),
        fetched_at=datetime.datetime.now().isoformat(timespec='seconds'),
        incomplete_last_bar=incomplete_last_bar,
        earnings_rows=earnings_rows, earnings_last=earnings_last,
        earnings_days_ago=earnings_days_ago, earnings_next=earnings_next, notes=notes,
    )
    if incomplete_last_bar:
        ctx.notes.append(
            'yfinance returned an incomplete final daily bar (volume but no open, high, low or '
            'close). this report drops it, but engines that fill missing values with zero read '
            'the latest price as zero and quietly decline the ticker - see the factor blocks')
    with _ctx_lock:
        _ctx_cache[ticker] = (time.time(), ctx)
    return ctx


# ==================== SECTIONS AND COMPOSITE ====================

#which report module corresponds to which logged strategy. only the ones that
#actually emit dated signals appear here - technicals, fundamentals, news and
#the rest are readings rather than calls, so there is nothing to score them
#against and they correctly carry no track record.
#10.0: 'clusters' came out with the module itself
TRACK_RECORD_KEYS = {'momentum': 'momentum', 'quality': 'quality',
                     'shareholder_yield': 'shareholder_yield', 'pead': 'pead',
                     'cortex': 'cortex'}


def _track_record_for(key):
    '''
    the module's own forward hit rate, once there is one.

    returns None until enough measured signals exist, so a verdict never
    arrives wearing a track record built on four trades.
    '''
    strategy = TRACK_RECORD_KEYS.get(key)
    if not strategy:
        return None
    try:
        import signal_store
        return signal_store.track_record(strategy)
    except Exception:
        return None


def _run_section(key, label, fn, ctx):
    '''fault isolation. a module that dies loses its own block and nothing else'''
    try:
        section = fn(ctx)
        if section.score is None and section.verdict not in ('NEUTRAL', 'UNAVAILABLE'):
            section.verdict = 'UNAVAILABLE'
        if section.track_record is None:
            section.track_record = _track_record_for(key)
        return section
    except Exception as e:
        return A.ReportSection(key, label, 'UNAVAILABLE', None, [],
                               commentary='this module could not be run for this ticker. the rest '
                                          'of the report is unaffected.',
                               data_status='unavailable',
                               error=f'{type(e).__name__}: {str(e)[:180]}')


def section_dict(s):
    weight = WEIGHTS.get(s.key, 0.0)
    return {'key': s.key, 'label': s.label, 'verdict': s.verdict,
            'score': None if s.score is None else round(float(s.score), 3),
            'metrics': s.metrics, 'commentary': s.commentary,
            'data_status': s.data_status, 'error': s.error,
            'track_record': s.track_record,
            'weight': weight, 'in_composite': weight > 0}


def _sign(x, dead=0.05):
    if x is None or abs(x) < dead:
        return 0
    return 1 if x > 0 else -1


def build_composite(sections):
    '''weighted mean over the modules that answered, plus how much they agree'''
    scored = [s for s in sections if s.score is not None and WEIGHTS.get(s.key, 0) > 0]
    total_modules = len(WEIGHTS)
    if not scored:
        return {'score': None, 'verdict': 'UNAVAILABLE', 'confidence': 'none',
                'n_contributing': 0, 'n_modules': total_modules, 'agreement_pct': None,
                'n_directional': 0, 'n_silent': 0,
                'disagreements': [], 'available_weight_pct': 0.0,
                'summary': f'no module returned a score out of {total_modules}'}

    weight = sum(WEIGHTS[s.key] for s in scored)
    composite = sum(s.score * WEIGHTS[s.key] for s in scored) / weight
    csign = _sign(composite)

    #agreement is measured over the modules that actually took a side. a
    #module sitting on zero because its edge has nothing to say today - no
    #gap for cortex, no earnings window for PEAD - is not disagreeing with
    #anything, and counting it as dissent would cap confidence below high
    #for structural reasons that have nothing to do with this ticker. those
    #zeroes still pull the composite toward neutral in the mean above, which
    #is the honest place for them to bite
    directional = [s for s in scored if _sign(s.score) != 0]
    silent = len(scored) - len(directional)
    agree = sum(1 for s in directional if csign != 0 and _sign(s.score) == csign)
    against = [s for s in directional if csign != 0 and _sign(s.score) == -csign]
    agreement_pct = round(agree / len(directional) * 100, 1) if directional else None

    if csign == 0 or not directional:
        confidence = 'low'
    elif agreement_pct >= 70:
        confidence = 'high'
    elif agreement_pct >= 50:
        confidence = 'medium'
    else:
        confidence = 'low'

    direction = 'up' if composite > 0 else 'down'
    disagreements = [{
        'label': s.label, 'score': round(s.score, 3), 'verdict': s.verdict,
        'note': f'{s.label.lower()} points {"up" if s.score > 0 else "down"} while the composite '
                f'points {direction}.',
        'expected': False,
    } for s in against]

    #PEAD rides an earnings gap, cortex fades one. on the same event they must
    #point opposite ways - that is the models working, not a fault
    pead = next((s for s in scored if s.key == 'pead'), None)
    ctx_cortex = next((s for s in scored if s.key == 'cortex'), None)
    if pead and ctx_cortex and _sign(pead.score) != 0 \
            and _sign(pead.score) == -_sign(ctx_cortex.score):
        disagreements.append({
            'label': 'PEAD vs Cortex', 'score': None, 'verdict': 'BY DESIGN',
            'note': 'PEAD rides an earnings gap and cortex fades one, so on the same event they '
                    'point opposite ways by construction. this pair contradicting each other is '
                    'expected behaviour rather than a fault - only one of them can be right about '
                    'this particular gap, and the report cannot tell you which.',
            'expected': True,
        })

    return {
        'score': round(composite, 3), 'verdict': A.verdict_for(composite),
        'confidence': confidence, 'n_contributing': len(scored),
        'n_modules': total_modules, 'agreement_pct': agreement_pct,
        'n_directional': len(directional), 'n_silent': silent,
        'available_weight_pct': round(weight / sum(WEIGHTS.values()) * 100, 1),
        'disagreements': disagreements,
        'summary': f'composite {composite:+.2f} from {len(scored)} of {total_modules} modules',
    }


# ==================== PDF RENDERING ====================

def _gtk_present():
    '''
    is the GTK runtime weasyprint needs actually on this machine.

    asked before importing weasyprint rather than after, because the import
    itself is what prints a wall of installation advice to the console on a
    windows box without it. an earlier version of this file swapped the
    stdout and stderr file descriptors to silence that. do not do that: on
    windows the swap can fail with "incorrect function" depending on what
    stdout is attached to, and a half-restored descriptor takes the browser
    subprocess down with it, which turns a cosmetic problem into a broken
    renderer. asking first costs nothing and breaks nothing.
    '''
    import ctypes.util
    return any(ctypes.util.find_library(name) for name in
               ('libgobject-2.0-0', 'gobject-2.0-0', 'gobject-2.0', 'libgobject-2.0.so.0'))


def _pdf_weasyprint(html, out_path, meta):
    if not _gtk_present():
        raise RuntimeError('GTK runtime not installed, weasyprint cannot render here')
    from weasyprint import HTML
    HTML(string=html, base_url=os.getcwd()).write_pdf(out_path)


def _pdf_playwright(html, out_path, meta):
    '''
    chromium print-to-pdf. header and footer templates are deliberately off:
    current chromium renders the @page margin boxes in the stylesheet, so
    turning its own footer on as well puts two overlapping footers on every
    page. the template owns the footer, prefer_css_page_size lets it own the
    margins too, so one file decides how the page looks.
    '''
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        #the default launch uses playwright's headless shell build. antivirus
        #products quarantine that one far more readily than the full browser,
        #so try the complete chromium before giving up on the renderer. the
        #first failure is kept and re-raised alongside the second, because
        #losing it makes this exact problem impossible to diagnose
        #playwright's own headless shell first, then its full chromium, then
        #whatever chrome the machine already has. an antivirus that eats one
        #of these rarely eats all three, and the installed chrome is the one
        #it is least likely to touch
        attempts, browser = [], None
        for kwargs in ({}, {'channel': 'chromium'}, {'channel': 'chrome'}):
            try:
                browser = p.chromium.launch(**kwargs)
                break
            except Exception as e:
                attempts.append(f'{kwargs.get("channel", "headless shell")} -> '
                                f'{type(e).__name__}: {e}')
        if browser is None:
            raise RuntimeError('every chromium launch failed. ' + ' || '.join(attempts))
        try:
            page = browser.new_page()
            page.set_content(html, wait_until='load')
            page.emulate_media(media='print')
            page.pdf(path=out_path, format='A4', print_background=True,
                     prefer_css_page_size=True, display_header_footer=False)
        finally:
            browser.close()


def _pdf_pdfkit(html, out_path, meta):
    import pdfkit
    pdfkit.from_string(html, out_path,
                       options={'quiet': '', 'enable-local-file-access': None})


RENDERERS = [('weasyprint', _pdf_weasyprint),
             ('playwright', _pdf_playwright),
             ('pdfkit', _pdf_pdfkit)]

_renderer_used = None


def render_pdf(html, out_path, meta):
    '''
    weasyprint first for css fidelity, playwright second because it ships its
    own chromium and needs nothing from the system, pdfkit last. the winner is
    remembered so a windows box without the GTK runtime does not retry
    weasyprint on every report just to fail the same way each time.
    '''
    global _renderer_used
    order = RENDERERS
    if _renderer_used:
        order = ([r for r in RENDERERS if r[0] == _renderer_used] +
                 [r for r in RENDERERS if r[0] != _renderer_used])
    failures = []
    for name, fn in order:
        try:
            fn(html, out_path, meta)
            _renderer_used = name
            return name
        except Exception as e:
            failures.append(f'{name}: {type(e).__name__}: {str(e)[:400]}')
    #the renderer tells you exactly what is wrong and the status bar in the
    #browser truncates it, so put the whole chain on the console and in a log
    #file. a renderer failure is nearly always environmental - a missing
    #browser, an antivirus quarantine - and none of that is diagnosable from
    #a sentence clipped at two hundred characters
    print('\n[report] every pdf renderer failed:')
    for line in failures:
        print(f'  {line}')
    try:
        os.makedirs('data', exist_ok=True)
        with open(os.path.join('data', 'report_render_error.log'), 'a', encoding='utf-8') as fh:
            fh.write(f'\n=== {datetime.datetime.now().isoformat(timespec="seconds")} '
                     f'{meta.get("ticker", "?")} ===\n')
            for line in failures:
                fh.write(line + '\n')
    except Exception:
        pass
    #lead with the thing to actually do about it. the detail follows, but the
    #first line has to survive being truncated in a status bar
    raise RuntimeError(
        'no pdf renderer available - the full error is in the merlin console. tried '
        + ' | '.join(failures))


_jinja = Environment(loader=FileSystemLoader('templates'),
                     autoescape=select_autoescape(['html']))


def build_html(payload):
    return _jinja.get_template('report_pdf.html').render(**payload)


# ==================== ORCHESTRATION ====================

def generate_report(ticker, progress=None):
    '''
    build one report for one ticker and return the payload the template eats.

    deliberately free of single-ticker assumptions beyond the argument, so
    batching over the portfolio holdings later is a loop around this call
    rather than a rewrite.
    '''
    ticker = normalise_ticker(ticker)
    total = len(ANALYSERS)
    step = progress or (lambda *a, **k: None)

    step('shared data', 0, total)
    ctx = build_context(ticker, progress=lambda msg: step(msg, 0, total))

    sections = []
    for i, (key, label, fn) in enumerate(ANALYSERS):
        step(label, i, total)
        sections.append(_run_section(key, label, fn, ctx))
    step('composing', total, total)

    composite = build_composite(sections)
    dicts = [section_dict(s) for s in sections]
    charts = {
        'price': report_charts.price_chart_png(ctx.prices, ticker, ctx.currency),
        'factor': report_charts.factor_chart_png([d for d in dicts if d['in_composite']]),
    }
    unavailable = [{'label': d['label'], 'error': d['error'] or 'no data'}
                   for d in dicts if d['data_status'] != 'ok']
    now = datetime.datetime.now()
    sym = A.currency_symbol(ctx.currency)
    return {
        'ticker': ticker, 'name': ctx.name, 'sector': ctx.sector, 'industry': ctx.industry,
        'currency': ctx.currency, 'currency_symbol': sym,
        'price': f'{sym}{ctx.price:,.2f}',
        'market_cap': A.engine('format_market_cap')(ctx.market_cap) if ctx.market_cap else 'n/a',
        'exchange': ctx.exchange, 'index_note': ctx.index_note,
        'generated': now.strftime('%Y-%m-%d %H:%M'),
        'generated_full': now.strftime('%d %B %Y at %H:%M'),
        'price_asof': pd.to_datetime(ctx.prices.index[-1]).strftime('%Y-%m-%d'),
        'bars': len(ctx.prices),
        'composite': composite, 'sections': dicts, 'charts': charts,
        'unavailable': unavailable, 'context_notes': ctx.notes,
        'weights': WEIGHTS, 'ranking_status': report_rank.ranking_status(),
        #10.0: the earnings diary in the header. a report read the morning of
        #a print is a different document from the same one read a month out
        'earnings_line': ctx.earnings_line,
        'earnings_next': ctx.earnings_next,
        'earnings_days': ctx.days_to_next_earnings,
        'earnings_imminent': ctx.earnings_imminent,
        'merlin_version': '10.0',
    }


# ==================== JOB STORE ====================

_jobs = {}
_jobs_lock = threading.Lock()
MAX_JOBS_KEPT = 40


def _set_job(job_id, **fields):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)


def _run_job(job_id, ticker):
    def progress(label, done, total):
        _set_job(job_id, current_module=label, done=done, total=total,
                 message=f'running {label} ({min(done + 1, total)} of {total})')
    try:
        _set_job(job_id, state='running', message='fetching shared data')
        payload = generate_report(ticker, progress=progress)
        _set_job(job_id, state='rendering', message='rendering the pdf')

        os.makedirs(REPORTS_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M')
        filename = f'{re.sub(r"[^A-Za-z0-9._-]", "_", payload["ticker"])}_{stamp}.pdf'
        path = os.path.join(REPORTS_DIR, filename)
        renderer = render_pdf(build_html(payload), path,
                              {'ticker': payload['ticker'],
                               'generated': payload['generated']})
        _set_job(job_id, state='done', message='report ready', path=path,
                 filename=filename, renderer=renderer, done=len(ANALYSERS),
                 composite=payload['composite']['score'],
                 verdict=payload['composite']['verdict'],
                 contributing=payload['composite']['n_contributing'],
                 n_modules=payload['composite']['n_modules'])
    except Exception as e:
        _set_job(job_id, state='error', error=f'{type(e).__name__}: {str(e)[:900]}',
                 message='report failed')


def start_job(raw_ticker):
    '''validate, then kick the run off in its own thread and hand back an id'''
    ticker = normalise_ticker(raw_ticker)
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        if len(_jobs) >= MAX_JOBS_KEPT:
            for old in sorted(_jobs, key=lambda k: _jobs[k]['created_at'])[:10]:
                _jobs.pop(old, None)
        _jobs[job_id] = {'job_id': job_id, 'ticker': ticker, 'state': 'queued',
                         'current_module': '', 'done': 0, 'total': len(ANALYSERS),
                         'message': 'queued', 'error': None, 'path': None,
                         'filename': None, 'renderer': None,
                         'created_at': datetime.datetime.now().isoformat(timespec='seconds')}
    threading.Thread(target=_run_job, args=(job_id, ticker), daemon=True).start()
    return job_id


def job_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def recent_reports(limit=20):
    '''everything on disk, newest first, so a re-download survives a restart'''
    if not os.path.isdir(REPORTS_DIR):
        return []
    out = []
    for fn in os.listdir(REPORTS_DIR):
        if not fn.lower().endswith('.pdf'):
            continue
        try:
            stat = os.stat(os.path.join(REPORTS_DIR, fn))
        except OSError:
            continue
        out.append({'filename': fn, 'ticker': fn.split('_')[0],
                    'generated': datetime.datetime.fromtimestamp(stat.st_mtime)
                                                  .strftime('%Y-%m-%d %H:%M'),
                    'size_kb': round(stat.st_size / 1024, 1),
                    'url': f'/api/report/file/{fn}'})
    out.sort(key=lambda r: r['generated'], reverse=True)
    return out[:limit]


def report_path(filename):
    '''resolve a filename inside reports/ only, never anywhere else on disk'''
    safe = os.path.basename(filename or '')
    if not safe.lower().endswith('.pdf'):
        return None
    full = os.path.abspath(os.path.join(REPORTS_DIR, safe))
    if not full.startswith(os.path.abspath(REPORTS_DIR) + os.sep):
        return None
    return full if os.path.exists(full) else None
