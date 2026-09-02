'''
cortex.py
gap-and-fade engine for merlin 9.0

the strategy in one paragraph: find stocks that gap hard at the open and then
do NOT carry on in the gap direction during the session. the gap was an
overreaction, so fade it. gap up means short, gap down means long. entry is
at the day-0 close, exit is on a calendar (default the next close) rather
than a trailing stop, which is what makes cortex structurally different from
every other merlin strategy.

two runs a day per market because the two filters can only be measured at
different times:

  morning run   (at the open)          - measures the gap, builds a watchlist
  afternoon run (just before the close) - measures the wing, opens the trades

9.0 change: IG is now the primary live data source, not yfinance. IG snapshots
are real time, so the morning run can go at 08:00 on the bell instead of
waiting out a 20 minute yahoo delay, and the afternoon run can go at 16:25
which is late enough to be honest about the day high and low but early enough
that you could actually place the trade into the closing auction.

snapshots are batched through /markets?epics= (up to 50 at a time) rather than
one call per ticker, so a full FTSE 350 scan is about 20 seconds rather than
the 15 minutes the original CFD project needed.

what IG does NOT do here: history. the backtester and the sweep stay on
yfinance, because IG's spread bet daily bars record a DFB rollover open rather
than the LSE auction open, which is the exact thing that made the original
spread bet cache undercount gaps. so live signals and backtested signals come
from different sources. that is a real limitation, not an oversight - see
compare_sources() for a way to keep an eye on how far apart they drift.

honest note baked into the defaults: on 2023-2026 FTSE 350 data this wins 53%
of the time for +0.09% a trade before costs. it looked like 67% on a five
month 2026 sample, which is where the original research came from. the only
two settings that held up across both halves were exiting on D1 and raising
the gap floor towards 7-8%. treat everything else as unproven.
'''

import os
import csv
import json
import time
import datetime
import threading

import numpy as np
import pandas as pd
import yfinance as yf

import earnings


# ==================== CONFIG ====================

DATA_DIR       = 'data'
CACHE_DIR      = os.path.join(DATA_DIR, 'cortex_cache')
PAPER_PATH     = os.path.join(DATA_DIR, 'cortex_paper.json')
WATCHLIST_PATH = os.path.join(DATA_DIR, 'cortex_watchlist.json')
SIGNAL_LOG     = os.path.join(DATA_DIR, 'cortex_signals.csv')

#defaults match the original research so results stay comparable. the sweep
#tab is where you find out whether they are actually the right numbers
DEFAULT_GAP_MIN  = 6.0    #minimum absolute gap at the open, in percent
DEFAULT_GAP_MAX  = 15.0   #above this it is usually a corporate action, skip
DEFAULT_WING     = 3.0    #max continued move in the gap direction, off the open
DEFAULT_EXIT_DAY = 1      #close on the Nth close after entry
BORDERLINE_BAND  = 1.0    #wing between wing and wing+this is logged, not traded

#original bucket exit table from the CFD research. kept so it can be compared
#against flat D1, which beat it on the full sample. off by default.
BUCKET_EXITS = [(6, 8, 1), (8, 9, 3), (9, 10, 2), (10, 15, 1)]

#10.0: how many calendar days either side of a reported result still counts
#as an earnings gap. one, so a company reporting before the london open is
#caught on the day and one reporting after the us close is caught the
#morning after. see EARNINGS NOTE above scan_gaps for what this is for
EARNINGS_GAP_WINDOW = 1

#paper book. flat stake per trade, no cap on how many run at once, which is
#what makes this a pure "does the signal work" test rather than a sizing test
# ==================== WHAT A TRADE COSTS ====================
#
# 10.0. cortex is a SPREAD BET book on IG, at five to one, and this is the
# model for that. it has been through two wrong versions to get here and both
# are worth knowing about, because the mistakes are the ordinary ones.
#
#   v1  one flat 0.25% round trip on everything. wrong because it charged a
#       US trade the same as an LSE one and a short the same as a long.
#   v2  a trading 212 SHARE model - spread, currency conversion, CFD markup
#       for shorts. wrong because we are not buying shares. there is no
#       trading 212 account in this at all.
#
# what a spread bet actually costs on IG, and what it does not:
#
#   no commission        IG's charge on a share spread bet is the dealing
#                        spread. there is no separate commission.
#   no stamp duty        spread bets are exempt. buying the share would cost
#                        0.5% on every UK purchase, so this is a real saving
#                        the share model was never crediting us with.
#   no currency fee      the bet is in pounds per point and settles in
#                        pounds, so a US name pays no conversion either way.
#                        the v2 model was charging 0.30% the round trip for
#                        this and it does not exist here.
#   the spread           paid twice, in and out. this is nearly the whole
#                        cost of a one to three day hold, and it is measured
#                        live where we can - see the note on SPREAD below.
#   overnight funding    charged EVERY night on the FULL exposure, and this
#                        is the piece the share model had backwards. a spread
#                        bet is a financed position, so:
#                            long  pays    (benchmark + IG's admin fee) / 365
#                            short receives (benchmark - IG's admin fee) / 365
#                        with the benchmark above the admin fee a short is
#                        CREDITED for holding overnight. v2 charged shorts
#                        financing and longs none, which is exactly backwards.
#
# LEVERAGE. five to one means the £100 committed to a position controls £500
# of exposure. it does not change the percentage move - a 1% gap fade is 1%
# whether the exposure is £100 or £500 - so it does not create edge. what it
# does is multiply the pounds, both ways, and put £500 of costs on £100 of
# committed money. an edge of +0.1% a trade becomes +0.5% on the money down;
# a loss of -0.1% becomes -0.5%. see CORTEX_LEVERAGE below.
#
# ---------------------------------------------------------------------------
# NUMBERS TO CHECK AGAINST YOUR OWN IG ACCOUNT. the funding admin fee and the
# benchmark rates below are the published shape of IG's share funding but
# they move, and the fallback spreads are estimates for when a live quote is
# not available. they are named constants in one place for that reason.
# ---------------------------------------------------------------------------

#what one position commits, and what that commitment controls
STAKE_GBP        = 100.0     #money down per position
CORTEX_LEVERAGE  = 5.0       #so the exposure is five times it
EXPOSURE_GBP     = STAKE_GBP * CORTEX_LEVERAGE

#SPREAD, per side, as a percentage of the exposure.
#
#these are fallbacks. the morning and afternoon runs already pull bid and
#offer from IG and currently throw the spread away - _gather_rows keeps the
#mid and drops the rest. when a live quote is available the real spread is
#recorded on the signal and charged instead of these, which is why they are
#deliberately conservative: a wrong assumption should cost us on paper rather
#than flatter us
IG_SPREAD_LSE_PCT = 0.10
IG_SPREAD_US_PCT  = 0.10

#OVERNIGHT FUNDING. IG's admin fee on share funding, annualised, applied over
#the benchmark for a long and under it for a short
IG_FUNDING_ADMIN_PCT = 2.5

#the overnight benchmark each currency's funding is quoted against, annual.
#GBP is SONIA, USD is the federal funds effective rate
IG_BENCHMARK_GBP_PCT = 4.0
IG_BENCHMARK_USD_PCT = 4.0

#kept under the old name because the sweep, the backtester and the tab all
#print it as the default cost assumption. an LSE long, one night
CORTEX_FEE_PCT = IG_SPREAD_LSE_PCT * 2


def _benchmark_for(market):
    return IG_BENCHMARK_USD_PCT if (market or 'LSE') != 'LSE' else IG_BENCHMARK_GBP_PCT


def spread_cost_pct(market='LSE', measured_pct=None):
    '''
    the dealing spread in and out again, as a percentage of exposure.

    measured_pct is the real spread off an IG quote, (offer - bid) / mid,
    when the run had one. it is used in preference to the fallback, because a
    measured number beats an assumed one every time.
    '''
    if measured_pct is not None and _finite(measured_pct) and measured_pct >= 0:
        return float(measured_pct)
    per_side = IG_SPREAD_LSE_PCT if (market or 'LSE') == 'LSE' else IG_SPREAD_US_PCT
    return per_side * 2


def funding_cost_pct(side='LONG', market='LSE', nights=1):
    '''
    overnight funding for the nights held, as a percentage of exposure.

    SIGNED, and the sign is the point: positive is a cost and negative is a
    credit. a long pays the benchmark plus IG's admin fee. a short pays the
    benchmark minus it, which while the benchmark is above the admin fee is
    money coming back rather than going out.

    nights are calendar nights, so a friday entry on a D1 exit is financed
    over the weekend whether or not the market opened.
    '''
    bench = _benchmark_for(market)
    if side == 'SHORT':
        #a short is on the lending side of the financing, so the benchmark
        #comes TO us and IG's admin fee comes off it. expressed as a cost
        #that is negative - a credit - whenever the benchmark is above the
        #admin fee, and a real charge whenever rates fall below it
        annual = IG_FUNDING_ADMIN_PCT - bench
    else:
        annual = bench + IG_FUNDING_ADMIN_PCT
    return annual / 365.0 * max(0, int(nights or 0))


def round_trip_cost_pct(side='LONG', market='LSE', nights=1, measured_spread_pct=None):
    '''
    everything one trade pays, as a percentage of the exposure.

    can come out NEGATIVE for a short held over a long weekend, which is not
    a bug: the funding credit can exceed a tight spread. that is a real
    feature of shorting a financed instrument and the share model could not
    express it at all.
    '''
    return round(spread_cost_pct(market, measured_spread_pct)
                 + funding_cost_pct(side, market, nights), 4)


def pl_gbp_for(return_pct, stake_gbp=None):
    '''
    pounds from a percentage move on the exposure.

    the percentage is on the EXPOSURE, so at five to one a 1% move on £100
    committed is £5. every return figure in this module is on exposure, which
    keeps it comparable with the gap, the backtest and every other strategy
    in merlin; the leverage shows up in the pounds and in return_on_stake_pct
    '''
    stake = STAKE_GBP if stake_gbp is None else stake_gbp
    return stake * CORTEX_LEVERAGE * (return_pct / 100.0)


def _nights_between(entry_date, exit_date):
    '''calendar nights between two iso dates, at least one'''
    try:
        a = pd.Timestamp(entry_date).normalize()
        b = pd.Timestamp(exit_date).normalize()
        return max(1, int((b - a).days))
    except Exception:
        return 1
PAPER_SCHEMA     = '9.0'

#backtest equity curve concurrency. cortex holds roughly 3-5 at a time at the
#default thresholds, so weighting each trade at 1/5 of the book is about right
BT_MAX_CONCURRENT = 5
BT_START_CAPITAL  = 10000.0

MARKETS = ('LSE', 'US')

#run times, now that IG is instant. the afternoon run sits at 16:25 rather
#than after the close on purpose: late enough that the day high and low are
#all but final, early enough that the trade could go into the closing auction
RUN_TIMES = {
    'LSE': {'tz': 'Europe/London', 'open': (8, 0),  'close': (16, 30),
            'morning': (8, 0), 'afternoon': (16, 25)},
    'US':  {'tz': 'America/New_York', 'open': (9, 30), 'close': (16, 0),
            'morning': (9, 30), 'afternoon': (15, 55)},
}


# ==================== UNIVERSES ====================

#FTSE 350 as validated in the original CFD project. dots become dashes for
#yahoo (BT.A -> BT-A.L). investment trusts are left in deliberately, they gap
#on NAV news and the strategy was researched with them included.
LSE_UNIVERSE = [
    '3IN','AAF','AAL','AAS','ABDN','ABF','ADM','AEP','AGT','AIE','AJB','ALFA','ALW','AML',
    'ANTO','AO','APN','ASHM','ASL','ATR','ATT','ATYM','AUTO','AV','AVON','AZN','BA','BAB','BAG',
    'BARC','BATS','BBOX','BBY','BCG','BEZ','BGEO','BGFD','BHMG','BKG','BLND','BME','BNKR',
    'BNZL','BOWL','BOY','BP','BPCR','BPT','BRBY','BREE','BRGE','BRSC','BRWM','BSIF','BT-A',
    'BTRW','BUT','BWY','BYG','BYIT','CBG','CCC','CCEP','CCH','CCL','CCR','CGT','CHG','CHRY',
    'CKN','CLDN','CMCX','CNA','COA','COST','CPG','CRDA','CSN','CTEC','CTY','CURY','CVSG','CWK',
    'CWR','DCC','DGE','DLN','DNLM','DOCS','DOM','DPLM','DRX','DSCV','EDIN','EDV','ELM','EMG',
    'ENOG','ENT','ESCT','EWG','EWI','EXPN','EZJ','FAN','FCH','FCIT','FCSS','FEML','FEV','FGEN',
    'FGP','FGT','FOUR','FRAS','FRES','FSG','FSV','GAMA','GAW','GBG','GCP','GDWN','GEN','GFRD',
    'GFTU','GLEN','GNC','GNS','GPE','GRG','GRI','GROW','GSCT','GSK','HAS','HBR','HFEL','HFG',
    'HGT','HICL','HIK','HILS','HLMA','HLN','HMSO','HOC','HRI','HSBA','HSL','HSX','HTG','HTWS',
    'HVPE','HWDN','HWG','IAD','IAG','IBST','ICG','ICGT','IEM','IGG','IHG','IHP','IMB','IMI',
    'INCH','INF','INPP','INVP','IPF','IPO','ITH','ITRK','ITV','IWG','JAM','JD','JDW','JEDT',
    'JEGI','JEMI','JFJ','JGGI','JII','JMAT','JMGI','JSG','JTC','JUP','KGF','KIE','KLR','KNOS',
    'LAND','LGEN','LLOY','LMP','LRE','LSEG','LWDB','MAB','MEGP','MGAM','MGNS','MKS','MNDI',
    'MNG','MNKS','MNTN','MONY','MOON','MRC','MRCH','MRO','MSLH','MTLN','MTO','MTRO','MUT','MYI',
    'N91','NAS','NBPE','NCC','NG','NWG','NXT','OCDO','OCI','ONT','OSB','OXB','OXIG','PAF','PAG',
    'PAGE','PCGH','PCT','PETS','PEY','PFD','PHI','PHP','PIN','PINT','PLUS','PNL','PNN','POLN',
    'PPET','PPH','PRN','PRU','PSH','PSN','PSON','PTEC','QLT','QQ','RAT','RCP','REL','RHIM',
    'RICA','RIO','RKT','RMV','RNK','ROR','RPI','RR','RS1','RSW','RTO','RTW','SAFE','SAGA',
    'SAIN','SBRY','SCT','SDLF','SDP','SDR','SEIT','SEQI','SGE','SGRO','SHAW','SHC','SHEL',
    'SMIN','SMT','SMWH','SN','SNR','SOI','SPI','SPX','SRE','SRP','SSE','SSPG','STAN','STJ',
    'SUPR','SVS','SVT','SYNC','TATE','TBCG','TCAP','TEM','TEP','TFIF','THG','THRG','THRL',
    'TMPL','TPK','TRIG','TRN','TRST','TRY','TSCO','TW','UEM','UKW','ULVR','USA','UTG','UU',
    'VCT','VEIL','VOD','VOF','VSVS','VTY','WEIR','WIX','WIZZ','WKP','WOSG','WPP','WTB','WWH',
    'XPS','ZIG',
]

#app.py injects merlin's existing get_stock_universe here at import time so
#cortex reuses the same US list as every other tab without importing app.py
#back (which would be circular)
_US_UNIVERSE_PROVIDER = None


def set_us_universe_provider(fn):
    '''app.py calls this once at startup with merlin's get_stock_universe'''
    global _US_UNIVERSE_PROVIDER
    _US_UNIVERSE_PROVIDER = fn


#2000 means "everything the provider has". cortex never silently trims a
#universe - the LSE list is all 349 FTSE 350 names with epics and the US list
#is merlin's full 572, topped up from wikipedia above that
UNIVERSE_ALL = 2000


def get_universe(market, count=UNIVERSE_ALL):
    '''plain ticker list for a market, before any yahoo suffix is added'''
    if market == 'LSE':
        return LSE_UNIVERSE[:count]
    if _US_UNIVERSE_PROVIDER is not None:
        try:
            return list(_US_UNIVERSE_PROVIDER(count))[:count]
        except Exception:
            pass
    #minimal fallback so the tab still works if the provider is not wired up
    return ['AAPL','MSFT','NVDA','AMZN','META','GOOGL','TSLA','AMD','NFLX','JPM'][:count]


def to_yf(ticker, market):
    '''yahoo symbol for a plain ticker'''
    if market == 'LSE':
        return f"{ticker.replace('.', '-')}.L"
    return ticker


# ==================== SMALL HELPERS ====================

def _finite(x):
    '''nan and inf slip through None checks and are truthy, so gate on this'''
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _pct(base, new):
    if not _finite(base) or not _finite(new) or base == 0:
        return None
    return (new - base) / base * 100.0


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _today_iso():
    return datetime.date.today().isoformat()


def exit_day_for_gap(abs_gap, exit_day=DEFAULT_EXIT_DAY, use_buckets=False):
    '''
    how many closes to hold. flat exit_day by default because that is what
    survived out-of-sample. the original bucket table is available for
    comparison but it was fitted on 150 signals and did not replicate.
    '''
    if not use_buckets:
        return int(exit_day)
    for lo, hi, days in BUCKET_EXITS:
        if lo <= abs_gap < hi:
            return days
    return int(exit_day)


def signal_direction(gap_pct):
    '''gap up means we short it, gap down means we buy it'''
    return 'SHORT' if gap_pct > 0 else 'LONG'


def trade_return_pct(side, entry, exit_price):
    '''signed so positive is always profitable, whichever way round we are'''
    raw = _pct(entry, exit_price)
    if raw is None:
        return None
    return raw if side == 'LONG' else -raw


# ==================== IG LIVE DATA ====================
#
#IG is the primary source for live scans from 9.0. the two things that make
#it worth the extra setup over yahoo:
#
#  1. it is real time, so the morning run goes at 08:00 on the bell and the
#     afternoon run at 16:25, early enough to actually place the trade
#  2. percentageChange in the snapshot is measured off yesterday's close, so
#     at the open it IS the gap, with no reliance on yahoo's daily open field
#
#and the one thing that makes it unsuitable for history: spread bet daily bars
#record a DFB rollover open, not the LSE auction open. the backtester stays on
#yahoo for exactly that reason.

IG_EPIC_MAP  = os.path.join(DATA_DIR, 'cortex_ig_epics.csv')
IG_BASE_URL  = 'https://demo-api.ig.com/gateway/deal'
IG_BATCH     = 50     #max epics per /markets call
IG_MIN_GAP_S = 2.0    #seconds between calls, keeps us under 30/minute

_epic_cache = {'loaded_at': 0, 'map': {}}


def load_ig_epics(force=False):
    '''ticker -> epic, cached for a minute so repeated calls are free'''
    if not force and time.time() - _epic_cache['loaded_at'] < 60:
        return _epic_cache['map']
    epics = {}
    if os.path.exists(IG_EPIC_MAP):
        try:
            with open(IG_EPIC_MAP, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    t = (row.get('ticker') or '').strip()
                    e = (row.get('epic') or '').strip()
                    if t and e:
                        epics[t] = e
        except Exception:
            pass
    _epic_cache['loaded_at'] = time.time()
    _epic_cache['map'] = epics
    return epics


def ig_configured():
    '''credentials present. does not prove they work, only that they exist'''
    return bool(os.getenv('IG_API_KEY') and os.getenv('IG_USERNAME')
                and os.getenv('IG_PASSWORD'))


def ig_available():
    return ig_configured() and bool(load_ig_epics())


def ig_status():
    '''what the UI shows so you can see at a glance why IG is or is not on'''
    epics = load_ig_epics()
    return {
        'credentials': ig_configured(),
        'epic_file': os.path.exists(IG_EPIC_MAP),
        'epics_mapped': len(epics),
        'ready': ig_configured() and bool(epics),
        'epic_file_path': IG_EPIC_MAP,
    }


class IGClient:
    '''
    minimal IG REST client - login, batched snapshots, logout.
    only /markets is used, which is the one endpoint that behaves the same on
    both the spread bet and CFD accounts.
    '''

    def __init__(self, account=None):
        import requests
        self.api_key    = os.getenv('IG_API_KEY')
        self.username   = os.getenv('IG_USERNAME')
        self.password   = os.getenv('IG_PASSWORD')
        self.account_id = account or os.getenv('IG_ACCOUNT_ID')
        self.session    = requests.Session()
        self.cst = self.tok = None
        self._last = 0.0

    def _headers(self, version='1'):
        return {'Content-Type': 'application/json; charset=UTF-8',
                'Accept': 'application/json; charset=UTF-8',
                'X-IG-API-KEY': self.api_key, 'CST': self.cst,
                'X-SECURITY-TOKEN': self.tok, 'Version': version}

    def login(self):
        r = self.session.post(
            f'{IG_BASE_URL}/session',
            json={'identifier': self.username, 'password': self.password},
            headers={'Content-Type': 'application/json; charset=UTF-8',
                     'Accept': 'application/json; charset=UTF-8',
                     'X-IG-API-KEY': self.api_key, 'Version': '2'}, timeout=15)
        r.raise_for_status()
        self.cst, self.tok = r.headers.get('CST'), r.headers.get('X-SECURITY-TOKEN')
        if not self.cst or not self.tok:
            raise ConnectionError('IG login returned no session tokens')
        #switching accounts reissues both tokens. losing them is what caused
        #the 401s in the original project, so capture them from the switch too
        if self.account_id and r.json().get('currentAccountId') != self.account_id:
            s = self.session.put(f'{IG_BASE_URL}/session',
                                 json={'accountId': self.account_id,
                                       'lightstreamerEndpoint': None},
                                 headers=self._headers('1'), timeout=15)
            s.raise_for_status()
            self.cst = s.headers.get('CST') or self.cst
            self.tok = s.headers.get('X-SECURITY-TOKEN') or self.tok

    def _throttle(self):
        wait = IG_MIN_GAP_S - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def snapshots(self, epics):
        '''
        batched market data. returns {epic: snapshot_dict}.

        tries /markets?epics= first (50 at a time, so a FTSE 350 scan is 7
        calls). if that shape is not what we expect on this account, falls
        back to one /markets/{epic} call per ticker, which is slow but always
        works. the fallback is why a bad batch response degrades rather than
        breaking the scan.
        '''
        out = {}
        remaining = []
        for chunk in _chunks(list(epics), IG_BATCH):
            self._throttle()
            try:
                r = self.session.get(f'{IG_BASE_URL}/markets',
                                     params={'epics': ','.join(chunk)},
                                     headers=self._headers('1'), timeout=25)
                r.raise_for_status()
                body = r.json() or {}
                details = body.get('marketDetails')
                if not isinstance(details, list):
                    remaining.extend(chunk)
                    continue
                for d in details:
                    epic = (d.get('instrument') or {}).get('epic')
                    snap = d.get('snapshot')
                    if epic and snap:
                        out[epic] = snap
                #anything the batch silently dropped gets retried singly
                remaining.extend([e for e in chunk if e not in out])
            except Exception:
                remaining.extend(chunk)

        for epic in remaining:
            self._throttle()
            try:
                r = self.session.get(f'{IG_BASE_URL}/markets/{epic}',
                                     headers=self._headers('3'), timeout=15)
                r.raise_for_status()
                snap = (r.json() or {}).get('snapshot')
                if snap:
                    out[epic] = snap
            except Exception:
                continue
        return out

    def logout(self):
        try:
            self.session.delete(f'{IG_BASE_URL}/session',
                                headers=self._headers('1'), timeout=10)
        except Exception:
            pass

    def __enter__(self):
        self.login()
        return self

    def __exit__(self, *a):
        self.logout()
        return False


def _ig_row_from_snapshot(snap, phase):
    '''
    turn one IG snapshot into the same row shape the yahoo path produces.

    at the open, current price is effectively the open price, so:
      gap_pct         = percentageChange   (move off yesterday's close)
      yesterday_close = current / (1 + pct/100)
      today_open      = current

    percentageChange is measured off yesterday's CLOSE, not today's open. that
    is the finding from the original project that the IG docs get wrong, and
    it is the reason the morning run has to happen at the open rather than
    being reconstructable later in the day.
    '''
    if not snap:
        return None
    bid, offer = snap.get('bid'), snap.get('offer')
    high, low  = snap.get('high'), snap.get('low')
    pct        = snap.get('percentageChange')
    status     = snap.get('marketStatus')

    if pct is None:
        return None

    #mid where we have both sides, otherwise fall back through high then low.
    #bid/offer go null outside market hours and during the opening auction
    spread_pct = None
    if _finite(bid) and _finite(offer):
        current = (float(bid) + float(offer)) / 2
        #the real dealing spread, as a percentage of the mid. this is what a
        #round trip actually costs on this name at this moment, and it used
        #to be computed and discarded one line above
        if current > 0:
            spread_pct = abs(float(offer) - float(bid)) / current * 100.0
    elif _finite(high) and _finite(low):
        current = (float(high) + float(low)) / 2
    elif _finite(high):
        current = float(high)
    else:
        return None

    denom = 1 + (float(pct) / 100.0)
    if denom == 0:
        return None
    yesterday_close = current / denom
    if yesterday_close <= 0 or current <= 0:
        return None

    day_high = float(high) if _finite(high) else current
    day_low  = float(low) if _finite(low) else current

    return {
        'bar_date':        _today_iso(),
        'yesterday_close': yesterday_close,
        'today_open':      current if phase == 'morning' else None,
        'current':         current,
        'day_high':        day_high,
        'day_low':         day_low,
        'gap_pct':         float(pct),
        'market_status':   status,
        'spread_pct':      spread_pct,
        'source':          'IG',
    }


# ==================== YAHOO FALLBACK + HISTORY ====================

_recent_cache = {}
_RECENT_TTL   = 120


def _normalise_download(df, symbols):
    '''yf.download gives a MultiIndex for many tickers and a flat frame for one'''
    out = {}
    if df is None or len(df) == 0:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        level0 = set(df.columns.get_level_values(0))
        for sym in symbols:
            if sym in level0:
                sub = df[sym].dropna(how='all')
                if len(sub):
                    out[sym] = sub
    else:
        if len(symbols) == 1:
            sub = df.dropna(how='all')
            if len(sub):
                out[symbols[0]] = sub
    return out


def fetch_recent_bars(symbols, period='10d', use_cache=True):
    '''batch daily OHLC from yahoo. used for the US fallback and for settling
    paper positions, which needs historical closes rather than live quotes'''
    key = (period, tuple(sorted(symbols)))
    if use_cache and key in _recent_cache:
        fetched_at, data = _recent_cache[key]
        if time.time() - fetched_at < _RECENT_TTL:
            return data
    out = {}
    for chunk in _chunks(list(symbols), 60):
        try:
            df = yf.download(chunk, period=period, interval='1d',
                             group_by='ticker', auto_adjust=False,
                             progress=False, threads=True)
            out.update(_normalise_download(df, chunk))
        except Exception:
            continue
    _recent_cache[key] = (time.time(), out)
    return out


def _yahoo_row(sub, phase):
    '''same row shape as the IG path, built from a delayed yahoo daily bar'''
    if sub is None or len(sub) < 2:
        return None
    try:
        closes = sub['Close']
        prev_close = None
        for i in range(len(sub) - 2, -1, -1):
            if _finite(closes.iloc[i]):
                prev_close = float(closes.iloc[i])
                break
        if prev_close is None:
            return None
        last = sub.iloc[-1]
        o, h, l, c = last['Open'], last['High'], last['Low'], last['Close']
        if not _finite(o):
            return None
        gap = _pct(prev_close, float(o))
        if gap is None:
            return None
        return {
            'bar_date':        pd.to_datetime(sub.index[-1]).strftime('%Y-%m-%d'),
            'yesterday_close': prev_close,
            'today_open':      float(o),
            'current':         float(c) if _finite(c) else float(o),
            'day_high':        float(h) if _finite(h) else float(o),
            'day_low':         float(l) if _finite(l) else float(o),
            'gap_pct':         gap,
            'market_status':   None,
            'source':          'yfinance',
        }
    except Exception:
        return None


def _cache_path(market, ticker):
    d = os.path.join(CACHE_DIR, market)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{ticker.replace('/', '_')}.csv")


def _read_cached_history(market, ticker):
    p = _cache_path(market, ticker)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p, parse_dates=['date'])
        return df if len(df) >= 30 else None
    except Exception:
        return None


def fetch_history(market, tickers, years=3, status=None, refresh=False):
    '''
    daily OHLC per ticker for the backtest and sweep, cached to
    data/cortex_cache/<market>/.

    yahoo, not IG, and deliberately so: IG spread bet daily bars use a DFB
    rollover open which differs from the LSE auction open on exactly the
    gapping stocks this strategy cares about.
    '''
    start = (datetime.date.today() - datetime.timedelta(days=int(years * 365) + 40)).isoformat()
    end   = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

    out, need = {}, []
    for t in tickers:
        cached = None if refresh else _read_cached_history(market, t)
        if cached is not None and len(cached):
            newest = pd.to_datetime(cached['date']).max().date()
            if (datetime.date.today() - newest).days <= 4:
                out[t] = cached
                continue
        need.append(t)

    if status is not None:
        status['message'] = f'{len(out)} tickers cached, downloading {len(need)}...'

    done = 0
    for chunk in _chunks(need, 50):
        syms = [to_yf(t, market) for t in chunk]
        try:
            raw = yf.download(syms, start=start, end=end, interval='1d',
                              group_by='ticker', auto_adjust=False,
                              progress=False, threads=True)
            frames = _normalise_download(raw, syms)
        except Exception:
            frames = {}
        for t, sym in zip(chunk, syms):
            sub = frames.get(sym)
            if sub is None or len(sub) < 30:
                continue
            df = pd.DataFrame({
                'date':  pd.to_datetime(sub.index).tz_localize(None),
                'open':  sub['Open'].values, 'high': sub['High'].values,
                'low':   sub['Low'].values,  'close': sub['Close'].values,
            }).dropna()
            if len(df) < 30:
                continue
            out[t] = df
            try:
                df.to_csv(_cache_path(market, t), index=False)
            except Exception:
                pass
        done += len(chunk)
        if status is not None:
            status['message'] = f'downloaded {done}/{len(need)} ({len(out)} usable)'
            status['progress'] = min(40, 5 + int(done / max(len(need), 1) * 35))
    return out


# ==================== MARKET CLOCK ====================

def _tz(name):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        return datetime.timezone.utc


def market_clock():
    '''
    open/closed state per market plus the next scheduled cortex run, as UTC
    timestamps so the browser can count down without arguing about timezones.
    weekends are handled, public holidays are not - a bank holiday will show
    as open, which the scan itself will catch when IG reports the market as
    not tradeable.
    '''
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    out = []
    for market, cfg in RUN_TIMES.items():
        tz = _tz(cfg['tz'])
        local = now_utc.astimezone(tz)

        def at(h, m, day_offset=0):
            d = (local + datetime.timedelta(days=day_offset)).replace(
                hour=h, minute=m, second=0, microsecond=0)
            return d

        open_t, close_t = at(*cfg['open']), at(*cfg['close'])
        weekday = local.weekday() < 5
        is_open = weekday and open_t <= local < close_t

        events = []
        for label, hm in (('morning run', cfg['morning']), ('afternoon run', cfg['afternoon'])):
            for offset in range(0, 8):
                t = at(*hm, day_offset=offset)
                if t > local and (local + datetime.timedelta(days=offset)).weekday() < 5:
                    events.append((t, label))
                    break
        events.sort()
        nxt_t, nxt_label = events[0] if events else (None, None)

        #the next open or close, whichever comes first
        if is_open:
            boundary, boundary_label = close_t, 'closes'
        else:
            b = open_t if local < open_t and weekday else None
            if b is None:
                for offset in range(1, 8):
                    cand = at(*cfg['open'], day_offset=offset)
                    if (local + datetime.timedelta(days=offset)).weekday() < 5:
                        b = cand
                        break
            boundary, boundary_label = b, 'opens'

        out.append({
            'market': market,
            'is_open': is_open,
            'local_time': local.strftime('%H:%M'),
            'timezone': cfg['tz'],
            'boundary_label': boundary_label,
            'boundary_local': boundary.strftime('%H:%M') if boundary else None,
            'boundary_utc': boundary.astimezone(datetime.timezone.utc).isoformat() if boundary else None,
            'next_run_label': nxt_label,
            'next_run_local': nxt_t.strftime('%H:%M') if nxt_t else None,
            'next_run_utc': nxt_t.astimezone(datetime.timezone.utc).isoformat() if nxt_t else None,
        })
    return {'now_utc': now_utc.isoformat(), 'markets': out}


# ==================== SCAN CORE ====================

def _gather_rows(market, tickers, phase, prefer_ig=True, status=None):
    '''
    one row per ticker from the best available source. IG where an epic exists
    and credentials are set, yahoo for everything else. returns
    (rows_by_ticker, meta) where meta records which source actually served it.
    '''
    rows, used_ig, ig_error = {}, 0, None
    epics = load_ig_epics() if prefer_ig else {}
    ig_tickers = [t for t in tickers if t in epics] if (prefer_ig and ig_configured()) else []

    if ig_tickers:
        if status is not None:
            status['message'] = f'IG snapshots for {len(ig_tickers)} {market} tickers...'
            status['progress'] = 15
        try:
            with IGClient() as ig:
                snaps = ig.snapshots([epics[t] for t in ig_tickers])
            by_epic = {epics[t]: t for t in ig_tickers}
            for epic, snap in snaps.items():
                t = by_epic.get(epic)
                if not t:
                    continue
                row = _ig_row_from_snapshot(snap, phase)
                if row:
                    rows[t] = row
                    used_ig += 1
        except Exception as e:
            ig_error = f'{type(e).__name__}: {str(e)[:160]}'

    fallback = [t for t in tickers if t not in rows]
    if fallback:
        if status is not None:
            status['message'] = f'yahoo bars for {len(fallback)} {market} tickers...'
            status['progress'] = 50
        syms = [to_yf(t, market) for t in fallback]
        bars = fetch_recent_bars(syms, period='10d', use_cache=False)
        for t, sym in zip(fallback, syms):
            row = _yahoo_row(bars.get(sym), phase)
            if row:
                rows[t] = row

    return rows, {'ig_rows': used_ig, 'yahoo_rows': len(rows) - used_ig,
                  'ig_error': ig_error, 'epics_available': len(epics)}


def _stamp_earnings(gappers):
    '''
    10.0: mark which of today's gaps landed on an earnings print.

    EARNINGS NOTE. cortex fades overreactions, and an earnings gap is not the
    same animal as a broker-note gap or a sector gap. one is the market
    repricing on genuinely new fundamental information, the other is flow. if
    the fade works on one and not the other then the gap floor is the wrong
    knob to be turning and this is the right one, but nobody knows yet
    because it has never been recorded. so it gets recorded, and the question
    can be asked in six months with real signals rather than argued about now.

    the cost is one yfinance lookup per gapper, not per universe member. a
    normal morning turns up ten to thirty gappers out of 349 names, and the
    lookups are cached for twelve hours, so a re-run costs nothing.

    this only works forwards. yahoo hands back the last report date and the
    next one, not a history, so there is no way to backfill the flag onto the
    signals already in the log - those keep an empty column and the sample
    starts from today.
    '''
    for g in gappers:
        try:
            yf_symbol = to_yf(g['ticker'], g.get('market') or 'LSE')
            info = earnings.earnings_gap(yf_symbol, g.get('bar_date'), EARNINGS_GAP_WINDOW)
            g['earnings_gap'] = info['flag']
            g['earnings_days_from_report'] = info['days_from_report']
            g['earnings_report_date'] = info['report_date']
        except Exception:
            g['earnings_gap'] = None
            g['earnings_days_from_report'] = None
            g['earnings_report_date'] = None
    return gappers


def scan_gaps(market, gap_min=DEFAULT_GAP_MIN, gap_max=DEFAULT_GAP_MAX,
              count=UNIVERSE_ALL, prefer_ig=True, status=None):
    '''
    morning run. keeps tickers whose open gapped by at least gap_min and less
    than gap_max off yesterday's close. gap_max exists because 15%+ moves are
    nearly always corporate actions rather than the overreaction this trades.
    '''
    tickers = get_universe(market, count)
    rows, meta = _gather_rows(market, tickers, 'morning', prefer_ig, status)

    if status is not None:
        status['message'] = 'measuring gaps...'
        status['progress'] = 80

    gappers, stale, halted = [], 0, 0
    today = _today_iso()
    for t, row in rows.items():
        if row['bar_date'] != today:
            stale += 1
        if row.get('market_status') and row['market_status'] != 'TRADEABLE':
            halted += 1
        gap = row['gap_pct']
        a = abs(gap)
        if a < gap_min or a >= gap_max:
            continue
        open_px = row.get('today_open') or row['current']
        gappers.append({
            'ticker': t, 'market': market, 'bar_date': row['bar_date'],
            'yesterday_close': round(row['yesterday_close'], 4),
            'today_open': round(open_px, 4),
            'gap_pct': round(gap, 2),
            'gap_direction': 'UP' if gap > 0 else 'DOWN',
            'side': signal_direction(gap),
            'market_status': row.get('market_status'),
            'source': row['source'],
        })

    gappers.sort(key=lambda r: -abs(r['gap_pct']))
    _stamp_earnings(gappers)
    earnings_gaps = sum(1 for g in gappers if g.get('earnings_gap') is True)
    payload = {'market': market, 'run_at': datetime.datetime.now().isoformat(),
               'bar_date': today, 'gap_min': gap_min, 'gap_max': gap_max,
               'scanned': len(rows), 'stale_bars': stale, 'not_tradeable': halted,
               'earnings_gaps': earnings_gaps,
               'gappers': gappers, **meta}
    _save_watchlist(market, payload)
    if status is not None:
        status['progress'] = 100
    return payload


def _save_watchlist(market, payload):
    os.makedirs(DATA_DIR, exist_ok=True)
    book = {}
    if os.path.exists(WATCHLIST_PATH):
        try:
            with open(WATCHLIST_PATH) as f:
                book = json.load(f)
        except Exception:
            book = {}
    book[market] = payload
    with open(WATCHLIST_PATH, 'w') as f:
        json.dump(book, f, indent=2)


def load_watchlist(market):
    if not os.path.exists(WATCHLIST_PATH):
        return None
    try:
        with open(WATCHLIST_PATH) as f:
            return json.load(f).get(market)
    except Exception:
        return None


def check_wings(market, wing=DEFAULT_WING, exit_day=DEFAULT_EXIT_DAY,
                use_buckets=False, prefer_ig=True, status=None):
    '''
    afternoon run. re-reads the watchlist and keeps only the stocks that did
    NOT continue in the gap direction by more than `wing` percent off the open.

    gap up   -> how far it ran above the open  (open_to_high)
    gap down -> how far it fell below the open (open_to_low)

    anything between wing and wing+1 is logged as borderline rather than
    silently binned, same as the original research. entry price is the live
    mid at the time of the run, which with IG at 16:25 is close enough to the
    close to trade on and close enough to the backtest to compare with.
    '''
    wl = load_watchlist(market)
    if not wl or not wl.get('gappers'):
        return {'market': market, 'error': 'no watchlist. run the morning scan first.',
                'confirmed': [], 'borderline': [], 'discarded': []}

    gappers = wl['gappers']
    rows, meta = _gather_rows(market, [g['ticker'] for g in gappers],
                              'afternoon', prefer_ig, status)

    if status is not None:
        status['message'] = 'applying wing filter...'
        status['progress'] = 80

    confirmed, borderline, discarded = [], [], []
    for g in gappers:
        row = rows.get(g['ticker'])
        if row is None:
            discarded.append({**g, 'reason': 'no end of day data'})
            continue

        o = g['today_open']
        oth = _pct(o, row['day_high']) or 0.0
        otl = _pct(o, row['day_low']) or 0.0
        wing_move = oth if g['gap_pct'] > 0 else abs(otl)

        rec = {**g,
               'day_high': round(row['day_high'], 4),
               'day_low': round(row['day_low'], 4),
               'open_to_high_pct': round(oth, 2),
               'open_to_low_pct': round(otl, 2),
               'wing_pct': round(wing_move, 2),
               'entry_price': round(row['current'], 4),
               #the spread measured on the quote we are entering against.
               #None on the yahoo path, which falls back to the assumption
               'spread_pct': row.get('spread_pct'),
               'exit_after_days': exit_day_for_gap(abs(g['gap_pct']), exit_day, use_buckets),
               'price_source': row['source'],
               'market_status': row.get('market_status')}

        if wing_move <= wing:
            confirmed.append(rec)
        elif wing_move <= wing + BORDERLINE_BAND:
            borderline.append({**rec, 'reason': f'wing {wing_move:.2f}% just over {wing:.1f}%'})
        else:
            discarded.append({**rec, 'reason': f'continued {wing_move:.2f}% in gap direction'})

    confirmed.sort(key=lambda r: -abs(r['gap_pct']))
    _append_signal_log(confirmed)

    return {'market': market, 'run_at': datetime.datetime.now().isoformat(),
            'bar_date': wl.get('bar_date'), 'wing': wing, 'exit_day': exit_day,
            'use_buckets': use_buckets, 'confirmed': confirmed,
            'borderline': borderline, 'discarded': discarded, **meta}


SIGNAL_FIELDS = ['bar_date', 'market', 'ticker', 'gap_pct', 'gap_direction', 'side',
                 'yesterday_close', 'today_open', 'day_high', 'day_low',
                 'wing_pct', 'entry_price', 'exit_after_days', 'price_source',
                 #10.0, and the whole point of the earnings flag - a column in
                 #the log is what makes "does the fade work on earnings gaps"
                 #answerable later instead of arguable now
                 'earnings_gap', 'earnings_days_from_report', 'earnings_report_date']


def _migrate_signal_log():
    '''
    bring an older signal log up to the current header.

    appending wider rows under a narrower header silently misaligns every
    column from that point on, and this file is the forward record - it is
    the one file in merlin that cannot be regenerated. so the header is
    rewritten once, old rows keep empty cells for the new columns, and the
    previous file is left behind as a .bak in case this goes wrong.
    '''
    if not os.path.exists(SIGNAL_LOG):
        return
    try:
        with open(SIGNAL_LOG, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            if header == SIGNAL_FIELDS:
                return
            rows = list(reader)
        #only ever widen. an unrecognised header is left alone rather than
        #being reshaped into something that loses columns
        if [h for h in header if h not in SIGNAL_FIELDS]:
            return
        os.replace(SIGNAL_LOG, SIGNAL_LOG + '.bak')
        with open(SIGNAL_LOG, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=SIGNAL_FIELDS, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f'\n[cortex] signal log widened to {len(SIGNAL_FIELDS)} columns, '
              f'{len(rows)} existing rows kept, old file saved as cortex_signals.csv.bak')
    except Exception as e:
        print(f'\n[cortex] could not migrate the signal log: {type(e).__name__}')


def _append_signal_log(rows):
    '''running record of every confirmed signal, for later review'''
    if not rows:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    _migrate_signal_log()
    new = not os.path.exists(SIGNAL_LOG)
    try:
        with open(SIGNAL_LOG, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=SIGNAL_FIELDS, extrasaction='ignore')
            if new:
                w.writeheader()
            w.writerows(rows)
    except Exception:
        pass


def recent_signals(limit=25):
    '''last N confirmed signals, newest first. used by the header ticker tape'''
    if not os.path.exists(SIGNAL_LOG):
        return []
    try:
        with open(SIGNAL_LOG, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        return list(reversed(rows))[:limit]
    except Exception:
        return []


def compare_sources(market='LSE', count=60):
    '''
    diagnostic. pulls the same tickers from IG and from yahoo and reports how
    far apart the two gap readings are. worth running once in a while, because
    live signals come from IG and backtested ones come from yahoo, and if
    those drift the paper book stops being comparable to the backtest.
    '''
    tickers = get_universe(market, count)
    ig_rows, _ = _gather_rows(market, tickers, 'morning', True, None)
    syms = [to_yf(t, market) for t in tickers]
    yb = fetch_recent_bars(syms, period='10d', use_cache=False)

    rows = []
    for t, sym in zip(tickers, syms):
        a = ig_rows.get(t)
        b = _yahoo_row(yb.get(sym), 'morning')
        if not a or not b or a['source'] != 'IG':
            continue
        rows.append({'ticker': t, 'ig_gap_pct': round(a['gap_pct'], 3),
                     'yahoo_gap_pct': round(b['gap_pct'], 3),
                     'diff_pct_points': round(a['gap_pct'] - b['gap_pct'], 3)})
    diffs = [abs(r['diff_pct_points']) for r in rows]
    return {'market': market, 'compared': len(rows),
            'mean_abs_diff': round(float(np.mean(diffs)), 3) if diffs else None,
            'max_abs_diff': round(float(np.max(diffs)), 3) if diffs else None,
            'rows': sorted(rows, key=lambda r: -abs(r['diff_pct_points']))[:25]}


# ==================== PAPER BOOK ====================
#
#flat GBP stake per trade, longs and shorts, no cap on how many run at once.
#P/L is a pure percentage of the stake, so no FX or pence-to-pounds mess:
#a 2% winner is 2 pounds on a 100 pound stake whether it was AAPL or VOD.
#
#exits are on a calendar, so the book self-heals. every maintenance pass
#counts how many trading bars have printed since entry rather than trusting
#the app to have been running each day. settlement uses yahoo daily closes
#because it needs a historical close, which is the one thing IG will not
#give us reliably.

def _init_paper():
    return {'schema': PAPER_SCHEMA, 'created': _today_iso(),
            'stake_gbp': STAKE_GBP, 'fee_pct': CORTEX_FEE_PCT,
            'realised_pl_gbp': 0.0, 'fees_paid_gbp': 0.0,
            'positions': [], 'closed': [], 'activity': [],
            'equity_curve': [{'date': _today_iso(), 'equity': 0.0}]}


def load_paper():
    if not os.path.exists(PAPER_PATH):
        book = _init_paper()
        save_paper(book)
        return book
    try:
        with open(PAPER_PATH) as f:
            book = json.load(f)
    except Exception:
        book = _init_paper()
    for k, v in _init_paper().items():
        book.setdefault(k, v)
    return book


def save_paper(book):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = PAPER_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(book, f, indent=2)
    os.replace(tmp, PAPER_PATH)


def reset_paper():
    book = _init_paper()
    save_paper(book)
    return book


def _log(book, msg):
    book['activity'].insert(0, {'at': datetime.datetime.now().isoformat(), 'msg': msg})
    book['activity'] = book['activity'][:200]


def open_paper_positions(signals, book=None):
    '''one flat-stake trade per confirmed signal, duplicates refused so
    running the afternoon button twice does not double the book'''
    book = book or load_paper()
    held = {(p['ticker'], p['entry_date']) for p in book['positions']}
    opened = []
    for s in signals:
        if (s['ticker'], s['bar_date']) in held:
            continue
        entry = s.get('entry_price')
        if not _finite(entry) or entry <= 0:
            continue
        #the entry half of the dealing spread, charged on the EXPOSURE not
        #the money down. funding is not charged here because it has not been
        #incurred yet - it accrues per night and settles with the trade
        entry_cost_pct = spread_cost_pct(s['market'], s.get('spread_pct')) / 2.0
        fee = round(pl_gbp_for(entry_cost_pct), 4)
        pos = {
            'id': f"{s['ticker']}-{s['bar_date']}-{int(time.time()*1000)%100000}",
            'ticker': s['ticker'], 'symbol': to_yf(s['ticker'], s['market']),
            'market': s['market'], 'side': s['side'],
            'entry_date': s['bar_date'], 'entry_price': round(float(entry), 4),
            'gap_pct': s['gap_pct'], 'wing_pct': s.get('wing_pct'),
            'stake_gbp': STAKE_GBP,
            'leverage': CORTEX_LEVERAGE,
            'exposure_gbp': round(STAKE_GBP * CORTEX_LEVERAGE, 2),
            #the real spread off the IG quote at entry, when the run had one.
            #charged in preference to the fallback for the whole life of the
            #trade, so a wide name is not settled at a liquid name's cost
            'spread_pct': s.get('spread_pct'),
            'exit_after_days': int(s.get('exit_after_days', DEFAULT_EXIT_DAY)),
            'entry_fee_gbp': round(fee, 4),
            'entry_source': s.get('price_source', 'unknown'),
        }
        book['positions'].append(pos)
        book['fees_paid_gbp'] = round(float(book.get('fees_paid_gbp', 0)) + fee, 4)
        opened.append(pos)
        _log(book, f"opened {pos['side']} {pos['ticker']} at {pos['entry_price']} "
                   f"(gap {pos['gap_pct']:+.1f}%, hold {pos['exit_after_days']}d)")
    if opened:
        save_paper(book)
    return book, opened


def maintain_paper(book=None):
    '''
    close anything whose scheduled exit date has printed. counts actual
    trading bars after entry rather than calendar days, so weekends and bank
    holidays look after themselves and a missed run is not a problem.
    '''
    book = book or load_paper()
    if not book['positions']:
        _mark_open(book)
        save_paper(book)
        return book, []

    symbols = sorted({p['symbol'] for p in book['positions']})
    bars = fetch_recent_bars(symbols, period='1mo', use_cache=True)

    closed = []
    for p in list(book['positions']):
        sub = bars.get(p['symbol'])
        if sub is None or len(sub) == 0:
            continue
        try:
            idx = pd.to_datetime(sub.index).tz_localize(None)
            after = sub[idx > pd.Timestamp(p['entry_date'])]
            if len(after) < p['exit_after_days']:
                continue
            bar = after.iloc[p['exit_after_days'] - 1]
            exit_price = float(bar['Close'])
            exit_date = pd.to_datetime(after.index[p['exit_after_days'] - 1]).strftime('%Y-%m-%d')
        except Exception:
            continue
        if not _finite(exit_price) or exit_price <= 0:
            continue

        gross_pct = trade_return_pct(p['side'], p['entry_price'], exit_price)
        if gross_pct is None:
            continue
        #what this trade actually cost, from its own side, market and the
        #nights it was really open. a friday entry on a D1 exit pays the
        #weekend's financing, which is why nights are counted on the calendar
        nights = _nights_between(p['entry_date'], exit_date)
        cost_pct = round_trip_cost_pct(p['side'], p.get('market'), nights,
                                       p.get('spread_pct'))
        net_pct = gross_pct - cost_pct
        #the exit fee is whatever is left of the round trip after the entry
        #half already booked, so fees_paid_gbp always reconciles with the
        #deduction taken out of P/L even for positions opened under an older
        #cost model
        total_fee = pl_gbp_for(cost_pct, p['stake_gbp'])
        exit_fee = total_fee - float(p.get('entry_fee_gbp', 0) or 0)
        #every percentage in here is on the exposure, so the pounds carry the
        #leverage. return_on_stake_pct is the same trade seen from the money
        #actually committed, which at five to one is five times the move
        pl_gbp = pl_gbp_for(net_pct, p['stake_gbp'])

        rec = {**p, 'exit_date': exit_date, 'exit_price': round(exit_price, 4),
               'gross_return_pct': round(gross_pct, 3),
               'return_pct': round(net_pct, 3), 'pl_gbp': round(pl_gbp, 2),
               'exit_fee_gbp': round(exit_fee, 4), 'days_held': p['exit_after_days'],
               #kept on the trade so the record says what it was charged and
               #why, rather than leaving it to be inferred from a constant
               #that may have moved since
               'cost_pct': cost_pct, 'nights_held': nights,
               'return_on_stake_pct': round(net_pct * CORTEX_LEVERAGE, 3),
               'spread_pct_charged': round(spread_cost_pct(p.get('market'), p.get('spread_pct')), 4),
               'funding_pct': round(funding_cost_pct(p['side'], p.get('market'), nights), 4),
               'exit_reason': f"scheduled D{p['exit_after_days']} close"}
        book['closed'].append(rec)
        book['positions'] = [x for x in book['positions'] if x['id'] != p['id']]
        book['realised_pl_gbp'] = round(float(book.get('realised_pl_gbp', 0)) + pl_gbp, 4)
        book['fees_paid_gbp'] = round(float(book.get('fees_paid_gbp', 0)) + exit_fee, 4)
        closed.append(rec)
        _log(book, f"closed {rec['side']} {rec['ticker']} at {rec['exit_price']} "
                   f"for {rec['return_pct']:+.2f}% ({rec['pl_gbp']:+.2f})")

    if closed:
        book['closed'].sort(key=lambda t: t['exit_date'])
        _rebuild_equity(book)
    _mark_open(book)
    save_paper(book)
    return book, closed


def _mark_open(book):
    '''mark open positions to the latest close so the tab shows live P/L'''
    if not book['positions']:
        return
    symbols = sorted({p['symbol'] for p in book['positions']})
    bars = fetch_recent_bars(symbols, period='5d', use_cache=True)
    today = _today_iso()
    for p in book['positions']:
        #10.0: an open position is now marked NET of what it will cost.
        #
        #it used to be marked gross while a closed one settled net, so every
        #open trade was flattered by its whole round trip and the equity curve
        #stepped down at settlement for no visible reason. these exits are on
        #a calendar, not a stop, so the cost is not a possibility - the trade
        #is going to close and it is going to pay. financing is only counted
        #for the nights actually served so far
        cost_pct = round_trip_cost_pct(p['side'], p.get('market'),
                                       _nights_between(p['entry_date'], today),
                                       p.get('spread_pct'))
        sub = bars.get(p['symbol'])
        price = None
        if sub is not None and len(sub):
            for i in range(len(sub) - 1, -1, -1):
                if _finite(sub['Close'].iloc[i]):
                    price = float(sub['Close'].iloc[i])
                    break
        if price is None:
            p.update({'current_price': p['entry_price'],
                      'unrealised_gross_pct': 0.0, 'cost_pct': cost_pct,
                      'unrealised_pct': round(-cost_pct, 3),
                      'unrealised_on_stake_pct': round(-cost_pct * CORTEX_LEVERAGE, 3),
                      'unrealised_gbp': round(pl_gbp_for(-cost_pct, p['stake_gbp']), 2),
                      'exposure_gbp': round(p['stake_gbp'] * CORTEX_LEVERAGE, 2),
                      'price_is_stale': True})
            continue
        gross = trade_return_pct(p['side'], p['entry_price'], price) or 0.0
        pct = gross - cost_pct
        p.update({'current_price': round(price, 4),
                  'unrealised_gross_pct': round(gross, 3), 'cost_pct': cost_pct,
                  'unrealised_pct': round(pct, 3),
                  'unrealised_on_stake_pct': round(pct * CORTEX_LEVERAGE, 3),
                  'unrealised_gbp': round(pl_gbp_for(pct, p['stake_gbp']), 2),
                  'exposure_gbp': round(p['stake_gbp'] * CORTEX_LEVERAGE, 2),
                  'price_is_stale': False})


def _rebuild_equity(book):
    '''
    cumulative realised P/L in pounds, one point per closed trade. anchored to
    the earliest entry rather than the book creation date, otherwise a
    backdated first trade makes the curve run backwards in time.
    '''
    closed = sorted(book['closed'], key=lambda x: x['exit_date'])
    if closed:
        anchor = min([t.get('entry_date') or t['exit_date'] for t in closed]
                     + [book.get('created', _today_iso())])
    else:
        anchor = book.get('created', _today_iso())
    curve, running = [{'date': anchor, 'equity': 0.0}], 0.0
    for t in closed:
        running += t['pl_gbp']
        curve.append({'date': t['exit_date'], 'equity': round(running, 2)})
    book['equity_curve'] = curve
    book['realised_pl_gbp'] = round(running, 4)


def paper_summary(book=None):
    book = book or load_paper()
    closed = book['closed']
    rets = [t['return_pct'] for t in closed]
    wins = [r for r in rets if r > 0]

    def _side_stats(ts):
        r = [t['return_pct'] for t in ts]
        return {'n': len(r),
                'win_rate_pct': round(len([x for x in r if x > 0]) / len(r) * 100, 1) if r else 0,
                'avg_pct': round(sum(r) / len(r), 3) if r else 0,
                'pl_gbp': round(sum(t['pl_gbp'] for t in ts), 2)}

    return {
        'schema': book.get('schema'), 'created': book.get('created'),
        'stake_gbp': STAKE_GBP, 'fee_pct': CORTEX_FEE_PCT,
        'leverage': CORTEX_LEVERAGE, 'exposure_gbp': EXPOSURE_GBP,
        #10.0: the tab prints the cost model rather than one blended number.
        #a long and a short are not charged the same thing, and on a financed
        #instrument they are not even charged the same SIGN
        'cost_model': {
            'lse_long':   round_trip_cost_pct('LONG',  'LSE', 1),
            'lse_short':  round_trip_cost_pct('SHORT', 'LSE', 1),
            'us_long':    round_trip_cost_pct('LONG',  'US',  1),
            'us_short':   round_trip_cost_pct('SHORT', 'US',  1),
            'spread_lse_pct': IG_SPREAD_LSE_PCT, 'spread_us_pct': IG_SPREAD_US_PCT,
            'funding_admin_pct': IG_FUNDING_ADMIN_PCT,
            'benchmark_gbp_pct': IG_BENCHMARK_GBP_PCT,
            'benchmark_usd_pct': IG_BENCHMARK_USD_PCT,
            'long_night_pct':  round(funding_cost_pct('LONG',  'LSE', 1), 4),
            'short_night_pct': round(funding_cost_pct('SHORT', 'LSE', 1), 4),
        },
        'open_count': len(book['positions']), 'closed_count': len(closed),
        'realised_pl_gbp': round(float(book.get('realised_pl_gbp', 0)), 2),
        'unrealised_pl_gbp': round(sum(p.get('unrealised_gbp', 0) or 0 for p in book['positions']), 2),
        'fees_paid_gbp': round(float(book.get('fees_paid_gbp', 0)), 2),
        #at five to one these are different numbers and both matter: the
        #first is what is tied up, the second is what is actually exposed
        'margin_committed_gbp': round(len(book['positions']) * STAKE_GBP, 2),
        'capital_at_risk_gbp': round(len(book['positions']) * STAKE_GBP, 2),
        'exposure_open_gbp': round(len(book['positions']) * EXPOSURE_GBP, 2),
        'win_rate_pct': round(len(wins) / len(rets) * 100, 1) if rets else 0,
        'avg_return_pct': round(sum(rets) / len(rets), 3) if rets else 0,
        'best_pct': round(max(rets), 2) if rets else 0,
        'worst_pct': round(min(rets), 2) if rets else 0,
        'long': _side_stats([t for t in closed if t['side'] == 'LONG']),
        'short': _side_stats([t for t in closed if t['side'] == 'SHORT']),
        'positions': book['positions'], 'closed': closed[-100:],
        'equity_curve': book.get('equity_curve', []),
        'activity': book.get('activity', [])[:40],
    }


# ==================== BACKTEST ====================

def cortex_backtest_trades(market, tickers, years=3,
                           gap_min=DEFAULT_GAP_MIN, gap_max=DEFAULT_GAP_MAX,
                           wing=DEFAULT_WING, exit_day=DEFAULT_EXIT_DAY,
                           use_buckets=False, fee_pct=None,
                           allow_short=True, status=None, histories=None):
    '''
    walks every ticker's daily history and simulates the strategy as the live
    tab runs it: gap at the open, wing over the session, enter at the day-0
    close, exit at the Nth close after.

    returns (trades, equity_curve) in the same shape merlin's backtester uses,
    so _bt_compute_stats and _bt_monte_carlo apply unchanged.
    '''
    #fee_pct None means work the cost out per trade from its side, market and
    #nights held. a number here overrides that with a flat charge, which is
    #what the sweep's sensitivity ladder wants
    if histories is None:
        histories = fetch_history(market, tickers, years=years, status=status)

    cutoff = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=int(years * 365)))
    trades, total = [], max(len(histories), 1)

    for n, (ticker, df) in enumerate(histories.items()):
        if status is not None and n % 25 == 0:
            status['message'] = f'simulating {market} {n}/{total}...'
            status['progress'] = min(90, 45 + int(n / total * 45))
        try:
            dates = df['date'].values
            o = df['open'].values.astype(float)
            h = df['high'].values.astype(float)
            l = df['low'].values.astype(float)
            c = df['close'].values.astype(float)
        except Exception:
            continue

        for i in range(1, len(df) - 1):
            if pd.Timestamp(dates[i]) < cutoff:
                continue
            yc, op = c[i - 1], o[i]
            if not (np.isfinite(yc) and np.isfinite(op)) or yc <= 0 or op <= 0:
                continue
            gap = (op - yc) / yc * 100.0
            a = abs(gap)
            if a < gap_min or a >= gap_max:
                continue

            side = signal_direction(gap)
            if side == 'SHORT' and not allow_short:
                continue

            wing_move = ((h[i] - op) / op * 100.0) if gap > 0 else abs((l[i] - op) / op * 100.0)
            if not np.isfinite(wing_move) or wing_move > wing:
                continue

            entry = c[i]
            if not np.isfinite(entry) or entry <= 0:
                continue
            hold = exit_day_for_gap(a, exit_day, use_buckets)
            j = i + hold
            if j >= len(df):
                continue
            exit_price = c[j]
            if not np.isfinite(exit_price) or exit_price <= 0:
                continue

            gross = trade_return_pct(side, entry, exit_price)
            if gross is None:
                continue
            #10.0: the backtest charges what the book charges. fee_pct is left
            #as an override so the sweep can still run its flat sensitivity
            #ladder, but the default is now the real per trade cost - side,
            #market and calendar nights held, the same function the paper book
            #settles on. before this the backtest priced a US short at 0.25%
            #and the live book would have paid more than twice that
            if fee_pct is None:
                nights = _nights_between(dates[i], dates[j])
                trade_cost = round_trip_cost_pct(side, market, nights)
            else:
                nights, trade_cost = hold, fee_pct
            trades.append({
                'ticker': ticker, 'strategy': f'cortex_{market.lower()}', 'side': side,
                'entry_date': pd.Timestamp(dates[i]).strftime('%Y-%m-%d'),
                'exit_date': pd.Timestamp(dates[j]).strftime('%Y-%m-%d'),
                'entry_price': round(float(entry), 2),
                'exit_price': round(float(exit_price), 2),
                'gross_return_pct': round(float(gross), 2),
                'return_pct': round(float(gross - trade_cost), 2),
                'fees_pct': round(float(trade_cost), 4), 'days_held': hold,
                'nights_held': int(nights),
                'gap_pct': round(float(gap), 2), 'exit_reason': f'scheduled_D{hold}',
            })

    trades.sort(key=lambda t: t['exit_date'])
    equity = [{'date': trades[0]['entry_date'] if trades else _today_iso(),
               'equity': BT_START_CAPITAL}]
    weight = 1.0 / BT_MAX_CONCURRENT
    for t in trades:
        last = equity[-1]['equity']
        equity.append({'date': t['exit_date'],
                       'equity': last * (1 + (t['return_pct'] / 100.0) * weight)})
    return trades, equity


# ==================== PARAMETER SWEEP ====================

sweep_status = {'active': False, 'progress': 0, 'message': '',
                'complete': False, 'results': None, 'error': None}


def run_sweep(market, years=4, count=UNIVERSE_ALL, allow_short=True,
              gap_grid=(5, 6, 7, 8), wing_grid=(2, 3, 4, 5), exit_grid=(1, 2, 3),
              fee_pct=None, status=None):
    '''
    grid search with the sample split in half by time. the in-sample half is
    for choosing, the out-of-sample half is the only number that means
    anything. on the original FTSE data every single top in-sample pick
    flipped to negative out of sample, which is exactly why this reports both
    columns and refuses to rank on the in-sample one.
    '''
    status = status if status is not None else sweep_status
    fee_pct = CORTEX_FEE_PCT if fee_pct is None else fee_pct
    tickers = get_universe(market, count)

    status['message'] = 'loading history...'
    histories = fetch_history(market, tickers, years=years, status=status)
    if not histories:
        return {'error': 'no history could be loaded'}

    all_dates = []
    for df in histories.values():
        if len(df):
            all_dates.append(pd.Timestamp(df['date'].iloc[0]))
            all_dates.append(pd.Timestamp(df['date'].iloc[-1]))
    lo, hi = min(all_dates), max(all_dates)
    split_str = (lo + (hi - lo) / 2).strftime('%Y-%m-%d')

    combos = [(g, w, e) for g in gap_grid for w in wing_grid for e in exit_grid]
    rows, done = [], 0
    for gap_min, wing, exit_day in combos:
        trades, _ = cortex_backtest_trades(
            market, tickers, years=years, gap_min=gap_min, wing=wing,
            exit_day=exit_day, fee_pct=fee_pct, allow_short=allow_short,
            histories=histories)
        done += 1
        status['progress'] = min(99, 45 + int(done / len(combos) * 54))
        status['message'] = f'tested {done}/{len(combos)} combinations'
        if len(trades) < 30:
            continue

        ins = [t['return_pct'] for t in trades if t['entry_date'] < split_str]
        oos = [t['return_pct'] for t in trades if t['entry_date'] >= split_str]
        if len(ins) < 15 or len(oos) < 15:
            continue
        allr = [t['return_pct'] for t in trades]
        rows.append({
            'gap_min': gap_min, 'wing': wing, 'exit_day': exit_day, 'n': len(allr),
            'win_pct': round(len([r for r in allr if r > 0]) / len(allr) * 100, 1),
            'avg_pct': round(float(np.mean(allr)), 3),
            'in_n': len(ins), 'in_avg': round(float(np.mean(ins)), 3),
            'out_n': len(oos), 'out_avg': round(float(np.mean(oos)), 3),
            'out_win_pct': round(len([r for r in oos if r > 0]) / len(oos) * 100, 1),
            'total_pl_pct': round(float(np.sum(allr)) / BT_MAX_CONCURRENT, 1),
        })

    rows.sort(key=lambda r: -r['out_avg'])
    best_is = sorted(rows, key=lambda r: -r['in_avg'])[:5]
    return {
        'market': market, 'years': years, 'fee_pct': fee_pct,
        'allow_short': allow_short, 'split_date': split_str,
        'tickers_evaluated': len(histories), 'rows': rows[:40],
        'in_sample_winners': [
            {'params': f"gap>={r['gap_min']} wing<={r['wing']} D{r['exit_day']}",
             'in_avg': r['in_avg'], 'out_avg': r['out_avg'], 'out_n': r['out_n']}
            for r in best_is],
        'current': next((r for r in rows
                         if r['gap_min'] == DEFAULT_GAP_MIN and r['wing'] == DEFAULT_WING
                         and r['exit_day'] == DEFAULT_EXIT_DAY), None),
    }
