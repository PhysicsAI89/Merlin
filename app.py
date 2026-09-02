'''
merlin 9.0 - multi-strategy signal engine

a local, single-user flask app for personal stock research. sixteen tabs,
each a strategy or a view onto one. data comes from yfinance, a scrape of
openinsider.com, and optionally the IG API for live LSE prices. all state is
json and csv under data/. no auth, no deployment, runs on one machine.

    pip install -r requirements.txt
    python app.py

the strategies, in order of how much the backtest supports them:

    shareholder yield   +129% over 5y, 0% chance of loss     own gold tab
    quality             +72% over 5y, 2.5% chance of loss    own gold tab
    relative momentum   modest                               edge tab
    insider clusters    forward-tested only                  edge tab
    earnings drift      forward-tested only                  edge tab
    cortex gap-fade     marginal, read the disclaimer        violet tabs

cortex lives in cortex.py and is self-contained. this module injects
get_stock_universe into it so it reuses the same US ticker list as every
other tab without importing app.py back, which would be circular.

9.0 changes: market clock strip and signal ticker tape in the header, both
refreshing on their own. every tab now carries its family colour and dot, so
none of them look half-finished. the afternoon cortex run no longer crashes
on the IG parameter rename. the parameter sweep stopped silently trimming
the universe to 400 names. the cortex book shows a message instead of an
empty chart until the first trade settles. IG credentials are read from a
.env file (see IG_SETUP.md) and cortex uses live prices whenever they are
present, falling back to delayed yahoo per ticker whenever they are not.

8.3 changes: cortex added, gap-and-fade on FTSE 350 and the US universe,
with its own paper book, parameter sweep and backtester rows.

8.2 changes: nan poisoning fixed. yfinance sometimes returns an incomplete
final row whose close is nan, and nan passes every `is None` check because
it is a perfectly valid float. one nan price or fx rate flowed into a
position's value, then into cash when positions closed, and from there
every number in the book (equity, sweep, fees) went nan. now every price
and fx fetch rejects non-finite values at source, every cash mutation has
a finite-check circuit breaker, and a one-off startup repair rebuilds a
poisoned book from first principles. also fixed: the 22:00 auto-run
scheduler never actually started - it was gated behind a werkzeug reloader
env var that never exists with use_reloader=False.

8.1 changes: fixed the beats-SPY column showing 0% (yfinance's incomplete
last row made the SPY total NaN, every comparison against NaN is False and
the json layer nulled the benchmark line). SPY buy-and-hold now appears as
its own row in backtest results with a viewable equity curve. momentum_12_1
retired from live trading after the 5y backtest (+8.31% total, 44% chance
of loss, sharpe 0.05 over 729 trades - fails the same bar that binned
meanrev and friends in 7.8). caps and score bonuses rebalanced toward the
proven quality and shareholder yield edges. dedicated shareholder yield
scanner tab. new vs SPY tab comparing the paper book against the index.

8.0 changes: starting capital raised to £10,000 with a one-off migration that
closes the old book, live paper trader now pays realistic broker/fx fees on
every fill, idle cash is swept into SPY during healthy regimes so the bot is
never dragged down by cash earning 0%, position sizes are volatility-scaled,
momentum signals get a frog-in-the-pan smoothness bonus, and the backtester
gained a monte carlo bootstrap plus a SPY benchmark line.
'''

import os, json, datetime, re, time, threading, warnings

#9.1: trust the certificate authorities windows already trusts. antivirus
#https scanning re-signs every tls connection with a locally generated root,
#which windows accepts and certifi has never heard of, so every yfinance call
#fails with "unable to get local issuer certificate" and looks for all the
#world like the network is down. this must run before yfinance is imported.
#see net_trust.py - it is a no-op on any machine that does not need it.
import net_trust
net_trust.install()

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import (LSTM, Dense, Dropout, Bidirectional,
                          Conv1D, GRU, BatchNormalization)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
warnings.filterwarnings('ignore')
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

#10.0: one module that knows when a company next reports. the paper trader
#uses it to stand off before a print, cortex to mark earnings gaps, the pdf
#report and the scan tables to say it out loud. see earnings.py
import earnings


#==================== .env ====================
#
#IG credentials for cortex live prices, read from a .env file in the project
#root so they never end up in the source. see ig_config.py for the format and
#the reasoning. merlin runs perfectly well without it - cortex just stays on
#delayed yahoo prices and nothing else notices.

import ig_config

ENV_PATH = ig_config.ENV_PATH
_env_keys_loaded = ig_config.load_env_file()


#paper trader constants
PAPER_PORTFOLIO_PATH = os.path.join('data', 'paper_portfolio.json')
STARTING_CASH_GBP = 10000.0
MAX_POSITIONS = 8
MIN_POSITION_PCT = 0.05
MAX_POSITION_PCT = 0.15
MIN_CONFIDENCE_TO_OPEN = 65
MIN_TRADE_GBP = 30.0
CASH_BUFFER_PCT = 0.99

# False = do exit checks only unless there is enough cash for a normal slot.
# True = allow tiny remainder trades using leftover cash.
ALLOW_REMAINDER_TRADES = False

# Require at least 90% of a normal slot before doing the expensive full entry scan.
MIN_ENTRY_SLOT_COVERAGE = 0.90

#AI paper trader allocation rules. The cycle may find dozens of signals,
#especially momentum-style ones, so we cap each strategy per cycle and rank
#by confidence with a small evidence-priority bonus. This stops one noisy
#strategy filling all 8 slots.
#
#7.8 change: removed meanrev, week52_high and low_beta_trend. The 5y backtest
#showed all three underperformed SPY (+91%) by a huge margin: meanrev +15.87%,
#week52_high +16.17%, low_beta_trend +8.81%. Anything that can't beat the
#index over 5y in a mostly bullish window does not earn a slot.
#8.1: momentum_12_1 retired after the 5y backtest: +8.31% total, monte carlo
#median +7.48% with a 44.1% chance of loss and a -45% 5th percentile, sharpe
#0.05 across 729 trades after fees. that fails the exact bar that binned
#meanrev, week52_high and low_beta_trend in 7.8. relative momentum stays on
#probation with reduced caps and no score bonus - it was weak (+25.78%) but
#robust (15.4% chance of loss) and the 8.0 regime gate plus smoothness bonus
#are not reflected in that backtest. quality (+72%, 2.5% chance of loss) and
#shareholder yield (+129%, 0% chance of loss in 1000 reshuffles) earned
#bigger allocations.
MAX_NEW_POSITIONS_PER_STRATEGY = {
    'cluster': 2,
    'pead': 2,
    'momentum': 2,          #relative 12-1 momentum, on probation
    'quality': 3,
    'shareholder_yield': 3,
}
STRATEGY_SCORE_BONUS = {
    'cluster': 4,
    'pead': 3,
    'momentum': 0,
    'quality': 2,
    'shareholder_yield': 3,
}
#group caps are stricter than per-strategy caps. This is the important part:
#momentum-style signals can throw 20+ names, so the AI can only open 3 total
#from the whole momentum group in one cycle.
STRATEGY_GROUP = {
    'cluster': 'event',
    'pead': 'event',
    'momentum': 'momentum',
    'quality': 'fundamental',
    'shareholder_yield': 'fundamental',
}
MAX_NEW_POSITIONS_PER_GROUP = {
    'event': 3,
    'momentum': 2,
    'fundamental': 4,
}
#portfolio-wide caps: total positions of each group held at any time across
#all cycles. without this, momentum-style signals can quietly fill 7-8 slots
#over a few weeks and leave no room for the event-driven edges when they
#finally appear. this enforces structural diversification.
MAX_PORTFOLIO_POSITIONS_PER_GROUP = {
    'event': 4,
    'momentum': 2,
    'fundamental': 5,
}

#list of strategies that were once supported but have been removed. used on
#startup to clean up any open paper positions still tagged with them, so
#the bot does not keep holding stocks under exit rules that no longer exist.
RETIRED_STRATEGIES = ['meanrev', 'week52_high', 'low_beta_trend', 'momentum_12_1']

#7.9 additions
#
#hard regime gate for the AI paper trader. when the current regime is in
#this set, the trader will NOT open new momentum-group positions, no matter
#how high the confidence score. existing momentum positions still close
#through their own trailing-stop rules so we do not force-sell into a panic.
#this hardens the previous soft -8 suppression which the bot could still
#override when nothing else looked good.
MOMENTUM_HOSTILE_REGIMES = {'bear_calm', 'bear_volatile', 'bull_volatile'}
MOMENTUM_STRATEGIES = {'momentum'}

#round-trip transaction cost assumption used by the backtester. covers
#bid-ask spread plus Trading 212 FX conversion fee on USD positions plus
#a small allowance for slippage. set to zero to see pre-cost numbers.
#  0.5% round trip = ~0.25% in + ~0.25% out
#  reasonable estimate for retail UK->US trading on a zero-commission broker
FEES_ROUND_TRIP_PCT = 0.5

#8.0: the live paper trader now pays the same costs a real Trading 212
#account would, charged per side (entry pays, exit pays):
#  fx fee 0.15% on any non-gbp position (the t212 currency conversion charge)
#  spread/slippage allowance 0.10% on everything
#a usd round trip therefore costs 0.5%, which matches FEES_ROUND_TRIP_PCT in
#the backtester, so live paper results and backtests stay directly comparable
FX_FEE_PCT = 0.15
SPREAD_SLIPPAGE_PCT = 0.10

#8.0: schema stamp on the paper portfolio file. on startup any file without
#this stamp gets migrated: open positions closed at market, the old book
#archived to data/paper_portfolio_v7_archive.json and a fresh £10,000
#portfolio created
PORTFOLIO_SCHEMA_VERSION = '8.0'

#8.0: index cash sweep. idle cash is the silent killer when the target is
#beating SPY - money earning 0% drags the whole book down between signals.
#when the regime says the market is in a healthy uptrend, spare cash is
#swept into SPY as a special position tagged index_sweep. it does not take
#up one of the 8 strategy slots and it is sold down automatically when a
#real signal needs the cash or the regime turns hostile. this is the classic
#200dma timing rule (faber 2007) applied to the cash pile
INDEX_SWEEP_ENABLED = True
INDEX_SWEEP_TICKER = 'SPY'
INDEX_SWEEP_REGIMES = {'strong_bull', 'bull_normal', 'narrow_bull', 'calm_chop'}
INDEX_SWEEP_MIN_GBP = 50.0
INDEX_SWEEP_CASH_FLOOR_GBP = 25.0

#8.0: volatility-scaled position sizing. two signals with the same confidence
#should not get the same £ if one of them moves three times as much per day.
#the confidence-based size is tilted by target_vol / realised_vol and clamped
#so the tilt stays moderate. calm names get a bigger slice, jumpy names a
#smaller one, which lifts risk-adjusted return without changing the signals
TARGET_POSITION_VOL = 0.30
VOL_SIZE_MIN_MULT = 0.6
VOL_SIZE_MAX_MULT = 1.3

#10.0: earnings blackout for the slow strategies.
#
#quality and shareholder yield are three to six month holds bought on gross
#profitability and on cash handed back. neither of those changes on a tuesday
#morning, but a print does, and a factor position opened four days before one
#is not the position that was backtested - it is a coin flip on the number,
#with the factor thesis along for the ride. so the trader stands off until
#the print is out of the way and the name comes back on the next cycle.
#
#the event strategies are deliberately not in this set. pead exists precisely
#to trade the days after a result, and an insider cluster that forms into a
#print is if anything a stronger signal, not a weaker one.
#
#set to 0 to switch the whole thing off. an unknown earnings date never
#blocks anything - see FAIL_OPEN in earnings.py for why
EARNINGS_BLACKOUT_DAYS = 5
EARNINGS_BLACKOUT_STRATEGIES = {'quality', 'shareholder_yield', 'momentum', 'momentum_12_1'}

app = Flask(__name__)

class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj): return None
            return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        return super().default(obj)

app.json_encoder = SafeJSONEncoder

def sanitise(obj):
    '''recursively replace nan/inf with None'''
    if isinstance(obj, dict): return {k: sanitise(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [sanitise(v) for v in obj]
    elif isinstance(obj, float):
        if obj != obj or obj == float('inf') or obj == float('-inf'): return None
        return obj
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.bool_,)): return bool(obj)
    return obj

training_status = {'active':False,'progress':0,'message':'','ticker':'','complete':False,'error':None,'backtest':None}
screener_status = {'active':False,'progress':0,'message':'','complete':False,'results':[],'error':None}
pead_status = {'active':False,'progress':0,'message':'','complete':False,'results':[],'error':None}
trade_recommendations = []

MODELS_DIR = 'models'
DATA_DIR = 'data'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

SEQUENCE_LENGTH = 60
NUM_ENSEMBLE = 3
TECH_MAX_SCORE = 13  #approximate max from technicals

# ==================== MARKET CONTEXT (new in 7.1) ====================

SECTOR_ETF_MAP = {
    'Technology':'XLK','Financial Services':'XLF','Healthcare':'XLV',
    'Consumer Cyclical':'XLY','Industrials':'XLI','Communication Services':'XLC',
    'Consumer Defensive':'XLP','Energy':'XLE','Real Estate':'XLRE',
    'Basic Materials':'XLB','Utilities':'XLU',
}
MARKET_FEATURE_COLS = ['SPY_Ret','SPY_SMA_Ratio','VIX_Level','VIX_Change','Sector_Ret','Sector_Relative']


def get_sector_etf(ticker):
    '''lookup the sector etf for a given ticker via yfinance info'''
    try:
        sector = (yf.Ticker(ticker).info or {}).get('sector', '')
        return SECTOR_ETF_MAP.get(sector, 'SPY')
    except:
        return 'SPY'


def fetch_market_context(start, end, sector_etf='SPY'):
    '''
    fetch SPY, VIX and the sector etf, return a dataframe aligned by date.
    these features tell the model what the wider market is doing, which often
    matters more than the stock's own technicals.
    '''
    try:
        spy = drop_incomplete_bars(yf.download('SPY', start=start, end=end, interval='1d', progress=False))
        vix = drop_incomplete_bars(yf.download('^VIX', start=start, end=end, interval='1d', progress=False))
        sec = (yf.download(sector_etf, start=start, end=end, interval='1d', progress=False)
               if sector_etf != 'SPY' else spy)
        for d in (spy, vix, sec):
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
        if spy.empty or vix.empty:
            return None
        sc, vc, ec = spy['Close'], vix['Close'], sec['Close']
        ctx = pd.DataFrame(index=spy.index)
        ctx['SPY_Ret'] = sc.pct_change().fillna(0)
        ctx['SPY_SMA_Ratio'] = (sc / sc.rolling(20, min_periods=1).mean() - 1).fillna(0)
        ctx['VIX_Level'] = (vc / 20.0 - 1).fillna(0)  #normalised around long-run avg ~20
        ctx['VIX_Change'] = vc.pct_change().fillna(0)
        ctx['Sector_Ret'] = ec.pct_change().fillna(0)
        return ctx
    except Exception as e:
        print(f'warning: market context fetch failed: {e}\n')
        return None


def add_market_features(df, market_ctx):
    '''merge market context into the indicator df, fill any gaps with zero'''
    if market_ctx is None:
        for c in MARKET_FEATURE_COLS:
            df[c] = 0.0
        return df
    merged = df.join(market_ctx, how='left')
    for c in ['SPY_Ret','SPY_SMA_Ratio','VIX_Level','VIX_Change','Sector_Ret']:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = merged[c].fillna(0)
    #stock return relative to its sector - a "did the stock beat its peers today" signal
    if 'Returns' in merged.columns:
        merged['Sector_Relative'] = merged['Returns'] - merged['Sector_Ret']
    else:
        merged['Sector_Relative'] = 0.0
    return merged


def make_verdict(ensemble_acc, naive_up, persistence, xgb_dict):
    '''
    compare ensemble against the best baseline. honest assessment for the user
    so they know if the deep models are actually earning their keep.
    '''
    best = max(naive_up, persistence)
    if xgb_dict and xgb_dict.get('held_out_test') is not None:
        best = max(best, xgb_dict['held_out_test'])
    margin = ensemble_acc - best
    if margin > 5:
        return {'rating':'strong','message':f'ensemble beats best baseline by {margin:.1f}% - genuinely adding value','best_baseline_acc':round(best,1),'margin':round(margin,1)}
    if margin > 1:
        return {'rating':'modest','message':f'ensemble edges out baselines by {margin:.1f}% - real but small gain','best_baseline_acc':round(best,1),'margin':round(margin,1)}
    if margin > -2:
        return {'rating':'matches_baseline','message':f'ensemble matches baselines (within {abs(margin):.1f}%) - the deep models are not earning their keep','best_baseline_acc':round(best,1),'margin':round(margin,1)}
    return {'rating':'underperforms','message':f'ensemble loses to a baseline by {abs(margin):.1f}% - simpler model would be better here','best_baseline_acc':round(best,1),'margin':round(margin,1)}


# ==================== OPENINSIDER SCRAPER ====================

def scrape_openinsider(trade_type='buy', min_value=10000, days=7, ceo_cfo_only=True, count=100):
    '''
    scrape openinsider.com using pandas.read_html which is far more
    robust at finding html tables than beautifulsoup class matching
    '''
    try:
        xp = '1' if trade_type in ('buy', 'both') else ''
        xs = '1' if trade_type in ('sell', 'both') else ''
        isceo = '1' if ceo_cfo_only else ''
        iscfo = '1' if ceo_cfo_only else ''
        #9.1: openinsider's vl filter is in THOUSANDS of dollars, not dollars.
        #merlin was passing 10000 meaning "$10,000" and openinsider read it as
        #$10,000,000, which threw away all but the very largest filings - the
        #scrape came back with 50 rows instead of 1000 and the cluster engine,
        #which needs three separate insiders on one ticker, could never find
        #a single one. divide before sending.
        vl = str(max(1, int(min_value / 1000))) if min_value else ''

        url = (
            f'http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh='
            f'&fd={days}&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago='
            f'&xp={xp}&xs={xs}&vl={vl}&vh=&ocl=&och='
            f'&sic1=-1&sicl=100&sich=9999'
            f'&isceo={isceo}&iscfo={iscfo}'
            f'&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh='
            f'&v2l=&v2h=&oc2l=&oc2h=&sortcol=1&cnt={count}&page=1'
        )

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }

        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            return {'trades': [], 'error': f'openinsider returned status {resp.status_code}', 'count': 0}

        #use pandas to find all tables in the page
        try:
            dfs = pd.read_html(resp.text)
        except ValueError:
            return {'trades': [], 'error': 'no tables found on page', 'count': 0}

        if not dfs:
            return {'trades': [], 'error': 'no tables found', 'count': 0}

        #find the biggest table (the data table is always the largest)
        df = max(dfs, key=len)

        if len(df) < 1:
            return {'trades': [], 'error': 'table was empty', 'count': 0}

        #normalise column names to lowercase for matching
        df.columns = [str(c).lower().strip() for c in df.columns]

        #openinsider columns vary but we need to find these key ones
        #try to identify columns by name patterns
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if 'filing' in cl and 'date' in cl: col_map['filing_date'] = c
            elif 'trade' in cl and 'date' in cl: col_map['trade_date'] = c
            elif cl == 'ticker' or 'ticker' in cl: col_map['ticker'] = c
            elif 'insider' in cl and 'name' in cl: col_map['insider_name'] = c
            elif cl == 'title': col_map['title'] = c
            elif 'trade' in cl and 'type' in cl: col_map['trade_type'] = c
            elif cl == 'price': col_map['price'] = c
            elif cl == 'qty': col_map['qty'] = c
            elif cl == 'owned': col_map['owned'] = c
            elif cl == 'value': col_map['value'] = c
            elif cl in ('δown', 'own', 'Δown') or 'own' in cl and cl != 'owned': col_map['delta_own'] = c

        #if we couldn't match by name, try by position (openinsider's known order)
        #columns: X, Filing Date, Trade Date, Ticker, Company Name, Insider Name, Title, Trade Type, Price, Qty, Owned, ΔOwn, Value, 1d, 1w, 1m, 6m
        cols = list(df.columns)
        if 'ticker' not in col_map and len(cols) >= 12:
            #positional fallback
            col_map = {
                'filing_date': cols[1] if len(cols) > 1 else None,
                'trade_date': cols[2] if len(cols) > 2 else None,
                'ticker': cols[3] if len(cols) > 3 else None,
                'company': cols[4] if len(cols) > 4 else None,
                'insider_name': cols[5] if len(cols) > 5 else None,
                'title': cols[6] if len(cols) > 6 else None,
                'trade_type': cols[7] if len(cols) > 7 else None,
                'price': cols[8] if len(cols) > 8 else None,
                'qty': cols[9] if len(cols) > 9 else None,
                'owned': cols[10] if len(cols) > 10 else None,
                'delta_own': cols[11] if len(cols) > 11 else None,
                'value': cols[12] if len(cols) > 12 else None,
            }

        trades = []
        for _, row in df.iterrows():
            try:
                ticker = str(row.get(col_map.get('ticker', ''), '')).strip().upper()
                if not ticker or ticker == 'NAN' or len(ticker) > 6:
                    continue

                trade_type_str = str(row.get(col_map.get('trade_type', ''), ''))
                tt_lower = trade_type_str.lower()
                if 'purchase' in tt_lower or 'buy' in tt_lower:
                    action = 'buy'
                elif 'sale' in tt_lower or 'sell' in tt_lower:
                    action = 'sell'
                else:
                    action = 'other'

                if action == 'other':
                    continue

                #parse price
                price_raw = str(row.get(col_map.get('price', ''), '0'))
                price_raw = re.sub(r'[^\d.]', '', price_raw)
                try: price = float(price_raw)
                except: price = 0

                #parse qty
                qty_raw = str(row.get(col_map.get('qty', ''), '0'))
                qty_raw = re.sub(r'[^\d]', '', qty_raw)
                try: qty = int(qty_raw) if qty_raw else 0
                except: qty = 0

                #parse value
                value_raw = str(row.get(col_map.get('value', ''), '0'))
                value_raw = re.sub(r'[^\d.]', '', value_raw)
                try: value = float(value_raw)
                except: value = 0

                #parse dates
                filing_date = str(row.get(col_map.get('filing_date', ''), ''))[:19]
                trade_date = str(row.get(col_map.get('trade_date', ''), ''))[:10]
                insider_name = str(row.get(col_map.get('insider_name', ''), ''))[:35]
                title = str(row.get(col_map.get('title', ''), ''))[:25]

                #parse delta own
                delta_raw = str(row.get(col_map.get('delta_own', ''), '0'))
                delta_raw = re.sub(r'[^\d.\-]', '', delta_raw)
                try: delta_own = float(delta_raw)
                except: delta_own = 0

                trades.append({
                    'filing_date': filing_date,
                    'trade_date': trade_date,
                    'ticker': ticker,
                    'insider_name': insider_name,
                    'title': title,
                    'action': action,
                    'trade_type': trade_type_str[:20],
                    'price': round(price, 2),
                    'qty': qty,
                    'delta_own': round(delta_own, 1),
                    'value': round(value, 0)
                })
            except:
                continue

        #sort by value descending
        trades.sort(key=lambda x: abs(x.get('value', 0)), reverse=True)

        return {
            'trades': trades,
            'count': len(trades),
            'filters': {'trade_type': trade_type, 'min_value': min_value, 'days': days, 'ceo_cfo_only': ceo_cfo_only}
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return {'trades': [], 'error': str(e), 'count': 0}


# ==================== EXPANDED STOCK UNIVERSE ====================

def get_stock_universe(count=500):
    '''
    get a list of stock tickers. uses a hardcoded core list
    plus dynamically fetches s&p 500 from wikipedia for larger scans
    '''
    core = [
        'AAPL','MSFT','AMZN','NVDA','GOOGL','GOOG','META','TSLA','BRK-B','UNH','XOM',
        'JNJ','JPM','V','PG','MA','AVGO','HD','CVX','MRK','ABBV','LLY','COST','PEP',
        'KO','ADBE','WMT','MCD','CSCO','CRM','ACN','TMO','ABT','DHR','NKE','NEE','LIN',
        'TXN','PM','UNP','RTX','LOW','QCOM','HON','INTC','INTU','AMAT','ISRG','AMGN',
        'BKNG','GS','CAT','BLK','AXP','BA','SBUX','GE','IBM','DIS','AMD','PYPL','SHOP',
        'PLTR','COIN','SOFI','F','GM','T','VZ','NFLX','UBER','ABNB','RBLX','DDOG','NET',
        'CRWD','PANW','NOW','WDAY','SNAP','MELI','BABA','JD','NIO','TSM','SONY','BP',
        'SHEL','GOLD','NEM','FCX','WFC','BAC','C','MS','SCHW','PFE','BMY','GILD','MRNA',
        'VRTX','SYK','MDT','CI','CVS','DE','ENPH','ARM','SMCI','MU','MRVL','ON','KLAC',
        'LRCX','ASML','WM','SHW','APD','PSA','O','VICI','ED','SO','DUK','AEP','XEL',
        'DKNG','MGM','AMC','GME','CELH','MNST','BROS','SPY','QQQ','DIA','IWM','ARKK',
        'XLF','XLE','XLV','RIVN','LCID','HOOD','SQ','ZM','ROKU','SPOT','TTWO','EA',
        'ATVI','RBLX','U','PINS','TWLO','SE','PDD','LI','XPEV','RIO','VALE','BHP',
        'AA','CLF','NUE','STLD','X','PNC','TFC','USB','AIG','MET','PRU','ALL','TRV',
        'CB','AFL','BIIB','REGN','ZTS','BDX','BSX','EW','HUM','CNC','ELV','WBA','MCK',
        'CAH','CNH','MOS','CF','ADM','TSN','GIS','K','CPB','CLX','CL','EL','CHD',
        'PLUG','FCEL','SEDG','RUN','CHPT','BLNK','WKHS','HYLN','DKNG','PENN','WYNN',
        'LVS','CZR','CLOV','OPEN','RKT','ASTS','OLPX','BIRD','FIGS','OATLY',
        'MCHP','SWKS','WCN','ECL','DD','PPG','BALL','PKG','IP','AVY','BLL',
        'EQR','AVB','UDR','MAA','ESS','INVH','SUI','NNN','WPC','ADC','GLPI',
        'WEC','CMS','AES','EIX','PCG','FIS','FISV','GPN','ADP','PAYX','CDNS',
        'SNPS','ANSS','KEYS','TDY','ZBRA','FTNT','ZS','OKTA','HUBS','TTD',
        'DOCU','BILL','CFLT','MDB','SNOW','ESTC','DDOG','S','MNDY','GTLB',
        'PATH','AI','IONQ','RGTI','QBTS','BBAI','SOUN','JOBY','LILM',
        'SOFI','AFRM','UPST','LC','NU','GRAB','CPNG','GLBE','TOST','CAVA',
        'BIRK','DUOL','CART','RDDT','APP','DASH','LYFT','ABNB','EXPE','MAR',
        'HLT','H','WH','IHG','RCL','CCL','NCLH','LUV','DAL','UAL','AAL',
        'FDX','UPS','XPO','CHRW','JBHT','ODFL','SAIA','KNX','WERN',
        'CMG','DPZ','YUM','QSR','SBUX','DRI','TXRH','EAT','CAKE','BJRI',
        'TGT','COST','DG','DLTR','FIVE','ROST','TJX','BURL','GPS','ANF',
        'LULU','NKE','UAA','CROX','ONON','DECK','SKX','VFC','HBI',
        'TSCO','ORLY','AZO','AAP','GPC','LKQ','MNRO',
        'COP','EOG','DVN','FANG','PXD','OXY','MPC','VLO','PSX','HES',
        'SLB','HAL','BKR','OKE','WMB','KMI','ET','EPD','MPLX','PAA',
        'ICE','CME','NDAQ','CBOE','MSCI','SPGI','MCO','FDS','VRSK',
        'AME','ROK','EMR','ETN','PH','DOV','ITW','SWK','IR','XYL',
        'A','WAT','TMO','DHR','IQV','CRL','MTD','PKI','BIO','TECH',
        'ZBH','STE','HOLX','ALGN','DXCM','PODD','ISRG','INSP','IRTC',
        'VEEV','HIMS','DOCS','TDOC','AMWL','GH','EXAS','ILMN','PACB',
        'NVAX','BNTX','MRNA','SRRK','VRTX','ALNY','BMRN','SGEN','EXEL',
        'PCVX','RARE','IONS','SRPT','UTHR','NBIX','INCY','ARGX',
        'PANW','FTNT','CRWD','ZS','S','OKTA','QLYS','TENB','RPD','VRNS',
        'CHKP','CYBR','SAIL','DDOG','DT','ESTC','NEWR','SUMO','PD',
        'WOOF','CHWY','ZM','FVRR','UPWK','ETSY','W','CVNA','RVLV','POSH',
        'REAL','GRPN','WISH','OSTK','BIGC','SHOP','WIX','SQSP','GDDY',
        'NET','FSLY','CDN','AKAM','LLAP','RKLB','LUNR','IRDM','BWXT',
        'NOC','LMT','RTX','GD','HII','TXT','LHX','LDOS','BAH','SAIC',
        'CARR','OTIS','JCI','TT','GNRC','SEDG','ENPH','FSLR','ARRY',
        'RUN','NOVA','MAXN','SHLS','STEM','BE','PLUG','FCEL','BLDP',
        'SLI','ALB','LAC','LTHM','PLL','MP','UUUU','CCJ','LEU','NXE',
        'GOLD','AEM','NEM','FNV','WPM','RGLD','AG','HL','CDE','MAG',
        'CLF','VALE','RIO','BHP','SCCO','FCX','TECK','AA','CENX','ACH',
        'CF','NTR','MOS','FMC','CTVA','DE','AGCO','CNHI','TTC','LII',
        'WSO','AAON','AZEK','TREX','BECN','BLD','OC','VMC','MLM','CX',
        'SUM','EXP','USCR','APOG','GMS','AWI','DOOR','JELD','MAS','PHM',
        'DHI','LEN','NVR','KBH','MDC','MHO','CCS','TOL','MTH','GRBK',
        'DLR','EQIX','CCI','AMT','SBAC','UNIT','LUMN','DISH','TMUS',
        'CHTR','CMCSA','PARA','WBD','FOX','FOXA','NWSA','NWS','NYT',
        'EB','LYV','MSGS','MSGE','DKNG','FLUT','MGM','BYD','WYNN',
    ]

    #deduplicate
    seen = set()
    unique = []
    for t in core:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    #if they want more than we have hardcoded, try fetching s&p 500 from wikipedia
    if count > len(unique):
        try:
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            sp_tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()
            for t in sp_tickers:
                if t not in seen:
                    seen.add(t)
                    unique.append(t)
        except:
            pass

    #try nasdaq 100 too
    if count > len(unique):
        try:
            ndx = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]
            ndx_tickers = ndx['Ticker'].tolist()
            for t in ndx_tickers:
                if t not in seen:
                    seen.add(t)
                    unique.append(t)
        except:
            pass

    return unique[:count]


# ==================== TECHNICALS ====================

def drop_incomplete_bars(df):
    '''
    9.1: drop any daily row that has no close.

    yfinance appends a final row carrying volume but no open, high, low or
    close while a session is still settling, and sometimes leaves one behind
    afterwards. every indicator below fills missing values with zero, so that
    row becomes a bar priced at nothing: the short moving averages sag, the
    bollinger position goes negative, the last close reads as zero and the
    penny-stock gates in the factor engines throw the ticker out. nothing
    raises, so it all looks like a real answer. it is not, and it hit LSE and
    US names alike depending on the time of day.

    this is the single place to stop it, because every technical path in
    merlin flows through compute_indicators.
    '''
    if df is None or len(df) == 0:
        return df
    for col in ('Close', 'close'):
        if col in df.columns:
            try:
                return df[df[col].notna()]
            except Exception:
                return df
    return df


def _dividend_yield_pct(info):
    '''
    9.1: dividend yield as a percentage, whichever way yfinance reports it.

    the dividendYield field changed from a fraction (0.035) to a percentage
    (3.5). code that still multiplies by a hundred turns a 3.5% payer into a
    350% one, and the "below one means it is a fraction" guess turns a 0.7%
    payer into a 71% one - which is how visa came to look like the best
    shareholder yield name in the universe.

    trailingAnnualDividendYield is still a fraction, so where both fields
    exist they settle the units between them instead of anybody guessing.
    '''
    def _f(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    dy = _f(info.get('dividendYield'))
    reference = _f(info.get('trailingAnnualDividendYield'))

    if dy is None or dy <= 0:
        #fall back to the fraction field when the headline one is missing
        return round(reference * 100, 3) if reference and reference > 0 else 0.0

    if reference and reference > 0:
        ref_pct = reference * 100.0
        #whichever reading of dy sits closer to the cross-check wins
        as_pct, as_fraction = dy, dy * 100.0
        if abs(as_pct - ref_pct) <= abs(as_fraction - ref_pct):
            return round(as_pct, 3)
        return round(as_fraction, 3)

    #no cross-check available. current yfinance hands this out as a
    #percentage, so take it at face value rather than guessing from size
    return round(dy, 3)


def compute_indicators(df):
    df = drop_incomplete_bars(df)
    df = df.copy()
    c = df['Close'].values.flatten().astype(float)
    h = df['High'].values.flatten().astype(float)
    l = df['Low'].values.flatten().astype(float)
    v = df['Volume'].values.flatten().astype(float)

    for w in [5,10,20,50]:
        df[f'SMA_{w}'] = pd.Series(c).rolling(window=w, min_periods=1).mean().values
    df['EMA_12'] = pd.Series(c).ewm(span=12, adjust=False).mean().values
    df['EMA_26'] = pd.Series(c).ewm(span=26, adjust=False).mean().values
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = pd.Series(df['MACD'].values).ewm(span=9, adjust=False).mean().values
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    delta = pd.Series(c).diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    df['RSI'] = (100 - (100 / (1 + gain/(loss+1e-10)))).values

    low14 = pd.Series(l).rolling(14, min_periods=1).min()
    high14 = pd.Series(h).rolling(14, min_periods=1).max()
    df['Stoch_K'] = (100*(pd.Series(c)-low14)/(high14-low14+1e-10)).values
    df['Stoch_D'] = pd.Series(df['Stoch_K']).rolling(3, min_periods=1).mean().values
    df['Williams_R'] = (-100*(high14-pd.Series(c))/(high14-low14+1e-10)).values

    sma20 = pd.Series(c).rolling(20, min_periods=1).mean()
    std20 = pd.Series(c).rolling(20, min_periods=1).std().fillna(0)
    df['BB_Upper'] = (sma20+2*std20).values
    df['BB_Lower'] = (sma20-2*std20).values
    df['BB_Width'] = ((df['BB_Upper']-df['BB_Lower'])/(sma20.values+1e-10))
    df['BB_Position'] = ((pd.Series(c)-df['BB_Lower'].values)/(df['BB_Upper'].values-df['BB_Lower'].values+1e-10)).values

    tr = pd.concat([pd.Series(h)-pd.Series(l), abs(pd.Series(h)-pd.Series(c).shift(1)), abs(pd.Series(l)-pd.Series(c).shift(1))], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14, min_periods=1).mean().values

    obv = [0.0]
    for i in range(1, len(c)):
        obv.append(obv[-1] + (v[i] if c[i]>c[i-1] else -v[i] if c[i]<c[i-1] else 0))
    df['OBV'] = obv
    df['OBV_SMA'] = pd.Series(obv).rolling(20, min_periods=1).mean().values

    df['Momentum_5'] = pd.Series(c).pct_change(5).fillna(0).values
    df['Momentum_10'] = pd.Series(c).pct_change(10).fillna(0).values
    df['Momentum_20'] = pd.Series(c).pct_change(20).fillna(0).values
    vol_sma = pd.Series(v).rolling(20, min_periods=1).mean()
    df['Vol_Ratio'] = (pd.Series(v)/(vol_sma+1e-10)).values
    df['Vol_Change'] = pd.Series(v).pct_change().fillna(0).values
    df['Returns'] = pd.Series(c).pct_change().fillna(0).values
    df['Returns_5d'] = pd.Series(c).pct_change(5).fillna(0).values
    df['Volatility_10'] = pd.Series(c).pct_change().rolling(10, min_periods=1).std().fillna(0).values
    df['Volatility_20'] = pd.Series(c).pct_change().rolling(20, min_periods=1).std().fillna(0).values
    df['Price_to_SMA20'] = (pd.Series(c)/(df['SMA_20'].values+1e-10)-1).values
    df['Price_to_SMA50'] = (pd.Series(c)/(df['SMA_50'].values+1e-10)-1).values
    df['SMA_Cross_5_20'] = (df['SMA_5']-df['SMA_20']).values
    df['SMA_Cross_20_50'] = (df['SMA_20']-df['SMA_50']).values
    df['Trend_Slope_5'] = pd.Series(c).rolling(5, min_periods=1).apply(lambda x: np.polyfit(range(len(x)),x,1)[0] if len(x)>1 else 0, raw=True).values
    df['Trend_Slope_10'] = pd.Series(c).rolling(10, min_periods=1).apply(lambda x: np.polyfit(range(len(x)),x,1)[0] if len(x)>1 else 0, raw=True).values

    if hasattr(df.index, 'dayofweek'):
        dow, month = df.index.dayofweek, df.index.month
    else:
        dow, month = pd.Series([0]*len(df)), pd.Series([1]*len(df))
    df['Day_Sin'] = np.sin(2*np.pi*dow/5)
    df['Day_Cos'] = np.cos(2*np.pi*dow/5)
    df['Month_Sin'] = np.sin(2*np.pi*month/12)
    df['Month_Cos'] = np.cos(2*np.pi*month/12)

    return df.replace([np.inf,-np.inf], 0).fillna(0)


FEATURE_COLS = [
    'Close','High','Low','Open','Volume','SMA_5','SMA_10','SMA_20','SMA_50',
    'EMA_12','EMA_26','MACD','MACD_Signal','MACD_Hist','RSI','Stoch_K','Stoch_D','Williams_R',
    'BB_Upper','BB_Lower','BB_Width','BB_Position','ATR','OBV','OBV_SMA',
    'Momentum_5','Momentum_10','Momentum_20','Vol_Ratio','Vol_Change','Returns','Returns_5d',
    'Volatility_10','Volatility_20','Price_to_SMA20','Price_to_SMA50',
    'SMA_Cross_5_20','SMA_Cross_20_50','Trend_Slope_5','Trend_Slope_10',
    'Day_Sin','Day_Cos','Month_Sin','Month_Cos',
    #market context features (new in 7.1)
    'SPY_Ret','SPY_SMA_Ratio','VIX_Level','VIX_Change','Sector_Ret','Sector_Relative'
]

def prepare_features(df):
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].replace([np.inf,-np.inf],0).fillna(0).copy(), available


# ==================== FUNDAMENTALS ====================

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info or {}
        mc = info.get('marketCap',0)
        pe = info.get('trailingPE', info.get('forwardPE'))
        fpe = info.get('forwardPE')
        peg = info.get('pegRatio')
        beta = info.get('beta')
        dy = _dividend_yield_pct(info)   #9.1: units settled in one place
        av = info.get('averageVolume',0)
        h52 = info.get('fiftyTwoWeekHigh',0)
        l52 = info.get('fiftyTwoWeekLow',0)
        cp = info.get('currentPrice', info.get('regularMarketPrice',0))
        rg = info.get('revenueGrowth')
        pm = info.get('profitMargins')
        dte = info.get('debtToEquity')
        roe = info.get('returnOnEquity')
        fcf = info.get('freeCashflow',0)
        eg = info.get('earningsGrowth')
        ptb = info.get('priceToBook')
        sector = info.get('sector','unknown')
        industry = info.get('industry','unknown')
        name = info.get('shortName', info.get('longName',ticker))

        rp = (cp-l52)/(h52-l52+1e-10) if h52 and l52 and cp else 0.5

        if mc>=1e12: mcs,mcc = f'${mc/1e12:.2f}T','mega-cap'
        elif mc>=10e9: mcs,mcc = f'${mc/1e9:.2f}B','large-cap'
        elif mc>=1e9: mcs,mcc = f'${mc/1e9:.2f}B','mid-cap'
        elif mc>=1e6: mcs,mcc = f'${mc/1e6:.1f}M','small-cap'
        else: mcs,mcc = 'n/a','unknown'

        avs = f'{av/1e6:.1f}M' if av>=1e6 else f'{av/1e3:.0f}K' if av>=1e3 else str(av)

        f = {
            'name':name,'sector':sector,'industry':industry,
            'market_cap':mc,'market_cap_str':mcs,'market_cap_class':mcc,
            'pe_ratio':round(pe,2) if pe else None,
            'forward_pe':round(fpe,2) if fpe else None,
            'peg_ratio':round(peg,2) if peg else None,
            'beta':round(beta,2) if beta else None,
            'dividend_yield':round(dy,2) if dy else 0,
            'avg_volume':av,'avg_volume_str':avs,
            'fifty_two_high':round(h52,2) if h52 else None,
            'fifty_two_low':round(l52,2) if l52 else None,
            'range_position':round(rp*100,1),
            'revenue_growth':round(rg*100,1) if rg else None,
            'profit_margins':round(pm*100,1) if pm else None,
            'debt_to_equity':round(dte,1) if dte else None,
            'return_on_equity':round(roe*100,1) if roe else None,
            'free_cashflow':fcf,
            'earnings_growth':round(eg*100,1) if eg else None,
            'price_to_book':round(ptb,2) if ptb else None,
        }
        f['assessment'] = _assess(f, ticker)
        return f
    except Exception as e:
        return {'error':str(e),'assessment':{'verdict':'unknown','summary':'could not fetch data','points':[],'strength':'unknown','strength_text':''}}


def _assess(f, ticker):
    points = []; score = 0
    pe = f.get('pe_ratio')
    fpe = f.get('forward_pe')
    if pe:
        if pe<0: points.append({'type':'warning','text':f'negative P/E ({pe}) means the company is currently losing money'}); score-=2
        elif pe<12: points.append({'type':'bullish','text':f'P/E of {pe} is low suggesting possible undervaluation'}); score+=2
        elif pe<20: points.append({'type':'neutral','text':f'P/E of {pe} is reasonable'}); score+=1
        elif pe<35: points.append({'type':'neutral','text':f'P/E of {pe} is above average suggesting investors expect growth'})
        elif pe<60: points.append({'type':'warning','text':f'P/E of {pe} is high. needs strong growth to justify'}); score-=1
        else: points.append({'type':'bearish','text':f'P/E of {pe} is very high. priced for perfection'}); score-=2
        if fpe and pe and fpe<pe*0.8: points.append({'type':'bullish','text':f'forward P/E of {fpe} below trailing suggests earnings growth expected'}); score+=1
        elif fpe and pe and fpe>pe*1.2: points.append({'type':'warning','text':f'forward P/E of {fpe} above trailing suggests earnings may decline'}); score-=1

    peg = f.get('peg_ratio')
    if peg:
        if peg<1: points.append({'type':'bullish','text':f'PEG of {peg} under 1.0 suggests undervaluation relative to growth'}); score+=2
        elif peg<1.5: points.append({'type':'neutral','text':f'PEG of {peg} suggests fair value relative to growth'})
        elif peg>2: points.append({'type':'bearish','text':f'PEG of {peg} suggests overpriced for growth rate'}); score-=1

    beta = f.get('beta')
    if beta:
        if beta>1.5: points.append({'type':'warning','text':f'beta of {beta} means significantly more volatile than the market'})
        elif beta>=0.8: points.append({'type':'neutral','text':f'beta of {beta} means it tracks the market fairly closely'})
        elif beta>=0: points.append({'type':'bullish','text':f'beta of {beta} means lower volatility than the market'}); score+=0.5

    pm = f.get('profit_margins')
    if pm is not None:
        if pm>20: points.append({'type':'bullish','text':f'profit margins of {pm}% are strong'}); score+=1
        elif pm>10: points.append({'type':'neutral','text':f'profit margins of {pm}% are decent'})
        elif pm>0: points.append({'type':'warning','text':f'profit margins of {pm}% are thin'}); score-=1
        else: points.append({'type':'bearish','text':f'negative margins of {pm}% mean the company is not profitable'}); score-=2

    rg = f.get('revenue_growth')
    if rg is not None:
        if rg>20: points.append({'type':'bullish','text':f'revenue growth of {rg}% is strong'}); score+=1
        elif rg>5: points.append({'type':'neutral','text':f'revenue growth of {rg}% is steady'})
        elif rg>0: points.append({'type':'warning','text':f'revenue growth of {rg}% is slow'})
        else: points.append({'type':'bearish','text':f'revenue declining at {rg}%'}); score-=1

    roe = f.get('return_on_equity')
    if roe is not None:
        if roe>20: points.append({'type':'bullish','text':f'return on equity of {roe}% is excellent'}); score+=1
        elif roe>10: points.append({'type':'neutral','text':f'return on equity of {roe}% is respectable'})
        elif roe>0: points.append({'type':'warning','text':f'return on equity of {roe}% is below average'})
        else: points.append({'type':'bearish','text':f'negative ROE of {roe}% is concerning'}); score-=1

    dte = f.get('debt_to_equity')
    if dte is not None:
        if dte>200: points.append({'type':'bearish','text':f'debt-to-equity of {dte} is very high'}); score-=2
        elif dte>100: points.append({'type':'warning','text':f'debt-to-equity of {dte} is elevated'}); score-=1
        elif dte>50: points.append({'type':'neutral','text':f'debt-to-equity of {dte} is manageable'})
        else: points.append({'type':'bullish','text':f'debt-to-equity of {dte} is low. strong balance sheet'}); score+=1

    div = f.get('dividend_yield',0)
    if div>4: points.append({'type':'bullish','text':f'dividend yield of {div}% is high'}); score+=0.5
    elif div>1.5: points.append({'type':'neutral','text':f'dividend yield of {div}%'})

    rp = f.get('range_position',50)
    if rp>90: points.append({'type':'warning','text':f'near 52-week high ({rp:.0f}% of range)'}); score-=0.5
    elif rp<15: points.append({'type':'bullish','text':f'near 52-week low ({rp:.0f}% of range)'}); score+=0.5

    if score>=3: verdict,summary = 'undervalued', f'fundamentals look strong for {ticker}. could be undervalued'
    elif score>=1: verdict,summary = 'fair_value', f'{ticker} appears reasonably priced'
    elif score>=-1: verdict,summary = 'fair_value', f'{ticker} is roughly fairly valued'
    elif score>=-3: verdict,summary = 'overvalued', f'{ticker} looks potentially overpriced'
    else: verdict,summary = 'overvalued', f'{ticker} raises several concerns. appears overvalued'

    ss = 0
    if pm and pm>15: ss+=1
    if rg and rg>10: ss+=1
    if roe and roe>15: ss+=1
    if dte is not None and dte<80: ss+=1
    if f.get('free_cashflow',0)>0: ss+=1

    if ss>=4: strength,st = 'strong','solid profitability, growth and healthy balance sheet'
    elif ss>=2: strength,st = 'moderate','some strengths but also areas to improve'
    else: strength,st = 'weak','profitability, growth or financial health are concerning'

    return {'verdict':verdict,'summary':summary,'score':round(score,1),'strength':strength,'strength_text':st,'points':points}


# ==================== NEWS & INSIDER ====================

def get_news_sentiment(ticker):
    try:
        raw = yf.Ticker(ticker).news
        if not raw: return {'articles':[],'overall_score':0,'summary':'no news','count':0}
        pos_w = {'surge','surges','soar','jump','jumps','gain','gains','rally','rise','rises','climb','high','record','beat','beats','strong','bullish','growth','profit','upgrade','buy','outperform','positive','boost','recovery','optimistic','breakthrough','expansion','earnings','dividend','approval','partnership','deal','success','win','higher','upside','above'}
        neg_w = {'drop','drops','fall','falls','decline','plunge','crash','sink','down','low','loss','losses','miss','weak','bearish','sell','downgrade','negative','cut','slash','warning','risk','fear','concern','recession','lawsuit','investigation','penalty','recall','bankruptcy','layoff','layoffs','debt','deficit','lower','below','worst','crisis','slump','tumble'}
        articles,total = [],0
        for item in raw[:15]:
            t,p,lk,dt = '','','',''
            if isinstance(item,dict) and 'content' in item:
                ct=item['content']; t=ct.get('title',''); pr=ct.get('provider',{}); p=pr.get('displayName','') if isinstance(pr,dict) else ''; cu=ct.get('canonicalUrl',{}); lk=cu.get('url','') if isinstance(cu,dict) else ''; dt=ct.get('pubDate','')
            elif isinstance(item,dict):
                t=item.get('title',''); p=item.get('publisher',''); lk=item.get('link','')
                if 'providerPublishTime' in item:
                    try: dt=datetime.datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M')
                    except: pass
            if not t: continue
            w=set(re.findall(r'\b\w+\b',t.lower())); ps=len(w&pos_w); ns=len(w&neg_w)
            if ps>ns: sent,sc='positive',min(ps-ns,3)
            elif ns>ps: sent,sc='negative',-min(ns-ps,3)
            else: sent,sc='neutral',0
            total+=sc; articles.append({'title':t,'publisher':p,'date':dt[:16] if dt else '','link':lk,'sentiment':sent,'score':sc})
        n=len(articles)
        if n==0: return {'articles':[],'overall_score':0,'summary':'no news','count':0}
        avg=total/n
        return {'articles':articles,'overall_score':round(avg,2),'summary':'positive' if avg>0.4 else 'negative' if avg<-0.4 else 'mixed','count':n}
    except: return {'articles':[],'overall_score':0,'summary':'error','count':0}


#9.1: what an insider filing actually says now. yfinance leaves the
#Transaction column blank and puts the wording in Text, so the whole tab was
#reading an empty field and reporting a confident neutral on every ticker -
#shell had 121 filings and two large CFO sells behind that zero.
#
#the vocabulary, counted across a spread of US and LSE names:
#    'Sale at price N per share'      -> sell
#    'Sold at price N per share'      -> sell
#    'Bought at price N per share'    -> buy
#    'Purchase at price N per share'  -> buy
#    'Stock Award(Grant) ...'         -> neither
#    'Stock Gift ...'                 -> neither
#    'Conversion of Exercise of derivative security ...' -> neither
#    'Exercise of Option ...'         -> neither
#
#awards, gifts and option exercises are deliberately not buys. an insider
#being handed shares says nothing about what they think of the price, and
#counting those as purchases is what makes naive insider signals useless.
INSIDER_BUY_WORDS  = ('bought', 'purchase')
INSIDER_SELL_WORDS = ('sale', 'sold', 'disposition')

SENIOR_INSIDER_TITLES = ('ceo', 'cfo', 'chief executive', 'chief financial', 'chief operating',
                         'president', 'chairman', 'director', 'officer', 'general counsel',
                         'vp', 'vice president')


def _classify_insider_row(transaction_text, description_text):
    '''transaction column first, then the description. buy, sell or other'''
    for blob in (transaction_text, description_text):
        low = (blob or '').strip().lower()
        if not low:
            continue
        if any(w in low for w in INSIDER_BUY_WORDS):
            return 'buy'
        if any(w in low for w in INSIDER_SELL_WORDS):
            return 'sell'
    return 'other'


def get_insider_activity(ticker):
    try:
        stock = yf.Ticker(ticker)
        transactions = []
        try:
            idf = stock.insider_transactions
            if idf is not None and not idf.empty:
                #parse a deeper slice than we display. the most recent rows
                #are often awards, and a signal built on five of those is
                #noise. display is trimmed at the end instead
                for _, row in idf.head(60).iterrows():
                    def cell(*names):
                        for n in names:
                            if n in row.index:
                                v = row.get(n)
                                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                                    s = str(v).strip()
                                    if s and s.lower() != 'nan':
                                        return s
                        return ''

                    name  = cell('Insider', 'Insider Trading', 'insider')
                    title = cell('Position', 'Title', 'position')
                    trans = cell('Transaction', 'transaction')
                    text  = cell('Text', 'text', 'Description')
                    date  = cell('Start Date', 'startDate', 'Date')

                    action = _classify_insider_row(trans, text)
                    is_exec = any(t in title.lower() for t in SENIOR_INSIDER_TITLES)
                    try: sv = int(float(str(row.get('Shares', 0)).replace(',', '')))
                    except Exception: sv = 0
                    try: vv = float(str(row.get('Value', 0)).replace(',', '').replace('$', ''))
                    except Exception: vv = 0
                    transactions.append({'name': name[:40], 'title': title[:30], 'action': action,
                                         'date': str(date)[:10], 'shares': sv, 'value': vv,
                                         'is_executive': is_exec, 'detail': text[:60]})
        except Exception:
            pass

        eb = sum(1 for t in transactions if t['is_executive'] and t['action'] == 'buy')
        es = sum(1 for t in transactions if t['is_executive'] and t['action'] == 'sell')
        ab = sum(1 for t in transactions if t['action'] == 'buy')
        a_s = sum(1 for t in transactions if t['action'] == 'sell')

        #LSE filings come back with no Position at all, so nothing is ever
        #flagged executive. fall back to the whole tape rather than scoring
        #zero on a name with real insider activity on it
        titled = any(t['title'] for t in transactions)
        if titled:
            net_b, net_s = eb, es
        else:
            net_b, net_s = ab, a_s

        score = min(net_b - net_s, 3) if net_b > net_s else -min(net_s - net_b, 3) if net_s > net_b else 0
        return {'transactions': transactions[:25], 'exec_buys': eb, 'exec_sells': es,
                'all_buys': ab, 'all_sells': a_s, 'parsed': len(transactions),
                'titles_available': titled,
                'sentiment': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral',
                'score': score}
    except Exception:
        return {'transactions': [], 'exec_buys': 0, 'exec_sells': 0, 'all_buys': 0,
                'all_sells': 0, 'parsed': 0, 'titles_available': False,
                'sentiment': 'unknown', 'score': 0}


# ==================== SCREENER ====================

def quick_score_stock(ticker):
    try:
        data = yf.download(ticker, period='6mo', interval='1d', progress=False)
        if data.empty or len(data)<50: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data = compute_indicators(data)
        c = float(data['Close'].iloc[-1])
        score, signals = 0, []
        rsi = float(data['RSI'].iloc[-1])
        if rsi<30: score+=2; signals.append('RSI oversold')
        elif rsi>70: score-=2; signals.append('RSI overbought')
        macd,ms=float(data['MACD'].iloc[-1]),float(data['MACD_Signal'].iloc[-1])
        mp,msp=float(data['MACD'].iloc[-2]),float(data['MACD_Signal'].iloc[-2])
        if macd>ms and mp<=msp: score+=2; signals.append('MACD bullish cross')
        elif macd<ms and mp>=msp: score-=2; signals.append('MACD bearish cross')
        elif macd>ms: score+=0.5
        s5,s20,s50=float(data['SMA_5'].iloc[-1]),float(data['SMA_20'].iloc[-1]),float(data['SMA_50'].iloc[-1])
        if s5>s20>s50: score+=2; signals.append('bullish SMA alignment')
        elif s5<s20<s50: score-=2; signals.append('bearish SMA alignment')
        if c>s20: score+=0.5
        if c>s50: score+=0.5
        bbp=float(data['BB_Position'].iloc[-1])
        if bbp<0.1: score+=1.5; signals.append('near lower BB')
        elif bbp>0.9: score-=1.5
        m5=float(data['Momentum_5'].iloc[-1])
        if m5>0.02: score+=1; signals.append('positive momentum')
        elif m5<-0.02: score-=1
        vr=float(data['Vol_Ratio'].iloc[-1])
        if vr>1.5 and m5>0: score+=1; signals.append('volume breakout')
        sl5=float(data['Trend_Slope_5'].iloc[-1])
        if sl5>0: score+=0.5
        else: score-=0.5
        sk=float(data['Stoch_K'].iloc[-1]); sd=float(data['Stoch_D'].iloc[-1])
        if sk<20 and sk>sd: score+=1.5; signals.append('stochastic bullish')
        elif sk>80 and sk<sd: score-=1.5
        return {'ticker':ticker,'price':round(c,2),'change_1d':round(float(data['Returns'].iloc[-1])*100,2),'change_5d':round(float(data['Returns_5d'].iloc[-1])*100,2),'rsi':round(rsi,1),'score':round(score,1),'signals':signals[:4],'direction':'bullish' if score>1 else 'bearish' if score<-1 else 'neutral'}
    except: return None


# ==================== MODELS ====================

def build_model_a(sl,nf):
    m=Sequential(); m.add(Conv1D(64,3,activation='relu',padding='same',input_shape=(sl,nf),kernel_regularizer=l2(1e-4))); m.add(Conv1D(32,3,activation='relu',padding='same')); m.add(Dropout(0.2)); m.add(Bidirectional(LSTM(80,return_sequences=True))); m.add(Dropout(0.25)); m.add(Bidirectional(LSTM(40))); m.add(Dropout(0.2)); m.add(Dense(32,activation='relu')); m.add(Dense(1,activation='tanh')); m.compile(optimizer='adam',loss='huber'); return m
def build_model_b(sl,nf):
    m=Sequential(); m.add(GRU(100,return_sequences=True,input_shape=(sl,nf))); m.add(Dropout(0.25)); m.add(GRU(50,return_sequences=True)); m.add(Dropout(0.25)); m.add(GRU(25)); m.add(Dropout(0.2)); m.add(Dense(32,activation='relu')); m.add(Dense(16,activation='relu')); m.add(Dense(1,activation='tanh')); m.compile(optimizer='adam',loss='huber'); return m
def build_model_c(sl,nf):
    m=Sequential(); m.add(LSTM(128,return_sequences=True,input_shape=(sl,nf))); m.add(BatchNormalization()); m.add(Dropout(0.3)); m.add(LSTM(64,return_sequences=True)); m.add(BatchNormalization()); m.add(Dropout(0.25)); m.add(LSTM(32)); m.add(Dropout(0.2)); m.add(Dense(48,activation='relu',kernel_regularizer=l2(1e-4))); m.add(Dense(16,activation='relu')); m.add(Dense(1,activation='tanh')); m.compile(optimizer='adam',loss='huber'); return m

MODEL_BUILDERS = [build_model_a, build_model_b, build_model_c]
MODEL_NAMES = ['conv-lstm', 'gru', 'deep-lstm']


# ==================== ROUTES ====================

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/fetch_data', methods=['POST'])
def fetch_data():
    try:
        ticker=request.json.get('ticker','').upper().strip()
        if not ticker: return jsonify({'error':'no ticker'}),400
        data=yf.download(ticker,period='10y',interval='1d',progress=False)
        if data.empty or len(data)<200: return jsonify({'error':f'not enough data for {ticker}'}),400
        if isinstance(data.columns,pd.MultiIndex): data.columns=data.columns.get_level_values(0)
        data=compute_indicators(data)
        #fetch and merge market context (new in 7.1)
        sector_etf = get_sector_etf(ticker)
        market_ctx = fetch_market_context(data.index.min(), data.index.max() + pd.Timedelta(days=2), sector_etf)
        data = add_market_features(data, market_ctx)
        data = data.replace([np.inf,-np.inf], 0).fillna(0)
        data.to_csv(os.path.join(DATA_DIR,f'{ticker}.csv'))
        cp=round(float(data['Close'].iloc[-1]),2); pp=round(float(data['Close'].iloc[-2]),2)
        return jsonify(sanitise({
            'ticker':ticker,'current_price':cp,'change':round(cp-pp,2),'change_pct':round(((cp-pp)/pp)*100,2),
            'data_points':len(data),'sector_etf':sector_etf,
            'chart_data':{'dates':[d.strftime('%Y-%m-%d') for d in data.index],'close':[round(float(v),2) for v in data['Close'].values],'volume':[int(v) for v in data['Volume'].values],'high':[round(float(v),2) for v in data['High'].values],'low':[round(float(v),2) for v in data['Low'].values],'open':[round(float(v),2) for v in data['Open'].values],'sma_20':[round(float(v),2) if not np.isnan(v) else None for v in data['SMA_20'].values],'sma_50':[round(float(v),2) if not np.isnan(v) else None for v in data['SMA_50'].values]},
            'news':get_news_sentiment(ticker),'insider':get_insider_activity(ticker),'fundamentals':get_fundamentals(ticker),
            #10.0: the earnings diary travels with the price data, so the
            #analyse tab can say "reports in 3 days" before you act on
            #anything else on the page
            'earnings':earnings.summary(ticker),
            'date_range':{'start':data.index[0].strftime('%Y-%m-%d'),'end':data.index[-1].strftime('%Y-%m-%d')}
        }))
    except Exception as e: return jsonify({'error':str(e)}),500


@app.route('/api/insider_screener', methods=['POST'])
def insider_screener():
    '''openinsider.com scraper endpoint'''
    try:
        trade_type = request.json.get('trade_type', 'buy')
        min_value = request.json.get('min_value', 10000)
        days = request.json.get('days', 7)
        ceo_cfo_only = request.json.get('ceo_cfo_only', True)
        count = request.json.get('count', 100)

        result = scrape_openinsider(
            trade_type=trade_type,
            min_value=min_value,
            days=days,
            ceo_cfo_only=ceo_cfo_only,
            count=count
        )
        return jsonify(sanitise(result))
    except Exception as e:
        return jsonify({'error': str(e), 'trades': [], 'count': 0}), 500


@app.route('/api/screener', methods=['POST'])
def run_screener():
    global screener_status
    if screener_status['active']: return jsonify({'error':'already running'}),400
    top_n=request.json.get('top_n',10); count=request.json.get('stock_count',200)
    screener_status={'active':True,'progress':0,'message':'starting...','complete':False,'results':[],'error':None}
    thread=threading.Thread(target=_run_screener,args=(top_n,count)); thread.daemon=True; thread.start()
    return jsonify({'status':'started'})

def _run_screener(top_n, count):
    global screener_status
    try:
        stocks = get_stock_universe(count)
        total=len(stocks); results=[]
        for i,t in enumerate(stocks):
            screener_status['progress']=int((i/total)*90)
            screener_status['message']=f'scanning {t} ({i+1}/{total})...'
            r=quick_score_stock(t)
            if r: results.append(r)
            if i%10==0 and i>0: time.sleep(0.5)
        results.sort(key=lambda x: x['score'], reverse=True)
        screener_status['message']='fetching details for top picks...'
        screener_status['progress']=92
        top=results[:top_n]
        for r in top:
            try:
                news=get_news_sentiment(r['ticker']); r['news_score']=news.get('overall_score',0)
                ins=get_insider_activity(r['ticker']); r['insider_sentiment']=ins.get('sentiment','unknown'); r['insider_score']=ins.get('score',0); r['exec_buys']=ins.get('exec_buys',0); r['exec_sells']=ins.get('exec_sells',0)
                fund=get_fundamentals(r['ticker']); r['pe_ratio']=fund.get('pe_ratio'); r['market_cap_str']=fund.get('market_cap_str','?'); r['beta']=fund.get('beta'); r['dividend_yield']=fund.get('dividend_yield',0); r['valuation']=fund.get('assessment',{}).get('verdict','unknown'); r['strength']=fund.get('assessment',{}).get('strength','unknown')
                r['combined_score']=round(r['score']+r['news_score']*0.5+r['insider_score']*0.8,1)
                time.sleep(0.3)
            except: r['combined_score']=r['score']
        top.sort(key=lambda x: x.get('combined_score',x['score']), reverse=True)
        screener_status.update({'results':top,'total_scanned':total,'progress':100,'complete':True,'active':False,'message':f'done! scanned {total} stocks.'})
    except Exception as e:
        screener_status.update({'error':str(e),'active':False})

@app.route('/api/screener_status')
def get_screener_status(): return jsonify(sanitise(screener_status))


@app.route('/api/train', methods=['POST'])
def train_route():
    global training_status
    if training_status['active']: return jsonify({'error':'already training'}),400
    ticker=request.json.get('ticker','').upper().strip(); epochs=request.json.get('epochs',50)
    if not ticker: return jsonify({'error':'no ticker'}),400
    if not os.path.exists(os.path.join(DATA_DIR,f'{ticker}.csv')): return jsonify({'error':'fetch data first'}),400
    training_status={'active':True,'progress':0,'message':'starting...','ticker':ticker,'complete':False,'error':None,'backtest':None}
    thread=threading.Thread(target=_train_ensemble,args=(ticker,epochs)); thread.daemon=True; thread.start()
    return jsonify({'status':'started','ticker':ticker})

def _train_ensemble(ticker, epochs):
    global training_status
    try:
        data=pd.read_csv(os.path.join(DATA_DIR,f'{ticker}.csv'),index_col=0,parse_dates=True)
        if isinstance(data.columns,pd.MultiIndex): data.columns=data.columns.get_level_values(0)
        features_df,feature_cols=prepare_features(data); nf=len(feature_cols); ci=feature_cols.index('Close')
        cp=features_df['Close'].values.astype(float)

        #raw returns plus a lightly-smoothed version for training (2-day ewma)
        #the model trains on smoothed returns (less single-day noise) but we
        #always evaluate against the actual unsmoothed direction
        rets_raw = np.diff(cp)/(cp[:-1]+1e-10)
        rets_smooth = pd.Series(rets_raw).ewm(span=2, adjust=False).mean().values
        rets_target = np.clip(rets_smooth, -0.15, 0.15)
        rets_actual = np.clip(rets_raw, -0.15, 0.15)

        #build raw sequences - scaling done per-split to avoid leakage
        X_raw, y_t, y_a = [], [], []
        for i in range(SEQUENCE_LENGTH, len(features_df.values)-1):
            X_raw.append(features_df.values[i-SEQUENCE_LENGTH:i])
            y_t.append(rets_target[i-1])
            y_a.append(rets_actual[i-1])
        X_raw=np.array(X_raw); y_t=np.array(y_t); y_a=np.array(y_a)

        #chronological split: last 20% is the held-out test set, never seen by training or scaler
        total=len(X_raw); ts=int(total*0.2); trs=total-ts
        X_pool, X_test_raw = X_raw[:trs], X_raw[trs:]
        y_pool, y_pool_actual = y_t[:trs], y_a[:trs]
        y_test, y_test_actual = y_t[trs:], y_a[trs:]

        #within the pool: last 15% is validation, first 85% is training (chronological)
        vs=int(trs*0.85)
        X_tr_raw, X_vl_raw = X_pool[:vs], X_pool[vs:]
        y_tr, y_vl = y_pool[:vs], y_pool[vs:]

        training_status['progress']=5; training_status['message']='fitting scaler on training data only...'
        #fit scaler ONLY on training portion of pool - this is the leak fix
        scaler=RobustScaler(); scaler.fit(X_tr_raw.reshape(-1, nf))
        def apply_scaler(X3d):
            s=X3d.shape; return scaler.transform(X3d.reshape(-1, nf)).reshape(s)
        Xt=apply_scaler(X_tr_raw); Xv=apply_scaler(X_vl_raw); Xte=apply_scaler(X_test_raw)

        #---- baselines ----
        training_status['progress']=10; training_status['message']='computing baselines...'
        naive_acc = float(np.mean(y_test_actual > 0) * 100)
        if len(y_test_actual)>1:
            pers_pred=np.concatenate([[0], y_test_actual[:-1]])
            pers_acc=float(np.mean(np.sign(pers_pred)==np.sign(y_test_actual))*100)
        else:
            pers_acc=50.0

        #walk-forward xgboost baseline: trains across 3 expanding folds within the pool,
        #then once on the full pool to score on the held-out test
        xgb_acc=None
        try:
            from xgboost import XGBRegressor
            Xpf, Xtf = X_pool[:,-1,:], X_test_raw[:,-1,:]
            n=len(Xpf); fz=n//4; fold_accs=[]
            for fi in range(3):
                tre=fz*(fi+1); vle=fz*(fi+2)
                if vle-tre < 20: continue
                m=XGBRegressor(n_estimators=120,max_depth=4,learning_rate=0.05,n_jobs=-1,verbosity=0)
                m.fit(Xpf[:tre], y_pool[:tre])
                vp=m.predict(Xpf[tre:vle]); va=y_pool_actual[tre:vle]
                fold_accs.append(float(np.mean(np.sign(vp)==np.sign(va))*100))
            xgb_final=XGBRegressor(n_estimators=120,max_depth=4,learning_rate=0.05,n_jobs=-1,verbosity=0)
            xgb_final.fit(Xpf, y_pool)
            xt_pred=xgb_final.predict(Xtf)
            xgb_acc={'walk_forward_folds':[round(f,1) for f in fold_accs],
                     'walk_forward_mean':round(float(np.mean(fold_accs)),1) if fold_accs else None,
                     'held_out_test':round(float(np.mean(np.sign(xt_pred)==np.sign(y_test_actual))*100),1)}
        except ImportError:
            print('xgboost not installed - skipping baseline\n')
        except Exception as e:
            print(f'xgb baseline failed: {e}\n')

        #---- train the lstm ensemble ----
        training_status['progress']=15; mr=[]; etp_1step=[]
        #sample weights: linearly increase from 0.5 to 1.5 across the training window
        #so recent samples carry more influence than ancient ones
        sw=np.linspace(0.5, 1.5, len(y_tr))
        for i in range(NUM_ENSEMBLE):
            mn=MODEL_NAMES[i]; training_status['message']=f'training {mn} ({i+1}/{NUM_ENSEMBLE})...'
            model=MODEL_BUILDERS[i](SEQUENCE_LENGTH,nf)
            es=EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True)
            rlr=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,min_lr=1e-6)
            for epoch in range(epochs):
                h=model.fit(Xt,y_tr,sample_weight=sw,epochs=1,batch_size=32,
                            validation_data=(Xv,y_vl),callbacks=[es,rlr],verbose=0)
                training_status['progress']=min(15+i*18+int((epoch+1)/epochs*16),70)
                training_status['message']=f'{mn} | epoch {epoch+1}/{epochs} | loss: {h.history["loss"][0]:.6f}'
                if es.stopped_epoch>0: break
            model.save(os.path.join(MODELS_DIR,f'{ticker}_model_{i}.keras'))
            tp=model.predict(Xte,verbose=0).flatten(); etp_1step.append(tp)
            da=float(np.mean(np.sign(tp)==np.sign(y_test_actual))*100)
            mr.append({'name':mn,'direction_accuracy':round(da,1)})

        #---- 1-step backtest (teacher forcing, same as before for comparison) ----
        training_status['progress']=75; training_status['message']='1-step backtest...'
        eavg=np.mean(etp_1step,axis=0)
        evote=np.sign(np.sum([np.sign(p) for p in etp_1step],axis=0))
        eda_1step=float(np.mean(evote==np.sign(y_test_actual))*100)
        tcp=cp[trs+SEQUENCE_LENGTH : trs+SEQUENCE_LENGTH+len(y_test_actual)]
        pp_1step=tcp*(1+eavg); ap=tcp*(1+y_test_actual); cl=min(60,len(ap))

        #---- honest multi-step backtest (autoregressive rollout) ----
        training_status['progress']=80; training_status['message']='multi-step backtest (the honest one)...'
        models=[load_model(os.path.join(MODELS_DIR,f'{ticker}_model_{i}.keras')) for i in range(NUM_ENSEMBLE)]
        rollout_len=min(60, len(y_test_actual))
        start_idx = trs + SEQUENCE_LENGTH
        actual_curve = [float(p) for p in ap[:rollout_len]]
        predicted_curve = []
        buf = data.iloc[: start_idx][['Open','High','Low','Close','Volume']].copy()
        #freeze the market context at the value right before rollout begins
        market_freeze={c: float(data[c].iloc[start_idx-1]) if c in data.columns else 0.0 for c in MARKET_FEATURE_COLS}
        current = float(cp[start_idx-1])
        vol_std = float(data['Close'].iloc[:start_idx].pct_change().dropna().std())
        for step in range(rollout_len):
            tdf=compute_indicators(buf.copy())
            for c,v in market_freeze.items(): tdf[c]=v
            tfdf,_=prepare_features(tdf)
            rec=tfdf[feature_cols].iloc[-SEQUENCE_LENGTH:]
            if len(rec)<SEQUENCE_LENGTH: break
            ss=apply_scaler(rec.values.reshape(1,SEQUENCE_LENGTH,nf))
            mrs=[float(np.clip(m.predict(ss,verbose=0)[0][0], -0.08, 0.08)) for m in models]
            vu=sum(1 for r in mrs if r>0); direction=1 if vu>len(mrs)/2 else -1
            combined=direction*np.mean(np.abs(mrs))
            new_price=current*(1+combined); predicted_curve.append(new_price)
            nd=buf.index[-1]+pd.Timedelta(days=1)
            while nd.weekday()>=5: nd+=pd.Timedelta(days=1)
            noise=vol_std*current
            buf=pd.concat([buf, pd.DataFrame({'Open':[current],'High':[max(new_price,current)+abs(noise*0.3)],
                                              'Low':[min(new_price,current)-abs(noise*0.3)],'Close':[new_price],
                                              'Volume':[float(buf['Volume'].iloc[-20:].mean())]}, index=[nd])])
            current=new_price

        if predicted_curve and len(predicted_curve)==len(actual_curve):
            ms_mae=float(mean_absolute_error(actual_curve, predicted_curve))
            ms_rmse=float(np.sqrt(mean_squared_error(actual_curve, predicted_curve)))
            #5-day-ahead direction accuracy: does the predicted curve get the trend over 5 days right?
            if len(predicted_curve)>=5:
                ms_dirs=[np.sign(predicted_curve[k]-predicted_curve[k-5])==np.sign(actual_curve[k]-actual_curve[k-5])
                         for k in range(5, len(predicted_curve))]
                ms_dir_acc=float(np.mean(ms_dirs)*100) if ms_dirs else 50.0
            else:
                ms_dir_acc=50.0
        else:
            ms_mae=0; ms_rmse=0; ms_dir_acc=50.0

        #---- assemble final backtest payload ----
        bt={
            'ensemble_direction_accuracy':round(eda_1step,1),
            'mae':round(float(mean_absolute_error(ap,pp_1step)),2),
            'rmse':round(float(np.sqrt(mean_squared_error(ap,pp_1step))),2),
            'mape':round(float(np.mean(np.abs((ap-pp_1step)/(ap+1e-10)))*100),2),
            'individual_models':mr,
            'val_actual':[round(float(p),2) for p in ap[-cl:]],
            'val_predictions':[round(float(p),2) for p in pp_1step[-cl:]],
            'multi_step':{
                'mae':round(ms_mae,2),'rmse':round(ms_rmse,2),
                'direction_accuracy_5d':round(ms_dir_acc,1),
                'rollout_length':len(predicted_curve),
                'actual':[round(float(p),2) for p in actual_curve],
                'predicted':[round(float(p),2) for p in predicted_curve],
            },
            'baselines':{
                'naive_always_up':round(naive_acc,1),
                'persistence':round(pers_acc,1),
                'xgboost':xgb_acc,
            },
            'verdict':make_verdict(eda_1step, naive_acc, pers_acc, xgb_acc)
        }
        meta={'feature_cols':feature_cols,'num_features':nf,'close_idx':ci,
              'scaler':{'center':scaler.center_.tolist(),'scale':scaler.scale_.tolist()},
              'backtest':bt,'model_names':MODEL_NAMES[:NUM_ENSEMBLE],'num_models':NUM_ENSEMBLE,
              'version':'7.1'}
        with open(os.path.join(MODELS_DIR,f'{ticker}_meta.json'),'w') as f: json.dump(meta,f)
        training_status.update({'message':'complete!','progress':100,'complete':True,'active':False,'backtest':bt})
    except Exception as e:
        import traceback; traceback.print_exc()
        training_status.update({'error':str(e),'active':False})

@app.route('/api/training_status')
def get_training_status(): return jsonify(sanitise(training_status))


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        ticker=request.json.get('ticker','').upper().strip(); tf=request.json.get('timeframe','1d')
        if not ticker: return jsonify({'error':'no ticker'}),400
        mp=os.path.join(MODELS_DIR,f'{ticker}_meta.json')
        if not os.path.exists(mp): return jsonify({'error':'train first'}),400
        with open(mp) as f: meta=json.load(f)
        #refuse to load pre-7.1 models because feature counts changed
        if meta.get('version','7.0') != '7.1':
            return jsonify({'error':f'this model was trained with an older version - please retrain {ticker}'}),400
        fc=meta['feature_cols']; nm=meta['num_models']
        models=[load_model(os.path.join(MODELS_DIR,f'{ticker}_model_{i}.keras')) for i in range(nm)]
        scaler=RobustScaler(); scaler.center_=np.array(meta['scaler']['center']); scaler.scale_=np.array(meta['scaler']['scale'])
        rd=pd.read_csv(os.path.join(DATA_DIR,f'{ticker}.csv'),index_col=0,parse_dates=True)
        if isinstance(rd.columns,pd.MultiIndex): rd.columns=rd.columns.get_level_values(0)
        buf=rd[['Open','High','Low','Close','Volume']].copy()
        cp=float(rd['Close'].iloc[-1]); dv=float(rd['Close'].pct_change().dropna().std())
        #freeze the market context features at last known values for the future rollout
        market_freeze={c: float(rd[c].iloc[-1]) if c in rd.columns else 0.0 for c in MARKET_FEATURE_COLS}
        tfm={'1d':1,'1w':5,'1m':22,'3m':66,'6m':132,'1y':252}; days=tfm.get(tf,1)
        amp=[[] for _ in range(nm)]; ep=[]; ub=[]; lb=[]
        #use the multi-step rmse for uncertainty bands when available because that's the honest one
        brmse=meta.get('backtest',{}).get('multi_step',{}).get('rmse') or meta.get('backtest',{}).get('rmse', cp*0.02)
        if brmse <= 0: brmse = cp*0.02
        for day in range(days):
            tdf=compute_indicators(buf.copy())
            for c,v in market_freeze.items(): tdf[c]=v
            tfdf,_=prepare_features(tdf)
            rec=tfdf[fc].iloc[-SEQUENCE_LENGTH:]
            if len(rec)<SEQUENCE_LENGTH: break
            ss=scaler.transform(rec.values).reshape(1,SEQUENCE_LENGTH,len(fc))
            mrs=[]
            for i,m in enumerate(models):
                pr=float(m.predict(ss,verbose=0)[0][0]); pr=np.clip(pr,-0.08,0.08); mrs.append(pr)
            vu=sum(1 for r in mrs if r>0); d=1 if vu>len(mrs)/2 else -1
            cr=d*np.mean(np.abs(mrs))
            prev=ep[-1] if ep else cp; pp=round(prev*(1+cr),2); ep.append(pp)
            for i,ret in enumerate(mrs):
                p=amp[i][-1] if amp[i] else cp; amp[i].append(round(p*(1+ret),2))
            u=brmse*np.sqrt(day+1)*0.8; ub.append(round(pp+1.96*u,2)); lb.append(round(pp-1.96*u,2))
            noise=dv*prev; nd=buf.index[-1]+pd.Timedelta(days=1)
            while nd.weekday()>=5: nd+=pd.Timedelta(days=1)
            nr=pd.DataFrame({'Open':[prev],'High':[max(pp,prev)+abs(noise*0.3)],'Low':[min(pp,prev)-abs(noise*0.3)],'Close':[pp],'Volume':[float(buf['Volume'].iloc[-20:].mean())]},index=[nd])
            buf=pd.concat([buf,nr])
        pd_dates=[]; cd=rd.index[-1]
        for _ in range(len(ep)):
            cd+=datetime.timedelta(days=1)
            while cd.weekday()>=5: cd+=datetime.timedelta(days=1)
            pd_dates.append(cd.strftime('%Y-%m-%d'))
        mv=[{'name':MODEL_NAMES[i],'direction':'up' if amp[i][-1]>cp else 'down','final_price':amp[i][-1],'change_pct':round(((amp[i][-1]-cp)/cp)*100,2),'prices':amp[i]} for i in range(nm) if amp[i]]
        news=get_news_sentiment(ticker); ins=get_insider_activity(ticker)
        analysis=_analyse_pred(cp,ep,pd_dates,news,ins,meta.get('backtest',{}),mv,dv)
        return jsonify(sanitise({'ticker':ticker,'timeframe':tf,'current_price':cp,'predictions':{'dates':pd_dates,'prices':ep,'upper_band':ub,'lower_band':lb},'model_votes':mv,'analysis':analysis,'news_sentiment':news,'insider':ins}))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error':str(e)}),500


def _analyse_pred(cp,pp,pd,news,ins,bt,mv,dv):
    if not pp: return {'action':'hold','reason':'no predictions'}
    mnp,mxp=min(pp),max(pp); mni,mxi=pp.index(mnp),pp.index(mxp); fp=pp[-1]
    oc=round(((fp-cp)/cp)*100,2)
    da=bt.get('ensemble_direction_accuracy',50); conf='low' if da<55 else 'medium' if da<65 else 'high'
    vu=sum(1 for v in mv if v['direction']=='up'); cons=f'{vu}/{len(mv)} models say up'
    ns=news.get('overall_score',0) if news else 0; nb='positive' if ns>0.3 else 'negative' if ns<-0.3 else 'neutral'
    isent=ins.get('sentiment','unknown') if ins else 'unknown'; eb=ins.get('exec_buys',0) if ins else 0; es_=ins.get('exec_sells',0) if ins else 0
    result={'current_price':cp,'predicted_low':round(mnp,2),'predicted_low_date':pd[mni],'predicted_high':round(mxp,2),'predicted_high_date':pd[mxi],'final_predicted_price':round(fp,2),'overall_change_pct':oc,'model_confidence':conf,'direction_accuracy':da,'news_bias':nb,'news_score':ns,'insider_sentiment':isent,'exec_buys':eb,'exec_sells':es_,'consensus':cons,'votes_up':vu,'votes_down':len(mv)-vu,'daily_volatility':round(dv*100,2)}
    threshold=0.015 if conf=='high' else 0.02 if conf=='medium' else 0.03; reasons=[]
    if mni<mxi:
        sw=(mxp-mnp)/mnp
        if sw>=threshold:
            result.update({'action':'buy_then_sell','buy_date':pd[mni],'buy_price':round(mnp,2),'sell_date':pd[mxi],'sell_price':round(mxp,2),'potential_profit_pct':round(sw*100,2)})
            reasons.append(f'{cons}. predicted dip to ${mnp:.2f} on {pd[mni]} then rise to ${mxp:.2f} on {pd[mxi]} ({sw*100:.1f}% gain)')
            if nb=='positive': reasons.append('news supports this')
            elif nb=='negative': reasons.append('news is negative so be cautious')
            if isent=='bullish': reasons.append(f'insiders buying ({eb} exec buys vs {es_} sells)')
            elif isent=='bearish': reasons.append(f'insiders selling ({es_} exec sells)')
            reasons.append(f'ensemble accuracy: {da:.0f}% ({conf})'); result['reason']='. '.join(reasons); return result
    if mxi<mni:
        dr=(mxp-mnp)/mxp
        if dr>=threshold:
            result.update({'action':'sell_then_buy','sell_date':pd[mxi],'sell_price':round(mxp,2),'buy_date':pd[mni],'buy_price':round(mnp,2),'potential_profit_pct':round(dr*100,2)})
            reasons.append(f'{cons}. peak at ${mxp:.2f} on {pd[mxi]} then drop to ${mnp:.2f} on {pd[mni]}')
            if isent=='bearish': reasons.append(f'insiders selling too ({es_} exec sells)')
            reasons.append(f'accuracy: {da:.0f}% ({conf})'); result['reason']='. '.join(reasons); return result
    if oc>2: result['action']='buy'; reasons.append(f'{cons}. upward trend {oc:+.1f}%')
    elif oc<-2: result['action']='sell'; reasons.append(f'{cons}. downward {oc:+.1f}%')
    else: result['action']='hold'; reasons.append(f'{cons}. no significant movement ({oc:+.1f}%)')
    if nb!='neutral': reasons.append(f'news: {nb} ({ns:+.1f})')
    if isent not in ('unknown','neutral'): reasons.append(f'insider: {isent} ({eb}B/{es_}S)')
    reasons.append(f'accuracy: {da:.0f}% ({conf})'); result['reason']='. '.join(reasons); return result


@app.route('/api/save_trade', methods=['POST'])
def save_trade():
    global trade_recommendations
    t=request.json; t['saved_at']=datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    trade_recommendations.append(t); return jsonify({'status':'saved'})

@app.route('/api/trades')
def get_trades(): return jsonify(trade_recommendations)

@app.route('/api/clear_trades', methods=['POST'])
def clear_trades():
    global trade_recommendations; trade_recommendations=[]; return jsonify({'status':'cleared'})


# ==================== PORTFOLIO PERSISTENCE ====================

PORTFOLIO_FILE = os.path.join(DATA_DIR, 'merlin_portfolio.json')

def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE): return []
    try:
        with open(PORTFOLIO_FILE, 'r') as f: return json.load(f)
    except: return []

def save_portfolio(positions):
    with open(PORTFOLIO_FILE, 'w') as f: json.dump(positions, f, indent=2)


@app.route('/api/portfolio_load', methods=['GET'])
def portfolio_load():
    '''return the saved portfolio positions'''
    return jsonify({'positions': load_portfolio()})


@app.route('/api/portfolio_save', methods=['POST'])
def portfolio_save():
    '''
    save the current portfolio. expects:
    positions: list of {ticker, shares, avg_cost, currency}
    overwrites whatever was there.
    '''
    try:
        positions = (request.json or {}).get('positions', [])
        #clean and validate
        cleaned = []
        for p in positions:
            ticker = str(p.get('ticker','')).upper().strip()
            shares = float(p.get('shares', 0) or 0)
            avg_cost = float(p.get('avg_cost', 0) or 0)
            currency = str(p.get('currency','USD'))
            if ticker and shares > 0:
                cleaned.append({'ticker':ticker,'shares':shares,'avg_cost':avg_cost,'currency':currency})
        save_portfolio(cleaned)
        return jsonify({'status':'saved','count':len(cleaned)})
    except Exception as e:
        return jsonify({'error':str(e)}), 500


@app.route('/api/portfolio_clear', methods=['POST'])
def portfolio_clear():
    save_portfolio([])
    return jsonify({'status':'cleared'})



# ==================== PORTFOLIO ANALYSER ====================

def get_fx_rates():
    '''fetch live conversion rates TO GBP'''
    fallback = {'USD': 0.787, 'GBP': 1.0, 'GBp': 0.01, 'EUR': 0.856, 'CAD': 0.58, 'AUD': 0.51}
    try:
        rates = {'GBP': 1.0, 'GBp': 0.01}
        # GBPUSD=X → 1 GBP = X USD → invert for USD→GBP
        # EURGBP=X → 1 EUR = X GBP → direct
        pairs = {
            'USD': ('GBPUSD=X', 'invert'),
            'EUR': ('EURGBP=X', 'direct'),
            'CAD': ('CADGBP=X', 'direct'),
            'AUD': ('AUDGBP=X', 'direct'),
        }
        for currency, (symbol, mode) in pairs.items():
            try:
                price = yf.Ticker(symbol).info.get('regularMarketPrice', 0)
                if price and price > 0:
                    rates[currency] = round(1 / price if mode == 'invert' else price, 6)
            except:
                pass
        for k, v in fallback.items():
            if k not in rates:
                rates[k] = v
        return rates
    except:
        return fallback


@app.route('/api/fx_rates', methods=['GET'])
def fx_rates_route():
    fx = get_fx_rates()
    return jsonify(sanitise({
        'rates': fx,
        'display': {
            'GBPUSD': round(1 / fx.get('USD', 0.787), 4),
            'GBPEUR': round(1 / fx.get('EUR', 0.856), 4),
            'GBPCAD': round(1 / fx.get('CAD', 0.58),  4),
            'GBPAUD': round(1 / fx.get('AUD', 0.51),  4),
        }
    }))


@app.route('/api/portfolio_analyse', methods=['POST'])
def portfolio_analyse():
    try:
        positions = request.json.get('positions', [])
        if not positions:
            return jsonify({'error': 'no positions provided'}), 400

        fx = get_fx_rates()   # fetch once for the whole portfolio
        results = []; total_value_gbp = 0; total_cost_gbp = 0; sectors = {}

        CURRENCY_SYMS = {'USD':'$','GBP':'£','GBp':'p','EUR':'€','CAD':'C$','AUD':'A$'}

        for pos in positions:
            ticker   = pos.get('ticker', '').upper().strip()
            shares   = float(pos.get('shares', 0) or 0)
            avg_cost = float(pos.get('avg_cost', 0) or 0)
            currency = pos.get('currency', 'USD')
            if not ticker or shares <= 0:
                continue
            try:
                tech = quick_score_stock(ticker) or {}
                cp   = float(tech.get('price', 0) or 0)
                if not cp:
                    info = yf.Ticker(ticker).info or {}
                    cp   = float(info.get('currentPrice', info.get('regularMarketPrice', 0)) or 0)
                #london-listed tickers return prices in pence. if user said GBP we need pounds
                if ticker.endswith('.L') and currency == 'GBP' and cp > 1000:
                    cp = cp / 100

                fund    = get_fundamentals(ticker)
                news    = get_news_sentiment(ticker)
                insider = get_insider_activity(ticker)

                # convert everything to GBP
                rate    = fx.get(currency, 1.0)
                cp_gbp  = cp * rate
                ac_gbp  = avg_cost * rate
                pv_gbp  = cp_gbp * shares
                pc_gbp  = ac_gbp * shares
                pl_gbp  = pv_gbp - pc_gbp
                pl_pct  = ((cp - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0

                total_value_gbp += pv_gbp
                total_cost_gbp  += pc_gbp

                sector = fund.get('sector', 'Unknown') or 'Unknown'
                sectors[sector] = sectors.get(sector, 0) + pv_gbp

                ts    = float(tech.get('score', 0) or 0)
                ns    = float(news.get('overall_score', 0) or 0)
                ins_s = float(insider.get('score', 0) or 0)
                fs    = float(fund.get('assessment', {}).get('score', 0) or 0)
                combined = ts + ns * 0.5 + ins_s * 0.8 + fs * 0.5
                rsi   = float(tech.get('rsi', 50) or 50)

                if rsi > 75 and pl_pct > 30:
                    action = 'trim'
                    reason = f'overbought (RSI {rsi:.0f}) with {pl_pct:.0f}% gain — consider taking profits'
                elif combined >= 4:
                    action = 'add'
                    reason = 'strong bullish signals across technicals, fundamentals and sentiment'
                elif combined >= 1.5:
                    action = 'hold'
                    reason = 'moderate signals, position looks healthy'
                elif combined <= -4:
                    action = 'sell' if pl_pct < -15 else 'trim'
                    reason = (f'bearish signals with {pl_pct:.0f}% loss — consider cutting'
                              if pl_pct < -15 else 'bearish signals — consider reducing exposure')
                elif combined <= -1.5:
                    action = 'trim'
                    reason = 'weak or mixed signals skewing bearish — consider trimming'
                else:
                    action = 'hold'
                    reason = 'mixed signals, maintain current position size'

                results.append(sanitise({
                    'ticker': ticker, 'name': fund.get('name', ticker),
                    'sector': sector, 'shares': shares,
                    'avg_cost': avg_cost, 'avg_cost_gbp': round(ac_gbp, 4),
                    'currency': currency, 'currency_symbol': CURRENCY_SYMS.get(currency, ''),
                    'fx_rate': rate,
                    'current_price': round(cp, 4), 'current_price_gbp': round(cp_gbp, 4),
                    'position_value_gbp': round(pv_gbp, 2),
                    'position_cost_gbp': round(pc_gbp, 2),
                    'pl_gbp': round(pl_gbp, 2), 'pl_pct': round(pl_pct, 2),
                    'action': action, 'action_reason': reason,
                    'tech_score': round(ts, 1), 'tech_direction': tech.get('direction', 'neutral'),
                    'tech_signals': tech.get('signals', []), 'rsi': round(rsi, 1),
                    'fund_verdict': fund.get('assessment', {}).get('verdict', 'unknown'),
                    'fund_score': round(fs, 1), 'pe_ratio': fund.get('pe_ratio'),
                    'market_cap_str': fund.get('market_cap_str', '?'), 'beta': fund.get('beta'),
                    'news_sentiment': news.get('summary', 'neutral'), 'news_score': round(ns, 2),
                    'insider_sentiment': insider.get('sentiment', 'neutral'),
                    'exec_buys': insider.get('exec_buys', 0), 'exec_sells': insider.get('exec_sells', 0),
                    'combined_score': round(combined, 1), 'weight_pct': 0,
                }))
                time.sleep(0.2)

            except Exception as e:
                results.append({
                    'ticker': ticker, 'shares': shares, 'avg_cost': avg_cost,
                    'currency': currency, 'currency_symbol': CURRENCY_SYMS.get(currency, ''),
                    'error': str(e), 'action': 'hold', 'action_reason': 'could not fetch data',
                    'combined_score': 0, 'weight_pct': 0, 'pl_gbp': 0, 'pl_pct': 0,
                    'current_price': 0, 'current_price_gbp': 0, 'position_value_gbp': 0,
                })

        for r in results:
            if total_value_gbp > 0:
                r['weight_pct'] = round(r.get('position_value_gbp', 0) / total_value_gbp * 100, 1)

        total_pl_gbp  = total_value_gbp - total_cost_gbp
        total_pl_pct  = (total_pl_gbp / total_cost_gbp * 100) if total_cost_gbp > 0 else 0
        sector_alloc  = {k: round(v / total_value_gbp * 100, 1) for k, v in sectors.items()} if total_value_gbp > 0 else {}
        weights       = sorted([(r['ticker'], r.get('weight_pct', 0)) for r in results], key=lambda x: x[1], reverse=True)
        top_w         = weights[0][1] if weights else 0
        con_risk      = 'high' if top_w > 30 else 'medium' if top_w > 20 else 'low'

        return jsonify(sanitise({
            'positions': results,
            'fx_rates': fx,
            'portfolio': {
                'total_value_gbp': round(total_value_gbp, 2),
                'total_cost_gbp':  round(total_cost_gbp, 2),
                'total_pl_gbp':    round(total_pl_gbp, 2),
                'total_pl_pct':    round(total_pl_pct, 2),
                'num_positions':   len(results),
                'sector_allocation': sector_alloc,
                'concentration_risk': con_risk,
                'top_weight': weights[0] if weights else None,
                'sells_count': len([r for r in results if r.get('action') in ('sell','trim')]),
                'buys_count':  len([r for r in results if r.get('action') == 'add']),
            }
        }))

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500



# ==================== NICHE EDGE SIGNALS (new in 7.1) ====================

def format_market_cap(mc):
    '''compact market cap string'''
    if not mc: return 'n/a'
    if mc >= 1e12: return f'${mc/1e12:.2f}T'
    if mc >= 1e9:  return f'${mc/1e9:.2f}B'
    if mc >= 1e6:  return f'${mc/1e6:.1f}M'
    return 'n/a'


# ---------------- INSIDER CLUSTER SIGNAL ---------------- #
# based on cohen, malloy and pomorski (2012, journal of finance) showing
# that 3+ insider buy clusters predict ~6-10% outperformance over 60 days

SENIOR_TITLES = ['ceo','cfo','cto','coo','president','chairman','director',
                 'chief executive','chief financial','chief operating','chief technology']

def get_insider_clusters(min_insiders=3, days=60, min_total_value=500000, senior_required=True):
    '''
    aggregate openinsider buys by ticker over a window and find tickers
    with 3+ unique senior insider buyers. returns clusters sorted by a
    composite confidence score.
    '''
    raw = scrape_openinsider(trade_type='buy', min_value=10000, days=days,
                             ceo_cfo_only=False, count=1000)
    if not raw or not raw.get('trades'):
        return []

    #group by ticker
    by_ticker = {}
    for t in raw['trades']:
        tk = t.get('ticker', '').strip().upper()
        if tk: by_ticker.setdefault(tk, []).append(t)

    clusters = []
    for ticker, trades in by_ticker.items():
        unique_names = set(); senior_count = 0; total_value = 0
        most_recent_date = ''
        for t in trades:
            name = t.get('insider_name','').strip()
            if not name: continue
            unique_names.add(name)
            total_value += t.get('value', 0)
            title_lower = t.get('title','').lower()
            if any(kw in title_lower for kw in SENIOR_TITLES):
                senior_count += 1
            fd = t.get('filing_date','')
            if fd > most_recent_date: most_recent_date = fd

        n_insiders = len(unique_names)
        #apply filters
        if n_insiders < min_insiders: continue
        if total_value < min_total_value: continue
        if senior_required and senior_count == 0: continue

        #confidence score 0-100
        size_score = min(n_insiders * 8, 35)
        senior_score = min(senior_count * 10, 30)
        value_score = min(total_value / 100000, 25)
        try:
            days_ago = (pd.Timestamp.now() - pd.to_datetime(most_recent_date[:10])).days
            recency_score = max(10 - days_ago/3, 0)
        except:
            recency_score = 5
        confidence = round(size_score + senior_score + value_score + recency_score, 1)

        clusters.append({
            'ticker': ticker, 'n_insiders': n_insiders, 'n_trades': len(trades),
            'senior_count': senior_count, 'total_value': total_value,
            'avg_value': total_value / max(len(trades), 1),
            'most_recent_date': most_recent_date[:10],
            'confidence': confidence, 'trades': trades[:10]
        })

    clusters.sort(key=lambda x: x['confidence'], reverse=True)
    return clusters


def enrich_cluster(cluster):
    '''add current price, sector and price move since the cluster began'''
    try:
        ticker = cluster['ticker']
        info = yf.Ticker(ticker).info or {}
        cluster['name'] = info.get('shortName', ticker)
        cluster['sector'] = info.get('sector', 'unknown')
        cluster['current_price'] = info.get('currentPrice', info.get('regularMarketPrice', 0))
        cluster['market_cap_str'] = format_market_cap(info.get('marketCap', 0))
        cluster['pe_ratio'] = info.get('trailingPE')
        #price move since cluster started
        hist = drop_incomplete_bars(yf.Ticker(ticker).history(period='3mo'))
        if not hist.empty and cluster.get('most_recent_date'):
            try:
                cd = pd.to_datetime(cluster['most_recent_date'])
                since = hist[hist.index >= cd]
                if len(since) > 1:
                    move = (hist['Close'].iloc[-1] - since['Close'].iloc[0]) / since['Close'].iloc[0] * 100
                    cluster['move_since_cluster'] = round(float(move), 2)
            except: pass
    except: pass
    return cluster


# ---------------- POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD) ---------------- #
# ball and brown (1968): stocks drift in the direction of earnings surprises
# for weeks afterwards, strongest in the first 2-3 weeks

def get_pead_signal(ticker, max_days_since=30):
    '''
    detect active PEAD opportunities with analyst revision component.
    upgraded per Chan Jegadeesh Lakonishok 1996, which showed that EPS
    surprise AND subsequent analyst revisions both predict return drift,
    and that combining them produces a stronger signal than either alone.
    confidence is now built from three components:
      surprise_score:  magnitude of EPS beat or miss
      timing_score:    drift window still open (closer to event = better)
      revision_score:  analysts revising estimates in same direction
    alignment with price action is kept as a bonus.
    '''
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        #9.1: the earnings_dates endpoint went stale. it returns twelve rows
        #whose newest event is well over a year old and no future dates at
        #all, for every ticker, so days_since was always past the window and
        #this module could never fire - the whole tab produced nothing and
        #PEAD's contribution to the paper trader was silently dead.
        #
        #two fields that are still current between them say the same thing:
        #    info['earningsTimestamp']  the date of the last report
        #    stock.earnings_history     epsActual and epsEstimate per quarter
        #the old path is kept underneath as a fallback in case the endpoint
        #comes back.
        latest_naive = None
        eps_actual = eps_estimate = None

        #earningsTimestamp is the date of the last report for some tickers and
        #the date of the next one for others - apple hands back a july date
        #that has been and gone, shell hands back an october date that has not
        #happened yet. only a past timestamp is a report we can measure drift
        #from, so a future one is treated as no usable event rather than as an
        #event sixty days in the future
        try:
            ts = info.get('earningsTimestamp')
            if ts and float(ts) > 1e9:
                stamp = pd.Timestamp(datetime.datetime.fromtimestamp(float(ts)))
                if stamp <= pd.Timestamp.now():
                    latest_naive = stamp
        except Exception:
            latest_naive = None

        if latest_naive is not None:
            try:
                eh = stock.earnings_history
                if eh is not None and not eh.empty:
                    row = eh.iloc[-1]
                    a = _safe_float(row.get('epsActual'))
                    e = _safe_float(row.get('epsEstimate'))
                    #the history is indexed by quarter end and the timestamp
                    #is the report date, so the quarter must sit just before
                    #the report rather than a year away from it
                    try:
                        q_end = pd.Timestamp(eh.index[-1])
                        gap = (latest_naive - q_end.tz_localize(None) if q_end.tz else latest_naive - q_end).days
                    except Exception:
                        gap = 0
                    if a is not None and e is not None and -5 <= gap <= 150:
                        eps_actual, eps_estimate = a, e
            except Exception:
                pass

        if eps_actual is None or eps_estimate is None:
            #fallback: the original earnings_dates path
            try:
                earnings = stock.earnings_dates
            except Exception:
                try: earnings = stock.get_earnings_dates(limit=12)
                except Exception: return None
            if earnings is None or earnings.empty: return None

            idx_tz = earnings.index.tz
            now = pd.Timestamp.now(tz=idx_tz) if idx_tz else pd.Timestamp.now()
            past = earnings[earnings.index <= now]
            if past.empty: return None

            latest = past.iloc[-1]; latest_date = past.index[-1]
            latest_naive = latest_date.tz_localize(None) if latest_date.tz else latest_date
            for col in ['Reported EPS','EPS Actual','Actual EPS']:
                if col in latest.index and not pd.isna(latest[col]):
                    eps_actual = latest[col]; break
            for col in ['EPS Estimate','Estimate','EPS Est']:
                if col in latest.index and not pd.isna(latest[col]):
                    eps_estimate = latest[col]; break

        if latest_naive is None: return None
        days_since = (pd.Timestamp.now() - latest_naive).days
        if days_since < 1 or days_since > max_days_since: return None
        if eps_actual is None or eps_estimate is None or eps_estimate == 0: return None

        surprise_pct = float((eps_actual - eps_estimate) / abs(eps_estimate) * 100)
        if surprise_pct > 5: direction = 'up'
        elif surprise_pct < -5: direction = 'down'
        else: return None

        #price move since earnings
        hist = drop_incomplete_bars(stock.history(period='3mo'))
        if hist.empty: return None
        current_price = float(hist['Close'].iloc[-1])
        try:
            #the history index is tz-aware and the event date is naive.
            #comparing the two raises, the bare except swallowed it and the
            #move came out as a flat zero every time, which quietly removed
            #the price-confirmation component from every confidence score
            idx = hist.index
            if getattr(idx, 'tz', None) is not None:
                idx = idx.tz_localize(None)
            after = hist[idx >= latest_naive]
            if len(after) > 0:
                earn_price = float(after['Close'].iloc[0])
                move = (current_price - earn_price) / earn_price * 100
            else: move = 0
        except Exception: move = 0

        #analyst revision component, the new bit
        #yfinance exposes a few revision-related fields. we use forward EPS
        #and current quarter estimates if available. the direction of revision
        #(estimates going up vs down) is what matters most.
        revision_score = 0
        revision_note = ''
        try:
            #current quarter and next quarter estimate trends
            eps_trend = None
            try:
                eps_trend = stock.eps_trend
            except Exception:
                try: eps_trend = stock.get_eps_trend()
                except Exception: eps_trend = None

            revision_pct = None
            if eps_trend is not None and not eps_trend.empty:
                #eps_trend has rows like '0q','+1q','0y','+1y' with columns
                #'current','7daysAgo','30daysAgo','60daysAgo','90daysAgo'.
                #compare current vs 30 days ago for current quarter row.
                try:
                    row_label = '0q' if '0q' in eps_trend.index else eps_trend.index[0]
                    row = eps_trend.loc[row_label]
                    cur = _safe_float(row.get('current'))
                    old = _safe_float(row.get('30daysAgo')) or _safe_float(row.get('60daysAgo'))
                    if cur is not None and old is not None and old != 0:
                        revision_pct = (cur - old) / abs(old) * 100
                except Exception:
                    revision_pct = None

            if revision_pct is not None:
                #aligned revisions add up to 15. opposing revisions subtract up to 10.
                if direction == 'up' and revision_pct > 0:
                    revision_score = min(revision_pct * 1.5, 15)
                    revision_note = f"analysts revised est +{revision_pct:.1f}% in last month"
                elif direction == 'up' and revision_pct < 0:
                    revision_score = max(revision_pct, -10)
                    revision_note = f"analysts revised est {revision_pct:.1f}% (against direction)"
                elif direction == 'down' and revision_pct < 0:
                    revision_score = min(abs(revision_pct) * 1.5, 15)
                    revision_note = f"analysts revised est {revision_pct:.1f}% in last month"
                elif direction == 'down' and revision_pct > 0:
                    revision_score = -min(revision_pct, 10)
                    revision_note = f"analysts revised est +{revision_pct:.1f}% (against direction)"
        except Exception:
            pass

        #confidence: bigger surprise + earlier in window + analyst confirmation + price confirming = better
        surprise_score = min(abs(surprise_pct) / 2, 35)   #was 40, made room for revisions
        timing_score = max(25 - days_since, 0)            #was 30, slightly reduced
        aligned = (move > 0 and direction == 'up') or (move < 0 and direction == 'down')
        alignment_score = 15 if aligned else 0
        if abs(move) > 15: alignment_score *= 0.5   #most of the move already happened
        confidence = round(surprise_score + timing_score + alignment_score + revision_score, 1)
        confidence = max(0, min(confidence, 95))    #clamp to sensible range

        return {
            'ticker': ticker, 'name': info.get('shortName', ticker),
            'sector': info.get('sector', 'unknown'),
            'earnings_date': latest_naive.strftime('%Y-%m-%d'),
            'days_since_earnings': int(days_since),
            'eps_actual': round(float(eps_actual), 3),
            'eps_estimate': round(float(eps_estimate), 3),
            'surprise_pct': round(surprise_pct, 1),
            'direction': direction,
            'current_price': round(current_price, 2),
            'move_since_earnings': round(move, 2),
            'revision_score': round(revision_score, 1),
            'revision_note': revision_note,
            'confidence': confidence,
            'market_cap_str': format_market_cap(info.get('marketCap', 0))
        }
    except:
        return None


# ---------------- VOLATILITY-GATED MEAN REVERSION ---------------- #
# only fires when market is calm (VIX low), stock is in uptrend,
# AND RSI is at an extreme. all gates must be open

def get_vix_level():
    '''current VIX close with safe fallback'''
    try:
        vix = yf.Ticker('^VIX').history(period='5d')
        if not vix.empty: return float(vix['Close'].iloc[-1])
    except: pass
    return 20.0


#check_mean_reversion removed in 7.8 - meanrev strategy retired after 5y
#backtest showed +15.87% total P/L vs SPY's +91.11% over the same window




# ---------------- EXISTING RELATIVE MOMENTUM SIGNAL ---------------- #
# This preserves the momentum strategy that was already in Merlin before the
# research-factor expansion. It is a 12-1 month momentum signal with two extra
# filters: it must be near the 52-week high and it must beat SPY momentum.

def get_momentum_signal(ticker, spy_mom=0.0):
    '''
    Existing Merlin momentum strategy: 12-1 month relative momentum.
    Measures return from ~12 months ago to ~1 month ago, deliberately skipping
    the most recent month to reduce short-term reversal noise. It then requires
    the stock to be near its 52-week high and outperforming SPY.
    '''
    try:
        hist = drop_incomplete_bars(yf.Ticker(ticker).history(period='1y', auto_adjust=True))
        if hist.empty or len(hist) < 240: return None
        closes = hist['Close']

        p_old = float(closes.iloc[0])
        p_skip = float(closes.iloc[-22])
        p_now = float(closes.iloc[-1])
        if p_old <= 0 or p_now <= 3: return None

        mom_12_1 = (p_skip - p_old) / p_old
        if mom_12_1 < 0.10: return None

        high_52w = float(closes.max())
        proximity = p_now / high_52w if high_52w > 0 else 0
        if proximity < 0.85: return None

        rel_mom = mom_12_1 - spy_mom
        if rel_mom <= 0: return None

        conf = 50.0
        conf += min(30, mom_12_1 * 100)
        if proximity >= 0.95: conf += 10
        elif proximity >= 0.90: conf += 5
        if rel_mom >= 0.10: conf += 10
        elif rel_mom >= 0.05: conf += 5

        #8.0: frog-in-the-pan smoothness (da gurun warachka 2014). momentum
        #built from lots of small up days continues far better than momentum
        #from a couple of violent gaps, because gradual information diffuses
        #slowly into prices. reward smooth paths, lightly penalise jumpy ones
        form_rets = closes.iloc[:-21].pct_change().dropna()
        pos_day_ratio = float((form_rets > 0).mean()) if len(form_rets) > 20 else 0.5
        if pos_day_ratio >= 0.55: conf += 5
        elif pos_day_ratio >= 0.52: conf += 3
        elif pos_day_ratio < 0.48: conf -= 4

        info = yf.Ticker(ticker).info or {}
        return {
            'ticker': ticker, 'name': info.get('shortName', ticker),
            'sector': info.get('sector', 'unknown'),
            'current_price': round(p_now, 2),
            'mom_12_1_pct': round(mom_12_1 * 100, 2),
            'proximity_to_52w_high_pct': round(proximity * 100, 1),
            'relative_mom_vs_spy_pct': round(rel_mom * 100, 2),
            'positive_day_ratio_pct': round(pos_day_ratio * 100, 1),
            'confidence': round(min(conf, 96), 1),
            'expected_horizon': '60-90 days',
            'market_cap_str': format_market_cap(info.get('marketCap', 0))
        }
    except Exception:
        return None

# ---------------- RESEARCH FACTOR SIGNALS FOR AI PAPER TRADER ---------------- #
# These four buy-only signals are designed for the auto trader. They are slower
# moving than PEAD/mean-reversion, so the exits below use wider stops and longer
# holding windows. All return the same candidate schema used by the paper trader:
# {ticker, strategy, confidence, direction, reason, ...}

FACTOR_STRATEGIES = ['momentum', 'quality', 'shareholder_yield']


def _safe_float(x, default=None):
    try:
        if x is None or pd.isna(x): return default
        return float(x)
    except Exception:
        return default


def _download_factor_history(ticker, period='2y'):
    """single price download used by all four research-factor signals"""
    try:
        data = yf.download(ticker, period=period, interval='1d', progress=False)
        if data.empty or len(data) < 252: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = compute_indicators(data)
        data['SMA_100'] = data['Close'].rolling(100, min_periods=1).mean()
        data['SMA_200'] = data['Close'].rolling(200, min_periods=1).mean()
        return data.replace([np.inf, -np.inf], 0).fillna(0)
    except Exception:
        return None


def _estimate_buyback_yield(ticker, info=None):
    """
    Estimate buyback yield as last annual repurchases / market cap.
    Falls back to share-count contraction if yfinance exposes shares history.
    Returns percentage points, e.g. 3.2 means 3.2%.
    """
    info = info or {}
    market_cap = _safe_float(info.get('marketCap'), 0) or 0
    if market_cap <= 0: return 0.0

    #cash-flow method: repurchases are normally negative cash-flow values
    try:
        cf = yf.Ticker(ticker).cashflow
        if cf is not None and not cf.empty:
            row_name = None
            for idx in cf.index:
                low = str(idx).lower()
                if 'repurchase' in low or 'buyback' in low:
                    row_name = idx; break
            if row_name is not None:
                vals = pd.to_numeric(cf.loc[row_name], errors='coerce').dropna()
                if len(vals):
                    repurchase = max(0.0, -float(vals.iloc[0]))
                    if repurchase > 0:
                        return round(min(repurchase / market_cap * 100, 20), 2)
    except Exception:
        pass

    #share-count method: if shares outstanding fell, treat it as buyback yield
    try:
        start = (datetime.datetime.now() - datetime.timedelta(days=540)).strftime('%Y-%m-%d')
        shares = yf.Ticker(ticker).get_shares_full(start=start)
        if shares is not None and len(shares) > 5:
            shares = shares.dropna().sort_index()
            old = float(shares.iloc[0]); new = float(shares.iloc[-1])
            if old > 0 and new > 0 and new < old:
                return round(min((old - new) / old * 100, 20), 2)
    except Exception:
        pass

    return 0.0


def get_research_factor_signals(ticker, min_confidence=65):
    """
    Return all qualifying research-backed factor signals for one ticker:
      1) 12-1 month momentum
      2) 52-week high momentum
      3) quality / gross profitability proxy
      4) shareholder yield / buyback yield
    """
    signals = []
    data = _download_factor_history(ticker, period='2y')
    if data is None or len(data) < 252: return signals

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    close = data['Close'].astype(float)
    volume = data['Volume'].astype(float) if 'Volume' in data.columns else pd.Series([0]*len(data), index=data.index)
    cp = float(close.iloc[-1])
    if cp <= 3: return signals  #avoid penny-stock noise

    avg_vol_20 = float(volume.tail(20).mean()) if len(volume) else 0
    if avg_vol_20 < 100_000: return signals  #liquidity gate

    sma50 = float(data['SMA_50'].iloc[-1])
    sma100 = float(data['SMA_100'].iloc[-1])
    sma200 = float(data['SMA_200'].iloc[-1])
    rsi = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns else 50.0
    high_52 = float(close.tail(252).max())
    low_52 = float(close.tail(252).min())
    ret_1m = float(close.iloc[-1] / close.iloc[-22] - 1) if len(close) > 22 else 0.0
    ret_6m = float(close.iloc[-1] / close.iloc[-127] - 1) if len(close) > 127 else 0.0
    ret_12_1 = float(close.iloc[-22] / close.iloc[-253] - 1) if len(close) > 253 else 0.0
    trend_ok = cp > sma200 and sma50 > sma200 * 0.98
    name = info.get('shortName', ticker)
    sector = info.get('sector', 'unknown')
    market_cap = _safe_float(info.get('marketCap'), 0) or 0

    #1) 12-1 month momentum: retired in 8.1. the 5y backtest showed +8.31%
    #total after fees with a 44.1% monte carlo chance of loss, a -45% 5th
    #percentile and sharpe 0.05 across 729 trades. fails the same bar that
    #binned meanrev, week52_high and low_beta_trend in 7.8. the branch stays
    #in the backtester (_bt_signal_fast) so the decision can be re-tested.

    #2) 52-week high momentum: removed in 7.8. The 5y backtest showed only
    #+16.17% total P/L over 498 trades (0.3% per trade avg) vs SPY +91%.
    #Average winner too small to justify a slot.

    #3) Quality/profitability: slower-moving long signal, used only when market confirms it
    pm = _safe_float(info.get('profitMargins'))
    roe = _safe_float(info.get('returnOnEquity'))
    rg = _safe_float(info.get('revenueGrowth'))
    dte = _safe_float(info.get('debtToEquity'))
    fcf = _safe_float(info.get('freeCashflow'))
    pe = _safe_float(info.get('trailingPE') or info.get('forwardPE'))
    fpe = _safe_float(info.get('forwardPE'))

    q = 0
    if pm is not None: q += 18 if pm > 0.15 else 10 if pm > 0.08 else 4 if pm > 0 else -8
    if roe is not None: q += 16 if roe > 0.15 else 9 if roe > 0.08 else 3 if roe > 0 else -6
    if rg is not None: q += 13 if rg > 0.08 else 7 if rg > 0.02 else 2 if rg > 0 else -5
    if fcf is not None and fcf > 0: q += 12
    if dte is not None: q += 12 if dte < 80 else 6 if dte < 150 else -8
    if pe is not None and pe > 0: q += 8 if pe < 25 else 4 if pe < 40 else -6
    if fpe is not None and pe is not None and fpe > 0 and fpe < pe: q += 4
    if cp > sma200: q += 6
    if ret_6m > 0: q += 4

    if q >= 62 and cp > sma200 * 0.97 and ret_6m > -0.05 and rsi < 78:
        confidence = round(min(q, 92), 1)
        if confidence >= min_confidence:
            signals.append({
                'ticker': ticker, 'strategy': 'quality', 'direction': 'buy',
                'confidence': confidence, 'current_price': round(cp, 2),
                'reason': f'quality score {q:.0f}: margins/ROE/FCF/debt pass, trend not broken',
                'quality_score': round(q, 1), 'profit_margin_pct': round(pm*100, 1) if pm is not None else None,
                'roe_pct': round(roe*100, 1) if roe is not None else None,
                'name': name, 'sector': sector, 'market_cap_str': format_market_cap(market_cap)
            })

    #4) Shareholder yield: dividend + estimated buyback yield, only with trend/quality guardrails
    #9.1: was `dy * 100 if dy < 1 else dy`, which is correct for a 3.5%
    #payer and wrong for every company yielding under one percent - those
    #arrived as yields in the seventies and carried the confidence score up
    #with them. units are settled by _dividend_yield_pct now
    dividend_yield_pct = _dividend_yield_pct(info)
    shareholder_base_ok = market_cap > 1e9 and cp > sma200 * 0.97 and ret_6m > -0.10 and (pm is None or pm > 0) and (fcf is None or fcf > 0)
    if shareholder_base_ok:
        buyback_yield_pct = _estimate_buyback_yield(ticker, info)
        shareholder_yield_pct = dividend_yield_pct + buyback_yield_pct
        if shareholder_yield_pct >= 3.0 and rsi < 78:
            confidence = 45
            confidence += min(shareholder_yield_pct * 4.0, 24)
            if buyback_yield_pct >= 2: confidence += 8
            if dividend_yield_pct >= 2: confidence += 5
            if cp > sma50: confidence += 5
            if q >= 50: confidence += 6
            confidence = round(min(confidence, 90), 1)
            if confidence >= min_confidence:
                signals.append({
                    'ticker': ticker, 'strategy': 'shareholder_yield', 'direction': 'buy',
                    'confidence': confidence, 'current_price': round(cp, 2),
                    'reason': f'shareholder yield {shareholder_yield_pct:.1f}% = dividend {dividend_yield_pct:.1f}% + buyback {buyback_yield_pct:.1f}%',
                    'shareholder_yield_pct': round(shareholder_yield_pct, 1),
                    'dividend_yield_pct': round(dividend_yield_pct, 1),
                    'buyback_yield_pct': round(buyback_yield_pct, 1),
                    'name': name, 'sector': sector, 'market_cap_str': format_market_cap(market_cap)
                })

    #10.0: stamp the earnings diary on every signal before it leaves.
    #
    #this costs nothing over the wire. the info blob above already carries
    #the earnings timestamps, so priming the cache from it means a 300 name
    #factor scan knows every report date without a single extra request.
    #the paper trader reads next_earnings_days off the signal for its
    #blackout and the scan tables print it, both for free
    try:
        earnings.prime_from_info(ticker, info)
        edays = earnings.days_to_next_earnings(ticker)
        for sig in signals:
            sig['next_earnings_days'] = edays
    except Exception:
        pass

    return signals




#get_low_beta_trend_signal removed in 7.8 - low_beta_trend strategy retired
#after 5y backtest showed +8.81% total P/L (0.06 sharpe) vs SPY +91.11%.
#Already flagged as weakest of the 9 in the 7.7 handover.






def _short_interest_risk_check(ticker, ret_1m=None):
    '''
    Risk filter applied before opening any position. Asquith Pathak Ritter
    2005 showed heavily short-sale-constrained stocks tend to underperform.
    The cleanest version of this signal combines high short interest with
    falling price. We treat high short interest alone as a yellow flag
    (lower priority) and high short interest with weak recent price action
    as a red flag (skip). Returns (allowed, priority_penalty, reason).
        allowed = False means do not open
        priority_penalty is subtracted from effective score when allowed=True
    '''
    try:
        info = yf.Ticker(ticker).info or {}
        short_pct = _safe_float(info.get('shortPercentOfFloat'))
        if short_pct is None: return True, 0, None

        #yfinance returns this as a decimal (0.15) most of the time but
        #occasionally as a percentage (15.0). normalise to percent.
        if short_pct < 1: short_pct = short_pct * 100

        if short_pct < 10:        #normal range, no penalty
            return True, 0, None

        #compute 1m return if not provided so we can check price confirmation
        if ret_1m is None:
            try:
                h = drop_incomplete_bars(yf.Ticker(ticker).history(period='2mo', auto_adjust=True))
                if len(h) > 22:
                    ret_1m = float(h['Close'].iloc[-1] / h['Close'].iloc[-22] - 1)
                else: ret_1m = 0.0
            except Exception:
                ret_1m = 0.0

        #red flag: heavily shorted AND falling
        if short_pct >= 20 and ret_1m < -0.05:
            return False, 0, f"short interest {short_pct:.1f}% with 1m return {ret_1m*100:+.1f}%"
        if short_pct >= 15 and ret_1m < -0.10:
            return False, 0, f"short interest {short_pct:.1f}% with 1m return {ret_1m*100:+.1f}%"

        #yellow flag: high short interest, neutral price. allowed but downgraded
        if short_pct >= 20:
            return True, 6, f"high short interest {short_pct:.1f}%"
        if short_pct >= 15:
            return True, 4, f"elevated short interest {short_pct:.1f}%"
        return True, 2, f"moderate short interest {short_pct:.1f}%"

    except Exception:
        return True, 0, None


def _candidate_effective_score(c, regime_data=None):
    '''
    confidence plus strategy evidence bonus, minus short interest penalty,
    plus regime adjustment. regime gives favoured strategies a small boost
    and suppressed strategies a penalty without disabling them entirely.
    '''
    base = float(c.get('confidence', 0)) + STRATEGY_SCORE_BONUS.get(c.get('strategy'), 0)
    base -= float(c.get('_short_penalty', 0))
    if regime_data is not None:
        sname = c.get('strategy', '')
        if sname in regime_data.get('favoured_strategies', []):
            base += 5.0
        elif sname in regime_data.get('suppressed_strategies', []):
            base -= 8.0
    return base


def _rank_paper_candidates(candidates, slots, held_tickers=None, held_positions=None, regime_data=None):
    '''
    Choose which signals the AI trader listens to.
    1) remove held tickers and low-confidence signals
    2) 7.9: hard regime gate. in bear or volatile regimes refuse all new
       momentum positions outright. previously a -8 score penalty, but in
       quiet weeks the trader would still open them when nothing else
       qualified. existing momentum positions are not force-closed, they
       still exit through their own trailing stop logic.
    3) keep only the best signal per ticker
    4) sort by adjusted confidence (regime-aware)
    5) enforce per-cycle caps so one strategy cannot fill all new slots
    6) enforce portfolio-wide group caps so momentum cannot dominate the book
       over multiple cycles.
    '''
    held_tickers = held_tickers or set()
    held_positions = held_positions or []
    if regime_data is None: regime_data = get_market_regime()
    regime_name = regime_data.get('regime', '')
    momentum_hostile = regime_name in MOMENTUM_HOSTILE_REGIMES

    #count what's already in the portfolio by group, used for the portfolio cap
    existing_group_counts = {}
    for p in held_positions:
        g = STRATEGY_GROUP.get(p.get('strategy'), p.get('strategy', 'unknown'))
        existing_group_counts[g] = existing_group_counts.get(g, 0) + 1

    best_by_ticker = {}
    short_filter_blocked = []
    earnings_blocked = []
    momentum_blocked_by_regime = 0
    for c in candidates:
        t = c.get('ticker')
        if not t or t in held_tickers: continue
        if c.get('confidence', 0) < MIN_CONFIDENCE_TO_OPEN: continue
        #7.9 hard regime gate
        if momentum_hostile and c.get('strategy') in MOMENTUM_STRATEGIES:
            momentum_blocked_by_regime += 1
            continue
        #10.0 earnings blackout. a three month factor position opened four
        #days before a print is a bet on the print, so the slow strategies
        #stand off and the name comes back on a later cycle. the factor
        #scanners already carry the day count on the signal, so the usual
        #path here is a dict read rather than a network call
        if EARNINGS_BLACKOUT_DAYS > 0 and c.get('strategy') in EARNINGS_BLACKOUT_STRATEGIES:
            n = c.get('next_earnings_days')
            if isinstance(n, int):
                blocked = 0 <= n <= EARNINGS_BLACKOUT_DAYS
                ereason = (f'reports in {n} day(s), inside the '
                           f'{EARNINGS_BLACKOUT_DAYS} day earnings blackout') if blocked else ''
            else:
                blocked, ereason = earnings.in_blackout(t, EARNINGS_BLACKOUT_DAYS)
            if blocked:
                earnings_blocked.append((t, ereason))
                continue
        #short interest risk filter: blocks heavily shorted falling names,
        #penalises moderately shorted names so they rank below clean signals
        allowed, penalty, sreason = _short_interest_risk_check(t)
        if not allowed:
            short_filter_blocked.append((t, sreason))
            continue
        if penalty:
            c = dict(c)  #copy so we don't mutate the original
            c['_short_penalty'] = penalty
            if sreason: c['reason'] = (c.get('reason', '') + f" | {sreason}").strip(' |')
        prev = best_by_ticker.get(t)
        if prev is None or _candidate_effective_score(c, regime_data) > _candidate_effective_score(prev, regime_data):
            best_by_ticker[t] = c

    ordered = sorted(best_by_ticker.values(),
                     key=lambda c: (_candidate_effective_score(c, regime_data), c.get('confidence', 0)),
                     reverse=True)
    picked, counts, group_counts = [], {}, {}
    for c in ordered:
        if len(picked) >= slots: break
        sname = c.get('strategy', 'unknown')
        group = STRATEGY_GROUP.get(sname, sname)
        strategy_cap = MAX_NEW_POSITIONS_PER_STRATEGY.get(sname, 2)
        group_cap = MAX_NEW_POSITIONS_PER_GROUP.get(group, strategy_cap)
        portfolio_group_cap = MAX_PORTFOLIO_POSITIONS_PER_GROUP.get(group, MAX_POSITIONS)
        if counts.get(sname, 0) >= strategy_cap:
            continue
        if group_counts.get(group, 0) >= group_cap:
            continue
        #portfolio-wide check: already held + about to add must stay under cap
        if existing_group_counts.get(group, 0) + group_counts.get(group, 0) >= portfolio_group_cap:
            continue
        picked.append(c)
        counts[sname] = counts.get(sname, 0) + 1
        group_counts[group] = group_counts.get(group, 0) + 1
    #attach diagnostics on the returned list so the caller can log them.
    #using a custom attribute on a list is messy, so we tag the first item
    #if there is one. caller can check candidates[0].get('_regime_blocks').
    if picked and momentum_blocked_by_regime:
        picked[0]['_regime_blocks'] = momentum_blocked_by_regime
    #10.0: same trick for the earnings blackout, with the tickers themselves
    #rather than a count, because "why did it skip BP" is the question that
    #actually gets asked when a name you expected does not appear
    if picked and earnings_blocked:
        picked[0]['_earnings_blocks'] = [t for t, _ in earnings_blocked]
    return picked

def _paper_exit_context(ticker):
    """price/trend context used by strategy-specific paper-trader exits"""
    try:
        data = yf.download(ticker, period='1y', interval='1d', progress=False)
        if data.empty or len(data) < 60: return {}
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = compute_indicators(data)
        close = data['Close'].astype(float)
        cp = float(close.iloc[-1])
        sma50 = float(close.rolling(50, min_periods=1).mean().iloc[-1])
        sma200 = float(close.rolling(200, min_periods=1).mean().iloc[-1])
        high_52 = float(close.tail(252).max())
        rsi = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns else None
        return {
            'price': cp, 'sma50': sma50, 'sma200': sma200, 'high_52': high_52,
            'below_sma50': cp < sma50, 'below_sma200': cp < sma200,
            'drawdown_from_52w_high_pct': (cp - high_52) / (high_52 + 1e-10) * 100,
            'rsi': rsi
        }
    except Exception:
        return {}

# ==================== NICHE SIGNAL ROUTES ====================

@app.route('/api/insider_clusters', methods=['POST'])
def insider_clusters_route():
    '''synchronous - openinsider scrape is fast enough'''
    try:
        d = request.json or {}
        clusters = get_insider_clusters(
            min_insiders=d.get('min_insiders', 3),
            days=d.get('days', 60),
            min_total_value=d.get('min_total_value', 500000),
            senior_required=d.get('senior_required', True)
        )
        if d.get('enrich', True):
            for c in clusters[:30]:
                enrich_cluster(c); time.sleep(0.1)
        return jsonify(sanitise({'clusters': clusters[:30], 'total_found': len(clusters)}))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e), 'clusters': []}), 500


@app.route('/api/pead_scan', methods=['POST'])
def pead_scan_route():
    global pead_status
    if pead_status['active']: return jsonify({'error': 'already running'}), 400
    count = (request.json or {}).get('count', 200)
    pead_status = {'active': True, 'progress': 0, 'message': 'starting...',
                   'complete': False, 'results': [], 'error': None}
    threading.Thread(target=_run_pead_scan, args=(count,), daemon=True).start()
    return jsonify({'status': 'started'})


def _run_pead_scan(count):
    global pead_status
    try:
        stocks = get_stock_universe(count); signals = []
        total = len(stocks)
        for i, t in enumerate(stocks):
            pead_status['progress'] = int((i / total) * 100)
            pead_status['message'] = f'checking {t} ({i+1}/{total})...'
            sig = get_pead_signal(t)
            if sig: signals.append(sig)
            if i % 10 == 0 and i > 0: time.sleep(0.3)
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        pead_status.update({'results': signals, 'progress': 100, 'complete': True,
                            'active': False, 'message': f'done - {len(signals)} PEAD signals found',
                            'total_scanned': total})
    except Exception as e:
        pead_status.update({'error': str(e), 'active': False})


@app.route('/api/pead_status')
def pead_status_route():
    return jsonify(sanitise(pead_status))


#mean reversion scan routes removed in 7.8 - strategy retired
#vix_now endpoint kept because the regime detection footer still uses it


@app.route('/api/vix_now')
def vix_now_route():
    '''quick fetch of current VIX, used by the regime footer'''
    v = get_vix_level()
    return jsonify({'vix': round(v, 2), 'gate_open_default': v < 18})



# ==================== MERLIN TRADE LOG ====================

LOG_FILE = os.path.join(DATA_DIR, 'merlin_trades.json')

def load_trade_log():
    if not os.path.exists(LOG_FILE): return []
    try:
        with open(LOG_FILE, 'r') as f: return json.load(f)
    except: return []

def save_trade_log(trades):
    with open(LOG_FILE, 'w') as f: json.dump(trades, f, indent=2)


def _gbp_pl_for_trade(trade, current_price_native=None):
    '''compute true P/L in GBP using FX at entry vs FX now, returns (pl_gbp, pl_pct_gbp, current_fx)'''
    try:
        ccy = trade.get('currency', 'USD')
        shares = float(trade.get('shares', 0) or 0)
        entry = float(trade.get('entry_price', 0) or 0)
        if not shares or not entry: return None, None, None

        if current_price_native is None:
            current_price_native = entry  #fallback
            try:
                hist = yf.Ticker(trade['ticker']).history(period='5d')
                if len(hist) > 0:
                    current_price_native = float(hist['Close'].iloc[-1])
                    #london pence handling, same logic as portfolio_analyse
                    if trade['ticker'].endswith('.L') and ccy == 'GBP' and current_price_native > 1000:
                        current_price_native = current_price_native / 100
            except Exception: pass

        #fx rate at entry, persisted on first compute so historic conversions stay stable
        fx_at_entry = trade.get('fx_at_entry_to_gbp')
        if fx_at_entry is None:
            fx_at_entry = _fetch_fx_to_gbp_on_date(ccy, trade.get('entry_date'))
        fx_now = _get_fx_to_gbp(ccy)

        cost_gbp = shares * entry * fx_at_entry
        value_gbp = shares * current_price_native * fx_now
        pl_gbp = value_gbp - cost_gbp
        pl_pct = (pl_gbp / cost_gbp * 100) if cost_gbp else 0
        return round(pl_gbp, 2), round(pl_pct, 2), fx_at_entry
    except Exception:
        return None, None, None


def _fetch_fx_to_gbp_on_date(ccy, date_str):
    '''historic fx rate to gbp on a date, falls back to current if history fails'''
    if ccy == 'GBP': return 1.0
    if ccy == 'GBp': return 0.01
    try:
        pair = f'{ccy}GBP=X'
        dt = pd.to_datetime(date_str)
        start = dt - pd.Timedelta(days=5)
        end = dt + pd.Timedelta(days=2)
        hist = yf.Ticker(pair).history(start=start, end=end)
        if len(hist) > 0:
            #closest available trading day to the entry date
            hist.index = hist.index.tz_localize(None) if hist.index.tz else hist.index
            target = hist.index[hist.index <= dt]
            if len(target) > 0:
                return float(hist.loc[target[-1], 'Close'])
    except Exception: pass
    return _get_fx_to_gbp(ccy)  #fall back to spot



@app.route('/api/log_trade', methods=['POST'])
def log_trade():
    '''
    log a merlin-influenced trade. expects:
    ticker, strategy (ensemble/insider_cluster/pead/mean_reversion/screener/manual),
    action (buy/short), entry_price, entry_date, shares, notes (optional),
    target_price (optional), stop_price (optional)
    '''
    try:
        d = request.json or {}
        trades = load_trade_log()
        trade = {
            'id': int(time.time() * 1000),
            'ticker': d.get('ticker', '').upper().strip(),
            'strategy': d.get('strategy', 'manual'),
            'action': d.get('action', 'buy'),
            'entry_price': float(d.get('entry_price', 0)),
            'entry_date': d.get('entry_date', datetime.datetime.now().strftime('%Y-%m-%d')),
            'shares': float(d.get('shares', 0)),
            'currency': d.get('currency', 'USD'),
            'fx_at_entry_to_gbp': _fetch_fx_to_gbp_on_date(d.get('currency', 'USD'),
                                    d.get('entry_date', datetime.datetime.now().strftime('%Y-%m-%d'))),
            'target_price': float(d.get('target_price', 0)) if d.get('target_price') else None,
            'stop_price': float(d.get('stop_price', 0)) if d.get('stop_price') else None,
            'notes': d.get('notes', '')[:300],
            'status': 'open',
            'exit_price': None, 'exit_date': None, 'realised_pl_pct': None,
            'logged_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        }
        if not trade['ticker'] or trade['entry_price'] <= 0:
            return jsonify({'error': 'ticker and entry price required'}), 400
        trades.append(trade)
        save_trade_log(trades)
        return jsonify({'status': 'logged', 'trade': trade})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/close_trade', methods=['POST'])
def close_trade():
    '''mark a trade as closed at the current price (or a user-supplied exit)'''
    try:
        d = request.json or {}
        tid = d.get('id')
        trades = load_trade_log()
        for t in trades:
            if t['id'] == tid:
                exit_price = d.get('exit_price')
                if not exit_price:
                    try: exit_price = float(yf.Ticker(t['ticker']).info.get('currentPrice', 0))
                    except: exit_price = 0
                if not exit_price: return jsonify({'error':'could not fetch exit price'}), 400
                t['exit_price'] = float(exit_price)
                t['exit_date'] = d.get('exit_date', datetime.datetime.now().strftime('%Y-%m-%d'))
                sign = 1 if t['action'] == 'buy' else -1
                t['realised_pl_pct'] = round(sign * (t['exit_price'] - t['entry_price']) / t['entry_price'] * 100, 2)
                t['status'] = 'closed'
                save_trade_log(trades)
                return jsonify({'status': 'closed', 'trade': t})
        return jsonify({'error': 'trade not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete_logged_trade', methods=['POST'])
def delete_logged_trade():
    try:
        tid = (request.json or {}).get('id')
        trades = load_trade_log()
        trades = [t for t in trades if t['id'] != tid]
        save_trade_log(trades)
        return jsonify({'status': 'deleted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trade_log', methods=['GET'])
def get_trade_log():
    '''fetch all logged trades with current prices and unrealised p/l'''
    try:
        trades = load_trade_log()
        if not trades:
            return jsonify({'trades': [], 'summary': _empty_summary()})

        #fetch current prices in one pass per unique open ticker
        open_trades = [t for t in trades if t['status'] == 'open']
        unique_tickers = list({t['ticker'] for t in open_trades})
        current_prices = {}
        for tk in unique_tickers:
            try:
                info = yf.Ticker(tk).info or {}
                p = info.get('currentPrice', info.get('regularMarketPrice', 0))
                if p: current_prices[tk] = float(p)
            except: pass

        #compute live p/l for open trades
        for t in trades:
            if t['status'] == 'open':
                cp = current_prices.get(t['ticker'])
                if cp:
                    t['current_price'] = round(cp, 2)
                    sign = 1 if t['action'] == 'buy' else -1
                    t['unrealised_pl_pct'] = round(sign * (cp - t['entry_price']) / t['entry_price'] * 100, 2)
                    pl_gbp, pl_pct_gbp, fx_at_entry = _gbp_pl_for_trade(t, cp)
                    if pl_gbp is not None:
                        t['unrealised_pl_gbp'] = pl_gbp
                        t['unrealised_pl_pct_gbp'] = pl_pct_gbp
                        t['fx_at_entry_to_gbp'] = fx_at_entry
                    t['unrealised_pl_value'] = round(t['unrealised_pl_pct'] / 100 * t['entry_price'] * t['shares'], 2)
                    try:
                        days_held = (pd.Timestamp.now() - pd.to_datetime(t['entry_date'])).days
                        t['days_held'] = int(days_held)
                    except: t['days_held'] = 0
                else:
                    t['current_price'] = None
                    t['unrealised_pl_pct'] = None
                    t['unrealised_pl_value'] = None
                    t['days_held'] = 0

        #summary stats by strategy
        summary = _trade_log_summary(trades)
        return jsonify(sanitise({'trades': trades, 'summary': summary}))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e), 'trades': []}), 500


def _empty_summary():
    return {'total_trades':0,'open':0,'closed':0,'win_rate':0,'avg_realised_pct':0,
            'total_realised_pct':0,'by_strategy':{}}


def _trade_log_summary(trades):
    #9.1: nan guard. `is not None` passes a nan straight through, because nan
    #is a perfectly valid float - the same trap the 8.2 notes describe for the
    #paper book. six trades closed on 2026-06-11 with a nan exit price, and
    #one of them was enough to turn every total on this tab into nan while
    #the win rate silently counted them as losses (nan > 0 is False).
    #trades with an unknowable P/L are now excluded from the numbers and
    #counted separately, so they are visible rather than quietly wrong.
    def _known(t):
        v = t.get('realised_pl_pct')
        try:
            return v is not None and np.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    all_closed = [t for t in trades if t['status'] == 'closed']
    closed = [t for t in all_closed if _known(t)]
    unknown = len(all_closed) - len(closed)
    open_ = [t for t in trades if t['status'] == 'open']
    wins = [t for t in closed if t['realised_pl_pct'] > 0]
    summary = {
        'total_trades': len(trades),
        'open': len(open_),
        'closed': len(closed),
        'closed_unknown_pl': unknown,
        'win_rate': round(len(wins)/len(closed)*100, 1) if closed else 0,
        'avg_realised_pct': round(float(np.mean([t['realised_pl_pct'] for t in closed])), 2) if closed else 0,
        'total_realised_pct': round(sum(t['realised_pl_pct'] for t in closed), 2) if closed else 0,
    }
    #break down by strategy
    strategies = {}
    for t in closed:
        s = t['strategy']
        strategies.setdefault(s, {'closed':0,'wins':0,'total_pct':0})
        strategies[s]['closed'] += 1
        if t['realised_pl_pct'] > 0: strategies[s]['wins'] += 1
        strategies[s]['total_pct'] += t['realised_pl_pct']
    for s in strategies:
        d = strategies[s]
        d['win_rate'] = round(d['wins']/d['closed']*100, 1) if d['closed'] else 0
        d['avg_pct'] = round(d['total_pct']/d['closed'], 2) if d['closed'] else 0
    #include open count per strategy too
    for t in open_:
        s = t['strategy']
        strategies.setdefault(s, {'closed':0,'wins':0,'total_pct':0,'win_rate':0,'avg_pct':0})
        strategies[s].setdefault('open', 0)
        strategies[s]['open'] = strategies[s].get('open', 0) + 1
    summary['by_strategy'] = strategies
    return summary

# ==================== AI PAPER TRADER ====================

paper_cycle_status = {'active': False, 'progress': 0, 'message': '', 'triggered_by': None}

def _try_with_retry(fn, *args, retries=2, **kwargs):
    '''retry a flaky function up to N times with brief backoff, returns None on total failure'''
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    return None

def _init_paper_portfolio():
    '''fresh paper portfolio at starting cash'''
    return {
        'cash_gbp': STARTING_CASH_GBP,
        'positions': [],
        'closed_trades': [],
        'equity_curve': [{
            'date': datetime.datetime.now().isoformat(),
            'equity_gbp': STARTING_CASH_GBP
        }],
        'cycles_run': 0,
        'last_cycle': None,
        'activity_log': [],
        'auto_run_enabled': False,
        'fees_paid_gbp': 0.0,
        'schema_version': PORTFOLIO_SCHEMA_VERSION
    }

def _load_paper_portfolio():
    if not os.path.exists(PAPER_PORTFOLIO_PATH):
        p = _init_paper_portfolio(); _save_paper_portfolio(p); return p
    try:
        with open(PAPER_PORTFOLIO_PATH, 'r') as f:
            p = json.load(f)
            if 'auto_run_enabled' not in p: p['auto_run_enabled'] = False
            return p
    except Exception:
        return _init_paper_portfolio()

def _save_paper_portfolio(portfolio):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PAPER_PORTFOLIO_PATH, 'w') as f:
        json.dump(portfolio, f, indent=2, default=str)


def _cleanup_retired_strategy_positions():
    '''
    on startup, close any open paper positions tagged with a retired strategy.
    7.8 retired meanrev, week52_high and low_beta_trend, so any positions still
    sitting in the file under those tags have no exit rules any more and would
    just hang there forever. close at current market price, return cash to the
    portfolio, log to both the paper closed_trades list and the main trade log
    so the books stay continuous. silent no-op if nothing needs cleaning.
    '''
    if not os.path.exists(PAPER_PORTFOLIO_PATH):
        return
    try:
        portfolio = _load_paper_portfolio()
    except Exception:
        return

    open_positions = portfolio.get('positions', [])
    orphans = [p for p in open_positions if p.get('strategy') in RETIRED_STRATEGIES]
    if not orphans:
        return

    print(f"\n[startup cleanup] found {len(orphans)} open position(s) tagged with retired strategies, closing them now")
    activity = portfolio.setdefault('activity_log', [])
    closed = portfolio.setdefault('closed_trades', [])
    kept = []

    for pos in open_positions:
        if pos.get('strategy') not in RETIRED_STRATEGIES:
            kept.append(pos)
            continue

        ticker = pos.get('ticker', '?')
        strat = pos.get('strategy', '?')
        entry_native = float(pos.get('entry_price_native', 0) or 0)
        shares = float(pos.get('shares', 0) or 0)
        ccy = pos.get('currency', 'USD')

        #try to get a live price, fall back to entry price if the network is down
        exit_price_native, _ = _paper_price_and_currency(ticker)
        if exit_price_native is None:
            exit_price_native = entry_native
            print(f"[startup cleanup] {ticker} ({strat}): could not fetch live price, closing at entry £{entry_native}")

        fx_at_entry = float(pos.get('fx_at_entry_to_gbp') or _get_fx_to_gbp(ccy))
        fx_now = _get_fx_to_gbp(ccy)
        cost_gbp = shares * entry_native * fx_at_entry
        proceeds_gbp = shares * exit_price_native * fx_now
        realised_pl_gbp = proceeds_gbp - cost_gbp
        realised_pl_pct = ((exit_price_native - entry_native) / entry_native * 100) if entry_native else 0.0

        portfolio['cash_gbp'] = float(portfolio.get('cash_gbp', 0)) + proceeds_gbp

        closed.append({
            'ticker': ticker, 'strategy': strat,
            'entry_price_native': entry_native, 'exit_price_native': exit_price_native,
            'shares': shares, 'currency': ccy,
            'entry_date': pos.get('entry_date'),
            'exit_date': datetime.datetime.now().isoformat(),
            'realised_pl_gbp': round(realised_pl_gbp, 2),
            'realised_pl_pct': round(realised_pl_pct, 2),
            'exit_reason': 'strategy_retired',
            'confidence_at_entry': pos.get('confidence_at_entry'),
        })

        msg = f"closed {ticker} ({strat}) strategy_retired £{realised_pl_gbp:+.2f} ({realised_pl_pct:+.2f}%)"
        activity.append({'date': datetime.datetime.now().isoformat(), 'event': msg})
        print(f"[startup cleanup] {msg}")

        #mirror to the main trade log so per-strategy stats stay honest
        try:
            _log_paper_trade_to_main_log(pos, exit_price_native, 'strategy_retired')
        except Exception as e:
            print(f"[startup cleanup] could not mirror {ticker} to main trade log: {e}")

    portfolio['positions'] = kept
    activity.append({'date': datetime.datetime.now().isoformat(),
                     'event': f'7.8 startup cleanup: closed {len(orphans)} orphaned position(s)'})
    _save_paper_portfolio(portfolio)
    print(f"[startup cleanup] done, {len(kept)} active position(s) remain\n")


def _migrate_portfolio_to_v8():
    '''
    8.0 one-off migration, runs once on the first startup after upgrading.
    1) closes every open paper position at the best available price (live,
       falling back to last cached, falling back to entry) and mirrors each
       close into the main trade log so the history survives
    2) archives the entire old portfolio file to
       data/paper_portfolio_v7_archive.json
    3) starts a fresh book at the new £10,000 with a clean equity curve
    portfolios already stamped with the 8.0 schema are left alone, so this
    is safe to run on every startup.
    '''
    if not os.path.exists(PAPER_PORTFOLIO_PATH):
        _save_paper_portfolio(_init_paper_portfolio())
        print(f"\n[8.0 migration] no existing portfolio, fresh book created at £{STARTING_CASH_GBP:,.0f}\n")
        return
    try:
        portfolio = _load_paper_portfolio()
    except Exception:
        _save_paper_portfolio(_init_paper_portfolio())
        return
    if portfolio.get('schema_version') == PORTFOLIO_SCHEMA_VERSION:
        return

    open_positions = portfolio.get('positions', [])
    print(f"\n[8.0 migration] migrating paper portfolio: closing {len(open_positions)} open position(s), resetting to £{STARTING_CASH_GBP:,.0f}")

    for pos in open_positions:
        ticker = pos.get('ticker', '?')
        strat = pos.get('strategy', '?')
        entry_native = float(pos.get('entry_price_native', 0) or 0)
        shares = float(pos.get('shares', 0) or 0)
        ccy = pos.get('currency', 'USD')

        #best available exit price: live, then last cached, then entry
        exit_price_native, _ = _paper_price_and_currency(ticker)
        if exit_price_native is None:
            exit_price_native = pos.get('current_price_native') or entry_native
            print(f"[8.0 migration] {ticker} ({strat}): no live price, closing at {exit_price_native}")

        realised_pl_pct = ((exit_price_native - entry_native) / entry_native * 100) if entry_native else 0.0
        print(f"[8.0 migration] closed {ticker} ({strat}) {shares:.4f} sh @ {exit_price_native:.2f} {ccy} ({realised_pl_pct:+.2f}%)")

        #mirror to the main trade log so per-strategy stats keep the history
        try:
            _log_paper_trade_to_main_log(pos, exit_price_native, 'closed_for_v8_reset')
        except Exception as e:
            print(f"[8.0 migration] could not mirror {ticker} to main trade log: {e}")

    #archive the whole old book, then start fresh
    try:
        archive_path = os.path.join('data', 'paper_portfolio_v7_archive.json')
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(archive_path, 'w') as f:
            json.dump(portfolio, f, indent=2, default=str)
        print(f"[8.0 migration] old portfolio archived to {archive_path}")
    except Exception as e:
        print(f"[8.0 migration] archive failed (continuing anyway): {e}")

    fresh = _init_paper_portfolio()
    fresh['activity_log'] = [{
        'date': datetime.datetime.now().isoformat(),
        'event': f'8.0 migration: closed {len(open_positions)} position(s), book reset to £{STARTING_CASH_GBP:,.0f} with live fees enabled'
    }]
    _save_paper_portfolio(fresh)
    print(f"[8.0 migration] done, fresh book at £{STARTING_CASH_GBP:,.0f}\n")


def _repair_nan_portfolio():
    '''
    8.2 one-off repair for books poisoned by nan prices or fx. runs on every
    startup but only acts when it finds non-finite numbers. the rebuild works
    from the accounting identity:
        cash = starting cash
             + realised P/L on every finite closed trade
             - (cost + entry fee) of every open position
    closed trades whose exit price was corrupted get voided: exit set to the
    entry price, realised P/L zero, the cash that went in fully refunded. we
    genuinely do not know what those exits were worth, so a flat void is the
    only honest reconstruction. positions whose size or cost is corrupted
    (the £nan sweep) are dropped entirely - reversing their unknown nan
    deduction is exactly what the identity above does. the corrupt file is
    archived to data/paper_portfolio_nan_archive.json first.
    '''
    if not os.path.exists(PAPER_PORTFOLIO_PATH):
        return
    try:
        p = _load_paper_portfolio()
    except Exception:
        return

    poisoned = not _finite(p.get('cash_gbp')) or not _finite(p.get('fees_paid_gbp', 0))
    for pos in p.get('positions', []):
        if not (_finite(pos.get('shares')) and _finite(pos.get('cost_gbp')) and _finite(pos.get('entry_price_native'))):
            poisoned = True
    for ct in p.get('closed_trades', []):
        if not (_finite(ct.get('realised_pl_gbp')) and _finite(ct.get('proceeds_gbp'))):
            poisoned = True
    for pt in p.get('equity_curve', []):
        if not _finite(pt.get('equity_gbp')):
            poisoned = True
    if not poisoned:
        return

    print('\n[8.2 repair] nan poisoning detected in the paper book, rebuilding from first principles')
    try:
        archive_path = os.path.join(DATA_DIR, 'paper_portfolio_nan_archive.json')
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(archive_path, 'w') as f:
            json.dump(p, f, indent=2, default=str)
        print(f'[8.2 repair] corrupt book archived to {archive_path}')
    except Exception as e:
        print(f'[8.2 repair] archive failed (continuing anyway): {e}')

    #open positions: keep the healthy, drop the unknowable
    kept_positions = []
    dropped = 0
    for pos in p.get('positions', []):
        healthy = (_finite(pos.get('shares')) and float(pos.get('shares') or 0) > 0
                   and _finite(pos.get('cost_gbp')) and float(pos.get('cost_gbp') or 0) > 0
                   and _finite(pos.get('entry_price_native')) and float(pos.get('entry_price_native') or 0) > 0)
        if healthy:
            if not _finite(pos.get('entry_fee_gbp')):
                pos['entry_fee_gbp'] = 0.0
            #clear poisoned mark-to-market leftovers, next cycle refreshes them
            for k in ('current_value_gbp', 'unrealised_pl_gbp', 'unrealised_pl_pct',
                      'current_price_native', 'peak_price_native', 'peak_gain_pct'):
                if k in pos and not _finite(pos.get(k)):
                    pos.pop(k, None)
            kept_positions.append(pos)
        else:
            dropped += 1
            print(f"[8.2 repair] dropped corrupt position {pos.get('ticker', '?')} ({pos.get('strategy', '?')}) - size/cost unknowable")
    p['positions'] = kept_positions

    #closed trades: void the corrupted exits. the exit becomes a flat sale
    #at the entry price with no exit fee, but the entry fee stays as a real
    #loss - that money genuinely left the account when the position opened,
    #only the exit price is unknowable. keeps the accounting identity
    #realised = proceeds - cost - entry_fee exactly true.
    voided = 0
    for ct in p.get('closed_trades', []):
        if _finite(ct.get('realised_pl_gbp')) and _finite(ct.get('proceeds_gbp')):
            continue
        cost = float(ct.get('cost_gbp')) if _finite(ct.get('cost_gbp')) else 0.0
        efee = float(ct.get('entry_fee_gbp')) if _finite(ct.get('entry_fee_gbp')) else 0.0
        entry_px = float(ct.get('entry_price_native')) if _finite(ct.get('entry_price_native')) else 0.0
        ct['entry_fee_gbp'] = efee
        ct['exit_price_native'] = entry_px
        ct['proceeds_gbp'] = cost
        ct['realised_pl_gbp'] = -efee
        ct['realised_pl_pct'] = round(-efee / cost * 100, 2) if cost else 0.0
        ct['exit_fee_gbp'] = 0.0
        ct['fees_gbp'] = efee
        ct['exit_reason'] = f"{ct.get('exit_reason', '')} (voided in 8.2 repair, exit price was corrupted)"
        voided += 1

    #rebuild cash from the identity, then fees from what survives
    realised_total = sum(float(ct.get('realised_pl_gbp') or 0) for ct in p.get('closed_trades', []))
    open_outlay = sum(float(pos.get('cost_gbp') or 0) + float(pos.get('entry_fee_gbp') or 0) for pos in p['positions'])
    p['cash_gbp'] = STARTING_CASH_GBP + realised_total - open_outlay

    fees = 0.0
    for pos in p['positions']:
        fees += float(pos.get('entry_fee_gbp') or 0)
    for ct in p.get('closed_trades', []):
        if _finite(ct.get('entry_fee_gbp')): fees += float(ct['entry_fee_gbp'])
        if _finite(ct.get('exit_fee_gbp')): fees += float(ct['exit_fee_gbp'])
    p['fees_paid_gbp'] = fees

    #equity curve: keep only real points
    p['equity_curve'] = [pt for pt in p.get('equity_curve', []) if _finite(pt.get('equity_gbp'))]

    p.setdefault('activity_log', []).append({
        'date': datetime.datetime.now().isoformat(),
        'event': f'8.2 repair: voided {voided} corrupted close(s), dropped {dropped} corrupt position(s), cash rebuilt to £{p["cash_gbp"]:.2f}'
    })
    _save_paper_portfolio(p)
    print(f"[8.2 repair] done: voided {voided} close(s), dropped {dropped} position(s), cash rebuilt to £{p['cash_gbp']:.2f}\n")


def _get_fx_to_gbp(from_currency):
    '''spot fx to gbp, london pence is already normalised before this'''
    if from_currency == 'GBP': return 1.0
    try:
        pair = f'{from_currency}GBP=X'
        hist = yf.Ticker(pair).history(period='1d')
        if len(hist) > 0:
            #8.2: nan fx was the other way the book got poisoned. only a
            #finite positive rate is allowed out of here
            closes = hist['Close'].dropna()
            if len(closes) > 0:
                rate = float(closes.iloc[-1])
                if _finite(rate) and rate > 0:
                    return rate
    except Exception: pass
    fallbacks = {'USD': 0.79, 'EUR': 0.85, 'CAD': 0.58, 'AUD': 0.52}
    return fallbacks.get(from_currency, 1.0)


def _finite(x):
    '''
    8.2: true only for real, usable numbers. nan and inf slip straight
    through `is None` checks and python truthiness (nan is truthy), which
    is exactly how one bad yfinance row poisoned the whole book. every
    number arriving from outside (prices, fx) and every value about to
    mutate cash goes through this first.
    '''
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _paper_fee_gbp(gross_gbp, currency):
    '''8.0: one-side trading cost in pounds. fx fee only applies off-gbp'''
    rate = SPREAD_SLIPPAGE_PCT + (FX_FEE_PCT if currency != 'GBP' else 0.0)
    return abs(gross_gbp) * rate / 100.0


def _recent_ann_vol(ticker_sym):
    '''8.0: annualised volatility from ~3 months of daily closes, None on failure'''
    try:
        hist = drop_incomplete_bars(yf.Ticker(ticker_sym).history(period='3mo'))
        if len(hist) < 30: return None
        rets = hist['Close'].pct_change().dropna()
        if rets.empty: return None
        return float(rets.std() * (252 ** 0.5))
    except Exception:
        return None


def _get_sweep_position(portfolio):
    '''8.0: the single index_sweep position if one exists'''
    for p in portfolio.get('positions', []):
        if p.get('strategy') == 'index_sweep': return p
    return None


def _non_sweep_positions(portfolio):
    '''8.0: real strategy positions, the sweep does not occupy a slot'''
    return [p for p in portfolio.get('positions', []) if p.get('strategy') != 'index_sweep']


def _sell_sweep(portfolio, amount_gbp_needed, reason, activity, now_iso):
    '''
    8.0: sell some or all of the SPY sweep to free cash, paying exit fees.
    amount_gbp_needed=None means sell the lot. returns the net £ freed.
    partial sales reduce shares and cost proportionally so the remaining
    slice keeps an honest basis.
    '''
    pos = _get_sweep_position(portfolio)
    if pos is None: return 0.0
    #8.2: `a or b` passes nan through (nan is truthy), so check each
    #candidate price properly and refuse corrupted share/cost figures
    if not _finite(pos.get('shares')) or not _finite(pos.get('cost_gbp')):
        return 0.0
    price = pos.get('current_price_native')
    if not _finite(price) or float(price) <= 0:
        price = pos.get('entry_price_native')
    if not _finite(price) or float(price) <= 0:
        return 0.0
    ccy = pos.get('currency', 'USD')
    fx = _get_fx_to_gbp(ccy)
    full_value = float(pos['shares']) * float(price) * fx
    if full_value <= 0: return 0.0
    if amount_gbp_needed is None or amount_gbp_needed >= full_value * 0.97:
        frac = 1.0
    else:
        #sell a touch extra so the exit fee does not leave us short
        frac = min(1.0, (amount_gbp_needed / full_value) * 1.03)
    sell_shares = float(pos['shares']) * frac
    gross = sell_shares * float(price) * fx
    fee = _paper_fee_gbp(gross, ccy)
    net = gross - fee
    cost_slice = float(pos['cost_gbp']) * frac
    entry_fee_slice = float(pos.get('entry_fee_gbp', 0) or 0) * frac
    realised = net - cost_slice - entry_fee_slice
    entry_px = float(pos.get('entry_price_native', 0) or 0)
    portfolio['cash_gbp'] = float(portfolio.get('cash_gbp', 0) or 0) + net
    portfolio['fees_paid_gbp'] = float(portfolio.get('fees_paid_gbp', 0) or 0) + fee
    portfolio.setdefault('closed_trades', []).append({
        'ticker': pos['ticker'], 'strategy': 'index_sweep',
        'entry_price_native': entry_px, 'exit_price_native': float(price),
        'shares': sell_shares, 'currency': ccy,
        'entry_date': pos.get('entry_date'), 'exit_date': now_iso,
        'exit_reason': reason,
        'realised_pl_gbp': round(realised, 2),
        'realised_pl_pct': round((float(price) - entry_px) / entry_px * 100, 2) if entry_px else 0.0,
        'exit_fee_gbp': round(fee, 2), 'proceeds_gbp': round(net, 2),
        'confidence_at_entry': None,
    })
    if frac >= 1.0:
        portfolio['positions'].remove(pos)
        activity.append(f"sweep: sold all SPY for £{net:.2f} net ({reason}, fee £{fee:.2f})")
    else:
        pos['shares'] = float(pos['shares']) - sell_shares
        pos['cost_gbp'] = float(pos['cost_gbp']) - cost_slice
        if pos.get('entry_fee_gbp') is not None:
            pos['entry_fee_gbp'] = float(pos.get('entry_fee_gbp', 0) or 0) - entry_fee_slice
        pos['current_value_gbp'] = float(pos['shares']) * float(price) * fx
        activity.append(f"sweep: sold £{net:.2f} of SPY to free cash ({reason}, fee £{fee:.2f})")
    return net


def _sweep_idle_cash(portfolio, regime_data, activity, now_iso):
    '''
    8.0: end-of-cycle step. when the regime is friendly, push spare cash
    above a small float into SPY so it earns market beta instead of nothing.
    entry pays its fee like any other fill.
    '''
    if not INDEX_SWEEP_ENABLED: return
    regime = (regime_data or {}).get('regime', 'neutral')
    if regime not in INDEX_SWEEP_REGIMES: return
    #8.2 circuit breaker: nan cash made `spare < min` False (nan comparisons
    #always fail) so the sweep happily parked £nan. refuse to act unless the
    #balance is a real number
    if not _finite(portfolio.get('cash_gbp')):
        activity.append('sweep: skipped - cash balance is corrupted, startup repair needed')
        return
    spare = float(portfolio['cash_gbp']) - INDEX_SWEEP_CASH_FLOOR_GBP
    if spare < INDEX_SWEEP_MIN_GBP: return
    price, ccy = _paper_price_and_currency(INDEX_SWEEP_TICKER)
    if not _finite(price) or (price or 0) <= 0:
        activity.append('sweep: could not price SPY, leaving cash idle this cycle')
        return
    fx = _get_fx_to_gbp(ccy)
    fee_rate = (SPREAD_SLIPPAGE_PCT + (FX_FEE_PCT if ccy != 'GBP' else 0.0)) / 100.0
    invest = spare / (1.0 + fee_rate)
    fee = invest * fee_rate
    shares = (invest / fx) / price
    portfolio['cash_gbp'] = float(portfolio.get('cash_gbp', 0) or 0) - (invest + fee)
    portfolio['fees_paid_gbp'] = float(portfolio.get('fees_paid_gbp', 0) or 0) + fee
    existing = _get_sweep_position(portfolio)
    if existing:
        old_shares = float(existing['shares'])
        old_entry = float(existing.get('entry_price_native', price) or price)
        new_shares = old_shares + shares
        existing['entry_price_native'] = ((old_shares * old_entry) + (shares * price)) / new_shares if new_shares else price
        existing['shares'] = new_shares
        existing['cost_gbp'] = float(existing['cost_gbp']) + invest
        existing['entry_fee_gbp'] = float(existing.get('entry_fee_gbp', 0) or 0) + fee
        existing['current_price_native'] = price
        existing['current_value_gbp'] = new_shares * price * fx
        activity.append(f"sweep: topped up SPY with £{invest:.2f} spare cash (fee £{fee:.2f})")
    else:
        portfolio['positions'].append({
            'ticker': INDEX_SWEEP_TICKER, 'strategy': 'index_sweep',
            'confidence_at_entry': None,
            'entry_price_native': price, 'shares': shares,
            'currency': ccy, 'fx_at_entry': fx,
            'cost_gbp': invest, 'entry_fee_gbp': fee,
            'entry_date': now_iso,
            'entry_reason': f'idle cash sweep in {regime} regime',
            'peak_price_native': price,
            'current_price_native': price,
            'current_value_gbp': invest,
        })
        activity.append(f"sweep: parked £{invest:.2f} spare cash in SPY ({regime} regime, fee £{fee:.2f})")

def _paper_price_and_currency(sym):
    '''latest close and normalised currency with retry, london pence becomes pounds'''
    for attempt in range(3):
        try:
            info = yf.Ticker(sym).info or {}
            raw_ccy = info.get('currency', 'USD')
            hist = yf.Ticker(sym).history(period='5d')
            #8.2: yfinance can hand back an incomplete final row (today's
            #bar before the close) whose value is nan. dropna first, and
            #reject anything non-finite or non-positive so a bad row reads
            #as a failed fetch and the caller's fallback chain takes over
            closes = hist['Close'].dropna() if len(hist) else hist
            if len(closes) == 0:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1)); continue
                return None, None
            price = float(closes.iloc[-1])
            if not _finite(price) or price <= 0:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1)); continue
                return None, None
            if raw_ccy == 'GBp': return price / 100.0, 'GBP'
            return price, raw_ccy
        except Exception:
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    return None, None

def _calculate_position_size_gbp(confidence, total_equity_gbp, available_cash_gbp, ann_vol=None):
    '''
    size 5pct to 15pct of equity scaled by confidence above 60.
    8.0: the result is then tilted by realised volatility - calmer names get
    a bigger slice, jumpier ones a smaller one, clamped 0.6x to 1.3x so the
    tilt never dominates the confidence signal
    '''
    if available_cash_gbp <= 0:
        return 0.0

    conf_clamped = max(60, min(100, confidence))
    pct = MIN_POSITION_PCT + (MAX_POSITION_PCT - MIN_POSITION_PCT) * ((conf_clamped - 60) / 40.0)
    target_gbp = total_equity_gbp * pct

    if ann_vol and ann_vol > 0:
        mult = max(VOL_SIZE_MIN_MULT, min(VOL_SIZE_MAX_MULT, TARGET_POSITION_VOL / ann_vol))
        target_gbp *= mult

    return min(target_gbp, available_cash_gbp * CASH_BUFFER_PCT)


def _mark_position_to_market(pos):
    '''
    7.9: robust mark-to-market for a single open paper position.
    returns (price_native, is_stale, fx_to_gbp) and never lets a position fall
    out of the equity calculation silently. fallback chain:
      1) live price from yfinance (fresh)
      2) last known current_price_native saved on the position object (stale)
      3) entry_price_native (stale, last resort, treats position as flat)
    the old code returned None from a failed fetch and the caller did
    `continue`, dropping the position from total_value_gbp. that caused the
    equity-collapse display bug when yfinance rate-limited the whole batch.
    '''
    live, ccy = _paper_price_and_currency(pos['ticker'])
    if _finite(live) and float(live) > 0:
        return float(live), False, _get_fx_to_gbp(pos.get('currency', ccy or 'USD'))
    #fall back to whatever we recorded last cycle, refusing stored nans
    cached = pos.get('current_price_native')
    if _finite(cached) and float(cached) > 0:
        return float(cached), True, _get_fx_to_gbp(pos.get('currency', 'USD'))
    #last resort: pretend nothing has moved since entry
    return float(pos['entry_price_native']), True, _get_fx_to_gbp(pos.get('currency', 'USD'))


def _paper_rsi(ticker_sym, period=14):
    try:
        hist = drop_incomplete_bars(yf.Ticker(ticker_sym).history(period='3mo'))
        if len(hist) < period + 1: return None
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return float(val) if not pd.isna(val) else None
    except Exception: return None

def _check_position_exit(position, current_price_native, exit_ctx=None):
    '''
    per-strategy exit rules for the AI paper trader.
    momentum-group strategies (momentum, momentum_12_1) use trailing stops:
    once profit hits +10%, the stop ratchets up to peak-5%. this captures
    the long right tail of momentum returns while still cutting losers early.
    peak_price_native is updated in the mark-to-market step.
    '''
    exit_ctx = exit_ctx or {}
    strategy = position['strategy']
    entry = position['entry_price_native']
    peak = position.get('peak_price_native', entry)
    pct_change = (current_price_native - entry) / entry
    peak_pct = (peak - entry) / entry
    drawdown_from_peak = (current_price_native - peak) / peak if peak > 0 else 0
    entry_dt = datetime.datetime.fromisoformat(position['entry_date'])
    days_held = (datetime.datetime.now() - entry_dt).days
    rsi = exit_ctx.get('rsi')

    if strategy == 'cluster':
        if pct_change <= -0.08: return True, 'stop_loss'
        if pct_change >= 0.15:  return True, 'target_hit'
        if days_held >= 60:     return True, 'time_exit'

    elif strategy == 'pead':
        if pct_change <= -0.06: return True, 'stop_loss'
        if pct_change >= 0.10:  return True, 'target_hit'
        if days_held >= 30:     return True, 'time_exit'

    elif strategy == 'momentum':
        #trailing stop logic: below +10% use fixed -9% stop. above +10% use
        #peak-5% trailing. no hard target so big winners keep running.
        if peak_pct < 0.10:
            if pct_change <= -0.09: return True, 'stop_loss'
        else:
            if drawdown_from_peak <= -0.05: return True, f'trailing_stop_at_{peak_pct*100:.0f}pct_peak'
        if exit_ctx.get('below_sma200'): return True, 'trend_broken_200dma'
        if exit_ctx.get('below_sma50') and pct_change < 0: return True, 'lost_50dma'
        if days_held >= 100:    return True, 'time_exit'

    elif strategy == 'momentum_12_1':
        #stricter momentum, slightly tighter trail (4%) once trailing engages
        if peak_pct < 0.10:
            if pct_change <= -0.08: return True, 'stop_loss'
        else:
            if drawdown_from_peak <= -0.04: return True, f'trailing_stop_at_{peak_pct*100:.0f}pct_peak'
        if exit_ctx.get('below_sma200'): return True, 'trend_broken_200dma'
        if exit_ctx.get('below_sma50') and pct_change < 0: return True, 'lost_50dma'
        if days_held >= 120:    return True, 'time_exit'

    elif strategy == 'quality':
        if pct_change <= -0.12: return True, 'stop_loss'
        if pct_change >= 0.20:  return True, 'target_hit'
        if exit_ctx.get('below_sma200'): return True, 'long_term_trend_broken'
        if days_held >= 180:    return True, 'time_exit'

    elif strategy == 'shareholder_yield':
        if pct_change <= -0.10: return True, 'stop_loss'
        if pct_change >= 0.15:  return True, 'target_hit'
        if exit_ctx.get('below_sma200'): return True, 'trend_broken_200dma'
        if days_held >= 180:    return True, 'time_exit'

    return False, None

def _log_paper_trade_to_main_log(pos, exit_price_native, exit_reason):
    try:
        trades = load_trade_log()
        realised_pct = (exit_price_native - pos['entry_price_native']) / pos['entry_price_native'] * 100
        trades.append({
            'id': int(time.time() * 1000),
            'ticker': pos['ticker'],
            'strategy': f"ai_paper_{pos['strategy']}",
            'action': 'buy', 'entry_price': pos['entry_price_native'],
            'entry_date': pos['entry_date'][:10], 'shares': pos['shares'],
            'currency': pos['currency'], 'target_price': None, 'stop_price': None,
            'notes': f"paper auto closed: {exit_reason}", 'status': 'closed',
            'exit_price': exit_price_native,
            'exit_date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'realised_pl_pct': round(realised_pct, 2),
            'logged_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        })
        save_trade_log(trades)
    except Exception: pass


def _execute_paper_cycle(scan_count=300, triggered_by='manual'):
    '''one full cycle. updates paper_cycle_status as it goes'''
    global paper_cycle_status
    paper_cycle_status = {'active': True, 'progress': 0,
                          'message': 'starting cycle...', 'triggered_by': triggered_by}
    portfolio = _load_paper_portfolio()
    activity = []
    now_iso = datetime.datetime.now().isoformat()
    #fetch regime once at the top of the cycle so all ranking uses the same view
    regime_data = get_market_regime()
    activity.append(f"regime: {regime_data.get('regime_label', '?')}")
    if regime_data.get('favoured_strategies'):
        activity.append(f"favoured: {', '.join(regime_data['favoured_strategies'])}")
    if regime_data.get('suppressed_strategies'):
        activity.append(f"suppressed: {', '.join(regime_data['suppressed_strategies'])}")
    try:
        #step 1 mark to market
        #7.9: never silently drop a position. _mark_position_to_market falls
        #back to cached or entry price when the live fetch fails, which fixes
        #the equity-collapse display bug from 7.8 (yfinance rate-limited the
        #whole batch and every position vanished from total_value_gbp).
        paper_cycle_status['message'] = 'marking positions to market...'
        total_value_gbp = 0.0
        stale_count = 0
        for pos in portfolio['positions']:
            try:
                price, is_stale, fx = _mark_position_to_market(pos)
                pos['current_price_native'] = price
                pos['price_is_stale'] = is_stale
                if is_stale: stale_count += 1
                #seed peak for legacy positions opened before trailing stops existed
                if 'peak_price_native' not in pos:
                    pos['peak_price_native'] = max(price, pos['entry_price_native'])
                #ratchet peak upward only, never down. only when the price is
                #fresh, so a stale fallback can not ratchet the trailing stop.
                if not is_stale and price > pos['peak_price_native']:
                    pos['peak_price_native'] = price
                pos['current_value_gbp'] = pos['shares'] * price * fx
                #8.0: unrealised P/L nets out the entry fee already paid
                pos['unrealised_pl_gbp'] = pos['current_value_gbp'] - pos['cost_gbp'] - float(pos.get('entry_fee_gbp', 0) or 0)
                pos['unrealised_pl_pct'] = (price - pos['entry_price_native']) / pos['entry_price_native'] * 100
                pos['peak_gain_pct'] = (pos['peak_price_native'] - pos['entry_price_native']) / pos['entry_price_native'] * 100
                total_value_gbp += pos['current_value_gbp']
            except Exception as e:
                activity.append(f"warning could not price {pos['ticker']}: {str(e)[:60]}")
        if stale_count:
            activity.append(f"warning: {stale_count}/{len(portfolio['positions'])} positions priced from stale cache (yfinance unavailable)")
        total_equity_gbp = portfolio['cash_gbp'] + total_value_gbp

        #step 2 check exits
        #7.9: skip exit evaluation on stale-priced positions. running stop-loss
        #logic on a fallback price could fire a spurious sale (eg if entry
        #price happens to be below the stop threshold for that strategy).
        paper_cycle_status['message'] = 'checking exit conditions...'
        to_close = []
        for pos in portfolio['positions']:
            if 'current_price_native' not in pos: continue
            if pos.get('price_is_stale'): continue
            #8.0: the sweep is managed by regime/cash logic, not strategy exits
            if pos.get('strategy') == 'index_sweep': continue
            exit_ctx = _paper_exit_context(pos['ticker'])
            should_exit, reason = _check_position_exit(pos, pos['current_price_native'], exit_ctx)
            if should_exit: to_close.append((pos, reason))
        for pos, reason in to_close:
            #8.0: the exit pays its fee like a real fill, and realised P/L
            #nets out both the entry and exit fees
            gross = pos['current_value_gbp']
            #8.2 circuit breaker: never let a non-finite value touch cash.
            #keep the position and try again next cycle with fresh prices
            if not _finite(gross) or not _finite(pos.get('cost_gbp')):
                activity.append(f"skipped closing {pos['ticker']} - marked value was corrupted by a bad data fetch, retrying next cycle")
                continue
            exit_fee = _paper_fee_gbp(gross, pos.get('currency', 'USD'))
            proceeds = gross - exit_fee
            entry_fee = float(pos.get('entry_fee_gbp', 0) or 0)
            realised_pl = proceeds - pos['cost_gbp'] - entry_fee
            realised_pct = realised_pl / pos['cost_gbp'] * 100
            portfolio['cash_gbp'] += proceeds
            portfolio['fees_paid_gbp'] = float(portfolio.get('fees_paid_gbp', 0) or 0) + exit_fee
            portfolio['closed_trades'].append({
                **pos, 'exit_date': now_iso,
                'exit_price_native': pos['current_price_native'],
                'exit_reason': reason, 'realised_pl_gbp': realised_pl,
                'realised_pl_pct': realised_pct, 'proceeds_gbp': proceeds,
                'exit_fee_gbp': round(exit_fee, 2),
                'fees_gbp': round(entry_fee + exit_fee, 2)
            })
            portfolio['positions'].remove(pos)
            _log_paper_trade_to_main_log(pos, pos['current_price_native'], reason)
            activity.append(f"closed {pos['ticker']} ({pos['strategy']}) {reason} £{realised_pl:+.2f} ({realised_pct:+.2f}%) after £{entry_fee + exit_fee:.2f} fees")
        #8.0 sweep step 1: if the regime has turned hostile, dump the SPY
        #sweep back to cash before anything else happens this cycle
        if INDEX_SWEEP_ENABLED and _get_sweep_position(portfolio) is not None:
            _regime_now = regime_data.get('regime', 'neutral')
            if _regime_now not in INDEX_SWEEP_REGIMES:
                _sell_sweep(portfolio, None, f'regime_{_regime_now}', activity, now_iso)

        #refresh cash/equity after exits before deciding whether to scan entries
        total_value_gbp = sum(p.get('current_value_gbp', p.get('cost_gbp', 0)) for p in portfolio['positions'])
        total_equity_gbp = portfolio['cash_gbp'] + total_value_gbp

        #8.0: the sweep does not occupy one of the strategy slots
        slots_left = MAX_POSITIONS - len(_non_sweep_positions(portfolio))
        target_slot_gbp = total_equity_gbp / MAX_POSITIONS if MAX_POSITIONS else 0
        cash_gbp = float(portfolio.get('cash_gbp', 0) or 0)

        #8.0 sweep step 2: if a new slot needs funding and the cash is parked
        #in SPY, sell just enough sweep to cover the slot before deciding
        if INDEX_SWEEP_ENABLED and slots_left > 0 and _get_sweep_position(portfolio) is not None:
            required_cash = target_slot_gbp * MIN_ENTRY_SLOT_COVERAGE
            if cash_gbp < required_cash:
                _sell_sweep(portfolio, required_cash - cash_gbp, 'fund_new_position', activity, now_iso)
                cash_gbp = float(portfolio.get('cash_gbp', 0) or 0)

        entry_scan_allowed = slots_left > 0 and cash_gbp >= MIN_TRADE_GBP

        if entry_scan_allowed and not ALLOW_REMAINDER_TRADES:
            required_cash = target_slot_gbp * MIN_ENTRY_SLOT_COVERAGE
            if cash_gbp < required_cash:
                entry_scan_allowed = False
                activity.append(
                    f"exit-only cycle: cash £{cash_gbp:.2f} below required £{required_cash:.2f} "
                    f"for a normal new slot, skipping entry scan"
                )

        if not entry_scan_allowed and slots_left <= 0:
            activity.append("exit-only cycle: max positions reached, skipping entry scan")
        elif not entry_scan_allowed and cash_gbp < MIN_TRADE_GBP:
            activity.append(f"exit-only cycle: cash £{cash_gbp:.2f} below minimum trade size, skipping entry scan")

        #step 3 scan for candidates
        candidates = []
        if entry_scan_allowed:
            held = set(p['ticker'] for p in portfolio['positions'])
            paper_cycle_status['message'] = 'scanning insider clusters...'
            paper_cycle_status['progress'] = 5
            try:
                clusters = _try_with_retry(get_insider_clusters, min_insiders=3, days=60,
                                            min_total_value=500000, senior_required=True) or []
                cluster_hits = 0
                for c in clusters[:25]:
                    if c['ticker'] not in held and c.get('confidence', 0) >= MIN_CONFIDENCE_TO_OPEN:
                        candidates.append({'ticker': c['ticker'], 'confidence': c['confidence'], 'strategy': 'cluster'})
                        cluster_hits += 1
                activity.append(f"cluster scan: {len(clusters)} found, {cluster_hits} qualify (conf >= {MIN_CONFIDENCE_TO_OPEN})")
            except Exception as e:
                activity.append(f"cluster scan failed: {str(e)[:80]}")
            try:
                universe = _try_with_retry(get_stock_universe, scan_count) or []
                if not universe:
                    activity.append('warning: stock universe fetch failed after retries, skipping scan')
                vix = _try_with_retry(get_vix_level)
                if vix is None: vix = 20.0  #safe fallback, used by the regime footer
                activity.append(f"VIX {vix:.1f}")
                #fetch SPY 12-1 momentum once so existing relative momentum does not refetch it per ticker
                try:
                    spy_hist = drop_incomplete_bars(yf.Ticker('SPY').history(period='1y', auto_adjust=True))
                    if len(spy_hist) >= 240:
                        spy_mom = (float(spy_hist['Close'].iloc[-22]) - float(spy_hist['Close'].iloc[0])) / float(spy_hist['Close'].iloc[0])
                    else:
                        spy_mom = 0.0
                except Exception:
                    spy_mom = 0.0
                activity.append(f"SPY 12-1 momentum {spy_mom*100:+.1f}%")
                pead_hits = 0
                factor_hits = {name: 0 for name in FACTOR_STRATEGIES}
                total = len(universe)
                for i, t in enumerate(universe):
                    paper_cycle_status['progress'] = int(10 + (i / max(total, 1)) * 85)
                    paper_cycle_status['message'] = f'scanning {t} ({i+1}/{total})'
                    if t in held: continue

                    #existing event-driven edge: post-earnings announcement drift
                    sig = _try_with_retry(get_pead_signal, t)
                    if sig and sig.get('confidence', 0) >= MIN_CONFIDENCE_TO_OPEN:
                        candidates.append({
                            'ticker': t, 'confidence': sig['confidence'],
                            'strategy': 'pead', 'reason': f"PEAD {sig.get('surprise_pct', '?')}% surprise"
                        })
                        pead_hits += 1

                    #existing Merlin momentum edge, preserved as its own strategy
                    sig = _try_with_retry(get_momentum_signal, t, spy_mom)
                    if sig and sig.get('confidence', 0) >= MIN_CONFIDENCE_TO_OPEN:
                        candidates.append({
                            'ticker': t, 'confidence': sig['confidence'],
                            'strategy': 'momentum',
                            'reason': f"relative 12-1 momentum {sig.get('mom_12_1_pct', '?')}%, vs SPY {sig.get('relative_mom_vs_spy_pct', '?')}%"
                        })
                        factor_hits['momentum'] = factor_hits.get('momentum', 0) + 1

                    #research-backed factor edges: stricter 12-1 momentum, quality, shareholder yield
                    factor_sigs = _try_with_retry(get_research_factor_signals, t, MIN_CONFIDENCE_TO_OPEN) or []
                    for fs in factor_sigs:
                        candidates.append({
                            'ticker': t, 'confidence': fs['confidence'],
                            'strategy': fs['strategy'], 'reason': fs.get('reason', '')
                        })
                        factor_hits[fs['strategy']] = factor_hits.get(fs['strategy'], 0) + 1

                    if i % 10 == 0 and i > 0: time.sleep(0.3)

                factor_msg = ', '.join([f"{k} {v}" for k, v in factor_hits.items()])
                activity.append(f"scanned {len(universe)} tickers: {pead_hits} pead, {factor_msg} qualify")
            except Exception as e:
                activity.append(f"scan loop failed: {str(e)[:80]}")

            #step 4 rank and open
            paper_cycle_status['message'] = 'opening positions...'
            paper_cycle_status['progress'] = 96
            slots = MAX_POSITIONS - len(_non_sweep_positions(portfolio))
            ranked_candidates = _rank_paper_candidates(candidates, slots,
                                                       held_tickers=set(p['ticker'] for p in portfolio['positions']),
                                                       held_positions=portfolio['positions'],
                                                       regime_data=regime_data)
            counts_msg = {}
            for rc in ranked_candidates:
                counts_msg[rc['strategy']] = counts_msg.get(rc['strategy'], 0) + 1
            activity.append('candidate allocation: ' + ', '.join([f'{k} {v}' for k, v in counts_msg.items()]) if counts_msg else 'candidate allocation: none')
            #7.9: surface the hard regime gate so the user can see why
            #momentum signals were ignored in bear or volatile markets
            if ranked_candidates and ranked_candidates[0].get('_regime_blocks'):
                activity.append(f"regime gate: blocked {ranked_candidates[0]['_regime_blocks']} momentum-group signal(s) in {regime_data.get('regime', '?')}")
                ranked_candidates[0].pop('_regime_blocks', None)
            #10.0 earnings blackout, named rather than counted
            if ranked_candidates and ranked_candidates[0].get('_earnings_blocks'):
                held_off = ranked_candidates[0]['_earnings_blocks']
                activity.append(f"earnings blackout: held off {len(held_off)} slow-factor signal(s) "
                                f"within {EARNINGS_BLACKOUT_DAYS} days of a print - {', '.join(held_off[:8])}")
                ranked_candidates[0].pop('_earnings_blocks', None)
            #note: short interest blocks happen inside _rank_paper_candidates,
            #they're not tracked here per-cycle. could add later if useful.
            seen_tickers = set()
            for cand in ranked_candidates:
                if portfolio['cash_gbp'] < MIN_TRADE_GBP:
                    break

                sym = cand['ticker']

                try:
                    entry, currency = _paper_price_and_currency(sym)
                    if not _finite(entry) or entry <= 0:
                        continue

                    fx = _get_fx_to_gbp(currency)
                    #8.0: realised volatility tilts the size (calm = bigger)
                    ann_vol = _recent_ann_vol(sym)
                    size_gbp = _calculate_position_size_gbp(
                        cand['confidence'],
                        total_equity_gbp,
                        portfolio['cash_gbp'],
                        ann_vol=ann_vol
                    )

                    if size_gbp < MIN_TRADE_GBP:
                        continue

                    #8.0: the entry pays its fee up front like a real broker.
                    #size_gbp is the total cash leaving the account, so the
                    #actual invested amount is size / (1 + fee rate)
                    fee_rate = (SPREAD_SLIPPAGE_PCT + (FX_FEE_PCT if currency != 'GBP' else 0.0)) / 100.0
                    cost_gbp = size_gbp / (1.0 + fee_rate)

                    #final safety guard so it can never overspend incl the fee
                    max_spend = portfolio['cash_gbp'] * CASH_BUFFER_PCT
                    if cost_gbp * (1.0 + fee_rate) > max_spend:
                        cost_gbp = max_spend / (1.0 + fee_rate)

                    shares = (cost_gbp / fx) / entry
                    entry_fee = cost_gbp * fee_rate

                    if cost_gbp < MIN_TRADE_GBP:
                        continue
                    portfolio['positions'].append({
                        'ticker': sym, 'strategy': cand['strategy'],
                        'confidence_at_entry': cand['confidence'],
                        'entry_price_native': entry, 'shares': shares,
                        'currency': currency, 'fx_at_entry': fx,
                        'cost_gbp': cost_gbp, 'entry_date': now_iso,
                        'entry_fee_gbp': entry_fee,
                        'entry_reason': cand.get('reason', ''),
                        #peak starts at entry, ratchets up in mark-to-market
                        'peak_price_native': entry,
                        'current_price_native': entry,
                        'current_value_gbp': cost_gbp,
                    })
                    portfolio['cash_gbp'] -= (cost_gbp + entry_fee)
                    portfolio['fees_paid_gbp'] = float(portfolio.get('fees_paid_gbp', 0) or 0) + entry_fee
                    seen_tickers.add(sym)
                    vol_note = f", vol {ann_vol*100:.0f}%" if ann_vol else ''
                    activity.append(f"opened {sym} ({cand['strategy']}) conf {cand['confidence']:.0f} £{cost_gbp:.2f} = {shares:.4f} @ {entry:.2f} {currency} (fee £{entry_fee:.2f}{vol_note})")
                except Exception as e:
                    activity.append(f"failed to open {sym}: {str(e)[:80]}")

        #step 5: sweep idle cash into SPY when the regime allows, then snapshot
        paper_cycle_status['message'] = 'finalising...'
        paper_cycle_status['progress'] = 98
        _sweep_idle_cash(portfolio, regime_data, activity, now_iso)
        final_pos_value = sum(p.get('current_value_gbp', p.get('cost_gbp', 0)) for p in portfolio['positions'])
        final_equity = portfolio['cash_gbp'] + final_pos_value
        if _finite(final_equity):
            portfolio['equity_curve'].append({'date': now_iso, 'equity_gbp': final_equity})
        else:
            activity.append('warning: equity came out non-finite this cycle, snapshot skipped')
        portfolio['cycles_run'] += 1
        portfolio['last_cycle'] = now_iso
        prefix = '[auto] ' if triggered_by == 'scheduler' else ''
        if not activity: activity.append(f'{prefix}no actions, no exits and no qualifying signals')
        portfolio['activity_log'] = (portfolio.get('activity_log', []) +
                                      [{'date': now_iso, 'event': prefix + a} for a in activity])[-200:]
        _save_paper_portfolio(portfolio)
        result = {'success': True, 'equity_gbp': final_equity,
                  'cash_gbp': portfolio['cash_gbp'],
                  'positions_count': len(_non_sweep_positions(portfolio)),
                  'fees_paid_gbp': round(float(portfolio.get('fees_paid_gbp', 0) or 0), 2),
                  'pl_gbp': final_equity - STARTING_CASH_GBP,
                  'pl_pct': (final_equity - STARTING_CASH_GBP) / STARTING_CASH_GBP * 100,
                  'activity': activity, 'cycles_run': portfolio['cycles_run']}
        paper_cycle_status = {'active': False, 'progress': 100, 'message': 'cycle complete', 'triggered_by': triggered_by}
        return result
    except Exception as e:
        paper_cycle_status = {'active': False, 'progress': 0, 'message': f'failed: {str(e)[:80]}', 'triggered_by': triggered_by}
        raise


def _scheduled_paper_cycle():
    '''called by apscheduler at 22:00 daily, respects the toggle'''
    portfolio = _load_paper_portfolio()
    if not portfolio.get('auto_run_enabled', False):
        print(f"\n[scheduler] daily cycle fired but auto-run is disabled, skipping")
        return
    print(f"\n[scheduler] starting daily paper cycle at {datetime.datetime.now()}")
    try:
        result = _execute_paper_cycle(scan_count=550, triggered_by='scheduler')
        print(f"[scheduler] cycle complete, equity £{result['equity_gbp']:.2f}, {result['positions_count']} positions")
    except Exception as e:
        print(f"[scheduler] cycle failed: {e}")


@app.route('/api/paper/state', methods=['GET'])
def paper_state():
    '''
    7.9: this route is what feeds the dashboard. previously when yfinance
    rate-limited or failed for all positions, this returned equity=cash and
    the UI showed a fake catastrophic loss. now we fall back to last known
    or entry price so the equity stays honest even when the live data is down.
    '''
    portfolio = _load_paper_portfolio()
    total_value_gbp = 0.0
    stale_count = 0
    for pos in portfolio['positions']:
        try:
            price, is_stale, fx = _mark_position_to_market(pos)
            pos['current_price_native'] = price
            pos['price_is_stale'] = is_stale
            if is_stale: stale_count += 1
            pos['current_value_gbp'] = pos['shares'] * price * fx
            #8.0: unrealised P/L nets out the entry fee already paid
            pos['unrealised_pl_gbp'] = pos['current_value_gbp'] - pos['cost_gbp'] - float(pos.get('entry_fee_gbp', 0) or 0)
            pos['unrealised_pl_pct'] = (price - pos['entry_price_native']) / pos['entry_price_native'] * 100
            total_value_gbp += pos['current_value_gbp']
        except Exception: continue
    equity_gbp = portfolio['cash_gbp'] + total_value_gbp
    return jsonify(sanitise({**portfolio, 'equity_gbp': equity_gbp,
                              'position_value_gbp': total_value_gbp,
                              'stale_price_count': stale_count,
                              'starting_cash_gbp': STARTING_CASH_GBP,
                              'fees_paid_gbp': float(portfolio.get('fees_paid_gbp', 0) or 0)}))


@app.route('/api/paper/reset', methods=['POST'])
def paper_reset():
    _save_paper_portfolio(_init_paper_portfolio())
    return jsonify({'success': True})


@app.route('/api/paper/cycle_progress', methods=['GET'])
def paper_cycle_progress():
    return jsonify(paper_cycle_status)


@app.route('/api/paper/toggle_auto', methods=['POST'])
def paper_toggle_auto():
    portfolio = _load_paper_portfolio()
    portfolio['auto_run_enabled'] = not portfolio.get('auto_run_enabled', False)
    _save_paper_portfolio(portfolio)
    return jsonify({'enabled': portfolio['auto_run_enabled']})


def _execute_paper_cycle_safe(scan_count, triggered_by):
    '''wrapper that runs the cycle in a background thread and catches any unhandled errors'''
    global paper_cycle_status
    try:
        _execute_paper_cycle(scan_count=scan_count, triggered_by=triggered_by)
    except Exception as e:
        import traceback; traceback.print_exc()
        paper_cycle_status = {'active': False, 'progress': 0,
                              'message': f'failed: {str(e)[:100]}', 'triggered_by': triggered_by}


@app.route('/api/paper/run_cycle', methods=['POST'])
def paper_run_cycle():
    '''fire and forget, returns immediately. Frontend polls cycle_progress for status'''
    if paper_cycle_status['active']:
        return jsonify({'error': 'a cycle is already running'}), 400
    scan_count = (request.json or {}).get('scan_count', 550)
    #flip active immediately so duplicate clicks bounce before the thread even starts
    paper_cycle_status['active'] = True
    paper_cycle_status['progress'] = 0
    paper_cycle_status['message'] = 'starting cycle...'
    threading.Thread(target=_execute_paper_cycle_safe,
                     args=(scan_count, 'manual'), daemon=True).start()
    return jsonify({'success': True, 'status': 'started'})


#schedule daily auto-run at 22:00 local time
_paper_scheduler = BackgroundScheduler(daemon=True)
_paper_scheduler.add_job(_scheduled_paper_cycle, 'cron', hour=22, minute=0,
                          id='daily_paper_cycle', replace_existing=True)
#8.2 fix: this used to be gated behind WERKZEUG_RUN_MAIN == 'true', which
#werkzeug only sets inside the reloader's child process. merlin runs with
#use_reloader=False (see the __main__ docstring for why), so that env var
#never exists and the scheduler never started - the 22:00 cycle has been
#dead the whole time while the UI said ENABLED. with the reloader off there
#is exactly one process, so just start it.
if not _paper_scheduler.running:
    _paper_scheduler.start()
    atexit.register(lambda: _paper_scheduler.shutdown(wait=False))
    print('\n[scheduler] paper trader scheduler started, daily cycle at 22:00 local time')


#==================== MARKET REGIME DETECTION ====================
#
#a simple regime classifier that looks at three signals:
#  1) SPY trend: where is the S&P 500 relative to its 200 day moving average,
#     and is the 200dma sloping up or down? this is the single biggest signal
#     for "are we in a bull or bear market".
#  2) VIX level: how stressed are options markets right now. high vix means
#     fear, low vix means complacency. mean reversion and short-term tactical
#     setups behave very differently across vix regimes.
#  3) sector breadth: what percentage of the major SPDR sector ETFs are
#     trading above their 50 day moving average. high breadth = bull market
#     participation is broad. low breadth = narrow leadership, fragile.
#
#regime cache is refreshed every 15 minutes so we don't hammer yfinance.

SECTOR_ETFS = ['XLK','XLV','XLF','XLY','XLP','XLE','XLI','XLB','XLU','XLRE','XLC']

_regime_cache = {'data': None, 'fetched_at': None}

def get_market_regime(force_refresh=False):
    '''
    returns the current market regime as a dict. cached for 15 minutes.
    safe to call from any route, never raises on data failures.
    '''
    global _regime_cache
    now = datetime.datetime.now()
    if not force_refresh and _regime_cache['fetched_at'] is not None:
        age = (now - _regime_cache['fetched_at']).total_seconds()
        if age < 900 and _regime_cache['data'] is not None:
            return _regime_cache['data']

    try:
        spy = drop_incomplete_bars(yf.Ticker('SPY').history(period='1y', auto_adjust=True))
        if spy.empty or len(spy) < 200:
            #not enough data, return neutral regime
            data = _neutral_regime_payload('insufficient SPY history')
            _regime_cache = {'data': data, 'fetched_at': now}
            return data

        close = spy['Close']
        spy_price = float(close.iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])
        sma200_slope_pct = float((close.rolling(200).mean().iloc[-1] - close.rolling(200).mean().iloc[-22]) / close.rolling(200).mean().iloc[-22] * 100)
        spy_above_200 = spy_price > sma200
        spy_pct_vs_200 = float((spy_price - sma200) / sma200 * 100)
        spy_1m_return = float((close.iloc[-1] - close.iloc[-22]) / close.iloc[-22] * 100) if len(close) > 22 else 0.0

        #vix
        vix = get_vix_level()
        if vix is None: vix = 20.0

        #sector breadth
        sectors_above_50 = 0
        sectors_checked = 0
        for etf in SECTOR_ETFS:
            try:
                h = drop_incomplete_bars(yf.Ticker(etf).history(period='4mo', auto_adjust=True))
                if h.empty or len(h) < 60: continue
                hc = h['Close']
                sma50 = float(hc.rolling(50).mean().iloc[-1])
                if float(hc.iloc[-1]) > sma50: sectors_above_50 += 1
                sectors_checked += 1
            except Exception: continue
        breadth_pct = round(sectors_above_50 / sectors_checked * 100, 1) if sectors_checked > 0 else 50.0

        #regime classification
        regime, label, explainer = _classify_regime(spy_above_200, sma200_slope_pct, vix, breadth_pct, spy_pct_vs_200)
        favoured, suppressed = _regime_strategy_biases(regime)

        data = {
            'regime': regime,
            'regime_label': label,
            'explainer': explainer,
            'spy_price': round(spy_price, 2),
            'spy_sma200': round(sma200, 2),
            'spy_pct_vs_200dma': round(spy_pct_vs_200, 2),
            'spy_above_200dma': bool(spy_above_200),
            'spy_sma200_slope_22d_pct': round(sma200_slope_pct, 2),
            'spy_1m_return_pct': round(spy_1m_return, 2),
            'vix_level': round(float(vix), 2),
            'sector_breadth_pct': breadth_pct,
            'sectors_above_50dma': sectors_above_50,
            'sectors_checked': sectors_checked,
            'favoured_strategies': favoured,
            'suppressed_strategies': suppressed,
            'fetched_at': now.isoformat(),
        }
        _regime_cache = {'data': data, 'fetched_at': now}
        return data
    except Exception as e:
        data = _neutral_regime_payload(f'regime fetch failed: {str(e)[:80]}')
        _regime_cache = {'data': data, 'fetched_at': now}
        return data


def _neutral_regime_payload(reason):
    return {
        'regime': 'neutral', 'regime_label': 'neutral / unknown',
        'explainer': reason,
        'spy_price': None, 'spy_sma200': None, 'spy_pct_vs_200dma': None,
        'spy_above_200dma': None, 'spy_sma200_slope_22d_pct': None,
        'spy_1m_return_pct': None, 'vix_level': None,
        'sector_breadth_pct': None, 'sectors_above_50dma': None, 'sectors_checked': 0,
        'favoured_strategies': [], 'suppressed_strategies': [],
        'fetched_at': datetime.datetime.now().isoformat(),
    }


def _classify_regime(above_200, slope_pct, vix, breadth, pct_vs_200):
    '''returns (regime_code, human_label, plain_english_explainer)'''
    if not above_200 and vix >= 25:
        return ('bear_volatile',
                'bear market / volatile',
                f'SPY is {abs(pct_vs_200):.1f}% below its 200-day average and VIX is elevated at {vix:.1f}. defensive strategies (quality, low-beta) are favoured. momentum and mean reversion are unreliable in this regime.')
    if not above_200:
        return ('bear_calm',
                'weak market / below trend',
                f'SPY is {abs(pct_vs_200):.1f}% below its 200-day average. trend has rolled over. quality and shareholder yield work better than momentum here.')
    if vix >= 25:
        return ('bull_volatile',
                'bull market / turbulent',
                f'SPY is in uptrend but VIX at {vix:.1f} signals stress. momentum suppressed, defensive factors favoured. avoid fresh mean reversion until volatility settles.')
    if vix < 16 and breadth >= 60 and slope_pct > 0.5:
        return ('strong_bull',
                'strong bull / broad participation',
                f'SPY above 200dma with positive slope, VIX calm at {vix:.1f}, {breadth:.0f}% of sectors above 50dma. classic momentum environment. mean reversion gate is open.')
    if vix < 18 and abs(pct_vs_200) < 4 and abs(slope_pct) < 1:
        return ('calm_chop',
                'calm sideways / range-bound',
                f'SPY hovering near its 200dma with low slope. VIX is calm at {vix:.1f}. mean reversion is the sweet spot here. momentum tends to chop.')
    if above_200 and breadth < 40:
        return ('narrow_bull',
                'narrow bull / weak breadth',
                f'SPY above 200dma but only {breadth:.0f}% of sectors above 50dma. leadership is narrow. be selective on momentum, lean to quality.')
    return ('bull_normal',
            'normal bull / mixed',
            f'SPY above 200dma, VIX at {vix:.1f}, breadth {breadth:.0f}%. balanced regime, no single strategy is dominant.')


def _regime_strategy_biases(regime):
    '''
    returns (favoured_strategies_list, suppressed_strategies_list).
    7.8: meanrev, week52_high and low_beta_trend retired, so they're stripped
    out of the bias maps. bear regimes now have a thinner playbook (just
    quality and shareholder_yield as defensive favourites).
    '''
    biases = {
        'strong_bull':    (['momentum', 'momentum_12_1', 'cluster'], []),
        'bull_normal':    (['momentum_12_1', 'cluster', 'pead'], []),
        'narrow_bull':    (['quality', 'momentum_12_1'], []),
        'calm_chop':      (['shareholder_yield', 'quality'], ['momentum']),
        'bull_volatile':  (['quality', 'shareholder_yield'], ['momentum']),
        'bear_calm':      (['quality', 'shareholder_yield'], ['momentum', 'momentum_12_1']),
        'bear_volatile':  (['quality'], ['momentum', 'momentum_12_1', 'cluster']),
        'neutral':        ([], []),
    }
    return biases.get(regime, ([], []))


@app.route('/api/market_regime', methods=['GET'])
def market_regime_route():
    '''returns the current market regime payload'''
    force = request.args.get('refresh') == '1'
    return jsonify(sanitise(get_market_regime(force_refresh=force)))


#regime-aware score bonus applied inside _rank_paper_candidates.
#favoured strategies get +5, suppressed get -8. this shifts the effective
#score ranking without disabling any strategy outright.
def _regime_score_adjustment(strategy_name, regime_data=None):
    if not regime_data: regime_data = get_market_regime()
    if strategy_name in regime_data.get('favoured_strategies', []):
        return 5.0
    if strategy_name in regime_data.get('suppressed_strategies', []):
        return -8.0
    return 0.0


#==================== MANUAL SCANNER ROUTES ====================
#
#the AI paper trader runs the 9 strategies automatically. but for the manual
#tabs we group them into two scanners: momentum (3 strategies) and factor
#(3 strategies). both share the same async pattern as the existing scanners.

momentum_scan_status = {'active': False, 'progress': 0, 'message': '',
                        'complete': False, 'results': None, 'error': None}
factor_scan_status =   {'active': False, 'progress': 0, 'message': '',
                        'complete': False, 'results': None, 'error': None}


def _run_momentum_scan(scan_count, min_conf):
    global momentum_scan_status
    try:
        universe = _try_with_retry(get_stock_universe, scan_count) or []
        if not universe:
            momentum_scan_status.update({'error': 'could not fetch universe', 'active': False, 'complete': True})
            return
        #spy momentum once
        try:
            spy_hist = drop_incomplete_bars(yf.Ticker('SPY').history(period='1y', auto_adjust=True))
            spy_mom = (float(spy_hist['Close'].iloc[-22]) - float(spy_hist['Close'].iloc[0])) / float(spy_hist['Close'].iloc[0]) if len(spy_hist) >= 240 else 0.0
        except Exception:
            spy_mom = 0.0

        signals = []
        total = len(universe)
        for i, t in enumerate(universe):
            momentum_scan_status['progress'] = int((i / max(total, 1)) * 100)
            momentum_scan_status['message'] = f'scanning {t} ({i+1}/{total})'
            try:
                rel = _try_with_retry(get_momentum_signal, t, spy_mom)
                if rel and rel.get('confidence', 0) >= min_conf:
                    rel['strategy'] = 'momentum'; signals.append(rel)
                #8.1: momentum_12_1 retired, the scan no longer collects it
            except Exception: pass
            if i % 10 == 0 and i > 0: time.sleep(0.3)

        signals.sort(key=lambda s: s.get('confidence', 0), reverse=True)
        momentum_scan_status.update({
            'progress': 100, 'complete': True, 'active': False,
            'message': f'done - {len(signals)} signals',
            'results': {'signals': signals, 'min_conf': min_conf, 'universe_size': total}
        })
    except Exception as e:
        momentum_scan_status.update({'error': str(e)[:200], 'active': False, 'complete': True})


def _run_factor_scan(scan_count, min_conf):
    global factor_scan_status
    try:
        universe = _try_with_retry(get_stock_universe, scan_count) or []
        if not universe:
            factor_scan_status.update({'error': 'could not fetch universe', 'active': False, 'complete': True})
            return

        signals = []
        total = len(universe)
        for i, t in enumerate(universe):
            factor_scan_status['progress'] = int((i / max(total, 1)) * 100)
            factor_scan_status['message'] = f'scanning {t} ({i+1}/{total})'
            try:
                factor_sigs = _try_with_retry(get_research_factor_signals, t, min_conf) or []
                for fs in factor_sigs:
                    if fs.get('strategy') in ('quality', 'shareholder_yield'):
                        signals.append(fs)
            except Exception: pass
            if i % 10 == 0 and i > 0: time.sleep(0.3)

        signals.sort(key=lambda s: s.get('confidence', 0), reverse=True)
        factor_scan_status.update({
            'progress': 100, 'complete': True, 'active': False,
            'message': f'done - {len(signals)} signals',
            'results': {'signals': signals, 'min_conf': min_conf, 'universe_size': total}
        })
    except Exception as e:
        factor_scan_status.update({'error': str(e)[:200], 'active': False, 'complete': True})


quality_scan_status = {'active': False, 'progress': 0, 'message': '',
                       'complete': False, 'results': None, 'error': None}

def _run_quality_scan(scan_count, min_conf):
    '''8.3: quality gets its own tab. same engine as the factor scan but
    filtered to quality alone - +72% over 5y with a 2.5% chance of loss,
    second only to shareholder yield, and it was buried in a mixed list'''
    global quality_scan_status
    try:
        universe = _try_with_retry(get_stock_universe, scan_count) or []
        if not universe:
            quality_scan_status.update({'error': 'could not fetch universe', 'active': False, 'complete': True})
            return

        signals = []
        total = len(universe)
        for i, t in enumerate(universe):
            quality_scan_status['progress'] = int((i / max(total, 1)) * 100)
            quality_scan_status['message'] = f'scanning {t} ({i+1}/{total})'
            try:
                factor_sigs = _try_with_retry(get_research_factor_signals, t, min_conf) or []
                for fs in factor_sigs:
                    if fs.get('strategy') == 'quality':
                        signals.append(fs)
            except Exception: pass
            if i % 10 == 0 and i > 0: time.sleep(0.3)

        signals.sort(key=lambda s: s.get('confidence', 0), reverse=True)
        quality_scan_status.update({
            'progress': 100, 'complete': True, 'active': False,
            'message': f'done - {len(signals)} signals',
            'results': {'signals': signals, 'min_conf': min_conf, 'universe_size': total}
        })
    except Exception as e:
        quality_scan_status.update({'error': str(e)[:200], 'active': False, 'complete': True})


@app.route('/api/quality_scan', methods=['POST'])
def quality_scan_route():
    global quality_scan_status
    if quality_scan_status.get('active'): return jsonify({'error': 'already running'}), 400
    body = request.json or {}
    scan_count = int(body.get('count', 300))
    min_conf = int(body.get('min_conf', 65))
    quality_scan_status = {'active': True, 'progress': 0, 'message': 'starting...',
                           'complete': False, 'results': None, 'error': None}
    threading.Thread(target=_run_quality_scan, args=(scan_count, min_conf), daemon=True).start()
    return jsonify({'success': True})


@app.route('/api/quality_scan_status')
def quality_scan_status_route():
    return jsonify(sanitise(quality_scan_status))

shareholder_scan_status = {'active': False, 'progress': 0, 'message': '',
                           'complete': False, 'results': None, 'error': None}

def _run_shareholder_scan(scan_count, min_conf):
    '''8.1: dedicated shareholder yield scan for the new tab. same engine as
    the factor scan but filtered to the strategy that earned it - +129% over
    5y with a 0% chance of loss across 1000 monte carlo reshuffles'''
    global shareholder_scan_status
    try:
        universe = _try_with_retry(get_stock_universe, scan_count) or []
        if not universe:
            shareholder_scan_status.update({'error': 'could not fetch universe', 'active': False, 'complete': True})
            return

        signals = []
        total = len(universe)
        for i, t in enumerate(universe):
            shareholder_scan_status['progress'] = int((i / max(total, 1)) * 100)
            shareholder_scan_status['message'] = f'scanning {t} ({i+1}/{total})'
            try:
                factor_sigs = _try_with_retry(get_research_factor_signals, t, min_conf) or []
                for fs in factor_sigs:
                    if fs.get('strategy') == 'shareholder_yield':
                        signals.append(fs)
            except Exception: pass
            if i % 10 == 0 and i > 0: time.sleep(0.3)

        signals.sort(key=lambda s: s.get('confidence', 0), reverse=True)
        shareholder_scan_status.update({
            'progress': 100, 'complete': True, 'active': False,
            'message': f'done - {len(signals)} signals',
            'results': {'signals': signals, 'min_conf': min_conf, 'universe_size': total}
        })
    except Exception as e:
        shareholder_scan_status.update({'error': str(e)[:200], 'active': False, 'complete': True})


@app.route('/api/shareholder_scan', methods=['POST'])
def shareholder_scan_route():
    global shareholder_scan_status
    if shareholder_scan_status.get('active'): return jsonify({'error': 'already running'}), 400
    body = request.json or {}
    scan_count = int(body.get('count', 300))
    min_conf = int(body.get('min_conf', 65))
    shareholder_scan_status = {'active': True, 'progress': 0, 'message': 'starting...',
                               'complete': False, 'results': None, 'error': None}
    threading.Thread(target=_run_shareholder_scan, args=(scan_count, min_conf), daemon=True).start()
    return jsonify({'success': True})


@app.route('/api/shareholder_scan_status')
def shareholder_scan_status_route():
    return jsonify(sanitise(shareholder_scan_status))


#8.1: vs SPY benchmark for the new tab. normalises the paper equity curve
#and SPY to 100 at the paper book's first date so growth is comparable.
#note SPY is in USD terms while the book is GBP - cable moves are part of
#the bot's number (it pays fx both ways) but not the index line.
@app.route('/api/benchmark/vs_spy')
def benchmark_vs_spy_route():
    portfolio = _load_paper_portfolio()
    curve = portfolio.get('equity_curve', [])
    if len(curve) < 2:
        return jsonify({'error': 'not enough paper history yet - run a few cycles and come back'})
    start_iso = str(curve[0]['date'])[:10]
    try:
        spy = drop_incomplete_bars(yf.Ticker('SPY').history(start=start_iso, auto_adjust=True))
    except Exception:
        spy = pd.DataFrame()
    if spy is None or spy.empty:
        return jsonify({'error': 'could not fetch SPY from yfinance right now, try again in a minute'})
    sc = spy['Close'].dropna()
    if len(sc) < 1:
        return jsonify({'error': 'SPY data came back empty, try again in a minute'})
    spy0 = float(sc.iloc[0])
    bot0 = float(curve[0].get('equity_gbp') or STARTING_CASH_GBP)
    if spy0 <= 0 or bot0 <= 0:
        return jsonify({'error': 'bad starting values, cannot normalise'})
    spy_series = [{'date': d.strftime('%Y-%m-%d'), 'value': round(float(v) / spy0 * 100, 3)} for d, v in sc.items()]
    bot_series = [{'date': str(p['date'])[:10], 'value': round(float(p.get('equity_gbp') or bot0) / bot0 * 100, 3)} for p in curve]
    bot_ret = (float(curve[-1].get('equity_gbp') or bot0) / bot0 - 1) * 100
    spy_ret = (float(sc.iloc[-1]) / spy0 - 1) * 100
    return jsonify(sanitise({
        'bot': bot_series, 'spy': spy_series,
        'bot_return_pct': round(bot_ret, 2), 'spy_return_pct': round(spy_ret, 2),
        'alpha_pct': round(bot_ret - spy_ret, 2),
        'start_date': start_iso, 'starting_cash_gbp': STARTING_CASH_GBP,
    }))


@app.route('/api/momentum_scan', methods=['POST'])
def momentum_scan_route():
    global momentum_scan_status
    if momentum_scan_status.get('active'): return jsonify({'error': 'already running'}), 400
    body = request.json or {}
    scan_count = int(body.get('count', 300))
    min_conf = int(body.get('min_conf', 65))
    momentum_scan_status = {'active': True, 'progress': 0, 'message': 'starting...',
                            'complete': False, 'results': None, 'error': None}
    threading.Thread(target=_run_momentum_scan, args=(scan_count, min_conf), daemon=True).start()
    return jsonify({'success': True, 'started': True})


@app.route('/api/momentum_scan_status')
def momentum_scan_status_route():
    return jsonify(sanitise(momentum_scan_status))


@app.route('/api/factor_scan', methods=['POST'])
def factor_scan_route():
    global factor_scan_status
    if factor_scan_status.get('active'): return jsonify({'error': 'already running'}), 400
    body = request.json or {}
    scan_count = int(body.get('count', 300))
    min_conf = int(body.get('min_conf', 65))
    factor_scan_status = {'active': True, 'progress': 0, 'message': 'starting...',
                          'complete': False, 'results': None, 'error': None}
    threading.Thread(target=_run_factor_scan, args=(scan_count, min_conf), daemon=True).start()
    return jsonify({'success': True, 'started': True})


@app.route('/api/factor_scan_status')
def factor_scan_status_route():
    return jsonify(sanitise(factor_scan_status))

#==================== CORTEX (8.3) ====================
#
#gap-and-fade. the engine lives in cortex.py, this is just the flask surface.
#two runs a day per market: the morning run measures the gap and builds a
#watchlist, the afternoon run measures the wing and opens the trades. exits
#are on a calendar rather than a stop, so the book is maintained on every
#call and self-heals if merlin was not running for a few days.

import cortex

#let cortex reuse merlin's existing US universe rather than keeping its own
cortex.set_us_universe_provider(get_stock_universe)

#pick the IG host and report what state IG is in. demo unless IG_ENV=live,
#because a wrong epic on a live account is a real order
IG_ENV = ig_config.configure_cortex(cortex)

cortex_status = {'active': False, 'progress': 0, 'message': '', 'run': None,
                 'complete': False, 'results': None, 'error': None}


def _cortex_morning(market, gap_min, gap_max, count, prefer_ig=True):
    global cortex_status
    try:
        res = cortex.scan_gaps(market, gap_min=gap_min, gap_max=gap_max,
                               count=count, prefer_ig=prefer_ig,
                               status=cortex_status)
        cortex_status.update({'active': False, 'complete': True, 'progress': 100,
                              'results': res,
                              'message': f"{len(res['gappers'])} gappers from {res['scanned']} scanned"})
    except Exception as e:
        cortex_status.update({'active': False, 'complete': True,
                              'error': f'{type(e).__name__}: {str(e)[:200]}'})


def _cortex_afternoon(market, wing, exit_day, use_buckets, prefer_ig, auto_trade):
    global cortex_status
    try:
        res = cortex.check_wings(market, wing=wing, exit_day=exit_day,
                                 use_buckets=use_buckets, prefer_ig=prefer_ig,
                                 status=cortex_status)
        #always maintain first so anything due is closed at the right historical
        #close before new trades are layered on top
        book, closed = cortex.maintain_paper()
        opened = []
        if auto_trade and res.get('confirmed'):
            book, opened = cortex.open_paper_positions(res['confirmed'], book)
            cortex.maintain_paper(book)
        res['paper_closed'] = len(closed)
        res['paper_opened'] = len(opened)
        cortex_status.update({'active': False, 'complete': True, 'progress': 100,
                              'results': res,
                              'message': (f"{len(res.get('confirmed', []))} confirmed, "
                                          f"{len(res.get('borderline', []))} borderline, "
                                          f"{len(opened)} opened, {len(closed)} closed")})
    except Exception as e:
        cortex_status.update({'active': False, 'complete': True,
                              'error': f'{type(e).__name__}: {str(e)[:200]}'})


@app.route('/api/cortex/morning', methods=['POST'])
def cortex_morning_route():
    global cortex_status
    if cortex_status.get('active'):
        return jsonify({'error': 'a cortex run is already going'}), 400
    b = request.json or {}
    market = (b.get('market') or 'LSE').upper()
    if market not in cortex.MARKETS:
        return jsonify({'error': f'unknown market {market}'}), 400
    cortex_status = {'active': True, 'progress': 0, 'message': 'starting morning scan...',
                     'run': 'morning', 'market': market, 'complete': False,
                     'results': None, 'error': None}
    threading.Thread(target=_cortex_morning,
                     args=(market, float(b.get('gap_min', cortex.DEFAULT_GAP_MIN)),
                           float(b.get('gap_max', cortex.DEFAULT_GAP_MAX)),
                           int(b.get('count', cortex.UNIVERSE_ALL)),
                           bool(b.get('prefer_ig', True))), daemon=True).start()
    return jsonify({'success': True, 'started': True})


@app.route('/api/cortex/afternoon', methods=['POST'])
def cortex_afternoon_route():
    global cortex_status
    if cortex_status.get('active'):
        return jsonify({'error': 'a cortex run is already going'}), 400
    b = request.json or {}
    market = (b.get('market') or 'LSE').upper()
    if market not in cortex.MARKETS:
        return jsonify({'error': f'unknown market {market}'}), 400
    cortex_status = {'active': True, 'progress': 0, 'message': 'starting wing check...',
                     'run': 'afternoon', 'market': market, 'complete': False,
                     'results': None, 'error': None}
    threading.Thread(target=_cortex_afternoon,
                     args=(market, float(b.get('wing', cortex.DEFAULT_WING)),
                           int(b.get('exit_day', cortex.DEFAULT_EXIT_DAY)),
                           bool(b.get('use_buckets', False)),
                           bool(b.get('prefer_ig', True)),
                           bool(b.get('auto_trade', True))), daemon=True).start()
    return jsonify({'success': True, 'started': True})


@app.route('/api/cortex/status')
def cortex_status_route():
    return jsonify(sanitise(cortex_status))


@app.route('/api/cortex/watchlist')
def cortex_watchlist_route():
    market = (request.args.get('market') or 'LSE').upper()
    return jsonify(sanitise({'market': market,
                             'watchlist': cortex.load_watchlist(market),
                             'ig_available': cortex.ig_available()}))


@app.route('/api/cortex/paper')
def cortex_paper_route():
    #a plain GET refreshes marks and closes anything due, so the tab is
    #always truthful even if the afternoon button has not been pressed
    try:
        book, _ = cortex.maintain_paper()
    except Exception:
        book = cortex.load_paper()
    return jsonify(sanitise(cortex.paper_summary(book)))


@app.route('/api/cortex/paper/reset', methods=['POST'])
def cortex_paper_reset_route():
    return jsonify(sanitise(cortex.paper_summary(cortex.reset_paper())))


def _cortex_sweep(market, years, count, allow_short):
    try:
        res = cortex.run_sweep(market, years=years, count=count,
                               allow_short=allow_short, status=cortex.sweep_status)
        cortex.sweep_status.update({'active': False, 'complete': True, 'progress': 100,
                                    'results': res, 'message': 'done'})
    except Exception as e:
        cortex.sweep_status.update({'active': False, 'complete': True,
                                    'error': f'{type(e).__name__}: {str(e)[:200]}'})


@app.route('/api/cortex/sweep', methods=['POST'])
def cortex_sweep_route():
    if cortex.sweep_status.get('active'):
        return jsonify({'error': 'sweep already running'}), 400
    b = request.json or {}
    market = (b.get('market') or 'LSE').upper()
    cortex.sweep_status.update({'active': True, 'progress': 0, 'complete': False,
                                'results': None, 'error': None,
                                'message': 'starting sweep...'})
    threading.Thread(target=_cortex_sweep,
                     args=(market, float(b.get('years', 4)), int(b.get('count', cortex.UNIVERSE_ALL)),
                           bool(b.get('allow_short', True))), daemon=True).start()
    return jsonify({'success': True, 'started': True})


@app.route('/api/cortex/sweep_status')
def cortex_sweep_status_route():
    return jsonify(sanitise(cortex.sweep_status))


@app.route('/api/cortex/ig_status')
def cortex_ig_status_route():
    '''
    whether the cortex tab can run on live IG prices or has to fall back to
    yahoo. drives the source badge, so it must never raise - a missing epic
    file or absent credentials are normal states, not errors.
    '''
    return jsonify(sanitise(cortex.ig_status()))


@app.route('/api/market_clock')
def market_clock_route():
    '''header clock strip. pure date arithmetic, no network, so it is cheap
    enough for the browser to refetch every few minutes'''
    return jsonify(sanitise(cortex.market_clock()))


@app.route('/api/earnings/radar')
def earnings_radar_route():
    '''
    10.0: every open position in either book, sorted by how close its next
    result is.

    the point of this one is the command deck. holding a name through a print
    is a decision, and it should be a decision rather than something you find
    out about on the day. the blackout stops the trader OPENING into a print;
    nothing stops a position it opened six weeks ago from walking into one,
    and this is the panel that tells you.

    only open positions are looked up - typically under a dozen tickers, all
    cached - so this stays cheap enough for the deck to call on every load.
    '''
    rows, seen = [], set()

    def add(ticker, book, extra=None):
        t = (ticker or '').upper()
        if not t or t in seen:
            return
        seen.add(t)
        try:
            row = earnings.summary(t)
        except Exception:
            row = {'ticker': t, 'days_to_next': None, 'text': 'earnings date unknown'}
        row['book'] = book
        row.update(extra or {})
        rows.append(row)

    #each row is guarded on its own. wrapping a whole loop in one try means a
    #single malformed position silently drops every position after it, which
    #is the kind of bug that only shows up as "why is BP not on the radar"
    def each(rows, fn):
        for row in (rows or []):
            try:
                fn(row)
            except Exception:
                continue

    try:
        pf = _load_paper_portfolio()
    except Exception:
        pf = {}
    each(pf.get('positions'),
         lambda pos: None if pos.get('strategy') == 'index_sweep' else
         add(pos.get('ticker'), 'ai trader', {'strategy': pos.get('strategy'),
                                              'pl_pct': pos.get('unrealised_pl_pct')}))
    try:
        cx_positions = cortex.load_paper().get('positions')
    except Exception:
        cx_positions = []
    #the cortex book already stores the yahoo symbol beside the plain ticker,
    #so use it rather than rebuilding it and risking to_yf on a missing name
    each(cx_positions,
         lambda pos: add(pos.get('symbol') or (cortex.to_yf(pos['ticker'], pos.get('market') or 'LSE')
                                               if pos.get('ticker') else None),
                         'cortex', {'strategy': 'cortex ' + str(pos.get('side', '')).lower()}))
    try:
        log = load_trade_log()
    except Exception:
        log = []
    each(log,
         lambda tr: add(tr.get('ticker'), 'trade log', {'strategy': tr.get('strategy')})
         if tr.get('status') == 'open' and not tr.get('exit_date') else None)

    #unknown dates sort last rather than first, which is what a radar is for
    rows.sort(key=lambda r: (r.get('days_to_next') is None, r.get('days_to_next') or 0))
    inside = [r for r in rows if r.get('days_to_next') is not None
              and 0 <= r['days_to_next'] <= EARNINGS_BLACKOUT_DAYS]
    return jsonify(sanitise({'rows': rows, 'count': len(rows),
                             'inside_blackout': len(inside),
                             'blackout_days': EARNINGS_BLACKOUT_DAYS}))


#registered after the radar route on purpose. werkzeug does rank a static
#rule above a dynamic one, but a path converter swallowing /api/earnings/radar
#would be a silent, confusing failure, so the guard makes it explicit rather
#than relying on the routing table to sort it out
@app.route('/api/earnings/<path:ticker>')
def earnings_route(ticker):
    '''one ticker's earnings diary. cached twelve hours, see earnings.py'''
    if ticker.strip('/').lower() == 'radar':
        return earnings_radar_route()
    try:
        return jsonify(sanitise(earnings.summary(ticker)))
    except Exception as e:
        return jsonify({'error': str(e)[:200]}), 500


@app.route('/api/signal_tape')
def signal_tape_route():
    '''
    the header ticker tape. reads local state files only, never the network,
    so it can never slow a page load down. shows the most recent activity
    across cortex, the ai paper book and the manual trade log.

    every block below swallows its exception on purpose: a missing or
    half-written state file should thin the tape out, never break the page.
    '''
    items = []
    try:
        for sig in cortex.recent_signals(20):
            items.append({'source': 'cortex', 'ticker': sig.get('ticker', ''),
                          'label': sig.get('side', ''),
                          'dir': 'up' if sig.get('side') == 'LONG' else 'down',
                          'when': sig.get('bar_date', '')})
    except Exception:
        pass
    try:
        pf = _load_paper_portfolio()
        for pos in (pf.get('positions') or [])[:12]:
            pct = pos.get('unrealised_pl_pct') or 0
            items.append({'source': pos.get('strategy', 'ai'), 'ticker': pos.get('ticker', ''),
                          'label': f'{pct:+.1f}%', 'dir': 'up' if pct >= 0 else 'down',
                          'when': str(pos.get('entry_date') or '')[:10]})
    except Exception:
        pass
    try:
        for t in (load_trade_log() or [])[-10:]:
            #the log stores realised_pl_pct, and it is only filled in once a
            #trade is closed. an open trade has no P/L yet, so colour it flat
            pct = t.get('realised_pl_pct')
            items.append({'source': t.get('strategy', 'manual'), 'ticker': t.get('ticker', ''),
                          'label': t.get('status', 'open'),
                          'dir': 'up' if (pct or 0) >= 0 else 'down',
                          'when': t.get('entry_date', '')})
    except Exception:
        pass
    return jsonify(sanitise({'items': items[:40]}))


#==================== PDF REPORT (9.1) ====================
#
#one ticker, every module merlin has, one downloadable pdf. report.py owns
#the adapters, the scoring and the layout; everything below is the flask
#surface and nothing more.
#
#the run is a background job with real progress rather than a spinner. a
#dozen modules against yfinance takes the better part of a minute on a cold
#cache, and a page that sits frozen for that long reads as broken.
#
#report.py is handed this module rather than importing it, exactly the way
#cortex is given get_stock_universe, so there is no circular import and
#report.py stays independently importable for testing.

import sys
import report
import report_rank

report.set_engine_module(sys.modules[__name__])


#==================== FORWARD RECORD (9.1) ====================
#
#every backtest merlin runs is a story about the past told with today's
#universe and today's accounts. the signal store is the opposite: it writes
#down what each strategy said on the day it said it and measures what
#happened next. nothing about that can be tuned afterwards.
#
#signals are logged whether or not the paper book trades them, because the
#paper book is one policy over these signals - position caps, regime gates,
#available cash - and conflating the policy with the signal leaves you unable
#to tell which of the two is not working.
#
#one pass over the universe collects every strategy at once. running the five
#scan tabs separately would mean five downloads per ticker for the same bars.

import signal_store
import datastore

signal_log_status = {'active': False, 'progress': 0, 'message': '', 'complete': False,
                     'error': None, 'logged': None, 'last_run': None}


def _collect_daily_signals(scan_count=300):
    '''one pass over the universe, every strategy, straight into the store'''
    global signal_log_status
    try:
        universe = _try_with_retry(get_stock_universe, scan_count) or []
        if not universe:
            signal_log_status.update({'error': 'could not fetch universe',
                                      'active': False, 'complete': True})
            return

        try:
            spy_hist = drop_incomplete_bars(yf.Ticker('SPY').history(period='1y', auto_adjust=True))
            spy_mom = ((float(spy_hist['Close'].iloc[-22]) - float(spy_hist['Close'].iloc[0]))
                       / float(spy_hist['Close'].iloc[0])) if len(spy_hist) >= 240 else 0.0
        except Exception:
            spy_mom = 0.0

        buckets = {'momentum': [], 'quality': [], 'shareholder_yield': [], 'pead': []}
        total = len(universe)
        for i, t in enumerate(universe):
            signal_log_status['progress'] = int(i / max(total, 1) * 90)
            signal_log_status['message'] = f'collecting {t} ({i + 1}/{total})'
            try:
                m = _try_with_retry(get_momentum_signal, t, spy_mom)
                if m: buckets['momentum'].append(m)
            except Exception: pass
            try:
                for fs in (_try_with_retry(get_research_factor_signals, t, 0) or []):
                    if fs.get('strategy') in buckets:
                        buckets[fs['strategy']].append(fs)
            except Exception: pass
            try:
                p = _try_with_retry(get_pead_signal, t)
                if p: buckets['pead'].append(p)
            except Exception: pass
            if i % 10 == 0 and i > 0: time.sleep(0.3)

        signal_log_status['message'] = 'collecting insider clusters'
        try:
            clusters = get_insider_clusters() or []
            buckets['cluster'] = [{'ticker': c['ticker'], 'confidence': c['confidence'],
                                   'n_insiders': c['n_insiders']} for c in clusters]
        except Exception:
            buckets['cluster'] = []

        #cortex keeps its own signal csv, so take today's confirmed rows from there
        try:
            today = datetime.date.today().isoformat()
            buckets['cortex'] = [{'ticker': f"{s['ticker']}.L" if s.get('market') == 'LSE'
                                  else s['ticker'],
                                  'confidence': 60, 'direction': s.get('side', 'buy').lower(),
                                  'current_price': s.get('entry_price')}
                                 for s in (cortex.recent_signals(200) or [])
                                 if str(s.get('bar_date', ''))[:10] == today]
        except Exception:
            buckets['cortex'] = []

        logged = {}
        for strategy, rows in buckets.items():
            if rows:
                logged[strategy] = signal_store.log_signals(strategy, rows)

        signal_log_status['message'] = 'measuring forward returns'
        signal_log_status['progress'] = 95
        measured = signal_store.update_forward_returns()

        signal_log_status.update({
            'progress': 100, 'complete': True, 'active': False,
            'logged': logged, 'measured': measured,
            'last_run': datetime.datetime.now().isoformat(timespec='seconds'),
            'message': 'logged ' + (', '.join(f'{k} {v}' for k, v in logged.items())
                                    or 'nothing') + f'; {measured} forward returns measured',
        })
    except Exception as e:
        signal_log_status.update({'error': str(e)[:200], 'active': False, 'complete': True})


def _scheduled_signal_log():
    print('\n[scheduler] nightly signal log starting')
    _collect_daily_signals(300)
    print(f"[scheduler] signal log done: {signal_log_status.get('message')}")


#21:00, an hour ahead of the paper cycle. the store records what every
#strategy said today regardless of what the book then does with it, which is
#the whole point - a signal the book had no cash for is still a signal, and
#its forward return still counts. registered here rather than beside the
#paper job because the function has to exist first.
_paper_scheduler.add_job(_scheduled_signal_log, 'cron', hour=21, minute=0,
                         id='daily_signal_log', replace_existing=True)
print('[scheduler] nightly signal log registered for 21:00 local time')


@app.route('/api/signals/collect', methods=['POST'])
def signals_collect_route():
    global signal_log_status
    if signal_log_status.get('active'):
        return jsonify({'error': 'a signal collection is already running'}), 400
    count = int((request.json or {}).get('count', 300))
    signal_log_status = {'active': True, 'progress': 0, 'message': 'starting...',
                         'complete': False, 'error': None, 'logged': None,
                         'last_run': signal_log_status.get('last_run')}
    threading.Thread(target=_collect_daily_signals, args=(count,), daemon=True).start()
    return jsonify({'success': True})


@app.route('/api/signals/status')
def signals_status_route():
    return jsonify(sanitise(signal_log_status))


@app.route('/api/scoreboard')
def scoreboard_route():
    '''
    the forward record. only signals whose horizon has fully elapsed appear,
    so this can never flatter itself with trades that are still open.
    '''
    horizon = int(request.args.get('horizon', signal_store.HEADLINE_HORIZON))
    board = signal_store.scoreboard(horizon=horizon)
    board['store'] = signal_store.stats()
    board['bars'] = datastore.stats()
    return jsonify(sanitise(board))


@app.route('/api/signals/remeasure', methods=['POST'])
def signals_remeasure_route():
    '''re-run the forward measurement without collecting anything new'''
    try:
        n = signal_store.update_forward_returns()
        return jsonify({'success': True, 'measured': n, 'stats': sanitise(signal_store.stats())})
    except Exception as e:
        return jsonify({'error': str(e)[:200]}), 500


@app.route('/api/report/generate', methods=['POST'])
def report_generate_route():
    body = request.json or {}
    try:
        job_id = report.start_job(body.get('ticker', ''))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'could not start the report: {str(e)[:160]}'}), 500
    return jsonify({'job_id': job_id})


@app.route('/api/report/status/<job_id>')
def report_status_route(job_id):
    job = report.job_status(job_id)
    if job is None:
        return jsonify({'error': 'unknown job id - merlin may have restarted'}), 404
    return jsonify(sanitise(job))


@app.route('/api/report/download/<job_id>')
def report_download_route(job_id):
    job = report.job_status(job_id)
    if job is None or not job.get('path') or not os.path.exists(job['path']):
        return jsonify({'error': 'no pdf for that job'}), 404
    return send_file(os.path.abspath(job['path']), mimetype='application/pdf',
                     as_attachment=True, download_name=job['filename'])


@app.route('/api/report/file/<path:filename>')
def report_file_route(filename):
    '''re-download by filename so old reports outlive the job store'''
    path = report.report_path(filename)
    if not path:
        return jsonify({'error': 'no such report'}), 404
    return send_file(path, mimetype='application/pdf', as_attachment=True,
                     download_name=os.path.basename(path))


@app.route('/api/report/recent')
def report_recent_route():
    return jsonify(sanitise({'reports': report.recent_reports(),
                             'rankings': report_rank.ranking_status()}))


@app.route('/api/report/rankings/build', methods=['POST'])
def report_rankings_build_route():
    '''
    the screener and the two factor ranks are cross-sectional models, so the
    honest single-name answer is a percentile. this builds that cross-section
    once and caches it for a day. it is slow - one pass over the universe -
    which is exactly why it is a separate job and never runs inside a report.
    '''
    count = int((request.json or {}).get('count', 150))
    if not report_rank.start_build(count):
        return jsonify({'error': 'a ranking build is already running'}), 400
    return jsonify({'success': True, 'count': count})


@app.route('/api/report/rankings/status')
def report_rankings_status_route():
    return jsonify(sanitise(report_rank.ranking_status()))


#==================== BACKTESTER ====================
#
#walks through historical data and simulates trades per strategy, to answer
#one question: do these strategies work in our universe over the past few
#years. it is not a production trading sim, and it is deliberately honest
#about where it is weak.
#
#what it covers:
#  - the four strategies that survive on OHLCV plus fundamentals alone:
#    momentum, momentum_12_1, quality, shareholder_yield
#  - cortex gap-fade as an optional extra row per market, via include_cortex.
#    it runs off cortex's own yahoo history cache rather than this module's,
#    because a gap needs the real auction open (see cortex.py on why IG daily
#    bars are wrong for history)
#
#what it cannot cover:
#  - cluster, because there is no historical insider filing data to replay
#  - pead, because yfinance historical EPS is too unreliable to trust
#    both of those stay forward-tested through the paper trader instead
#
#the limitations that matter when reading a result:
#  - the universe is today's list, so survivorship bias flatters every number
#  - fills are assumed at the close with no slippage or market impact
#  - one position per ticker per signal, equal-weighted, no portfolio cap
#  - round-trip fees ARE modelled and subtracted from every trade, at
#    FEES_ROUND_TRIP_PCT (0.5% by default, set at the top of this file).
#    fees are usually what separates a real edge from noise here: a
#    thousand-trade strategy paying 0.5% a round trip can lose several times
#    its gross alpha to costs, while quality at ~264 trades barely feels it.
#    this is why meanrev, week52_high and low_beta_trend were retired - they
#    underperformed SPY badly over the 5y backtest once costs were counted.

backtest_status = {'active': False, 'progress': 0, 'message': '',
                   'complete': False, 'results': None, 'error': None}

BACKTESTABLE_STRATEGIES = ['momentum', 'momentum_12_1', 'quality', 'shareholder_yield']

#exit rules used by the backtester. these mirror _check_position_exit but
#operate purely on historical price arrays. trailing logic preserved.
def _bt_check_exit(strategy, entry_price, peak_price, current_price, days_held, rsi=None,
                   below_sma50=False, below_sma200=False, drawdown_from_52w=0):
    pct = (current_price - entry_price) / entry_price
    peak_pct = (peak_price - entry_price) / entry_price
    dd_peak = (current_price - peak_price) / peak_price if peak_price > 0 else 0

    if strategy == 'momentum':
        if peak_pct < 0.10:
            if pct <= -0.09: return 'stop_loss'
        else:
            if dd_peak <= -0.05: return 'trailing_stop'
        if below_sma200: return 'trend_broken_200dma'
        if below_sma50 and pct < 0: return 'lost_50dma'
        if days_held >= 100: return 'time_exit'

    elif strategy == 'momentum_12_1':
        if peak_pct < 0.10:
            if pct <= -0.08: return 'stop_loss'
        else:
            if dd_peak <= -0.04: return 'trailing_stop'
        if below_sma200: return 'trend_broken_200dma'
        if below_sma50 and pct < 0: return 'lost_50dma'
        if days_held >= 120: return 'time_exit'

    elif strategy == 'quality':
        if pct <= -0.12: return 'stop_loss'
        if pct >= 0.20: return 'target_hit'
        if below_sma200: return 'trend_broken_200dma'
        if days_held >= 180: return 'time_exit'

    elif strategy == 'shareholder_yield':
        if pct <= -0.10: return 'stop_loss'
        if pct >= 0.15: return 'target_hit'
        if below_sma200: return 'trend_broken_200dma'
        if days_held >= 180: return 'time_exit'

    return None


def _bt_precompute(histories, all_dates_idx, status_dict=None, progress_start=18, progress_end=24):
    '''
    pre-compute all per-ticker indicator arrays ONCE so the simulation loop
    can do O(1) array lookups instead of O(n) pandas slicing. this is the
    single biggest speedup in the backtester. typically 50-100x faster than
    the naive "slice the dataframe by date every iteration" approach.

    for each ticker we cache:
      close, sma50, sma100, sma200, rsi, rolling_252_high as numpy arrays
      position_at_date: integer array of length len(all_dates) where
                       position_at_date[i] = the last index in this ticker's
                       history that is <= all_dates_idx[i], or -1 if none

    yfinance sometimes returns tz-aware indexes which fail or silently
    produce garbage when searchsorted'd against a tz-naive calendar. we
    normalise both sides to tz-naive before the searchsort.
    '''
    pre = {}
    tickers = list(histories.keys())
    total = len(tickers)
    #ensure the calendar side is tz-naive
    if all_dates_idx.tz is not None:
        all_dates_idx = all_dates_idx.tz_localize(None)
    for ti, ticker in enumerate(tickers):
        if status_dict is not None:
            pct = progress_start + int((ti / max(total, 1)) * max(progress_end - progress_start, 1))
            status_dict['progress'] = pct
            status_dict['message'] = f'pre-computing {ticker} ({ti+1}/{total})'
        try:
            h = histories[ticker]
            #normalise this ticker's index to tz-naive too
            if h.index.tz is not None:
                h = h.copy(); h.index = h.index.tz_localize(None)
            close = h['Close'].astype(float)
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            position_at_date = h.index.searchsorted(all_dates_idx, side='right') - 1
            pre[ticker] = {
                'close': close.values,
                'sma50': close.rolling(50).mean().values,
                'sma100': close.rolling(100).mean().values,
                'sma200': close.rolling(200).mean().values,
                'rsi': rsi.values,
                'high_252': close.rolling(252).max().values,
                'index': h.index,
                'position_at_date': position_at_date,
            }
        except Exception:
            #one bad ticker shouldn't break the whole backtest, skip it
            continue
    return pre


def _bt_signal_fast(strategy, ticker_pre, idx, info, spy_pre, spy_idx):
    '''
    fast signal evaluation using pre-computed arrays. all indicators are
    looked up by integer index in O(1) time. NO pandas dataframe slicing.
    '''
    try:
        close = ticker_pre['close']
        n = len(close)
        if idx < 253 or idx >= n: return False, 0
        cp = float(close[idx])
        if cp <= 3: return False, 0
        sma50 = float(ticker_pre['sma50'][idx])
        sma100 = float(ticker_pre['sma100'][idx])
        sma200 = float(ticker_pre['sma200'][idx])
        rsi = float(ticker_pre['rsi'][idx])
        high_52 = float(ticker_pre['high_252'][idx])
        if sma200 != sma200 or sma200 <= 0: return False, 0  #nan check

        ret_1m = float(close[idx] / close[idx-22] - 1) if idx >= 22 else 0
        ret_6m = float(close[idx] / close[idx-127] - 1) if idx >= 127 else 0
        ret_12_1 = float(close[idx-22] / close[idx-253] - 1) if idx >= 253 else 0

        if strategy == 'momentum':
            if spy_pre is None or spy_idx is None or spy_idx < 253: return False, 0
            spy_close = spy_pre['close']
            spy_mom = float((spy_close[spy_idx-22] - spy_close[spy_idx-253]) / spy_close[spy_idx-253]) if spy_idx >= 253 else 0
            mom_12_1 = ret_12_1
            if mom_12_1 < 0.10: return False, 0
            proximity = cp / high_52 if high_52 > 0 else 0
            if proximity < 0.85: return False, 0
            rel_mom = mom_12_1 - spy_mom
            if rel_mom <= 0: return False, 0
            conf = 50 + min(30, mom_12_1*100) + (10 if proximity >= 0.95 else 5 if proximity >= 0.9 else 0) + (10 if rel_mom >= 0.10 else 5 if rel_mom >= 0.05 else 0)
            return True, min(conf, 95)

        if strategy == 'momentum_12_1':
            trend_ok = cp > sma200 and sma50 > sma200 * 0.98
            if ret_12_1 > 0.20 and ret_6m > 0.08 and trend_ok and -0.08 < ret_1m < 0.25 and rsi < 76:
                conf = 45 + min(ret_12_1*100*0.45, 25) + min(max(ret_6m,0)*100*0.25, 12)
                if cp > sma50: conf += 6
                if sma50 > sma100 > sma200: conf += 6
                if 45 <= rsi <= 68: conf += 4
                return True, min(conf, 95)
            return False, 0

        if strategy in ('quality', 'shareholder_yield'):
            if info is None: return False, 0
            pm = _safe_float(info.get('profitMargins'))
            roe = _safe_float(info.get('returnOnEquity'))
            rg = _safe_float(info.get('revenueGrowth'))
            dte = _safe_float(info.get('debtToEquity'))
            fcf = _safe_float(info.get('freeCashflow'))
            pe = _safe_float(info.get('trailingPE'))

            if strategy == 'quality':
                q = 0
                if pm is not None: q += 18 if pm > 0.15 else 10 if pm > 0.08 else 4 if pm > 0 else -8
                if roe is not None: q += 16 if roe > 0.15 else 9 if roe > 0.08 else 3 if roe > 0 else -6
                if rg is not None: q += 13 if rg > 0.08 else 7 if rg > 0.02 else 2 if rg > 0 else -5
                if fcf is not None and fcf > 0: q += 12
                if dte is not None: q += 12 if dte < 80 else 6 if dte < 150 else -8
                if pe is not None and pe > 0: q += 8 if pe < 25 else 4 if pe < 40 else -6
                if cp > sma200: q += 6
                if ret_6m > 0: q += 4
                if q >= 62 and cp > sma200 * 0.97 and ret_6m > -0.05 and rsi < 78:
                    return True, min(q, 92)
                return False, 0

            if strategy == 'shareholder_yield':
                dy = _dividend_yield_pct(info) / 100.0   #9.1: units settled, kept as a fraction here
                div_pct = dy * 100 if dy < 1 else dy
                mc = _safe_float(info.get('marketCap'), 0) or 0
                if mc < 1e9 or cp <= sma200 * 0.97 or ret_6m < -0.10: return False, 0
                if div_pct < 2: return False, 0
                conf = 50 + min(div_pct * 4, 24)
                if cp > sma50: conf += 5
                return True, min(conf, 90)

        return False, 0
    except Exception:
        return False, 0


def _bt_rsi(close, period=14):
    try:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    except Exception: return 50.0


def _bt_compute_stats(trades, equity_curve, starting_capital):
    '''compute the headline performance metrics for a list of closed trades'''
    if not trades:
        return {'n_trades': 0, 'win_rate_pct': 0, 'avg_return_pct': 0,
                'median_return_pct': 0, 'best_pct': 0, 'worst_pct': 0,
                'total_pl_pct': 0, 'sharpe': 0, 'max_drawdown_pct': 0,
                'avg_days_held': 0}
    returns = [t['return_pct'] for t in trades]
    wins = [r for r in returns if r > 0]
    win_rate = len(wins) / len(returns) * 100
    avg_ret = sum(returns) / len(returns)
    sorted_r = sorted(returns)
    median = sorted_r[len(sorted_r)//2]
    days = [t['days_held'] for t in trades]
    avg_days = sum(days) / len(days)

    if equity_curve:
        peak = equity_curve[0]['equity']
        max_dd = 0
        for pt in equity_curve:
            if pt['equity'] > peak: peak = pt['equity']
            dd = (pt['equity'] - peak) / peak * 100
            if dd < max_dd: max_dd = dd
        final = equity_curve[-1]['equity']
        total_pl_pct = (final - starting_capital) / starting_capital * 100
    else:
        max_dd = 0; total_pl_pct = 0

    if len(returns) > 1:
        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r)**2 for r in returns) / (len(returns)-1)
        std = var**0.5
        per_trade_sharpe = mean_r / std if std > 0 else 0
        ann_factor = (252 / max(avg_days, 1)) ** 0.5
        sharpe = per_trade_sharpe * ann_factor
    else: sharpe = 0

    return {
        'n_trades': len(trades),
        'win_rate_pct': round(win_rate, 1),
        'avg_return_pct': round(avg_ret, 2),
        'median_return_pct': round(median, 2),
        'best_pct': round(max(returns), 2),
        'worst_pct': round(min(returns), 2),
        'total_pl_pct': round(total_pl_pct, 2),
        'sharpe': round(sharpe, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'avg_days_held': round(avg_days, 1),
    }


def _bt_monte_carlo(trades, starting_capital, max_concurrent=8, spy_total_pl_pct=None, n_sims=1000):
    '''
    8.0: bootstrap the closed-trade returns to separate skill from sequencing
    luck. resamples the trade list with replacement n_sims times, recompounds
    equity exactly the way the main loop does, and reports the spread of
    outcomes. a strategy whose 5th percentile is still positive is robust.
    a strategy whose median collapses versus the headline number got lucky
    with trade ordering.
    '''
    if not trades or len(trades) < 10:
        return None
    rets = np.array([t['return_pct'] for t in trades], dtype=float)
    w = 1.0 / max_concurrent
    n = len(rets)
    rng = np.random.default_rng(42)
    samples = rng.choice(rets, size=(n_sims, n), replace=True)
    growth = 1.0 + (samples / 100.0) * w
    paths = starting_capital * np.cumprod(growth, axis=1)
    finals = paths[:, -1]
    totals = (finals - starting_capital) / starting_capital * 100.0
    peaks = np.maximum.accumulate(paths, axis=1)
    dds = (paths - peaks) / peaks * 100.0
    max_dds = dds.min(axis=1)
    out = {
        'n_sims': n_sims,
        'p5_total_pl_pct': round(float(np.percentile(totals, 5)), 2),
        'p25_total_pl_pct': round(float(np.percentile(totals, 25)), 2),
        'median_total_pl_pct': round(float(np.percentile(totals, 50)), 2),
        'p75_total_pl_pct': round(float(np.percentile(totals, 75)), 2),
        'p95_total_pl_pct': round(float(np.percentile(totals, 95)), 2),
        'prob_loss_pct': round(float((totals < 0).mean() * 100), 1),
        'median_max_drawdown_pct': round(float(np.percentile(max_dds, 50)), 2),
        'worst5pct_max_drawdown_pct': round(float(np.percentile(max_dds, 5)), 2),
    }
    if spy_total_pl_pct is not None and np.isfinite(spy_total_pl_pct):
        out['prob_beat_spy_pct'] = round(float((totals > spy_total_pl_pct).mean() * 100), 1)
    return out


def _run_backtest(tickers, years, strategies, scan_freq_days,
                  include_cortex=False, cortex_markets=('US', 'LSE')):
    '''
    fast backtester using pre-computed indicator arrays. simulation walks
    business-day calendar with O(1) lookups per ticker per date.
    '''
    global backtest_status
    try:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=int(years * 365.25))
        warmup_days = 260
        fetch_start = start_date - datetime.timedelta(days=int(warmup_days * 1.5))

        #fetch
        backtest_status['message'] = 'fetching historical data...'
        backtest_status['progress'] = 2
        histories = {}
        ticker_infos = {}
        for i, t in enumerate(tickers):
            backtest_status['message'] = f'fetching {t} ({i+1}/{len(tickers)})'
            backtest_status['progress'] = 2 + int((i / max(len(tickers), 1)) * 15)
            try:
                h = drop_incomplete_bars(yf.Ticker(t).history(start=fetch_start, end=end_date, auto_adjust=True))
                if h.empty or len(h) < 260: continue
                histories[t] = h
                try: ticker_infos[t] = yf.Ticker(t).info or {}
                except Exception: ticker_infos[t] = {}
            except Exception: continue

        try:
            spy_hist = drop_incomplete_bars(yf.Ticker('SPY').history(start=fetch_start, end=end_date, auto_adjust=True))
        except Exception: spy_hist = pd.DataFrame()

        if not histories:
            backtest_status.update({'error': 'no ticker histories could be fetched', 'active': False, 'complete': True})
            return

        backtest_status['message'] = 'pre-computing indicators...'
        backtest_status['progress'] = 18
        all_dates = pd.date_range(start_date, end_date, freq='B')
        all_dates_idx = pd.DatetimeIndex(all_dates)
        #pre-compute updates progress 18->24 ticker by ticker so the user sees it move
        pre = _bt_precompute(histories, all_dates_idx, status_dict=backtest_status,
                              progress_start=18, progress_end=24)
        spy_pre = None
        spy_position_at_date = None
        if not spy_hist.empty:
            backtest_status['message'] = 'pre-computing SPY...'
            backtest_status['progress'] = 25
            spy_pre_dict = _bt_precompute({'SPY': spy_hist}, all_dates_idx)
            spy_pre = spy_pre_dict.get('SPY')
            if spy_pre is not None:
                spy_position_at_date = spy_pre['position_at_date']

        per_strategy_trades = {s: [] for s in strategies}
        equity = {s: [{'date': start_date.isoformat(), 'equity': 10000.0}] for s in strategies}
        open_positions = {s: {} for s in strategies}
        max_concurrent = 8
        starting_capital = 10000.0

        backtest_status['message'] = 'walking through history...'
        backtest_status['progress'] = 26

        total_dates = len(all_dates)
        for di, current_date in enumerate(all_dates):
            #update progress and a "still alive" heartbeat message every few days
            #so the user can see the simulation actively progressing
            if di % 3 == 0:
                pct = 26 + int((di / max(total_dates, 1)) * 72)
                backtest_status['progress'] = pct
                total_open = sum(len(open_positions[s]) for s in strategies)
                total_closed = sum(len(per_strategy_trades[s]) for s in strategies)
                backtest_status['message'] = f'day {di+1}/{total_dates} ({current_date.strftime("%Y-%m-%d")}) - {total_open} open positions, {total_closed} trades closed'

            #step A: mark to market and check exits for all open positions
            for strategy in strategies:
                to_close = []
                for ticker, pos in open_positions[strategy].items():
                    p = pre.get(ticker)
                    if p is None: continue
                    idx = int(p['position_at_date'][di])
                    if idx < 0 or idx >= len(p['close']): continue
                    cp = float(p['close'][idx])
                    if cp != cp: continue  #nan
                    if cp > pos['peak_price']: pos['peak_price'] = cp
                    days_held = (current_date - pos['entry_date']).days
                    sma50 = float(p['sma50'][idx]) if p['sma50'][idx] == p['sma50'][idx] else cp
                    sma200 = float(p['sma200'][idx]) if p['sma200'][idx] == p['sma200'][idx] else cp
                    high_52 = float(p['high_252'][idx]) if p['high_252'][idx] == p['high_252'][idx] else cp
                    rsi = float(p['rsi'][idx]) if p['rsi'][idx] == p['rsi'][idx] else 50.0
                    below_sma50 = cp < sma50
                    below_sma200 = cp < sma200
                    dd_52 = (cp - high_52) / high_52 * 100 if high_52 > 0 else 0
                    reason = _bt_check_exit(strategy, pos['entry_price'], pos['peak_price'],
                                            cp, days_held, rsi=rsi,
                                            below_sma50=below_sma50, below_sma200=below_sma200,
                                            drawdown_from_52w=dd_52)
                    if reason:
                        #7.9: net out round-trip fees from each trade's return
                        gross_pct = (cp - pos['entry_price']) / pos['entry_price'] * 100
                        ret_pct = gross_pct - FEES_ROUND_TRIP_PCT
                        per_strategy_trades[strategy].append({
                            'ticker': ticker, 'strategy': strategy,
                            'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                            'exit_date': current_date.strftime('%Y-%m-%d'),
                            'entry_price': round(pos['entry_price'], 2),
                            'exit_price': round(cp, 2),
                            'return_pct': round(ret_pct, 2),
                            'gross_return_pct': round(gross_pct, 2),
                            'fees_pct': FEES_ROUND_TRIP_PCT,
                            'days_held': days_held,
                            'exit_reason': reason,
                        })
                        last_eq = equity[strategy][-1]['equity']
                        position_weight = 1.0 / max_concurrent
                        new_eq = last_eq * (1 + (ret_pct/100) * position_weight)
                        equity[strategy].append({'date': current_date.isoformat(), 'equity': new_eq})
                        to_close.append(ticker)
                for t in to_close:
                    del open_positions[strategy][t]

            #step B: every scan_freq_days, look for new signals
            if di % scan_freq_days == 0:
                spy_idx = int(spy_position_at_date[di]) if spy_position_at_date is not None else -1
                for strategy in strategies:
                    if len(open_positions[strategy]) >= max_concurrent: continue
                    candidates = []
                    for ticker, p in pre.items():
                        if ticker in open_positions[strategy]: continue
                        idx = int(p['position_at_date'][di])
                        if idx < 253: continue
                        qual, conf = _bt_signal_fast(strategy, p, idx, ticker_infos.get(ticker, {}), spy_pre, spy_idx)
                        if qual and conf >= 65:
                            candidates.append((ticker, conf, float(p['close'][idx])))
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    slots_open = max_concurrent - len(open_positions[strategy])
                    for ticker, conf, entry_price in candidates[:slots_open]:
                        if entry_price != entry_price or entry_price <= 0: continue
                        open_positions[strategy][ticker] = {
                            'entry_price': entry_price,
                            'peak_price': entry_price,
                            'entry_date': current_date,
                            'confidence': conf,
                        }

        #close any still-open positions at the end at last available price
        for strategy in strategies:
            for ticker, pos in open_positions[strategy].items():
                p = pre.get(ticker)
                if p is None: continue
                last_idx = len(p['close']) - 1
                cp = float(p['close'][last_idx])
                if cp != cp: continue
                #7.9: net out round-trip fees here too
                gross_pct = (cp - pos['entry_price']) / pos['entry_price'] * 100
                ret_pct = gross_pct - FEES_ROUND_TRIP_PCT
                days_held = (end_date - pos['entry_date']).days
                per_strategy_trades[strategy].append({
                    'ticker': ticker, 'strategy': strategy,
                    'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': end_date.strftime('%Y-%m-%d'),
                    'entry_price': round(pos['entry_price'], 2),
                    'exit_price': round(cp, 2),
                    'return_pct': round(ret_pct, 2),
                    'gross_return_pct': round(gross_pct, 2),
                    'fees_pct': FEES_ROUND_TRIP_PCT,
                    'days_held': days_held,
                    'exit_reason': 'backtest_end',
                })
                last_eq = equity[strategy][-1]['equity']
                position_weight = 1.0 / max_concurrent
                new_eq = last_eq * (1 + (ret_pct/100) * position_weight)
                equity[strategy].append({'date': end_date.isoformat(), 'equity': new_eq})

        #8.0: SPY buy-and-hold over the same window - the number to beat
        #8.1 fix: yfinance can return an incomplete final row (today's bar
        #before the close) whose Close is NaN. that NaN poisoned the total,
        #every monte carlo comparison against NaN returned False (hence the
        #beats-SPY column reading 0% everywhere) and the json encoder nulled
        #the benchmark line. dropna plus an isfinite guard kills both symptoms.
        spy_total_pl_pct = None
        spy_win = None
        try:
            if not spy_hist.empty:
                sc = spy_hist['Close'].dropna()
                idx = sc.index.tz_localize(None) if sc.index.tz is not None else sc.index
                win = sc[idx >= pd.Timestamp(start_date)]
                if len(win) > 1 and float(win.iloc[0]) > 0:
                    val = float((win.iloc[-1] - win.iloc[0]) / win.iloc[0] * 100)
                    if np.isfinite(val):
                        spy_total_pl_pct = round(val, 2)
                        spy_win = win
        except Exception:
            spy_total_pl_pct = None
            spy_win = None

        per_strategy_stats = {}
        for strategy in strategies:
            per_strategy_stats[strategy] = _bt_compute_stats(per_strategy_trades[strategy], equity[strategy], starting_capital)
            per_strategy_stats[strategy]['trades'] = per_strategy_trades[strategy][-50:]
            per_strategy_stats[strategy]['equity_curve'] = equity[strategy]
            #8.0: monte carlo bootstrap of this strategy's trades
            per_strategy_stats[strategy]['monte_carlo'] = _bt_monte_carlo(
                per_strategy_trades[strategy], starting_capital,
                max_concurrent=max_concurrent, spy_total_pl_pct=spy_total_pl_pct)

        #8.1: SPY buy-and-hold gets its own row so the benchmark sits right
        #next to the strategies with a viewable equity curve
        result_strategies = list(strategies)
        if spy_win is not None and spy_total_pl_pct is not None:
            try:
                base = float(spy_win.iloc[0])
                spy_curve = [{'date': d.isoformat(), 'equity': round(starting_capital * float(v) / base, 2)}
                             for d, v in spy_win.items()]
                spy_rets = spy_win.pct_change().dropna()
                spy_sharpe = round(float(spy_rets.mean() / spy_rets.std() * (252 ** 0.5)), 2) if len(spy_rets) > 2 and float(spy_rets.std()) > 0 else 0
                peak = -1e18; max_dd = 0.0
                for pt in spy_curve:
                    if pt['equity'] > peak: peak = pt['equity']
                    dd = (pt['equity'] - peak) / peak * 100
                    if dd < max_dd: max_dd = dd
                window_days = (end_date - start_date).days
                per_strategy_stats['spy_buy_hold'] = {
                    'n_trades': 1, 'win_rate_pct': 100.0 if spy_total_pl_pct > 0 else 0.0,
                    'avg_return_pct': spy_total_pl_pct, 'median_return_pct': spy_total_pl_pct,
                    'best_pct': spy_total_pl_pct, 'worst_pct': spy_total_pl_pct,
                    'total_pl_pct': spy_total_pl_pct, 'sharpe': spy_sharpe,
                    'max_drawdown_pct': round(max_dd, 2), 'avg_days_held': window_days,
                    'trades': [{
                        'ticker': 'SPY', 'strategy': 'spy_buy_hold',
                        'entry_date': start_date.strftime('%Y-%m-%d'),
                        'exit_date': end_date.strftime('%Y-%m-%d'),
                        'entry_price': round(base, 2), 'exit_price': round(float(spy_win.iloc[-1]), 2),
                        'return_pct': spy_total_pl_pct, 'gross_return_pct': spy_total_pl_pct,
                        'fees_pct': 0, 'days_held': window_days, 'exit_reason': 'benchmark_hold',
                    }],
                    'equity_curve': spy_curve, 'monte_carlo': None,
                }
                result_strategies.append('spy_buy_hold')
            except Exception:
                pass
                #--- cortex gap-and-fade rows -------------------------------------
        if include_cortex:
            for cx_market in cortex_markets:
                try:
                    backtest_status['message'] = f'simulating cortex {cx_market}...'
                    #cortex needs the full universe to find gaps at all. the
                    #main backtest's 40-ticker default left it almost nothing
                    #to work with, which is why the US row looked so bad
                    cx_tickers = cortex.get_universe(cx_market)

                    cx_trades, cx_equity = cortex.cortex_backtest_trades(
                        cx_market, cx_tickers, years=years,
                        gap_min=cortex.DEFAULT_GAP_MIN, gap_max=cortex.DEFAULT_GAP_MAX,
                        wing=cortex.DEFAULT_WING, exit_day=cortex.DEFAULT_EXIT_DAY,
                        fee_pct=FEES_ROUND_TRIP_PCT, allow_short=True,
                        status=None)
                    if not cx_trades:
                        continue

                    name = f'cortex_{cx_market.lower()}'
                    stats = _bt_compute_stats(cx_trades, cx_equity, starting_capital)
                    stats['trades'] = cx_trades[-50:]
                    stats['equity_curve'] = cx_equity
                    stats['monte_carlo'] = _bt_monte_carlo(
                        cx_trades, starting_capital,
                        max_concurrent=cortex.BT_MAX_CONCURRENT,
                        spy_total_pl_pct=spy_total_pl_pct)
                    per_strategy_stats[name] = stats
                    result_strategies.append(name)
                except Exception as e:
                    #a cortex failure must never take the whole backtest down
                    print(f'  cortex backtest ({cx_market}) failed: {e}\n')         
        backtest_status.update({
            'progress': 100, 'complete': True, 'active': False,
            'message': f'done - {sum(len(per_strategy_trades[s]) for s in strategies)} trades simulated (fees {FEES_ROUND_TRIP_PCT}% per round trip)',
            'results': {
                'strategies': result_strategies,
                'tickers_evaluated': list(histories.keys()),
                'years': years,
                'scan_freq_days': scan_freq_days,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'fees_round_trip_pct': FEES_ROUND_TRIP_PCT,
                'spy_total_pl_pct': spy_total_pl_pct,
                'per_strategy': per_strategy_stats,
            }
        })
    except Exception as e:
        backtest_status.update({'error': f'{type(e).__name__}: {str(e)[:200]}', 'active': False, 'complete': True})


@app.route('/api/backtest/run', methods=['POST'])
def backtest_run_route():
    global backtest_status
    if backtest_status.get('active'): return jsonify({'error': 'backtest already running'}), 400
    body = request.json or {}
    years = float(body.get('years', 3))
    scan_freq_days = int(body.get('scan_freq_days', 5))
    custom_tickers = body.get('tickers')
    strategies = body.get('strategies') or BACKTESTABLE_STRATEGIES
    #honour requested universe size, default 40 if not specified
    count = int(body.get('count', 40))
    #8.3: cortex rides along as its own rows rather than a merlin strategy
    include_cortex = bool(body.get('include_cortex', True))
    cortex_markets = body.get('cortex_markets') or ['US', 'LSE']
    if custom_tickers:
        tickers = [t.strip().upper() for t in custom_tickers if t.strip()]
    else:
        tickers = (_try_with_retry(get_stock_universe, count) or [])[:count]

    backtest_status = {'active': True, 'progress': 0, 'message': 'starting backtest...',
                       'complete': False, 'results': None, 'error': None}
    threading.Thread(target=_run_backtest,
                     args=(tickers, years, strategies, scan_freq_days,
                           include_cortex, cortex_markets), daemon=True).start()
    return jsonify({'success': True, 'started': True, 'tickers_queued': len(tickers)})


@app.route('/api/backtest/status')
def backtest_status_route():
    return jsonify(sanitise(backtest_status))


if __name__=='__main__':
    '''
    debug=True keeps the nice tracebacks and template auto-refresh.
    use_reloader=False turns off the file watcher entirely. on windows
    the watchdog-based reloader is over-eager about yfinance cache writes
    inside site-packages and kept killing background scans mid-run. the
    newer exclude_patterns API also raised ValueError on conflicting
    patterns. simplest and safest is just to disable the reloader.
    the cost: you have to stop and restart manually after editing app.py.
    that's worth it to keep the AI paper cycle and backtester alive.

    7.8 startup: close any open paper positions tagged with retired strategies
    (meanrev, week52_high, low_beta_trend) so the bot does not hold them
    forever under exit rules that no longer exist.

    7.9 changes: equity display bug fixed (positions no longer vanish when
    yfinance fails). hard regime gate added for momentum strategies in bear
    and volatile regimes. backtester now nets out a 0.5% round-trip fee on
    every trade so the numbers reflect what a real Trading 212 account
    would see.

    8.0 startup: one-off migration closes every open paper position, archives
    the old book to data/paper_portfolio_v7_archive.json and starts a fresh
    £10,000 portfolio. live fills now pay fees (0.15% fx each way off-gbp
    plus 0.10% spread/slippage each way), idle cash sweeps into SPY during
    healthy regimes, position sizes are volatility-scaled and momentum
    signals carry a smoothness bonus. the backtester reports a SPY benchmark
    and a 1000-run monte carlo bootstrap per strategy.

    8.2 startup: one-off repair rebuilds any book poisoned by nan prices or
    fx (corrupted closes voided at entry price, unknowable positions dropped,
    cash reconstructed from the accounting identity). all price and fx
    fetches now reject non-finite values at source and every cash mutation
    has a circuit breaker. the 22:00 scheduler actually starts now - it was
    gated behind a reloader env var that never exists with the reloader off.
    '''
    _migrate_portfolio_to_v8()
    _repair_nan_portfolio()
    _cleanup_retired_strategy_positions()
    app.run(debug=True, port=5000, use_reloader=False)

