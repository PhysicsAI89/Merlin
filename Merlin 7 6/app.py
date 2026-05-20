'''
merlin 7.2 - ensemble prediction engine
adds market context features, honest multi-step backtest, baselines comparison,
walk-forward xgboost validation and a leak-free training pipeline
'''

import os, json, datetime, re, time, threading, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
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



#paper trader constants
PAPER_PORTFOLIO_PATH = os.path.join('data', 'paper_portfolio.json')
STARTING_CASH_GBP = 1000.0
MAX_POSITIONS = 8
MIN_POSITION_PCT = 0.05
MAX_POSITION_PCT = 0.15
MIN_CONFIDENCE_TO_OPEN = 65

# AI paper trader allocation rules. The cycle may find dozens of signals,
# especially momentum-style ones, so we cap each strategy per cycle and rank
# by confidence with a small evidence-priority bonus. This stops one noisy
# strategy filling all 8 slots.
MAX_NEW_POSITIONS_PER_STRATEGY = {
    'cluster': 2,
    'pead': 2,
    'meanrev': 2,
    'momentum': 3,          # existing relative 12-1 momentum strategy
    'momentum_12_1': 3,     # stricter 12-1 momentum factor
    'week52_high': 2,
    'quality': 2,
    'shareholder_yield': 2,
    'low_beta_trend': 2,
}
STRATEGY_SCORE_BONUS = {
    'cluster': 4,
    'pead': 3,
    'momentum': 2,
    'momentum_12_1': 2,
    'week52_high': 1.5,
    'meanrev': 1,
    'quality': 0.5,
    'shareholder_yield': 0,
    'low_beta_trend': 0,
}
# group caps are stricter than per-strategy caps. This is the important part:
# momentum-style signals can throw 20+ names, so the AI can only open 3 total
# from the whole momentum group in one cycle.
STRATEGY_GROUP = {
    'cluster': 'event',
    'pead': 'event',
    'meanrev': 'tactical',
    'momentum': 'momentum',
    'momentum_12_1': 'momentum',
    'week52_high': 'momentum',
    'quality': 'fundamental',
    'shareholder_yield': 'fundamental',
    'low_beta_trend': 'defensive',
}
MAX_NEW_POSITIONS_PER_GROUP = {
    'event': 3,
    'tactical': 2,
    'momentum': 3,
    'fundamental': 3,
    'defensive': 2,
}
# portfolio-wide caps: total positions of each group held at any time across
# all cycles. without this, momentum-style signals can quietly fill 7-8 slots
# over a few weeks and leave no room for rare tactical signals like meanrev
# when they finally appear. this enforces structural diversification.
MAX_PORTFOLIO_POSITIONS_PER_GROUP = {
    'event': 4,
    'tactical': 3,
    'momentum': 4,
    'fundamental': 3,
    'defensive': 2,
}

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
mr_status = {'active':False,'progress':0,'message':'','complete':False,'results':None,'error':None}
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
        spy = yf.download('SPY', start=start, end=end, interval='1d', progress=False)
        vix = yf.download('^VIX', start=start, end=end, interval='1d', progress=False)
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
        vl = str(int(min_value)) if min_value else ''

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

def compute_indicators(df):
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
        dy = info.get('dividendYield',0)
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
            'dividend_yield':round(dy*100,2) if dy else 0,
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


def get_insider_activity(ticker):
    try:
        stock = yf.Ticker(ticker); transactions = []
        try:
            idf = stock.insider_transactions
            if idf is not None and not idf.empty:
                for _,row in idf.head(20).iterrows():
                    name=str(row.get('Insider Trading',row.get('insider','')))
                    title=str(row.get('Position',row.get('position','')))
                    trans=str(row.get('Transaction',row.get('transaction','')))
                    date=str(row.get('Start Date',row.get('startDate','')))
                    shares=row.get('Shares',row.get('shares',0))
                    value=row.get('Value',row.get('value',0))
                    tl=trans.lower()
                    action='buy' if any(w in tl for w in ['purchase','buy','acquisition']) else 'sell' if any(w in tl for w in ['sale','sell','disposition']) else 'other'
                    is_exec=any(t in title.lower() for t in ['ceo','cfo','chief executive','chief financial','president','chairman','director','officer','vp','vice president'])
                    try: sv=int(float(str(shares).replace(',','')))
                    except: sv=0
                    try: vv=float(str(value).replace(',','').replace('$',''))
                    except: vv=0
                    transactions.append({'name':name[:40],'title':title[:30],'action':action,'date':str(date)[:10],'shares':sv,'value':vv,'is_executive':is_exec})
        except: pass
        eb=sum(1 for t in transactions if t['is_executive'] and t['action']=='buy')
        es=sum(1 for t in transactions if t['is_executive'] and t['action']=='sell')
        return {'transactions':transactions,'exec_buys':eb,'exec_sells':es,'all_buys':sum(1 for t in transactions if t['action']=='buy'),'all_sells':sum(1 for t in transactions if t['action']=='sell'),'sentiment':'bullish' if eb>es else 'bearish' if es>eb else 'neutral','score':min(eb-es,3) if eb>es else -min(es-eb,3) if es>eb else 0}
    except: return {'transactions':[],'exec_buys':0,'exec_sells':0,'all_buys':0,'all_sells':0,'sentiment':'unknown','score':0}


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
        hist = yf.Ticker(ticker).history(period='3mo')
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
        #try both api paths since yfinance has changed it
        try:
            earnings = stock.earnings_dates
        except:
            try: earnings = stock.get_earnings_dates(limit=12)
            except: return None
        if earnings is None or earnings.empty: return None

        #find most recent past earnings event
        idx_tz = earnings.index.tz
        now = pd.Timestamp.now(tz=idx_tz) if idx_tz else pd.Timestamp.now()
        past = earnings[earnings.index <= now]
        if past.empty: return None

        latest = past.iloc[-1]; latest_date = past.index[-1]
        latest_naive = latest_date.tz_localize(None) if latest_date.tz else latest_date
        days_since = (pd.Timestamp.now() - latest_naive).days
        if days_since < 1 or days_since > max_days_since: return None

        #find surprise across various column names yfinance has used
        eps_actual = None; eps_estimate = None
        for col in ['Reported EPS','EPS Actual','Actual EPS']:
            if col in latest.index and not pd.isna(latest[col]):
                eps_actual = latest[col]; break
        for col in ['EPS Estimate','Estimate','EPS Est']:
            if col in latest.index and not pd.isna(latest[col]):
                eps_estimate = latest[col]; break
        if eps_actual is None or eps_estimate is None or eps_estimate == 0: return None

        surprise_pct = float((eps_actual - eps_estimate) / abs(eps_estimate) * 100)
        if surprise_pct > 5: direction = 'up'
        elif surprise_pct < -5: direction = 'down'
        else: return None

        #price move since earnings
        hist = stock.history(period='3mo')
        if hist.empty: return None
        current_price = float(hist['Close'].iloc[-1])
        try:
            after = hist[hist.index >= latest_naive]
            if len(after) > 0:
                earn_price = float(after['Close'].iloc[0])
                move = (current_price - earn_price) / earn_price * 100
            else: move = 0
        except: move = 0

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


def check_mean_reversion(ticker, vix_level, max_vix=18):
    '''detect a mean-reversion setup that passes all four gates'''
    if vix_level >= max_vix: return None
    try:
        data = yf.download(ticker, period='1y', interval='1d', progress=False)
        if data.empty or len(data) < 200: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = compute_indicators(data)

        cp = float(data['Close'].iloc[-1])
        sma200 = float(pd.Series(data['Close'].values).rolling(200, min_periods=1).mean().iloc[-1])
        rsi = float(data['RSI'].iloc[-1])
        bb_pos = float(data['BB_Position'].iloc[-1])
        vol_ratio = float(data['Vol_Ratio'].iloc[-1])

        #uptrend gate: must be within 5% of or above 200dma
        if cp < sma200 * 0.95: return None
        #rsi extreme gate
        if 30 < rsi < 70: return None
        direction = 'buy' if rsi < 30 else 'sell'

        #confidence components
        rsi_score = min((30 - rsi) * 2, 35) if rsi < 30 else min((rsi - 70) * 2, 35)
        bb_score = max(15 * (1 - bb_pos), 0) if direction == 'buy' else max(15 * bb_pos, 0)
        vol_score = min((vol_ratio - 1) * 5, 15) if vol_ratio > 1 else 0
        vix_score = max((max_vix - vix_level) * 1.5, 0)
        confidence = round(rsi_score + bb_score + vol_score + vix_score, 1)

        info = yf.Ticker(ticker).info or {}
        return {
            'ticker': ticker, 'name': info.get('shortName', ticker),
            'sector': info.get('sector', 'unknown'),
            'direction': direction, 'current_price': round(cp, 2),
            'rsi': round(rsi, 1), 'bb_position': round(bb_pos * 100, 1),
            'sma200': round(sma200, 2),
            'price_vs_sma200_pct': round((cp - sma200) / sma200 * 100, 2),
            'vol_ratio': round(vol_ratio, 2), 'vix_level': round(vix_level, 1),
            'confidence': confidence, 'expected_horizon': '5-10 days',
            'market_cap_str': format_market_cap(info.get('marketCap', 0))
        }
    except: return None




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
        hist = yf.Ticker(ticker).history(period='1y', auto_adjust=True)
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

        info = yf.Ticker(ticker).info or {}
        return {
            'ticker': ticker, 'name': info.get('shortName', ticker),
            'sector': info.get('sector', 'unknown'),
            'current_price': round(p_now, 2),
            'mom_12_1_pct': round(mom_12_1 * 100, 2),
            'proximity_to_52w_high_pct': round(proximity * 100, 1),
            'relative_mom_vs_spy_pct': round(rel_mom * 100, 2),
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

FACTOR_STRATEGIES = ['momentum', 'momentum_12_1', 'week52_high', 'quality', 'shareholder_yield', 'low_beta_trend']


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

    #1) 12-1 month momentum: strong medium-term winners, but avoid recent exhaustion/crashes
    if ret_12_1 > 0.20 and ret_6m > 0.08 and trend_ok and -0.08 < ret_1m < 0.25 and rsi < 76:
        confidence = 45
        confidence += min(ret_12_1 * 100 * 0.45, 25)
        confidence += min(max(ret_6m, 0) * 100 * 0.25, 12)
        if cp > sma50: confidence += 6
        if sma50 > sma100 > sma200: confidence += 6
        if 45 <= rsi <= 68: confidence += 4
        confidence = round(min(confidence, 95), 1)
        if confidence >= min_confidence:
            signals.append({
                'ticker': ticker, 'strategy': 'momentum_12_1', 'direction': 'buy',
                'confidence': confidence, 'current_price': round(cp, 2),
                'reason': f'12-1 momentum {ret_12_1*100:.1f}%, 6m {ret_6m*100:.1f}%, price above 200dma',
                'rsi': round(rsi, 1), 'ret_12_1_pct': round(ret_12_1*100, 1),
                'name': name, 'sector': sector, 'market_cap_str': format_market_cap(market_cap)
            })

    #2) 52-week high momentum: buy strength near highs, but not if wildly overbought
    high_ratio = cp / (high_52 + 1e-10)
    if high_ratio >= 0.93 and ret_6m > 0.05 and cp > sma50 and cp > sma200 and 45 <= rsi <= 78 and ret_1m > -0.03:
        confidence = 48
        confidence += min((high_ratio - 0.93) / 0.07 * 18, 18)
        confidence += min(max(ret_6m, 0) * 100 * 0.25, 12)
        if sma50 > sma100 > sma200: confidence += 8
        if avg_vol_20 > 500_000: confidence += 3
        confidence = round(min(confidence, 93), 1)
        if confidence >= min_confidence:
            signals.append({
                'ticker': ticker, 'strategy': 'week52_high', 'direction': 'buy',
                'confidence': confidence, 'current_price': round(cp, 2),
                'reason': f'price is {high_ratio*100:.1f}% of 52w high with positive 6m trend',
                'rsi': round(rsi, 1), 'range_position_pct': round((cp-low_52)/(high_52-low_52+1e-10)*100, 1),
                'name': name, 'sector': sector, 'market_cap_str': format_market_cap(market_cap)
            })

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
    dy = _safe_float(info.get('dividendYield'), 0) or 0
    dividend_yield_pct = dy * 100 if dy < 1 else dy
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

    return signals




def get_low_beta_trend_signal(ticker, min_confidence=65):
    '''
    Defensive trend signal: low-beta, liquid stocks in a positive trend.
    Useful when momentum exists but the trader should avoid very high-beta names.
    This is deliberately conservative and buy-only.
    '''
    data = _download_factor_history(ticker, period='2y')
    if data is None or len(data) < 252: return None
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    close = data['Close'].astype(float)
    volume = data['Volume'].astype(float) if 'Volume' in data.columns else pd.Series([0]*len(data), index=data.index)
    cp = float(close.iloc[-1])
    if cp <= 3 or float(volume.tail(20).mean()) < 100_000: return None

    beta = _safe_float(info.get('beta'))
    if beta is None or beta <= 0 or beta > 1.10: return None

    sma50 = float(data['SMA_50'].iloc[-1])
    sma100 = float(data['SMA_100'].iloc[-1])
    sma200 = float(data['SMA_200'].iloc[-1])
    rsi = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns else 50.0
    ret_6m = float(close.iloc[-1] / close.iloc[-127] - 1) if len(close) > 127 else 0.0
    ret_12_1 = float(close.iloc[-22] / close.iloc[-253] - 1) if len(close) > 253 else 0.0
    if not (cp > sma50 > sma100 * 0.98 and sma50 > sma200 and ret_6m > 0.04 and ret_12_1 > 0.08 and 40 <= rsi <= 72):
        return None

    confidence = 48
    confidence += min((1.10 - beta) * 20, 12)
    confidence += min(ret_6m * 100 * 0.35, 14)
    confidence += min(ret_12_1 * 100 * 0.25, 14)
    if sma50 > sma100 > sma200: confidence += 8
    if 45 <= rsi <= 65: confidence += 4
    confidence = round(min(confidence, 90), 1)
    if confidence < min_confidence: return None

    return {
        'ticker': ticker,
        'strategy': 'low_beta_trend',
        'direction': 'buy',
        'confidence': confidence,
        'current_price': round(cp, 2),
        'reason': f'low-beta trend: beta {beta:.2f}, 6m {ret_6m*100:.1f}%, price above 50/200dma',
        'beta': round(beta, 2),
        'rsi': round(rsi, 1),
        'ret_6m_pct': round(ret_6m*100, 1),
        'name': info.get('shortName', ticker),
        'sector': info.get('sector', 'unknown'),
        'market_cap_str': format_market_cap(info.get('marketCap', 0))
    }

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
                h = yf.Ticker(ticker).history(period='2mo', auto_adjust=True)
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


def _candidate_effective_score(c):
    '''confidence plus a small strategy evidence-priority bonus, minus any
    short interest penalty so heavily shorted names rank below clean ones.'''
    base = float(c.get('confidence', 0)) + STRATEGY_SCORE_BONUS.get(c.get('strategy'), 0)
    return base - float(c.get('_short_penalty', 0))


def _rank_paper_candidates(candidates, slots, held_tickers=None, held_positions=None):
    '''
    Choose which signals the AI trader listens to.
    1) remove held tickers and low-confidence signals
    2) keep only the best signal per ticker
    3) sort by adjusted confidence
    4) enforce per-cycle caps so one strategy cannot fill all new slots
    5) enforce portfolio-wide group caps so momentum cannot dominate the book
       over multiple cycles. this leaves room for rare tactical signals.
    '''
    held_tickers = held_tickers or set()
    held_positions = held_positions or []

    #count what's already in the portfolio by group, used for the portfolio cap
    existing_group_counts = {}
    for p in held_positions:
        g = STRATEGY_GROUP.get(p.get('strategy'), p.get('strategy', 'unknown'))
        existing_group_counts[g] = existing_group_counts.get(g, 0) + 1

    best_by_ticker = {}
    short_filter_blocked = []
    for c in candidates:
        t = c.get('ticker')
        if not t or t in held_tickers: continue
        if c.get('confidence', 0) < MIN_CONFIDENCE_TO_OPEN: continue
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
        if prev is None or _candidate_effective_score(c) > _candidate_effective_score(prev):
            best_by_ticker[t] = c

    ordered = sorted(best_by_ticker.values(),
                     key=lambda c: (_candidate_effective_score(c), c.get('confidence', 0)),
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


@app.route('/api/mean_reversion_scan', methods=['POST'])
def mr_scan_route():
    global mr_status
    if mr_status['active']: return jsonify({'error': 'already running'}), 400
    d = request.json or {}
    count = d.get('count', 200); max_vix = d.get('max_vix', 18)
    mr_status = {'active': True, 'progress': 0, 'message': 'starting...',
                 'complete': False, 'results': None, 'error': None}
    threading.Thread(target=_run_mr_scan, args=(count, max_vix), daemon=True).start()
    return jsonify({'status': 'started'})


def _run_mr_scan(count, max_vix):
    global mr_status
    try:
        vix = get_vix_level()
        if vix >= max_vix:
            mr_status.update({'results': {'signals': [], 'vix_level': vix, 'gate_open': False,
                                          'message': f'VIX at {vix:.1f} is above threshold {max_vix} - gate is closed'},
                              'progress': 100, 'complete': True, 'active': False,
                              'message': 'market too turbulent - gate closed'})
            return
        stocks = get_stock_universe(count); signals = []
        total = len(stocks)
        for i, t in enumerate(stocks):
            mr_status['progress'] = int((i / total) * 100)
            mr_status['message'] = f'checking {t} ({i+1}/{total})...'
            sig = check_mean_reversion(t, vix, max_vix)
            if sig: signals.append(sig)
            if i % 10 == 0 and i > 0: time.sleep(0.3)
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        mr_status.update({'results': {'signals': signals, 'vix_level': vix, 'gate_open': True,
                                      'message': f'VIX at {vix:.1f} - {len(signals)} mean reversion signals found'},
                          'progress': 100, 'complete': True, 'active': False,
                          'message': f'done - {len(signals)} signals'})
    except Exception as e:
        mr_status.update({'error': str(e), 'active': False})


@app.route('/api/mr_status')
def mr_status_route():
    return jsonify(sanitise(mr_status))


@app.route('/api/vix_now')
def vix_now_route():
    '''quick fetch of current VIX for the mean reversion tab to display live'''
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
    closed = [t for t in trades if t['status'] == 'closed' and t.get('realised_pl_pct') is not None]
    open_ = [t for t in trades if t['status'] == 'open']
    wins = [t for t in closed if t['realised_pl_pct'] > 0]
    summary = {
        'total_trades': len(trades),
        'open': len(open_),
        'closed': len(closed),
        'win_rate': round(len(wins)/len(closed)*100, 1) if closed else 0,
        'avg_realised_pct': round(np.mean([t['realised_pl_pct'] for t in closed]), 2) if closed else 0,
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
        'auto_run_enabled': False
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

def _get_fx_to_gbp(from_currency):
    '''spot fx to gbp, london pence is already normalised before this'''
    if from_currency == 'GBP': return 1.0
    try:
        pair = f'{from_currency}GBP=X'
        hist = yf.Ticker(pair).history(period='1d')
        if len(hist) > 0: return float(hist['Close'].iloc[-1])
    except Exception: pass
    fallbacks = {'USD': 0.79, 'EUR': 0.85, 'CAD': 0.58, 'AUD': 0.52}
    return fallbacks.get(from_currency, 1.0)

def _paper_price_and_currency(sym):
    '''latest close and normalised currency with retry, london pence becomes pounds'''
    for attempt in range(3):
        try:
            info = yf.Ticker(sym).info or {}
            raw_ccy = info.get('currency', 'USD')
            hist = yf.Ticker(sym).history(period='5d')
            if len(hist) == 0:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1)); continue
                return None, None
            price = float(hist['Close'].iloc[-1])
            if raw_ccy == 'GBp': return price / 100.0, 'GBP'
            return price, raw_ccy
        except Exception:
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    return None, None

def _calculate_position_size_gbp(confidence, total_equity_gbp, available_cash_gbp):
    '''size 5pct to 15pct of equity scaled by confidence above 60'''
    conf_clamped = max(60, min(100, confidence))
    pct = MIN_POSITION_PCT + (MAX_POSITION_PCT - MIN_POSITION_PCT) * ((conf_clamped - 60) / 40.0)
    target_gbp = total_equity_gbp * pct
    return min(target_gbp, available_cash_gbp * 0.99)

def _paper_rsi(ticker_sym, period=14):
    try:
        hist = yf.Ticker(ticker_sym).history(period='3mo')
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
    '''per-strategy exit rules for the AI paper trader'''
    exit_ctx = exit_ctx or {}
    strategy = position['strategy']
    entry = position['entry_price_native']
    pct_change = (current_price_native - entry) / entry
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

    elif strategy == 'meanrev':
        if pct_change <= -0.05: return True, 'stop_loss'
        if pct_change >= 0.08:  return True, 'bounce_target_hit'
        if days_held >= 14:     return True, 'time_exit'
        if rsi is not None and 40 <= rsi <= 60: return True, 'rsi_normalised'

    elif strategy == 'momentum':
        if pct_change <= -0.09: return True, 'stop_loss'
        if pct_change >= 0.22:  return True, 'target_hit'
        if exit_ctx.get('below_sma200'): return True, 'trend_broken_200dma'
        if exit_ctx.get('below_sma50') and pct_change < 0: return True, 'lost_50dma'
        if days_held >= 100:    return True, 'time_exit'

    elif strategy == 'momentum_12_1':
        if pct_change <= -0.08: return True, 'stop_loss'
        if pct_change >= 0.18:  return True, 'target_hit'
        if exit_ctx.get('below_sma200'): return True, 'trend_broken_200dma'
        if exit_ctx.get('below_sma50') and pct_change < 0: return True, 'lost_50dma'
        if days_held >= 120:    return True, 'time_exit'

    elif strategy == 'week52_high':
        if pct_change <= -0.07: return True, 'stop_loss'
        if pct_change >= 0.15:  return True, 'target_hit'
        if exit_ctx.get('drawdown_from_52w_high_pct', 0) <= -8: return True, 'failed_breakout'
        if exit_ctx.get('below_sma50') and pct_change < 0: return True, 'lost_50dma'
        if days_held >= 90:     return True, 'time_exit'

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

    elif strategy == 'low_beta_trend':
        if pct_change <= -0.06: return True, 'stop_loss'
        if pct_change >= 0.12:  return True, 'target_hit'
        if exit_ctx.get('below_sma200'): return True, 'trend_broken_200dma'
        if exit_ctx.get('below_sma50') and pct_change < 0: return True, 'lost_50dma'
        if days_held >= 120:    return True, 'time_exit'

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
    try:
        #step 1 mark to market
        paper_cycle_status['message'] = 'marking positions to market...'
        total_value_gbp = 0.0
        for pos in portfolio['positions']:
            try:
                price, _ = _paper_price_and_currency(pos['ticker'])
                if price is None: continue
                pos['current_price_native'] = price
                fx = _get_fx_to_gbp(pos['currency'])
                pos['current_value_gbp'] = pos['shares'] * price * fx
                pos['unrealised_pl_gbp'] = pos['current_value_gbp'] - pos['cost_gbp']
                pos['unrealised_pl_pct'] = (price - pos['entry_price_native']) / pos['entry_price_native'] * 100
                total_value_gbp += pos['current_value_gbp']
            except Exception as e:
                activity.append(f"warning could not price {pos['ticker']}: {str(e)[:60]}")
        total_equity_gbp = portfolio['cash_gbp'] + total_value_gbp

        #step 2 check exits
        paper_cycle_status['message'] = 'checking exit conditions...'
        to_close = []
        for pos in portfolio['positions']:
            if 'current_price_native' not in pos: continue
            exit_ctx = _paper_exit_context(pos['ticker'])
            should_exit, reason = _check_position_exit(pos, pos['current_price_native'], exit_ctx)
            if should_exit: to_close.append((pos, reason))
        for pos, reason in to_close:
            proceeds = pos['current_value_gbp']
            realised_pl = proceeds - pos['cost_gbp']
            realised_pct = realised_pl / pos['cost_gbp'] * 100
            portfolio['cash_gbp'] += proceeds
            portfolio['closed_trades'].append({
                **pos, 'exit_date': now_iso,
                'exit_price_native': pos['current_price_native'],
                'exit_reason': reason, 'realised_pl_gbp': realised_pl,
                'realised_pl_pct': realised_pct, 'proceeds_gbp': proceeds
            })
            portfolio['positions'].remove(pos)
            _log_paper_trade_to_main_log(pos, pos['current_price_native'], reason)
            activity.append(f"closed {pos['ticker']} ({pos['strategy']}) {reason} £{realised_pl:+.2f} ({realised_pct:+.2f}%)")

        #step 3 scan for candidates
        candidates = []
        if len(portfolio['positions']) < MAX_POSITIONS and portfolio['cash_gbp'] > 50:
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
                if vix is None: vix = 20.0  #safe fallback, closes the meanrev gate
                activity.append(f"VIX {vix:.1f}, meanrev gate {'open' if vix < 18 else 'closed'}")
                #fetch SPY 12-1 momentum once so existing relative momentum does not refetch it per ticker
                try:
                    spy_hist = yf.Ticker('SPY').history(period='1y', auto_adjust=True)
                    if len(spy_hist) >= 240:
                        spy_mom = (float(spy_hist['Close'].iloc[-22]) - float(spy_hist['Close'].iloc[0])) / float(spy_hist['Close'].iloc[0])
                    else:
                        spy_mom = 0.0
                except Exception:
                    spy_mom = 0.0
                activity.append(f"SPY 12-1 momentum {spy_mom*100:+.1f}%")
                pead_hits = 0; mr_hits = 0
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

                    #existing tactical edge: volatility-gated mean reversion
                    if vix < 18:
                        sig = _try_with_retry(check_mean_reversion, t, vix, 18)
                        if sig and sig.get('confidence', 0) >= MIN_CONFIDENCE_TO_OPEN:
                            candidates.append({
                                'ticker': t, 'confidence': sig['confidence'],
                                'strategy': 'meanrev', 'reason': f"mean-reversion {sig.get('direction')} RSI {sig.get('rsi')}"
                            })
                            mr_hits += 1

                    #existing Merlin momentum edge, preserved as its own strategy
                    sig = _try_with_retry(get_momentum_signal, t, spy_mom)
                    if sig and sig.get('confidence', 0) >= MIN_CONFIDENCE_TO_OPEN:
                        candidates.append({
                            'ticker': t, 'confidence': sig['confidence'],
                            'strategy': 'momentum',
                            'reason': f"relative 12-1 momentum {sig.get('mom_12_1_pct', '?')}%, vs SPY {sig.get('relative_mom_vs_spy_pct', '?')}%"
                        })
                        factor_hits['momentum'] = factor_hits.get('momentum', 0) + 1

                    #new research-backed factor edges: stricter 12-1 momentum, 52w high, quality, shareholder yield
                    factor_sigs = _try_with_retry(get_research_factor_signals, t, MIN_CONFIDENCE_TO_OPEN) or []
                    low_beta_sig = _try_with_retry(get_low_beta_trend_signal, t, MIN_CONFIDENCE_TO_OPEN)
                    if low_beta_sig:
                        factor_sigs.append(low_beta_sig)
                    for fs in factor_sigs:
                        candidates.append({
                            'ticker': t, 'confidence': fs['confidence'],
                            'strategy': fs['strategy'], 'reason': fs.get('reason', '')
                        })
                        factor_hits[fs['strategy']] = factor_hits.get(fs['strategy'], 0) + 1

                    if i % 10 == 0 and i > 0: time.sleep(0.3)

                factor_msg = ', '.join([f"{k} {v}" for k, v in factor_hits.items()])
                activity.append(f"scanned {len(universe)} tickers: {pead_hits} pead, {mr_hits} meanrev, {factor_msg} qualify")
            except Exception as e:
                activity.append(f"scan loop failed: {str(e)[:80]}")

            #step 4 rank and open
            paper_cycle_status['message'] = 'opening positions...'
            paper_cycle_status['progress'] = 96
            slots = MAX_POSITIONS - len(portfolio['positions'])
            ranked_candidates = _rank_paper_candidates(candidates, slots,
                                                       held_tickers=set(p['ticker'] for p in portfolio['positions']),
                                                       held_positions=portfolio['positions'])
            counts_msg = {}
            for rc in ranked_candidates:
                counts_msg[rc['strategy']] = counts_msg.get(rc['strategy'], 0) + 1
            activity.append('candidate allocation: ' + ', '.join([f'{k} {v}' for k, v in counts_msg.items()]) if counts_msg else 'candidate allocation: none')
            #note: short interest blocks happen inside _rank_paper_candidates,
            #they're not tracked here per-cycle. could add later if useful.
            seen_tickers = set()
            for cand in ranked_candidates:
                if portfolio['cash_gbp'] < 50: break
                sym = cand['ticker']
                try:
                    entry, currency = _paper_price_and_currency(sym)
                    if entry is None: continue
                    fx = _get_fx_to_gbp(currency)
                    size_gbp = _calculate_position_size_gbp(cand['confidence'], total_equity_gbp, portfolio['cash_gbp'])
                    if size_gbp < 30: continue
                    shares = (size_gbp / fx) / entry
                    cost_gbp = shares * entry * fx
                    portfolio['positions'].append({
                        'ticker': sym, 'strategy': cand['strategy'],
                        'confidence_at_entry': cand['confidence'],
                        'entry_price_native': entry, 'shares': shares,
                        'currency': currency, 'fx_at_entry': fx,
                        'cost_gbp': cost_gbp, 'entry_date': now_iso,
                        'entry_reason': cand.get('reason', ''),
                    })
                    portfolio['cash_gbp'] -= cost_gbp
                    seen_tickers.add(sym)
                    activity.append(f"opened {sym} ({cand['strategy']}) conf {cand['confidence']:.0f} £{cost_gbp:.2f} = {shares:.4f} @ {entry:.2f} {currency}")
                except Exception as e:
                    activity.append(f"failed to open {sym}: {str(e)[:80]}")

        #step 5 snapshot and persist
        paper_cycle_status['message'] = 'finalising...'
        paper_cycle_status['progress'] = 98
        final_pos_value = sum(p.get('current_value_gbp', p.get('cost_gbp', 0)) for p in portfolio['positions'])
        final_equity = portfolio['cash_gbp'] + final_pos_value
        portfolio['equity_curve'].append({'date': now_iso, 'equity_gbp': final_equity})
        portfolio['cycles_run'] += 1
        portfolio['last_cycle'] = now_iso
        prefix = '[auto] ' if triggered_by == 'scheduler' else ''
        if not activity: activity.append(f'{prefix}no actions, no exits and no qualifying signals')
        portfolio['activity_log'] = (portfolio.get('activity_log', []) +
                                      [{'date': now_iso, 'event': prefix + a} for a in activity])[-200:]
        _save_paper_portfolio(portfolio)
        result = {'success': True, 'equity_gbp': final_equity,
                  'cash_gbp': portfolio['cash_gbp'],
                  'positions_count': len(portfolio['positions']),
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
    portfolio = _load_paper_portfolio()
    total_value_gbp = 0.0
    for pos in portfolio['positions']:
        try:
            price, _ = _paper_price_and_currency(pos['ticker'])
            if price is None: continue
            pos['current_price_native'] = price
            fx = _get_fx_to_gbp(pos['currency'])
            pos['current_value_gbp'] = pos['shares'] * price * fx
            pos['unrealised_pl_gbp'] = pos['current_value_gbp'] - pos['cost_gbp']
            pos['unrealised_pl_pct'] = (price - pos['entry_price_native']) / pos['entry_price_native'] * 100
            total_value_gbp += pos['current_value_gbp']
        except Exception: continue
    equity_gbp = portfolio['cash_gbp'] + total_value_gbp
    return jsonify(sanitise({**portfolio, 'equity_gbp': equity_gbp, 'position_value_gbp': total_value_gbp}))


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
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    if not _paper_scheduler.running:
        _paper_scheduler.start()
        atexit.register(lambda: _paper_scheduler.shutdown(wait=False))
        print('\n[scheduler] paper trader scheduler started, daily cycle at 22:00 local time')


if __name__=='__main__':
    '''
    debug=True gives template auto-refresh and good tracebacks but the
    Werkzeug reloader on Windows uses watchdog which fires false positives
    on yfinance cache writes inside site-packages. that kills any long
    running background task mid-scan (mean reversion, AI paper cycle).
    exclude_patterns stops the reloader watching anything outside the
    project directory, which is what we actually want.
    '''
    app.run(
        debug=True,
        port=5000,
        exclude_patterns=[
            '*/site-packages/*', '*\\site-packages\\*',
            '*/AppData/*',       '*\\AppData\\*',
            '*/__pycache__/*',   '*\\__pycache__\\*',
            '*.pyc'
        ]
    )

