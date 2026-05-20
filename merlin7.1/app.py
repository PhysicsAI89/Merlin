'''
merlin 7.1 - ensemble prediction engine
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




if __name__=='__main__':
    app.run(debug=True, port=5000)
