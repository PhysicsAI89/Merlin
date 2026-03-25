'''
merlin 5.5 - ensemble prediction engine
fundamentals dashboard, ai valuation assessment, screener, insider tracking
'''

import os, json, datetime, re, time, threading, warnings
import numpy as np
import pandas as pd
import yfinance as yf
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

#fix nan/numpy values breaking json serialisation
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

def sanitise_for_json(obj):
    '''recursively replace nan/inf with None in dicts and lists'''
    if isinstance(obj, dict):
        return {k: sanitise_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitise_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if obj != obj or obj == float('inf') or obj == float('-inf'):
            return None
        return obj
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
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


# ==================== FUNDAMENTALS ====================

def get_fundamentals(ticker):
    '''
    pull key fundamental metrics from yfinance and generate
    a plain-english ai assessment of valuation and strength
    '''
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', info.get('forwardPE', None))
        forward_pe = info.get('forwardPE', None)
        peg_ratio = info.get('pegRatio', None)
        beta = info.get('beta', None)
        dividend_yield = info.get('dividendYield', 0)
        avg_volume = info.get('averageVolume', 0)
        avg_volume_10d = info.get('averageVolume10days', 0)
        fifty_two_high = info.get('fiftyTwoWeekHigh', 0)
        fifty_two_low = info.get('fiftyTwoWeekLow', 0)
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        revenue_growth = info.get('revenueGrowth', None)
        profit_margins = info.get('profitMargins', None)
        debt_to_equity = info.get('debtToEquity', None)
        return_on_equity = info.get('returnOnEquity', None)
        free_cashflow = info.get('freeCashflow', 0)
        total_revenue = info.get('totalRevenue', 0)
        sector = info.get('sector', 'unknown')
        industry = info.get('industry', 'unknown')
        name = info.get('shortName', info.get('longName', ticker))
        earnings_growth = info.get('earningsGrowth', None)
        book_value = info.get('bookValue', None)
        price_to_book = info.get('priceToBook', None)

        #calculate additional metrics
        if fifty_two_high and fifty_two_low and current_price:
            range_position = (current_price - fifty_two_low) / (fifty_two_high - fifty_two_low + 1e-10)
        else:
            range_position = 0.5

        #format market cap
        if market_cap >= 1e12:
            mc_str = f'${market_cap/1e12:.2f}T'
            mc_class = 'mega-cap'
        elif market_cap >= 1e9:
            mc_str = f'${market_cap/1e9:.2f}B'
            mc_class = 'large-cap' if market_cap >= 10e9 else 'mid-cap'
        elif market_cap >= 1e6:
            mc_str = f'${market_cap/1e6:.1f}M'
            mc_class = 'small-cap'
        else:
            mc_str = 'n/a'
            mc_class = 'unknown'

        #format avg volume
        if avg_volume >= 1e6:
            vol_str = f'{avg_volume/1e6:.1f}M'
        elif avg_volume >= 1e3:
            vol_str = f'{avg_volume/1e3:.0f}K'
        else:
            vol_str = str(avg_volume)

        fundamentals = {
            'name': name,
            'sector': sector,
            'industry': industry,
            'market_cap': market_cap,
            'market_cap_str': mc_str,
            'market_cap_class': mc_class,
            'pe_ratio': round(pe_ratio, 2) if pe_ratio else None,
            'forward_pe': round(forward_pe, 2) if forward_pe else None,
            'peg_ratio': round(peg_ratio, 2) if peg_ratio else None,
            'beta': round(beta, 2) if beta else None,
            'dividend_yield': round(dividend_yield * 100, 2) if dividend_yield else 0,
            'avg_volume': avg_volume,
            'avg_volume_str': vol_str,
            'fifty_two_high': round(fifty_two_high, 2) if fifty_two_high else None,
            'fifty_two_low': round(fifty_two_low, 2) if fifty_two_low else None,
            'range_position': round(range_position * 100, 1),
            'revenue_growth': round(revenue_growth * 100, 1) if revenue_growth else None,
            'profit_margins': round(profit_margins * 100, 1) if profit_margins else None,
            'debt_to_equity': round(debt_to_equity, 1) if debt_to_equity else None,
            'return_on_equity': round(return_on_equity * 100, 1) if return_on_equity else None,
            'free_cashflow': free_cashflow,
            'earnings_growth': round(earnings_growth * 100, 1) if earnings_growth else None,
            'price_to_book': round(price_to_book, 2) if price_to_book else None,
        }

        #generate ai assessment
        fundamentals['assessment'] = _assess_fundamentals(fundamentals, ticker)

        return fundamentals

    except Exception as e:
        return {'error': str(e), 'assessment': {'verdict': 'unknown', 'summary': 'could not fetch fundamental data', 'points': []}}


def _assess_fundamentals(f, ticker):
    '''
    generate a plain-english assessment of whether the stock
    looks overvalued, undervalued or fairly priced based on
    its fundamental metrics. not actual financial advice obviously.
    '''
    points = []
    score = 0  #positive = bullish, negative = bearish

    #p/e analysis
    pe = f.get('pe_ratio')
    fpe = f.get('forward_pe')
    if pe:
        if pe < 0:
            points.append({'type': 'warning', 'text': f'negative P/E ({pe}) means the company is currently losing money'})
            score -= 2
        elif pe < 12:
            points.append({'type': 'bullish', 'text': f'P/E of {pe} is low which could suggest undervaluation or a mature slow-growth business'})
            score += 2
        elif pe < 20:
            points.append({'type': 'neutral', 'text': f'P/E of {pe} is reasonable for an established company'})
            score += 1
        elif pe < 35:
            points.append({'type': 'neutral', 'text': f'P/E of {pe} is above average which suggests investors expect solid growth'})
        elif pe < 60:
            points.append({'type': 'warning', 'text': f'P/E of {pe} is high. the stock needs to deliver strong growth to justify this valuation'})
            score -= 1
        else:
            points.append({'type': 'bearish', 'text': f'P/E of {pe} is very high. this is priced for perfection and any earnings miss could cause a sharp drop'})
            score -= 2

        if fpe and pe:
            if fpe < pe * 0.8:
                points.append({'type': 'bullish', 'text': f'forward P/E of {fpe} is well below trailing P/E suggesting earnings are expected to grow'})
                score += 1
            elif fpe > pe * 1.2:
                points.append({'type': 'warning', 'text': f'forward P/E of {fpe} is above trailing P/E suggesting earnings may decline'})
                score -= 1

    #peg ratio
    peg = f.get('peg_ratio')
    if peg:
        if peg < 1:
            points.append({'type': 'bullish', 'text': f'PEG ratio of {peg} (under 1.0) suggests the stock is undervalued relative to its growth rate'})
            score += 2
        elif peg < 1.5:
            points.append({'type': 'neutral', 'text': f'PEG ratio of {peg} suggests fair value relative to growth'})
        elif peg > 2:
            points.append({'type': 'bearish', 'text': f'PEG ratio of {peg} suggests the stock may be overpriced for its growth rate'})
            score -= 1

    #beta
    beta = f.get('beta')
    if beta:
        if beta > 1.5:
            points.append({'type': 'warning', 'text': f'beta of {beta} means this stock is significantly more volatile than the market. expect bigger swings in both directions'})
        elif beta > 1.1:
            points.append({'type': 'neutral', 'text': f'beta of {beta} means slightly more volatile than the market'})
        elif beta >= 0.8:
            points.append({'type': 'neutral', 'text': f'beta of {beta} means it tracks the market fairly closely'})
        elif beta >= 0:
            points.append({'type': 'bullish', 'text': f'beta of {beta} means lower volatility than the market which is good for stability'})

    #profit margins
    margins = f.get('profit_margins')
    if margins is not None:
        if margins > 20:
            points.append({'type': 'bullish', 'text': f'profit margins of {margins}% are strong which indicates good pricing power'})
            score += 1
        elif margins > 10:
            points.append({'type': 'neutral', 'text': f'profit margins of {margins}% are decent'})
        elif margins > 0:
            points.append({'type': 'warning', 'text': f'profit margins of {margins}% are thin. the company doesn\'t have much room for error'})
            score -= 1
        else:
            points.append({'type': 'bearish', 'text': f'negative profit margins of {margins}% mean the company is not profitable'})
            score -= 2

    #revenue growth
    rg = f.get('revenue_growth')
    if rg is not None:
        if rg > 20:
            points.append({'type': 'bullish', 'text': f'revenue growth of {rg}% is strong'})
            score += 1
        elif rg > 5:
            points.append({'type': 'neutral', 'text': f'revenue growth of {rg}% is steady'})
        elif rg > 0:
            points.append({'type': 'warning', 'text': f'revenue growth of {rg}% is slow'})
        else:
            points.append({'type': 'bearish', 'text': f'revenue is declining at {rg}%'})
            score -= 1

    #return on equity
    roe = f.get('return_on_equity')
    if roe is not None:
        if roe > 20:
            points.append({'type': 'bullish', 'text': f'return on equity of {roe}% is excellent which means the company is efficient at generating profit from shareholders\' investment'})
            score += 1
        elif roe > 10:
            points.append({'type': 'neutral', 'text': f'return on equity of {roe}% is respectable'})
        elif roe > 0:
            points.append({'type': 'warning', 'text': f'return on equity of {roe}% is below average'})
        else:
            points.append({'type': 'bearish', 'text': f'negative return on equity of {roe}% is concerning'})
            score -= 1

    #debt to equity
    dte = f.get('debt_to_equity')
    if dte is not None:
        if dte > 200:
            points.append({'type': 'bearish', 'text': f'debt-to-equity of {dte} is very high. the company is heavily leveraged'})
            score -= 2
        elif dte > 100:
            points.append({'type': 'warning', 'text': f'debt-to-equity of {dte} is elevated. worth keeping an eye on'})
            score -= 1
        elif dte > 50:
            points.append({'type': 'neutral', 'text': f'debt-to-equity of {dte} is manageable'})
        else:
            points.append({'type': 'bullish', 'text': f'debt-to-equity of {dte} is low. the company has a strong balance sheet'})
            score += 1

    #dividend
    div = f.get('dividend_yield', 0)
    if div > 4:
        points.append({'type': 'bullish', 'text': f'dividend yield of {div}% is high which provides good income'})
        score += 0.5
    elif div > 1.5:
        points.append({'type': 'neutral', 'text': f'dividend yield of {div}% provides a reasonable income stream'})
    elif div > 0:
        points.append({'type': 'neutral', 'text': f'small dividend yield of {div}%'})

    #52-week range position
    rp = f.get('range_position', 50)
    if rp > 90:
        points.append({'type': 'warning', 'text': f'trading near 52-week high ({rp:.0f}% of range). could be running hot'})
        score -= 0.5
    elif rp < 15:
        points.append({'type': 'bullish', 'text': f'trading near 52-week low ({rp:.0f}% of range). could be a value opportunity or there\'s a reason it\'s down'})
        score += 0.5

    #overall verdict
    if score >= 3:
        verdict = 'undervalued'
        summary = f'the fundamentals look strong for {ticker}. the combination of metrics suggests this stock could be undervalued at its current price'
    elif score >= 1:
        verdict = 'fair_value'
        summary = f'{ticker} appears to be reasonably priced based on its fundamentals. no major red flags but no screaming bargain either'
    elif score >= -1:
        verdict = 'fair_value'
        summary = f'{ticker} is roughly fairly valued. there are some positive and negative signals that largely balance out'
    elif score >= -3:
        verdict = 'overvalued'
        summary = f'{ticker} looks like it could be overpriced relative to its fundamentals. the metrics suggest caution'
    else:
        verdict = 'overvalued'
        summary = f'the fundamentals for {ticker} raise several concerns. the stock appears overvalued or the company is facing significant challenges'

    #fundamental strength
    strength_score = 0
    if margins and margins > 15: strength_score += 1
    if rg and rg > 10: strength_score += 1
    if roe and roe > 15: strength_score += 1
    if dte is not None and dte < 80: strength_score += 1
    if f.get('free_cashflow', 0) > 0: strength_score += 1

    if strength_score >= 4:
        strength = 'strong'
        strength_text = 'this company has strong fundamentals with solid profitability, growth and a healthy balance sheet'
    elif strength_score >= 2:
        strength = 'moderate'
        strength_text = 'the company has decent fundamentals with some strengths but also areas that could improve'
    else:
        strength = 'weak'
        strength_text = 'the company has weak fundamentals. profitability, growth or financial health are concerning'

    return {
        'verdict': verdict,
        'summary': summary,
        'score': round(score, 1),
        'strength': strength,
        'strength_text': strength_text,
        'points': points
    }


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
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
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
    'Day_Sin','Day_Cos','Month_Sin','Month_Cos'
]

def prepare_features(df):
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].replace([np.inf,-np.inf],0).fillna(0).copy(), available


# ==================== NEWS & INSIDER ====================

def get_news_sentiment(ticker):
    try:
        raw = yf.Ticker(ticker).news
        if not raw: return {'articles':[],'overall_score':0,'summary':'no recent news found','count':0}
        pos_w = {'surge','surges','soar','jump','jumps','gain','gains','rally','rise','rises','climb','high','record','beat','beats','strong','bullish','growth','profit','upgrade','buy','outperform','positive','boost','recovery','optimistic','breakthrough','expansion','earnings','dividend','approval','partnership','deal','success','win','higher','upside','above'}
        neg_w = {'drop','drops','fall','falls','decline','plunge','crash','sink','down','low','loss','losses','miss','weak','bearish','sell','downgrade','negative','cut','slash','warning','risk','fear','concern','recession','lawsuit','investigation','penalty','recall','bankruptcy','layoff','layoffs','debt','deficit','lower','below','worst','crisis','slump','tumble'}
        articles, total = [], 0
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
    except: return {'articles':[],'overall_score':0,'summary':'could not fetch','count':0}


def get_insider_activity(ticker):
    try:
        stock = yf.Ticker(ticker)
        transactions = []
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
        ab=sum(1 for t in transactions if t['action']=='buy')
        asl=sum(1 for t in transactions if t['action']=='sell')
        isent='bullish' if eb>es else 'bearish' if es>eb else 'neutral'
        isc=min(eb-es,3) if eb>es else -min(es-eb,3) if es>eb else 0
        return {'transactions':transactions,'exec_buys':eb,'exec_sells':es,'all_buys':ab,'all_sells':asl,'sentiment':isent,'score':isc}
    except: return {'transactions':[],'exec_buys':0,'exec_sells':0,'all_buys':0,'all_sells':0,'sentiment':'unknown','score':0}


# ==================== SCREENER ====================

STOCK_UNIVERSE = [
    'AAPL','MSFT','AMZN','NVDA','GOOGL','META','TSLA','BRK-B','UNH','XOM',
    'JNJ','JPM','V','PG','MA','AVGO','HD','CVX','MRK','ABBV',
    'LLY','COST','PEP','KO','ADBE','WMT','MCD','CSCO','CRM','ACN',
    'TMO','ABT','DHR','NKE','NEE','LIN','TXN','PM','UNP','RTX',
    'LOW','QCOM','HON','INTC','INTU','AMAT','ISRG','AMGN','BKNG','GS',
    'CAT','BLK','AXP','BA','SBUX','GE','IBM','DIS','AMD','PYPL',
    'SHOP','PLTR','COIN','SOFI','F','GM','T','VZ','NFLX','UBER',
    'ABNB','RBLX','DDOG','NET','CRWD','PANW','NOW','WDAY','SNAP','MELI',
    'BABA','JD','NIO','TSM','SONY','BP','SHEL','GOLD','NEM','FCX',
    'WFC','BAC','C','MS','SCHW','PFE','BMY','GILD','MRNA','VRTX',
    'SYK','MDT','CI','CVS','DE','ENPH','ARM','SMCI','MU','MRVL',
    'ON','KLAC','LRCX','ASML','WM','SHW','APD','PSA','O','VICI',
    'ED','SO','DUK','AEP','XEL','DKNG','MGM','AMC','GME','CELH',
    'MNST','BROS','SPY','QQQ','DIA','IWM','ARKK','XLF','XLE','XLV'
]


def quick_score_stock(ticker):
    try:
        data = yf.download(ticker, period='6mo', interval='1d', progress=False)
        if data.empty or len(data) < 50: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data = compute_indicators(data)
        c = float(data['Close'].iloc[-1])
        score, signals = 0, []

        rsi = float(data['RSI'].iloc[-1])
        if rsi < 30: score += 2; signals.append('RSI oversold')
        elif rsi > 70: score -= 2; signals.append('RSI overbought')

        macd,ms = float(data['MACD'].iloc[-1]), float(data['MACD_Signal'].iloc[-1])
        mp,msp = float(data['MACD'].iloc[-2]), float(data['MACD_Signal'].iloc[-2])
        if macd>ms and mp<=msp: score += 2; signals.append('MACD bullish cross')
        elif macd<ms and mp>=msp: score -= 2; signals.append('MACD bearish cross')
        elif macd>ms: score += 0.5

        s5,s20,s50 = float(data['SMA_5'].iloc[-1]), float(data['SMA_20'].iloc[-1]), float(data['SMA_50'].iloc[-1])
        if s5>s20>s50: score += 2; signals.append('bullish SMA alignment')
        elif s5<s20<s50: score -= 2; signals.append('bearish SMA alignment')
        if c>s20: score += 0.5
        if c>s50: score += 0.5

        bbp = float(data['BB_Position'].iloc[-1])
        if bbp < 0.1: score += 1.5; signals.append('near lower BB')
        elif bbp > 0.9: score -= 1.5

        m5 = float(data['Momentum_5'].iloc[-1])
        if m5 > 0.02: score += 1; signals.append('positive momentum')
        elif m5 < -0.02: score -= 1

        vr = float(data['Vol_Ratio'].iloc[-1])
        if vr > 1.5 and m5 > 0: score += 1; signals.append('volume breakout')

        return {
            'ticker': ticker, 'price': round(c, 2),
            'change_1d': round(float(data['Returns'].iloc[-1])*100, 2),
            'change_5d': round(float(data['Returns_5d'].iloc[-1])*100, 2),
            'rsi': round(rsi, 1), 'score': round(score, 1),
            'signals': signals[:4],
            'direction': 'bullish' if score > 1 else 'bearish' if score < -1 else 'neutral'
        }
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
def index():
    return render_template('index.html')


@app.route('/api/fetch_data', methods=['POST'])
def fetch_data():
    try:
        ticker = request.json.get('ticker','').upper().strip()
        if not ticker: return jsonify({'error':'no ticker'}), 400

        data = yf.download(ticker, period='10y', interval='1d', progress=False)
        if data.empty or len(data)<200: return jsonify({'error':f'not enough data for {ticker}'}), 400
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

        data = compute_indicators(data)
        data.to_csv(os.path.join(DATA_DIR, f'{ticker}.csv'))

        chart_data = {
            'dates':[d.strftime('%Y-%m-%d') for d in data.index],
            'close':[round(float(v),2) for v in data['Close'].values],
            'volume':[int(v) for v in data['Volume'].values],
            'high':[round(float(v),2) for v in data['High'].values],
            'low':[round(float(v),2) for v in data['Low'].values],
            'open':[round(float(v),2) for v in data['Open'].values],
            'sma_20':[round(float(v),2) if not np.isnan(v) else None for v in data['SMA_20'].values],
            'sma_50':[round(float(v),2) if not np.isnan(v) else None for v in data['SMA_50'].values],
        }

        cp = round(float(data['Close'].iloc[-1]),2)
        pp = round(float(data['Close'].iloc[-2]),2)
        ch = round(cp-pp,2)
        chp = round((ch/pp)*100,2)

        return jsonify(sanitise_for_json({
            'ticker':ticker, 'current_price':cp, 'change':ch, 'change_pct':chp,
            'data_points':len(data), 'chart_data':chart_data,
            'news':get_news_sentiment(ticker),
            'insider':get_insider_activity(ticker),
            'fundamentals':get_fundamentals(ticker),
            'date_range':{'start':data.index[0].strftime('%Y-%m-%d'),'end':data.index[-1].strftime('%Y-%m-%d')}
        }))
    except Exception as e:
        return jsonify({'error':str(e)}), 500


@app.route('/api/screener', methods=['POST'])
def run_screener():
    global screener_status
    if screener_status['active']: return jsonify({'error':'already running'}), 400
    top_n = request.json.get('top_n',10)
    count = request.json.get('stock_count',200)
    screener_status = {'active':True,'progress':0,'message':'starting...','complete':False,'results':[],'error':None}
    thread = threading.Thread(target=_run_screener, args=(top_n,count)); thread.daemon=True; thread.start()
    return jsonify({'status':'started'})

def _run_screener(top_n, count):
    global screener_status
    try:
        stocks = STOCK_UNIVERSE[:min(count,len(STOCK_UNIVERSE))]
        total = len(stocks); results = []
        for i,t in enumerate(stocks):
            screener_status['progress'] = int((i/total)*90)
            screener_status['message'] = f'scanning {t} ({i+1}/{total})...'
            r = quick_score_stock(t)
            if r: results.append(r)
            if i%10==0 and i>0: time.sleep(0.5)

        results.sort(key=lambda x: x['score'], reverse=True)
        screener_status['message'] = 'fetching details for top picks...'
        screener_status['progress'] = 92

        top = results[:top_n]
        for r in top:
            try:
                news = get_news_sentiment(r['ticker'])
                r['news_score'] = news.get('overall_score',0)
                ins = get_insider_activity(r['ticker'])
                r['insider_sentiment'] = ins.get('sentiment','unknown')
                r['insider_score'] = ins.get('score',0)
                r['exec_buys'] = ins.get('exec_buys',0)
                r['exec_sells'] = ins.get('exec_sells',0)
                #fundamentals for screener
                fund = get_fundamentals(r['ticker'])
                r['pe_ratio'] = fund.get('pe_ratio')
                r['market_cap_str'] = fund.get('market_cap_str','?')
                r['beta'] = fund.get('beta')
                r['dividend_yield'] = fund.get('dividend_yield',0)
                r['valuation'] = fund.get('assessment',{}).get('verdict','unknown')
                r['strength'] = fund.get('assessment',{}).get('strength','unknown')
                r['combined_score'] = round(r['score'] + r['news_score']*0.5 + r['insider_score']*0.8, 1)
                time.sleep(0.3)
            except:
                r['combined_score'] = r['score']

        top.sort(key=lambda x: x.get('combined_score', x['score']), reverse=True)
        screener_status['results'] = top
        screener_status['total_scanned'] = total
        screener_status['progress'] = 100
        screener_status['complete'] = True
        screener_status['active'] = False
        screener_status['message'] = f'done! scanned {total} stocks.'
    except Exception as e:
        screener_status['error'] = str(e)
        screener_status['active'] = False

@app.route('/api/screener_status')
def get_screener_status():
    return jsonify(sanitise_for_json(screener_status))


@app.route('/api/train', methods=['POST'])
def train_route():
    global training_status
    if training_status['active']: return jsonify({'error':'already training'}), 400
    ticker = request.json.get('ticker','').upper().strip()
    epochs = request.json.get('epochs',50)
    if not ticker: return jsonify({'error':'no ticker'}), 400
    if not os.path.exists(os.path.join(DATA_DIR, f'{ticker}.csv')): return jsonify({'error':'fetch data first'}), 400
    training_status = {'active':True,'progress':0,'message':'starting...','ticker':ticker,'complete':False,'error':None,'backtest':None}
    thread = threading.Thread(target=_train_ensemble, args=(ticker,epochs)); thread.daemon=True; thread.start()
    return jsonify({'status':'started','ticker':ticker})

def _train_ensemble(ticker, epochs):
    global training_status
    try:
        data = pd.read_csv(os.path.join(DATA_DIR,f'{ticker}.csv'), index_col=0, parse_dates=True)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        features_df, feature_cols = prepare_features(data)
        nf = len(feature_cols); ci = feature_cols.index('Close')
        cp = features_df['Close'].values.astype(float)
        rets = np.clip(np.diff(cp)/(cp[:-1]+1e-10), -0.15, 0.15)
        scaler = RobustScaler(); sd = scaler.fit_transform(features_df.values)

        X,y = [],[]
        for i in range(SEQUENCE_LENGTH, len(sd)-1):
            X.append(sd[i-SEQUENCE_LENGTH:i]); y.append(rets[i-1])
        X,y = np.array(X), np.array(y)

        total=len(X); ts=int(total*0.2); trs=total-ts
        Xtr,Xte,ytr,yte = X[:trs],X[trs:],y[:trs],y[trs:]
        vs=int(trs*0.85); Xt,Xv,yt,yv = Xtr[:vs],Xtr[vs:],ytr[:vs],ytr[vs:]

        training_status['progress']=10
        mr,etp = [],[]
        for i in range(NUM_ENSEMBLE):
            mn=MODEL_NAMES[i]; training_status['message']=f'training {mn} ({i+1}/3)...'
            model=MODEL_BUILDERS[i](SEQUENCE_LENGTH,nf)
            es=EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True)
            rlr=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,min_lr=1e-6)
            for epoch in range(epochs):
                h=model.fit(Xt,yt,epochs=1,batch_size=32,validation_data=(Xv,yv),callbacks=[es,rlr],verbose=0)
                training_status['progress']=min(10+i*25+int((epoch+1)/epochs*22),85)
                training_status['message']=f'{mn} | epoch {epoch+1}/{epochs} | loss: {h.history["loss"][0]:.6f}'
                if es.stopped_epoch>0: break
            model.save(os.path.join(MODELS_DIR,f'{ticker}_model_{i}.keras'))
            tp=model.predict(Xte,verbose=0).flatten(); etp.append(tp)
            da=np.mean((tp>0)==(yte>0))*100
            mr.append({'name':mn,'direction_accuracy':round(float(da),1)})

        eavg=np.mean(etp,axis=0); evote=np.sign(np.sum([np.sign(p) for p in etp],axis=0))
        eda=np.mean((evote>0)==(yte>0))*100
        tcp=cp[trs+SEQUENCE_LENGTH:trs+SEQUENCE_LENGTH+len(yte)]
        pp=tcp*(1+eavg); ap=tcp*(1+yte)
        cl=min(60,len(ap))
        bt={'ensemble_direction_accuracy':round(float(eda),1),
            'mae':round(float(mean_absolute_error(ap,pp)),2),
            'rmse':round(float(np.sqrt(mean_squared_error(ap,pp))),2),
            'mape':round(float(np.mean(np.abs((ap-pp)/(ap+1e-10)))*100),2),
            'individual_models':mr,
            'val_actual':[round(float(p),2) for p in ap[-cl:]],
            'val_predictions':[round(float(p),2) for p in pp[-cl:]]}

        meta={'feature_cols':feature_cols,'num_features':nf,'close_idx':ci,
              'scaler':{'center':scaler.center_.tolist(),'scale':scaler.scale_.tolist()},
              'backtest':bt,'model_names':MODEL_NAMES[:NUM_ENSEMBLE],'num_models':NUM_ENSEMBLE}
        with open(os.path.join(MODELS_DIR,f'{ticker}_meta.json'),'w') as f: json.dump(meta,f)

        training_status.update({'message':'complete!','progress':100,'complete':True,'active':False,'backtest':bt})
    except Exception as e:
        import traceback; traceback.print_exc()
        training_status.update({'error':str(e),'active':False})

@app.route('/api/training_status')
def get_training_status():
    return jsonify(sanitise_for_json(training_status))


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        ticker=request.json.get('ticker','').upper().strip(); tf=request.json.get('timeframe','1d')
        if not ticker: return jsonify({'error':'no ticker'}),400
        mp=os.path.join(MODELS_DIR,f'{ticker}_meta.json')
        if not os.path.exists(mp): return jsonify({'error':'train first'}),400
        with open(mp) as f: meta=json.load(f)
        fc=meta['feature_cols']; nm=meta['num_models']
        models=[load_model(os.path.join(MODELS_DIR,f'{ticker}_model_{i}.keras')) for i in range(nm)]
        scaler=RobustScaler(); scaler.center_=np.array(meta['scaler']['center']); scaler.scale_=np.array(meta['scaler']['scale'])

        rd=pd.read_csv(os.path.join(DATA_DIR,f'{ticker}.csv'),index_col=0,parse_dates=True)
        if isinstance(rd.columns,pd.MultiIndex): rd.columns=rd.columns.get_level_values(0)
        buf=rd[['Open','High','Low','Close','Volume']].copy()
        cp=float(rd['Close'].iloc[-1]); dv=float(rd['Close'].pct_change().dropna().std())
        tfm={'1d':1,'1w':5,'1m':22,'3m':66,'6m':132,'1y':252}; days=tfm.get(tf,1)

        amp=[[] for _ in range(nm)]; ep=[]; ub=[]; lb=[]
        brmse=meta.get('backtest',{}).get('rmse',cp*0.02)

        for day in range(days):
            tdf=compute_indicators(buf.copy()); tfdf,_=prepare_features(tdf)
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
        analysis=_analyse(cp,ep,pd_dates,news,ins,meta.get('backtest',{}),mv,dv)

        return jsonify(sanitise_for_json({'ticker':ticker,'timeframe':tf,'current_price':cp,
            'predictions':{'dates':pd_dates,'prices':ep,'upper_band':ub,'lower_band':lb},
            'model_votes':mv,'analysis':analysis,'news_sentiment':news,'insider':ins}))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error':str(e)}),500


def _analyse(cp,pp,pd,news,ins,bt,mv,dv):
    if not pp: return {'action':'hold','reason':'no predictions'}
    mnp,mxp=min(pp),max(pp); mni,mxi=pp.index(mnp),pp.index(mxp); fp=pp[-1]
    oc=round(((fp-cp)/cp)*100,2)
    da=bt.get('ensemble_direction_accuracy',50); conf='low' if da<55 else 'medium' if da<65 else 'high'
    vu=sum(1 for v in mv if v['direction']=='up'); cons=f'{vu}/{len(mv)} models say up'
    ns=news.get('overall_score',0) if news else 0; nb='positive' if ns>0.3 else 'negative' if ns<-0.3 else 'neutral'
    isent=ins.get('sentiment','unknown') if ins else 'unknown'; eb=ins.get('exec_buys',0); es_=ins.get('exec_sells',0)
    result={'current_price':cp,'predicted_low':round(mnp,2),'predicted_low_date':pd[mni],'predicted_high':round(mxp,2),'predicted_high_date':pd[mxi],'final_predicted_price':round(fp,2),'overall_change_pct':oc,'model_confidence':conf,'direction_accuracy':da,'news_bias':nb,'news_score':ns,'insider_sentiment':isent,'insider_score':ins.get('score',0) if ins else 0,'exec_buys':eb,'exec_sells':es_,'consensus':cons,'votes_up':vu,'votes_down':len(mv)-vu,'daily_volatility':round(dv*100,2)}
    threshold=0.015 if conf=='high' else 0.02 if conf=='medium' else 0.03
    reasons=[]
    if mni<mxi:
        sw=(mxp-mnp)/mnp
        if sw>=threshold:
            result.update({'action':'buy_then_sell','buy_date':pd[mni],'buy_price':round(mnp,2),'sell_date':pd[mxi],'sell_price':round(mxp,2),'potential_profit_pct':round(sw*100,2)})
            reasons.append(f'{cons}. predicted dip to ${mnp:.2f} on {pd[mni]} then rise to ${mxp:.2f} on {pd[mxi]} ({sw*100:.1f}% gain)')
            if nb=='positive': reasons.append('news supports this')
            elif nb=='negative': reasons.append('news is negative so be cautious')
            if isent=='bullish': reasons.append(f'insiders are buying ({eb} exec buys vs {es_} sells)')
            elif isent=='bearish': reasons.append(f'but insiders are selling ({es_} exec sells)')
            reasons.append(f'ensemble accuracy: {da:.0f}% ({conf})'); result['reason']='. '.join(reasons); return result
    if mxi<mni:
        dr=(mxp-mnp)/mxp
        if dr>=threshold:
            result.update({'action':'sell_then_buy','sell_date':pd[mxi],'sell_price':round(mxp,2),'buy_date':pd[mni],'buy_price':round(mnp,2),'potential_profit_pct':round(dr*100,2)})
            reasons.append(f'{cons}. peak at ${mxp:.2f} on {pd[mxi]} then drop to ${mnp:.2f} on {pd[mni]}')
            if isent=='bearish': reasons.append(f'insiders selling too ({es_} exec sells)')
            reasons.append(f'ensemble accuracy: {da:.0f}% ({conf})'); result['reason']='. '.join(reasons); return result
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

if __name__=='__main__':
    app.run(debug=True, port=5000)
