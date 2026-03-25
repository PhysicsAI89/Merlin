'''
merlin 5.4 - ensemble prediction engine with stock screener and insider tracking
scans up to 2000 stocks, ranks by technical strength + news + insider activity,
shows ceo/cfo buying and selling data
'''

import os
import json
import datetime
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
import threading
import re
import warnings
import time
warnings.filterwarnings('ignore')

app = Flask(__name__)

training_status = {
    'active': False, 'progress': 0, 'message': '',
    'ticker': '', 'complete': False, 'error': None, 'backtest': None
}

screener_status = {
    'active': False, 'progress': 0, 'message': '',
    'complete': False, 'results': [], 'error': None
}

trade_recommendations = []

MODELS_DIR = 'models'
DATA_DIR = 'data'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

SEQUENCE_LENGTH = 60
NUM_ENSEMBLE = 3


#s&p 500 + popular stocks (trimmed for speed, expandable)
STOCK_UNIVERSE = [
    'AAPL','MSFT','AMZN','NVDA','GOOGL','META','TSLA','BRK-B','UNH','XOM',
    'JNJ','JPM','V','PG','MA','AVGO','HD','CVX','MRK','ABBV',
    'LLY','COST','PEP','KO','ADBE','WMT','MCD','CSCO','CRM','ACN',
    'TMO','ABT','DHR','NKE','NEE','LIN','TXN','PM','UNP','RTX',
    'LOW','QCOM','HON','INTC','INTU','AMAT','ISRG','AMGN','BKNG','GS',
    'CAT','BLK','AXP','BA','SBUX','GE','MMM','IBM','DIS','AMD',
    'PYPL','SQ','SHOP','SNOW','PLTR','COIN','RIVN','LCID','SOFI','HOOD',
    'F','GM','T','VZ','CMCSA','NFLX','ROKU','SPOT','ZM','UBER',
    'ABNB','DASH','RBLX','U','DDOG','NET','CRWD','ZS','OKTA','PANW',
    'NOW','WDAY','TEAM','HUBS','TTD','SNAP','PINS','TWLO','SQ','MELI',
    'SE','BABA','JD','PDD','NIO','XPEV','LI','TSM','SONY','TM',
    'BP','SHEL','RIO','VALE','BHP','GOLD','NEM','FCX','CLF','AA',
    'X','STLD','NUE','CMC','WFC','BAC','C','MS','SCHW','USB',
    'PNC','TFC','AIG','MET','PRU','ALL','TRV','CB','AFL','PFE',
    'BMY','GILD','BIIB','REGN','VRTX','MRNA','ZTS','SYK','BDX','MDT',
    'EW','BSX','CI','HUM','CNC','ELV','CVS','WBA','MCK','CAH',
    'DE','CNH','AGCO','FMC','MOS','CF','ADM','BG','TSN','HRL',
    'SJM','GIS','K','CPB','CAG','CLX','CL','EL','PG','CHD',
    'PLUG','FCEL','ENPH','SEDG','RUN','NOVA','SPWR','BE','CHPT','BLNK',
    'RIVN','LCID','FSR','GOEV','WKHS','RIDE','NKLA','HYLN','XL','ARVL',
    'SPY','QQQ','DIA','IWM','VTI','VOO','ARKK','XLF','XLE','XLV',
    'DKNG','PENN','MGM','WYNN','LVS','CZR','RSI','GENI','SKLZ',
    'AMC','GME','BB','BBBY','CLOV','WISH','SOFI','OPEN','UWMC','RKT',
    'ASTS','MNST','CELH','OLPX','BIRD','FIGS','BROS','DTC','OATLY',
    'ARM','SMCI','MU','MRVL','ON','SWKS','MCHP','KLAC','LRCX','ASML',
    'WM','RSG','WCN','ECL','SHW','APD','DD','EMN','PPG','BALL',
    'PKG','IP','WRK','AVY','SEE','BLL','AMCR','CCK','OI','GPK',
    'PSA','EQR','AVB','UDR','CPT','MAA','ESS','INVH','AMH','SUI',
    'O','NNN','WPC','STOR','ADC','VICI','GLPI','MGP','RHP','PK',
    'ED','SO','DUK','AEP','XEL','WEC','CMS','AES','EIX','PCG'
]


def compute_indicators(df):
    '''expanded technical indicators'''
    df = df.copy()
    c = df['Close'].values.flatten().astype(float)
    h = df['High'].values.flatten().astype(float)
    l = df['Low'].values.flatten().astype(float)
    v = df['Volume'].values.flatten().astype(float)

    for w in [5, 10, 20, 50]:
        df[f'SMA_{w}'] = pd.Series(c).rolling(window=w, min_periods=1).mean().values
    df['EMA_12'] = pd.Series(c).ewm(span=12, adjust=False).mean().values
    df['EMA_26'] = pd.Series(c).ewm(span=26, adjust=False).mean().values
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = pd.Series(df['MACD'].values).ewm(span=9, adjust=False).mean().values
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    delta = pd.Series(c).diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = (100 - (100 / (1 + rs))).values

    low14 = pd.Series(l).rolling(window=14, min_periods=1).min()
    high14 = pd.Series(h).rolling(window=14, min_periods=1).max()
    df['Stoch_K'] = (100 * (pd.Series(c) - low14) / (high14 - low14 + 1e-10)).values
    df['Stoch_D'] = pd.Series(df['Stoch_K']).rolling(window=3, min_periods=1).mean().values
    df['Williams_R'] = (-100 * (high14 - pd.Series(c)) / (high14 - low14 + 1e-10)).values

    sma20 = pd.Series(c).rolling(window=20, min_periods=1).mean()
    std20 = pd.Series(c).rolling(window=20, min_periods=1).std().fillna(0)
    df['BB_Upper'] = (sma20 + 2 * std20).values
    df['BB_Lower'] = (sma20 - 2 * std20).values
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / (sma20.values + 1e-10))
    df['BB_Position'] = ((pd.Series(c) - df['BB_Lower'].values) / (df['BB_Upper'].values - df['BB_Lower'].values + 1e-10)).values

    tr1 = pd.Series(h) - pd.Series(l)
    tr2 = abs(pd.Series(h) - pd.Series(c).shift(1))
    tr3 = abs(pd.Series(l) - pd.Series(c).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14, min_periods=1).mean().values

    obv = [0.0]
    for i in range(1, len(c)):
        if c[i] > c[i-1]: obv.append(obv[-1] + v[i])
        elif c[i] < c[i-1]: obv.append(obv[-1] - v[i])
        else: obv.append(obv[-1])
    df['OBV'] = obv
    df['OBV_SMA'] = pd.Series(obv).rolling(window=20, min_periods=1).mean().values

    df['Momentum_5'] = pd.Series(c).pct_change(periods=5).fillna(0).values
    df['Momentum_10'] = pd.Series(c).pct_change(periods=10).fillna(0).values
    df['Momentum_20'] = pd.Series(c).pct_change(periods=20).fillna(0).values

    vol_sma = pd.Series(v).rolling(window=20, min_periods=1).mean()
    df['Vol_Ratio'] = (pd.Series(v) / (vol_sma + 1e-10)).values
    df['Vol_Change'] = pd.Series(v).pct_change().fillna(0).values
    df['Returns'] = pd.Series(c).pct_change().fillna(0).values
    df['Returns_5d'] = pd.Series(c).pct_change(periods=5).fillna(0).values
    df['Volatility_10'] = pd.Series(c).pct_change().rolling(window=10, min_periods=1).std().fillna(0).values
    df['Volatility_20'] = pd.Series(c).pct_change().rolling(window=20, min_periods=1).std().fillna(0).values
    df['Price_to_SMA20'] = (pd.Series(c) / (df['SMA_20'].values + 1e-10) - 1).values
    df['Price_to_SMA50'] = (pd.Series(c) / (df['SMA_50'].values + 1e-10) - 1).values
    df['SMA_Cross_5_20'] = (df['SMA_5'] - df['SMA_20']).values
    df['SMA_Cross_20_50'] = (df['SMA_20'] - df['SMA_50']).values

    df['Trend_Slope_5'] = pd.Series(c).rolling(window=5, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True).values
    df['Trend_Slope_10'] = pd.Series(c).rolling(window=10, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True).values

    if hasattr(df.index, 'dayofweek'):
        dow = df.index.dayofweek
        month = df.index.month
    else:
        dow = pd.Series([0] * len(df))
        month = pd.Series([1] * len(df))
    df['Day_Sin'] = np.sin(2 * np.pi * dow / 5)
    df['Day_Cos'] = np.cos(2 * np.pi * dow / 5)
    df['Month_Sin'] = np.sin(2 * np.pi * month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * month / 12)

    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df


FEATURE_COLS = [
    'Close','High','Low','Open','Volume',
    'SMA_5','SMA_10','SMA_20','SMA_50',
    'EMA_12','EMA_26','MACD','MACD_Signal','MACD_Hist',
    'RSI','Stoch_K','Stoch_D','Williams_R',
    'BB_Upper','BB_Lower','BB_Width','BB_Position',
    'ATR','OBV','OBV_SMA',
    'Momentum_5','Momentum_10','Momentum_20',
    'Vol_Ratio','Vol_Change','Returns','Returns_5d',
    'Volatility_10','Volatility_20',
    'Price_to_SMA20','Price_to_SMA50',
    'SMA_Cross_5_20','SMA_Cross_20_50',
    'Trend_Slope_5','Trend_Slope_10',
    'Day_Sin','Day_Cos','Month_Sin','Month_Cos'
]


def prepare_features(df):
    available = [c for c in FEATURE_COLS if c in df.columns]
    features = df[available].copy()
    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    return features, available


def get_insider_activity(ticker):
    '''
    fetch insider transactions (ceo, cfo, directors buying/selling).
    this is the feature from merlin 4.2 brought back.
    '''
    try:
        stock = yf.Ticker(ticker)

        transactions = []
        try:
            insider_df = stock.insider_transactions
            if insider_df is not None and not insider_df.empty:
                for _, row in insider_df.head(20).iterrows():
                    name = str(row.get('Insider Trading', row.get('insider', '')))
                    title = str(row.get('Position', row.get('position', '')))
                    transaction = str(row.get('Transaction', row.get('transaction', '')))
                    date = str(row.get('Start Date', row.get('startDate', '')))
                    shares = row.get('Shares', row.get('shares', 0))
                    value = row.get('Value', row.get('value', 0))

                    #work out if it's a buy or sell
                    trans_lower = transaction.lower()
                    if any(w in trans_lower for w in ['purchase', 'buy', 'acquisition']):
                        action = 'buy'
                    elif any(w in trans_lower for w in ['sale', 'sell', 'disposition']):
                        action = 'sell'
                    else:
                        action = 'other'

                    #check if ceo or cfo
                    title_lower = title.lower()
                    is_exec = any(t in title_lower for t in [
                        'ceo', 'cfo', 'chief executive', 'chief financial',
                        'president', 'chairman', 'director', 'officer',
                        'vp', 'vice president', 'evp', 'svp', 'coo', 'cto'
                    ])

                    try:
                        shares_val = int(float(str(shares).replace(',', '')))
                    except:
                        shares_val = 0
                    try:
                        value_val = float(str(value).replace(',', '').replace('$', ''))
                    except:
                        value_val = 0

                    transactions.append({
                        'name': name[:40],
                        'title': title[:30],
                        'action': action,
                        'transaction': transaction[:40],
                        'date': str(date)[:10],
                        'shares': shares_val,
                        'value': value_val,
                        'is_executive': is_exec
                    })
        except:
            pass

        #also try insider_purchases for a summary
        purchases_summary = {}
        try:
            purchases = stock.insider_purchases
            if purchases is not None and not purchases.empty:
                total_buys = 0
                total_sells = 0
                for _, row in purchases.iterrows():
                    label = str(row.iloc[0]).lower() if len(row) > 0 else ''
                    count = 0
                    try:
                        count = int(row.iloc[1]) if len(row) > 1 else 0
                    except:
                        pass
                    if 'buy' in label or 'purchase' in label:
                        total_buys += count
                    elif 'sell' in label or 'sale' in label:
                        total_sells += count
                purchases_summary = {
                    'total_buys': total_buys,
                    'total_sells': total_sells,
                    'net_sentiment': 'bullish' if total_buys > total_sells else 'bearish' if total_sells > total_buys else 'neutral'
                }
        except:
            pass

        #score insider activity
        exec_buys = sum(1 for t in transactions if t['is_executive'] and t['action'] == 'buy')
        exec_sells = sum(1 for t in transactions if t['is_executive'] and t['action'] == 'sell')
        all_buys = sum(1 for t in transactions if t['action'] == 'buy')
        all_sells = sum(1 for t in transactions if t['action'] == 'sell')

        if exec_buys > exec_sells:
            insider_sentiment = 'bullish'
            insider_score = min(exec_buys - exec_sells, 3)
        elif exec_sells > exec_buys:
            insider_sentiment = 'bearish'
            insider_score = -min(exec_sells - exec_buys, 3)
        else:
            insider_sentiment = 'neutral'
            insider_score = 0

        return {
            'transactions': transactions,
            'summary': purchases_summary,
            'exec_buys': exec_buys,
            'exec_sells': exec_sells,
            'all_buys': all_buys,
            'all_sells': all_sells,
            'sentiment': insider_sentiment,
            'score': insider_score
        }

    except Exception as e:
        return {
            'transactions': [],
            'summary': {},
            'exec_buys': 0, 'exec_sells': 0,
            'all_buys': 0, 'all_sells': 0,
            'sentiment': 'unknown',
            'score': 0,
            'error': str(e)
        }


def quick_score_stock(ticker):
    '''
    fast technical scoring for the screener. no ml needed.
    scores based on technicals + news + insider activity.
    returns a dict with the score and reasons.
    '''
    try:
        data = yf.download(ticker, period='6mo', interval='1d', progress=False)
        if data.empty or len(data) < 50:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = compute_indicators(data)
        c = float(data['Close'].iloc[-1])
        score = 0
        signals = []

        #rsi
        rsi = float(data['RSI'].iloc[-1])
        if rsi < 30:
            score += 2
            signals.append('RSI oversold')
        elif rsi > 70:
            score -= 2
            signals.append('RSI overbought')
        elif 40 < rsi < 60:
            score += 0.5

        #macd crossover
        macd = float(data['MACD'].iloc[-1])
        macd_sig = float(data['MACD_Signal'].iloc[-1])
        macd_prev = float(data['MACD'].iloc[-2])
        macd_sig_prev = float(data['MACD_Signal'].iloc[-2])
        if macd > macd_sig and macd_prev <= macd_sig_prev:
            score += 2
            signals.append('MACD bullish crossover')
        elif macd < macd_sig and macd_prev >= macd_sig_prev:
            score -= 2
            signals.append('MACD bearish crossover')
        elif macd > macd_sig:
            score += 0.5

        #sma crossovers
        sma5 = float(data['SMA_5'].iloc[-1])
        sma20 = float(data['SMA_20'].iloc[-1])
        sma50 = float(data['SMA_50'].iloc[-1])
        if sma5 > sma20 > sma50:
            score += 2
            signals.append('bullish SMA alignment')
        elif sma5 < sma20 < sma50:
            score -= 2
            signals.append('bearish SMA alignment')
        if c > sma20:
            score += 0.5
        if c > sma50:
            score += 0.5

        #bollinger band position
        bb_pos = float(data['BB_Position'].iloc[-1])
        if bb_pos < 0.1:
            score += 1.5
            signals.append('near lower Bollinger band')
        elif bb_pos > 0.9:
            score -= 1.5
            signals.append('near upper Bollinger band')

        #momentum
        mom5 = float(data['Momentum_5'].iloc[-1])
        mom20 = float(data['Momentum_20'].iloc[-1])
        if mom5 > 0.02 and mom20 > 0:
            score += 1
            signals.append('positive momentum')
        elif mom5 < -0.02 and mom20 < 0:
            score -= 1

        #volume spike
        vol_ratio = float(data['Vol_Ratio'].iloc[-1])
        if vol_ratio > 1.5 and mom5 > 0:
            score += 1
            signals.append('high volume breakout')

        #trend slope
        slope = float(data['Trend_Slope_5'].iloc[-1])
        if slope > 0:
            score += 0.5
        else:
            score -= 0.5

        #stochastic
        stoch_k = float(data['Stoch_K'].iloc[-1])
        stoch_d = float(data['Stoch_D'].iloc[-1])
        if stoch_k < 20 and stoch_k > stoch_d:
            score += 1.5
            signals.append('stochastic bullish')
        elif stoch_k > 80 and stoch_k < stoch_d:
            score -= 1.5

        price = round(c, 2)
        change_1d = round(float(data['Returns'].iloc[-1]) * 100, 2)
        change_5d = round(float(data['Returns_5d'].iloc[-1]) * 100, 2)

        return {
            'ticker': ticker,
            'price': price,
            'change_1d': change_1d,
            'change_5d': change_5d,
            'rsi': round(rsi, 1),
            'score': round(score, 1),
            'signals': signals[:4],
            'direction': 'bullish' if score > 1 else 'bearish' if score < -1 else 'neutral'
        }
    except Exception as e:
        return None


def get_news_sentiment(ticker):
    '''fetch news handling both old and new yfinance formats'''
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news
        if not raw_news:
            return {'articles': [], 'overall_score': 0, 'summary': 'no recent news found', 'count': 0}

        positive_words = {
            'surge','surges','soar','soars','jump','jumps','gain','gains',
            'rally','rallies','rise','rises','climb','climbs','high',
            'record','beat','beats','exceed','exceeds','strong','bullish',
            'growth','profit','upgrade','buy','outperform','positive',
            'boost','boosts','recover','recovery','optimistic','upbeat',
            'breakthrough','innovation','expand','expansion','revenue',
            'earnings','dividend','approval','partnership','deal','success',
            'win','wins','top','tops','higher','upside','above'
        }
        negative_words = {
            'drop','drops','fall','falls','decline','declines','plunge',
            'plunges','crash','crashes','sink','sinks','down','low',
            'loss','losses','miss','misses','weak','bearish','sell',
            'downgrade','underperform','negative','cut','cuts','slash',
            'warning','risk','fear','concern','worry','recession',
            'lawsuit','investigation','fine','penalty','recall',
            'bankruptcy','layoff','layoffs','debt','deficit','lower',
            'below','worst','trouble','crisis','slump','tumble'
        }

        articles = []
        total_score = 0
        for item in raw_news[:15]:
            title, publisher, link, pub_date = '', '', '', ''
            if isinstance(item, dict) and 'content' in item:
                content = item['content']
                title = content.get('title', '')
                p = content.get('provider', {})
                publisher = p.get('displayName', '') if isinstance(p, dict) else ''
                cu = content.get('canonicalUrl', {})
                link = cu.get('url', '') if isinstance(cu, dict) else ''
                pub_date = content.get('pubDate', '')
            elif isinstance(item, dict):
                title = item.get('title', '')
                publisher = item.get('publisher', '')
                link = item.get('link', '')
                if 'providerPublishTime' in item:
                    try: pub_date = datetime.datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M')
                    except: pass
            if not title: continue
            words = set(re.findall(r'\b\w+\b', title.lower()))
            pos = len(words & positive_words)
            neg = len(words & negative_words)
            if pos > neg: sentiment, sc = 'positive', min(pos - neg, 3)
            elif neg > pos: sentiment, sc = 'negative', -min(neg - pos, 3)
            else: sentiment, sc = 'neutral', 0
            total_score += sc
            articles.append({'title': title, 'publisher': publisher, 'date': pub_date[:16] if pub_date else '', 'link': link, 'sentiment': sentiment, 'score': sc})

        n = len(articles)
        if n == 0: return {'articles': [], 'overall_score': 0, 'summary': 'no recent news found', 'count': 0}
        avg = total_score / n
        summary = 'news sentiment is generally positive' if avg > 0.4 else 'news sentiment is generally negative' if avg < -0.4 else 'news sentiment is mixed or neutral'
        return {'articles': articles, 'overall_score': round(avg, 2), 'summary': summary, 'count': n}
    except:
        return {'articles': [], 'overall_score': 0, 'summary': 'could not fetch news', 'count': 0}


#model builders (same 3 architectures from 5.3)
def build_model_a(sl, nf):
    m = Sequential()
    m.add(Conv1D(64, 3, activation='relu', padding='same', input_shape=(sl, nf), kernel_regularizer=l2(1e-4)))
    m.add(Conv1D(32, 3, activation='relu', padding='same'))
    m.add(Dropout(0.2))
    m.add(Bidirectional(LSTM(80, return_sequences=True)))
    m.add(Dropout(0.25))
    m.add(Bidirectional(LSTM(40)))
    m.add(Dropout(0.2))
    m.add(Dense(32, activation='relu'))
    m.add(Dense(1, activation='tanh'))
    m.compile(optimizer='adam', loss='huber')
    return m

def build_model_b(sl, nf):
    m = Sequential()
    m.add(GRU(100, return_sequences=True, input_shape=(sl, nf)))
    m.add(Dropout(0.25))
    m.add(GRU(50, return_sequences=True))
    m.add(Dropout(0.25))
    m.add(GRU(25))
    m.add(Dropout(0.2))
    m.add(Dense(32, activation='relu'))
    m.add(Dense(16, activation='relu'))
    m.add(Dense(1, activation='tanh'))
    m.compile(optimizer='adam', loss='huber')
    return m

def build_model_c(sl, nf):
    m = Sequential()
    m.add(LSTM(128, return_sequences=True, input_shape=(sl, nf)))
    m.add(BatchNormalization())
    m.add(Dropout(0.3))
    m.add(LSTM(64, return_sequences=True))
    m.add(BatchNormalization())
    m.add(Dropout(0.25))
    m.add(LSTM(32))
    m.add(Dropout(0.2))
    m.add(Dense(48, activation='relu', kernel_regularizer=l2(1e-4)))
    m.add(Dense(16, activation='relu'))
    m.add(Dense(1, activation='tanh'))
    m.compile(optimizer='adam', loss='huber')
    return m

MODEL_BUILDERS = [build_model_a, build_model_b, build_model_c]
MODEL_NAMES = ['conv-lstm', 'gru', 'deep-lstm']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/fetch_data', methods=['POST'])
def fetch_data():
    try:
        ticker = request.json.get('ticker', '').upper().strip()
        if not ticker: return jsonify({'error': 'no ticker provided'}), 400

        data = yf.download(ticker, period='10y', interval='1d', progress=False)
        if data.empty or len(data) < 200:
            return jsonify({'error': f'not enough data for {ticker}.'}), 400

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = compute_indicators(data)
        data.to_csv(os.path.join(DATA_DIR, f'{ticker}.csv'))

        chart_data = {
            'dates': [d.strftime('%Y-%m-%d') for d in data.index],
            'close': [round(float(v), 2) for v in data['Close'].values],
            'volume': [int(v) for v in data['Volume'].values],
            'high': [round(float(v), 2) for v in data['High'].values],
            'low': [round(float(v), 2) for v in data['Low'].values],
            'open': [round(float(v), 2) for v in data['Open'].values],
            'sma_20': [round(float(v), 2) if not np.isnan(v) else None for v in data['SMA_20'].values],
            'sma_50': [round(float(v), 2) if not np.isnan(v) else None for v in data['SMA_50'].values],
            'rsi': [round(float(v), 2) if not np.isnan(v) else None for v in data['RSI'].values],
        }

        current_price = round(float(data['Close'].iloc[-1]), 2)
        prev_price = round(float(data['Close'].iloc[-2]), 2)
        change = round(current_price - prev_price, 2)
        change_pct = round((change / prev_price) * 100, 2)

        news = get_news_sentiment(ticker)
        insider = get_insider_activity(ticker)

        return jsonify({
            'ticker': ticker, 'current_price': current_price,
            'change': change, 'change_pct': change_pct,
            'data_points': len(data), 'chart_data': chart_data,
            'news': news, 'insider': insider,
            'date_range': {'start': data.index[0].strftime('%Y-%m-%d'), 'end': data.index[-1].strftime('%Y-%m-%d')}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/screener', methods=['POST'])
def run_screener():
    '''start the stock screener in a background thread'''
    global screener_status
    if screener_status['active']:
        return jsonify({'error': 'screener already running'}), 400

    top_n = request.json.get('top_n', 10)
    stock_count = request.json.get('stock_count', 300)

    screener_status = {
        'active': True, 'progress': 0,
        'message': 'starting screener...',
        'complete': False, 'results': [], 'error': None,
        'top_n': top_n
    }

    thread = threading.Thread(target=_run_screener, args=(top_n, stock_count))
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'screener started'})


def _run_screener(top_n, stock_count):
    '''scan stocks and rank by technical score'''
    global screener_status
    try:
        stocks_to_scan = STOCK_UNIVERSE[:min(stock_count, len(STOCK_UNIVERSE))]
        total = len(stocks_to_scan)
        results = []

        for i, ticker in enumerate(stocks_to_scan):
            screener_status['progress'] = int((i / total) * 90)
            screener_status['message'] = f'scanning {ticker} ({i+1}/{total})...'

            result = quick_score_stock(ticker)
            if result:
                results.append(result)

            #rate limit: don't hammer yahoo
            if i % 10 == 0 and i > 0:
                time.sleep(0.5)

        #sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)

        #add news and insider data for top picks
        screener_status['message'] = 'fetching details for top picks...'
        screener_status['progress'] = 92

        top_results = results[:top_n]
        for r in top_results:
            try:
                news = get_news_sentiment(r['ticker'])
                r['news_score'] = news.get('overall_score', 0)
                r['news_summary'] = news.get('summary', '')

                insider = get_insider_activity(r['ticker'])
                r['insider_sentiment'] = insider.get('sentiment', 'unknown')
                r['insider_score'] = insider.get('score', 0)
                r['exec_buys'] = insider.get('exec_buys', 0)
                r['exec_sells'] = insider.get('exec_sells', 0)

                #combine scores
                r['combined_score'] = round(r['score'] + r['news_score'] * 0.5 + r['insider_score'] * 0.8, 1)
                time.sleep(0.3)
            except:
                r['news_score'] = 0
                r['insider_sentiment'] = 'unknown'
                r['insider_score'] = 0
                r['combined_score'] = r['score']

        #re-sort by combined score
        top_results.sort(key=lambda x: x.get('combined_score', x['score']), reverse=True)

        screener_status['results'] = top_results
        screener_status['total_scanned'] = total
        screener_status['progress'] = 100
        screener_status['complete'] = True
        screener_status['active'] = False
        screener_status['message'] = f'screener complete! scanned {total} stocks.'

    except Exception as e:
        screener_status['error'] = str(e)
        screener_status['active'] = False
        screener_status['message'] = f'error: {str(e)}'


@app.route('/api/screener_status')
def get_screener_status():
    return jsonify(screener_status)


@app.route('/api/train', methods=['POST'])
def train_model_route():
    global training_status
    if training_status['active']:
        return jsonify({'error': 'training already in progress'}), 400

    ticker = request.json.get('ticker', '').upper().strip()
    epochs = request.json.get('epochs', 50)
    if not ticker: return jsonify({'error': 'no ticker'}), 400

    csv_path = os.path.join(DATA_DIR, f'{ticker}.csv')
    if not os.path.exists(csv_path):
        return jsonify({'error': f'no data for {ticker}. fetch first.'}), 400

    training_status = {
        'active': True, 'progress': 0, 'message': 'starting...',
        'ticker': ticker, 'complete': False, 'error': None, 'backtest': None
    }
    thread = threading.Thread(target=_train_ensemble, args=(ticker, epochs))
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'training started', 'ticker': ticker})


def _train_ensemble(ticker, epochs):
    global training_status
    try:
        training_status['message'] = 'loading data...'
        training_status['progress'] = 2

        data = pd.read_csv(os.path.join(DATA_DIR, f'{ticker}.csv'), index_col=0, parse_dates=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        features_df, feature_cols = prepare_features(data)
        num_features = len(feature_cols)
        close_idx = feature_cols.index('Close')

        close_prices = features_df['Close'].values.astype(float)
        returns = np.diff(close_prices) / (close_prices[:-1] + 1e-10)
        returns = np.clip(returns, -0.15, 0.15)

        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(features_df.values)

        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(scaled_data) - 1):
            X.append(scaled_data[i - SEQUENCE_LENGTH:i])
            y.append(returns[i - 1])
        X = np.array(X)
        y = np.array(y)

        total = len(X)
        test_size = int(total * 0.2)
        train_size = total - test_size
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        val_split = int(train_size * 0.85)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        training_status['progress'] = 10
        model_results = []
        ensemble_test_preds = []

        for i in range(NUM_ENSEMBLE):
            mn = MODEL_NAMES[i]
            training_status['message'] = f'training model {i+1}/3 ({mn})...'
            model = MODEL_BUILDERS[i](SEQUENCE_LENGTH, num_features)

            es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
            rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

            for epoch in range(epochs):
                h = model.fit(X_tr, y_tr, epochs=1, batch_size=32,
                             validation_data=(X_val, y_val), callbacks=[es, rlr], verbose=0)
                base = 10 + i * 25
                training_status['progress'] = min(base + int((epoch + 1) / epochs * 22), 85)
                training_status['message'] = f'{mn} | epoch {epoch+1}/{epochs} | loss: {h.history["loss"][0]:.6f} | val: {h.history["val_loss"][0]:.6f}'
                if es.stopped_epoch > 0: break

            model.save(os.path.join(MODELS_DIR, f'{ticker}_model_{i}.keras'))
            tp = model.predict(X_test, verbose=0).flatten()
            ensemble_test_preds.append(tp)
            da = np.mean((tp > 0) == (y_test > 0)) * 100
            model_results.append({'name': mn, 'direction_accuracy': round(float(da), 1)})

        training_status['message'] = 'computing ensemble metrics...'
        training_status['progress'] = 88

        eavg = np.mean(ensemble_test_preds, axis=0)
        evote = np.sign(np.sum([np.sign(p) for p in ensemble_test_preds], axis=0))
        eda = np.mean((evote > 0) == (y_test > 0)) * 100

        tcp = close_prices[train_size + SEQUENCE_LENGTH:train_size + SEQUENCE_LENGTH + len(y_test)]
        pp = tcp * (1 + eavg)
        ap = tcp * (1 + y_test)
        mae = mean_absolute_error(ap, pp)
        rmse = np.sqrt(mean_squared_error(ap, pp))
        mape = np.mean(np.abs((ap - pp) / (ap + 1e-10))) * 100

        cl = min(60, len(ap))
        backtest = {
            'ensemble_direction_accuracy': round(float(eda), 1),
            'mae': round(float(mae), 2), 'rmse': round(float(rmse), 2),
            'mape': round(float(mape), 2),
            'individual_models': model_results,
            'val_actual': [round(float(p), 2) for p in ap[-cl:]],
            'val_predictions': [round(float(p), 2) for p in pp[-cl:]]
        }

        training_status['progress'] = 95
        meta = {
            'feature_cols': feature_cols, 'num_features': num_features,
            'close_idx': close_idx,
            'scaler': {'center': scaler.center_.tolist(), 'scale': scaler.scale_.tolist()},
            'backtest': backtest, 'model_names': MODEL_NAMES[:NUM_ENSEMBLE],
            'num_models': NUM_ENSEMBLE
        }
        with open(os.path.join(MODELS_DIR, f'{ticker}_meta.json'), 'w') as f:
            json.dump(meta, f)

        training_status['message'] = 'ensemble training complete!'
        training_status['progress'] = 100
        training_status['complete'] = True
        training_status['active'] = False
        training_status['backtest'] = backtest
    except Exception as e:
        import traceback; traceback.print_exc()
        training_status['error'] = str(e)
        training_status['active'] = False


@app.route('/api/training_status')
def get_training_status():
    return jsonify(training_status)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        ticker = request.json.get('ticker', '').upper().strip()
        timeframe = request.json.get('timeframe', '1d')
        if not ticker: return jsonify({'error': 'no ticker'}), 400

        meta_path = os.path.join(MODELS_DIR, f'{ticker}_meta.json')
        if not os.path.exists(meta_path):
            return jsonify({'error': f'no models for {ticker}. train first.'}), 400

        with open(meta_path) as f: meta = json.load(f)
        feature_cols = meta['feature_cols']

        models = []
        for i in range(meta['num_models']):
            mp = os.path.join(MODELS_DIR, f'{ticker}_model_{i}.keras')
            if not os.path.exists(mp): return jsonify({'error': f'model {i} missing'}), 400
            models.append(load_model(mp))

        scaler = RobustScaler()
        scaler.center_ = np.array(meta['scaler']['center'])
        scaler.scale_ = np.array(meta['scaler']['scale'])

        raw_data = pd.read_csv(os.path.join(DATA_DIR, f'{ticker}.csv'), index_col=0, parse_dates=True)
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)

        ohlcv_buffer = raw_data[['Open','High','Low','Close','Volume']].copy()
        current_price = float(raw_data['Close'].iloc[-1])
        returns = raw_data['Close'].pct_change().dropna()
        daily_vol = float(returns.std())

        tfmap = {'1d':1,'1w':5,'1m':22,'3m':66,'6m':132,'1y':252}
        days = tfmap.get(timeframe, 1)

        all_model_preds = [[] for _ in range(meta['num_models'])]
        ensemble_prices = []
        upper_band, lower_band = [], []
        backtest_rmse = meta.get('backtest', {}).get('rmse', current_price * 0.02)

        for day in range(days):
            temp_df = compute_indicators(ohlcv_buffer.copy())
            temp_features, _ = prepare_features(temp_df)
            recent = temp_features[feature_cols].iloc[-SEQUENCE_LENGTH:]
            if len(recent) < SEQUENCE_LENGTH: break

            scaled_seq = scaler.transform(recent.values)
            input_seq = scaled_seq.reshape(1, SEQUENCE_LENGTH, len(feature_cols))

            model_returns = []
            for i, model in enumerate(models):
                pr = float(model.predict(input_seq, verbose=0)[0][0])
                pr = np.clip(pr, -0.08, 0.08)
                model_returns.append(pr)

            votes_up = sum(1 for r in model_returns if r > 0)
            direction = 1 if votes_up > len(model_returns) / 2 else -1
            magnitude = np.mean(np.abs(model_returns))
            consensus_return = direction * magnitude

            prev = ensemble_prices[-1] if ensemble_prices else current_price
            pred_price = round(prev * (1 + consensus_return), 2)
            ensemble_prices.append(pred_price)

            for i, ret in enumerate(model_returns):
                p = all_model_preds[i][-1] if all_model_preds[i] else current_price
                all_model_preds[i].append(round(p * (1 + ret), 2))

            unc = backtest_rmse * np.sqrt(day + 1) * 0.8
            upper_band.append(round(pred_price + 1.96 * unc, 2))
            lower_band.append(round(pred_price - 1.96 * unc, 2))

            noise = daily_vol * prev
            next_date = ohlcv_buffer.index[-1] + pd.Timedelta(days=1)
            while next_date.weekday() >= 5: next_date += pd.Timedelta(days=1)
            new_row = pd.DataFrame({
                'Open': [prev], 'High': [max(pred_price, prev) + abs(noise*0.3)],
                'Low': [min(pred_price, prev) - abs(noise*0.3)],
                'Close': [pred_price], 'Volume': [float(ohlcv_buffer['Volume'].iloc[-20:].mean())]
            }, index=[next_date])
            ohlcv_buffer = pd.concat([ohlcv_buffer, new_row])

        pred_dates = []
        cd = raw_data.index[-1]
        for _ in range(len(ensemble_prices)):
            cd += datetime.timedelta(days=1)
            while cd.weekday() >= 5: cd += datetime.timedelta(days=1)
            pred_dates.append(cd.strftime('%Y-%m-%d'))

        model_votes = []
        for i in range(meta['num_models']):
            preds = all_model_preds[i]
            if preds:
                final = preds[-1]
                d = 'up' if final > current_price else 'down'
                cp = round(((final - current_price) / current_price) * 100, 2)
                model_votes.append({'name': MODEL_NAMES[i], 'direction': d, 'final_price': final, 'change_pct': cp, 'prices': preds})

        news = get_news_sentiment(ticker)
        insider = get_insider_activity(ticker)
        analysis = _analyse(current_price, ensemble_prices, pred_dates, news, insider, meta.get('backtest', {}), model_votes, daily_vol)

        return jsonify({
            'ticker': ticker, 'timeframe': timeframe, 'current_price': current_price,
            'predictions': {'dates': pred_dates, 'prices': ensemble_prices, 'upper_band': upper_band, 'lower_band': lower_band},
            'model_votes': model_votes, 'analysis': analysis,
            'news_sentiment': news, 'insider': insider
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _analyse(current_price, pred_prices, pred_dates, news, insider, backtest, model_votes, daily_vol):
    if not pred_prices: return {'action': 'hold', 'reason': 'no predictions'}

    min_p, max_p = min(pred_prices), max(pred_prices)
    min_i, max_i = pred_prices.index(min_p), pred_prices.index(max_p)
    final = pred_prices[-1]
    oc = round(((final - current_price) / current_price) * 100, 2)

    da = backtest.get('ensemble_direction_accuracy', 50)
    conf = 'low' if da < 55 else 'medium' if da < 65 else 'high'

    vu = sum(1 for v in model_votes if v['direction'] == 'up')
    cons = f'{vu}/{len(model_votes)} models say up'

    ns = news.get('overall_score', 0) if news else 0
    nb = 'positive' if ns > 0.3 else 'negative' if ns < -0.3 else 'neutral'

    isent = insider.get('sentiment', 'unknown') if insider else 'unknown'
    iscore = insider.get('score', 0) if insider else 0
    ebuys = insider.get('exec_buys', 0) if insider else 0
    esells = insider.get('exec_sells', 0) if insider else 0

    result = {
        'current_price': current_price,
        'predicted_low': round(min_p, 2), 'predicted_low_date': pred_dates[min_i],
        'predicted_high': round(max_p, 2), 'predicted_high_date': pred_dates[max_i],
        'final_predicted_price': round(final, 2), 'overall_change_pct': oc,
        'model_confidence': conf, 'direction_accuracy': da,
        'news_bias': nb, 'news_score': ns,
        'insider_sentiment': isent, 'insider_score': iscore,
        'exec_buys': ebuys, 'exec_sells': esells,
        'consensus': cons, 'votes_up': vu, 'votes_down': len(model_votes) - vu,
        'daily_volatility': round(daily_vol * 100, 2),
    }

    threshold = 0.015 if conf == 'high' else 0.02 if conf == 'medium' else 0.03
    reasons = []

    if min_i < max_i:
        swing = (max_p - min_p) / min_p
        if swing >= threshold:
            result['action'] = 'buy_then_sell'
            result['buy_date'] = pred_dates[min_i]
            result['buy_price'] = round(min_p, 2)
            result['sell_date'] = pred_dates[max_i]
            result['sell_price'] = round(max_p, 2)
            result['potential_profit_pct'] = round(swing * 100, 2)
            reasons.append(f'{cons}. predicted dip to ${min_p:.2f} on {pred_dates[min_i]} then rise to ${max_p:.2f} on {pred_dates[max_i]} ({swing*100:.1f}% gain)')
            if nb == 'positive': reasons.append('news sentiment supports this')
            elif nb == 'negative': reasons.append('news sentiment is negative so proceed with caution')
            if isent == 'bullish': reasons.append(f'insider activity is bullish ({ebuys} exec buys vs {esells} sells)')
            elif isent == 'bearish': reasons.append(f'but insiders have been selling ({esells} exec sells vs {ebuys} buys)')
            reasons.append(f'ensemble direction accuracy: {da:.0f}% ({conf} confidence)')
            result['reason'] = '. '.join(reasons)
            return result

    if max_i < min_i:
        drop = (max_p - min_p) / max_p
        if drop >= threshold:
            result['action'] = 'sell_then_buy'
            result['sell_date'] = pred_dates[max_i]
            result['sell_price'] = round(max_p, 2)
            result['buy_date'] = pred_dates[min_i]
            result['buy_price'] = round(min_p, 2)
            result['potential_profit_pct'] = round(drop * 100, 2)
            reasons.append(f'{cons}. predicted peak at ${max_p:.2f} on {pred_dates[max_i]} then drop to ${min_p:.2f} on {pred_dates[min_i]}')
            if nb == 'negative': reasons.append('negative news supports this')
            if isent == 'bearish': reasons.append(f'insiders are selling too ({esells} exec sells)')
            reasons.append(f'ensemble direction accuracy: {da:.0f}% ({conf} confidence)')
            result['reason'] = '. '.join(reasons)
            return result

    if oc > 2: result['action'] = 'buy'; reasons.append(f'{cons}. upward trend of {oc:+.1f}% predicted')
    elif oc < -2: result['action'] = 'sell'; reasons.append(f'{cons}. downward trend of {oc:+.1f}% predicted')
    else: result['action'] = 'hold'; reasons.append(f'{cons}. no significant movement ({oc:+.1f}%). hold or stay out')

    if nb != 'neutral': reasons.append(f'news: {nb} ({ns:+.1f})')
    if isent != 'unknown' and isent != 'neutral':
        reasons.append(f'insider activity: {isent} ({ebuys} exec buys / {esells} sells)')
    reasons.append(f'ensemble accuracy: {da:.0f}% ({conf})')
    result['reason'] = '. '.join(reasons)
    return result


@app.route('/api/save_trade', methods=['POST'])
def save_trade():
    global trade_recommendations
    trade = request.json
    trade['saved_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    trade_recommendations.append(trade)
    return jsonify({'status': 'saved', 'total': len(trade_recommendations)})

@app.route('/api/trades')
def get_trades():
    return jsonify(trade_recommendations)

@app.route('/api/clear_trades', methods=['POST'])
def clear_trades():
    global trade_recommendations
    trade_recommendations = []
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
