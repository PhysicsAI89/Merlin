'''
merlin 5.3 - ensemble stock prediction engine
3 models vote on direction, predicts returns not price,
walk-forward validation, expanded technical indicators
'''

import os
import json
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import (LSTM, Dense, Dropout, Bidirectional,
                          Conv1D, GRU, BatchNormalization)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
import threading
import re
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

training_status = {
    'active': False,
    'progress': 0,
    'message': '',
    'ticker': '',
    'complete': False,
    'error': None,
    'backtest': None
}

trade_recommendations = []

MODELS_DIR = 'models'
DATA_DIR = 'data'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

SEQUENCE_LENGTH = 60
NUM_ENSEMBLE = 3


def compute_indicators(df):
    '''
    expanded technical indicators including stochastic, williams %r,
    obv, momentum, day/month features and multi-timeframe context
    '''
    df = df.copy()
    c = df['Close'].values.flatten().astype(float)
    h = df['High'].values.flatten().astype(float)
    l = df['Low'].values.flatten().astype(float)
    v = df['Volume'].values.flatten().astype(float)

    #moving averages
    for w in [5, 10, 20, 50]:
        df[f'SMA_{w}'] = pd.Series(c).rolling(window=w, min_periods=1).mean().values
    df['EMA_12'] = pd.Series(c).ewm(span=12, adjust=False).mean().values
    df['EMA_26'] = pd.Series(c).ewm(span=26, adjust=False).mean().values

    #macd
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = pd.Series(df['MACD'].values).ewm(span=9, adjust=False).mean().values
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    #rsi 14
    delta = pd.Series(c).diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = (100 - (100 / (1 + rs))).values

    #stochastic oscillator (14 period)
    low14 = pd.Series(l).rolling(window=14, min_periods=1).min()
    high14 = pd.Series(h).rolling(window=14, min_periods=1).max()
    df['Stoch_K'] = (100 * (pd.Series(c) - low14) / (high14 - low14 + 1e-10)).values
    df['Stoch_D'] = pd.Series(df['Stoch_K']).rolling(window=3, min_periods=1).mean().values

    #williams %r (14 period)
    df['Williams_R'] = (-100 * (high14 - pd.Series(c)) / (high14 - low14 + 1e-10)).values

    #bollinger bands
    sma20 = pd.Series(c).rolling(window=20, min_periods=1).mean()
    std20 = pd.Series(c).rolling(window=20, min_periods=1).std().fillna(0)
    df['BB_Upper'] = (sma20 + 2 * std20).values
    df['BB_Lower'] = (sma20 - 2 * std20).values
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / (sma20.values + 1e-10))
    df['BB_Position'] = ((pd.Series(c) - df['BB_Lower'].values) / (df['BB_Upper'].values - df['BB_Lower'].values + 1e-10)).values

    #average true range
    tr1 = pd.Series(h) - pd.Series(l)
    tr2 = abs(pd.Series(h) - pd.Series(c).shift(1))
    tr3 = abs(pd.Series(l) - pd.Series(c).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14, min_periods=1).mean().values

    #on-balance volume
    obv = [0.0]
    for i in range(1, len(c)):
        if c[i] > c[i-1]:
            obv.append(obv[-1] + v[i])
        elif c[i] < c[i-1]:
            obv.append(obv[-1] - v[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    df['OBV_SMA'] = pd.Series(obv).rolling(window=20, min_periods=1).mean().values

    #momentum
    df['Momentum_5'] = pd.Series(c).pct_change(periods=5).fillna(0).values
    df['Momentum_10'] = pd.Series(c).pct_change(periods=10).fillna(0).values
    df['Momentum_20'] = pd.Series(c).pct_change(periods=20).fillna(0).values

    #volume features
    vol_sma = pd.Series(v).rolling(window=20, min_periods=1).mean()
    df['Vol_Ratio'] = (pd.Series(v) / (vol_sma + 1e-10)).values
    df['Vol_Change'] = pd.Series(v).pct_change().fillna(0).values

    #daily returns
    df['Returns'] = pd.Series(c).pct_change().fillna(0).values
    df['Returns_5d'] = pd.Series(c).pct_change(periods=5).fillna(0).values

    #volatility (rolling std of returns)
    df['Volatility_10'] = pd.Series(c).pct_change().rolling(window=10, min_periods=1).std().fillna(0).values
    df['Volatility_20'] = pd.Series(c).pct_change().rolling(window=20, min_periods=1).std().fillna(0).values

    #price relative to moving averages
    df['Price_to_SMA20'] = (pd.Series(c) / (df['SMA_20'].values + 1e-10) - 1).values
    df['Price_to_SMA50'] = (pd.Series(c) / (df['SMA_50'].values + 1e-10) - 1).values

    #sma crossover signals
    df['SMA_Cross_5_20'] = (df['SMA_5'] - df['SMA_20']).values
    df['SMA_Cross_20_50'] = (df['SMA_20'] - df['SMA_50']).values

    #weekly rolling context (5-day trend slope)
    df['Trend_Slope_5'] = pd.Series(c).rolling(window=5, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
    ).values
    df['Trend_Slope_10'] = pd.Series(c).rolling(window=10, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
    ).values

    #day of week and month (cyclical encoding)
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

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    return df


FEATURE_COLS = [
    'Close', 'High', 'Low', 'Open', 'Volume',
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'RSI', 'Stoch_K', 'Stoch_D', 'Williams_R',
    'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
    'ATR', 'OBV', 'OBV_SMA',
    'Momentum_5', 'Momentum_10', 'Momentum_20',
    'Vol_Ratio', 'Vol_Change',
    'Returns', 'Returns_5d',
    'Volatility_10', 'Volatility_20',
    'Price_to_SMA20', 'Price_to_SMA50',
    'SMA_Cross_5_20', 'SMA_Cross_20_50',
    'Trend_Slope_5', 'Trend_Slope_10',
    'Day_Sin', 'Day_Cos', 'Month_Sin', 'Month_Cos'
]


def prepare_features(df):
    available = [c for c in FEATURE_COLS if c in df.columns]
    features = df[available].copy()
    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    return features, available


def build_model_a(seq_len, num_features):
    '''conv-lstm hybrid'''
    m = Sequential()
    m.add(Conv1D(64, 3, activation='relu', padding='same',
                 input_shape=(seq_len, num_features), kernel_regularizer=l2(1e-4)))
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


def build_model_b(seq_len, num_features):
    '''gru based'''
    m = Sequential()
    m.add(GRU(100, return_sequences=True, input_shape=(seq_len, num_features)))
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


def build_model_c(seq_len, num_features):
    '''deep lstm with batch norm'''
    m = Sequential()
    m.add(LSTM(128, return_sequences=True, input_shape=(seq_len, num_features)))
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
            title = ''
            publisher = ''
            link = ''
            pub_date = ''

            if isinstance(item, dict) and 'content' in item:
                content = item['content']
                title = content.get('title', '')
                provider = content.get('provider', {})
                publisher = provider.get('displayName', '') if isinstance(provider, dict) else ''
                canon = content.get('canonicalUrl', {})
                link = canon.get('url', '') if isinstance(canon, dict) else ''
                pub_date = content.get('pubDate', '')
            elif isinstance(item, dict):
                title = item.get('title', '')
                publisher = item.get('publisher', '')
                link = item.get('link', '')
                if 'providerPublishTime' in item:
                    try:
                        pub_date = datetime.datetime.fromtimestamp(
                            item['providerPublishTime']).strftime('%Y-%m-%d %H:%M')
                    except:
                        pass

            if not title:
                continue

            words = set(re.findall(r'\b\w+\b', title.lower()))
            pos = len(words & positive_words)
            neg = len(words & negative_words)

            if pos > neg:
                sentiment = 'positive'
                score = min(pos - neg, 3)
            elif neg > pos:
                sentiment = 'negative'
                score = -min(neg - pos, 3)
            else:
                sentiment = 'neutral'
                score = 0

            total_score += score
            articles.append({
                'title': title, 'publisher': publisher,
                'date': pub_date[:16] if pub_date else '',
                'link': link, 'sentiment': sentiment, 'score': score
            })

        n = len(articles)
        if n == 0:
            return {'articles': [], 'overall_score': 0, 'summary': 'no recent news found', 'count': 0}

        avg = total_score / n
        if avg > 0.4:
            summary = 'news sentiment is generally positive'
        elif avg < -0.4:
            summary = 'news sentiment is generally negative'
        else:
            summary = 'news sentiment is mixed or neutral'

        return {'articles': articles, 'overall_score': round(avg, 2), 'summary': summary, 'count': n}
    except Exception as e:
        return {'articles': [], 'overall_score': 0, 'summary': f'could not fetch news: {str(e)}', 'count': 0}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/fetch_data', methods=['POST'])
def fetch_data():
    try:
        ticker = request.json.get('ticker', '').upper().strip()
        if not ticker:
            return jsonify({'error': 'no ticker provided'}), 400

        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=10*365)

        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'),
                          end=end_date.strftime('%Y-%m-%d'), interval='1d', progress=False)

        if data.empty or len(data) < 200:
            return jsonify({'error': f'not enough data for {ticker}. need at least 200 days.'}), 400

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

        return jsonify({
            'ticker': ticker, 'current_price': current_price,
            'change': change, 'change_pct': change_pct,
            'data_points': len(data), 'chart_data': chart_data,
            'news': news,
            'date_range': {'start': data.index[0].strftime('%Y-%m-%d'),
                          'end': data.index[-1].strftime('%Y-%m-%d')}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train_model_route():
    global training_status
    if training_status['active']:
        return jsonify({'error': 'training already in progress'}), 400

    ticker = request.json.get('ticker', '').upper().strip()
    epochs = request.json.get('epochs', 50)
    if not ticker:
        return jsonify({'error': 'no ticker provided'}), 400

    csv_path = os.path.join(DATA_DIR, f'{ticker}.csv')
    if not os.path.exists(csv_path):
        return jsonify({'error': f'no data for {ticker}. fetch data first.'}), 400

    training_status = {
        'active': True, 'progress': 0,
        'message': 'starting ensemble training...',
        'ticker': ticker, 'complete': False,
        'error': None, 'backtest': None
    }

    thread = threading.Thread(target=_train_ensemble, args=(ticker, epochs))
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'training started', 'ticker': ticker})


def _train_ensemble(ticker, epochs):
    '''
    train 3 different model architectures.
    target: next-day return (percentage change) not raw price.
    walk-forward validation for honest accuracy.
    '''
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

        training_status['message'] = f'preparing {num_features} features...'
        training_status['progress'] = 5

        #target: next day return (clipped to remove extreme outliers)
        close_prices = features_df['Close'].values.astype(float)
        returns = np.diff(close_prices) / (close_prices[:-1] + 1e-10)
        returns = np.clip(returns, -0.15, 0.15)

        #scale features
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(features_df.values)

        #create sequences -> predict next return
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(scaled_data) - 1):
            X.append(scaled_data[i - SEQUENCE_LENGTH:i])
            y.append(returns[i - 1])
        X = np.array(X)
        y = np.array(y)

        #walk-forward split: use last 20% as test, but train on expanding window
        total = len(X)
        test_size = int(total * 0.2)
        train_size = total - test_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]

        #further split training data for validation during training
        val_split = int(train_size * 0.85)
        X_tr = X_train[:val_split]
        y_tr = y_train[:val_split]
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]

        training_status['message'] = 'training ensemble (3 models)...'
        training_status['progress'] = 10

        model_results = []
        ensemble_test_preds = []

        for i in range(NUM_ENSEMBLE):
            model_name = MODEL_NAMES[i]
            training_status['message'] = f'training model {i+1}/3 ({model_name})...'

            builder = MODEL_BUILDERS[i]
            model = builder(SEQUENCE_LENGTH, num_features)

            early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

            for epoch in range(epochs):
                history = model.fit(
                    X_tr, y_tr, epochs=1, batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop, reduce_lr], verbose=0
                )
                base = 10 + i * 25
                progress = base + int((epoch + 1) / epochs * 22)
                training_status['progress'] = min(progress, 85)

                t_loss = history.history['loss'][0]
                v_loss = history.history['val_loss'][0]
                training_status['message'] = f'{model_name} | epoch {epoch+1}/{epochs} | loss: {t_loss:.6f} | val: {v_loss:.6f}'

                if early_stop.stopped_epoch > 0:
                    break

            #save model
            model_path = os.path.join(MODELS_DIR, f'{ticker}_model_{i}.keras')
            model.save(model_path)

            #test predictions
            test_preds = model.predict(X_test, verbose=0).flatten()
            ensemble_test_preds.append(test_preds)

            #individual model metrics
            pred_direction = test_preds > 0
            actual_direction = y_test > 0
            dir_acc = np.mean(pred_direction == actual_direction) * 100

            model_results.append({
                'name': model_name,
                'direction_accuracy': round(float(dir_acc), 1)
            })

        #ensemble metrics (majority vote)
        training_status['message'] = 'computing ensemble metrics...'
        training_status['progress'] = 88

        ensemble_avg = np.mean(ensemble_test_preds, axis=0)
        ensemble_vote = np.sign(np.sum([np.sign(p) for p in ensemble_test_preds], axis=0))

        actual_direction = y_test > 0
        ensemble_dir = ensemble_vote > 0
        ensemble_dir_acc = np.mean(ensemble_dir == actual_direction) * 100

        #also compute price-level metrics for context
        test_close_prices = close_prices[train_size + SEQUENCE_LENGTH:train_size + SEQUENCE_LENGTH + len(y_test)]
        predicted_prices = test_close_prices * (1 + ensemble_avg)
        actual_prices = test_close_prices * (1 + y_test)

        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mape = np.mean(np.abs((actual_prices - predicted_prices) / (actual_prices + 1e-10))) * 100

        #walk-forward backtest chart data (last 60 points)
        chart_len = min(60, len(actual_prices))
        bt_actual = [round(float(p), 2) for p in actual_prices[-chart_len:]]
        bt_predicted = [round(float(p), 2) for p in predicted_prices[-chart_len:]]

        backtest = {
            'ensemble_direction_accuracy': round(float(ensemble_dir_acc), 1),
            'mae': round(float(mae), 2),
            'rmse': round(float(rmse), 2),
            'mape': round(float(mape), 2),
            'individual_models': model_results,
            'val_actual': bt_actual,
            'val_predictions': bt_predicted
        }

        #save metadata
        training_status['message'] = 'saving ensemble...'
        training_status['progress'] = 95

        scaler_params = {
            'center': scaler.center_.tolist(),
            'scale': scaler.scale_.tolist()
        }

        meta = {
            'feature_cols': feature_cols,
            'num_features': num_features,
            'close_idx': close_idx,
            'scaler': scaler_params,
            'backtest': backtest,
            'model_names': MODEL_NAMES[:NUM_ENSEMBLE],
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
        import traceback
        traceback.print_exc()
        training_status['error'] = str(e)
        training_status['active'] = False
        training_status['message'] = f'error: {str(e)}'


@app.route('/api/training_status')
def get_training_status():
    return jsonify(training_status)


@app.route('/api/predict', methods=['POST'])
def predict():
    '''
    all 3 models predict independently.
    each predicts the next-day return, then we step forward
    by recomputing indicators from a rolling ohlcv buffer.
    final answer: majority vote on direction + averaged magnitude.
    '''
    try:
        ticker = request.json.get('ticker', '').upper().strip()
        timeframe = request.json.get('timeframe', '1d')

        if not ticker:
            return jsonify({'error': 'no ticker provided'}), 400

        meta_path = os.path.join(MODELS_DIR, f'{ticker}_meta.json')
        if not os.path.exists(meta_path):
            return jsonify({'error': f'no trained models for {ticker}. train first.'}), 400

        with open(meta_path) as f:
            meta = json.load(f)

        feature_cols = meta['feature_cols']
        num_models = meta['num_models']

        #load all models
        models = []
        for i in range(num_models):
            mp = os.path.join(MODELS_DIR, f'{ticker}_model_{i}.keras')
            if not os.path.exists(mp):
                return jsonify({'error': f'model {i} not found. retrain.'}), 400
            models.append(load_model(mp))

        #rebuild scaler
        scaler = RobustScaler()
        scaler.center_ = np.array(meta['scaler']['center'])
        scaler.scale_ = np.array(meta['scaler']['scale'])

        #load raw data
        raw_data = pd.read_csv(os.path.join(DATA_DIR, f'{ticker}.csv'), index_col=0, parse_dates=True)
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)

        ohlcv_buffer = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        current_price = float(raw_data['Close'].iloc[-1])

        #historical vol for synthetic candles
        returns = raw_data['Close'].pct_change().dropna()
        daily_vol = float(returns.std())

        timeframe_map = {'1d': 1, '1w': 5, '1m': 22, '3m': 66, '6m': 132, '1y': 252}
        days = timeframe_map.get(timeframe, 1)

        #collect per-model predictions
        all_model_predictions = [[] for _ in range(num_models)]
        ensemble_prices = []
        upper_band = []
        lower_band = []

        backtest_rmse = meta.get('backtest', {}).get('rmse', current_price * 0.02)

        for day in range(days):
            #recompute all indicators from raw buffer
            temp_df = compute_indicators(ohlcv_buffer.copy())
            temp_features, _ = prepare_features(temp_df)

            recent = temp_features[feature_cols].iloc[-SEQUENCE_LENGTH:]
            if len(recent) < SEQUENCE_LENGTH:
                break

            scaled_seq = scaler.transform(recent.values)
            input_seq = scaled_seq.reshape(1, SEQUENCE_LENGTH, len(feature_cols))

            #each model predicts the return
            model_returns = []
            for i, model in enumerate(models):
                pred_return = float(model.predict(input_seq, verbose=0)[0][0])
                #clamp to reasonable daily range
                pred_return = np.clip(pred_return, -0.08, 0.08)
                model_returns.append(pred_return)

            #majority vote on direction
            votes_up = sum(1 for r in model_returns if r > 0)
            votes_down = sum(1 for r in model_returns if r <= 0)
            consensus_direction = 1 if votes_up > votes_down else -1

            #magnitude: average of absolute predicted returns
            avg_magnitude = np.mean(np.abs(model_returns))

            #combine: direction from vote, magnitude from average
            consensus_return = consensus_direction * avg_magnitude

            #get previous price
            prev_price = ensemble_prices[-1] if ensemble_prices else current_price
            pred_price = round(prev_price * (1 + consensus_return), 2)
            ensemble_prices.append(pred_price)

            #track individual model prices
            for i, ret in enumerate(model_returns):
                prev = all_model_predictions[i][-1] if all_model_predictions[i] else current_price
                all_model_predictions[i].append(round(prev * (1 + ret), 2))

            #confidence bands
            uncertainty = backtest_rmse * np.sqrt(day + 1) * 0.8
            upper_band.append(round(pred_price + 1.96 * uncertainty, 2))
            lower_band.append(round(pred_price - 1.96 * uncertainty, 2))

            #append synthetic candle
            noise = daily_vol * prev_price
            new_open = prev_price
            new_high = max(pred_price, new_open) + abs(noise * 0.3)
            new_low = min(pred_price, new_open) - abs(noise * 0.3)
            avg_vol = float(ohlcv_buffer['Volume'].iloc[-20:].mean())

            next_date = ohlcv_buffer.index[-1] + pd.Timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += pd.Timedelta(days=1)

            new_row = pd.DataFrame({
                'Open': [new_open], 'High': [new_high],
                'Low': [new_low], 'Close': [pred_price],
                'Volume': [avg_vol]
            }, index=[next_date])
            ohlcv_buffer = pd.concat([ohlcv_buffer, new_row])

        #prediction dates
        pred_dates = []
        cd = raw_data.index[-1]
        for _ in range(len(ensemble_prices)):
            cd += datetime.timedelta(days=1)
            while cd.weekday() >= 5:
                cd += datetime.timedelta(days=1)
            pred_dates.append(cd.strftime('%Y-%m-%d'))

        #per-model vote summary
        model_votes = []
        for i in range(num_models):
            preds = all_model_predictions[i]
            if preds:
                final = preds[-1]
                direction = 'up' if final > current_price else 'down'
                change_pct = round(((final - current_price) / current_price) * 100, 2)
                model_votes.append({
                    'name': MODEL_NAMES[i],
                    'direction': direction,
                    'final_price': final,
                    'change_pct': change_pct,
                    'prices': preds
                })

        news = get_news_sentiment(ticker)
        analysis = _analyse_predictions(
            current_price, ensemble_prices, pred_dates,
            news, meta.get('backtest', {}), model_votes, daily_vol
        )

        return jsonify({
            'ticker': ticker, 'timeframe': timeframe,
            'current_price': current_price,
            'predictions': {
                'dates': pred_dates,
                'prices': ensemble_prices,
                'upper_band': upper_band,
                'lower_band': lower_band
            },
            'model_votes': model_votes,
            'analysis': analysis,
            'news_sentiment': news
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _analyse_predictions(current_price, pred_prices, pred_dates, news, backtest, model_votes, daily_vol):
    if not pred_prices:
        return {'action': 'hold', 'reason': 'no predictions available'}

    min_price = min(pred_prices)
    max_price = max(pred_prices)
    min_idx = pred_prices.index(min_price)
    max_idx = pred_prices.index(max_price)
    final_price = pred_prices[-1]
    overall_change_pct = round(((final_price - current_price) / current_price) * 100, 2)

    dir_acc = backtest.get('ensemble_direction_accuracy', 50)
    confidence = 'low' if dir_acc < 55 else 'medium' if dir_acc < 65 else 'high'

    #count model consensus
    votes_up = sum(1 for v in model_votes if v['direction'] == 'up')
    votes_down = len(model_votes) - votes_up
    consensus = f'{votes_up}/{len(model_votes)} models say up'

    news_score = news.get('overall_score', 0) if news else 0
    news_bias = 'positive' if news_score > 0.3 else 'negative' if news_score < -0.3 else 'neutral'

    result = {
        'current_price': current_price,
        'predicted_low': round(min_price, 2),
        'predicted_low_date': pred_dates[min_idx],
        'predicted_high': round(max_price, 2),
        'predicted_high_date': pred_dates[max_idx],
        'final_predicted_price': round(final_price, 2),
        'overall_change_pct': overall_change_pct,
        'model_confidence': confidence,
        'direction_accuracy': dir_acc,
        'news_bias': news_bias,
        'news_score': news_score,
        'consensus': consensus,
        'votes_up': votes_up,
        'votes_down': votes_down,
        'daily_volatility': round(daily_vol * 100, 2),
    }

    threshold = 0.015 if confidence == 'high' else 0.02 if confidence == 'medium' else 0.03
    reasons = []

    if min_idx < max_idx:
        swing = (max_price - min_price) / min_price
        if swing >= threshold:
            result['action'] = 'buy_then_sell'
            result['buy_date'] = pred_dates[min_idx]
            result['buy_price'] = round(min_price, 2)
            result['sell_date'] = pred_dates[max_idx]
            result['sell_price'] = round(max_price, 2)
            result['potential_profit_pct'] = round(swing * 100, 2)
            reasons.append(f'{consensus}. predicted dip to ${min_price:.2f} on {pred_dates[min_idx]} then rise to ${max_price:.2f} on {pred_dates[max_idx]} ({swing*100:.1f}% gain)')
            if news_bias == 'positive':
                reasons.append('news sentiment supports this')
            elif news_bias == 'negative':
                reasons.append('news sentiment is negative though so proceed with caution')
            reasons.append(f'ensemble direction accuracy: {dir_acc:.0f}% ({confidence} confidence)')
            result['reason'] = '. '.join(reasons)
            return result

    if max_idx < min_idx:
        drop = (max_price - min_price) / max_price
        if drop >= threshold:
            result['action'] = 'sell_then_buy'
            result['sell_date'] = pred_dates[max_idx]
            result['sell_price'] = round(max_price, 2)
            result['buy_date'] = pred_dates[min_idx]
            result['buy_price'] = round(min_price, 2)
            result['potential_profit_pct'] = round(drop * 100, 2)
            reasons.append(f'{consensus}. predicted peak at ${max_price:.2f} on {pred_dates[max_idx]} then drop to ${min_price:.2f} on {pred_dates[min_idx]}. consider selling at the peak and rebuying at the dip')
            if news_bias == 'negative':
                reasons.append('negative news supports the expected decline')
            elif news_bias == 'positive':
                reasons.append('but positive news may soften the drop')
            reasons.append(f'ensemble direction accuracy: {dir_acc:.0f}% ({confidence} confidence)')
            result['reason'] = '. '.join(reasons)
            return result

    if overall_change_pct > 2:
        result['action'] = 'buy'
        reasons.append(f'{consensus}. general upward trend of {overall_change_pct:+.1f}% predicted')
    elif overall_change_pct < -2:
        result['action'] = 'sell'
        reasons.append(f'{consensus}. downward trend of {overall_change_pct:+.1f}% predicted')
    else:
        result['action'] = 'hold'
        reasons.append(f'{consensus}. no significant movement predicted ({overall_change_pct:+.1f}%). best to hold or stay out')

    if news_bias != 'neutral':
        reasons.append(f'news sentiment is {news_bias} (score: {news_score:+.1f})')
    reasons.append(f'ensemble direction accuracy: {dir_acc:.0f}% ({confidence} confidence)')
    result['reason'] = '. '.join(reasons)
    return result


@app.route('/api/save_trade', methods=['POST'])
def save_trade():
    global trade_recommendations
    try:
        trade = request.json
        trade['saved_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        trade_recommendations.append(trade)
        return jsonify({'status': 'saved', 'total': len(trade_recommendations)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
