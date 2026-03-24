'''
merlin 5.2 - stock market prediction tool
fixed prediction loop, news sentiment and model architecture
'''

import os
import json
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten
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
    'error': None
}

trade_recommendations = []

MODEL_DIR = 'model'
DATA_DIR = 'data'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

SEQUENCE_LENGTH = 60


def compute_technical_indicators(df):
    '''
    compute all technical indicators on raw ohlcv data.
    returns the dataframe with new columns added.
    '''
    df = df.copy()

    close = df['Close'].values.flatten().astype(float)
    high = df['High'].values.flatten().astype(float)
    low = df['Low'].values.flatten().astype(float)
    volume = df['Volume'].values.flatten().astype(float)

    n = len(close)

    #simple moving averages
    df['SMA_10'] = pd.Series(close).rolling(window=10, min_periods=1).mean().values
    df['SMA_20'] = pd.Series(close).rolling(window=20, min_periods=1).mean().values
    df['SMA_50'] = pd.Series(close).rolling(window=50, min_periods=1).mean().values

    #exponential moving averages
    df['EMA_12'] = pd.Series(close).ewm(span=12, adjust=False).mean().values
    df['EMA_26'] = pd.Series(close).ewm(span=26, adjust=False).mean().values

    #macd
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = pd.Series(df['MACD'].values).ewm(span=9, adjust=False).mean().values

    #rsi 14
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = (100 - (100 / (1 + rs))).values

    #bollinger bands
    sma20 = pd.Series(close).rolling(window=20, min_periods=1).mean()
    std20 = pd.Series(close).rolling(window=20, min_periods=1).std().fillna(0)
    df['BB_Upper'] = (sma20 + 2 * std20).values
    df['BB_Lower'] = (sma20 - 2 * std20).values
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / (sma20.values + 1e-10))

    #average true range
    tr1 = pd.Series(high) - pd.Series(low)
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14, min_periods=1).mean().values

    #volume ratio
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=1).mean()
    df['Vol_Ratio'] = (pd.Series(volume) / (vol_sma + 1e-10)).values

    #rate of change
    df['ROC_5'] = pd.Series(close).pct_change(periods=5).fillna(0).values
    df['ROC_10'] = pd.Series(close).pct_change(periods=10).fillna(0).values

    #daily returns
    df['Returns'] = pd.Series(close).pct_change().fillna(0).values

    #replace any remaining nans or infs
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    return df


FEATURE_COLS = [
    'Close', 'High', 'Low', 'Open', 'Volume',
    'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI',
    'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR',
    'Vol_Ratio', 'ROC_5', 'ROC_10', 'Returns'
]


def prepare_features(df):
    '''select feature columns that exist in the dataframe'''
    available = [c for c in FEATURE_COLS if c in df.columns]
    features = df[available].copy()
    features = features.replace([np.inf, -np.inf], 0)
    features = features.fillna(0)
    return features, available


def get_news_sentiment(ticker):
    '''
    fetch news from yfinance and score sentiment.
    handles both old and new yfinance api formats.
    '''
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news

        if not raw_news:
            return {'articles': [], 'overall_score': 0, 'summary': 'no recent news found', 'count': 0}

        positive_words = {
            'surge', 'surges', 'soar', 'soars', 'jump', 'jumps', 'gain', 'gains',
            'rally', 'rallies', 'rise', 'rises', 'climb', 'climbs', 'high',
            'record', 'beat', 'beats', 'exceed', 'exceeds', 'strong', 'bullish',
            'growth', 'profit', 'upgrade', 'buy', 'outperform', 'positive',
            'boost', 'boosts', 'recover', 'recovery', 'optimistic', 'upbeat',
            'breakthrough', 'innovation', 'expand', 'expansion', 'revenue',
            'earnings', 'dividend', 'approval', 'partnership', 'deal', 'success',
            'win', 'wins', 'top', 'tops', 'higher', 'up', 'upside', 'above'
        }

        negative_words = {
            'drop', 'drops', 'fall', 'falls', 'decline', 'declines', 'plunge',
            'plunges', 'crash', 'crashes', 'sink', 'sinks', 'down', 'low',
            'loss', 'losses', 'miss', 'misses', 'weak', 'bearish', 'sell',
            'downgrade', 'underperform', 'negative', 'cut', 'cuts', 'slash',
            'warning', 'risk', 'fear', 'concern', 'worry', 'recession',
            'lawsuit', 'investigation', 'fine', 'penalty', 'recall',
            'bankruptcy', 'layoff', 'layoffs', 'debt', 'deficit', 'lower',
            'below', 'worst', 'trouble', 'crisis', 'slump', 'tumble'
        }

        articles = []
        total_score = 0

        #handle both old format (list of dicts) and new format (list with nested content)
        items_to_process = []
        for item in raw_news[:15]:
            title = ''
            publisher = ''
            link = ''
            pub_date = ''

            #new yfinance format (0.2.36+)
            if isinstance(item, dict) and 'content' in item:
                content = item['content']
                title = content.get('title', '')
                publisher = content.get('provider', {}).get('displayName', '') if isinstance(content.get('provider'), dict) else ''
                link = content.get('canonicalUrl', {}).get('url', '') if isinstance(content.get('canonicalUrl'), dict) else ''
                pub_date = content.get('pubDate', '')
            #old yfinance format
            elif isinstance(item, dict):
                title = item.get('title', '')
                publisher = item.get('publisher', '')
                link = item.get('link', '')
                if 'providerPublishTime' in item:
                    try:
                        pub_date = datetime.datetime.fromtimestamp(
                            item['providerPublishTime']
                        ).strftime('%Y-%m-%d %H:%M')
                    except:
                        pub_date = ''

            if not title:
                continue

            #score the headline
            title_lower = title.lower()
            words = set(re.findall(r'\b\w+\b', title_lower))
            pos_count = len(words & positive_words)
            neg_count = len(words & negative_words)

            if pos_count > neg_count:
                sentiment = 'positive'
                score = min(pos_count - neg_count, 3)
            elif neg_count > pos_count:
                sentiment = 'negative'
                score = -min(neg_count - pos_count, 3)
            else:
                sentiment = 'neutral'
                score = 0

            total_score += score

            articles.append({
                'title': title,
                'publisher': publisher,
                'date': pub_date[:16] if pub_date else '',
                'link': link,
                'sentiment': sentiment,
                'score': score
            })

        num_articles = len(articles)
        if num_articles == 0:
            return {'articles': [], 'overall_score': 0, 'summary': 'no recent news found', 'count': 0}

        avg_score = total_score / num_articles

        if avg_score > 0.4:
            summary = 'news sentiment is generally positive'
        elif avg_score < -0.4:
            summary = 'news sentiment is generally negative'
        else:
            summary = 'news sentiment is mixed or neutral'

        return {
            'articles': articles,
            'overall_score': round(avg_score, 2),
            'summary': summary,
            'count': num_articles
        }

    except Exception as e:
        return {
            'articles': [],
            'overall_score': 0,
            'summary': f'could not fetch news: {str(e)}',
            'count': 0
        }


def recompute_features_for_prediction(ohlcv_buffer):
    '''
    given a buffer of raw ohlcv rows (as a dataframe with columns
    Open, High, Low, Close, Volume), recompute all technical indicators
    and return the last SEQUENCE_LENGTH rows as a scaled feature array.
    this is the fix for the flat-line bug: we recompute everything
    from raw price data each step instead of just patching one column.
    '''
    df = ohlcv_buffer.copy()
    df = compute_technical_indicators(df)
    features, cols = prepare_features(df)
    return features, cols


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

        data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d',
            progress=False
        )

        if data.empty or len(data) < 100:
            return jsonify({'error': f'not enough data found for {ticker}. check the symbol is correct.'}), 400

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = compute_technical_indicators(data)

        csv_path = os.path.join(DATA_DIR, f'{ticker}.csv')
        data.to_csv(csv_path)

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
            'ticker': ticker,
            'current_price': current_price,
            'change': change,
            'change_pct': change_pct,
            'data_points': len(data),
            'chart_data': chart_data,
            'news': news,
            'date_range': {
                'start': data.index[0].strftime('%Y-%m-%d'),
                'end': data.index[-1].strftime('%Y-%m-%d')
            }
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
        return jsonify({'error': f'no data found for {ticker}. fetch data first.'}), 400

    training_status = {
        'active': True,
        'progress': 0,
        'message': 'starting training...',
        'ticker': ticker,
        'complete': False,
        'error': None
    }

    thread = threading.Thread(target=_train_model, args=(ticker, epochs))
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'training started', 'ticker': ticker})


def _train_model(ticker, epochs):
    global training_status
    try:
        training_status['message'] = 'loading data...'
        training_status['progress'] = 5

        csv_path = os.path.join(DATA_DIR, f'{ticker}.csv')
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        features_df, feature_cols = prepare_features(data)
        num_features = len(feature_cols)
        close_idx = feature_cols.index('Close')

        training_status['message'] = f'scaling {num_features} features...'
        training_status['progress'] = 10

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features_df.values)

        close_scaler = MinMaxScaler(feature_range=(0, 1))
        close_scaler.fit(features_df[['Close']].values)

        training_status['message'] = 'creating sequences...'
        training_status['progress'] = 15

        X, y = [], []
        for i in range(len(scaled_data) - SEQUENCE_LENGTH):
            X.append(scaled_data[i:i + SEQUENCE_LENGTH])
            y.append(scaled_data[i + SEQUENCE_LENGTH, close_idx])
        X = np.array(X)
        y = np.array(y)

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        training_status['message'] = 'building model...'
        training_status['progress'] = 18

        model = Sequential()
        #conv1d to extract local patterns
        model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same',
                         input_shape=(SEQUENCE_LENGTH, num_features),
                         kernel_regularizer=l2(1e-4)))
        model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same',
                         kernel_regularizer=l2(1e-4)))
        model.add(Dropout(0.2))
        #bidirectional lstms
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(Dropout(0.25))
        model.add(Bidirectional(LSTM(50)))
        model.add(Dropout(0.25))
        #dense head
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(1e-4)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='huber')

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

        training_status['message'] = 'training model...'
        training_status['progress'] = 20

        stopped_early = False
        for epoch in range(epochs):
            history = model.fit(
                X_train, y_train,
                epochs=1,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            progress = 20 + int((epoch + 1) / epochs * 60)
            train_loss = history.history['loss'][0]
            val_loss = history.history['val_loss'][0]
            training_status['progress'] = progress
            training_status['message'] = f'epoch {epoch + 1}/{epochs} | loss: {train_loss:.6f} | val_loss: {val_loss:.6f}'

            if early_stop.stopped_epoch > 0 and not stopped_early:
                training_status['message'] = f'early stopping at epoch {epoch + 1}'
                stopped_early = True
                break

        #backtest
        training_status['message'] = 'running backtest...'
        training_status['progress'] = 85

        val_predictions = model.predict(X_val, verbose=0)
        val_pred_prices = close_scaler.inverse_transform(val_predictions.reshape(-1, 1)).flatten()
        val_actual_prices = close_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

        mae = mean_absolute_error(val_actual_prices, val_pred_prices)
        rmse = np.sqrt(mean_squared_error(val_actual_prices, val_pred_prices))
        mape = np.mean(np.abs((val_actual_prices - val_pred_prices) / (val_actual_prices + 1e-10))) * 100

        actual_direction = np.diff(val_actual_prices) > 0
        pred_direction = np.diff(val_pred_prices) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100

        backtest_results = {
            'mae': round(float(mae), 2),
            'rmse': round(float(rmse), 2),
            'mape': round(float(mape), 2),
            'direction_accuracy': round(float(direction_accuracy), 1),
            'val_predictions': [round(float(p), 2) for p in val_pred_prices[-60:]],
            'val_actual': [round(float(a), 2) for a in val_actual_prices[-60:]],
        }

        training_status['message'] = 'saving model...'
        training_status['progress'] = 95

        model_path = os.path.join(MODEL_DIR, f'{ticker}_model.keras')
        model.save(model_path)

        #save scaler parameters for reconstruction
        scaler_info = {
            'data_min': scaler.data_min_.tolist(),
            'data_max': scaler.data_max_.tolist(),
            'data_range': scaler.data_range_.tolist(),
            'close_min': float(close_scaler.data_min_[0]),
            'close_max': float(close_scaler.data_max_[0]),
        }

        model_meta = {
            'feature_cols': feature_cols,
            'num_features': num_features,
            'close_idx': close_idx,
            'backtest': backtest_results,
            'scaler': scaler_info
        }

        with open(os.path.join(MODEL_DIR, f'{ticker}_meta.json'), 'w') as f:
            json.dump(model_meta, f)

        training_status['message'] = 'training complete!'
        training_status['progress'] = 100
        training_status['complete'] = True
        training_status['active'] = False
        training_status['backtest'] = backtest_results

    except Exception as e:
        training_status['error'] = str(e)
        training_status['active'] = False
        training_status['message'] = f'error: {str(e)}'


@app.route('/api/training_status')
def get_training_status():
    return jsonify(training_status)


@app.route('/api/predict', methods=['POST'])
def predict():
    '''
    the key fix in 5.2: instead of just patching the close column
    in a stale feature vector, we maintain a rolling buffer of raw
    ohlcv data and recompute ALL technical indicators from scratch
    at each prediction step. this stops the flat-line problem.
    '''
    try:
        ticker = request.json.get('ticker', '').upper().strip()
        timeframe = request.json.get('timeframe', '1d')

        if not ticker:
            return jsonify({'error': 'no ticker provided'}), 400

        model_path = os.path.join(MODEL_DIR, f'{ticker}_model.keras')
        meta_path = os.path.join(MODEL_DIR, f'{ticker}_meta.json')

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            return jsonify({'error': f'no trained model found for {ticker}. train one first.'}), 400

        model = load_model(model_path)

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        feature_cols = meta['feature_cols']
        close_idx = meta['close_idx']

        #rebuild scalers from saved params
        scaler_info = meta.get('scaler', {})

        csv_path = os.path.join(DATA_DIR, f'{ticker}.csv')
        raw_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)

        #we need the raw ohlcv for recomputing indicators
        #keep a generous buffer for indicator warm-up (need ~50 rows for sma50)
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        ohlcv_buffer = raw_data[ohlcv_cols].copy()

        #fit scalers on the full dataset with indicators
        full_features_df, _ = prepare_features(raw_data)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(full_features_df[feature_cols].values)

        close_scaler = MinMaxScaler(feature_range=(0, 1))
        close_scaler.fit(full_features_df[['Close']].values)

        timeframe_map = {
            '1d': 1,
            '1w': 5,
            '1m': 22,
            '3m': 66,
            '6m': 132,
            '1y': 252
        }
        days_to_predict = timeframe_map.get(timeframe, 1)

        current_price = float(raw_data['Close'].iloc[-1])

        #historical volatility for confidence bands
        returns = raw_data['Close'].pct_change().dropna()
        daily_vol = float(returns.std())
        avg_daily_return = float(returns.mean())

        #the fix: maintain a rolling raw ohlcv buffer
        #and recompute indicators from scratch each step
        predictions = []
        upper_band = []
        lower_band = []

        backtest_rmse = meta.get('backtest', {}).get('rmse', current_price * 0.02)

        for day in range(days_to_predict):
            #recompute all indicators from the full ohlcv buffer
            temp_df = compute_technical_indicators(ohlcv_buffer.copy())
            temp_features, _ = prepare_features(temp_df)

            #get the last SEQUENCE_LENGTH rows, scale them
            recent = temp_features[feature_cols].iloc[-SEQUENCE_LENGTH:]
            if len(recent) < SEQUENCE_LENGTH:
                #not enough data, pad with what we have
                break

            scaled_seq = scaler.transform(recent.values)
            input_seq = scaled_seq.reshape(1, SEQUENCE_LENGTH, len(feature_cols))

            #predict
            pred_scaled = model.predict(input_seq, verbose=0)[0][0]
            pred_price = float(close_scaler.inverse_transform([[pred_scaled]])[0][0])

            #sanity clamp: limit daily move to 5x historical vol
            prev_price = predictions[-1] if predictions else current_price
            implied_return = (pred_price - prev_price) / (prev_price + 1e-10)
            max_move = max(daily_vol * 5, 0.02)
            clamped_return = np.clip(implied_return, -max_move, max_move)
            pred_price = prev_price * (1 + clamped_return)
            pred_price = round(float(pred_price), 2)

            predictions.append(pred_price)

            #confidence bands (95% ci using backtest rmse, grows with sqrt of time)
            uncertainty = backtest_rmse * np.sqrt(day + 1) * 1.0
            upper_band.append(round(pred_price + 1.96 * uncertainty, 2))
            lower_band.append(round(pred_price - 1.96 * uncertainty, 2))

            #append a synthetic candle to the ohlcv buffer
            #simulate realistic ohlc from predicted close
            noise = daily_vol * pred_price
            new_open = prev_price
            new_high = max(pred_price, new_open) + abs(noise * 0.3)
            new_low = min(pred_price, new_open) - abs(noise * 0.3)
            new_close = pred_price
            avg_volume = float(ohlcv_buffer['Volume'].iloc[-20:].mean())

            last_date = ohlcv_buffer.index[-1]
            next_date = last_date + pd.Timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += pd.Timedelta(days=1)

            new_row = pd.DataFrame({
                'Open': [new_open],
                'High': [new_high],
                'Low': [new_low],
                'Close': [new_close],
                'Volume': [avg_volume]
            }, index=[next_date])

            ohlcv_buffer = pd.concat([ohlcv_buffer, new_row])

        #prediction dates
        last_date = raw_data.index[-1]
        pred_dates = []
        current_date = last_date
        for _ in range(len(predictions)):
            current_date += datetime.timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += datetime.timedelta(days=1)
            pred_dates.append(current_date.strftime('%Y-%m-%d'))

        news = get_news_sentiment(ticker)
        analysis = _analyse_predictions(
            current_price, predictions, pred_dates,
            news, meta.get('backtest', {}), daily_vol
        )

        return jsonify({
            'ticker': ticker,
            'timeframe': timeframe,
            'current_price': current_price,
            'predictions': {
                'dates': pred_dates,
                'prices': predictions,
                'upper_band': upper_band,
                'lower_band': lower_band
            },
            'analysis': analysis,
            'news_sentiment': news
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _analyse_predictions(current_price, pred_prices, pred_dates, news, backtest, daily_vol):
    '''analyse predictions factoring in confidence, news and volatility'''
    if not pred_prices:
        return {'action': 'hold', 'reason': 'no predictions available'}

    min_price = min(pred_prices)
    max_price = max(pred_prices)
    min_idx = pred_prices.index(min_price)
    max_idx = pred_prices.index(max_price)
    final_price = pred_prices[-1]

    overall_change_pct = round(((final_price - current_price) / current_price) * 100, 2)

    direction_acc = backtest.get('direction_accuracy', 50)
    confidence = 'low' if direction_acc < 55 else 'medium' if direction_acc < 65 else 'high'

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
        'direction_accuracy': direction_acc,
        'news_bias': news_bias,
        'news_score': news_score,
        'daily_volatility': round(daily_vol * 100, 2),
    }

    #threshold based on confidence
    base_threshold = 0.02
    if confidence == 'low':
        threshold = base_threshold * 1.5
    elif confidence == 'high':
        threshold = base_threshold * 0.75
    else:
        threshold = base_threshold

    reasons = []

    #buy then sell (dip before peak)
    if min_idx < max_idx:
        swing_pct = (max_price - min_price) / min_price
        if swing_pct >= threshold:
            result['action'] = 'buy_then_sell'
            result['buy_date'] = pred_dates[min_idx]
            result['buy_price'] = round(min_price, 2)
            result['sell_date'] = pred_dates[max_idx]
            result['sell_price'] = round(max_price, 2)
            result['potential_profit_pct'] = round(swing_pct * 100, 2)
            reasons.append(f'model predicts a dip to ${min_price:.2f} on {pred_dates[min_idx]} followed by a rise to ${max_price:.2f} on {pred_dates[max_idx]} ({swing_pct*100:.1f}% potential gain)')
            if news_bias == 'positive':
                reasons.append('news sentiment supports this outlook')
            elif news_bias == 'negative':
                reasons.append('however news sentiment is negative so proceed with caution')
            reasons.append(f'model direction accuracy: {direction_acc:.0f}% ({confidence} confidence)')
            result['reason'] = '. '.join(reasons)
            return result

    #sell then rebuy (peak before dip)
    if max_idx < min_idx:
        drop_pct = (max_price - min_price) / max_price
        if drop_pct >= threshold:
            result['action'] = 'sell_then_buy'
            result['sell_date'] = pred_dates[max_idx]
            result['sell_price'] = round(max_price, 2)
            result['buy_date'] = pred_dates[min_idx]
            result['buy_price'] = round(min_price, 2)
            result['potential_profit_pct'] = round(drop_pct * 100, 2)
            reasons.append(f'model predicts a peak at ${max_price:.2f} on {pred_dates[max_idx]} then a drop to ${min_price:.2f} on {pred_dates[min_idx]}. if you hold this stock consider selling at the peak and rebuying at the dip')
            if news_bias == 'negative':
                reasons.append('negative news sentiment supports the expected decline')
            elif news_bias == 'positive':
                reasons.append('but positive news sentiment may soften the predicted drop')
            reasons.append(f'model direction accuracy: {direction_acc:.0f}% ({confidence} confidence)')
            result['reason'] = '. '.join(reasons)
            return result

    #general trend
    if overall_change_pct > 2:
        result['action'] = 'buy'
        reasons.append(f'model predicts a general upward trend of {overall_change_pct:+.1f}% over this period')
    elif overall_change_pct < -2:
        result['action'] = 'sell'
        reasons.append(f'model predicts a downward trend of {overall_change_pct:+.1f}% over this period')
    else:
        result['action'] = 'hold'
        reasons.append(f'no significant movement predicted ({overall_change_pct:+.1f}%). best to hold or stay out')

    if news_bias != 'neutral':
        reasons.append(f'news sentiment is {news_bias} (score: {news_score:+.1f})')
    reasons.append(f'model direction accuracy: {direction_acc:.0f}% ({confidence} confidence)')
    result['reason'] = '. '.join(reasons)
    return result


@app.route('/api/backtest')
def get_backtest():
    ticker = request.args.get('ticker', '').upper().strip()
    meta_path = os.path.join(MODEL_DIR, f'{ticker}_meta.json')
    if not os.path.exists(meta_path):
        return jsonify({'error': 'no model found'}), 404
    with open(meta_path) as f:
        meta = json.load(f)
    return jsonify(meta.get('backtest', {}))


@app.route('/api/news', methods=['POST'])
def get_news():
    ticker = request.json.get('ticker', '').upper().strip()
    if not ticker:
        return jsonify({'error': 'no ticker'}), 400
    return jsonify(get_news_sentiment(ticker))


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
