'''
merlin 5.1 - stock market prediction tool
rebuilt with technical indicators, news sentiment, backtesting
and prediction logic that doesn't collapse to zero
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
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import threading
import re

app = Flask(__name__)

#store training status globally
training_status = {
    'active': False,
    'progress': 0,
    'message': '',
    'ticker': '',
    'complete': False,
    'error': None
}

#store trade recommendations
trade_recommendations = []

MODEL_DIR = 'model'
DATA_DIR = 'data'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

SEQUENCE_LENGTH = 60


def compute_technical_indicators(df):
    '''
    add technical indicators to the dataframe so the model
    has more than just raw price to learn from
    '''
    close = df['Close'].values.flatten()
    high = df['High'].values.flatten()
    low = df['Low'].values.flatten()
    volume = df['Volume'].values.flatten()

    #simple moving averages
    df['SMA_10'] = pd.Series(close).rolling(window=10).mean().values
    df['SMA_20'] = pd.Series(close).rolling(window=20).mean().values
    df['SMA_50'] = pd.Series(close).rolling(window=50).mean().values

    #exponential moving averages
    df['EMA_12'] = pd.Series(close).ewm(span=12, adjust=False).mean().values
    df['EMA_26'] = pd.Series(close).ewm(span=26, adjust=False).mean().values

    #macd
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = pd.Series(df['MACD']).ewm(span=9, adjust=False).mean().values

    #rsi (14 period)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = (100 - (100 / (1 + rs))).values

    #bollinger bands
    sma20 = pd.Series(close).rolling(window=20).mean()
    std20 = pd.Series(close).rolling(window=20).std()
    df['BB_Upper'] = (sma20 + 2 * std20).values
    df['BB_Lower'] = (sma20 - 2 * std20).values
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / (sma20.values + 1e-10))

    #average true range (volatility)
    tr1 = pd.Series(high) - pd.Series(low)
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean().values

    #volume moving average ratio
    vol_sma = pd.Series(volume, dtype=float).rolling(window=20).mean()
    df['Vol_Ratio'] = (pd.Series(volume, dtype=float) / (vol_sma + 1e-10)).values

    #price rate of change
    df['ROC_5'] = pd.Series(close).pct_change(periods=5).values
    df['ROC_10'] = pd.Series(close).pct_change(periods=10).values

    #daily returns
    df['Returns'] = pd.Series(close).pct_change().values

    return df


def prepare_features(df):
    '''select and clean feature columns'''
    feature_cols = [
        'Close', 'Volume', 'SMA_10', 'SMA_20', 'SMA_50',
        'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI',
        'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR',
        'Vol_Ratio', 'ROC_5', 'ROC_10', 'Returns'
    ]
    available = [c for c in feature_cols if c in df.columns]
    features = df[available].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna()
    return features, available


def get_news_sentiment(ticker):
    '''
    grab recent news for a ticker and do basic sentiment scoring
    using keyword matching
    '''
    try:
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return {'articles': [], 'overall_score': 0, 'summary': 'no recent news found', 'count': 0}

        positive_words = [
            'surge', 'surges', 'soar', 'soars', 'jump', 'jumps', 'gain', 'gains',
            'rally', 'rallies', 'rise', 'rises', 'climb', 'climbs', 'high',
            'record', 'beat', 'beats', 'exceed', 'exceeds', 'strong', 'bullish',
            'growth', 'profit', 'upgrade', 'buy', 'outperform', 'positive',
            'boost', 'boosts', 'recover', 'recovery', 'optimistic', 'upbeat',
            'breakthrough', 'innovation', 'expand', 'expansion', 'revenue',
            'earnings', 'dividend', 'approval', 'partnership', 'deal'
        ]

        negative_words = [
            'drop', 'drops', 'fall', 'falls', 'decline', 'declines', 'plunge',
            'plunges', 'crash', 'crashes', 'sink', 'sinks', 'down', 'low',
            'loss', 'losses', 'miss', 'misses', 'weak', 'bearish', 'sell',
            'downgrade', 'underperform', 'negative', 'cut', 'cuts', 'slash',
            'warning', 'risk', 'fear', 'concern', 'worry', 'recession',
            'lawsuit', 'investigation', 'fine', 'penalty', 'recall',
            'bankruptcy', 'layoff', 'layoffs', 'debt', 'deficit'
        ]

        articles = []
        total_score = 0

        for item in news[:10]:
            title = item.get('title', '')
            publisher = item.get('publisher', '')
            link = item.get('link', '')
            pub_date = ''
            if 'providerPublishTime' in item:
                pub_date = datetime.datetime.fromtimestamp(
                    item['providerPublishTime']
                ).strftime('%Y-%m-%d %H:%M')

            title_lower = title.lower()
            words = re.findall(r'\b\w+\b', title_lower)
            pos_count = sum(1 for w in words if w in positive_words)
            neg_count = sum(1 for w in words if w in negative_words)

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
                'date': pub_date,
                'link': link,
                'sentiment': sentiment,
                'score': score
            })

        num_articles = len(articles)
        avg_score = total_score / max(num_articles, 1)

        if avg_score > 0.5:
            summary = 'news sentiment is generally positive'
        elif avg_score < -0.5:
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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/fetch_data', methods=['POST'])
def fetch_data():
    '''fetch stock data for a given ticker'''
    try:
        ticker = request.json.get('ticker', '').upper().strip()
        if not ticker:
            return jsonify({'error': 'no ticker provided'}), 400

        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=5*365)

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
    '''kick off model training in a background thread'''
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
    '''background training with improved architecture and multi-feature input'''
    global training_status
    try:
        training_status['message'] = 'loading and preparing data...'
        training_status['progress'] = 5

        csv_path = os.path.join(DATA_DIR, f'{ticker}.csv')
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        features_df, feature_cols = prepare_features(data)
        num_features = len(feature_cols)

        training_status['message'] = f'scaling {num_features} features...'
        training_status['progress'] = 10

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features_df.values)

        close_idx = feature_cols.index('Close')
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
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(SEQUENCE_LENGTH, num_features)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='huber')

        early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

        training_status['message'] = 'training model...'
        training_status['progress'] = 20

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

            if early_stop.stopped_epoch > 0:
                training_status['message'] = f'early stopping at epoch {epoch + 1} (no improvement for 8 epochs)'
                break

        #backtest on validation set
        training_status['message'] = 'running backtest on validation data...'
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

        model_meta = {
            'feature_cols': feature_cols,
            'num_features': num_features,
            'close_idx': close_idx,
            'backtest': backtest_results
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
    predictions that stay sane by clamping daily returns
    to historically reasonable bounds and decaying confidence
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

        csv_path = os.path.join(DATA_DIR, f'{ticker}.csv')
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        features_df, _ = prepare_features(data)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(features_df[feature_cols].values)

        close_scaler = MinMaxScaler(feature_range=(0, 1))
        close_scaler.fit(features_df[['Close']].values)

        scaled_data = scaler.transform(features_df[feature_cols].values)

        timeframe_map = {
            '1d': 1,
            '1w': 5,
            '1m': 22,
            '3m': 66,
            '6m': 132,
            '1y': 252
        }
        days_to_predict = timeframe_map.get(timeframe, 1)

        last_sequence = scaled_data[-SEQUENCE_LENGTH:]
        current_price = float(features_df['Close'].iloc[-1])

        #historical stats for bounding predictions
        returns = features_df['Close'].pct_change().dropna()
        daily_vol = float(returns.std())
        avg_daily_return = float(returns.mean())

        predictions = []
        current_seq = last_sequence.copy()

        for day in range(days_to_predict):
            input_seq = current_seq.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
            pred_scaled = model.predict(input_seq, verbose=0)[0][0]

            pred_price = float(close_scaler.inverse_transform([[pred_scaled]])[0][0])

            prev_price = predictions[-1] if predictions else current_price

            implied_return = (pred_price - prev_price) / (prev_price + 1e-10)

            #clamp daily returns to 3x historical volatility
            max_daily_move = max(daily_vol * 3, 0.015)
            clamped_return = np.clip(implied_return, -max_daily_move, max_daily_move)

            #decay confidence for further out predictions
            confidence_decay = 1.0 / (1.0 + day * 0.008)
            adjusted_return = clamped_return * confidence_decay + avg_daily_return * (1 - confidence_decay)

            predicted_price = prev_price * (1 + adjusted_return)
            predictions.append(round(float(predicted_price), 2))

            #slide window forward
            new_row = current_seq[-1].copy()
            new_row[close_idx] = pred_scaled
            current_seq = np.vstack([current_seq[1:], new_row.reshape(1, -1)])

        #prediction dates skipping weekends
        last_date = features_df.index[-1]
        pred_dates = []
        current_date = last_date
        for _ in range(days_to_predict):
            current_date += datetime.timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += datetime.timedelta(days=1)
            pred_dates.append(current_date.strftime('%Y-%m-%d'))

        news = get_news_sentiment(ticker)
        analysis = _analyse_predictions(current_price, predictions, pred_dates, news, meta.get('backtest', {}))

        return jsonify({
            'ticker': ticker,
            'timeframe': timeframe,
            'current_price': current_price,
            'predictions': {
                'dates': pred_dates,
                'prices': predictions
            },
            'analysis': analysis,
            'news_sentiment': news
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _analyse_predictions(current_price, pred_prices, pred_dates, news, backtest):
    '''find optimal buy/sell points factoring in news and model confidence'''
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
    }

    base_threshold = 0.02
    if confidence == 'low':
        threshold = base_threshold * 1.5
    elif confidence == 'high':
        threshold = base_threshold * 0.8
    else:
        threshold = base_threshold

    reasons = []

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
