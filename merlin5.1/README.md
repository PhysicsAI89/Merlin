# Merlin 5.0

LSTM stock prediction engine with a clean web interface.

## What it does

- Type in any stock ticker (e.g. V for Visa, AAPL for Apple, TSLA for Tesla)
- Fetches 5 years of daily price data automatically
- Shows you an interactive price chart with range controls
- Train an LSTM model on the data with adjustable epochs
- Run predictions across multiple timeframes: 1 day, 1 week, 1 month, 3 months, 6 months or 1 year
- Tells you the best time to buy and sell (or to hold if nothing looks worthwhile)
- Save trade recommendations to a table for reference

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in your browser.

## How to use

1. Type a ticker symbol in the input box and hit fetch data
2. Once the chart loads, set your epochs (25 is a decent default) and click train model
3. Wait for training to finish (the progress bar will keep you updated)
4. Pick a timeframe and the model will predict price movements
5. If it spots a good trade opportunity, save it to the table

## Model details

The model is a 3-layer LSTM with dropout regularisation:
- LSTM(128) -> Dropout(0.2)
- LSTM(64) -> Dropout(0.2)
- LSTM(32) -> Dropout(0.2)
- Dense(16, relu) -> Dense(1)

Uses a 60-day lookback window with MinMaxScaler normalisation.
Early stopping on validation loss with patience of 5 epochs.

## Notes

- This is not financial advice. The model learns from historical patterns and there is no guarantee those patterns will repeat.
- Longer prediction timeframes (6m, 1y) are less reliable than shorter ones due to compounding prediction error.
- Training a model takes a minute or two depending on your hardware and the number of epochs.
