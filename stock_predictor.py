import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import logging
from requests.exceptions import HTTPError
from datetime import datetime, timedelta

# Required: ta library for indicators
try:
    from ta.trend import MACD
    from ta.momentum import rsi, StochasticOscillator
    from ta.volatility import BollingerBands, AverageTrueRange
except ImportError:
    logging.error("ta library not installed. Install with: pip install ta")
    exit(1)

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function to get the last trading day
def get_last_trading_day(current_date=None):
    try:
        from pandas_market_calendars import get_calendar
        nyse = get_calendar('NYSE')
        if current_date is None:
            current_date = pd.Timestamp.today()
        schedule = nyse.schedule(start_date=current_date - pd.Timedelta(days=7), end_date=current_date)
        last_trading_day = schedule.index[-1].strftime('%Y-%m-%d')
        logging.info(f"Last trading day: {last_trading_day}")
        return last_trading_day
    except ImportError:
        logging.warning("pandas_market_calendars not installed. Using rule-based fallback.")
        if current_date is None:
            current_date = datetime.now()
        for i in range(7):
            check_date = current_date - timedelta(days=i)
            if check_date.weekday() < 5:  # Monday (0) to Friday (4)
                if check_date.month == 7 and check_date.day == 4:
                    continue
                last_trading_day = check_date.strftime('%Y-%m-%d')
                logging.info(f"Last trading day (fallback): {last_trading_day}")
                return last_trading_day
        logging.error("Could not determine last trading day.")
        exit(1)

# Step 1: Set up Alpaca API
ALPACA_API_KEY = "ALPACA_API_KEY" 
ALPACA_SECRET_KEY = "ALPACA_SECRET_KEY" 
BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading for testing
try:
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')
    logging.info("Alpaca API initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Alpaca API: {e}")
    exit(1)

# Step 2: Get stock symbol from user input
while True:
    symbol = input("Enter stock symbol (e.g., NVDL, AAPL, CCL): ").strip().upper()
    if symbol.isalnum() and len(symbol) <= 10:  # Basic validation for stock symbols
        break
    else:
        logging.warning("Invalid stock symbol. Please enter a valid ticker (e.g., NVDL, AAPL, CCL).")

# Step 3: Fetch historical data
timeframe = '1Day'
start_date = '2022-01-01'  # Extended for more data
end_date = get_last_trading_day()  # Dynamic end date
try:
    bars = api.get_bars(symbol, timeframe, start=start_date, end=end_date, adjustment='raw').df
    if bars.empty:
        logging.error(f"No data returned for {symbol}. Check symbol, date range, or API access.")
        exit(1)
    logging.info(f"Fetched {len(bars)} bars for {symbol}")
except HTTPError as e:
    logging.error(f"HTTP Error fetching data for {symbol}: {e}")
    exit(1)
except Exception as e:
    logging.error(f"Error fetching data for {symbol}: {e}")
    exit(1)

# Debug: Print DataFrame columns and index
logging.info(f"DataFrame columns: {bars.columns.tolist()}")
logging.info(f"DataFrame index: {bars.index.name}")

# Reset index to ensure timestamp is a column
bars = bars.reset_index()

# Check for timestamp column and rename if necessary
if 'timestamp' not in bars.columns and 'time' in bars.columns:
    bars = bars.rename(columns={'time': 'timestamp'})
elif 'timestamp' not in bars.columns:
    logging.error("No 'timestamp' or 'time' column found in data. Available columns: %s", bars.columns.tolist())
    exit(1)

# Select relevant columns and create a copy
data = bars[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
data.loc[:, 'timestamp'] = pd.to_datetime(data['timestamp'])

# Step 4: Visualize data
fig = go.Figure(data=[
    go.Candlestick(x=data['timestamp'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=f'{symbol} Price'),
    go.Bar(x=data['timestamp'], y=data['volume'], name='Volume', yaxis='y2', opacity=0.3)
])
fig.update_layout(
    title=f'{symbol} Price and Volume',
    yaxis_title='Price (USD)',
    yaxis2=dict(title='Volume', overlaying='y', side='right'),
    xaxis_title='Date'
)
fig.write_html(f"{symbol}_price_volume.html")
fig.show()

# Step 5: Prepare data
try:
    data.loc[:, 'next_close'] = data['close'].shift(-1)
    data.loc[:, 'target'] = (data['next_close'] > data['close']).astype(int)
    data.loc[:, 'ma5'] = data['close'].rolling(window=5).mean()
    data.loc[:, 'ma10'] = data['close'].rolling(window=10).mean()
    data.loc[:, 'ema12'] = data['close'].ewm(span=12, adjust=False).mean()
    try:
        data.loc[:, 'rsi'] = rsi(data['close'], window=14)
        macd_indicator = MACD(data['close'], window_fast=12, window_slow=26, window_sign=9)
        data.loc[:, 'macd'] = macd_indicator.macd()
        data.loc[:, 'macd_signal'] = macd_indicator.macd_signal()
        bb = BollingerBands(close=data['close'], window=20, window_dev=2)
        data.loc[:, 'bb_high'] = bb.bollinger_hband()
        data.loc[:, 'bb_low'] = bb.bollinger_lband()
        atr = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=14)
        data.loc[:, 'atr'] = atr.average_true_range()
        stoch = StochasticOscillator(high=data['high'], low=data['low'], close=data['close'], window=14, smooth_window=3)
        data.loc[:, 'stoch'] = stoch.stoch()
        features = ['close', 'ma5', 'ma10', 'ema12', 'rsi', 'macd', 'macd_signal', 'bb_high', 'bb_low', 'atr', 'stoch']
    except Exception as e:
        logging.warning(f"Failed to compute indicators: {e}. Falling back to basic features.")
        features = ['close', 'ma5', 'ma10', 'ema12']
    data = data.dropna()
    logging.info(f"Data after preprocessing: {len(data)} rows")

    # Check class balance
    class_counts = data['target'].value_counts()
    logging.info(f"Class distribution: {class_counts.to_dict()}")

    # Check if enough data remains for LSTM
    if len(data) < 30:
        logging.error(f"Insufficient data after preprocessing ({len(data)} rows). Need at least 30 rows.")
        exit(1)

    # Scale features individually
    scalers = {}
    X = np.zeros((len(data), len(features)))
    for i, feature in enumerate(features):
        scalers[feature] = MinMaxScaler()
        X[:, i] = scalers[feature].fit_transform(data[[feature]].values).flatten()

    y = data['target'].values

    # Create sequences
    sequence_length = 20  # Increased for longer-term patterns
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Check if sequences are sufficient
    if len(X_seq) < 10:
        logging.error(f"Too few sequences ({len(X_seq)}) for training. Try a shorter sequence length or more data.")
        exit(1)
    logging.info(f"Created {len(X_seq)} sequences for training")
except Exception as e:
    logging.error(f"Error in data preparation: {e}")
    exit(1)

# Step 6: Split data
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]
logging.info(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

# Step 7: Compute class weights
try:
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    logging.info(f"Class weights: {class_weight_dict}")
except Exception as e:
    logging.warning(f"Error computing class weights: {e}. Using no class weights.")
    class_weight_dict = None

# Step 8: Build and train LSTM
try:
    model = Sequential([
        Input(shape=(sequence_length, len(features))),
        LSTM(30),  # Single layer with more units
        Dropout(0.2),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        class_weight=class_weight_dict,
        verbose=1
    )
except Exception as e:
    logging.error(f"Error in model training: {e}")
    exit(1)

# Step 9: Evaluate model
try:
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
except Exception as e:
    logging.error(f"Error in model evaluation: {e}")
    exit(1)

# Step 10: Make prediction
try:
    last_sequence = X[-sequence_length:].reshape(1, sequence_length, len(features))
    prediction = model.predict(last_sequence)[0][0]
    if prediction > 0.5:
        print(f"Prediction: Tomorrow's close for {symbol} will be HIGHER than today's (Buy signal)")
    else:
        print(f"Prediction: Tomorrow's close for {symbol} will be LOWER or equal to today's (Sell/No-Buy signal)")
except Exception as e:
    logging.error(f"Error in prediction: {e}")
    exit(1)

# Step 11: Visualize training history
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, len(history.history['accuracy']) + 1)), y=history.history['accuracy'], mode='lines', name='Training Accuracy'))
fig.add_trace(go.Scatter(x=list(range(1, len(history.history['val_accuracy']) + 1)), y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy'))
fig.update_layout(title=f'Model Accuracy Over Epochs for {symbol}', xaxis_title='Epoch', yaxis_title='Accuracy')
fig.write_html(f"{symbol}_accuracy_plot.html")
fig.show()

# Step 12: Backtesting
try:
    predictions = model.predict(X_test)
    buy_signals = (predictions > 0.5).astype(int)
    correct = (buy_signals.flatten() == y_test).mean()
    print(f"Backtest Accuracy: {correct * 100:.2f}%")
except Exception as e:
    logging.error(f"Error in backtesting: {e}")

# Save model and prediction
model.save(f'{symbol}_lstm_model.keras')
with open(f'{symbol}_prediction.txt', 'w') as f:
    f.write(f"Prediction for {symbol}: {'Buy' if prediction > 0.5 else 'Sell/No-Buy'}\n")