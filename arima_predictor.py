import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
import logging
from datetime import datetime, timedelta




# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fetch data
symbol = 'CCL'
start_date = '2021-01-01'
end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
try:
    data = yf.download(symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True)
    if data.empty:
        logging.error(f"No data returned for {symbol}. Check symbol or date range.")
        exit(1)
    data = data[['Close']].rename(columns={'Close': 'close'})
    data.index = pd.to_datetime(data.index)
    # Create a complete business day index
    full_index = pd.date_range(start=start_date, end=end_date, freq='B')
    data = data.reindex(full_index).ffill()  # Forward-fill missing dates
    data = data.asfreq('B')  # Set business day frequency
    # Transform to log prices
    data['log_close'] = np.log(data['close'])
    logging.info(f"Fetched {len(data)} bars for {symbol}")
except Exception as e:
    logging.error(f"Error fetching data for {symbol}: {e}")
    exit(1)

# Check stationarity with ADF test
def check_stationarity(series):
    result = adfuller(series.dropna())
    logging.info(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    if result[1] < 0.05:
        logging.info("Series is stationary (p < 0.05)")
    else:
        logging.warning("Series is not stationary (p >= 0.05). Using log transformation and differencing.")

check_stationarity(data['log_close'])

# Prepare data
try:
    data['returns'] = data['log_close'].pct_change()
    data = data.dropna()
    logging.info(f"Data after preprocessing: {len(data)} rows")
except Exception as e:
    logging.error(f"Error in data preparation: {e}")
    exit(1)

# Split data for backtesting
train_size = int(len(data) * 0.8)
train_data, test_data = data['log_close'][:train_size], data['log_close'][train_size:]

# Select best ARIMA order based on AIC
orders = [(1,1,1), (2,1,1), (1,2,1), (2,2,1)]
best_aic = float('inf')
best_order = None
best_model = None
try:
    for order in orders:
        model = ARIMA(train_data, order=order)
        model_fit = model.fit(method='css-mle', maxiter=200)
        aic = model_fit.aic
        logging.info(f"Order {order} AIC: {aic}")
        if aic < best_aic:
            best_aic = aic
            best_order = order
            best_model = model_fit
    logging.info(f"Best order: {best_order} with AIC: {best_aic}")
except Exception as e:
    logging.error(f"Error in ARIMA model selection: {e}")
    exit(1)

# Check residuals
try:
    residuals = best_model.resid
    check_stationarity(residuals)
    logging.info(f"Residual mean: {np.mean(residuals):.4f}, std: {np.std(residuals):.4f}")
except Exception as e:
    logging.error(f"Error in residual analysis: {e}")

# Fit best ARIMA model and forecast
try:
    forecast = best_model.forecast(steps=1)
    next_log_close = float(forecast.iloc[0])  # Convert to scalar
    next_close = np.exp(next_log_close)  # Convert back to price
    last_log_close = float(data['log_close'].iloc[-1])  # Convert to scalar
    last_close = np.exp(last_log_close)  # Convert back to price
    # Validate against raw close
    raw_last_close = float(data['close'].iloc[-1])
    if abs(last_close - raw_last_close) > 0.01:
        logging.warning(f"Log transformation mismatch: log-derived last close {last_close:.2f} vs raw {raw_last_close:.2f}")
        last_close = raw_last_close
    logging.info(f"Last close: {last_close:.2f}, Predicted close: {next_close:.2f}")
    if next_close > last_close * 1.01:
        print(f"ARIMA Prediction: Tomorrow's close for {symbol} will be HIGHER than today's by at least 1% (Buy signal)")
    else:
        print(f"ARIMA Prediction: Tomorrow's close for {symbol} will be LOWER or within 1% of today's (Sell/No-Buy signal)")
except Exception as e:
    logging.error(f"Error in ARIMA modeling: {e}")
    exit(1)

# Backtest ARIMA on test set
try:
    predictions = []
    actual = []
    test_indices = test_data.index[1:]  # For plotting
    for i in range(len(test_data) - 1):
        model = ARIMA(data['log_close'][:train_size + i], order=best_order)
        model_fit = model.fit(method='css-mle', maxiter=200)
        pred = model_fit.forecast(steps=1).iloc[0]
        predictions.append(np.exp(pred))  # Convert back to price
        actual.append(np.exp(data['log_close'].iloc[train_size + i + 1]))
    # Calculate accuracy (predicting direction)
    predictions = np.array(predictions)
    actual = np.array(actual)
    test_closes = np.exp(data['log_close'].iloc[train_size:train_size + len(predictions)].values)
    direction_pred = (predictions > test_closes * 1.01).astype(int)
    direction_actual = (actual > test_closes * 1.01).astype(int)
    accuracy = (direction_pred == direction_actual).mean()
    print(f"ARIMA Backtest Accuracy: {accuracy * 100:.2f}%")

    # Visualize predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_indices, y=actual, mode='lines', name='Actual Close', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=test_indices, y=predictions, mode='lines', name='Predicted Close', line=dict(color='#ff7f0e')))
    fig.update_layout(
        title=f'ARIMA Predictions vs Actual for {symbol}',
        xaxis_title='Date',
        yaxis_title='Close Price',
        template='plotly_dark'
    )
    fig.write_html(f"{symbol}_arima_predictions.html")
    fig.show()
except Exception as e:
    logging.error(f"Error in ARIMA backtesting: {e}")

# Save prediction
with open(f'{symbol}_arima_prediction.txt', 'w') as f:
    f.write(f"ARIMA Prediction for {symbol}: {'Buy' if next_close > last_close * 1.01 else 'Sell/No-Buy'}\n")