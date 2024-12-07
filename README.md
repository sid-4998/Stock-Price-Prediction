# Stock-Price-Prediction
This program predicts stock prices using the ARIMA,SARIMA,PROPHET,LSTM,CHRONOS,LINEAR REGRESSION models. It also detects and removes outliers from the stock price data to improve the accuracy of predictions.

#Goal
This program shows that LSTM model and Chronos Model fits best for stock price prediction as time series dataset. 

To implement stock price prediction using multiple models such as LSTM, ARIMA, Prophet, SARIMA, Linear Regression, Ridge Regression, and a Deterministic Model, you need to approach each method with its specific configuration and implementation. Below is an overview of how you can combine these models and their implementation steps.
Steps to Implement
1.	Load Stock Price Data: Use a single dataset for training all models.
2.	Preprocessing: Handle missing values, feature scaling, and train-test split.
3.	Modeling:
o	Implement ARIMA and SARIMA for time series modeling.
o	Use LSTM for deep learning-based time series forecasting.
o	Use Linear Regression and Ridge Regression for standard regression techniques.
o	Implement Prophet for time-series forecasting using trend decomposition.
o	Implement Deterministic models using moving averages or other simple statistical techniques.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Fit ARIMA model
arima_model = ARIMA(train_data, order=(5,1,0)).fit()
arima_pred = arima_model.forecast(steps=len(test_data))

# Evaluate ARIMA
mse_arima = mean_squared_error(test_data, arima_pred)

# Fit SARIMA model
sarima_model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
sarima_pred = sarima_model.forecast(steps=len(test_data))

# Evaluate SARIMA
mse_sarima = mean_squared_error(test_data, sarima_pred)

# Prepare data for Prophet
prophet_data = close_prices.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_data[:train_size])

# Predict
future = prophet_model.make_future_dataframe(periods=len(test_data))
prophet_pred = prophet_model.predict(future)

# Evaluate Prophet
mse_prophet = mean_squared_error(test_data['Close'], prophet_pred['yhat'][-len(test_data):])

# Prepare data for LSTM (reshaping for 3D tensor input)
train_scaled = train_data.values.reshape((train_data.shape[0], 1, 1))

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(train_scaled.shape[1], train_scaled.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

# Train LSTM
lstm_model.fit(train_scaled, train_data.values, epochs=100, batch_size=32, verbose=1)

# Predict using LSTM
lstm_pred = lstm_model.predict(test_data.values.reshape((test_data.shape[0], 1, 1)))

# Evaluate LSTM
mse_lstm = mean_squared_error(test_data, lstm_pred)

# Prepare data for Linear Regression
X_train = np.arange(len(train_data)).reshape(-1, 1)
y_train = train_data.values

X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict using Linear Regression
lr_pred = lr_model.predict(X_test)

# Evaluate Linear Regression
mse_lr = mean_squared_error(test_data, lr_pred)

# Train Ridge Regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Predict using Ridge Regression
ridge_pred = ridge_model.predict(X_test)

# Evaluate Ridge Regression
mse_ridge = mean_squared_error(test_data, ridge_pred)

# Simple moving average model
window = 5
deterministic_pred = test_data.rolling(window=window).mean().dropna()

# Evaluate Deterministic Model
mse_deterministic = mean_squared_error(test_data[window-1:], deterministic_pred)

