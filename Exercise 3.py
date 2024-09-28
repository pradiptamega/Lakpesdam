# -*- coding: utf-8 -*-
"""
Exercise 3 Pelatihan LAKPESDAM Python 2024

M Pradipta
"""

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

stock_symbol = "NFLX"
start_date = "2022-01-01"
end_date = "2024-05-31"
prediction_end_date = "2025-05-31"  # Prediksi 1 tahun

data = yf.download(stock_symbol, start=start_date, end=end_date)

data['dates_numeric'] = data.index.map(pd.Timestamp.timestamp)

# Features (X) and target (y)
x = data['dates_numeric'].values.reshape(-1, 1)  # Reshape for sklearn
y = data['Adj Close']

# Fit the linear regression model
model = LinearRegression()
model.fit(x, y)

# Predict values
y_pred = model.predict(x)

# buat prediksi
future_dates = pd.date_range(start=end_date, end=prediction_end_date, freq='B')
future_dates_numeric = future_dates.map(pd.Timestamp.timestamp).values.reshape(-1, 1)

future_pred = model.predict(future_dates_numeric)

# Plotting
plt.figure(figsize=(15,6))
plt.plot(data.index, data["Adj Close"], label="Adj Close", color='blue')
plt.plot(data.index, y_pred, label="Interpolation", linestyle='--', color='red')
plt.plot(future_dates, future_pred, label="Extrapolation", linestyle='--', color='green')
plt.grid(linestyle=":")
plt.ylabel("Price ($)")
plt.title(f"Netflix stock price from {start_date} to {prediction_end_date}")
plt.legend()
plt.show()