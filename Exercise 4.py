# -*- coding: utf-8 -*-
"""
Exercise 3 Pelatihan LAKPESDAM Python 2024

M pradipta
"""

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

class StockPredictor:
    def __init__(self, symbol, start_date, end_date, prediction_end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.prediction_end_date = prediction_end_date
        self.data = None
        self.model = LinearRegression()
        self.future_dates = None
        self.future_predictions = None

    def download_data(self):
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        self.data['dates_numeric'] = self.data.index.map(pd.Timestamp.timestamp)

    def model_interpolasi(self):
        x = self.data['dates_numeric'].values.reshape(-1, 1)
        y = self.data['Adj Close']
        self.model.fit(x, y)

    def prediksi_ekstrapolasi(self):
        x = self.data['dates_numeric'].values.reshape(-1, 1)
        self.data['Predicted'] = self.model.predict(x)
        self.future_dates = pd.date_range(start=self.end_date, end=self.prediction_end_date, freq='B')
        future_dates_numeric = self.future_dates.map(pd.Timestamp.timestamp).values.reshape(-1, 1)
        self.future_predictions = self.model.predict(future_dates_numeric)

    def plot(self):
        plt.figure(figsize=(15,6))
        plt.plot(self.data.index, self.data["Adj Close"], label="Adj Close", color='blue')
        plt.plot(self.data.index, self.data['Predicted'], label="Interpolation", linestyle='--', color='red')
        plt.plot(self.future_dates, self.future_predictions, label="Extrapolation", linestyle='--', color='green')
        plt.grid(linestyle=":")
        plt.ylabel("Price ($)")
        plt.title(f"{self.symbol} stock price from {self.start_date} to {self.prediction_end_date}")
        plt.legend()
        plt.show()

# Recall method
stock_predictor = StockPredictor('NFLX', '2022-01-01', '2024-05-31', '2025-05-31')
stock_predictor.download_data()
stock_predictor.model_interpolasi()
stock_predictor.prediksi_ekstrapolasi()
stock_predictor.plot()