import pandas as pd
import numpy as np
from arch import arch_model

class GARCHModel:
    def __init__(self):
        self.model = None

    def fit(self, prices):
        """
        Fit a GARCH model to the closing prices.

        :param prices: A DataFrame containing stock data with a 'Close' column.
        """
        if 'Close' not in prices.columns:
            raise ValueError("The input data must have a 'Close' column.")

        # Calculate returns
        prices['Return'] = prices['Close'].pct_change().dropna()

        # Fit the GARCH(1, 1) model
        self.model = arch_model(prices['Return'].dropna(), vol='Garch', p=20, q=20)
        self.model = self.model.fit(disp='off')

    def predict_future_prices(self, prices, steps_ahead=30):
        """
        Predict future prices using the fitted GARCH model.

        :param prices: A DataFrame containing stock data with a 'Close' column.
        :param steps_ahead: Number of steps ahead to forecast.
        :return: A DataFrame with predicted future prices.
        """
        if self.model is None:
            raise ValueError("The GARCH model must be fitted before prediction.")

        # Forecast future volatility
        forecast = self.model.forecast(horizon=steps_ahead)
        variance_forecast = forecast.variance.iloc[-1]
        mean_forecast = forecast.mean.iloc[-1]

        # Use the last close price as the base for predictions
        last_close = prices['Close'].iloc[-1]

        # Convert returns to prices
        future_prices = [last_close]
        for mean, variance in zip(mean_forecast, variance_forecast):
            # Assume log-normal distribution for prices
            future_return = mean + np.sqrt(variance) * np.random.normal()
            next_price = future_prices[-1] * (1 + future_return)
            future_prices.append(next_price)

        # Create a DataFrame for the results
        future_dates = pd.date_range(start=prices['Date'].iloc[-1], periods=steps_ahead, freq='D')
        future_prices_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_prices[1:steps_ahead+1]})
        return future_prices_df