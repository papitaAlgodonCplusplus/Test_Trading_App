import numpy as np
import pandas as pd

class GBMM:
    def __init__(self):
        pass

    def _calculate_gbm_params(self, prices):
        """
        Calculate drift (mu) and volatility (sigma) based on historical prices.
        """
        log_returns = np.log(prices / prices.shift(1)).dropna()
        mu = log_returns.mean()
        sigma = log_returns.std()
        return mu, sigma

    def _simulate_gbm(self, S0, mu, sigma, steps, dt):
        """
        Simulate GBM for a given number of steps.
        """
        W = np.random.normal(0, np.sqrt(dt), steps)  # Brownian motion increments
        W = np.cumsum(W)  # Cumulative Brownian motion
        t = np.linspace(0, steps * dt, steps)
        S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
        return S

    def predict_future_prices(self, prices_df, steps_ahead=30):
        """
        Predict future prices using Geometric Brownian Motion.
        
        Parameters:
            prices_df (pd.DataFrame): DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            steps_ahead (int): Number of future steps to predict
        
        Returns:
            pd.DataFrame: DataFrame with future dates and predicted prices
        """
        # Extract Close prices
        close_prices = prices_df['Close']
        last_price = close_prices.iloc[-1]

        # Calculate GBM parameters
        mu, sigma = self._calculate_gbm_params(close_prices)
        dt = 1 / 1440

        # Simulate future prices
        future_prices = self._simulate_gbm(last_price, mu, sigma, steps_ahead, dt)

        # Create future dates
        last_date = pd.to_datetime(prices_df['Date'].iloc[-1])
        future_dates = [last_date + pd.Timedelta(minutes=i + 1) for i in range(steps_ahead)]

        # Create DataFrame for future prices
        future_prices_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_prices})

        return future_prices_df
