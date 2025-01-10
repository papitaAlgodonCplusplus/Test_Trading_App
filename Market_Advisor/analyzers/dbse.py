import pandas as pd
import numpy as np

class EURUSDPricePredictor:
    def __init__(self, volatility=0.01):
        self.volatility = volatility

    def predict_future_prices(self, prices, steps_ahead, volatility=0.01):
        """
        Predict future EUR/USD prices based on the mock BSDE model.
        """
        if prices.empty:
            raise ValueError("The prices series is empty.")
        last_price = prices.iloc[-1]
        dt = 1 / len(prices)
        future_returns = np.random.normal(0, volatility, steps_ahead) * np.sqrt(dt)
        future_prices = last_price * np.exp(np.cumsum(future_returns))
        return future_prices

    def predict(self, data, steps_ahead = 30):
        if isinstance(data, str):
            from io import StringIO
            data = pd.read_csv(StringIO(data))
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Input data must contain the following columns: {required_columns}")
        predicted_prices = self.predict_future_prices(data['Close'], steps_ahead, self.volatility)
        last_date = pd.to_datetime(data['Date'].iloc[-1])
        future_dates = [last_date + pd.Timedelta(minutes=i+1) for i in range(steps_ahead)]
        predicted_data = pd.DataFrame({
            'Date': future_dates,
            'Open': predicted_prices,
            'High': predicted_prices * (1 + self.volatility / 2),  # Simulate a small range
            'Low': predicted_prices * (1 - self.volatility / 2),
            'Close': predicted_prices,
            'Volume': np.random.randint(1, 10, size=steps_ahead)  # Simulate random volumes
        })

        return predicted_data
