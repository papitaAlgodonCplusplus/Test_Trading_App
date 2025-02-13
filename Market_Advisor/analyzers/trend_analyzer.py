import pandas as pd

class TrendAnalyzer:
    def analyze_trend(self, data):
        # Convert the input data to a DataFrame
        df = pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        
        # Ensure 'Close' column is float for comparison
        df['Close'] = df['Close'].astype(float)
        
        # Check if there are at least 20 data points
        if len(df) < 20:
            return -1  # Not enough data to analyze
        
        # Get the last 20 closing prices
        last_20_closes = df['Close'].tail(20).values
        
        # Check if each close is greater than the previous one
        for i in range(1, len(last_20_closes)):
            if last_20_closes[i] <= last_20_closes[i - 1]:
                return -1  # Trend is not consistently upward
        
        return 1  # Prices have been going up for the last 20 data points