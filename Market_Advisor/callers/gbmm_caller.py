import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from analyzers.gbmm import GBMM

def indicate():
    prices_df = pd.read_csv('data/real_time_data.csv')
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    train_df, test_df = train_test_split(prices_df, test_size=0.1, shuffle=False)
    gbmm = GBMM()
    future_prices_df = gbmm.predict_future_prices(train_df, steps_ahead=10)
    print(f"Is last predicted price higher than the last first predicted price? {future_prices_df['Predicted_Close'].iloc[-1] > future_prices_df['Predicted_Close'].iloc[0]}")
    if future_prices_df['Predicted_Close'].iloc[-1] > future_prices_df['Predicted_Close'].iloc[0]:
        return 1
    elif future_prices_df['Predicted_Close'].iloc[-1] < future_prices_df['Predicted_Close'].iloc[0]:
        return -1
    else:
        return 0

if __name__ == "__main__":
    indicate()
