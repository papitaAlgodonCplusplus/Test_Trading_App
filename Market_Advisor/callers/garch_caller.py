import sys
import os
import pandas as pd
import plotly.graph_objects as go
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from analyzers.garch import GARCHModel

def indicate():
    predictor = GARCHModel()
    data = pd.read_csv('data/real_time_data.csv')
    predictor.fit(data)
    predicted_prices = predictor.predict_future_prices(data)
    predicted_prices_list = predicted_prices['Predicted_Close'].tolist()
    last_predicted_price = predicted_prices_list[-1]
    last_actual_price = data['Close'].iloc[-1]
    print(f"Last actual price: {last_actual_price}, Last predicted price: {last_predicted_price}")
    if last_predicted_price > last_actual_price:
        return 1
    elif last_predicted_price < last_actual_price:
        return -1
    else:
        return 0