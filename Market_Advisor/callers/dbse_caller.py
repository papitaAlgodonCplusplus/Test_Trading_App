import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from analyzers.dbse import EURUSDPricePredictor

def indicate():
    predictor = EURUSDPricePredictor()
    data = pd.read_csv('data/real_time_data.csv')
    predicted_data = predictor.predict(data)
    last_actual_price = data['Close'].iloc[-1]
    last_predicted_price = predicted_data['Close'].iloc[-1]
    print(f"Last actual price: {last_actual_price}, Last predicted price: {last_predicted_price}")
    if last_predicted_price > last_actual_price:
        return 1
    elif last_predicted_price < last_actual_price:
        return -1
    else:
        return 0
    
if __name__ == '__main__':
    indicate()