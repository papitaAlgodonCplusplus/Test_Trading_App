import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from analyzers.nbeats import NBEATS

def indicate():
    # nbeats = NBEATS()
    # data = pd.read_csv("data/real_time_data.csv")
    # future_prices = nbeats.predict_future_prices(prices_df=data, steps_ahead=20)
    # first_predicted_price = future_prices[0]
    # last_predicted_price = future_prices[-1]
    # print(f"First predicted price: {first_predicted_price}, Last predicted price: {last_predicted_price}")
    # if last_predicted_price > first_predicted_price:
    #     return 1
    # elif last_predicted_price < first_predicted_price:
    #     return -1
    # else:
    #     return 0
    return 0
    
if __name__ == "__main__":
    indicate()
