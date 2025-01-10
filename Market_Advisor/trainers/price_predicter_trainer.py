import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analyzers.price_preditcter import LSTMPricePredictor
if __name__ == '__main__':
    data = pd.read_csv('data/real_time_data.csv')
    price_predictor = LSTMPricePredictor()
    price_predictor.train(data)
