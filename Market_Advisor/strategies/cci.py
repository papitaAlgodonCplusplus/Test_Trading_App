import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

def main(file_path="data/real_time_data.csv", context_window=120):
    data = pd.read_csv(file_path)
    data = data.iloc[-context_window:]
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(by="Date", ascending=False)
    data = data.reset_index(drop=True)

    def compute_cci(data, period=5):
        data["TP"] = (data["High"] + data["Low"] + data["Close"]) / 3
        data["SMA"] = data["TP"].rolling(window=period).mean()
        data["MAD"] = data["TP"].rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        data["CCI"] = (data["TP"] - data["SMA"]) / (0.015 * data["MAD"])
        return data

    data = compute_cci(data)
    low_minima_indices = argrelextrema(data["Low"].values, np.less, order=2)[0]
    cci_minima_indices = argrelextrema(data["CCI"].values, np.less, order=2)[0]

    def find_divergence(low_indices, cci_indices, data):
        for i in range(1, len(low_indices)):
            low_diff = data["Low"].iloc[low_indices[i]] - data["Low"].iloc[low_indices[i-1]]
            cci_diff = data["CCI"].iloc[cci_indices[i]] - data["CCI"].iloc[cci_indices[i-1]]
            if low_diff < 0 and cci_diff > 0:
                return "bullish", data["Date"].iloc[low_indices[i]]
            elif low_diff > 0 and cci_diff < 0:
                return "bearish", data["Date"].iloc[low_indices[i]]
        return None, None

    divergence, date = find_divergence(low_minima_indices, cci_minima_indices, data)
    print(f"Diveregence: {divergence}, Date: {date}")
    if divergence == "bullish":
        print(f"Buy long here {date}")
    elif divergence == "bearish":
        print(f"Sell short here {date}")
    else:
        print("No divergence found")
        
if __name__ == '__main__':
    main()