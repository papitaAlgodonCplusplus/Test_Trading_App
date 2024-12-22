import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
def detect_stop_hunt(df, PIP=0.0001):
    threshold = 5 * PIP
    for i in range(2, len(df)):
        prev_low = df.loc[i - 1, 'Low']
        curr_low = df.loc[i, 'Low']
        prev_high = df.loc[i - 1, 'High']
        curr_high = df.loc[i, 'High']
        if curr_low < prev_low - threshold:
            return 'low', curr_low, df.loc[i, 'Date']
        if curr_high > prev_high + threshold:
            return 'high', curr_high, df.loc[i, 'Date']
    return None, None, None
    
def stop_hunt_calculator(csv_path, context_window=120):
    data = pd.read_csv(csv_path)
    data = data.iloc[-context_window:]
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date', ascending=False)
    data = data.reset_index(drop=True)
    PIP = 0.0001
    direction, stop_hunt_price, stop_hunt_time = detect_stop_hunt(data, PIP)
    
    if direction:
        print(f"Last stop-hunt detected: {direction} at {stop_hunt_price} on {stop_hunt_time}")
        break_threshold = 1 * PIP
        buy_signal = stop_hunt_price + 3 * PIP
        sell_signal = stop_hunt_price - 3 * PIP
        for i in range(len(data)):
            price = data.loc[i, 'Close']
            timestamp = data.loc[i, 'Date']
            if direction == 'low' and price > stop_hunt_price + break_threshold:
                print(f"Price broke above stop-hunt level at {timestamp}, setting buy-long at {buy_signal} and sell-short at {sell_signal}")
                if price >= buy_signal:
                    print(f"Buy-long signal hit at {timestamp}, price: {price}")
                    return 1, timestamp
                elif price <= sell_signal:
                    print(f"Sell-short signal hit at {timestamp}, price: {price}")
                    return -1, timestamp
            elif direction == 'high' and price < stop_hunt_price - break_threshold:
                print(f"Price broke below stop-hunt level at {timestamp}, setting buy-long at {buy_signal} and sell-short at {sell_signal}")
                if price >= buy_signal:
                    print(f"Buy-long signal hit at {timestamp}, price: {price}")
                    return 1, timestamp
                elif price <= sell_signal:
                    print(f"Sell-short signal hit at {timestamp}, price: {price}")
                    return -1, timestamp
        return 0, None
    else:
        print("No stop-hunt detected in the data.")
        return 0, None