import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Date'])

def find_last_retracement(data):
    """
    Identifies the last retracement point where a trend reverses direction.
    """
    trend = None
    retracement_point = None
    for i in range(1, len(data)):
        if data['Close'][i] > data['Close'][i-1]:
            current_trend = 'up'
        else:
            current_trend = 'down'
        if trend and current_trend != trend:
            retracement_point = data.iloc[i - 1]
            break
        trend = current_trend
    return retracement_point

def calculate_fibonacci_levels(data, retracement_point):
    """
    Calculates Fibonacci retracement levels from the initial trend to the retracement point.
    """
    start_price = data['Low'].iloc[0] if data['Close'].iloc[0] < retracement_point['Close'] else data['High'].iloc[0]
    end_price = retracement_point['Close']
    diff = abs(end_price - start_price)
    fib_levels = {
        '0': start_price,
        '0.618': start_price + 0.618 * diff,
        '1': end_price,
        '1.618': start_price + 1.618 * diff
    }
    return fib_levels

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculates the MACD line, signal line, and histogram.
    """
    data['EMA12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal']
    return data

def determine_trade_signal(data, fib_0_618, date):
    """
    Checks if the MACD line crosses above or below the signal line at the 0.618 level.
    """
    closest_idx = (data['Close'] - fib_0_618).abs().idxmin()
    macd_value = data['MACD'].iloc[closest_idx]
    signal_value = data['Signal'].iloc[closest_idx]
    if macd_value > signal_value and data['Close'].iloc[closest_idx] > date:
        print(f"Set buy long order at date: {data['Date'].iloc[closest_idx]}")
        return 1, data['Date'].iloc[closest_idx]
    elif macd_value < signal_value and data['Close'].iloc[closest_idx] > date:
        print(f"Set sell short order at date: {data['Date'].iloc[closest_idx]}")
        return -1, data['Date'].iloc[closest_idx]
    else:
        print("No trade signal found.")
        return 0, None
        
def fibonacci_calculator(file_path, context_window=120):
    data = load_data(file_path)
    data = data.iloc[-context_window:]
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date', ascending=False)
    data = data.reset_index(drop=True)
    retracement_point = find_last_retracement(data)
    if retracement_point is None:
        print("No retracement found.")
        return
    print(f"Retracement point: {retracement_point['Date']}")
    fib_levels = calculate_fibonacci_levels(data, retracement_point)
    print(f"Fibonacci Levels: {fib_levels}")
    data = calculate_macd(data)
    date_index = data.loc[data['Date'] == retracement_point['Date']].index[0]
    trade_signal, trade_date = determine_trade_signal(data, fib_levels['0.618'], date_index)
    return trade_signal, trade_date
