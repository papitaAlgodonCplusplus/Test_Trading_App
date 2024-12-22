import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

def detect_support_resistance(df):
    support = df['Low'].min()
    resistance = df['High'].max()
    return support, resistance

def verify_bounces(df, support, resistance, margin=0.0002, threshold=3):
    recent_data = df[-threshold:]
    bounces = 0
    for i in range(len(recent_data)):
        low = recent_data.iloc[i]['Low']
        high = recent_data.iloc[i]['High']
        if abs(low - support) <= margin or abs(high - resistance) <= margin:
            bounces += 1
    if bounces >= threshold:
        print(f"Price has bounced within the support/resistance levels for at least {threshold} data points.")
        return True
    else:
        print(f"Price has NOT bounced within the support/resistance levels for at least {threshold} data points.")
        return False

def check_attempts(above_middle, df, middle_line):
    for i in range(1, len(above_middle)):
        current_index = above_middle.index[i]
        low_attempt = df.loc[current_index, 'Low']
        close_price = df.loc[current_index, 'Close']
        date = df.loc[current_index, 'Date']
        if low_attempt < middle_line and close_price > middle_line:
            return 1, date
    return 0, None
            
def price_range_calculator(file_path="data/real_time_data.csv", context_window=30):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df[-context_window:]
    df = df.sort_values(by='Date', ascending=False)
    df = df.reset_index(drop=True)
    support, resistance = detect_support_resistance(df)
    middle_line = (support + resistance) / 2
    if verify_bounces(df, support, resistance):
        above_middle = df[df['Close'] > middle_line]
        signal, date = check_attempts(above_middle, df, middle_line)
        return signal, date
    else:
        return 0, None