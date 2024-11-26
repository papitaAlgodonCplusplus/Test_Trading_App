import pandas as pd

def load_stock_data(stock_name):
    '''Load stock data from a CSV file'''
    file_path = f"data/{stock_name}.csv"
    df = pd.read_csv(file_path)
    return df

def get_stock_prices(stock_name):
    '''Return a list containing stock close prices from a CSV file'''
    df = load_stock_data(stock_name)
    return df['Close'].tolist()

def get_dates(stock_name):
    '''Return a list of dates from a CSV file'''
    df = load_stock_data(stock_name)
    return pd.to_datetime(df['Date']).tolist()

def get_times(stock_name):
    '''Return a list of times from a CSV file'''
    df = load_stock_data(stock_name)
    
    # Specify the format explicitly (example: '%Y-%m-%d %H:%M:%S' for '2023-11-23 12:30:00')
    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    # Handle invalid parsing (e.g., rows with NaT)
    if df['Time'].isna().any():
        print("Warning: Some timestamps could not be parsed and were set to NaT")
    
    return df['Time'].tolist()

def get_key_levels(stock_name):
    '''Compute key levels (significant highs, lows, and midpoints)'''
    df = load_stock_data(stock_name)
    highs = df['High'].tolist()
    lows = df['Low'].tolist()
    midpoints = [(high + low) / 2 for high, low in zip(highs, lows)]

    # Combine significant levels into a single list
    key_levels = sorted(set(highs + lows + midpoints))
    return key_levels
