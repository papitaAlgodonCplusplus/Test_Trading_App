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