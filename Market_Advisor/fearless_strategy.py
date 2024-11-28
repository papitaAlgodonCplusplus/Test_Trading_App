import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from ta.momentum import RSIIndicator
from market_predicter import obtain_final_predictions
from itertools import product
from datetime import datetime
import time
import numpy as np
import threading
from plotter import Plotter

def calculate_indicators(data):
    """Calculate technical indicators."""
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
    data['High_Liquidity'] = data['High'].rolling(window=5).max()
    data['Low_Liquidity'] = data['Low'].rolling(window=5).min()

    # True Range and Liquidity Pool
    data['Prev_Close'] = data['Close'].shift(1)
    data['True_Range'] = data[['High', 'Low', 'Prev_Close']].apply(
        lambda row: max(
            row['High'] - row['Low'],
            abs(row['High'] - row['Prev_Close']),
            abs(row['Low'] - row['Prev_Close'])
        ),
        axis=1
    )

    rolling_window = 14
    rolling_min = data['True_Range'].rolling(window=rolling_window).min()
    rolling_max = data['True_Range'].rolling(window=rolling_window).max()
    data['Liquidity_Pool'] = (data['True_Range'] - rolling_min) / (rolling_max - rolling_min)
    data['Liquidity_Pool'] = data['Liquidity_Pool'].fillna(0)

    # Bollinger Bands
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['SMA_20'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Lower'] = data['SMA_20'] - 2 * data['Close'].rolling(window=20).std()

    # Average True Range
    data['ATR'] = data['True_Range'].rolling(window=rolling_window).mean()

    # Order Block Validity
    data['Order_Block_Valid'] = (
        (data['Low'].shift(1) > data['Low']) & (data['High'].shift(-1) < data['Low'])
    )
    return data

def get_conditions(data, level):
    """Define bullish and bearish conditions based on analysis level."""
    bullish = (data['MACD'] > data['Signal_Line'])
    bearish = (data['MACD'] < data['Signal_Line'])

    if level >= 1:
        bullish &= (data['RSI'] < 70)
        bearish &= (data['RSI'] > 30)
    if level >= 2:
        bullish &= (data['Close'] > data['SMA_20'])
        bearish &= (data['Close'] < data['SMA_20'])
    if level >= 3:
        bullish &= (data['Liquidity_Pool'] > 0.5)
        bearish &= (data['Liquidity_Pool'] < 0.5)
        bullish |= data['Order_Block_Valid'] & (data['Close'] > data['Low'].shift(1))
        bearish |= data['Order_Block_Valid'] & (data['Close'] < data['High'].shift(1))
    if level >= 4:
        bullish &= (data['Close'] <= data['BB_Lower']) & (data['RSI'] < 70)
        bearish &= (data['Close'] >= data['BB_Upper']) & (data['RSI'] > 30)
        volatility_threshold = data['ATR'].mean() * 0.5  # Example: 50% of the mean ATR
        bullish &= (data['ATR'] > volatility_threshold) & (data['ATR'] < 2 * volatility_threshold)
        bearish &= (data['ATR'] > volatility_threshold) & (data['ATR'] < 2 * volatility_threshold)

    return bullish, bearish

def define_actions(data, probabilities, threshold=0.5, analysis_level=0, zigzag=False):
    """Define trading actions based on indicators."""
    bullish, bearish = get_conditions(data, analysis_level)

    data['Action'] = 'Hold'
    data.loc[bullish, 'Action'] = 'Buy'
    data.loc[bearish, 'Action'] = 'Sell'

    
    if threshold > 0.0:
        data.loc[(data['Action'] == 'Hold') & (probabilities > 1 - threshold), 'Action'] = 'Buy'
        data.loc[(data['Action'] == 'Hold') & (probabilities < threshold), 'Action'] = 'Sell'

    
    if zigzag:
        data['Action'] = data['Action'].where(data['Action'] != data['Action'].shift(), 'Hold')
        return data

def run_simulation(data, initial_capital=10000, reverse=False, risk_amount=None, expected_profit=None, log=False):
    """Run a trading simulation."""
    capital, units_hold, profit_vault = initial_capital, 0, 0
    profit_over_time = []
    vault_values = []

    for i, row in data.iterrows():
        action, price = row['Action'], row['Close']
        if reverse:
            action = 'Buy' if action == 'Sell' else 'Sell' if action == 'Buy' else 'Hold'

        if action == 'Buy' and capital > 0:
            invest = min(risk_amount or capital, capital)
            units_hold = invest / price
            capital -= invest
            data.loc[i, 'Action'] = 'Buy'
        elif action == 'Sell' and units_hold > 0:
            capital += units_hold * price
            units_hold = 0
            data.loc[i, 'Action'] = 'Sell'
        else:
            data.loc[i, 'Action'] = 'Hold'

        profit_over_time.append(capital - initial_capital)
        if log:
           print(f"Date: {row['Date']}, Capital: {capital:.2f}, Units: {units_hold:.2f}, Profit: {profit_over_time[-1]:.2f}, Action: {action}, Price: {price:.2f}")

        if profit_over_time[-1] > 0:
            profit_vault += profit_over_time[-1]
            capital = initial_capital

        vault_values.append(profit_vault)    

        if expected_profit and profit_vault >= expected_profit:
            capital += units_hold * price
            units_hold = 0
            return data, capital, profit_vault

    if units_hold > 0:
        capital += units_hold * data['Close'].iloc[-1]
        profit_over_time[-1] = capital - initial_capital

    data['Profit'] = profit_over_time
    data['Vault'] = vault_values
    return data, capital, profit_vault

def main_logic(file_path, plotter, reverse=False, multiply_factor = None, expected_profit=1):
    """Run the trading simulation."""
    data = pd.read_csv(file_path, parse_dates=['Date'])

    print(f"\033[95m --- Obtained New Data ---\033[0m")
    print(f"\033[95mLast date in dataset: {data['Date'].iloc[-1]}\033[0m")
    print(f"\033[95m -------------------------\033[0m")

    if multiply_factor is not None:
        data['Close'] = data['Close'] * multiply_factor
        data['High'] = data['High'] * multiply_factor
        data['Low'] = data['Low'] * multiply_factor
        data['Open'] = data['Open'] * multiply_factor
        
    calculate_indicators(data)

    predictions, probabilities = obtain_final_predictions(file_path)
    if reverse:
        predictions = 1 - predictions
        
    if len(predictions) < len(data):
        predictions = np.pad(predictions, (0, len(data) - len(predictions)), constant_values=3)

    if len(probabilities) < len(data):
        probabilities = np.pad(probabilities, (0, len(data) - len(probabilities)), constant_values=3)

    results = []
    analysis_levels = [0, 1, 2, 3, 4]
    threshold_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    initial_capital = 50000 if multiply_factor is None else 50000 * multiply_factor
    risk_options = [(initial_capital * 0.01) * i for i in range(1, 80, 5)]

    for level, threshold, risk_amount in product(analysis_levels, threshold_options, risk_options):
        define_actions(data, probabilities, threshold, level, True)
        data, capital, profit_vault = run_simulation(data, initial_capital, reverse, risk_amount, expected_profit=expected_profit)
        debt_profit = profit_vault - abs(capital - initial_capital)
        results.append({
            "level": level,
            "threshold": threshold,
            "risk_amount": risk_amount,
            "final_capital": capital,
            "profit_vault": profit_vault,
            "debt_profit": debt_profit
        })

    sorted_results = sorted(results, key=lambda x: x["debt_profit"], reverse=True)
    current_iteration = 0
    for result in sorted_results:
        print(f"\033[91mAnalysis Level: {result['level']} with Risk Amount: {result['risk_amount']:.2f}, and Threshold: {result['threshold']}\033[0m")
        print(f"\033[94mFinal capital: {result['final_capital']:.2f}\033[0m")  
        print(f"\033[93mProfit Vault: {result['profit_vault']:.2f}\033[0m")  
        print(f"\033[92mDebt Profit: {result['debt_profit']:.2f}\033[0m")  
        print("-" * 50)
        current_iteration += 1
           
        if result['debt_profit'] <= 0 or current_iteration >= 3:
            break

    best_results = sorted_results[:3]

    output_file = "best_hyperparameters.txt"
    with open(output_file, 'a') as f:
        f.write(f"\n--- Best Hyperparameters Round ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
        for i, result in enumerate(best_results, start=1):
            f.write(f"Rank {i}:\n")
            f.write(f"  Analysis Level: {result['level']}\n")
            f.write(f"  Threshold: {result['threshold']}\n")
            f.write(f"  Risk Amount: {result['risk_amount']:.2f}\n")
            f.write(f"  Final Capital: {result['final_capital']:.2f}\n")
            f.write(f"  Profit Vault: {result['profit_vault']:.2f}\n")
            f.write(f"  Debt Profit: {result['debt_profit']:.2f}\n")
            f.write("-" * 50 + "\n")

    print(f"Best hyperparameters saved to {output_file}")

    print(f"\033[91m --- Obtained New Data ---\033[0m")
    print(f"\033[91mLast date in dataset: {data['Date'].iloc[-1]}\033[0m")
    print(f"\033[91m -------------------------\033[0m")


    result = best_results[0]
    predictions, probabilities = obtain_final_predictions(file_path)
    
    if len(predictions) < len(data):
        predictions = np.pad(predictions, (0, len(data) - len(predictions)), constant_values=3)

    if len(probabilities) < len(data):
        probabilities = np.pad(probabilities, (0, len(data) - len(probabilities)), constant_values=3)

    define_actions(data, probabilities, result['threshold'], result['level'], True)
    data, capital, profit_vault = run_simulation(data, initial_capital, reverse, result['risk_amount'], expected_profit=None, log=False)
    data['Profit'] = data['Profit'].fillna(0)

    for row in data.itertuples():
        # Print colored
        if row.Action == 'Buy':
            print(f"\033[91m{row.Date}: {row.Action} at {row.Close:.2f}\033[0m")
        elif row.Action == 'Sell':
            print(f"\033[92m{row.Date}: {row.Action} at {row.Close:.2f}\033[0m")

    # Update plotter
    plotter.data = data
    plotter.predictions = predictions
    plotter.title_suffix = "Real-Time Simulation"
    plotter.update_figure()

def schedule_main(file_path, plotter):
    """Run the main function every 60 seconds."""
    while True:
        try:
            main_logic(file_path, plotter, reverse=True)
            print(f"Updated at {datetime.now()}")
        except Exception as e:
            print(f"Error occurred: {e}")
        time.sleep(60)

import traceback

if __name__ == "__main__":
    try:
        real_time_csv = "real_time_data.csv"
        plotter = Plotter()
        plotter.start()  # Start the Dash app in a separate thread

        # Run the main function in a loop
        scheduler_thread = threading.Thread(target=schedule_main, args=(real_time_csv, plotter))
        scheduler_thread.daemon = True
        scheduler_thread.start()

        # Keep the main thread alive
        while True:
            time.sleep(50)

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()