import pandas as pd
from analyzers.market_predicter import obtain_final_predictions
from datetime import datetime
import numpy as np
from itertools import product
from analyzers.actions import define_actions
from analyzers.simulator import run_simulation
from termcolor import colored
from tqdm import tqdm

def print_colored_sentence(sentence):
    colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
    words = sentence.split()
    for i, word in enumerate(words):
        color = colors[i % len(colors)]
        print(colored(word, color), end=" ")
    print()

def process_simulation_window(data, probabilities, analysis_levels, threshold_options, risk_options,
                              initial_capital, reverse, expected_profit, context_window, output_file, file_path):
    """Process a single simulation window."""
    sorted_results = process_simulation(data, probabilities, analysis_levels, threshold_options, risk_options,
                                        initial_capital, reverse, expected_profit, context_window)
    print_results(sorted_results)

    # Process best results
    best_results = sorted_results[:3]
    result = best_results[0]
    predictions, probabilities = obtain_final_predictions(file_path)
    predictions, probabilities = pad_predictions_and_probabilities(predictions, probabilities, len(data))

    define_actions(data, probabilities, result['threshold'], result['level'], True, context_window=context_window)
    data, _, _ = run_simulation(data, initial_capital, reverse, result['risk_amount'], expected_profit=None, log=False)
    data['Profit'] = data['Profit'].fillna(0)

    print_actions(data)
    write_to_file(output_file, best_results)
    return data, predictions, probabilities

def pad_vault(data):
    best_vault = max(data.loc[0, 'Vault'], data.loc[0, 'Profit'])
    for i in range(0, len(data)):
        print_colored_sentence(f"Date: {data.loc[i, 'Date']}, Vault: {data.loc[i, 'Vault']}, Profit: {data.loc[i, 'Profit']}, Best Vault: {best_vault}")
        if data.loc[i, 'Profit'] > 0:
            best_vault += data.loc[i, 'Profit']
            data.loc[i, 'Vault'] = best_vault
        if data.loc[i, 'Vault'] < best_vault:
            data.loc[i, 'Vault'] = best_vault

def pad_predictions_and_probabilities(predictions, probabilities, target_length):
    """Ensure predictions and probabilities are of the same length as target."""
    if len(predictions) < target_length:
        predictions = np.pad(predictions, (0, target_length - len(predictions)), constant_values=3)
    if len(probabilities) < target_length:
        probabilities = np.pad(probabilities, (0, target_length - len(probabilities)), constant_values=3)
    return predictions, probabilities


def process_simulation(data, probabilities, analysis_levels, threshold_options, risk_options, initial_capital, reverse, expected_profit, context_window):
    """Run simulations for all parameter combinations and return sorted results."""
    results = []
    total_combinations = len(analysis_levels) * len(threshold_options) * len(risk_options)
    
    with tqdm(total=total_combinations, desc="Running simulations") as pbar:
        for level, threshold, risk_amount in product(analysis_levels, threshold_options, risk_options):
            define_actions(data, probabilities, threshold, level, True, context_window=context_window)
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
            pbar.update(1)
    
    return sorted(results, key=lambda x: x["debt_profit"], reverse=True)


def update_simulation_data(data, data_for_simulation):
    """Update main data with the simulation results."""
    data.loc[data_for_simulation.index, 'Action'] = data_for_simulation['Action']
    data.loc[data_for_simulation.index, 'Profit'] = data_for_simulation['Profit']
    if 'Vault' not in data.columns:
        data['Vault'] = 0
    data['Vault'] = data['Vault'].combine_first(data_for_simulation['Vault']).ffill()
    return max(data_for_simulation['Profit'])

def print_results(results):
    current_iteration = 0
    for result in results:
        print(f"\033[91mAnalysis Level: {result['level']} with Risk Amount: {result['risk_amount']:.2f}, and Threshold: {result['threshold']}\033[0m")
        print(f"\033[94mFinal capital: {result['final_capital']:.2f}\033[0m")  
        print(f"\033[93mProfit Vault: {result['profit_vault']:.2f}\033[0m")  
        print(f"\033[92mDebt Profit: {result['debt_profit']:.2f}\033[0m")  
        print("-" * 50)
        current_iteration += 1
        
        if result['debt_profit'] <= 0 or current_iteration >= 3:
            break

def print_actions(data):
    for row in data.itertuples():
        if row.Action == 'Buy':
            print(f"\033[91m{row.Date}: {row.Action} at {row.Close:.2f}\033[0m")
        elif row.Action == 'Sell':
            print(f"\033[92m{row.Date}: {row.Action} at {row.Close:.2f}\033[0m")

            
def write_to_file(output_file, best_results):
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

def load_data(file_path, reverse=False, multiply_factor=None):
    data = pd.read_csv(file_path, parse_dates=['Date'])

    print(f"\033[95m --- Obtained New Data ---\033[0m")
    print(f"\033[95mLast date in dataset: {data['Date'].iloc[-1]}\033[0m")
    print(f"\033[95m -------------------------\033[0m")

    if multiply_factor is not None:
        data['Close'] = data['Close'] * multiply_factor
        data['High'] = data['High'] * multiply_factor
        data['Low'] = data['Low'] * multiply_factor
        data['Open'] = data['Open'] * multiply_factor

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

    output_file = "best_hyperparameters.txt"

    return data, predictions, probabilities, results, analysis_levels, threshold_options, risk_options, initial_capital, output_file