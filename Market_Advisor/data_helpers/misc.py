import pandas as pd
from analyzers.market_predicter import obtain_final_predictions
from datetime import datetime
import numpy as np
from termcolor import colored

def read_last_written_row(file_path="last_row.txt"):
    """Read the last written row from a file."""
    try:
        with open(file_path, "r") as file:
            return int(file.read().strip())  # Read and convert to integer
    except (FileNotFoundError, ValueError):
        return 0  # Default to 0 if the file doesn't exist or is empty
    
def read_actions_from_file(file_path):
    """Read actions from a file and return as a DataFrame."""
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        actions = []
        for line in lines:
            # Safely split the line into Timestamp and Action
            parts = line.strip().split(": ", 1)  # Split on the first ": "
            if len(parts) == 2:
                timestamp, action = parts
                actions.append({"Timestamp": timestamp, "Action": action})
            else:
                print(f"Skipping malformed line: {line.strip()}")

        # Convert to DataFrame
        actions_df = pd.DataFrame(actions)
        if not actions_df.empty:
            actions_df["Timestamp"] = pd.to_datetime(actions_df["Timestamp"], errors="coerce")  # Convert to datetime
            actions_df.dropna(subset=["Timestamp"], inplace=True)  # Drop rows with invalid timestamps
            actions_df.set_index("Timestamp", inplace=True)  # Set Timestamp as the index
        else:
            print("No valid data found in the file.")
            return None

        return actions_df

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading actions from file: {e}")
        return None

def write_last_written_row(last_written_row, file_path="last_row.txt"):
    """Write the last written row to a file."""
    with open(file_path, "w") as file:
        file.write(str(last_written_row))

def write_actions_to_file(data, file_path="actions.txt"):
    """Write actions to a file."""
    with open(file_path, "w") as file:
        for row in data.itertuples():
            formatted_date = row.Date.isoformat() if isinstance(row.Date, pd.Timestamp) else str(row.Date)
            file.write(f"{formatted_date}: {row.Action}\n")

def print_colored_sentence(sentence):
    colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
    words = sentence.split()
    for i, word in enumerate(words):
        color = colors[i % len(colors)]
        print(colored(word, color), end=" ")
    print()

def pad_vault(data):
    data.reset_index(drop=True, inplace=True)
    data.fillna(0, inplace=True) 
    if 'Vault' not in data.columns:
        data['Vault'] = 0
    if 'Profit' not in data.columns:
        data['Profit'] = 0
    best_vault = max(data.loc[0, 'Vault'], data.loc[0, 'Profit'])
    for i in range(0, len(data)):
        if data.loc[i, 'Profit'] > 0:
            best_vault += data.loc[i, 'Profit']
            data.loc[i, 'Vault'] = best_vault
        if data.loc[i, 'Vault'] < best_vault:
            data.loc[i, 'Vault'] = best_vault
        if data.loc[i, 'Profit'] > data.loc[max(0, i-1), 'Profit']:
            data.loc[i, 'Action'] = 'Sell'
        if data.loc[i, 'Profit'] < data.loc[max(0, i-1), 'Profit'] and data.loc[i, 'Profit'] < 0:
            data.loc[i, 'Action'] = 'Buy'

def pad_predictions_and_probabilities(predictions, probabilities, target_length):
    """Ensure predictions and probabilities are of the same length as target."""
    if len(predictions) < target_length:
        predictions = np.pad(predictions, (0, target_length - len(predictions)), constant_values=3)
    if len(probabilities) < target_length:
        probabilities = np.pad(probabilities, (0, target_length - len(probabilities)), constant_values=3)
    return predictions, probabilities

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

def calculate_data_length(file_path):
    data = pd.read_csv(file_path)
    return len(data)

def load_data(file_path, reverse=False, initial_capital=50000, multiply_factor=None):
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
    analysis_levels = [0, 1, 2, 3, 4] #, 5
    threshold_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # risk_options = [(initial_capital * 0.01) * i for i in range(1, 100, 5)]
    risk_options = [initial_capital * 0.95]

    output_file = "best_hyperparameters.txt"

    return data, predictions, probabilities, results, analysis_levels, threshold_options, risk_options, 0, output_file