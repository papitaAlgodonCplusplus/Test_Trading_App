import pandas as pd
from analyzers.market_predicter import obtain_final_predictions
from datetime import datetime
import numpy as np
from termcolor import colored

def check_for_expired_actions(data, patience):
    last_buy_action = data.loc[data['Action'] == "Buy"].tail(1)
    last_sell_action = data.loc[data['Action'] == "Sell"].tail(1)
    last_sell_action_index = last_sell_action.index[0] if not last_sell_action.empty else 0
    last_buy_action_index = last_buy_action.index[0] if not last_buy_action.empty else 0
    if last_buy_action_index > last_sell_action_index and len(data) - last_buy_action_index > patience:
        data.loc[len(data) - 1, 'Action'] = "Sell"
        print_colored_sentence(f"last_buy_action_index: {last_buy_action_index}, last_sell_action_index: {last_sell_action_index}, len(data): {len(data)}")
        return
    
    if 'Stop Loss Long' in data.columns:
        stop_loss = data.loc[last_buy_action_index, 'Stop Loss Long'] if not last_buy_action.empty else 0
        if stop_loss > 0 and data.loc[len(data) - 1, 'Close'] <= stop_loss:
            data.loc[len(data) - 1, 'Action'] = "Sell"
            print_colored_sentence(f"last_buy_action_index: {last_buy_action_index}, last_sell_action_index: {last_sell_action_index}, len(data): {len(data)}")
            return
        
    last_sell_short_action = data.loc[data['Action'] == "Sell Short"].tail(1)
    last_buy_to_cover_action = data.loc[data['Action'] == "Buy to Cover"].tail(1)
    last_buy_to_cover_action_index = last_buy_to_cover_action.index[0] if not last_buy_to_cover_action.empty else 0
    last_sell_short_action_index = last_sell_short_action.index[0] if not last_sell_short_action.empty else 0
    if last_sell_short_action_index > last_buy_to_cover_action_index and len(data) - last_sell_short_action_index > patience:
        data.loc[len(data) - 1, 'Action'] = "Buy to Cover"
        print_colored_sentence(f"last_sell_short_action_index: {last_sell_short_action_index}, last_buy_to_cover_action_index: {last_buy_to_cover_action_index}, len(data): {len(data)}")
        return

    if 'Stop Loss Short' in data.columns:
        stop_loss = data.loc[last_sell_short_action_index, 'Stop Loss Short'] if not last_sell_short_action.empty else 0
        if stop_loss > 0 and data.loc[len(data) - 1, 'Close'] >= stop_loss:
            data.loc[len(data) - 1, 'Action'] = "Buy to Cover"
            print_colored_sentence(f"last_sell_short_action_index: {last_sell_short_action_index}, last_buy_to_cover_action_index: {last_buy_to_cover_action_index}, len(data): {len(data)}")
            return
    
def check_not_repeated_last_action(data):
    """
    Check the last action in the data. If it's a 'Buy' or 'Sell Short' action,
    ensure the previous corresponding 'Sell' or 'Buy to Cover' action exists
    in the data. If not, set the last action as 'Hold'.
    
    Parameters:
        data (pd.DataFrame): DataFrame with at least a column named 'Action'.
    """
    if len(data) < 2:
        return data
    
    last_action_index = len(data) - 1
    last_action = data.iloc[last_action_index]['Action']
    if last_action in ['Buy', 'Sell Short']:
        for i in range(last_action_index - 1, -1, -1):
            previous_action = data.iloc[i]['Action']
            if (last_action == 'Buy' and previous_action == 'Sell') or \
               (last_action == 'Sell Short' and previous_action == 'Buy to Cover'):
                return
            if previous_action == last_action:
                data.at[last_action_index, 'Action'] = 'Hold'
                return
    elif last_action in ['Sell', 'Buy to Cover']:
        for i in range(last_action_index - 1, -1, -1):
            previous_action = data.iloc[i]['Action']
            if (last_action == 'Sell' and previous_action == 'Buy') or \
               (last_action == 'Buy to Cover' and previous_action == 'Sell Short'):
                return
            if previous_action == last_action:
                data.at[last_action_index, 'Action'] = 'Hold'
                return
    return

def cement(data, cemented_data):
    if cemented_data is not None:
        data.loc[cemented_data.index, "Action"] = cemented_data["Action"]
        data.loc[cemented_data.index, "Vault"] = cemented_data["Vault"].astype(float)
        data.loc[cemented_data.index, "Profit"] = cemented_data["Profit"]
        data.loc[cemented_data.index, "Risk/Reward Ratios"] = cemented_data["Risk/Reward Ratios"]
        data.loc[cemented_data.index, "Take Profit Long"] = cemented_data["Take Profit Long"]
        data.loc[cemented_data.index, "Stop Loss Long"] = cemented_data["Stop Loss Long"]
        data.loc[cemented_data.index, "Take Profit Short"] = cemented_data["Take Profit Short"].astype(float)
        data.loc[cemented_data.index, "Stop Loss Short"] = cemented_data["Stop Loss Short"].astype(float)
  
def pad_zeros_with_last_nonzero(lst, close_prices):
    padded_list = lst[:]
    padded_list = [float(x) for x in padded_list]  # Ensure all elements are floats
    last_nonzero = None
    for value in padded_list:
        if value != 0:
            last_nonzero = value
            break
    if last_nonzero is None:
        last_nonzero = float(close_prices[0])
    for i in range(len(padded_list)):
        if padded_list[i] == 0 and last_nonzero is not None:
            padded_list[i] = last_nonzero
        elif padded_list[i] != 0:
            last_nonzero = padded_list[i]
    return padded_list

def pad_take_profits_and_stop_losses(data):
    data['Take Profit Long'] = data['Take Profit Long'].astype(float)
    data['Take Profit Short'] = data['Take Profit Short'].astype(float)
    data['Stop Loss Long'] = data['Stop Loss Long'].astype(float)
    data['Stop Loss Short'] = data['Stop Loss Short'].astype(float)
    
    data.loc[:, 'Take Profit Long'] = pad_zeros_with_last_nonzero(data.loc[:, 'Take Profit Long'], data.loc[:, 'Close'])
    data.loc[:, 'Take Profit Short'] = pad_zeros_with_last_nonzero(data.loc[:, 'Take Profit Short'], data.loc[:, 'Close'])
    data.loc[:, 'Stop Loss Long'] = pad_zeros_with_last_nonzero(data.loc[:, 'Stop Loss Long'], data.loc[:, 'Close'])
    data.loc[:, 'Stop Loss Short'] = pad_zeros_with_last_nonzero(data.loc[:, 'Stop Loss Short'], data.loc[:, 'Close'])

def print_colored_sentence(sentence):
    colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
    words = sentence.split()
    for i, word in enumerate(words):
        color = colors[i % len(colors)]
        print(colored(word, color), end=" ")
    print()

def pad_vault(data):
    data['Vault'] = data['Vault'].astype(float)
    best_vault = max(data.loc[0, 'Vault'], data.loc[0, 'Profit'])
    for i in range(0, len(data)):
        print_colored_sentence(f"Date: {data.loc[i, 'Date']}, Vault: {data.loc[i, 'Vault']}, Profit: {data.loc[i, 'Profit']}, Best Vault: {best_vault}, Action: {data.loc[i, 'Action']}")
        if data.loc[i, 'Profit'] > 0:
            best_vault += float(data.loc[i, 'Profit'])
            data.loc[i, 'Vault'] = best_vault
        if data.loc[i, 'Vault'] < best_vault:
            data.loc[i, 'Vault'] = best_vault

def calculate_profit_summatory(data):
    data['Profit Summatory'] = data['Profit'].cumsum()

def wins_and_losses_count(data):
    wins, losses, break_evens = 0, 0, 0
    if 'Win/Loss Ratio' not in data.columns:
        data['Win/Loss Ratio'] = 0
    data['Win/Loss Ratio'] = data['Win/Loss Ratio'].astype(float)
    for row in data.itertuples():
        if row.Profit > 0:
            wins += 1
        elif row.Profit < 0:
            losses += 1
        else:
            break_evens += 1
        total = wins + losses
        data.loc[row.Index, 'Win/Loss Ratio'] = (wins - losses) / total if total > 0 else 0

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
    data['Vault'] = data['Vault'].combine_first(data_for_simulation['Vault'].replace('', np.nan)).ffill()
    return max(data_for_simulation['Profit'].replace('', np.nan).fillna(0))

def print_results(results):
    current_iteration = 0
    for result in results:
        print(f"\033[91mAnalysis Level: {result['level']} with Risk Amount: {result['risk_amount']:.2f}, and Threshold: {result['threshold']}\033[0m")
        print(f"\033[94mFinal capital: {result['final_capital']:.2f}\033[0m")  
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
        elif row.Action == 'Sell Short':
            print(f"\033[93m{row.Date}: {row.Action} at {row.Close:.2f}\033[0m")
        elif row.Action == 'Buy to Cover':
            print(f"\033[94m{row.Date}: {row.Action} at {row.Close:.2f}\033[0m")
          
def write_to_file(output_file, best_results):
    with open(output_file, 'a') as f:
        f.write(f"\n--- Best Hyperparameters Round ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
        for i, result in enumerate(best_results, start=1):
            f.write(f"Rank {i}:\n")
            f.write(f"  Analysis Level: {result['level']}\n")
            f.write(f"  Threshold: {result['threshold']}\n")
            f.write(f"  Risk Amount: {result['risk_amount']:.2f}\n")
            f.write(f"  Final Capital: {result['final_capital']:.2f}\n")
            f.write(f"  Debt Profit: {result['debt_profit']:.2f}\n")
            f.write(f"  Patience: {result['patience']}\n")
            f.write(f"  Take Profit: {result['take_profit']}\n")
            f.write(f"  Stop Loss: {result['stop_loss']}\n")
            f.write("-" * 50 + "\n")

    print(f"Best hyperparameters saved to {output_file}")

import pandas as pd
import numpy as np

def load_data(
    file_path, 
    reverse=False, 
    multiply_factor=None, 
    initial_capital=50000, 
    deep_calculation=False, 
    bot_type=1
):
    data = pd.read_csv(file_path, parse_dates=['Date'])

    print(f"\033[95m --- Obtained New Data ---\033[0m")
    print(f"\033[95mLast date in dataset: {data['Date'].iloc[-1]}\033[0m")
    print(f"\033[95m -------------------------\033[0m")

    if multiply_factor is not None:
        data['Close'] = data['Close'] * multiply_factor
        data['High'] = data['High'] * multiply_factor
        data['Low'] = data['Low'] * multiply_factor
        data['Open'] = data['Open'] * multiply_factor

    predictions, probabilities = None, None

    if len(data) > 50:
        predictions, probabilities = obtain_final_predictions(file_path)
        if reverse:
            predictions = 1 - predictions
            
        if len(predictions) < len(data):
            predictions = np.pad(predictions, (0, len(data) - len(predictions)), constant_values=3)

        if len(probabilities) < len(data):
            probabilities = np.pad(probabilities, (0, len(data) - len(probabilities)), constant_values=3)

    bot_configs = {
        1: {
            'analysis_levels': [0],
            'threshold_options': [0.0],
            'risk_options': [5],
            'patience_options': [350],
            'take_profit_options': [0.1],
            'stop_loss_options': [0.01],
        },
        2: {
            'analysis_levels': [4],
            'threshold_options': [0.0],
            'risk_options': [0.5],
            'patience_options': [500],
            'take_profit_options': [0.05],
            'stop_loss_options': [0.001],
        },
        3: {
            'analysis_levels': [0],
            'threshold_options': [0.4],
            'risk_options': [5],
            'patience_options': [100],
            'take_profit_options': [0.2],
            'stop_loss_options': [0.025],
        },
        4: {
            'analysis_levels': [1],
            'threshold_options': [0.0],
            'risk_options': [1],
            'patience_options': [100],
            'take_profit_options': [0.05],
            'stop_loss_options': [0.01],
        },
        5: {
            'analysis_levels': [3],
            'threshold_options': [0.2],
            'risk_options': [1],
            'patience_options': [500],
            'take_profit_options': [0.1],
            'stop_loss_options': [0.05],
        }
    }

    if not deep_calculation:
        config = bot_configs.get(bot_type, bot_configs[1])
        analysis_levels = config['analysis_levels']
        threshold_options = config['threshold_options']
        risk_options = config['risk_options']
        patience_options = config['patience_options']
        take_profit_options = config['take_profit_options']
        stop_loss_options = config['stop_loss_options']
    else:
        analysis_levels = [0, 1, 2, 3, 4]
        threshold_options = np.arange(0.0, 0.4, 0.1)
        risk_options = [1, 5]
        patience_options = [50, 150, 300, 500]
        take_profit_options = [0.01, 0.05, 0.1, 0.15, 0.2]
        stop_loss_options = [0.001, 0.005, 0.01, 0.015, 0.02]

    output_file = "best_hyperparameters.txt"

    if 'Action' not in data.columns:
        data['Action'] = 'Hold'

    if 'Vault' not in data.columns:
        data['Vault'] = 0

    return (
        data, 
        predictions, 
        probabilities, 
        analysis_levels, 
        threshold_options, 
        risk_options, 
        patience_options, 
        take_profit_options, 
        stop_loss_options, 
        initial_capital, 
        output_file
    )
